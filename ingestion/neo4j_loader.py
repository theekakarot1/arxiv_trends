"""
neo4j_loader.py
---------------
All Neo4j interactions: connection, schema creation, data ingestion, and
vector-index management.

Graph schema (nodes)
---------------------
  Paper       – title, entry_id, published, summary, doi, journal_ref
  Author      – name
  Category    – name
  Model       – name  (Models & Algorithms entities)
  Dataset     – name
  Metrics     – name
  Libraries   – name  (Libraries & Frameworks entities)
  Tasks       – name
  Concepts    – name  (Theories & Concepts entities)
  Institute   – name
  Chunk       – text, embedding  (vector-searchable chunks of Paper content)

Relationships
-------------
  (Paper)-[:WRITTEN_BY]           -> (Author)
  (Paper)-[:IN_CATEGORY]          -> (Category)
  (Paper)-[:MODEL_ALGORITHM_USED] -> (Model)
  (Paper)-[:DATASET_USED]         -> (Dataset)
  (Paper)-[:METRICS_USED]         -> (Metrics)
  (Paper)-[:LIBRARY_FRAMEWORK_USED] -> (Libraries)
  (Paper)-[:TASK_PERFORMED]       -> (Tasks)
  (Paper)-[:THEORIES_CONCEPTS_USED] -> (Concepts)
  (Paper)-[:INSTITUTE]            -> (Institute)
  (Paper)-[:HAS_CHUNK]            -> (Chunk)
  (Paper)-[:CITES]                -> (Paper)   ← citation graph
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict

import nest_asyncio
from dotenv import load_dotenv
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from neo4j import GraphDatabase, exceptions
from sentence_transformers import SentenceTransformer

nest_asyncio.apply()
load_dotenv()

logger = logging.getLogger(__name__)

NEO4J_URI      = os.getenv("NEO4J_URI")
NEO4J_USER     = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Shared embedding model used throughout the ingestion pipeline.
# all-MiniLM-L6-v2 produces 384-dimensional embeddings.
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Chunk parameters — update these after running notebooks/chunk_size_analysis.ipynb.
# These values are the final size/overlap applied WITHIN each section after
# the markdown header splitter has already separated the paper into sections.
CHUNK_SIZE    = 832
CHUNK_OVERLAP = 0

# Markdown headers that signal a new logical section in arXiv papers.
# pymupdf4llm renders section titles as ## and subsection titles as ###.
_HEADERS_TO_SPLIT = [
    ("#",   "section"),
    ("##",  "section"),
    ("###", "subsection"),
]


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def connect_to_neo4j():
    """Open and verify a Neo4j driver.  Returns None on failure."""
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        logger.info("Connected to Neo4j at %s", NEO4J_URI)
        return driver
    except exceptions.ServiceUnavailable as exc:
        logger.error("Cannot connect to Neo4j at %s: %s", NEO4J_URI, exc)
        return None


def close_neo4j_driver(driver) -> None:
    if driver:
        driver.close()
        logger.info("Neo4j connection closed.")


def run_cypher_query(driver, query: str, parameters: dict | None = None):
    """Execute a Cypher query and return (records, summary, keys)."""
    if not driver:
        return None, None, None
    records, summary, keys = driver.execute_query(
        query,
        parameters_=parameters or {},
        database_="616caddc",
    )
    return records, summary, keys


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def create_chunks(doc_list: list[list]) -> list:
    """Split papers into section-aware, overlapping chunks for vector search.

    Two-stage pipeline
    ------------------
    Stage 1 — MarkdownHeaderTextSplitter
        Splits the paper's markdown at section/subsection boundaries (# / ## /
        ###).  Each resulting split carries section metadata:
            {"section": "Introduction"} or {"subsection": "3.2 Experimental Setup"}
        This prevents a chunk from spanning two unrelated sections (e.g. the end
        of Methods and the start of Results in the same chunk).

    Stage 2 — RecursiveCharacterTextSplitter
        Splits long sections that exceed CHUNK_SIZE into overlapping windows.
        Inherits the section/subsection metadata from Stage 1 so every final
        chunk knows which part of the paper it came from.

    Why this is better than a single RecursiveCharacterTextSplitter
    ---------------------------------------------------------------
    - Chunks never cross section boundaries — a retrieval result is always
      from one coherent part of the paper.
    - The `section` metadata is stored on every Neo4j Chunk node, enabling
      structured queries like:
          MATCH (c:Chunk) WHERE c.section = 'Methods' ...
    - The paper's markdown structure (produced by pymupdf4llm in arxiv_loader)
      is used as a first-class splitting signal, not discarded.

    Parameters
    ----------
    doc_list : list[list[Document]]
        Nested list as returned by arxiv_loader.get_arxiv_documents().
        Each Document's page_content is expected to be clean Markdown with
        ## section headers.
    """
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADERS_TO_SPLIT,
        strip_headers=False,   # keep the heading inside the chunk text
    )
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
        is_separator_regex=False,
    )

    flat_docs = [doc for sublist in doc_list for doc in sublist]
    all_chunks: list = []

    for doc in flat_docs:
        # Stage 1: split on markdown headers
        section_docs = header_splitter.split_text(doc.page_content)

        # Propagate the parent document's metadata (entry_id, title, etc.)
        # into each section document, then merge with header metadata.
        for sec_doc in section_docs:
            merged_metadata = {**doc.metadata, **sec_doc.metadata}
            sec_doc.metadata = merged_metadata

        # Stage 2: split long sections by character count
        final_chunks = char_splitter.split_documents(section_docs)
        all_chunks.extend(final_chunks)

    return all_chunks


# ---------------------------------------------------------------------------
# Core data ingestion
# ---------------------------------------------------------------------------

def ingest_papers(doc_details: list[dict], driver) -> None:
    """Write Paper nodes and all related entity nodes/relationships to Neo4j.

    Uses MERGE so the function is safe to call multiple times — existing nodes
    are never duplicated.
    """
    for doc in doc_details:
        doc = defaultdict(lambda: None, doc)
        entry_id = doc["entry_id"]
        if not entry_id:
            logger.warning("Skipping document with no entry_id.")
            continue

        # ---- Paper node ------------------------------------------------
        run_cypher_query(
            driver,
            """
            MERGE (p:Paper {entry_id: $entry_id})
            SET
                p.title       = $title,
                p.published   = datetime($published),
                p.summary     = $summary,
                p.doi         = $doi,
                p.journal_ref = $journal_ref
            """,
            {
                "entry_id":    entry_id,
                "title":       doc["Title"],
                "published":   doc["Published"],
                "summary":     doc["Summary"],
                "doi":         doc["doi"],
                "journal_ref": doc["journal_ref"],
            },
        )

        # ---- Helper: ingest a list of entity names with one relationship --
        def _ingest_entities(label: str, rel: str, values: list[str] | None) -> None:
            if not values:
                return
            run_cypher_query(
                driver,
                f"""
                MATCH (p:Paper {{entry_id: $entry_id}})
                UNWIND $values AS val
                MERGE (e:{label} {{name: val}})
                MERGE (p)-[:{rel}]->(e)
                """,
                {"entry_id": entry_id, "values": values},
            )

        # ---- Authors ---------------------------------------------------
        authors = [a.strip() for a in (doc["Authors"] or "").split(",") if a.strip()]
        _ingest_entities("Author",    "WRITTEN_BY",             authors)

        # ---- Taxonomy / structured metadata ----------------------------
        _ingest_entities("Category",  "IN_CATEGORY",            doc["categories"])
        _ingest_entities("Model",     "MODEL_ALGORITHM_USED",   doc["Models & Algorithms"])
        _ingest_entities("Dataset",   "DATASET_USED",           doc["Datasets"])
        _ingest_entities("Metrics",   "METRICS_USED",           doc["Metrics"])
        _ingest_entities("Libraries", "LIBRARY_FRAMEWORK_USED", doc["Libraries & Frameworks"])
        _ingest_entities("Tasks",     "TASK_PERFORMED",         doc["Tasks"])
        _ingest_entities("Concepts",  "THEORIES_CONCEPTS_USED", doc["Theories & Concepts"])
        _ingest_entities("Institute", "INSTITUTE",              doc["Institutions"])

    logger.info("Paper ingestion complete (%d documents).", len(doc_details))


def ingest_citations(doc_details: list[dict], driver) -> None:
    """Create CITES relationships between Paper nodes.

    For each paper we stored a 'cited_arxiv_ids' list (extracted in
    arxiv_loader._extract_cited_ids).  This function walks that list and
    creates (citing_paper)-[:CITES]->(cited_paper) edges.

    Papers cited but not yet ingested are created as stub nodes so the graph
    remains consistent; their metadata fields will be populated if those papers
    are ingested in a future run.

    This enables queries like:
      "What papers cite 'Attention is All You Need'?"
      "Which papers published after 2020 build on paper X?"
    """
    for doc in doc_details:
        citing_id   = doc.get("entry_id")
        cited_ids   = doc.get("cited_arxiv_ids", [])

        if not citing_id or not cited_ids:
            continue

        run_cypher_query(
            driver,
            """
            MATCH (citing:Paper {entry_id: $citing_id})
            UNWIND $cited_ids AS cited_id
            MERGE (cited:Paper {entry_id: cited_id})
            MERGE (citing)-[:CITES]->(cited)
            """,
            {"citing_id": citing_id, "cited_ids": cited_ids},
        )

    logger.info("Citation relationships ingested.")


# ---------------------------------------------------------------------------
# Vector index + chunk embeddings
# ---------------------------------------------------------------------------

def create_vector_index(driver) -> None:
    """Create the Neo4j vector index on Chunk nodes (idempotent)."""
    run_cypher_query(
        driver,
        """
        CREATE VECTOR INDEX `chunk-embeddings` IF NOT EXISTS
        FOR (c:Chunk) ON c.embedding
        OPTIONS {
            indexConfig: {
                `vector.dimensions`: 384,
                `vector.similarity_function`: 'cosine'
            }
        }
        """,
    )
    logger.info("Vector index 'chunk-embeddings' ensured.")


def ingest_chunks_embeddings(driver, chunks: list) -> None:
    """Embed each chunk and store it as a Chunk node linked to its Paper.

    Each Chunk node stores:
      - text        : the chunk content
      - embedding   : 384-dim vector for cosine similarity search
      - section     : top-level section name from the paper (e.g. "Introduction")
      - subsection  : subsection name if present (e.g. "3.2 Experimental Setup")

    The section/subsection fields come from the MarkdownHeaderTextSplitter
    metadata added in create_chunks().  They enable structured retrieval:

        MATCH (c:Chunk)<-[:HAS_CHUNK]-(p:Paper)
        WHERE c.section = 'Methods'
        RETURN c.text, p.title
    """
    query = """
    MATCH (p:Paper {entry_id: $parent_paper_id})
    CREATE (c:Chunk {
        text:       $chunk_text,
        embedding:  $chunk_embedding,
        section:    $section,
        subsection: $subsection
    })
    CREATE (p)-[:HAS_CHUNK]->(c)
    """
    for chunk in chunks:
        parent_id = chunk.metadata.get("entry_id")
        if not parent_id:
            logger.debug("Chunk has no entry_id in metadata; skipping.")
            continue

        embedding = embedding_model.encode(chunk.page_content).tolist()
        run_cypher_query(
            driver,
            query,
            {
                "parent_paper_id": parent_id,
                "chunk_text":      chunk.page_content,
                "chunk_embedding": embedding,
                "section":         chunk.metadata.get("section"),
                "subsection":      chunk.metadata.get("subsection"),
            },
        )

    logger.info("Chunk embeddings ingested (%d chunks).", len(chunks))