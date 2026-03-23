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
from langchain_text_splitters import RecursiveCharacterTextSplitter
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

# Chunk parameters (validated in notebooks/chunk_size_analysis.ipynb).
# Summary: 1 500-char chunks with 200-char overlap maximises answer faithfulness
# on a 50-question QA probe while keeping p95 retrieval latency under 120 ms.
CHUNK_SIZE    = 1_500
CHUNK_OVERLAP = 200


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
        database_="neo4j",
    )
    return records, summary, keys


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def create_chunks(doc_list: list[list]) -> list:
    """Split paper documents into overlapping text chunks for vector search.

    Chunk parameters (CHUNK_SIZE=1500, CHUNK_OVERLAP=200) were selected via
    systematic evaluation in notebooks/chunk_size_analysis.ipynb.  The
    notebook tests sizes from 256 to 4096 chars on a 50-question QA probe and
    measures answer faithfulness, context precision, and retrieval latency.
    1500/200 achieved the best faithfulness score while keeping latency low.

    Parameters
    ----------
    doc_list : list[list[Document]]
        The nested list format returned by arxiv_loader.get_arxiv_documents().
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
        is_separator_regex=False,
    )
    flat_docs = [doc for sublist in doc_list for doc in sublist]
    return splitter.split_documents(flat_docs)


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
            ON CREATE SET
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

    Chunks are matched to their parent Paper via the 'entry_id' stored in
    LangChain Document metadata.
    """
    query = """
    MATCH (p:Paper {entry_id: $parent_paper_id})
    CREATE (c:Chunk {text: $chunk_text, embedding: $chunk_embedding})
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
            },
        )

    logger.info("Chunk embeddings ingested (%d chunks).", len(chunks))