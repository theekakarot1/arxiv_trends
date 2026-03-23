"""
retriever.py
------------
Vector search against the Neo4j chunk-embeddings index.
Returns the top-k candidate chunks for a given user query.
"""

from __future__ import annotations

import logging

from ingestion.neo4j_loader import run_cypher_query
from retrieval.embedder import embedding_model

logger = logging.getLogger(__name__)


def get_chunks_from_neo4j(driver, user_query: str, k: int = 50) -> list[dict]:
    """Perform a vector search over Chunk nodes and return the top-k results.

    Each result dict contains:
        text        – chunk text
        score       – cosine similarity score
        metadata    – dict with paperTitle, paperId, section, subsection
    """
    if not driver:
        logger.error("Neo4j driver is not connected.")
        return []

    query_embedding = embedding_model.encode(user_query).tolist()

    cypher = """
    CALL db.index.vector.queryNodes('chunk-embeddings', $k, $query_embedding)
    YIELD node AS chunk, score
    MATCH (chunk)<-[:HAS_CHUNK]-(p:Paper)
    RETURN
        chunk.text       AS text,
        score,
        p.title          AS paperTitle,
        p.entry_id       AS paperId,
        chunk.section    AS section,
        chunk.subsection AS subsection
    ORDER BY score DESC
    LIMIT $k
    """
    try:
        records, _, _ = run_cypher_query(
            driver,
            cypher,
            {"query_embedding": query_embedding, "k": k},
        )
        return [
            {
                "text":  record["text"],
                "score": record["score"],
                "metadata": {
                    "paperTitle":  record["paperTitle"],
                    "paperId":     record["paperId"],
                    "section":     record["section"],
                    "subsection":  record["subsection"],
                },
            }
            for record in records
        ]
    except Exception as exc:
        logger.error("Error during Neo4j vector retrieval: %s", exc)
        return []