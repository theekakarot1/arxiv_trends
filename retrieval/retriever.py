from __future__ import annotations

import logging

from ingestion.neo4j_loader import run_cypher_query
from retrieval.embedder import embedding_model

logger = logging.getLogger(__name__)

# Shared Cypher — used by both sync and async paths
_VECTOR_SEARCH_CYPHER = """
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

_VECTOR_SEARCH_FILTERED_CYPHER = """
CALL db.index.vector.queryNodes('chunk-embeddings', $k, $query_embedding)
YIELD node AS chunk, score
MATCH (chunk)<-[:HAS_CHUNK]-(p:Paper)
WHERE p.entry_id IN $paper_ids
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


def _format_records(records) -> list[dict]:
    """Convert Neo4j records to the standard chunk dict format."""
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


# ---------------------------------------------------------------------------
# Synchronous interface (existing — unchanged)
# ---------------------------------------------------------------------------

def get_chunks_from_neo4j(driver, user_query: str, k: int = 50) -> list[dict]:
    """Perform a vector search over Chunk nodes and return the top-k results.

    Synchronous — uses the existing neo4j sync driver from neo4j_loader.
    Called by the legacy reranker and any non-async code paths.
    """
    if not driver:
        logger.error("Neo4j driver is not connected.")
        return []

    query_embedding = embedding_model.encode(user_query).tolist()

    try:
        records, _, _ = run_cypher_query(
            driver,
            _VECTOR_SEARCH_CYPHER,
            {"query_embedding": query_embedding, "k": k},
        )
        return _format_records(records)
    except Exception as exc:
        logger.error("Error during Neo4j vector retrieval: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Async interface (used by LangGraph nodes)
# ---------------------------------------------------------------------------

async def get_chunks_from_neo4j_async(
    user_query: str,
    k: int = 50,
    neo4j_uri: str = "",
    neo4j_user: str = "",
    neo4j_password: str = "",
    neo4j_database: str = "neo4j",
) -> list[dict]:
    """
    Async version of get_chunks_from_neo4j.
    Uses AsyncGraphDatabase so it does not block the event loop.
    Falls back to empty list on any error.
    """
    from neo4j import AsyncGraphDatabase

    query_embedding = embedding_model.encode(user_query).tolist()
    driver = AsyncGraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    try:
        async with driver.session(database=neo4j_database) as session:
            result = await session.run(
                _VECTOR_SEARCH_CYPHER,
                {"query_embedding": query_embedding, "k": k},
            )
            records = [r async for r in result]
        return _format_records(records)
    except Exception as exc:
        logger.error("Async vector retrieval failed: %s", exc)
        return []
    finally:
        await driver.close()


async def get_chunks_filtered_async(
    user_query: str,
    paper_ids: list[str],
    k: int = 50,
    neo4j_uri: str = "",
    neo4j_user: str = "",
    neo4j_password: str = "",
    neo4j_database: str = "616caddc",
) -> list[dict]:
    """
    Async vector search scoped to a specific list of paper entry_ids.
    Used in the hybrid citation path after Cypher has returned citing paper IDs.
    Falls back to unfiltered search if paper_ids is empty.
    """
    if not paper_ids:
        logger.warning("get_chunks_filtered_async called with no paper_ids — falling back to unfiltered")
        return await get_chunks_from_neo4j_async(
            user_query, k, neo4j_uri, neo4j_user, neo4j_password, neo4j_database
        )

    from neo4j import AsyncGraphDatabase

    query_embedding = embedding_model.encode(user_query).tolist()
    driver = AsyncGraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    try:
        async with driver.session(database=neo4j_database) as session:
            result = await session.run(
                _VECTOR_SEARCH_FILTERED_CYPHER,
                {"query_embedding": query_embedding, "k": k, "paper_ids": paper_ids},
            )
            records = [r async for r in result]
        return _format_records(records)
    except Exception as exc:
        logger.error("Async filtered retrieval failed: %s", exc)
        return []
    finally:
        await driver.close()