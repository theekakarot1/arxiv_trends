from __future__ import annotations

import logging
import os

import cohere
from dotenv import load_dotenv
from langchain_core.documents import Document

from retrieval.retriever import (
    get_chunks_from_neo4j,
    get_chunks_filtered_async,
    get_chunks_from_neo4j_async,
)

load_dotenv()
logger = logging.getLogger(__name__)

_COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
_co = cohere.ClientV2(api_key=_COHERE_API_KEY) if _COHERE_API_KEY else None


def _rerank_chunks(chunks: list[dict], query: str, top_n: int) -> list[dict]:
    """
    Run Cohere reranking on a list of chunks.
    Returns reranked top_n chunks, or the original top_n by score on failure.
    Shared by both sync and async callers.
    """
    if not _co:
        logger.warning("Cohere client not configured — skipping rerank")
        return chunks[:top_n]

    try:
        doc_texts = [chunk["text"] for chunk in chunks]
        result    = _co.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=doc_texts,
            top_n=top_n,
        )
        reranked: list[dict] = []
        for r in result.results:
            chunk = dict(chunks[r.index])
            chunk["rerank_score"] = r.relevance_score
            reranked.append(chunk)
        return reranked

    except Exception as exc:
        logger.error("Cohere reranking failed: %s — using score fallback", exc)
        return chunks[:top_n]


# ---------------------------------------------------------------------------
# Synchronous interface (existing — unchanged behaviour)
# ---------------------------------------------------------------------------

def retrieve_and_rerank(driver, user_query: str, top_n: int = 10) -> list[dict]:
    """
    Two-stage retrieval for sync callers.
    Accepts an existing neo4j sync driver.
    """
    initial_chunks = get_chunks_from_neo4j(driver, user_query, k=50)
    if not initial_chunks:
        logger.warning("No chunks retrieved from Neo4j.")
        return []
    return _rerank_chunks(initial_chunks, user_query, top_n)


def get_retriever_documents(driver, user_query: str) -> list[Document]:
    """LangChain-compatible retriever interface for sync callers."""
    chunks = retrieve_and_rerank(driver, user_query)
    return [
        Document(page_content=chunk["text"], metadata=chunk["metadata"])
        for chunk in chunks
    ]


# ---------------------------------------------------------------------------
# Async interface (used by LangGraph nodes)
# ---------------------------------------------------------------------------

async def retrieve_and_rerank_async(
    user_query: str,
    top_n: int = 10,
    k: int = 50,
    paper_ids: list[str] | None = None,
    neo4j_uri: str = "",
    neo4j_user: str = "",
    neo4j_password: str = "",
    neo4j_database: str = "616caddc",
) -> list[dict]:
    """
    Async two-stage retrieval used by LangGraph nodes.

    Parameters
    ----------
    paper_ids : list[str] | None
        When provided, restricts vector search to chunks belonging to
        these papers (used in the hybrid citation path).
        When None, searches all chunks.
    """
    connection_kwargs = dict(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        neo4j_database=neo4j_database,
    )

    if paper_ids is not None:
        initial_chunks = await get_chunks_filtered_async(
            user_query, paper_ids, k=k, **connection_kwargs
        )
    else:
        initial_chunks = await get_chunks_from_neo4j_async(
            user_query, k=k, **connection_kwargs
        )

    if not initial_chunks:
        logger.warning("Async retrieval returned no chunks.")
        return []

    return _rerank_chunks(initial_chunks, user_query, top_n)