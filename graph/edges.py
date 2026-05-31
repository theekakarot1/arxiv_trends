from __future__ import annotations

from graph.state import AppState


def route_after_query_understanding(state: AppState) -> str:
    """
    After classifying the query, decide the first retrieval step.

    Routing logic:
      - If a paper title was extracted → paper_resolution
        (both cypher and hybrid paths need to know the entry_id first)
      - If intent is semantic → vector_retrieval
        (no paper title needed, go straight to vector search)
    """
    entities    = state.get("entities", {})
    intent      = state.get("intent", "semantic")
    paper_title = entities.get("paper_title")

    if paper_title and intent in ("cypher", "hybrid"):
        return "paper_resolution"

    if intent == "semantic":
        return "vector_retrieval"

    # cypher intent without a specific paper title (e.g. "how many papers in 2024")
    return "cypher_generation"


def route_after_paper_resolution(state: AppState) -> str:
    """
    After attempting to resolve a paper title, decide next step.

      - If paper not found → not_found
      - If intent is cypher → cypher_generation (metadata query)
      - If intent is hybrid → cypher_generation (CITES traversal)
    """
    if state.get("not_found_key") == "paper_not_found":
        return "not_found"

    intent = state.get("intent", "cypher")

    if intent == "hybrid":
        return "cypher_generation"   # generate CITES traversal first

    return "cypher_generation"       # generate metadata/count query


def route_after_cypher_execution(state: AppState) -> str:
    """
    After running Cypher, decide next step based on results and intent.

      - Empty results → not_found
      - hybrid intent → filtered_retrieval (vector search on citing papers)
      - plot output   → plot node
      - text/number   → answer node
    """
    if state.get("not_found_key") in ("no_results", "no_citing_papers"):
        return "not_found"

    intent      = state.get("intent", "cypher")
    output_type = state.get("output_type", "text")

    if intent == "hybrid":
        return "filtered_retrieval"

    if output_type == "plot":
        return "plot"

    return "answer"


def route_after_retrieval(state: AppState) -> str:
    """
    After vector retrieval (semantic or filtered), decide next step.

      - Empty results → not_found
      - Otherwise → rerank
    """
    chunks = state.get("retrieved_chunks", [])

    if not chunks:
        return "not_found"

    return "rerank"


def route_after_rerank(state: AppState) -> str:
    """
    After reranking, always go to answer.
    (Plot is only triggered from Cypher results, not vector retrieval.)
    """
    return "answer"


def route_after_answer(state: AppState) -> str:
    """After generating an answer, always update memory."""
    return "memory_update"


def route_after_not_found(state: AppState) -> str:
    """After not-found message, still update memory so the turn is recorded."""
    return "memory_update"