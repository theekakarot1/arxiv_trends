from __future__ import annotations

import operator
from typing import Annotated, Any

from typing_extensions import TypedDict


class AppState(TypedDict, total=False):
    # ---- Input -------------------------------------------------------
    # The user's current raw query
    user_query: str

    # Conversation history — list of {"role": str, "content": str}
    # Annotated with operator.add so LangGraph appends new messages
    chat_history: Annotated[list[dict], operator.add]

    # ---- Query understanding outputs ---------------------------------
    # "semantic" | "cypher" | "hybrid"
    intent: str

    # "text" | "plot" | "number"
    output_type: str

    # Extracted entities: paper_title, author, year, topic
    entities: dict[str, Any]

    # One-sentence explanation of the classification
    query_reasoning: str

    # ---- Paper resolution outputs ------------------------------------
    # entry_id of the resolved paper (None if not found or not applicable)
    resolved_paper_id: str | None

    # Canonical title from the database
    resolved_paper_title: str | None

    # ---- Cypher generation outputs -----------------------------------
    generated_cypher: str | None
    cypher_parameters: dict[str, Any]

    # ---- Retrieval outputs -------------------------------------------
    # Results from Cypher execution — list of record dicts
    cypher_results: list[dict]

    # entry_ids of papers found via CITES traversal (hybrid path)
    citing_paper_ids: list[str]

    # Chunks from vector retrieval — list of {text, score, metadata}
    retrieved_chunks: list[dict]

    # ---- Generation outputs ------------------------------------------
    # Final text answer for the user
    answer: str | None

    # Python code string for plot generation
    plot_code: str | None

    # Rendered plotly figure (not serialised — held in memory only)
    plot_figure: Any | None

    # ---- Control flow ------------------------------------------------
    # Populated when retrieval finds nothing — drives the not_found node
    not_found_reason: str | None

    # Template key from prompts.NOT_FOUND_MESSAGES
    not_found_key: str | None

    # Error message if a node failed (for logging and user display)
    error: str | None

    # LangSmith run ID for the current turn (used for feedback submission)
    langsmith_run_id: str | None

    # ---- Memory ------------------------------------------------------
    # Summarised older history when token limit is hit
    history_summary: str | None