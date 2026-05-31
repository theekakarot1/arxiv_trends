from __future__ import annotations

from config import CHARS_PER_TOKEN


def estimate_tokens(messages: list[dict]) -> int:
    """Rough token estimate: total chars / CHARS_PER_TOKEN."""
    return sum(len(m.get("content", "")) for m in messages) // CHARS_PER_TOKEN


def format_history_for_display(messages: list[dict]) -> list[dict]:
    """
    Filter history to only user/assistant turns for display.
    Strips internal system messages (e.g. history summaries).
    """
    return [
        m for m in messages
        if m.get("role") in ("user", "assistant")
    ]


def build_initial_state(
    user_query: str,
    chat_history: list[dict],
) -> dict:
    """
    Build the initial state dict for a new LangGraph invocation.
    Called once per user query from main.py.
    """
    return {
        "user_query":         user_query,
        "chat_history":       chat_history,
        "intent":             None,
        "output_type":        None,
        "entities":           {},
        "query_reasoning":    None,
        "resolved_paper_id":  None,
        "resolved_paper_title": None,
        "generated_cypher":   None,
        "cypher_parameters":  {},
        "cypher_results":     [],
        "citing_paper_ids":   [],
        "retrieved_chunks":   [],
        "answer":             None,
        "plot_code":          None,
        "plot_figure":        None,
        "not_found_reason":   None,
        "not_found_key":      None,
        "error":              None,
        "langsmith_run_id":   None,
        "history_summary":    None,
    }