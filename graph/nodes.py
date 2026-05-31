from __future__ import annotations

import json
import logging
from typing import Any, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from config import (
    CHARS_PER_TOKEN,
    HISTORY_TOKEN_LIMIT,
    NEO4J_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    RERANK_TOP_N,
    VECTOR_SEARCH_K,
)
from graph.state import AppState
from prompts import (
    ANSWER_HUMAN,
    ANSWER_SYSTEM,
    CYPHER_GENERATION_SYSTEM,
    MEMORY_SUMMARISATION_HUMAN,
    MEMORY_SUMMARISATION_SYSTEM,
    NOT_FOUND_MESSAGES,
    PAPER_RESOLUTION_SYSTEM,
    PLOT_GENERATION_SYSTEM,
    QUERY_UNDERSTANDING_HUMAN,
    QUERY_UNDERSTANDING_SYSTEM,
)
from schemas import (
    CypherGenerationOutput,
    PaperResolutionOutput,
    QueryUnderstandingOutput,
)
from session.tools.neo4j_mcp import Neo4jMCPTools, resolve_paper
from wrappers.plot_tools import execute_plot_code
from retrieval.reranker import retrieve_and_rerank_async

logger = logging.getLogger(__name__)

T = TypeVar("T")

_NEO4J_CONN = dict(
    neo4j_uri=NEO4J_URI,
    neo4j_user=NEO4J_USER,
    neo4j_password=NEO4J_PASSWORD,
    neo4j_database=NEO4J_DATABASE,
)


# ---------------------------------------------------------------------------
# LLM call helpers
# ---------------------------------------------------------------------------

async def _call_structured(
    llm: BaseChatModel,
    schema: type[T],
    system: str,
    human: str,
) -> T | None:
    """
    Call the LLM expecting output that conforms to a Pydantic schema.

    Uses llm.with_structured_output(schema) which routes to the provider's
    native JSON mode (Gemini response_schema, OpenAI function calling).

    Returns the validated Pydantic instance, or None on any failure.
    Logs the specific validation error so you can see exactly which field
    failed and what the LLM returned.
    """
    structured_llm = llm.with_structured_output(schema)
    messages = [SystemMessage(content=system), HumanMessage(content=human)]
    try:
        result = await structured_llm.ainvoke(messages)
        return result
    except ValidationError as exc:
        logger.error(
            "Pydantic validation failed for %s:\n%s",
            schema.__name__,
            exc.json(indent=2),
        )
        return None
    except Exception as exc:
        logger.error(
            "Structured LLM call failed for %s: %s",
            schema.__name__,
            exc,
        )
        return None


async def _call_llm_text(
    llm: BaseChatModel,
    system: str,
    human: str,
) -> str:
    """
    Call the LLM expecting freeform text output.
    Used for: answer synthesis, memory summarisation, plot code generation.
    """
    try:
        messages = [SystemMessage(content=system), HumanMessage(content=human)]
        response = await llm.ainvoke(messages)
        return response.content.strip()
    except Exception as exc:
        logger.error("LLM text call failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Node 1: Query Understanding
# ---------------------------------------------------------------------------

async def query_understanding_node(state: AppState, llm: BaseChatModel) -> dict:
    """
    Classify the user query into intent + output_type + entities.

    Uses QueryUnderstandingOutput schema — Pydantic enforces:
      - intent is one of: semantic | cypher | hybrid
      - output_type is one of: text | plot | number
      - year is a 4-digit string or None (validator rejects malformed values)
      - topic is always populated (required field)
    """
    query = state.get("user_query", "")
    if not query:
        return {"error": "Empty query received."}

    logger.info("Query understanding: '%s'", query[:100])

    result: QueryUnderstandingOutput | None = await _call_structured(
        llm,
        schema=QueryUnderstandingOutput,
        system=QUERY_UNDERSTANDING_SYSTEM,
        human=QUERY_UNDERSTANDING_HUMAN.format(query=query),
    )

    if result is None:
        # Pydantic validation failed or LLM call errored — safe fallback
        logger.warning("Query understanding failed — falling back to semantic intent")
        return {
            "intent":          "semantic",
            "output_type":     "text",
            "entities":        {"topic": query, "paper_title": None, "author": None, "year": None},
            "query_reasoning": "Fallback: structured output validation failed",
        }

    logger.info(
        "Intent=%s | Output=%s | Topic=%s | Paper=%s",
        result.intent,
        result.output_type,
        result.entities.topic,
        result.entities.paper_title,
    )

    return {
        "intent":          result.intent,
        "output_type":     result.output_type,
        "entities":        result.entities.model_dump(),
        "query_reasoning": result.reasoning,
    }


# ---------------------------------------------------------------------------
# Node 2: Paper Resolution
# ---------------------------------------------------------------------------

async def paper_resolution_node(state: AppState, llm: BaseChatModel) -> dict:
    """
    Resolve the paper title from the query to a Neo4j entry_id.

    Uses PaperResolutionOutput schema — Pydantic enforces:
      - matched is a real bool (not a string "true")
      - confidence is one of: high | medium | low
      - empty string entry_id is coerced to None
    """
    entities  = state.get("entities", {})
    raw_title = entities.get("paper_title")

    if not raw_title:
        return {"resolved_paper_id": None, "resolved_paper_title": None}

    logger.info("Resolving paper: '%s'", raw_title)
    candidates = await resolve_paper(raw_title)

    if not candidates:
        logger.info("No candidates found for '%s'", raw_title)
        return {
            "resolved_paper_id":    None,
            "resolved_paper_title": None,
            "not_found_reason":     NOT_FOUND_MESSAGES["paper_not_found"].format(title=raw_title),
            "not_found_key":        "paper_not_found",
        }

    result: PaperResolutionOutput | None = await _call_structured(
        llm,
        schema=PaperResolutionOutput,
        system=PAPER_RESOLUTION_SYSTEM.format(
            raw_title=raw_title,
            search_results=json.dumps(candidates, indent=2),
        ),
        human="Does any database result match the user's paper mention?",
    )

    if result is None or not result.matched:
        reason = "Pydantic validation failed" if result is None else "No match found"
        logger.info("Paper not resolved for '%s': %s", raw_title, reason)
        return {
            "resolved_paper_id":    None,
            "resolved_paper_title": None,
            "not_found_reason":     NOT_FOUND_MESSAGES["paper_not_found"].format(title=raw_title),
            "not_found_key":        "paper_not_found",
        }

    logger.info(
        "Resolved '%s' → '%s' (confidence: %s)",
        raw_title,
        result.canonical_title,
        result.confidence,
    )
    return {
        "resolved_paper_id":    result.entry_id,
        "resolved_paper_title": result.canonical_title,
    }


# ---------------------------------------------------------------------------
# Node 3: Cypher Generation
# ---------------------------------------------------------------------------

async def cypher_generation_node(
    state: AppState,
    llm: BaseChatModel,
    schema: str,
) -> dict:
    """
    Generate a Cypher query grounded in the live database schema.

    Uses CypherGenerationOutput schema — Pydantic enforces:
      - cypher field is non-empty
      - code fence stripping (validator removes ``` if LLM included them)
      - write operation guard (validator raises if MERGE/CREATE/DELETE present)
      - parameters is a dict (never a list or string)
    """
    entities = state.get("entities", {})
    intent   = state.get("intent", "cypher")
    query    = state.get("user_query", "")

    enriched = dict(entities)
    if state.get("resolved_paper_id"):
        enriched["resolved_entry_id"] = state["resolved_paper_id"]
    if state.get("citing_paper_ids"):
        enriched["citing_paper_ids"] = state["citing_paper_ids"]

    logger.info("Generating Cypher | intent='%s'", intent)

    result: CypherGenerationOutput | None = await _call_structured(
        llm,
        schema=CypherGenerationOutput,
        system=CYPHER_GENERATION_SYSTEM.format(
            schema=schema,
            intent=intent,
            query=query,
            entities=json.dumps(enriched, indent=2),
        ),
        human="Write the Cypher query.",
    )

    if result is None:
        logger.error("Cypher generation failed for: '%s'", query[:100])
        return {
            "error":            "Failed to generate a valid Cypher query.",
            "not_found_reason": NOT_FOUND_MESSAGES["no_results"],
            "not_found_key":    "no_results",
        }

    logger.info("Cypher generated: %s", result.cypher[:200])
    logger.debug("Cypher parameters: %s", result.parameters)

    return {
        "generated_cypher":  result.cypher,
        "cypher_parameters": result.parameters,
    }


# ---------------------------------------------------------------------------
# Node 4: Cypher Execution
# ---------------------------------------------------------------------------

async def cypher_execution_node(
    state: AppState,
    mcp_tools: Neo4jMCPTools,
) -> dict:
    """Execute the generated Cypher via the MCP server. No LLM call."""
    cypher     = state.get("generated_cypher")
    parameters = state.get("cypher_parameters", {})

    if not cypher:
        return {
            "not_found_reason": NOT_FOUND_MESSAGES["no_results"],
            "not_found_key":    "no_results",
        }

    logger.info("Executing Cypher via MCP")
    results = await mcp_tools.read_cypher(cypher, parameters)

    if not results:
        logger.info("Cypher execution returned empty results")
        return {
            "cypher_results":   [],
            "not_found_reason": NOT_FOUND_MESSAGES["no_results"],
            "not_found_key":    "no_results",
        }

    logger.info("Cypher returned %d records", len(results))

    citing_ids: list[str] = []
    if state.get("intent") == "hybrid":
        citing_ids = [
            r.get("entry_id") or r.get("citing_entry_id", "")
            for r in results
            if r.get("entry_id") or r.get("citing_entry_id")
        ]

    return {
        "cypher_results":   results,
        "citing_paper_ids": citing_ids,
    }


# ---------------------------------------------------------------------------
# Node 5: Vector Retrieval (semantic path)
# ---------------------------------------------------------------------------

async def vector_retrieval_node(state: AppState) -> dict:
    """
    Semantic vector search + Cohere rerank. No LLM call.
    Delegates to retrieval.reranker.retrieve_and_rerank_async.
    """
    query = state.get("user_query", "")
    logger.info("Vector retrieval: '%s'", query[:80])

    chunks = await retrieve_and_rerank_async(
        user_query=query,
        top_n=RERANK_TOP_N,
        k=VECTOR_SEARCH_K,
        paper_ids=None,
        **_NEO4J_CONN,
    )

    if not chunks:
        return {
            "retrieved_chunks": [],
            "not_found_reason": NOT_FOUND_MESSAGES["no_results"],
            "not_found_key":    "no_results",
        }

    logger.info("Vector retrieval: %d chunks after rerank", len(chunks))
    return {"retrieved_chunks": chunks}


# ---------------------------------------------------------------------------
# Node 6: Filtered Vector Retrieval (hybrid path)
# ---------------------------------------------------------------------------

async def filtered_retrieval_node(state: AppState) -> dict:
    """
    Vector search scoped to citing papers found via CITES traversal. No LLM call.
    Delegates to retrieval.reranker.retrieve_and_rerank_async with paper_ids.
    """
    query     = state.get("user_query", "")
    paper_ids = state.get("citing_paper_ids", [])

    logger.info(
        "Filtered retrieval: '%s' | %d paper IDs", query[:80], len(paper_ids)
    )

    chunks = await retrieve_and_rerank_async(
        user_query=query,
        top_n=RERANK_TOP_N,
        k=VECTOR_SEARCH_K,
        paper_ids=paper_ids,
        **_NEO4J_CONN,
    )

    if not chunks:
        return {
            "retrieved_chunks": [],
            "not_found_reason": NOT_FOUND_MESSAGES["no_citing_papers"].format(
                title=state.get("resolved_paper_title", "the requested paper")
            ),
            "not_found_key": "no_citing_papers",
        }

    logger.info("Filtered retrieval: %d chunks after rerank", len(chunks))
    return {"retrieved_chunks": chunks}


# ---------------------------------------------------------------------------
# Node 7: Rerank (pass-through)
# ---------------------------------------------------------------------------

async def rerank_node(state: AppState) -> dict:
    """
    Pass-through — reranking already happens inside the retrieval nodes.
    Exists to keep graph wiring intact for future retrieval paths that
    might not have integrated reranking.
    """
    return {"retrieved_chunks": state.get("retrieved_chunks", [])}


# ---------------------------------------------------------------------------
# Node 8: Answer
# Plain text — freeform prose. Not structured output.
# ---------------------------------------------------------------------------

async def answer_node(state: AppState, llm: BaseChatModel) -> dict:
    """
    Synthesise the final prose answer from retrieved context.
    Uses _call_llm_text because the output is freeform — Pydantic schema
    would be wrong here since the answer content cannot be predetermined.
    """
    query    = state.get("user_query", "")
    chunks   = state.get("retrieved_chunks", [])
    cypher_r = state.get("cypher_results", [])

    context_parts: list[str] = []

    for chunk in chunks:
        meta    = chunk.get("metadata", {})
        section = meta.get("section") or ""
        paper   = meta.get("paperTitle") or ""
        header  = "[Paper: " + paper + (f" | Section: {section}" if section else "") + "]"
        context_parts.append(f"{header}\n{chunk['text']}")

    for record in cypher_r:
        context_parts.append(json.dumps(record, indent=2))

    if not context_parts:
        return {"answer": "I could not find enough context to answer your question."}

    context = "\n\n---\n\n".join(context_parts)

    history = state.get("chat_history", [])
    history_text = ""
    if history:
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in history[-6:]
        )

    system = ANSWER_SYSTEM.format(context=context)
    if history_text:
        system += f"\n\n## Previous conversation\n{history_text}"

    answer = await _call_llm_text(
        llm,
        system=system,
        human=ANSWER_HUMAN.format(query=query),
    )

    if not answer:
        answer = "I was unable to generate an answer. Please try again."

    logger.info("Answer generated (%d chars)", len(answer))
    return {"answer": answer}


# ---------------------------------------------------------------------------
# Node 9: Plot
# Plain text — LLM writes Python code. Not structured output.
# ---------------------------------------------------------------------------

async def plot_node(state: AppState, llm: BaseChatModel) -> dict:
    """
    Generate a Plotly chart from Cypher results.
    LLM writes the plot function as freeform code — not structured output
    because the function body cannot be expressed as a Pydantic schema.
    """
    data  = state.get("cypher_results", [])
    query = state.get("user_query", "")

    if not data:
        return {"answer": "No data was returned to plot.", "plot_figure": None}

    plot_code = await _call_llm_text(
        llm,
        system=PLOT_GENERATION_SYSTEM.format(
            data=json.dumps(data[:50], indent=2),
            query=query,
        ),
        human="Write the generate_plot function.",
    )

    if not plot_code:
        return {"answer": "Failed to generate plot code.", "plot_figure": None}

    fig, err = execute_plot_code(plot_code, data)

    if err:
        logger.error("Plot execution error: %s", err)
        return {
            "answer":      f"Chart generation failed: {err}",
            "plot_code":   plot_code,
            "plot_figure": None,
        }

    return {
        "answer":      "Here is the chart based on your query.",
        "plot_code":   plot_code,
        "plot_figure": fig,
    }


# ---------------------------------------------------------------------------
# Node 10: Not Found
# No LLM call — static templates from prompts.py
# ---------------------------------------------------------------------------

async def not_found_node(state: AppState) -> dict:
    """Return a clear user-facing message when retrieval returns nothing."""
    reason = state.get("not_found_reason") or NOT_FOUND_MESSAGES["no_results"]
    logger.info("Not found: %s", reason[:100])
    return {"answer": reason}


# ---------------------------------------------------------------------------
# Node 11: Memory Update
# Plain text — LLM writes a summary paragraph. Not structured output.
# ---------------------------------------------------------------------------

async def memory_update_node(state: AppState, llm: BaseChatModel) -> dict:
    """
    Append the current turn to history.
    Summarises older turns when token budget is exceeded.
    Summary is freeform prose — not structured output.
    """
    query  = state.get("user_query", "")
    answer = state.get("answer", "")

    new_messages = [
        {"role": "user",      "content": query},
        {"role": "assistant", "content": answer},
    ]

    history = list(state.get("chat_history", []))
    history.extend(new_messages)

    history_tokens = sum(len(m["content"]) for m in history) // CHARS_PER_TOKEN

    if history_tokens > HISTORY_TOKEN_LIMIT:
        logger.info("History at %d est. tokens — summarising", history_tokens)
        fresh        = history[-4:]
        to_summarise = history[:-4]
        conversation = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in to_summarise
        )

        summary = await _call_llm_text(
            llm,
            system=MEMORY_SUMMARISATION_SYSTEM,
            human=MEMORY_SUMMARISATION_HUMAN.format(conversation=conversation),
        )

        if summary:
            history = [
                {"role": "system", "content": f"[Earlier conversation summary]: {summary}"}
            ] + fresh

    return {"chat_history": new_messages}