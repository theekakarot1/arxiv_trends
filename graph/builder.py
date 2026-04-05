"""
graph/builder.py
----------------
Constructs the compiled LangGraph StateGraph.

The graph is built once and cached. It receives the LLM, MCP tools,
and schema as dependencies — keeping nodes pure and testable.

Graph structure:
  START
    ↓
  query_understanding
    ↓ (conditional)
    ├── paper_resolution  ──→  cypher_generation
    ├── vector_retrieval  ──→  rerank ──→ answer
    └── cypher_generation ──→  cypher_execution
                                  ↓ (conditional)
                                  ├── filtered_retrieval → rerank → answer
                                  ├── plot
                                  ├── answer
                                  └── not_found
  all terminal paths → memory_update → END
"""

from __future__ import annotations

import functools
import logging

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph


from ..graph.edges import (
    route_after_answer,
    route_after_cypher_execution,
    route_after_not_found,
    route_after_paper_resolution,
    route_after_query_understanding,
    route_after_rerank,
    route_after_retrieval,
)
from ..graph.nodes import (
    answer_node,
    cypher_execution_node,
    cypher_generation_node,
    filtered_retrieval_node,
    memory_update_node,
    not_found_node,
    paper_resolution_node,
    plot_node,
    query_understanding_node,
    rerank_node,
    vector_retrieval_node,
)
from ..graph.state import AppState
from ..session.tools.neo4j_mcp import Neo4jMCPTools

logger = logging.getLogger(__name__)


def build_graph(
    llm: BaseChatModel,
    mcp_tools: Neo4jMCPTools,
    schema: str,
) -> "CompiledStateGraph":
    """
    Build and compile the LangGraph StateGraph.

    Parameters
    ----------
    llm       : The LLM client (provider-agnostic BaseChatModel)
    mcp_tools : Initialised MCP tool wrapper for Neo4j
    schema    : Pre-fetched Neo4j schema string (from get_neo4j_schema())

    Returns
    -------
    A compiled LangGraph graph ready to invoke.
    """
    graph = StateGraph(AppState)

    # ----------------------------------------------------------------
    # Bind dependencies to node functions using functools.partial
    # This keeps node functions pure (no global state) and testable.
    # ----------------------------------------------------------------

    def _node(func, **kwargs):
        """Create a partial with bound dependencies."""
        return functools.partial(func, **kwargs)

    # ----------------------------------------------------------------
    # Register nodes
    # ----------------------------------------------------------------

    graph.add_node(
        "query_understanding",
        _node(query_understanding_node, llm=llm),
    )
    graph.add_node(
        "paper_resolution",
        _node(paper_resolution_node, llm=llm),
    )
    graph.add_node(
        "cypher_generation",
        _node(cypher_generation_node, llm=llm, schema=schema),
    )
    graph.add_node(
        "cypher_execution",
        _node(cypher_execution_node, mcp_tools=mcp_tools),
    )
    graph.add_node(
        "vector_retrieval",
        vector_retrieval_node,         # no extra deps — uses config directly
    )
    graph.add_node(
        "filtered_retrieval",
        filtered_retrieval_node,
    )
    graph.add_node(
        "rerank",
        rerank_node,
    )
    graph.add_node(
        "answer",
        _node(answer_node, llm=llm),
    )
    graph.add_node(
        "plot",
        _node(plot_node, llm=llm),
    )
    graph.add_node(
        "not_found",
        not_found_node,
    )
    graph.add_node(
        "memory_update",
        _node(memory_update_node, llm=llm),
    )

    # ----------------------------------------------------------------
    # Register edges
    # ----------------------------------------------------------------

    # Entry point
    graph.add_edge(START, "query_understanding")

    # After query understanding — conditional branch
    graph.add_conditional_edges(
        "query_understanding",
        route_after_query_understanding,
        {
            "paper_resolution": "paper_resolution",
            "vector_retrieval": "vector_retrieval",
            "cypher_generation": "cypher_generation",
        },
    )

    # After paper resolution — conditional branch
    graph.add_conditional_edges(
        "paper_resolution",
        route_after_paper_resolution,
        {
            "not_found":        "not_found",
            "cypher_generation": "cypher_generation",
        },
    )

    # Cypher always flows through generation → execution
    graph.add_edge("cypher_generation", "cypher_execution")

    # After cypher execution — conditional branch
    graph.add_conditional_edges(
        "cypher_execution",
        route_after_cypher_execution,
        {
            "not_found":          "not_found",
            "filtered_retrieval": "filtered_retrieval",
            "plot":               "plot",
            "answer":             "answer",
        },
    )

    # Vector retrieval → rerank (conditional on empty results)
    graph.add_conditional_edges(
        "vector_retrieval",
        route_after_retrieval,
        {
            "not_found": "not_found",
            "rerank":    "rerank",
        },
    )

    # Filtered retrieval → rerank (conditional on empty results)
    graph.add_conditional_edges(
        "filtered_retrieval",
        route_after_retrieval,
        {
            "not_found": "not_found",
            "rerank":    "rerank",
        },
    )

    # Rerank → answer (always)
    graph.add_conditional_edges(
        "rerank",
        route_after_rerank,
        {"answer": "answer"},
    )

    # Answer → memory update → END
    graph.add_conditional_edges(
        "answer",
        route_after_answer,
        {"memory_update": "memory_update"},
    )

    # Plot → memory update → END
    graph.add_edge("plot", "memory_update")

    # Not found → memory update → END
    graph.add_conditional_edges(
        "not_found",
        route_after_not_found,
        {"memory_update": "memory_update"},
    )

    # Memory update is always the final node
    graph.add_edge("memory_update", END)

    compiled = graph.compile()
    logger.info("LangGraph graph compiled successfully")
    return compiled