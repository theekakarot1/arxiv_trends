"""
tools/neo4j_mcp.py
------------------
Manages the connection to the mcp-neo4j-cypher MCP server and exposes
the tools the LangGraph nodes use.

## Why this design

The mcp-neo4j-cypher server runs as a subprocess (via uvx) and communicates
over stdio. LangChain's MCP adapter wraps its tools as standard LangChain
StructuredTool objects, making them callable from LangGraph nodes just like
any other tool.

## Tools exposed by the MCP server
  - read_neo4j_cypher   : execute a read-only Cypher query
  - write_neo4j_cypher  : execute a write Cypher query (disabled — read-only mode)

## Schema retrieval
get_neo4j_schema from the MCP server requires APOC, which is not available
on AuraDB free tier. We implement schema retrieval ourselves using native
Cypher (db.labels(), db.relationshipTypes(), db.schema.visualization()).
This is called once at startup and cached.

## Lifecycle
The MCP server process must stay alive for the duration of the app session.
It is started once and managed via an async context manager that Streamlit
caches as a resource.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from neo4j import AsyncGraphDatabase
from config import (
    MCP_SERVER_ARGS,
    MCP_SERVER_COMMAND,
    MCP_SERVER_ENV,
    NEO4J_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema retrieval — native Cypher (no APOC required)
# ---------------------------------------------------------------------------

async def get_neo4j_schema() -> str:
    """
    Retrieve the graph schema using native Neo4j Cypher.
    Works on AuraDB free tier without APOC.

    Returns a human-readable schema description that the Cypher Generation
    node includes in its prompt so the LLM generates valid queries.
    """
    driver = AsyncGraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
    )
    try:
        async with driver.session(database=NEO4J_DATABASE) as session:
            # Node labels
            label_result = await session.run("CALL db.labels() YIELD label RETURN label")
            labels = [r["label"] async for r in label_result]

            # Relationship types
            rel_result = await session.run(
                "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
            )
            rel_types = [r["relationshipType"] async for r in rel_result]

            # Property keys per label (sample one node per label)
            label_props: dict[str, list[str]] = {}
            for label in labels:
                try:
                    prop_result = await session.run(
                        f"MATCH (n:{label}) RETURN keys(n) AS props LIMIT 1"
                    )
                    record = await prop_result.single()
                    label_props[label] = record["props"] if record else []
                except Exception:
                    label_props[label] = []

            # Relationships with source → target
            schema_result = await session.run(
                """
                MATCH (a)-[r]->(b)
                RETURN DISTINCT
                    labels(a)[0]    AS from_label,
                    type(r)         AS rel_type,
                    labels(b)[0]    AS to_label
                LIMIT 200
                """
            )
            relationships = [
                f"(:{r['from_label']})-[:{r['rel_type']}]->(:{r['to_label']})"
                async for r in schema_result
            ]

    finally:
        await driver.close()

    # Format as readable schema string for the prompt
    lines = ["## Node labels and their properties"]
    for label in sorted(labels):
        props = label_props.get(label, [])
        prop_str = ", ".join(props) if props else "no properties sampled"
        lines.append(f"  :{label}  [{prop_str}]")

    lines.append("\n## Relationship types")
    for rt in sorted(rel_types):
        lines.append(f"  :{rt}")

    lines.append("\n## Relationships (from → rel → to)")
    for rel in sorted(set(relationships)):
        lines.append(f"  {rel}")

    schema_text = "\n".join(lines)
    logger.info("Schema retrieved: %d labels, %d rel types", len(labels), len(rel_types))
    return schema_text


# ---------------------------------------------------------------------------
# MCP tool wrappers — used by LangGraph nodes
# ---------------------------------------------------------------------------

class Neo4jMCPTools:
    """
    Wraps the mcp-neo4j-cypher MCP server tools for use in LangGraph nodes.

    Usage
    -----
    tools = await Neo4jMCPTools.create()

    # In a LangGraph node:
    results = await tools.read_cypher(
        "MATCH (p:Paper) WHERE p.published.year = $year RETURN p.title LIMIT 10",
        {"year": 2024}
    )
    """

    def __init__(self, langchain_tools: list) -> None:
        self._tools = {t.name: t for t in langchain_tools}
        logger.info(
            "MCP tools loaded: %s", list(self._tools.keys())
        )

    @classmethod
    async def create(cls) -> "Neo4jMCPTools":
        """
        Start the MCP server subprocess and load its tools.
        This is an async classmethod so it can be awaited at startup.
        """
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        from langchain_mcp_adapters.tools import load_mcp_tools

        server_params = StdioServerParameters(
            command=MCP_SERVER_COMMAND,
            args=MCP_SERVER_ARGS,
            env=MCP_SERVER_ENV,
        )

        # We hold references to read/write so the session stays open.
        # The caller is responsible for calling .close() when done.
        read, write = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _start_mcp_sync(server_params),
        )

        session = ClientSession(read, write)
        await session.initialize()

        tools = await load_mcp_tools(session)
        instance = cls(tools)
        instance._session = session
        return instance

    async def read_cypher(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict]:
        """
        Execute a read-only Cypher query via the MCP server.

        Returns a list of record dicts.
        Falls back to empty list on error (with logging).
        """
        tool = self._tools.get("read_neo4j_cypher")
        if tool is None:
            logger.error("read_neo4j_cypher tool not found in MCP session")
            return []

        payload: dict[str, Any] = {"query": query}
        if parameters:
            payload["parameters"] = parameters

        try:
            result = await tool.ainvoke(payload)
            # MCP returns a JSON string — parse it
            if isinstance(result, str):
                parsed = json.loads(result)
                # The server returns {"results": [...]} or just [...]
                return parsed.get("results", parsed) if isinstance(parsed, dict) else parsed
            return result if isinstance(result, list) else []
        except Exception as exc:
            logger.error("MCP read_cypher failed: %s | query: %s", exc, query[:200])
            return []

    async def close(self) -> None:
        """Close the MCP session gracefully."""
        try:
            if hasattr(self, "_session"):
                await self._session.close()
        except Exception as exc:
            logger.warning("Error closing MCP session: %s", exc)


def _start_mcp_sync(server_params: Any) -> tuple:
    """
    Helper to start the MCP stdio client synchronously
    (for use inside run_in_executor).
    This is needed because stdio_client is an async context manager
    but we need to start it from a sync context during Streamlit init.
    """
    # This approach uses a dedicated event loop for the MCP connection.
    loop = asyncio.new_event_loop()
    return loop.run_until_complete(_start_mcp_async(server_params))


async def _start_mcp_async(server_params: Any) -> tuple:
    from mcp.client.stdio import stdio_client
    # stdio_client returns (read_stream, write_stream)
    # We enter the context manager and return the streams.
    # The context manager keeps the subprocess alive.
    cm = stdio_client(server_params)
    read, write = await cm.__aenter__()
    return read, write


# ---------------------------------------------------------------------------
# Paper resolution helper (uses Neo4j directly — not via MCP)
# Fuzzy title match using native Cypher
# ---------------------------------------------------------------------------

async def resolve_paper(title: str) -> list[dict]:
    """
    Fuzzy-match a paper title against Paper nodes in Neo4j.

    Returns up to 5 candidate matches with entry_id, title, published.
    Returns [] if nothing is found.
    """
    driver = AsyncGraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
    )
    try:
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.run(
                """
                MATCH (p:Paper)
                WHERE toLower(p.title) CONTAINS toLower($title)
                RETURN p.entry_id    AS entry_id,
                       p.title       AS title,
                       p.published   AS published,
                       p.summary     AS summary
                ORDER BY p.published DESC
                LIMIT 5
                """,
                {"title": title},
            )
            records = [
                {
                    "entry_id":  r["entry_id"],
                    "title":     r["title"],
                    "published": str(r["published"]),
                    "summary":   r["summary"],
                }
                async for r in result
            ]
    finally:
        await driver.close()

    logger.info(
        "Paper resolution for '%s': %d candidates found", title, len(records)
    )
    return records