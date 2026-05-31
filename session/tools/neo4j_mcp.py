from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any
import shutil
import os
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
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Background event loop — one per process, lives forever
# ---------------------------------------------------------------------------

class _BgLoop:
    """
    A single asyncio event loop running in a daemon thread.

    All MCP I/O and LangGraph coroutines are submitted here so that:
      - The subprocess stdin/stdout streams stay on the loop that created them.
      - No coroutine ever crosses loop boundaries.
    """

    _instance: "_BgLoop | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        t = threading.Thread(
            target=self._run, daemon=True, name="neo4j-mcp-bg-loop"
        )
        t.start()
        logger.info("Background event loop started (thread: %s)", t.name)

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def submit(self, coro, timeout: float = 60):
        """Submit a coroutine and block the calling thread until it completes."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    @classmethod
    def get(cls) -> "_BgLoop":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance


def run_on_bg_loop(coro, timeout: float = 60):
    """Public helper used by main.py to replace run_async()."""
    return _BgLoop.get().submit(coro, timeout=timeout)


# ---------------------------------------------------------------------------
# Schema retrieval — native Cypher (no APOC required)
# ---------------------------------------------------------------------------

async def get_neo4j_schema() -> str:
    """
    Retrieve the graph schema using native Neo4j Cypher.
    Works on AuraDB free tier without APOC.
    """
    driver = AsyncGraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
    )
    try:
        async with driver.session(database=NEO4J_DATABASE) as session:
            label_result = await session.run(
                "CALL db.labels() YIELD label RETURN label"
            )
            labels = [r["label"] async for r in label_result]

            rel_result = await session.run(
                "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
            )
            rel_types = [r["relationshipType"] async for r in rel_result]

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

            schema_result = await session.run(
                """
                MATCH (a)-[r]->(b)
                RETURN DISTINCT
                    labels(a)[0]  AS from_label,
                    type(r)       AS rel_type,
                    labels(b)[0]  AS to_label
                LIMIT 200
                """
            )
            relationships = [
                f"(:{r['from_label']})-[:{r['rel_type']}]->(:{r['to_label']})"
                async for r in schema_result
            ]
    finally:
        await driver.close()

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
# Neo4jMCPTools
# ---------------------------------------------------------------------------

class Neo4jMCPTools:
    """
    Wraps the mcp-neo4j-cypher MCP server tools for use in LangGraph nodes.

    Lifecycle
    ---------
    tools = Neo4jMCPTools.create()   # sync, safe to call from Streamlit

    # In a LangGraph node (already running on bg loop):
    results = await tools.read_cypher("MATCH (p:Paper) RETURN p.title LIMIT 5")
    """

    def __init__(
        self,
        langchain_tools: list,
        session: ClientSession,   # already-entered ClientSession
        _cm_stdio: Any,           # already-entered stdio_client ctx mgr
    ) -> None:
        self._tools     = {t.name: t for t in langchain_tools}
        self._session   = session
        self._cm_stdio  = _cm_stdio
        logger.info("MCP tools loaded: %s", list(self._tools.keys()))

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(cls) -> "Neo4jMCPTools":
        """
        Synchronous factory — safe to call from Streamlit's sync context
        or from @st.cache_resource.

        Starts the MCP server subprocess on the background loop and returns
        a ready-to-use instance.  Raises on failure (caller should catch).
        """
        return _BgLoop.get().submit(cls._create_async(), timeout=120)

    @staticmethod
    async def _create_async() -> "Neo4jMCPTools":
        # Resolve full path so subprocess finds uvx regardless of thread PATH
        uvx_path = shutil.which(MCP_SERVER_COMMAND)
        if not uvx_path:
            raise FileNotFoundError(
                f"'{MCP_SERVER_COMMAND}' not found on PATH. "
                f"PATH={os.environ.get('PATH', '<empty>')}"
            )
        logger.info("Resolved MCP command: %s", uvx_path)

        params = StdioServerParameters(
            command=uvx_path,
            args=MCP_SERVER_ARGS,
            env={**os.environ, **MCP_SERVER_ENV},  # full env — don't strip PATH/TEMP
        )

        # ── Step 1: enter stdio_client, keep _cm_stdio alive to prevent GC ──
        _cm_stdio = stdio_client(params)
        read, write = await _cm_stdio.__aenter__()
        logger.info("stdio streams ready")

        # ── Step 2: enter ClientSession — THIS starts the background receive  ──
        # ── loop that reads server responses. Without __aenter__(), the loop  ──
        # ── never starts and initialize() hangs waiting for a reply forever.  ──
        _cm_session = ClientSession(read, write)
        await _cm_session.__aenter__()
        logger.info("ClientSession entered (receive loop started)")

        await _cm_session.initialize()
        logger.info("MCP session initialised successfully")

        tools = await load_mcp_tools(_cm_session)
        logger.info("Tools loaded: %s", [t.name for t in tools])

        return Neo4jMCPTools(tools, _cm_session, _cm_stdio)

    # ------------------------------------------------------------------
    # Cypher execution
    # ------------------------------------------------------------------

    async def read_cypher(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict]:
        """
        Execute a read-only Cypher query via the MCP server.

        This is an async method — call it with `await` from LangGraph nodes.
        Those nodes are invoked via run_on_bg_loop(graph.ainvoke(...)), so
        they run on the same loop as the MCP session. No cross-loop issues.
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
            if isinstance(result, str):
                parsed = json.loads(result)
                return (
                    parsed.get("results", parsed)
                    if isinstance(parsed, dict)
                    else parsed
                )
            return result if isinstance(result, list) else []
        except Exception as exc:
            logger.error("MCP read_cypher failed: %s | query: %.200s", exc, query)
            return []

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Exit both context managers in reverse order."""
        try:
            await self._session.__aexit__(None, None, None)   # stops receive loop
            await self._cm_stdio.__aexit__(None, None, None)  # kills subprocess
            logger.info("MCP session closed")
        except Exception as exc:
            logger.warning("MCP close error: %s", exc)


# ---------------------------------------------------------------------------
# Paper resolution helper (direct Neo4j — not via MCP)
# ---------------------------------------------------------------------------

async def resolve_paper(title: str) -> list[dict]:
    """Fuzzy-match a paper title against Paper nodes in Neo4j."""
    driver = AsyncGraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
    )
    try:
        async with driver.session(database=NEO4J_DATABASE) as session:
            result = await session.run(
                """
                MATCH (p:Paper)
                WHERE toLower(p.title) CONTAINS toLower($title)
                RETURN p.entry_id  AS entry_id,
                       p.title     AS title,
                       p.published AS published,
                       p.summary   AS summary
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

    logger.info("Paper resolution for '%s': %d candidates", title, len(records))
    return records