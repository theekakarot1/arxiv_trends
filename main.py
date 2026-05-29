"""
main.py
-------
Streamlit application entry point.

Run with:
    streamlit run app/main.py

Features:
  - Multi-provider LLM configuration with live validation
  - LangGraph-powered query processing (semantic / cypher / hybrid)
  - Plotly chart rendering for visualisation queries
  - Thumbs up/down feedback per response (sent to LangSmith)
  - Conversation memory with automatic summarisation
  - Full observability via LangSmith tracing
"""

from __future__ import annotations
from langsmith import Client
import asyncio
import logging
import sys
from pathlib import Path
import concurrent.futures
import streamlit as st

# Make the app/ directory importable when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    APP_ICON,
    APP_TITLE,
    LANGSMITH_API_KEY,
    LANGSMITH_ENABLED,
    LANGSMITH_PROJECT,
    PROVIDER_DISPLAY_NAMES,
    SUPPORTED_PROVIDERS,
)
from graph.builder import build_graph
from llm_factory import create_llm, get_provider_fields
from memory.manager import build_initial_state, format_history_for_display
from session.tools.neo4j_mcp import Neo4jMCPTools, get_neo4j_schema

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/app.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Async helpers
# ---------------------------------------------------------------------------

def run_async(coro):
    """Run an async coroutine from Streamlit's sync context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Cached resources — initialised once per session
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Connecting to Neo4j MCP server…")
def get_mcp_tools() -> Neo4jMCPTools | None:
    """Start the MCP server subprocess and load tools. Cached for the session."""
    try:
        tools = run_async(Neo4jMCPTools.create())
        logger.info("MCP tools initialised")
        return tools
    except Exception as exc:
        logger.error("Failed to initialise MCP tools: %s", exc)
        return None


@st.cache_resource(show_spinner="Fetching Neo4j schema…")
def get_schema() -> str:
    """Fetch the database schema once and cache it."""
    try:
        schema = run_async(get_neo4j_schema())
        logger.info("Schema cached (%d chars)", len(schema))
        return schema
    except Exception as exc:
        logger.error("Schema fetch failed: %s", exc)
        return "Schema unavailable."


def get_compiled_graph(llm):
    """Build the LangGraph graph. Cached by LLM identity."""
    mcp_tools = get_mcp_tools()
    schema    = get_schema()

    if mcp_tools is None:
        st.error(
            "Could not connect to Neo4j MCP server. "
            "Check your Neo4j credentials in .env and ensure `uvx` is installed."
        )
        st.stop()

    return build_graph(llm, mcp_tools, schema)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def _init_session():
    defaults = {
        "messages":         [],      # display messages: {role, content, figure?, run_id?}
        "chat_history":     [],      # full history passed to graph
        "llm":              None,
        "graph":            None,
        "provider":         None,
        "llm_configured":   False,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ---------------------------------------------------------------------------
# Sidebar — LLM configuration
# ---------------------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.title(f"{APP_ICON} {APP_TITLE}")
        st.divider()

        # ---- LLM Provider selection ------------------------------------
        st.subheader("LLM Configuration")

        provider = st.selectbox(
            "Provider",
            SUPPORTED_PROVIDERS,
            format_func=lambda p: PROVIDER_DISPLAY_NAMES[p],
            key="selected_provider",
        )

        fields = get_provider_fields(provider)
        field_values: dict[str, str] = {}

        for field in fields:
            if field["type"] == "password":
                field_values[field["key"]] = st.text_input(
                    field["label"],
                    type="password",
                    key=f"field_{field['key']}",
                )
            else:
                field_values[field["key"]] = st.text_input(
                    field["label"],
                    key=f"field_{field['key']}",
                )

        if st.button("Connect", use_container_width=True, type="primary"):
            with st.spinner("Validating API key…"):
                llm, err = create_llm(
                    provider=provider,
                    api_key=field_values.get("api_key", ""),
                    endpoint=field_values.get("endpoint", ""),
                    deployment=field_values.get("deployment", ""),
                    validate=True,
                )

            if err:
                st.error(f"Connection failed: {err}")
                st.session_state.llm_configured = False
            else:
                st.session_state.llm           = llm
                st.session_state.provider       = provider
                st.session_state.graph          = get_compiled_graph(llm)
                st.session_state.llm_configured = True
                st.success(f"Connected to {PROVIDER_DISPLAY_NAMES[provider]}")
                logger.info("LLM configured: provider=%s", provider)

        st.divider()

        # ---- Status indicators -----------------------------------------
        st.subheader("Status")

        neo4j_ok  = get_mcp_tools() is not None
        schema_ok = get_schema() != "Schema unavailable."

        col1, col2 = st.columns(2)
        col1.metric("Neo4j MCP", "✓" if neo4j_ok  else "✗")
        col2.metric("Schema",    "✓" if schema_ok else "✗")

        if LANGSMITH_ENABLED:
            st.caption(f"LangSmith: {LANGSMITH_PROJECT}")
        else:
            st.caption("LangSmith: disabled (set LANGCHAIN_TRACING_V2=true)")

        st.divider()

        # ---- Clear conversation ----------------------------------------
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages     = []
            st.session_state.chat_history = []
            st.rerun()


# ---------------------------------------------------------------------------
# Feedback submission
# ---------------------------------------------------------------------------

def submit_feedback(run_id: str | None, score: int, message_idx: int):
    """Send thumbs up/down feedback to LangSmith."""
    if not LANGSMITH_ENABLED or not run_id:
        return

    try:
        client = Client(api_key=LANGSMITH_API_KEY)
        client.create_feedback(
            run_id=run_id,
            key="user_feedback",
            score=score,          # 1 = thumbs up, 0 = thumbs down
            comment=f"User feedback on message {message_idx}",
        )
        logger.info("Feedback submitted: run_id=%s score=%d", run_id, score)
    except Exception as exc:
        logger.warning("Feedback submission failed: %s", exc)


# ---------------------------------------------------------------------------
# Main chat interface
# ---------------------------------------------------------------------------

def render_chat():
    st.title(APP_TITLE)

    if not st.session_state.llm_configured:
        st.info(
            "Configure your LLM provider in the sidebar to begin. "
            "Your API key is validated before any queries are sent."
        )
        return

    # Display existing messages
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Render plot if present
            if message.get("figure") is not None:
                st.plotly_chart(message["figure"], use_container_width=True)

            # Feedback buttons (only on assistant messages)
            if message["role"] == "assistant":
                run_id = message.get("run_id")
                col1, col2, col3 = st.columns([1, 1, 8])

                feedback_key = f"feedback_{idx}"

                if col1.button("👍", key=f"up_{idx}"):
                    submit_feedback(run_id, score=1, message_idx=idx)
                    st.session_state[feedback_key] = "positive"

                if col2.button("👎", key=f"down_{idx}"):
                    submit_feedback(run_id, score=0, message_idx=idx)
                    st.session_state[feedback_key] = "negative"

                if st.session_state.get(feedback_key) == "positive":
                    col3.caption("Thanks for the feedback!")
                elif st.session_state.get(feedback_key) == "negative":
                    col3.caption("Thanks — we'll improve.")

    # Chat input
    if prompt := st.chat_input("Ask about arXiv papers, trends, or authors…"):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process with LangGraph
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                result = _run_graph(prompt)

            answer = result.get("answer") or "Something went wrong. Please try again."
            figure = result.get("plot_figure")
            run_id = result.get("langsmith_run_id")
            error  = result.get("error")

            # Show error if graph failed
            if error:
                st.error(f"An error occurred: {error}")
                logger.error("Graph error: %s", error)

            st.markdown(answer)

            if figure is not None:
                st.plotly_chart(figure, use_container_width=True)

            # Debug expander: show query classification
            if st.session_state.get("show_debug"):
                with st.expander("Query analysis"):
                    st.write({
                        "intent":      result.get("intent"),
                        "output_type": result.get("output_type"),
                        "entities":    result.get("entities"),
                        "reasoning":   result.get("query_reasoning"),
                    })

        # Save assistant message
        assistant_msg: dict = {
            "role":    "assistant",
            "content": answer,
            "run_id":  run_id,
        }
        if figure is not None:
            assistant_msg["figure"] = figure

        st.session_state.messages.append(assistant_msg)

        # Update chat history for next turn
        st.session_state.chat_history = list(
            result.get("chat_history", st.session_state.chat_history)
        )


def _run_graph(user_query: str) -> dict:
    """Invoke the LangGraph graph and return the final state."""
    graph = st.session_state.graph

    initial_state = build_initial_state(
        user_query=user_query,
        chat_history=list(st.session_state.chat_history),
    )

    try:
        final_state = run_async(graph.ainvoke(initial_state))
        return final_state
    except Exception as exc:
        logger.error("Graph invocation failed: %s", exc, exc_info=True)
        return {
            "answer": "An unexpected error occurred. Please try again.",
            "error":  str(exc),
        }


# ---------------------------------------------------------------------------
# Sidebar extras
# ---------------------------------------------------------------------------

def render_sidebar_extras():
    with st.sidebar:
        st.divider()
        st.subheader("Settings")
        st.session_state.show_debug = st.toggle(
            "Show query analysis",
            value=False,
            help="Show how the system classified your query (intent, entities, etc.)",
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    _init_session()
    render_sidebar()
    render_sidebar_extras()
    render_chat()


if __name__ == "__main__":
    main()