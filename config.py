"""
config.py
---------
Single source of truth for all configuration.
All environment variables, constants, and tunable parameters live here.
Import this module everywhere — never read os.getenv() directly in other files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent.parent
LOG_DIR  = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Neo4j
# ---------------------------------------------------------------------------
NEO4J_URI      = os.getenv("NEO4J_URI", "")
NEO4J_USER     = os.getenv("NEO4J_USER", "616caddc")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "616caddc")

# ---------------------------------------------------------------------------
# LLM providers
# Supported: "gemini", "openai", "azure_openai"
# ---------------------------------------------------------------------------
DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")

GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL        = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")

OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL        = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

AZURE_OPENAI_API_KEY     = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_DEPLOYMENT  = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

# ---------------------------------------------------------------------------
# LangSmith observability
# Set LANGCHAIN_TRACING_V2=true in .env to enable
# ---------------------------------------------------------------------------
LANGSMITH_API_KEY   = os.getenv("LANGCHAIN_API_KEY", "")
LANGSMITH_PROJECT   = os.getenv("LANGCHAIN_PROJECT", "arxiv_trends")
LANGSMITH_ENDPOINT  = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
LANGSMITH_ENABLED   = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"

# Set the env vars LangChain checks internally
if LANGSMITH_ENABLED:
    os.environ["LANGCHAIN_TRACING_V2"]  = "true"
    os.environ["LANGCHAIN_API_KEY"]     = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"]     = LANGSMITH_PROJECT
    os.environ["LANGCHAIN_ENDPOINT"]    = LANGSMITH_ENDPOINT

# ---------------------------------------------------------------------------
# Cohere reranker
# ---------------------------------------------------------------------------
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")

# ---------------------------------------------------------------------------
# Retrieval parameters
# ---------------------------------------------------------------------------
VECTOR_SEARCH_K   = int(os.getenv("VECTOR_SEARCH_K", "50"))   # initial candidates
RERANK_TOP_N      = int(os.getenv("RERANK_TOP_N", "10"))       # after Cohere rerank
MIN_SCORE_THRESHOLD = float(os.getenv("MIN_SCORE_THRESHOLD", "0.5"))

# ---------------------------------------------------------------------------
# Memory / conversation history
# ---------------------------------------------------------------------------
# Max tokens of history before summarisation is triggered
HISTORY_TOKEN_LIMIT = int(os.getenv("HISTORY_TOKEN_LIMIT", "4000"))
# Rough chars-per-token estimate for quick checks without calling a tokeniser
CHARS_PER_TOKEN = 4

# ---------------------------------------------------------------------------
# MCP server (mcp-neo4j-cypher)
# The server is launched as a subprocess via uvx.
# No external service needed — runs locally.
# ---------------------------------------------------------------------------
MCP_SERVER_COMMAND = "uvx"
MCP_SERVER_ARGS    = ["mcp-neo4j-cypher"]
MCP_SERVER_ENV: dict[str, str] = {
    "NEO4J_URI":      NEO4J_URI,
    "NEO4J_USERNAME": NEO4J_USER,
    "NEO4J_PASSWORD": NEO4J_PASSWORD,
    "NEO4J_DATABASE": NEO4J_DATABASE,
    "NEO4J_READ_ONLY": "true",       # safety: only allow read queries via MCP
}

# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
APP_TITLE       = "arXiv Trend Analyser"
APP_ICON        = "📈"
MAX_CHAT_HEIGHT = 600   # px

# ---------------------------------------------------------------------------
# Supported LLM providers (shown in UI)
# ---------------------------------------------------------------------------
SUPPORTED_PROVIDERS = ["gemini", "openai", "azure_openai"]

PROVIDER_DISPLAY_NAMES = {
    "gemini":       "Google Gemini",
    "openai":       "OpenAI",
    "azure_openai": "Azure OpenAI",
}

PROVIDER_REQUIRED_FIELDS: dict[str, list[str]] = {
    "gemini":       ["api_key"],
    "openai":       ["api_key"],
    "azure_openai": ["api_key", "endpoint", "deployment"],
}