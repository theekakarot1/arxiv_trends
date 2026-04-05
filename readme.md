# arXiv Trend Analyser — Setup Guide

## Prerequisites

- Python 3.10+
- An active Neo4j AuraDB instance (free tier works)
- At least one LLM provider API key (Gemini, OpenAI, or Azure OpenAI)
- `uv` package manager (for running the MCP server)

---

## Step 1 — Install `uv`

The Neo4j MCP server is run via `uvx`, which comes with `uv`.

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Verify
uvx --version
```

`uvx` runs packages from PyPI in isolated environments without
permanently installing them. `mcp-neo4j-cypher` will be downloaded
automatically the first time the app starts.

---

## Step 2 — Install Python dependencies

```bash
pip install -r requirements.txt
```

---

## Step 3 — Configure environment variables

Create a `.env` file in the project root (same level as `requirements.txt`):

```env
# ── Neo4j ──────────────────────────────────────────────────────────────────
NEO4J_URI=neo4j+ssc://xxxxxxxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

# ── LLM providers (fill in the ones you have) ──────────────────────────────
GEMINI_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_DEPLOYMENT=gpt-4o

# ── Cohere reranker ────────────────────────────────────────────────────────
COHERE_API_KEY=your_cohere_key

# ── LangSmith observability (optional but recommended) ─────────────────────
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=arxiv_trends
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

### Getting free API keys

| Service | Free tier | Link |
|---|---|---|
| Google Gemini | Yes — Gemini 2.0 Flash is free | https://aistudio.google.com/apikey |
| Cohere | Yes — free tier for reranking | https://cohere.com |
| LangSmith | Yes — free developer tier | https://smith.langchain.com |
| Neo4j AuraDB | Yes — free instance | https://neo4j.com/cloud/aura-free |

---

## Step 4 — Run the ingestion pipeline

Before running the app, make sure your Neo4j database is populated:

```bash
# From the project root
python run_ingestion.py
```

This fetches arXiv papers, runs NER, stores Paper/Chunk nodes,
and builds citation relationships. See `ingestion/` for details.

---

## Step 5 — Run the app

```bash
streamlit run app/main.py
```

The app will:
1. Start the Neo4j MCP server subprocess (via `uvx mcp-neo4j-cypher`)
2. Fetch the database schema and cache it
3. Open the UI at `http://localhost:8501`

---

## How the Neo4j MCP server works

The `mcp-neo4j-cypher` package runs as a **subprocess** on your local
machine. It is NOT a cloud service — your Neo4j credentials stay local.

```
Your app (Python)
    ↕  stdio
mcp-neo4j-cypher process
    ↕  bolt://
Neo4j AuraDB
```

When you start the Streamlit app, it launches `uvx mcp-neo4j-cypher`
as a background process. The app communicates with it over stdin/stdout.
When the app stops, the subprocess is automatically cleaned up.

### Why not use the MCP server's `get_neo4j_schema` tool?

That tool requires the APOC plugin, which is not available on AuraDB
free tier. We implemented schema retrieval ourselves using native
Cypher (`CALL db.labels()`, `CALL db.relationshipTypes()`) which
works on all Neo4j instances. Everything else uses the MCP server.

---

## LangSmith setup (optional but highly recommended)

LangSmith gives you full observability into every query:
- Which nodes were visited in the graph
- What the LLM received and returned at each step
- Token counts and latency per step
- User feedback (thumbs up/down) tied to specific runs

1. Create a free account at https://smith.langchain.com
2. Generate an API key under Settings → API Keys
3. Add it to your `.env` as `LANGCHAIN_API_KEY`
4. Set `LANGCHAIN_TRACING_V2=true`

No code changes needed — tracing is automatic for all LangChain/LangGraph calls.

---

## Using the app

### Configuring your LLM

On first launch, the sidebar shows "LLM Configuration". Select your
provider, enter your API key, and click **Connect**. The app validates
the key by making a test call before accepting it.

You can switch providers mid-session by entering new credentials and
clicking Connect again.

### Query types the system handles

| Query type | Example | How it works |
|---|---|---|
| Semantic | "What are the key challenges in training LLMs?" | Vector similarity search over paper chunks |
| Structured filter | "How many cs.AI papers were published in 2024?" | Cypher query via MCP server |
| Plot | "Show me a chart of papers per year" | Cypher → data → LLM generates Plotly code |
| Citation lookup | "Which papers cite the BERT paper?" | Cypher CITES traversal |
| Hybrid | "What improvements have been proposed over Attention is All You Need?" | CITES traversal → filtered vector search |

### Feedback

Every assistant response has a 👍 / 👎 button. Clicking it sends
feedback to LangSmith, where it is attached to the specific run
that generated that response. Over time this builds an evaluation
dataset you can use to measure improvements.

### Debug mode

Toggle "Show query analysis" in the sidebar to see how the system
classified your query — useful for understanding why a particular
retrieval strategy was chosen.

---

## Project structure

```
arxiv_trends/
├── run_ingestion.py            Ingestion pipeline entry point
├── ingestion/                  Data loading, chunking, NER, Neo4j writes
├── retrieval/                  Legacy retriever (still used during ingestion eval)
├── notebooks/                  Chunk size analysis
└── app/
    ├── main.py                 Streamlit UI
    ├── config.py               All env vars and constants
    ├── prompts.py              All LLM prompts
    ├── llm_factory.py          Provider selection + validation
    ├── graph/
    │   ├── state.py            AppState TypedDict
    │   ├── nodes.py            All LangGraph node functions
    │   ├── edges.py            All conditional edge routing functions
    │   └── builder.py          Graph construction
    ├── tools/
    │   ├── neo4j_mcp.py        MCP server connection + schema retrieval
    │   └── plot_tools.py       Safe Plotly code execution
    └── memory/
        └── manager.py          History utilities
```

---

## Troubleshooting

### "Could not connect to Neo4j MCP server"

1. Confirm `uvx` is installed: `uvx --version`
2. Test the MCP server directly:
   ```bash
   NEO4J_URI=your_uri NEO4J_USERNAME=neo4j NEO4J_PASSWORD=your_pass \
   uvx mcp-neo4j-cypher
   ```
   If it prints a startup message, it works.
3. Check your `.env` has the correct `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`.

### "API key validation failed"

- Gemini: ensure you are using an API key from Google AI Studio,
  not Google Cloud. They are different.
- Azure OpenAI: the endpoint must be the full URL including
  `https://` and ending in `.openai.azure.com`.
- OpenAI: keys start with `sk-`.

### "Schema unavailable"

Your Neo4j instance is reachable but the schema query failed.
Run this in Neo4j Browser to verify:
```cypher
CALL db.labels() YIELD label RETURN label
```
If this works, the schema fetch will work.

### LangSmith traces not appearing

- Confirm `LANGCHAIN_TRACING_V2=true` in your `.env`
- Confirm the `LANGCHAIN_API_KEY` is correct
- Traces appear in the LangSmith UI under the project name
  set in `LANGCHAIN_PROJECT` (default: `arxiv_trends`)