"""
run_ingestion.py
----------------
Orchestrates the full ingestion pipeline:

  1. Connect to Neo4j.
  2. Fetch new arXiv papers (skipping already-ingested ones).
  3. Run NER to extract entities from full text.
  4. Ingest Paper nodes + entity nodes + relationships into Neo4j.
  5. Ingest CITES relationships derived from in-text citation patterns.
  6. Create/ensure the vector index on Chunk nodes.
  7. Embed and store text chunks for semantic retrieval.

Run from the project root:
    python run_ingestion.py

Logging is written to stdout and to logs/ingestion.log.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# ---- Logging setup --------------------------------------------------------
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / "ingestion.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ---- Local imports (after logging is configured) --------------------------
from ingestion.arxiv_loader import get_arxiv_documents
from ingestion.arxiv_ner import group_by_entity
from ingestion.neo4j_loader import (
    close_neo4j_driver,
    connect_to_neo4j,
    create_chunks,
    create_vector_index,
    ingest_citations,
    ingest_chunks_embeddings,
    ingest_papers,
)


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Connect to Neo4j
    # ------------------------------------------------------------------
    logger.info("=== Step 1/7 : Connecting to Neo4j ===")
    driver = connect_to_neo4j()
    if driver is None:
        logger.critical("Could not connect to Neo4j.  Aborting.")
        sys.exit(1)

    try:
        # ------------------------------------------------------------------
        # 2. Fetch new arXiv papers (incremental — skips existing ones)
        # ------------------------------------------------------------------
        logger.info("=== Step 2/7 : Fetching new arXiv papers ===")
        doc_list, doc_details = get_arxiv_documents(driver=driver)

        if not doc_details:
            logger.info("No new papers to ingest.  Database is already up to date.")
            return

        logger.info("Downloaded %d new papers.", len(doc_details))

        # ------------------------------------------------------------------
        # 3. NER — extract entities from full text
        # ------------------------------------------------------------------
        logger.info("=== Step 3/7 : Running NER on paper content ===")
        doc_details = group_by_entity(doc_details)

        # ------------------------------------------------------------------
        # 4. Ingest Paper + entity nodes
        # ------------------------------------------------------------------
        logger.info("=== Step 4/7 : Ingesting Paper and entity nodes ===")
        ingest_papers(doc_details, driver)

        # ------------------------------------------------------------------
        # 5. Ingest CITES relationships
        # ------------------------------------------------------------------
        logger.info("=== Step 5/7 : Ingesting citation relationships ===")
        ingest_citations(doc_details, driver)

        # ------------------------------------------------------------------
        # 6. Create / ensure vector index
        # ------------------------------------------------------------------
        logger.info("=== Step 6/7 : Ensuring vector index on Chunk nodes ===")
        create_vector_index(driver)

        # ------------------------------------------------------------------
        # 7. Embed and store text chunks
        # ------------------------------------------------------------------
        logger.info("=== Step 7/7 : Chunking and embedding paper content ===")
        chunks = create_chunks(doc_list)
        logger.info("Created %d chunks from %d documents.", len(chunks), len(doc_list))
        ingest_chunks_embeddings(driver, chunks)

        logger.info("=== Ingestion complete. ===")

    finally:
        close_neo4j_driver(driver)


if __name__ == "__main__":
    main()