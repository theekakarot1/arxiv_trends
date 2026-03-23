"""
arxiv_loader.py
---------------
Responsible for fetching arXiv paper metadata and full-text content.

Key behaviours
  - Skips papers already present in Neo4j  (idempotent / incremental ingestion).
  - Extracts cited arXiv IDs from each paper's text so the graph loader can
    later build CITES relationships between Paper nodes.
  - Retries transient arXiv API failures with exponential back-off.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime

import arxiv
from langchain_community.document_loaders import ArxivLoader

logger = logging.getLogger(__name__)

_arxiv_client = arxiv.Client()

# Maximum allowed paper size in characters.
# Papers larger than this are almost certainly scanned books mis-classified as
# arXiv submissions and would bloat Neo4j with useless data.
_MAX_CONTENT_CHARS = 1_000_000


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_arxiv_id(raw: str) -> str:
    """Return the bare, version-free arXiv ID from any representation.

    Examples
    --------
    'https://arxiv.org/abs/2301.07041v2' -> '2301.07041'
    '2301.07041v2'                        -> '2301.07041'
    '2301.07041'                          -> '2301.07041'
    """
    bare = raw.split("abs/")[-1].strip()
    return re.sub(r"v\d+$", "", bare)


def _extract_cited_ids(page_content: str) -> list[str]:
    """Heuristically extract arXiv IDs that are cited inside a paper.

    Scans for common citation patterns:
      - arXiv:2301.07041   (inline citation)
      - arxiv.org/abs/2301.07041  (URL in reference list)

    Returns a deduplicated list of bare IDs (no version suffix).
    """
    patterns = [
        r"arXiv[:\s]+(\d{4}\.\d{4,5})",
        r"arxiv\.org/abs/(\d{4}\.\d{4,5})",
    ]
    found: set[str] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, page_content, re.IGNORECASE):
            found.add(_parse_arxiv_id(match.group(1)))
    return list(found)


def _safe_search(query: str, max_retries: int = 3, base_sleep: float = 5.0) -> list:
    """Execute an arXiv search with exponential back-off on transient errors."""
    for attempt in range(1, max_retries + 1):
        try:
            search = arxiv.Search(
                query,
                max_results=100,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending,
            )
            return list(_arxiv_client.results(search))
        except Exception as exc:
            logger.warning("arXiv search attempt %d/%d failed: %s", attempt, max_retries, exc)
            if attempt < max_retries:
                sleep_secs = base_sleep * (2 ** (attempt - 1))
                logger.info("Retrying in %.0fs …", sleep_secs)
                time.sleep(sleep_secs)
            else:
                logger.error("All retries exhausted for query: %s", query)
                return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_existing_paper_ids(driver) -> set[str]:
    """Return the set of bare arXiv IDs already stored in Neo4j.

    Called before downloading so incremental runs skip already-ingested papers.
    Returns an empty set when the driver is None or the query fails.
    """
    if driver is None:
        return set()
    try:
        records, _, _ = driver.execute_query(
            "MATCH (p:Paper) RETURN p.entry_id AS entry_id",
            database_="neo4j",
        )
        return {
            _parse_arxiv_id(r["entry_id"])
            for r in records
            if r["entry_id"]
        }
    except Exception as exc:
        logger.error("Could not fetch existing paper IDs from Neo4j: %s", exc)
        return set()


def load_single_paper(arxiv_id: str) -> tuple[list, dict]:
    """Download full text and metadata for one paper.

    Returns
    -------
    (doc_list, detail_dict)
        doc_list    – one-element list containing a LangChain Document.
        detail_dict – metadata dict with extra keys:
                        'page_content'    : full paper text
                        'cited_arxiv_ids' : list of arXiv IDs cited by this paper
    Returns ([], {}) on any error or if the paper is too large.
    """
    try:
        loader = ArxivLoader(
            query=arxiv_id,
            load_max_docs=1,
            load_all_available_meta=True,
            load_full_documents=True,
        )
        docs = loader.load()
        if not docs:
            logger.warning("ArxivLoader returned no documents for %s", arxiv_id)
            return [], {}

        doc = docs[0]
        if len(doc.page_content) > _MAX_CONTENT_CHARS:
            logger.warning(
                "Skipping %s — content too large (%d chars)", arxiv_id, len(doc.page_content)
            )
            return [], {}

        detail: dict = dict(doc.metadata)
        detail["page_content"]    = doc.page_content
        detail["cited_arxiv_ids"] = _extract_cited_ids(doc.page_content)

        return [doc], detail

    except Exception as exc:
        logger.error("Failed to load paper %s: %s", arxiv_id, exc)
        return [], {}


def get_arxiv_documents(driver=None) -> tuple[list, list]:
    """Fetch new arXiv papers (cs.AI, 2017-01 → 2025-04), skipping existing ones.

    Parameters
    ----------
    driver : neo4j.Driver | None
        Active Neo4j driver.  When provided, papers already in the database
        are skipped so that repeated runs are safe and efficient.

    Returns
    -------
    doc_list    : list[list[Document]]  – one inner list per paper.
    doc_details : list[dict]            – one metadata dict per paper,
                                          including 'cited_arxiv_ids'.
    """
    # ---- Build bi-monthly date ranges 2017-01 through 2025-04 ----------
    date_ranges: list[tuple[str, str]] = []
    for year in range(2017, 2026):
        month_pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]
        for start_m, end_m in month_pairs:
            if year == 2025 and start_m > 4:
                break
            # Correct end-of-month day
            if end_m == 2:
                end_day = 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28
            elif end_m in (4, 6, 9, 11):
                end_day = 30
            else:
                end_day = 31
            date_ranges.append((f"{year}{start_m:02d}01", f"{year}{end_m:02d}{end_day:02d}"))

    # ---- Load already-ingested IDs to skip them -------------------------
    existing_ids: set[str] = get_existing_paper_ids(driver)
    if existing_ids:
        logger.info("Found %d papers already in Neo4j — these will be skipped.", len(existing_ids))

    # ---- Search arXiv for candidate papers ------------------------------
    candidate_ids: list[str] = []
    for start_date, end_date in date_ranges:
        results = _safe_search(f"cat:cs.AI AND submittedDate:[{start_date} TO {end_date}]")
        if not results:
            continue

        start_label = datetime.strptime(start_date, "%Y%m%d").strftime("%b %d, %Y")
        end_label   = datetime.strptime(end_date,   "%Y%m%d").strftime("%b %d, %Y")
        logger.info("  %s → %s : %d results", start_label, end_label, len(results))

        for result in results:
            has_pdf = any(link.title == "pdf" for link in result.links)
            if not has_pdf:
                continue
            paper_id = _parse_arxiv_id(result.entry_id)
            if paper_id not in existing_ids:
                candidate_ids.append(paper_id)

    # Deduplicate while preserving insertion order
    seen: set[str] = set()
    unique_ids = [p for p in candidate_ids if p not in seen and not seen.add(p)]  # type: ignore[func-returns-value]

    logger.info(
        "Downloading %d new papers (skipped %d already ingested).",
        len(unique_ids),
        len(existing_ids),
    )

    # ---- Download full text for each new paper --------------------------
    doc_list:    list[list] = []
    doc_details: list[dict] = []

    for paper_id in unique_ids:
        docs, detail = load_single_paper(paper_id)
        if docs and detail:
            doc_list.append(docs)
            doc_details.append(detail)

    return doc_list, doc_details