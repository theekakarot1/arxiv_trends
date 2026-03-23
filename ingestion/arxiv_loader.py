"""
arxiv_loader.py
---------------
Responsible for fetching arXiv paper metadata and full-text content.

Extraction strategy
  Raw PDF text from pymupdf (used by ArxivLoader) has several problems for
  chunking: single \n line-wraps inside paragraphs, missing paragraph breaks,
  column-layout artefacts, arXiv watermarks bleeding in mid-text, and equations
  rendering as symbol garbage.

  We use two complementary approaches:
    1. pymupdf4llm.to_markdown() — converts the PDF to Markdown, preserving
       section headings as ## markers, tables, and lists.  This is the primary
       extraction path and feeds the structure-aware chunker in neo4j_loader.py.
    2. clean_arxiv_text() — applied to the markdown output to remove residual
       noise (watermarks, excessive blank lines, hyphenation artefacts).

  The resulting text is stored as `page_content` on the LangChain Document so
  the rest of the pipeline is unchanged.

Key behaviours
  - Skips papers already present in Neo4j  (idempotent / incremental ingestion).
  - Extracts cited arXiv IDs from the cleaned text for the citation graph.
  - Retries transient arXiv API failures with exponential back-off.
  - Downloads the actual PDF to a temp file, extracts with pymupdf4llm, then
    deletes the temp file immediately to avoid disk accumulation.
"""

from __future__ import annotations

import logging
import re
import tempfile
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import arxiv
import pymupdf4llm
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_arxiv_client = arxiv.Client()

# Papers larger than this (chars) after extraction are almost certainly
# mis-classified books.  Skip them to protect Neo4j.
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


def clean_arxiv_text(text: str) -> str:
    """Remove common noise from pymupdf4llm markdown output of arXiv papers.

    Problems addressed
    ------------------
    1. arXiv watermark lines  — e.g. "arXiv:2301.07041v2 [cs.AI] 15 Jan 2023"
       that appear at the top of every page and bleed into the extracted text.
    2. Page-number lines      — standalone integers on their own line.
    3. Hyphenation artefacts  — words broken across lines with a trailing hyphen
       get re-joined: "computa-\ntion" → "computation".
    4. Excessive blank lines  — more than two consecutive blank lines collapsed
       to two so the markdown header splitter sees clean paragraph boundaries.
    5. Trailing whitespace    — stripped from every line.
    """
    # 1. arXiv watermark (appears as a full line)
    text = re.sub(
        r"^arXiv:\d{4}\.\d{4,5}(?:v\d+)?\s*\[[\w.]+\].*$",
        "",
        text,
        flags=re.MULTILINE,
    )
    # 2. Standalone page numbers
    text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.MULTILINE)

    # 3. Re-join hyphenated line-breaks  ("computa-\ntion" → "computation")
    text = re.sub(r"-\n(\S)", r"\1", text)

    # 4. Collapse 3+ consecutive blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 5. Trailing whitespace per line
    text = "\n".join(line.rstrip() for line in text.splitlines())

    return text.strip()


def _extract_cited_ids(text: str) -> list[str]:
    """Heuristically extract arXiv IDs cited inside a paper.

    Scans for common citation patterns:
      arXiv:2301.07041        (inline citation)
      arxiv.org/abs/2301.07041  (URL in reference list)

    Returns a deduplicated list of bare IDs (no version suffix).
    """
    patterns = [
        r"arXiv[:\s]+(\d{4}\.\d{4,5})",
        r"arxiv\.org/abs/(\d{4}\.\d{4,5})",
    ]
    found: set[str] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
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


def _fetch_arxiv_result(arxiv_id: str):
    """Return the arxiv.Result object for a given ID, or None on failure."""
    try:
        search = arxiv.Search(id_list=[arxiv_id])
        results = list(_arxiv_client.results(search))
        return results[0] if results else None
    except Exception as exc:
        logger.error("Could not fetch arXiv result for %s: %s", arxiv_id, exc)
        return None


def _extract_markdown(pdf_url: str) -> str | None:
    """Download a PDF to a temp file, extract markdown, delete the temp file.

    Returns the cleaned markdown string, or None on failure.
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        urllib.request.urlretrieve(pdf_url, tmp_path)
        raw_md = pymupdf4llm.to_markdown(str(tmp_path))
        return clean_arxiv_text(raw_md)

    except Exception as exc:
        logger.error("PDF extraction failed for %s: %s", pdf_url, exc)
        return None

    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()


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
    """Download and extract one arXiv paper as structured Markdown.

    Pipeline
    --------
    1. Fetch paper metadata via the arXiv API.
    2. Download the PDF to a temp file.
    3. Extract structured Markdown using pymupdf4llm (preserves ## headings).
    4. Clean noise with clean_arxiv_text().
    5. Build a LangChain Document with metadata and markdown content.
    6. Delete the temp PDF.

    Returns
    -------
    (doc_list, detail_dict)
        doc_list    – one-element list containing a LangChain Document whose
                      page_content is clean Markdown with ## section headers.
        detail_dict – metadata dict with extra keys:
                        'page_content'    : the clean markdown text
                        'cited_arxiv_ids' : list of arXiv IDs cited by this paper
    Returns ([], {}) on any error or if the paper is too large.
    """
    result = _fetch_arxiv_result(arxiv_id)
    if result is None:
        return [], {}

    # Find the PDF link
    pdf_url = next(
        (link.href for link in result.links if link.title == "pdf"),
        None,
    )
    if not pdf_url:
        logger.warning("No PDF link found for %s", arxiv_id)
        return [], {}

    # Extract markdown from the PDF
    markdown_text = _extract_markdown(pdf_url)
    if not markdown_text:
        return [], {}

    if len(markdown_text) > _MAX_CONTENT_CHARS:
        logger.warning(
            "Skipping %s — content too large (%d chars)", arxiv_id, len(markdown_text)
        )
        return [], {}

    # Build metadata dict mirroring the old ArxivLoader format so the rest of
    # the pipeline (NER, neo4j_loader) is unaffected.
    metadata = {
        "entry_id":   arxiv_id,
        "Title":      result.title,
        "Authors":    ", ".join(a.name for a in result.authors),
        "Published":  result.published.isoformat() if result.published else None,
        "Summary":    result.summary,
        "categories": result.categories,
        "doi":        result.doi,
        "journal_ref":result.journal_ref,
    }

    doc = Document(page_content=markdown_text, metadata=metadata)

    detail: dict = dict(metadata)
    detail["page_content"]    = markdown_text
    detail["cited_arxiv_ids"] = _extract_cited_ids(markdown_text)

    return [doc], detail


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