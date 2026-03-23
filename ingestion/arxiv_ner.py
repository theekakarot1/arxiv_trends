"""
arxiv_ner.py
------------
Runs the custom spaCy NER model over raw paper text and merges the
extracted entities into each document's metadata dict.

The NER model (arxiv_ner_model/) was trained on LLM-annotated arXiv
abstracts and full texts.  It recognises seven entity types:
  Models & Algorithms, Datasets, Metrics, Libraries & Frameworks,
  Tasks, Theories & Concepts, Institutions.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from ingestion.ner_model import model as ner_model

logger = logging.getLogger(__name__)


def group_by_entity(doc_details: list[dict]) -> list[dict]:
    """Annotate each document dict with NER-extracted entity lists.

    For every paper in doc_details, the full page_content is passed through
    the spaCy NER model.  Extracted entities are deduplicated per label and
    stored as lists under their label name, e.g.:

        doc["Models & Algorithms"] = ["BERT", "GPT-2", "ResNet"]
        doc["Datasets"]            = ["ImageNet", "SQuAD"]

    These keys map directly to the Neo4j relationship types created in
    neo4j_loader.ingest_papers().

    Parameters
    ----------
    doc_details : list[dict]
        List of metadata dicts as returned by arxiv_loader.get_arxiv_documents().
        Each dict must have a 'page_content' key.

    Returns
    -------
    The same list with entity keys added in-place.
    """
    for doc in doc_details:
        content = doc.get("page_content", "")
        if not content:
            logger.warning("Document %s has no page_content; skipping NER.", doc.get("entry_id"))
            continue

        try:
            ner_doc = ner_model(content)
        except Exception as exc:
            logger.error("NER failed for %s: %s", doc.get("entry_id"), exc)
            continue

        entity_dict: dict[str, set[str]] = defaultdict(set)
        for ent in ner_doc.ents:
            # Normalise whitespace and store unique values only.
            entity_dict[ent.label_].add(" ".join(ent.text.split()))

        # Merge into the document dict as sorted lists for determinism.
        for label, values in entity_dict.items():
            doc[label] = sorted(values)

    return doc_details