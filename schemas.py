"""
schemas.py
----------
Pydantic models for every structured LLM call in the application.

Why Pydantic + with_structured_output instead of manual json.loads()
----------------------------------------------------------------------
1. Type safety — field types are enforced at parse time. If the LLM
   returns year as an integer when a string is expected, Pydantic
   coerces or raises, not silently passes the wrong type downstream.

2. Enum validation — Literal types mean the LLM cannot return
   "Semantic" or "SEMANTIC" when "semantic" is required. Invalid
   values raise ValidationError immediately.

3. Required vs optional fields — Required fields that are missing
   raise ValidationError. Optional fields get their default values.
   No more silent .get("field", fallback) hiding LLM failures.

4. Provider-native JSON mode — with_structured_output uses the
   provider's built-in JSON schema enforcement (Gemini response_schema,
   OpenAI function calling / response_format) which is more reliable
   than prompting the LLM to produce valid JSON.

5. No code fence stripping — the old _call_llm_json had to manually
   strip ```json fences. with_structured_output bypasses that entirely.

One model per LLM call. Plain text responses (answer synthesis,
memory summarisation, plot code) are not structured — they stay
as _call_llm_text because their output is freeform prose or code.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Query Understanding
# Used by: query_understanding_node
# ---------------------------------------------------------------------------

class QueryEntities(BaseModel):
    """Entities extracted from the user query."""

    paper_title: str | None = Field(
        default=None,
        description="Exact paper title as written by the user, or null if none mentioned.",
    )
    author: str | None = Field(
        default=None,
        description="Author name if mentioned, else null.",
    )
    year: str | None = Field(
        default=None,
        description="4-digit year if mentioned, else null. Always a string.",
    )
    topic: str = Field(
        description="Main research topic. Always populated, infer from the query.",
    )

    @field_validator("year")
    @classmethod
    def validate_year(cls, v: str | None) -> str | None:
        if v is None:
            return v
        # Strip any surrounding whitespace or quotes the LLM might add
        v = str(v).strip().strip("'\"")
        if not v.isdigit() or len(v) != 4:
            return None   # reject malformed years silently
        return v


class QueryUnderstandingOutput(BaseModel):
    """Structured output for the query understanding node."""

    intent: Literal["semantic", "cypher", "hybrid"] = Field(
        description=(
            "semantic: concept/topic search via vector similarity. "
            "cypher: structured graph query (counts, listings, citations metadata). "
            "hybrid: specific paper named AND content from inside it is needed."
        ),
    )
    output_type: Literal["text", "plot", "number"] = Field(
        description=(
            "text: prose answer expected. "
            "plot: user explicitly wants a chart or visualisation. "
            "number: user wants a count or specific numeric fact."
        ),
    )
    entities: QueryEntities = Field(
        description="Entities extracted from the query.",
    )
    reasoning: str = Field(
        description="One sentence explaining the classification decision.",
        max_length=300,
    )


# ---------------------------------------------------------------------------
# Paper Resolution
# Used by: paper_resolution_node
# ---------------------------------------------------------------------------

class PaperResolutionOutput(BaseModel):
    """Structured output for the paper resolution node."""

    matched: bool = Field(
        description="True if a database result clearly matches the user's paper mention.",
    )
    entry_id: str | None = Field(
        default=None,
        description="The entry_id from the database if matched, else null.",
    )
    canonical_title: str | None = Field(
        default=None,
        description="The exact title string from the database if matched, else null.",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence level of the match.",
    )
    reasoning: str = Field(
        description="One sentence explaining the match decision.",
        max_length=300,
    )

    @field_validator("entry_id", "canonical_title", mode="before")
    @classmethod
    def empty_string_to_none(cls, v: Any) -> Any:
        """Some LLMs return empty string instead of null."""
        if v == "":
            return None
        return v


# ---------------------------------------------------------------------------
# Cypher Generation
# Used by: cypher_generation_node
# ---------------------------------------------------------------------------

class CypherGenerationOutput(BaseModel):
    """Structured output for the Cypher generation node."""

    cypher: str = Field(
        description=(
            "The complete Cypher READ query. "
            "Must not include WRITE operations. "
            "Must not be wrapped in markdown code fences."
        ),
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Query parameters as key-value pairs. "
            "e.g. {\"year\": 2024, \"title\": \"attention\"}"
        ),
    )
    reasoning: str = Field(
        description="One sentence explaining the query strategy chosen.",
        max_length=300,
    )

    @field_validator("cypher")
    @classmethod
    def strip_code_fences(cls, v: str) -> str:
        """Strip any markdown code fences the LLM included despite instructions."""
        v = v.strip()
        if v.startswith("```"):
            lines = v.split("\n")
            # Remove first line (```cypher or ```) and last line (```)
            inner = lines[1:] if lines[-1].strip() == "```" else lines[1:]
            v = "\n".join(inner).strip()
            if v.endswith("```"):
                v = v[:-3].strip()
        return v

    @field_validator("cypher")
    @classmethod
    def reject_write_operations(cls, v: str) -> str:
        """Hard guard: reject any query containing write keywords."""
        write_keywords = ["CREATE ", "MERGE ", "DELETE ", "SET ", "REMOVE ", "DROP "]
        v_upper = v.upper()
        for kw in write_keywords:
            if kw in v_upper:
                raise ValueError(
                    f"Cypher query contains write operation '{kw.strip()}'. "
                    "Only READ queries are permitted."
                )
        return v