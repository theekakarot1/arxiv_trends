"""
llm_factory.py
--------------
Creates and validates LLM clients for all supported providers.

Supported providers: gemini, openai, azure_openai
All return a LangChain BaseChatModel so the rest of the app
is completely provider-agnostic.

Validation: makes a cheap test call (single token) to confirm
the API key and endpoint are valid before accepting them.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel

from config import (
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
    GEMINI_MODEL,
    OPENAI_MODEL,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _test_llm(llm: BaseChatModel) -> tuple[bool, str]:
    """
    Make a minimal call to verify the LLM is reachable and the key is valid.
    Returns (success, error_message).
    """
    try:
        llm.invoke("Hi")
        return True, ""
    except Exception as exc:
        msg = str(exc)
        # Trim long error messages for display
        if len(msg) > 200:
            msg = msg[:200] + "..."
        return False, msg


def _validate_fields(provider: str, **kwargs: Any) -> tuple[bool, str]:
    """Check that all required fields for a provider are non-empty."""
    required: dict[str, list[str]] = {
        "gemini":       ["api_key"],
        "openai":       ["api_key"],
        "azure_openai": ["api_key", "endpoint", "deployment"],
    }
    missing = [f for f in required.get(provider, []) if not kwargs.get(f)]
    if missing:
        return False, f"Missing required fields: {', '.join(missing)}"
    return True, ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_llm(
    provider: str,
    api_key: str,
    endpoint: str = "",
    deployment: str = "",
    model: str = "",
    temperature: float = 0.1,
    validate: bool = True,
) -> tuple[BaseChatModel | None, str]:
    """
    Create and optionally validate an LLM client.

    Parameters
    ----------
    provider    : "gemini" | "openai" | "azure_openai"
    api_key     : API key for the provider
    endpoint    : Azure endpoint URL (azure_openai only)
    deployment  : Azure deployment name (azure_openai only)
    model       : Override the default model name
    temperature : Sampling temperature (default 0.1 for factual tasks)
    validate    : If True, makes a test call to verify credentials

    Returns
    -------
    (llm, error_message)
    llm is None if creation or validation failed.
    """
    # Step 1 — field presence check
    ok, err = _validate_fields(
        provider, api_key=api_key, endpoint=endpoint, deployment=deployment
    )
    if not ok:
        return None, err

    # Step 2 — build the LLM object
    try:
        llm = _build_llm(provider, api_key, endpoint, deployment, model, temperature)
    except ImportError as exc:
        pkg = str(exc).split("'")[1] if "'" in str(exc) else str(exc)
        return None, (
            f"Required package not installed: {pkg}. "
            f"Run: pip install {_install_hint(provider)}"
        )
    except Exception as exc:
        return None, f"Failed to initialise {provider} client: {exc}"

    # Step 3 — live validation
    if validate:
        ok, err = _test_llm(llm)
        if not ok:
            return None, f"API key validation failed for {provider}: {err}"

    logger.info("LLM created successfully: provider=%s", provider)
    return llm, ""


def _build_llm(
    provider: str,
    api_key: str,
    endpoint: str,
    deployment: str,
    model: str,
    temperature: float,
) -> BaseChatModel:
    """Construct the provider-specific LangChain chat model."""

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model or GEMINI_MODEL,
            google_api_key=api_key,
            temperature=temperature,
            max_retries=2,
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model or OPENAI_MODEL,
            api_key=api_key,
            temperature=temperature,
            max_retries=2,
        )

    elif provider == "azure_openai":
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_endpoint=endpoint,
            azure_deployment=deployment or AZURE_OPENAI_DEPLOYMENT,
            api_key=api_key,
            api_version=AZURE_OPENAI_API_VERSION,
            temperature=temperature,
            max_retries=2,
        )

    else:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Supported: gemini, openai, azure_openai"
        )


def _install_hint(provider: str) -> str:
    hints = {
        "gemini":       "langchain-google-genai",
        "openai":       "langchain-openai",
        "azure_openai": "langchain-openai",
    }
    return hints.get(provider, "")


def get_provider_fields(provider: str) -> list[dict]:
    """
    Return the list of UI field specs for a provider.
    Used by the Streamlit sidebar to render the right input fields.
    """
    base_fields = [
        {"key": "api_key", "label": "API Key", "type": "password"},
    ]
    extra: dict[str, list[dict]] = {
        "azure_openai": [
            {"key": "endpoint",   "label": "Azure Endpoint",    "type": "text"},
            {"key": "deployment", "label": "Deployment Name",   "type": "text"},
        ]
    }
    return base_fields + extra.get(provider, [])