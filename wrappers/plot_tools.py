"""
tools/plot_tools.py
-------------------
Handles Plotly chart generation from LLM-generated code.

Security note: LLM-generated code is executed with exec(). This is contained
to a restricted namespace with only the libraries needed for plotting.
The MCP server runs in read-only mode so no data mutation can happen,
and the exec namespace has no access to os, sys, or file system.
"""

from __future__ import annotations

import logging
import textwrap
from typing import Any

logger = logging.getLogger(__name__)

# Safe namespace for exec — only plotting libraries
_SAFE_EXEC_GLOBALS: dict[str, Any] = {}


def _build_safe_globals() -> dict[str, Any]:
    """Build a restricted globals dict for exec."""
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    return {
        "__builtins__": {
            "len": len, "range": range, "enumerate": enumerate,
            "list": list, "dict": dict, "str": str, "int": int,
            "float": float, "bool": bool, "print": print,
            "zip": zip, "map": map, "filter": filter,
            "sorted": sorted, "min": min, "max": max, "sum": sum,
            "round": round, "abs": abs,
        },
        "pd": pd,
        "px": px,
        "go": go,
    }


def execute_plot_code(
    code: str,
    data: list[dict],
) -> tuple[Any | None, str]:
    """
    Execute LLM-generated plot code safely.

    Parameters
    ----------
    code : str
        Python code defining a function `generate_plot(data)`.
        Must return a plotly Figure.
    data : list[dict]
        The data to pass to the function.

    Returns
    -------
    (figure, error_message)
    figure is None if execution failed.
    """
    # Dedent and clean the code
    code = textwrap.dedent(code).strip()

    # Add required imports at the top
    full_code = (
        "import pandas as pd\n"
        "import plotly.express as px\n"
        "import plotly.graph_objects as go\n\n"
        + code
    )

    safe_globals = _build_safe_globals()
    local_scope: dict[str, Any] = {}

    try:
        exec(full_code, safe_globals, local_scope)  # noqa: S102
    except SyntaxError as exc:
        err = f"Syntax error in generated code: {exc}"
        logger.error(err)
        return None, err
    except Exception as exc:
        err = f"Error executing generated code: {exc}"
        logger.error(err)
        return None, err

    plot_func = local_scope.get("generate_plot")
    if not callable(plot_func):
        err = "Generated code did not define a callable `generate_plot(data)` function."
        logger.error(err)
        return None, err

    try:
        fig = plot_func(data)
        return fig, ""
    except Exception as exc:
        err = f"Error running generate_plot(data): {exc}"
        logger.error(err)
        return None, err