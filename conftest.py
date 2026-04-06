"""
Pytest Configuration
====================

Project-level pytest configuration and hooks.
"""

from __future__ import annotations

import os
import sys
from typing import Any


def pytest_configure(config: Any) -> None:
    """
    Configure pytest for the project test suite.

    Notes
    -----
    GitHub Actions on Linux occasionally hangs near the end of the suite when
    running with xdist enabled (e.g., ``-n 2``). This hook forces a single-process
    run in that environment to avoid deadlocks / stuck workers and CI timeouts.
    """
    if os.environ.get("GITHUB_ACTIONS") == "true" and sys.platform.startswith("linux"):
        # Disable xdist / parallel execution.
        if getattr(config.option, "numprocesses", None):
            config.option.numprocesses = 0
        if getattr(config.option, "dist", None):
            config.option.dist = "no"

    # NOTE: The *Colour* warning filters are registered here rather than via the
    # ``filterwarnings`` ini option on purpose: an
    # ``ignore::colour.utilities.ColourWarning`` *ini* entry is resolved during
    # pytest's initial-conftest phase, forcing ``import colour`` *before*
    # *pytest-cov* starts recording in each xdist worker, which leaves all
    # import-time code measured untraced (reported as ``0%`` coverage).
    # ``addinivalue_line`` appends the same filters post-configure, deferring
    # their category resolution (and the ``import colour`` it triggers) to test
    # time, under coverage, while preserving the per-test filtering behaviour.
    for warning in (
        "ignore::colour.utilities.ColourWarning",
        "ignore::colour.utilities.ColourRuntimeWarning",
        "ignore::colour.utilities.ColourUsageWarning",
    ):
        config.addinivalue_line("filterwarnings", warning)
