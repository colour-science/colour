# conftest.py
from __future__ import annotations

import os
import sys


def pytest_configure(config) -> None:
    # GitHub Actions + Linux: desabilita xdist para evitar deadlocks/child processes
    # que podem travar o fim da suite e estourar 6h.
    if os.environ.get("GITHUB_ACTIONS") == "true" and sys.platform.startswith("linux"):
        # Se xdist estiver ativo, forçamos execução single-process.
        if getattr(config.option, "numprocesses", None):
            config.option.numprocesses = 0
        if getattr(config.option, "dist", None):
            config.option.dist = "no"
