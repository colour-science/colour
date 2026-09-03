"""
Pytest Configuration
====================

Configure *pytest* with array backend fixtures for *Array API* testing.
"""

from __future__ import annotations

import os
import sys
import typing

import numpy as np
import pytest

if typing.TYPE_CHECKING:
    from colour.hints import Generator, ModuleType

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.utilities import (
    array_api_enable,
    set_default_complex_dtype,
    set_default_float_dtype,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "xp",
]

try:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
except ImportError:
    jnp = None

try:
    import torch

    torch.set_default_dtype(torch.float64)
except ImportError:
    torch = None


_TEST_BACKENDS: str | None = os.environ.get("COLOUR_SCIENCE__TEST_BACKENDS")
"""
Optional comma-separated list of backend parameter ids (``numpy``, ``jax``,
``torch``, ``torch-mps``, ``torch-cuda``) restricting the :func:`xp` fixture
parametrisation. Unset yields every installed backend.
"""


def _make_backend_parameters() -> list:
    """Build the parametrised backend list."""

    params = [pytest.param((np, "numpy"), id="numpy")]

    if jnp is not None:
        params.append(pytest.param((jnp, "jax"), id="jax"))

    if torch is not None:
        params.append(pytest.param((torch, "torch"), id="torch"))
        if torch.backends.mps.is_available():
            params.append(pytest.param((torch, "torch-mps"), id="torch-mps"))
        if torch.cuda.is_available():
            params.append(pytest.param((torch, "torch-cuda"), id="torch-cuda"))

    if _TEST_BACKENDS is None:
        return params

    requested = [token.strip() for token in _TEST_BACKENDS.split(",") if token.strip()]
    available = {str(parameter.id): parameter for parameter in params}

    missing = [backend for backend in requested if backend not in available]
    if missing:
        pytest.exit(
            f"COLOUR_SCIENCE__TEST_BACKENDS requests unavailable backend(s): "
            f"{', '.join(missing)}; available: {', '.join(available)}.",
            returncode=1,
        )

    return [available[backend] for backend in requested]


@pytest.fixture(params=_make_backend_parameters())
def xp(request: pytest.FixtureRequest) -> Generator[ModuleType, None, None]:
    """
    Parametrised array namespace fixture.

    Yields :mod:`numpy` and, when available, :mod:`jax.numpy` and
    :mod:`torch`. Non-NumPy backends automatically enable Array API dispatch
    for the duration of the test. The ``torch-mps`` and ``torch-cuda`` variants
    additionally set the matching default device. The ``torch-mps`` variant
    also sets the default dtype to ``float32``.
    """

    backend, variant = request.param

    if variant == "numpy":
        yield backend
    elif variant == "torch-cuda":
        with array_api_enable(True):
            default_device = torch.get_default_device()  # pyright: ignore
            torch.set_default_device("cuda")  # pyright: ignore

            try:
                yield backend
            finally:
                torch.set_default_device(default_device)  # pyright: ignore
    elif variant == "torch-mps":
        with array_api_enable(True):
            default_dtype = torch.get_default_dtype()  # pyright: ignore
            torch.set_default_dtype(torch.float32)  # pyright: ignore
            torch.set_default_device("mps")  # pyright: ignore
            set_default_float_dtype(np.float32)
            set_default_complex_dtype(np.complex64)

            # Relax test tolerance for float32 precision. A per-test
            # ``@pytest.mark.mps_tolerance_absolute(value)`` marker overrides
            # the ``5e-4`` default for tests whose float32 deltas need more
            # headroom. Tests that thread
            # :attr:`colour.constants.TOLERANCE_ABSOLUTE_TESTS` honour it, as
            # do ``xp_assert_close`` calls relying on the default tolerances
            # (resolved at call time); hard-coded tolerance literals do not.
            marker = request.node.get_closest_marker("mps_tolerance_absolute")
            tolerance = marker.args[0] if marker else 5e-4
            # The original tolerance is snapshotted into a local BEFORE the
            # sweep: the sweep patches every module holding the constant,
            # including this ``conftest`` module, so restoring from the
            # module-level name in the ``finally`` would restore the patched
            # value and leak the relaxed tolerance across the whole worker.
            tolerance_original = TOLERANCE_ABSOLUTE_TESTS
            for module in sys.modules.values():
                if hasattr(module, "TOLERANCE_ABSOLUTE_TESTS"):
                    module.TOLERANCE_ABSOLUTE_TESTS = tolerance  # pyright: ignore

            # Tests that cannot pass at any sane tolerance under float32
            # (large-magnitude radiometry, divergent solvers, hard-coded
            # tolerance literals) opt in to a strict expected failure via
            # ``@pytest.mark.mps_xfail("reason")``. ``strict=True`` makes
            # an unexpected pass a CI failure, so the marker stays honest
            # as *MPS* support improves.
            xfail_marker = request.node.get_closest_marker("mps_xfail")
            if xfail_marker is not None:
                request.node.add_marker(
                    pytest.mark.xfail(
                        reason=xfail_marker.args[0]
                        if xfail_marker.args
                        else "MPS float32",
                        raises=(AssertionError, RuntimeError, TypeError),
                        strict=True,
                    )
                )

            try:
                yield backend
            finally:
                torch.set_default_device("cpu")  # pyright: ignore
                torch.set_default_dtype(default_dtype)  # pyright: ignore
                set_default_float_dtype(np.float64)
                set_default_complex_dtype(np.complex128)

                for module in sys.modules.values():
                    if hasattr(module, "TOLERANCE_ABSOLUTE_TESTS"):
                        module.TOLERANCE_ABSOLUTE_TESTS = (  # pyright: ignore
                            tolerance_original
                        )
    else:
        with array_api_enable(True):
            yield backend
