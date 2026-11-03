"""
Backend-specialised interpolator implementations.

Each submodule (:mod:`.numpy`, :mod:`.torch`, :mod:`.jax`) provides the same
public interpolator classes (``LinearInterpolator``,
``NearestNeighbourInterpolator``, ``NullInterpolator``, ``SpragueInterpolator``,
``CubicSplineInterpolator``, ``PchipInterpolator``, ``KernelInterpolator`` and
``Extrapolator``), specialised for one array backend and sharing an identical
API.

These submodules are public. Import one to pin a specific backend instead of the
data-driven dispatch of :mod:`colour.algebra.interpolation`::

    from colour.algebra.interpolation.backends import jax as jax_backend

    interpolator = jax_backend.LinearInterpolator(x, y)

Pinning guarantees a native backend regardless of the input array namespace or
the global Array API dispatch state (:func:`colour.utilities.is_array_api_enabled`)
-- e.g. to keep a *PyTorch* or *JAX* implementation for automatic differentiation
or ``jax.jit``, which the dispatcher would otherwise select only for framework
data with dispatch enabled. The *NumPy* backend is always importable; the
*PyTorch* and *JAX* backends require their respective framework.

Submodules are imported lazily on attribute access, so referencing
``backends.torch`` does not import *PyTorch* until used. The kernel functions
shared by the kernel interpolators live in :mod:`._kernels`.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import ModuleType

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

# Public backend submodules, resolved lazily so importing this package never
# pulls in *PyTorch* or *JAX*. ``__all__`` is intentionally left unset: a star
# import would force every framework to import eagerly and fail where one is
# absent.
_BACKENDS = ("numpy", "torch", "jax")


def __getattr__(name: str) -> ModuleType | Any:
    """Import a backend submodule lazily on attribute access (PEP 562)."""

    if name in _BACKENDS:
        return importlib.import_module(f"{__name__}.{name}")

    error = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(error)


def __dir__() -> list[str]:
    """Return the module contents, including the lazy backend submodules."""

    return sorted({*globals(), *_BACKENDS})
