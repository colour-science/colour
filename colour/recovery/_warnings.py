"""Warnings shared by spectral recovery methods."""

from __future__ import annotations

from colour.utilities import array_namespace, usage_warning

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = []


def warn_if_optimisation_detaches(XYZ: object, method: str) -> None:
    """Warn when a recovery optimisation detaches a backend array."""

    namespace = array_namespace(XYZ).__name__.split(".")
    if not {"jax", "torch"}.intersection(namespace):
        return

    usage_warning(
        f'"{method}" spectral recovery uses "SciPy" optimisation and does not '
        "preserve PyTorch or JAX automatic differentiation graphs. Use "
        'method="Gaussian" or method="Mallett 2019" for gradient-preserving '
        "spectral recovery at regular points."
    )
