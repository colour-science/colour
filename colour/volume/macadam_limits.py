"""
Optimal Colour Stimuli - MacAdam Limits
=======================================

Defines the objects related to *Optimal Colour Stimuli* computations.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay

from colour.constants import EPSILON
from colour.hints import ArrayLike, Literal, NDArrayFloat
from colour.models import xyY_to_XYZ
from colour.utilities import (
    CACHE_REGISTRY,
    is_caching_enabled,
    validate_method,
)
from colour.volume import OPTIMAL_COLOUR_STIMULI_ILLUMINANTS

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "is_within_macadam_limits",
]

_CACHE_OPTIMAL_COLOUR_STIMULI_XYZ: dict = CACHE_REGISTRY.register_cache(
    f"{__name__}._CACHE_OPTIMAL_COLOUR_STIMULI_XYZ"
)

_CACHE_OPTIMAL_COLOUR_STIMULI_XYZ_TRIANGULATIONS: dict = CACHE_REGISTRY.register_cache(
    f"{__name__}._CACHE_OPTIMAL_COLOUR_STIMULI_XYZ_TRIANGULATIONS"
)


def _XYZ_optimal_colour_stimuli(
    illuminant: Literal["A", "C", "D65"] | str = "D65",
) -> NDArrayFloat:
    """
    Return given illuminant *Optimal Colour Stimuli* in *CIE XYZ* tristimulus
    values and caches it if not existing.

    Parameters
    ----------
    illuminant
        Illuminant name.

    Returns
    -------
    :class:`numpy.ndarray`
        Illuminant *Optimal Colour Stimuli*.
    """

    illuminant = validate_method(
        illuminant,
        tuple(OPTIMAL_COLOUR_STIMULI_ILLUMINANTS),
        '"{0}" illuminant is invalid, it must be one of {1}!',
    )

    optimal_colour_stimuli = OPTIMAL_COLOUR_STIMULI_ILLUMINANTS[illuminant]

    vertices = _CACHE_OPTIMAL_COLOUR_STIMULI_XYZ.get(illuminant)

    if is_caching_enabled() and vertices is not None:
        return vertices

    _CACHE_OPTIMAL_COLOUR_STIMULI_XYZ[illuminant] = vertices = (
        xyY_to_XYZ(optimal_colour_stimuli) / 100
    )

    return vertices


def is_within_macadam_limits(
    xyY: ArrayLike,
    illuminant: Literal["A", "C", "D65"] | str = "D65",
    tolerance: float = 100 * EPSILON,
) -> NDArrayFloat:
    """
    Return whether given *CIE xyY* colourspace array is within MacAdam limits
    of given illuminant.

    Parameters
    ----------
    xyY
        *CIE xyY* colourspace array.
    illuminant
        Illuminant name.
    tolerance
        Tolerance allowed in the inside-triangle check.

    Returns
    -------
    :class:`numpy.ndarray`
        Whether given *CIE xyY* colourspace array is within MacAdam limits.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``xyY``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    Examples
    --------
    >>> is_within_macadam_limits(np.array([0.3205, 0.4131, 0.51]), "A")
    array(True, dtype=bool)
    >>> a = np.array([[0.3205, 0.4131, 0.51], [0.0005, 0.0031, 0.001]])
    >>> is_within_macadam_limits(a, "A")
    array([ True, False], dtype=bool)
    """

    optimal_colour_stimuli = _XYZ_optimal_colour_stimuli(illuminant)
    triangulation = _CACHE_OPTIMAL_COLOUR_STIMULI_XYZ_TRIANGULATIONS.get(illuminant)

    if triangulation is None:
        _CACHE_OPTIMAL_COLOUR_STIMULI_XYZ_TRIANGULATIONS[illuminant] = (
            triangulation
        ) = Delaunay(optimal_colour_stimuli)

    simplex = triangulation.find_simplex(xyY_to_XYZ(xyY), tol=tolerance)
    simplex = np.where(simplex >= 0, True, False)

    return simplex
