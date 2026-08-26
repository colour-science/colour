"""
CIE Illuminant D Series Correlated Colour Temperature
=====================================================

Define the *CIE Illuminant D Series* correlated colour temperature
:math:`T_{cp}` computation objects.

-   :func:`colour.temperature.xy_to_CCT_CIE_D`: Compute correlated colour
    temperature :math:`T_{cp}` of a *CIE Illuminant D Series* from its
    *CIE xy* chromaticity coordinates.
-   :func:`colour.temperature.CCT_to_xy_CIE_D`: Compute *CIE xy*
    chromaticity coordinates of a *CIE Illuminant D Series* from its
    correlated colour temperature :math:`T_{cp}`.

References
----------
-   :cite:`Wyszecki2000z` : Wyszecki, Günther, & Stiles, W. S. (2000). CIE
    Method of Calculating D-Illuminants. In Color Science: Concepts and
    Methods, Quantitative Data and Formulae (pp. 145-146). Wiley.
    ISBN:978-0-471-39918-6
"""

from __future__ import annotations

import typing

from colour.colorimetry import daylight_locus_function

if typing.TYPE_CHECKING:
    from colour.hints import ArrayLike, NDArrayFloat

from colour.temperature.common import (
    CCT_INVERSION_GRID_SAMPLES,
    solve_CCT_Newton,
    x0_CCT_grid,
)
from colour.utilities import (
    array_namespace,
    as_float,
    as_float_array,
    optional,
    tstack,
    usage_warning,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "xy_to_CCT_CIE_D",
    "CCT_to_xy_CIE_D",
]


def xy_to_CCT_CIE_D(
    xy: ArrayLike, optimisation_kwargs: dict | None = None
) -> NDArrayFloat:
    """
    Compute the correlated colour temperature :math:`T_{cp}` of a
    *CIE Illuminant D Series* from the specified *CIE xy* chromaticity
    coordinates.

    Parameters
    ----------
    xy
        *CIE xy* chromaticity coordinates.
    optimisation_kwargs
        Inversion parameters forwarded to
        :func:`colour.temperature.x0_CCT_grid` and
        :func:`colour.temperature.solve_CCT_Newton`. Accepted keys are
        ``samples`` (grid density for the initial guess, default
        :attr:`colour.temperature.CCT_INVERSION_GRID_SAMPLES`),
        ``newton_iterations``, ``backtrack_iterations`` and ``tolerance``
        (forwarded to :func:`solve_CCT_Newton`).

    Returns
    -------
    :class:`numpy.ndarray`
        Correlated colour temperature :math:`T_{cp}`.

    Warnings
    --------
    The *CIE Illuminant D Series* method does not provide an analytical inverse
    transformation to compute the correlated colour temperature :math:`T_{cp}`
    from the specified *CIE xy* chromaticity coordinates. The current
    implementation seeds a damped *Gauss-Newton* iteration with a
    nearest-neighbour lookup against a coarse grid sampled from the
    analytical forward, vectorised across all input samples.

    References
    ----------
    :cite:`Wyszecki2000z`

    Examples
    --------
    >>> xy_to_CCT_CIE_D([0.31270775, 0.32911283])  # doctest: +ELLIPSIS
    np.float64(6504.389564...)
    """

    optimisation_kwargs = dict(optional(optimisation_kwargs, {}))

    xy = as_float_array(xy)

    x0 = x0_CCT_grid(
        CCT_to_xy_CIE_D,
        xy,
        (4000.0, 25000.0),
        samples=optimisation_kwargs.pop("samples", CCT_INVERSION_GRID_SAMPLES),
    )

    return as_float(solve_CCT_Newton(CCT_to_xy_CIE_D, xy, x0=x0, **optimisation_kwargs))


def CCT_to_xy_CIE_D(CCT: ArrayLike) -> NDArrayFloat:
    """
    Compute the *CIE xy* chromaticity coordinates of a
    *CIE Illuminant D Series* from the specified correlated colour temperature
    :math:`T_{cp}`.

    Parameters
    ----------
    CCT
        Correlated colour temperature :math:`T_{cp}`.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE xy* chromaticity coordinates.

    Raises
    ------
    ValueError
        If the correlated colour temperature is not in the appropriate
        domain.

    References
    ----------
    :cite:`Wyszecki2000z`

    Examples
    --------
    >>> CCT_to_xy_CIE_D(6504.38938305)  # doctest: +ELLIPSIS
    array([0.3127077..., 0.3291128...])
    """

    CCT = as_float_array(CCT)

    xp = array_namespace(CCT)

    if xp.any(xp.logical_or(CCT < 4000, CCT > 25000)):
        usage_warning(
            "Correlated colour temperature must be in domain "
            "[4000, 25000], unpredictable results may occur!"
        )

    CCT_3 = CCT**3
    CCT_2 = CCT**2

    x = as_float(
        xp.where(
            CCT <= 7000,
            -4.607 * 10**9 / CCT_3
            + 2.9678 * 10**6 / CCT_2
            + 0.09911 * 10**3 / CCT
            + 0.244063,
            -2.0064 * 10**9 / CCT_3
            + 1.9018 * 10**6 / CCT_2
            + 0.24748 * 10**3 / CCT
            + 0.23704,
        )
    )

    y = daylight_locus_function(x)

    return tstack([x, y])
