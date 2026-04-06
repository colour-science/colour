"""
Hernandez-Andres, Lee and Romero (1999) Correlated Colour Temperature
=====================================================================

Define *Hernandez-Andres et al. (1999)* correlated colour temperature
:math:`T_{cp}` computation objects.

-   :func:`colour.temperature.xy_to_CCT_Hernandez1999`: Compute correlated
    colour temperature :math:`T_{cp}` from specified *CIE xy* chromaticity
    coordinates using *Hernandez-Andres, Lee and Romero (1999)* method.
-   :func:`colour.temperature.CCT_to_xy_Hernandez1999`: Compute *CIE xy*
    chromaticity coordinates from specified correlated colour temperature
    :math:`T_{cp}` using *Hernandez-Andres, Lee and Romero (1999)* method.

References
----------
-   :cite:`Hernandez-Andres1999a` : Hernández-Andrés, J., Lee, R. L., &
    Romero, J. (1999). Calculating correlated color temperatures across the
    entire gamut of daylight and skylight chromaticities. Applied Optics,
    38(27), 5703. doi:10.1364/AO.38.005703
"""

from __future__ import annotations

import typing

from colour.algebra import sdiv, sdiv_mode

if typing.TYPE_CHECKING:
    from colour.hints import ArrayLike, NDArrayFloat

from colour.temperature.common import solve_xy_Newton
from colour.utilities import (
    array_namespace,
    as_float,
    as_float_array,
    optional,
    tsplit,
    usage_warning,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "xy_to_CCT_Hernandez1999",
    "CCT_to_xy_Hernandez1999",
]


def xy_to_CCT_Hernandez1999(xy: ArrayLike) -> NDArrayFloat:
    """
    Compute the correlated colour temperature :math:`T_{cp}` from the
    specified *CIE xy* chromaticity coordinates using
    *Hernandez-Andres et al. (1999)* method.

    Parameters
    ----------
    xy
        *CIE xy* chromaticity coordinates.

    Returns
    -------
    :class:`numpy.ndarray`
        Correlated colour temperature :math:`T_{cp}`.

    References
    ----------
    :cite:`Hernandez-Andres1999a`

    Examples
    --------
    >>> import numpy as np
    >>> xy = np.array([0.31270, 0.32900])
    >>> xy_to_CCT_Hernandez1999(xy)  # doctest: +ELLIPSIS
    np.float64(6500.7420431...)
    """

    xp = array_namespace(xy)

    x, y = tsplit(xy)

    with sdiv_mode():
        n = sdiv(x - 0.3366, y - 0.1735)

    CCT = (
        -949.86315
        + 6253.80338 * xp.exp(-n / 0.92159)
        + 28.70599 * xp.exp(-n / 0.20039)
        + 0.00004 * xp.exp(-n / 0.07125)
    )

    n = xp.where(CCT > 50000, (x - 0.3356) / (y - 0.1691), n)

    CCT = xp.where(
        CCT > 50000,
        36284.48953
        + 0.00228 * xp.exp(-n / 0.07861)
        + 5.4535e-36 * xp.exp(-n / 0.01543),
        CCT,
    )

    return as_float(CCT)


def CCT_to_xy_Hernandez1999(
    CCT: ArrayLike, optimisation_kwargs: dict | None = None
) -> NDArrayFloat:
    """
    Compute the *CIE xy* chromaticity coordinates from the specified
    correlated colour temperature :math:`T_{cp}` using
    *Hernandez-Andres et al. (1999)* method.

    Parameters
    ----------
    CCT
        Correlated colour temperature :math:`T_{cp}`.
    optimisation_kwargs
        Inversion parameters forwarded to
        :func:`colour.temperature.solve_xy_Newton`. Accepted keys are
        ``x0``, ``reference_xy``, ``reference_weight``,
        ``newton_iterations``, ``backtrack_iterations`` and ``tolerance``.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE xy* chromaticity coordinates.

    Warnings
    --------
    *Hernandez-Andres et al. (1999)* method for computing *CIE xy*
    chromaticity coordinates from the specified correlated colour
    temperature is not a bijective function and might produce unexpected
    results. It is provided for consistency with other correlated colour
    temperature computation methods but should be avoided for practical
    applications. The current implementation seeds a *Tikhonov*-
    regularised damped *Gauss-Newton* iteration anchored to the
    *CIE Standard Illuminant D65* chromaticity coordinates, vectorised
    across all input samples.

    References
    ----------
    :cite:`Hernandez-Andres1999a`

    Examples
    --------
    >>> CCT_to_xy_Hernandez1999(6500.7420431786531)  # doctest: +ELLIPSIS
    array([0.3127..., 0.329...])
    """

    usage_warning(
        '"Hernandez-Andres et al. (1999)" method for computing "CIE xy" '
        "chromaticity coordinates from given correlated colour temperature is "
        "not a bijective function and might produce unexpected results. It is "
        "given for consistency with other correlated colour temperature "
        "computation methods but should be avoided for practical applications."
    )

    optimisation_kwargs = dict(optional(optimisation_kwargs, {}))

    CCT = as_float_array(CCT)

    return as_float_array(
        solve_xy_Newton(xy_to_CCT_Hernandez1999, CCT, **optimisation_kwargs)
    )
