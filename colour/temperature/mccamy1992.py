"""
McCamy (1992) Correlated Colour Temperature
===========================================

Define the *McCamy (1992)* correlated colour temperature :math:`T_{cp}`
computation objects.

-   :func:`colour.temperature.xy_to_CCT_McCamy1992`: Compute correlated
    colour temperature :math:`T_{cp}` from specified *CIE xy* chromaticity
    coordinates using the *McCamy (1992)* method.
-   :func:`colour.temperature.CCT_to_xy_McCamy1992`: Compute *CIE xy*
    chromaticity coordinates from specified correlated colour temperature
    :math:`T_{cp}` using the *McCamy (1992)* method.

References
----------
-   :cite:`Wikipedia2001` : Wikipedia. (2001). Approximation. Retrieved June
    28, 2014, from http://en.wikipedia.org/wiki/Color_temperature#Approximation
"""

from __future__ import annotations

import typing

from colour.algebra import sdiv, sdiv_mode

if typing.TYPE_CHECKING:
    from colour.hints import ArrayLike, NDArrayFloat

from colour.temperature.common import solve_xy_Newton
from colour.utilities import (
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
    "xy_to_CCT_McCamy1992",
    "CCT_to_xy_McCamy1992",
]


def xy_to_CCT_McCamy1992(xy: ArrayLike) -> NDArrayFloat:
    """
    Compute the correlated colour temperature :math:`T_{cp}` from the
    specified *CIE xy* chromaticity coordinates using the *McCamy (1992)*
    method.

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
    :cite:`Wikipedia2001`

    Examples
    --------
    >>> import numpy as np
    >>> xy = np.array([0.31270, 0.32900])
    >>> xy_to_CCT_McCamy1992(xy)  # doctest: +ELLIPSIS
    np.float64(6505.0805913...)
    """

    x, y = tsplit(xy)

    with sdiv_mode():
        n = sdiv(x - 0.3320, y - 0.1858)

    CCT = -449 * n**3 + 3525 * n**2 - 6823.3 * n + 5520.33

    return as_float(CCT)


def CCT_to_xy_McCamy1992(
    CCT: ArrayLike, optimisation_kwargs: dict | None = None
) -> NDArrayFloat:
    """
    Compute the *CIE xy* chromaticity coordinates from the specified
    correlated colour temperature :math:`T_{cp}` using the *McCamy (1992)*
    method.

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
    The *McCamy (1992)* method for computing *CIE xy* chromaticity
    coordinates from the specified correlated colour temperature is not
    a bijective function and might produce unexpected results. It is
    provided for consistency with other correlated colour temperature
    computation methods but should be avoided for practical
    applications. The current implementation seeds a *Tikhonov*-
    regularised damped *Gauss-Newton* iteration anchored to the
    *CIE Standard Illuminant D65* chromaticity coordinates, vectorised
    across all input samples.

    References
    ----------
    :cite:`Wikipedia2001`

    Examples
    --------
    >>> CCT_to_xy_McCamy1992(6505.0805913074782)  # doctest: +ELLIPSIS
    array([0.3127..., 0.329...])
    """

    usage_warning(
        '"McCamy (1992)" method for computing "CIE xy" chromaticity '
        "coordinates from given correlated colour temperature is not a "
        "bijective function and might produce unexpected results. It is given "
        "for consistency with other correlated colour temperature computation "
        "methods but should be avoided for practical applications."
    )

    optimisation_kwargs = dict(optional(optimisation_kwargs, {}))

    CCT = as_float_array(CCT)

    return as_float_array(
        solve_xy_Newton(xy_to_CCT_McCamy1992, CCT, **optimisation_kwargs)
    )
