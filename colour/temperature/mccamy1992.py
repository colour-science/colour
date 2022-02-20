"""
McCamy (1992) Correlated Colour Temperature
===========================================

Defines the *McCamy (1992)* correlated colour temperature :math:`T_{cp}`
computations objects:

-   :func:`colour.temperature.xy_to_CCT_McCamy1992`: Correlated colour
    temperature :math:`T_{cp}` computation of given *CIE xy* chromaticity
    coordinates using *McCamy (1992)* method.
-   :func:`colour.temperature.xy_to_CCT_McCamy1992`: *CIE xy* chromaticity
    coordinates computation of given correlated colour temperature
    :math:`T_{cp}` using *McCamy (1992)* method.

References
----------
-   :cite:`Wikipedia2001` : Wikipedia. (2001). Approximation. Retrieved June
    28, 2014, from http://en.wikipedia.org/wiki/Color_temperature#Approximation
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from colour.colorimetry import CCS_ILLUMINANTS
from colour.hints import (
    ArrayLike,
    Dict,
    FloatingOrArrayLike,
    FloatingOrNDArray,
    NDArray,
    Optional,
)
from colour.utilities import as_float_array, as_float, tsplit, usage_warning

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "xy_to_CCT_McCamy1992",
    "CCT_to_xy_McCamy1992",
]


def xy_to_CCT_McCamy1992(xy: ArrayLike) -> FloatingOrNDArray:
    """
    Return the correlated colour temperature :math:`T_{cp}` from given
    *CIE xy* chromaticity coordinates using *McCamy (1992)* method.

    Parameters
    ----------
    xy
        *CIE xy* chromaticity coordinates.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Correlated colour temperature :math:`T_{cp}`.

    References
    ----------
    :cite:`Wikipedia2001`

    Examples
    --------
    >>> import numpy as np
    >>> xy = np.array([0.31270, 0.32900])
    >>> xy_to_CCT_McCamy1992(xy)  # doctest: +ELLIPSIS
    6505.0805913...
    """

    x, y = tsplit(xy)

    n = (x - 0.3320) / (y - 0.1858)
    CCT = -449 * n**3 + 3525 * n**2 - 6823.3 * n + 5520.33

    return as_float(CCT)


def CCT_to_xy_McCamy1992(
    CCT: FloatingOrArrayLike, optimisation_kwargs: Optional[Dict] = None
) -> NDArray:
    """
    Return the *CIE xy* chromaticity coordinates from given correlated colour
    temperature :math:`T_{cp}` using *McCamy (1992)* method.

    Parameters
    ----------
    CCT
        Correlated colour temperature :math:`T_{cp}`.
    optimisation_kwargs
        Parameters for :func:`scipy.optimize.minimize` definition.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE xy* chromaticity coordinates.

    Warnings
    --------
    *McCamy (1992)* method for computing *CIE xy* chromaticity coordinates
    from given correlated colour temperature is not a bijective function and
    might produce unexpected results. It is given for consistency with other
    correlated colour temperature computation methods but should be avoided
    for practical applications. The current implementation relies on
    optimization using :func:`scipy.optimize.minimize` definition and thus has
    reduced precision and poor performance.

    References
    ----------
    :cite:`Wikipedia2001`

    Examples
    --------
    >>> CCT_to_xy_McCamy1992(6505.0805913074782)  # doctest: +ELLIPSIS
    array([ 0.3127...,  0.329...])
    """

    usage_warning(
        '"McCamy (1992)" method for computing "CIE xy" '
        "chromaticity coordinates from given correlated colour "
        "temperature is not a bijective function and might produce "
        "unexpected results. It is given for consistency with other "
        "correlated colour temperature computation methods but "
        "should be avoided for practical applications."
    )

    CCT = as_float_array(CCT)
    shape = list(CCT.shape)
    CCT = np.atleast_1d(CCT.reshape([-1, 1]))

    def objective_function(
        xy: ArrayLike, CCT: FloatingOrArrayLike
    ) -> FloatingOrNDArray:
        """Objective function."""

        objective = np.linalg.norm(
            xy_to_CCT_McCamy1992(xy) - as_float_array(CCT)
        )

        return as_float(objective)

    optimisation_settings = {
        "method": "Nelder-Mead",
        "options": {
            "fatol": 1e-10,
        },
    }
    if optimisation_kwargs is not None:
        optimisation_settings.update(optimisation_kwargs)

    xy = as_float_array(
        [
            minimize(
                objective_function,
                x0=CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"][
                    "D65"
                ],
                args=(CCT_i,),
                **optimisation_settings,
            ).x
            for CCT_i in as_float_array(CCT)
        ]
    )

    return np.reshape(xy, (shape + [2]))
