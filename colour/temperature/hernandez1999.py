# -*- coding: utf-8 -*-
"""
Hernandez-Andres, Lee and Romero (1999) Correlated Colour Temperature
=====================================================================

Defines the *Hernandez-Andres et al. (1999)* correlated colour temperature
:math:`T_{cp}` computations objects:

-   :func:`colour.temperature.xy_to_CCT_Hernandez1999`: Correlated colour
    temperature :math:`T_{cp}` computation of given *CIE xy* chromaticity
    coordinates using *Hernandez-Andres, Lee and Romero (1999)* method.
-   :func:`colour.temperature.CCT_to_xy_Hernandez1999`: *CIE xy* chromaticity
    coordinates computation of given correlated colour temperature
    :math:`T_{cp}` using *Hernandez-Andres, Lee and Romero (1999)* method.

References
----------
-   :cite:`Hernandez-Andres1999a` : Hernández-Andrés, J., Lee, R. L., &
    Romero, J. (1999). Calculating correlated color temperatures across the
    entire gamut of daylight and skylight chromaticities. Applied Optics,
    38(27), 5703. doi:10.1364/AO.38.005703
"""

import numpy as np
from scipy.optimize import minimize

from colour.colorimetry import CCS_ILLUMINANTS
from colour.utilities import as_float_array, as_float, tsplit, usage_warning

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'xy_to_CCT_Hernandez1999',
    'CCT_to_xy_Hernandez1999',
]


def xy_to_CCT_Hernandez1999(xy):
    """
    Returns the correlated colour temperature :math:`T_{cp}` from given
    *CIE xy* chromaticity coordinates using *Hernandez-Andres et al. (1999)*
    method.

    Parameters
    ----------
    xy : array_like
        *CIE xy* chromaticity coordinates.

    Returns
    -------
    numeric
        Correlated colour temperature :math:`T_{cp}`.

    References
    ----------
    :cite:`Hernandez-Andres1999a`

    Examples
    --------
    >>> xy = np.array([0.31270, 0.32900])
    >>> xy_to_CCT_Hernandez1999(xy)  # doctest: +ELLIPSIS
    6500.7420431...
    """

    x, y = tsplit(xy)

    n = (x - 0.3366) / (y - 0.1735)
    CCT = (-949.86315 + 6253.80338 * np.exp(-n / 0.92159) +
           28.70599 * np.exp(-n / 0.20039) + 0.00004 * np.exp(-n / 0.07125))

    n = np.where(CCT > 50000, (x - 0.3356) / (y - 0.1691), n)

    CCT = np.where(
        CCT > 50000,
        36284.48953 + 0.00228 * np.exp(-n / 0.07861) +
        5.4535e-36 * np.exp(-n / 0.01543),
        CCT,
    )

    return as_float(CCT)


def CCT_to_xy_Hernandez1999(CCT, optimisation_kwargs=None):
    """
    Returns the *CIE xy* chromaticity coordinates from given correlated colour
    temperature :math:`T_{cp}` using *Hernandez-Andres et al. (1999)* method.

    Parameters
    ----------
    CCT : numeric or array_like
        Correlated colour temperature :math:`T_{cp}`.
    optimisation_kwargs : dict_like, optional
        Parameters for :func:`scipy.optimize.minimize` definition.

    Returns
    -------
    ndarray
        *CIE xy* chromaticity coordinates.

    Warnings
    --------
    *Hernandez-Andres et al. (1999)* method for computing *CIE xy* chromaticity
    coordinates from given correlated colour temperature is not a bijective
    function and might produce unexpected results. It is given for consistency
    with other correlated colour temperature computation methods but should be
    avoided for practical applications. The current implementation relies on
    optimization using :func:`scipy.optimize.minimize` definition and thus has
    reduced precision and poor performance.

    References
    ----------
    :cite:`Hernandez-Andres1999a`

    Examples
    --------
    >>> CCT_to_xy_Hernandez1999(6500.7420431786531)  # doctest: +ELLIPSIS
    array([ 0.3127...,  0.329...])
    """

    usage_warning('"Hernandez-Andres et al. (1999)" method for computing '
                  '"CIE xy" chromaticity coordinates from given correlated '
                  'colour temperature is not a bijective function and and'
                  'might produce unexpected results. It is given for '
                  'consistency with other correlated colour temperature '
                  'computation methods but should be avoided for practical '
                  'applications.')

    CCT = as_float_array(CCT)
    shape = list(CCT.shape)
    CCT = np.atleast_1d(CCT.reshape([-1, 1]))

    def objective_function(xy, CCT):
        """
        Objective function.
        """

        objective = np.linalg.norm(xy_to_CCT_Hernandez1999(xy) - CCT)

        return objective

    optimisation_settings = {
        'method': 'Nelder-Mead',
        'options': {
            'fatol': 1e-10,
        },
    }
    if optimisation_kwargs is not None:
        optimisation_settings.update(optimisation_kwargs)

    CCT = as_float_array([
        minimize(
            objective_function,
            x0=CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65'],
            args=(CCT_i, ),
            **optimisation_settings).x for CCT_i in CCT
    ])

    return as_float(CCT.reshape(shape + [2]))
