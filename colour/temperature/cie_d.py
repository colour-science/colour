# -*- coding: utf-8 -*-
"""
CIE Illuminant D Series Correlated Colour Temperature
=====================================================

Defines the *CIE Illuminant D Series* correlated colour temperature
:math:`T_{cp} computations objects:

-   :func:`colour.temperature.xy_to_CCT_CIE_D`: Correlated colour temperature
    :math:`T_{cp}` computation of a *CIE Illuminant D Series* from its *CIE xy*
    chromaticity coordinates.
-   :func:`colour.temperature.CCT_to_xy_CIE_D`: *CIE xy* chromaticity
    coordinates computation of a *CIE Illuminant D Series* from its correlated
    colour temperature :math:`T_{cp}`.

References
----------
-   :cite:`Wyszecki2000z` : Wyszecki, Günther, & Stiles, W. S. (2000). CIE
    Method of Calculating D-Illuminants. In Color Science: Concepts and
    Methods, Quantitative Data and Formulae (pp. 145-146). Wiley.
    ISBN:978-0-471-39918-6
"""

import numpy as np
from scipy.optimize import minimize

from colour.colorimetry import daylight_locus_function
from colour.utilities import as_float_array, as_float, tstack, usage_warning

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'xy_to_CCT_CIE_D',
    'CCT_to_xy_CIE_D',
]


def xy_to_CCT_CIE_D(xy, optimisation_kwargs=None):
    """
    Returns the correlated colour temperature :math:`T_{cp}` of a
    *CIE Illuminant D Series* from its *CIE xy* chromaticity coordinates.

    Parameters
    ----------
    xy : array_like
        *CIE xy* chromaticity coordinates.
    optimisation_kwargs : dict_like, optional
        Parameters for :func:`scipy.optimize.minimize` definition.

    Returns
    -------
    ndarray
        Correlated colour temperature :math:`T_{cp}`.

    Warnings
    --------
    The *CIE Illuminant D Series* method does not give an analytical inverse
    transformation to compute the correlated colour temperature :math:`T_{cp}`
    from given *CIE xy* chromaticity coordinates, the current implementation
    relies on optimization using :func:`scipy.optimize.minimize` definition and
    thus has reduced precision and poor performance.

    References
    ----------
    :cite:`Wyszecki2000z`

    Examples
    --------
    >>> xy_to_CCT_CIE_D(np.array([0.31270775, 0.32911283]))
    ... # doctest: +ELLIPSIS
    6504.3895840...
    """

    xy = as_float_array(xy)
    shape = xy.shape
    xy = np.atleast_1d(xy.reshape([-1, 2]))

    def objective_function(CCT, xy):
        """
        Objective function.
        """

        objective = np.linalg.norm(CCT_to_xy_CIE_D(CCT) - xy)

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
            x0=6500,
            args=(xy_i, ),
            **optimisation_settings).x for xy_i in xy
    ])

    return as_float(CCT.reshape(shape[:-1]))


def CCT_to_xy_CIE_D(CCT):
    """
    Returns the *CIE xy* chromaticity coordinates of a
    *CIE Illuminant D Series* from its correlated colour temperature
    :math:`T_{cp}`.

    Parameters
    ----------
    CCT : numeric or array_like
        Correlated colour temperature :math:`T_{cp}`.

    Returns
    -------
    ndarray
        *CIE xy* chromaticity coordinates.

    Raises
    ------
    ValueError
        If the correlated colour temperature is not in appropriate domain.

    References
    ----------
    :cite:`Wyszecki2000z`

    Examples
    --------
    >>> CCT_to_xy_CIE_D(6504.38938305)  # doctest: +ELLIPSIS
    array([ 0.3127077...,  0.3291128...])
    """

    CCT = as_float_array(CCT)

    if np.any(CCT[np.asarray(np.logical_or(CCT < 4000, CCT > 25000))]):
        usage_warning(('Correlated colour temperature must be in domain '
                       '[4000, 25000], unpredictable results may occur!'))

    CCT_3 = CCT ** 3
    CCT_2 = CCT ** 2

    x = np.where(
        CCT <= 7000,
        -4.607 * 10 ** 9 / CCT_3 + 2.9678 * 10 ** 6 / CCT_2 +
        0.09911 * 10 ** 3 / CCT + 0.244063,
        -2.0064 * 10 ** 9 / CCT_3 + 1.9018 * 10 ** 6 / CCT_2 +
        0.24748 * 10 ** 3 / CCT + 0.23704,
    )

    y = daylight_locus_function(x)

    xy = tstack([x, y])

    return xy
