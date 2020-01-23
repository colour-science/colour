# -*- coding: utf-8 -*-
"""
Krystek (1985) Correlated Colour Temperature
============================================

Defines *Krystek (1985)* correlated colour temperature :math:`T_{cp}`
computations objects:

-   :func:`colour.temperature.uv_to_CCT_Krystek1985`: Correlated colour
    temperature :math:`T_{cp}` computation of given *CIE UCS* colourspace *uv*
    chromaticity coordinates using *Krystek (1985)* method.
-   :func:`colour.temperature.CCT_to_uv_Krystek1985`: *CIE UCS* colourspace
    *uv* chromaticity coordinates computation of given correlated colour
    temperature :math:`T_{cp}` using *Krystek (1985)* method.

See Also
--------
`Colour Temperature & Correlated Colour Temperature Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/temperature/cct.ipynb>`_

References
----------
-   :cite:`Krystek1985b` : Krystek, M. (1985). An algorithm to calculate
    correlated colour temperature. Color Research & Application, 10(1),
    38-40. doi:10.1002/col.5080100109
"""

from __future__ import division, unicode_literals

import numpy as np
from scipy.optimize import minimize

from colour.utilities import as_float_array, as_numeric, tstack

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['uv_to_CCT_Krystek1985', 'CCT_to_uv_Krystek1985']


def uv_to_CCT_Krystek1985(uv, optimisation_parameters=None):
    """
    Returns the correlated colour temperature :math:`T_{cp}` from given
    *CIE UCS* colourspace *uv* chromaticity coordinates using *Krystek (1985)*
    method.

    Parameters
    ----------
    uv : array_like
         *CIE UCS* colourspace *uv* chromaticity coordinates.
    optimisation_parameters : dict_like, optional
        Parameters for :func:`scipy.optimize.minimize` definition.

    Returns
    -------
    ndarray
        Correlated colour temperature :math:`T_{cp}`.

    Warnings
    --------
    *Krystek (1985)* does not give an analytical inverse transformation to
    compute the correlated colour temperature :math:`T_{cp}` from given
    *CIE UCS* colourspace *uv* chromaticity coordinates, the current
    implementation relies on optimization using :func:`scipy.optimize.minimize`
    definition and thus has reduced precision and poor performance.

    Notes
    -----
    -   *Krystek (1985)* method computations are valid for correlated colour
        temperature :math:`T_{cp}` normalised to domain [1000, 15000].

    References
    ----------
    :cite:`Krystek1985b`

    Examples
    --------
    >>> uv_to_CCT_Krystek1985(np.array([0.20047203, 0.31029290]))
    ... # doctest: +ELLIPSIS
    6504.3894290...
    """

    uv = as_float_array(uv)
    shape = uv.shape
    uv = np.atleast_1d(uv.reshape([-1, 2]))

    def objective_function(CCT, uv):
        """
        Objective function.
        """

        objective = np.linalg.norm(CCT_to_uv_Krystek1985(CCT) - uv)

        return objective

    optimisation_settings = {
        'method': 'Nelder-Mead',
        'options': {
            'fatol': 1e-10,
        },
    }
    if optimisation_parameters is not None:
        optimisation_settings.update(optimisation_parameters)

    CCT = as_float_array([
        minimize(
            objective_function,
            x0=6500,
            args=(uv_i, ),
            **optimisation_settings).x for uv_i in uv
    ])

    return as_numeric(CCT.reshape(shape[:-1]))


def CCT_to_uv_Krystek1985(CCT):
    """
    Returns the *CIE UCS* colourspace *uv* chromaticity coordinates from given
    correlated colour temperature :math:`T_{cp}` using *Krystek (1985)* method.

    Parameters
    ----------
    CCT : array_like
        Correlated colour temperature :math:`T_{cp}`.

    Returns
    -------
    ndarray
        *CIE UCS* colourspace *uv* chromaticity coordinates.

    Notes
    -----
    -   *Krystek (1985)* method computations are valid for correlated colour
        temperature :math:`T_{cp}` normalised to domain [1000, 15000].

    References
    ----------
    :cite:`Krystek1985b`

    Examples
    --------
    >>> CCT_to_uv_Krystek1985(6504.38938305)  # doctest: +ELLIPSIS
    array([ 0.2004720...,  0.3102929...])
    """

    T = as_float_array(CCT)

    u = ((0.860117757 + 1.54118254 * 10 ** -4 * T +
          1.28641212 * 10 ** -7 * T ** 2) /
         (1 + 8.42420235 * 10 ** -4 * T + 7.08145163 * 10 ** -7 * T ** 2))
    v = ((0.317398726 + 4.22806245 * 10 ** -5 * T +
          4.20481691 * 10 ** -8 * T ** 2) /
         (1 - 2.89741816 * 10 ** -5 * T + 1.61456053 * 10 ** -7 * T ** 2))

    return tstack([u, v])
