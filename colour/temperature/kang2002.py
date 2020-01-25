# -*- coding: utf-8 -*-
"""
Kang, Moon, Hong, Lee, Cho and Kim (2002) Correlated Colour Temperature
=======================================================================

Defines *Kang et al. (2002)* correlated colour temperature :math:`T_{cp}`
computations objects:

-   :func:`colour.temperature.xy_to_CCT_Kang2002`: Correlated colour
    temperature :math:`T_{cp}` of given *CIE xy* chromaticity coordinates
    computation  using *Kang, Moon, Hong, Lee, Cho and Kim (2002)* method.
-   :func:`colour.temperature.CCT_to_xy_Kang2002`: *CIE xy* chromaticity
    coordinates computation of given correlated colour temperature
    :math:`T_{cp}` using *Kang, Moon, Hong, Lee, Cho and Kim (2002)* method.

See Also
--------
`Colour Temperature & Correlated Colour Temperature Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/temperature/cct.ipynb>`_

References
----------
-   :cite:`Kang2002a` : Kang, B., Moon, O., Hong, C., Lee, H., Cho, B., &
    Kim, Y. (2002). Design of advanced color: Temperature control system for
    HDTV applications. Journal of the Korean Physical Society, 41(6), 865-871.
    Retrieved from http://cat.inist.fr/?aModele=afficheN&cpsidt=14448733
"""

from __future__ import division, unicode_literals

import numpy as np
from scipy.optimize import minimize

from colour.utilities import as_float_array, as_numeric, tstack, usage_warning

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['xy_to_CCT_Kang2002', 'CCT_to_xy_Kang2002']


def xy_to_CCT_Kang2002(xy, optimisation_parameters=None):
    """
    Returns the correlated colour temperature :math:`T_{cp}` from given
    *CIE xy* chromaticity coordinates using *Kang et al. (2002)* method.

    Parameters
    ----------
    xy : array_like
        *CIE xy* chromaticity coordinates.
    optimisation_parameters : dict_like, optional
        Parameters for :func:`scipy.optimize.minimize` definition.

    Returns
    -------
    ndarray
        Correlated colour temperature :math:`T_{cp}`.

    Warnings
    --------
    *Kang et al. (2002)* does not give an analytical inverse transformation to
    compute the correlated colour temperature :math:`T_{cp}` from given
    *CIE xy* chromaticity coordinates, the current implementation relies on
    optimization using :func:`scipy.optimize.minimize` definition and thus has
    reduced precision and poor performance.

    References
    ----------
    :cite:`Kang2002a`

    Examples
    --------
    >>> xy_to_CCT_Kang2002(np.array([0.31342600, 0.32359597]))
    ... # doctest: +ELLIPSIS
    6504.3893128...
    """

    xy = as_float_array(xy)
    shape = xy.shape
    xy = np.atleast_1d(xy.reshape([-1, 2]))

    def objective_function(CCT, xy):
        """
        Objective function.
        """

        objective = np.linalg.norm(CCT_to_xy_Kang2002(CCT) - xy)

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
            args=(xy_i, ),
            **optimisation_settings).x for xy_i in xy
    ])

    return as_numeric(CCT.reshape(shape[:-1]))


def CCT_to_xy_Kang2002(CCT):
    """
    Returns the *CIE xy* chromaticity coordinates from given correlated colour
    temperature :math:`T_{cp}` using *Kang et al. (2002)* method.

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
    :cite:`Kang2002a`

    Examples
    --------
    >>> CCT_to_xy_Kang2002(6504.38938305)  # doctest: +ELLIPSIS
    array([ 0.313426 ...,  0.3235959...])
    """

    CCT = as_float_array(CCT)

    if np.any(CCT[np.asarray(np.logical_or(CCT < 1667, CCT > 25000))]):
        usage_warning(('Correlated colour temperature must be in domain '
                       '[1667, 25000], unpredictable results may occur!'))

    x = np.where(
        CCT <= 4000,
        -0.2661239 * 10 ** 9 / CCT ** 3 - 0.2343589 * 10 ** 6 / CCT ** 2 +
        0.8776956 * 10 ** 3 / CCT + 0.179910,
        -3.0258469 * 10 ** 9 / CCT ** 3 + 2.1070379 * 10 ** 6 / CCT ** 2 +
        0.2226347 * 10 ** 3 / CCT + 0.24039,
    )

    cnd_l = [CCT <= 2222, np.logical_and(CCT > 2222, CCT <= 4000), CCT > 4000]
    i = -1.1063814 * x ** 3 - 1.34811020 * x ** 2 + 2.18555832 * x - 0.20219683
    j = -0.9549476 * x ** 3 - 1.37418593 * x ** 2 + 2.09137015 * x - 0.16748867
    k = 3.0817580 * x ** 3 - 5.8733867 * x ** 2 + 3.75112997 * x - 0.37001483
    y = np.select(cnd_l, [i, j, k])

    xy = tstack([x, y])

    return xy
