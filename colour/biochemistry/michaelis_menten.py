# -*- coding: utf-8 -*-
"""
Michaelis-Menten Kinetics
=========================

Implements support for *Michaelis-Menten* kinetics, a model of enzyme kinetics:

-   :func:`colour.biochemistry.reaction_rate_MichealisMenten`
-   :func:`colour.biochemistry.substrate_concentration_MichealisMenten`

References
----------
-   :cite:`Wikipedia2003d` : Wikipedia. (2003). Michaelis-Menten kinetics.
    Retrieved April 29, 2017, from
    https://en.wikipedia.org/wiki/Michaelis%E2%80%93Menten_kinetics
"""

from __future__ import division, unicode_literals

from colour.utilities import as_float_array, as_float
import colour.ndarray as np

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'reaction_rate_MichealisMenten', 'substrate_concentration_MichealisMenten'
]


def reaction_rate_MichealisMenten(S, V_max, K_m):
    """
    Describes the rate of enzymatic reactions, by relating reaction rate
    :math:`v` to concentration of a substrate :math:`S`.

    Parameters
    ----------
    S : array_like
        Concentration of a substrate :math:`S`.
    V_max : array_like
        Maximum rate :math:`V_{max}` achieved by the system, at saturating
        substrate concentration.
    K_m : array_like
        Substrate concentration :math:`V_{max}` at which the reaction rate is
        half of :math:`V_{max}`.

    Returns
    -------
    array_like
        Reaction rate :math:`v`.

    References
    ----------
    :cite:`Wikipedia2003d`

    Notes
    -----
    On CuPy, this returns an array even for length of 1.

    Examples
    --------
    >>> reaction_rate_MichealisMenten(0.5, 2.5, 0.8)  # doctest: +ELLIPSIS
    0.9615384...
    """

    S = as_float_array(S)
    V_max = as_float_array(V_max)
    K_m = as_float_array(K_m)

    v = (V_max * S) / (K_m + S)

    if np.__name__ == 'cupy' and v.size == 1:
        return as_float(v)

    return v


def substrate_concentration_MichealisMenten(v, V_max, K_m):
    """
    Describes the rate of enzymatic reactions, by relating concentration of a
    substrate :math:`S` to reaction rate :math:`v`.

    Parameters
    ----------
    v : array_like
        Reaction rate :math:`v`.
    V_max : array_like
        Maximum rate :math:`V_{max}` achieved by the system, at saturating
        substrate concentration.
    K_m : array_like
        Substrate concentration :math:`V_{max}` at which the reaction rate is
        half of :math:`V_{max}`.

    Returns
    -------
    array_like
        Concentration of a substrate :math:`S`.

    References
    ----------
    :cite:`Wikipedia2003d`

    Examples
    --------
    >>> substrate_concentration_MichealisMenten(0.961538461538461, 2.5, 0.8)
    ... # doctest: +ELLIPSIS
    0.4999999...
    """

    v = as_float_array(v)
    V_max = as_float_array(V_max)
    K_m = as_float_array(K_m)

    S = (v * K_m) / (V_max - v)

    if np.__name__ == 'cupy' and S.size == 1:
        return as_float(S)

    return S
