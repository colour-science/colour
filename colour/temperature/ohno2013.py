# -*- coding: utf-8 -*-
"""
Ohno (2013) Correlated Colour Temperature
=========================================

Defines *Ohno (2013)* correlated colour temperature :math:`T_{cp}` computations
objects:

-   :func:`colour.temperature.uv_to_CCT_Ohno2013`: Correlated colour
    temperature :math:`T_{cp}` and :math:`\\Delta_{uv}` computation of given
    *CIE UCS* colourspace *uv* chromaticity coordinates using *Ohno (2013)*
    method.
-   :func:`colour.temperature.CCT_to_uv_Ohno2013`: *CIE UCS* colourspace *uv*
    chromaticity coordinates computation of given correlated colour temperature
    :math:`T_{cp}`, :math:`\\Delta_{uv}` using *Ohno (2013)* method.

See Also
--------
`Colour Temperature & Correlated Colour Temperature Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/temperature/cct.ipynb>`_

References
----------
-   :cite:`Ohno2014a` : Ohno, Y. (2014). Practical Use and Calculation of CCT
    and Duv. LEUKOS, 10(1), 47-55. doi:10.1080/15502724.2014.839020
"""

from __future__ import division, unicode_literals

import numpy as np
from collections import namedtuple

from colour.colorimetry import (
    DEFAULT_SPECTRAL_SHAPE, STANDARD_OBSERVERS_CMFS, sd_blackbody, sd_to_XYZ)
from colour.models import UCS_to_uv, XYZ_to_UCS
from colour.utilities import as_float_array, runtime_warning, tsplit

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'PLANCKIAN_TABLE_TUVD', 'CCT_MINIMAL', 'CCT_MAXIMAL', 'CCT_SAMPLES',
    'CCT_CALCULATION_ITERATIONS', 'planckian_table',
    'planckian_table_minimal_distance_index', 'uv_to_CCT_Ohno2013',
    'CCT_to_uv_Ohno2013'
]

PLANCKIAN_TABLE_TUVD = namedtuple('PlanckianTable_Tuvdi',
                                  ('Ti', 'ui', 'vi', 'di'))

CCT_MINIMAL = 1000
CCT_MAXIMAL = 100000
CCT_SAMPLES = 10
CCT_CALCULATION_ITERATIONS = 6


def planckian_table(uv, cmfs, start, end, count):
    """
    Returns a planckian table from given *CIE UCS* colourspace *uv*
    chromaticity coordinates, colour matching functions and temperature range
    using *Ohno (2013)* method.

    Parameters
    ----------
    uv : array_like
        *uv* chromaticity coordinates.
    cmfs : XYZ_ColourMatchingFunctions
        Standard observer colour matching functions.
    start : numeric
        Temperature range start in kelvins.
    end : numeric
        Temperature range end in kelvins.
    count : int
        Temperatures count in the planckian table.

    Returns
    -------
    list
        Planckian table.

    Examples
    --------
    >>> from colour import DEFAULT_SPECTRAL_SHAPE, STANDARD_OBSERVERS_CMFS
    >>> from pprint import pprint
    >>> cmfs = (
    ...     STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer'].
    ...     copy().align(DEFAULT_SPECTRAL_SHAPE)
    ... )
    >>> uv = np.array([0.1978, 0.3122])
    >>> pprint(planckian_table(uv, cmfs, 1000, 1010, 10))
    ... # doctest: +ELLIPSIS
    [PlanckianTable_Tuvdi(Ti=1000.0, \
ui=0.4479628..., vi=0.3546296..., di=0.2537355...),
     PlanckianTable_Tuvdi(Ti=1001.1111111..., \
ui=0.4477030..., vi=0.3546521..., di=0.2534831...),
     PlanckianTable_Tuvdi(Ti=1002.2222222..., \
ui=0.4474434..., vi=0.3546746..., di=0.2532310...),
     PlanckianTable_Tuvdi(Ti=1003.3333333..., \
ui=0.4471842..., vi=0.3546970..., di=0.2529792...),
     PlanckianTable_Tuvdi(Ti=1004.4444444..., \
ui=0.4469252..., vi=0.3547194..., di=0.2527277...),
     PlanckianTable_Tuvdi(Ti=1005.5555555..., \
ui=0.4466666..., vi=0.3547417..., di=0.2524765...),
     PlanckianTable_Tuvdi(Ti=1006.6666666..., \
ui=0.4464083..., vi=0.3547640..., di=0.2522256...),
     PlanckianTable_Tuvdi(Ti=1007.7777777..., \
ui=0.4461502..., vi=0.3547862..., di=0.2519751...),
     PlanckianTable_Tuvdi(Ti=1008.8888888..., \
ui=0.4458925..., vi=0.3548084..., di=0.2517248...),
     PlanckianTable_Tuvdi(Ti=1010.0, \
ui=0.4456351..., vi=0.3548306..., di=0.2514749...)]
    """

    ux, vx = uv

    cmfs = cmfs.copy().trim(DEFAULT_SPECTRAL_SHAPE)

    shape = cmfs.shape

    table = []
    for Ti in np.linspace(start, end, count):
        sd = sd_blackbody(Ti, shape)
        XYZ = sd_to_XYZ(sd, cmfs)
        XYZ /= np.max(XYZ)
        UVW = XYZ_to_UCS(XYZ)
        ui, vi = UCS_to_uv(UVW)
        di = np.hypot(ux - ui, vx - vi)
        table.append(PLANCKIAN_TABLE_TUVD(Ti, ui, vi, di))

    return table


def planckian_table_minimal_distance_index(planckian_table_):
    """
    Returns the shortest distance index in given planckian table using
    *Ohno (2013)* method.

    Parameters
    ----------
    planckian_table_ : list
        Planckian table.

    Returns
    -------
    int
        Shortest distance index.

    Examples
    --------
    >>> from colour import DEFAULT_SPECTRAL_SHAPE, STANDARD_OBSERVERS_CMFS
    >>> cmfs = (
    ...     STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer'].
    ...     copy().align(DEFAULT_SPECTRAL_SHAPE)
    ... )
    >>> uv = np.array([0.1978, 0.3122])
    >>> table = planckian_table(uv, cmfs, 1000, 1010, 10)
    >>> planckian_table_minimal_distance_index(table)
    9
    """

    distances = [x.di for x in planckian_table_]
    return distances.index(min(distances))


def _uv_to_CCT_Ohno2013(
        uv,
        cmfs=STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer']
        .copy().trim(DEFAULT_SPECTRAL_SHAPE),
        start=CCT_MINIMAL,
        end=CCT_MAXIMAL,
        count=CCT_SAMPLES,
        iterations=CCT_CALCULATION_ITERATIONS):
    """
    Returns the correlated colour temperature :math:`T_{cp}` and
    :math:`\\Delta_{uv}` from given *CIE UCS* colourspace *uv* chromaticity
    coordinates, colour matching functions and temperature range using
    *Ohno (2013)* method.

    The iterations parameter defines the calculations precision: The higher its
    value, the more planckian tables will be generated through cascade
    expansion in order to converge to the exact solution.

    Parameters
    ----------
    uv : array_like
        *CIE UCS* colourspace *uv* chromaticity coordinates.
    cmfs : XYZ_ColourMatchingFunctions, optional
        Standard observer colour matching functions.
    start : numeric, optional
        Temperature range start in kelvins.
    end : numeric, optional
        Temperature range end in kelvins.
    count : int, optional
        Temperatures count in the planckian tables.
    iterations : int, optional
        Number of planckian tables to generate.

    Returns
    -------
    ndarray
        Correlated colour temperature :math:`T_{cp}`, :math:`\\Delta_{uv}`.
    """

    # Ensuring we do at least one iteration to initialise variables.
    iterations = max(iterations, 1)

    # Planckian table creation through cascade expansion.
    for _i in range(iterations):
        table = planckian_table(uv, cmfs, start, end, count)
        index = planckian_table_minimal_distance_index(table)
        if index == 0:
            runtime_warning(
                ('Minimal distance index is on lowest planckian table bound, '
                 'unpredictable results may occur!'))
            index += 1
        elif index == len(table) - 1:
            runtime_warning(
                ('Minimal distance index is on highest planckian table bound, '
                 'unpredictable results may occur!'))
            index -= 1

        start = table[index - 1].Ti
        end = table[index + 1].Ti

    _ux, vx = uv

    Tuvdip, Tuvdi, Tuvdin = (table[index - 1], table[index], table[index + 1])
    Tip, uip, vip, dip = Tuvdip.Ti, Tuvdip.ui, Tuvdip.vi, Tuvdip.di
    Ti, di = Tuvdi.Ti, Tuvdi.di
    Tin, uin, vin, din = Tuvdin.Ti, Tuvdin.ui, Tuvdin.vi, Tuvdin.di

    # Triangular solution.
    l = np.hypot(uin - uip, vin - vip)  # noqa
    x = (dip ** 2 - din ** 2 + l ** 2) / (2 * l)
    T = Tip + (Tin - Tip) * (x / l)

    vtx = vip + (vin - vip) * (x / l)
    sign = 1 if vx - vtx >= 0 else -1
    D_uv = (dip ** 2 - x ** 2) ** (1 / 2) * sign

    # Parabolic solution.
    if np.abs(D_uv) >= 0.002:
        X = (Tin - Ti) * (Tip - Tin) * (Ti - Tip)
        a = (Tip * (din - di) + Ti * (dip - din) + Tin * (di - dip)) * X ** -1
        b = (-(Tip ** 2 * (din - di) + Ti ** 2 * (dip - din) + Tin ** 2 *
               (di - dip)) * X ** -1)
        c = (
            -(dip * (Tin - Ti) * Ti * Tin + di *
              (Tip - Tin) * Tip * Tin + din * (Ti - Tip) * Tip * Ti) * X ** -1)

        T = -b / (2 * a)

        D_uv = sign * (a * T ** 2 + b * T + c)

    return np.array([T, D_uv])


def uv_to_CCT_Ohno2013(
        uv,
        cmfs=STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer'],
        start=CCT_MINIMAL,
        end=CCT_MAXIMAL,
        count=CCT_SAMPLES,
        iterations=CCT_CALCULATION_ITERATIONS):
    """
    Returns the correlated colour temperature :math:`T_{cp}` and
    :math:`\\Delta_{uv}` from given *CIE UCS* colourspace *uv* chromaticity
    coordinates, colour matching functions and temperature range using
    *Ohno (2013)* method.

    The iterations parameter defines the calculations precision: The higher its
    value, the more planckian tables will be generated through cascade
    expansion in order to converge to the exact solution.

    Parameters
    ----------
    uv : array_like
        *CIE UCS* colourspace *uv* chromaticity coordinates.
    cmfs : XYZ_ColourMatchingFunctions, optional
        Standard observer colour matching functions.
    start : numeric, optional
        Temperature range start in kelvins.
    end : numeric, optional
        Temperature range end in kelvins.
    count : int, optional
        Temperatures count in the planckian tables.
    iterations : int, optional
        Number of planckian tables to generate.

    Returns
    -------
    ndarray
        Correlated colour temperature :math:`T_{cp}`, :math:`\\Delta_{uv}`.

    References
    ----------
    :cite:`Ohno2014a`

    Examples
    --------
    >>> from colour import DEFAULT_SPECTRAL_SHAPE, STANDARD_OBSERVERS_CMFS
    >>> cmfs = (
    ...     STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer'].
    ...     copy().align(DEFAULT_SPECTRAL_SHAPE)
    ... )
    >>> uv = np.array([0.1978, 0.3122])
    >>> uv_to_CCT_Ohno2013(uv, cmfs)  # doctest: +ELLIPSIS
    array([  6.5074738...e+03,   3.2233461...e-03])
    """

    uv = as_float_array(uv)

    CCT_D_uv = [
        _uv_to_CCT_Ohno2013(a, cmfs, start, end, count, iterations)
        for a in np.reshape(uv, (-1, 2))
    ]

    return as_float_array(CCT_D_uv).reshape(uv.shape)


def _CCT_to_uv_Ohno2013(
        CCT_D_uv,
        cmfs=STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer']):
    """
    Returns the *CIE UCS* colourspace *uv* chromaticity coordinates from given
    correlated colour temperature :math:`T_{cp}`, :math:`\\Delta_{uv}` and
    colour matching functions using *Ohno (2013)* method.

    Parameters
    ----------
    CCT_D_uv : ndarray
        Correlated colour temperature :math:`T_{cp}`, :math:`\\Delta_{uv}`.
    cmfs : XYZ_ColourMatchingFunctions, optional
        Standard observer colour matching functions.

    Returns
    -------
    ndarray
        *CIE UCS* colourspace *uv* chromaticity coordinates.
    """

    CCT, D_uv = tsplit(CCT_D_uv)

    cmfs = cmfs.copy().trim(DEFAULT_SPECTRAL_SHAPE)

    shape = cmfs.shape

    delta = 0.01

    sd = sd_blackbody(CCT, shape)
    XYZ = sd_to_XYZ(sd, cmfs)
    XYZ *= 1 / np.max(XYZ)
    UVW = XYZ_to_UCS(XYZ)
    u0, v0 = UCS_to_uv(UVW)

    if D_uv == 0:
        return np.array([u0, v0])
    else:
        sd = sd_blackbody(CCT + delta, shape)
        XYZ = sd_to_XYZ(sd, cmfs)
        XYZ *= 1 / np.max(XYZ)
        UVW = XYZ_to_UCS(XYZ)
        u1, v1 = UCS_to_uv(UVW)

        du = u0 - u1
        dv = v0 - v1

        u = u0 - D_uv * (dv / np.hypot(du, dv))
        v = v0 + D_uv * (du / np.hypot(du, dv))

        return np.array([u, v])


def CCT_to_uv_Ohno2013(
        CCT_D_uv,
        cmfs=STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer']):
    """
    Returns the *CIE UCS* colourspace *uv* chromaticity coordinates from given
    correlated colour temperature :math:`T_{cp}`, :math:`\\Delta_{uv}` and
    colour matching functions using *Ohno (2013)* method.

    Parameters
    ----------
    CCT_D_uv : ndarray
        Correlated colour temperature :math:`T_{cp}`, :math:`\\Delta_{uv}`.
    cmfs : XYZ_ColourMatchingFunctions, optional
        Standard observer colour matching functions.

    Returns
    -------
    ndarray
        *CIE UCS* colourspace *uv* chromaticity coordinates.

    References
    ----------
    :cite:`Ohno2014a`

    Examples
    --------
    >>> from colour import DEFAULT_SPECTRAL_SHAPE, STANDARD_OBSERVERS_CMFS
    >>> cmfs = (
    ...     STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer'].
    ...     copy().align(DEFAULT_SPECTRAL_SHAPE)
    ... )
    >>> CCT_D_uv = np.array([6507.4342201047066, 0.003223690901513])
    >>> CCT_to_uv_Ohno2013(CCT_D_uv, cmfs)  # doctest: +ELLIPSIS
    array([ 0.1977999...,  0.3122004...])
    """

    CCT_D_uv = as_float_array(CCT_D_uv)

    uv = [_CCT_to_uv_Ohno2013(a, cmfs) for a in np.reshape(CCT_D_uv, (-1, 2))]

    return as_float_array(uv).reshape(CCT_D_uv.shape)
