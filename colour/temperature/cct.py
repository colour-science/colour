#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Correlated Colour Temperature :math:`T_{cp}`
============================================

Defines correlated colour temperature :math:`T_{cp}` computations objects:

-   :func:`uv_to_CCT_ohno2013`: Correlated colour temperature :math:`T_{cp}`
    and :math:`\Delta_{uv}` computation of given *CIE UCS* colourspace *uv*
    chromaticity coordinates using *Yoshi Ohno (2013)* method.
-   :func:`CCT_to_uv_ohno2013`: *CIE UCS* colourspace *uv* chromaticity
    coordinates computation of given correlated colour temperature
    :math:`T_{cp}`, :math:`\Delta_{uv}` using *Yoshi Ohno (2013)* method.
-   :func:`uv_to_CCT_robertson1968`: Correlated colour temperature
    :math:`T_{cp}` and :math:`\Delta_{uv}` computation of given *CIE UCS*
    colourspace *uv* chromaticity coordinates using *Robertson (1968)* method.
-   :func:`CCT_to_uv_robertson1968`: *CIE UCS* colourspace *uv* chromaticity
    coordinates computation of given correlated colour temperature
    :math:`T_{cp}` and :math:`\Delta_{uv}` using *Robertson (1968)* method.
-   :func:`xy_to_CCT_mccamy1992`: Correlated colour temperature :math:`T_{cp}`
    computation of given *CIE XYZ* colourspace *xy* chromaticity coordinates
    using *McCamy (1992)* method.
-   :func:`xy_to_CCT_hernandez1999`: Correlated colour temperature
    :math:`T_{cp}` computation of given *CIE XYZ* colourspace *xy* chromaticity
    coordinates using *Hernandez-Andres, Lee & Romero (1999)* method.
-   :func:`CCT_to_xy_kang2002`: *CIE XYZ* colourspace *xy* chromaticity
    coordinates computation of given correlated colour temperature
    :math:`T_{cp}` using *Kang, Moon, Hong, Lee, Cho and Kim (2002)* method.
-   :func:`CCT_to_xy_illuminant_D`: *CIE XYZ* colourspace *xy* chromaticity
    coordinates computation of *CIE Illuminant D Series* from given correlated
    colour temperature :math:`T_{cp}` of that *CIE Illuminant D Series*.

See Also
--------
`Colour Temperature & Correlated Colour Temperature IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/master/notebooks/temperature/cct.ipynb>`_  # noqa

References
----------
.. [1]  http://en.wikipedia.org/wiki/Color_temperature
"""

from __future__ import division, unicode_literals

import math
import numpy as np
from collections import namedtuple

from colour.colorimetry import (
    STANDARD_OBSERVERS_CMFS,
    blackbody_spd,
    spectral_to_XYZ)
from colour.models import UCS_to_uv, XYZ_to_UCS
from colour.utilities import CaseInsensitiveMapping, warning

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['PLANCKIAN_TABLE_TUVD',
           'CCT_MINIMAL',
           'CCT_MAXIMAL',
           'CCT_SAMPLES',
           'CCT_CALCULATION_ITERATIONS',
           'ROBERTSON_ISOTEMPERATURE_LINES_DATA',
           'ROBERTSON_ISOTEMPERATURE_LINES_RUVT',
           'ROBERTSON_ISOTEMPERATURE_LINES',
           'planckian_table',
           'planckian_table_minimal_distance_index',
           'uv_to_CCT_ohno2013',
           'CCT_to_uv_ohno2013',
           'uv_to_CCT_robertson1968',
           'CCT_to_uv_robertson1968',
           'UV_TO_CCT_METHODS',
           'uv_to_CCT',
           'CCT_TO_UV_METHODS',
           'CCT_to_uv',
           'xy_to_CCT_mccamy1992',
           'xy_to_CCT_hernandez1999',
           'CCT_to_xy_kang2002',
           'CCT_to_xy_illuminant_D',
           'XY_TO_CCT_METHODS',
           'xy_to_CCT',
           'CCT_TO_XY_METHODS',
           'CCT_to_xy']

PLANCKIAN_TABLE_TUVD = namedtuple('PlanckianTable_Tuvdi',
                                  ('Ti', 'ui', 'vi', 'di'))

CCT_MINIMAL = 1000
CCT_MAXIMAL = 100000
CCT_SAMPLES = 10
CCT_CALCULATION_ITERATIONS = 6

ROBERTSON_ISOTEMPERATURE_LINES_DATA = (
    (0, 0.18006, 0.26352, -0.24341),
    (10, 0.18066, 0.26589, -0.25479),
    (20, 0.18133, 0.26846, -0.26876),
    (30, 0.18208, 0.27119, -0.28539),
    (40, 0.18293, 0.27407, -0.30470),
    (50, 0.18388, 0.27709, -0.32675),
    (60, 0.18494, 0.28021, -0.35156),
    (70, 0.18611, 0.28342, -0.37915),
    (80, 0.18740, 0.28668, -0.40955),
    (90, 0.18880, 0.28997, -0.44278),
    (100, 0.19032, 0.29326, -0.47888),
    (125, 0.19462, 0.30141, -0.58204),
    (150, 0.19962, 0.30921, -0.70471),
    (175, 0.20525, 0.31647, -0.84901),
    (200, 0.21142, 0.32312, -1.0182),
    (225, 0.21807, 0.32909, -1.2168),
    (250, 0.22511, 0.33439, -1.4512),
    (275, 0.23247, 0.33904, -1.7298),
    (300, 0.24010, 0.34308, -2.0637),
    (325, 0.24792, 0.34655, -2.4681),  # 0.24702 ---> 0.24792 Bruce Lindbloom
    (350, 0.25591, 0.34951, -2.9641),
    (375, 0.26400, 0.35200, -3.5814),
    (400, 0.27218, 0.35407, -4.3633),
    (425, 0.28039, 0.35577, -5.3762),
    (450, 0.28863, 0.35714, -6.7262),
    (475, 0.29685, 0.35823, -8.5955),
    (500, 0.30505, 0.35907, -11.324),
    (525, 0.31320, 0.35968, -15.628),
    (550, 0.32129, 0.36011, -23.325),
    (575, 0.32931, 0.36038, -40.770),
    (600, 0.33724, 0.36051, -116.45))
"""
*Robertson* iso-temperature lines.

ROBERTSON_ISOTEMPERATURE_LINES_DATA : tuple
    (Reciprocal Megakelvin,
    CIE 1960 Chromaticity Coordinate *u*,
    CIE 1960 Chromaticity Coordinate *v*,
    Slope)

Notes
-----
-   A correction has been done by *Bruce Lindbloom* for *325* Megakelvin
    temperature: 0.24702 ---> 0.24792

References
----------
.. [2]  **Wyszecki & Stiles**,
        *Color Science - Concepts and Methods Data and Formulae -
        Second Edition*,
        Wiley Classics Library Edition, published 2000, ISBN-10: 0-471-39918-3,
        page  228.
"""

ROBERTSON_ISOTEMPERATURE_LINES_RUVT = namedtuple(
    'WyszeckiRobertson_ruvt', ('r', 'u', 'v', 't'))

ROBERTSON_ISOTEMPERATURE_LINES = [
    ROBERTSON_ISOTEMPERATURE_LINES_RUVT(*x)
    for x in ROBERTSON_ISOTEMPERATURE_LINES_DATA]


def planckian_table(uv, cmfs, start, end, count):
    """
    Returns a planckian table from given *CIE UCS* colourspace *uv*
    chromaticity coordinates, colour matching functions and temperature range
    using *Yoshi Ohno (2013)* method.

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
    >>> from colour import STANDARD_OBSERVERS_CMFS
    >>> from pprint import pprint
    >>> cmfs = 'CIE 1931 2 Degree Standard Observer'
    >>> cmfs = STANDARD_OBSERVERS_CMFS.get(cmfs)
    >>> pprint(planckian_table((0.1978, 0.3122), cmfs, 1000, 1010, 10))  # noqa  # doctest: +ELLIPSIS
    [PlanckianTable_Tuvdi(Ti=1000.0, ui=0.4480108..., vi=0.3546249..., di=0.2537821...),
     PlanckianTable_Tuvdi(Ti=1001.1111111..., ui=0.4477508..., vi=0.3546475..., di=0.2535294...),
     PlanckianTable_Tuvdi(Ti=1002.2222222..., ui=0.4474910..., vi=0.3546700..., di=0.2532771...),
     PlanckianTable_Tuvdi(Ti=1003.3333333..., ui=0.4472316..., vi=0.3546924..., di=0.2530251...),
     PlanckianTable_Tuvdi(Ti=1004.4444444..., ui=0.4469724..., vi=0.3547148..., di=0.2527734...),
     PlanckianTable_Tuvdi(Ti=1005.5555555..., ui=0.4467136..., vi=0.3547372..., di=0.2525220...),
     PlanckianTable_Tuvdi(Ti=1006.6666666..., ui=0.4464550..., vi=0.3547595..., di=0.2522710...),
     PlanckianTable_Tuvdi(Ti=1007.7777777..., ui=0.4461968..., vi=0.3547817..., di=0.2520202...),
     PlanckianTable_Tuvdi(Ti=1008.8888888..., ui=0.4459389..., vi=0.3548040..., di=0.2517697...),
     PlanckianTable_Tuvdi(Ti=1010.0, ui=0.4456812..., vi=0.3548261..., di=0.2515196...)]
    """

    ux, vx = uv

    shape = cmfs.shape

    table = []
    for Ti in np.linspace(start, end, count):
        spd = blackbody_spd(Ti, shape)
        XYZ = spectral_to_XYZ(spd, cmfs)
        XYZ *= 1 / np.max(XYZ)
        UVW = XYZ_to_UCS(XYZ)
        ui, vi = UCS_to_uv(UVW)
        di = math.sqrt((ux - ui) ** 2 + (vx - vi) ** 2)
        table.append(PLANCKIAN_TABLE_TUVD(Ti, ui, vi, di))

    return table


def planckian_table_minimal_distance_index(planckian_table):
    """
    Returns the shortest distance index in given planckian table using
    *Yoshi Ohno (2013)* method.

    Parameters
    ----------
    planckian_table : list
        Planckian table.

    Returns
    -------
    int
        Shortest distance index.

    Examples
    --------
    >>> from colour import STANDARD_OBSERVERS_CMFS
    >>> cmfs = 'CIE 1931 2 Degree Standard Observer'
    >>> cmfs = STANDARD_OBSERVERS_CMFS.get(cmfs)
    >>> table = planckian_table((0.1978, 0.3122), cmfs, 1000, 1010, 10)
    >>> planckian_table_minimal_distance_index(table)
    9
    """

    distances = [x.di for x in planckian_table]
    return distances.index(min(distances))


def uv_to_CCT_ohno2013(uv,
                       cmfs=STANDARD_OBSERVERS_CMFS.get(
                           'CIE 1931 2 Degree Standard Observer'),
                       start=CCT_MINIMAL,
                       end=CCT_MAXIMAL,
                       count=CCT_SAMPLES,
                       iterations=CCT_CALCULATION_ITERATIONS):
    """
    Returns the correlated colour temperature :math:`T_{cp}` and
    :math:`\Delta_{uv}` from given *CIE UCS* colourspace *uv* chromaticity
    coordinates, colour matching functions and temperature range using
    *Yoshi Ohno (2013)* method.

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
    tuple
        Correlated colour temperature :math:`T_{cp}`, :math:`\Delta_{uv}`.

    References
    ----------
    .. [3]  **Yoshi Ohno**, `Practical Use and Calculation of CCT and Duv
            <http://dx.doi.org/10.1080/15502724.2014.839020>`_,
            DOI: http://dx.doi.org/10.1080/15502724.2014.839020

    Examples
    --------
    >>> from colour import STANDARD_OBSERVERS_CMFS
    >>> cmfs = 'CIE 1931 2 Degree Standard Observer'
    >>> cmfs = STANDARD_OBSERVERS_CMFS.get(cmfs)
    >>> uv_to_CCT_ohno2013((0.1978, 0.3122), cmfs)  # doctest: +ELLIPSIS
    (6507.5470349..., 0.0032236...)
    """

    # Ensuring we do at least one iteration to initialise variables.
    if iterations <= 0:
        iterations = 1

    # Planckian table creation through cascade expansion.
    for i in range(iterations):
        table = planckian_table(uv, cmfs, start, end, count)
        index = planckian_table_minimal_distance_index(table)
        if index == 0:
            warning(
                ('Minimal distance index is on lowest planckian table bound, '
                 'unpredictable results may occur!'))
            index += 1
        elif index == len(table) - 1:
            warning(
                ('Minimal distance index is on highest planckian table bound, '
                 'unpredictable results may occur!'))
            index -= 1

        start = table[index - 1].Ti
        end = table[index + 1].Ti

    ux, vx = uv

    Tuvdip, Tuvdi, Tuvdin = (table[index - 1], table[index], table[index + 1])
    Tip, uip, vip, dip = Tuvdip.Ti, Tuvdip.ui, Tuvdip.vi, Tuvdip.di
    Ti, ui, vi, di = Tuvdi.Ti, Tuvdi.ui, Tuvdi.vi, Tuvdi.di
    Tin, uin, vin, din = Tuvdin.Ti, Tuvdin.ui, Tuvdin.vi, Tuvdin.di

    # Triangular solution.
    l = math.sqrt((uin - uip) ** 2 + (vin - vip) ** 2)
    x = (dip ** 2 - din ** 2 + l ** 2) / (2 * l)
    T = Tip + (Tin - Tip) * (x / l)

    vtx = vip + (vin - vip) * (x / l)
    sign = 1 if vx - vtx >= 0 else -1
    Duv = (dip ** 2 - x ** 2) ** (1 / 2) * sign

    # Parabolic solution.
    if Duv < 0.002:
        X = (Tin - Ti) * (Tip - Tin) * (Ti - Tip)
        a = (Tip * (din - di) + Ti * (dip - din) + Tin * (di - dip)) * X ** -1
        b = (-(Tip ** 2 * (din - di) + Ti ** 2 * (dip - din) + Tin ** 2 *
               (di - dip)) * X ** -1)
        c = (-(dip * (Tin - Ti) * Ti * Tin + di * (Tip - Tin) * Tip * Tin
               + din * (Ti - Tip) * Tip * Ti) * X ** -1)

        T = -b / (2 * a)

        Duv = sign * (a * T ** 2 + b * T + c)

    return T, Duv


def CCT_to_uv_ohno2013(CCT,
                       Duv=0,
                       cmfs=STANDARD_OBSERVERS_CMFS.get(
                           'CIE 1931 2 Degree Standard Observer')):
    """
    Returns the *CIE UCS* colourspace *uv* chromaticity coordinates from given
    correlated colour temperature :math:`T_{cp}`, :math:`\Delta_{uv}` and
    colour matching functions using *Yoshi Ohno (2013)* method.

    Parameters
    ----------
    CCT : numeric
        Correlated colour temperature :math:`T_{cp}`.
    Duv : numeric, optional
        :math:`\Delta_{uv}`.
    cmfs : XYZ_ColourMatchingFunctions, optional
        Standard observer colour matching functions.

    Returns
    -------
    tuple
        *CIE UCS* colourspace *uv* chromaticity coordinates.

    References
    ----------
    .. [4]  **Yoshi Ohno**, `Practical Use and Calculation of CCT and Duv
            <http://dx.doi.org/10.1080/15502724.2014.839020>`_,
            DOI: http://dx.doi.org/10.1080/15502724.2014.839020

    Examples
    --------
    >>> from colour import STANDARD_OBSERVERS_CMFS
    >>> cmfs = 'CIE 1931 2 Degree Standard Observer'
    >>> cmfs = STANDARD_OBSERVERS_CMFS.get(cmfs)
    >>> CCT = 6507.4342201047066
    >>> Duv = 0.003223690901512735
    >>> CCT_to_uv_ohno2013(CCT, Duv, cmfs)  # doctest: +ELLIPSIS
    (0.1978003..., 0.3122005...)
    """

    shape = cmfs.shape
    delta = 0.01

    spd = blackbody_spd(CCT, shape)
    XYZ = spectral_to_XYZ(spd, cmfs)
    XYZ *= 1 / np.max(XYZ)
    UVW = XYZ_to_UCS(XYZ)
    u0, v0 = UCS_to_uv(UVW)

    if Duv == 0:
        return u0, v0
    else:
        spd = blackbody_spd(CCT + delta, shape)
        XYZ = spectral_to_XYZ(spd, cmfs)
        XYZ *= 1 / np.max(XYZ)
        UVW = XYZ_to_UCS(XYZ)
        u1, v1 = UCS_to_uv(UVW)

        du = u0 - u1
        dv = v0 - v1

        u = u0 - Duv * (dv / math.sqrt(du ** 2 + dv ** 2))
        v = v0 + Duv * (du / math.sqrt(du ** 2 + dv ** 2))

        return u, v


def uv_to_CCT_robertson1968(uv):
    """
    Returns the correlated colour temperature :math:`T_{cp}` and
    :math:`\Delta_{uv}` from given *CIE UCS* colourspace *uv* chromaticity
    coordinates using *Alan R. Roberston (1968)* method.

    Parameters
    ----------
    uv : array_like
        *CIE UCS* colourspace *uv* chromaticity coordinates.

    Returns
    -------
    tuple
        Correlated colour temperature :math:`T_{cp}`, :math:`\Delta_{uv}`.

    References
    ----------
    .. [5]  **Wyszecki & Stiles**,
            *Color Science - Concepts and Methods Data and Formulae -
            Second Edition*,
            Wiley Classics Library Edition, published 2000,
            ISBN-10: 0-471-39918-3,
            page  227.
    .. [6]  *Adobe DNG SDK 1.3.0.0*:
            *dng_sdk_1_3/dng_sdk/source/dng_temperature.cpp*:
            *dng_temperature::Set_xy_coord*.

    Examples
    --------
    >>> uv = (0.19374137599822966, 0.31522104394059397)
    >>> uv_to_CCT_robertson1968(uv)  # doctest: +ELLIPSIS
    (6500.0162879..., 0.0083333...)
    """

    u, v = uv

    last_dt = last_dv = last_du = 0.0

    for i in range(1, 31):
        wr_ruvt = ROBERTSON_ISOTEMPERATURE_LINES[i]
        wr_ruvt_previous = ROBERTSON_ISOTEMPERATURE_LINES[i - 1]

        du = 1.0
        dv = wr_ruvt.t

        length = math.sqrt(1 + dv * dv)

        du /= length
        dv /= length

        uu = u - wr_ruvt.u
        vv = v - wr_ruvt.v

        dt = -uu * dv + vv * du

        if dt <= 0 or i == 30:
            if dt > 0.0:
                dt = 0.0

            dt = -dt

            if i == 1:
                f = 0.0
            else:
                f = dt / (last_dt + dt)

            T = 1.0e6 / (wr_ruvt_previous.r * f + wr_ruvt.r * (1 - f))

            uu = u - (wr_ruvt_previous.u * f + wr_ruvt.u * (1 - f))
            vv = v - (wr_ruvt_previous.v * f + wr_ruvt.v * (1 - f))

            du = du * (1 - f) + last_du * f
            dv = dv * (1 - f) + last_dv * f

            length = math.sqrt(du * du + dv * dv)

            du /= length
            dv /= length

            Duv = uu * du + vv * dv

            break

        last_dt = dt
        last_du = du
        last_dv = dv

    return T, -Duv


def CCT_to_uv_robertson1968(CCT, Duv=0):
    """
    Returns the *CIE UCS* colourspace *uv* chromaticity coordinates from given
    correlated colour temperature :math:`T_{cp}` and :math:`\Delta_{uv}` using
    *Alan R. Roberston (1968)* method.

    Parameters
    ----------
    CCT : numeric
        Correlated colour temperature :math:`T_{cp}`.
    Duv : numeric
        :math:`\Delta_{uv}`.

    Returns
    -------
    tuple
        *CIE UCS* colourspace *uv* chromaticity coordinates.

    References
    ----------
    .. [7]  **Wyszecki & Stiles**,
            *Color Science - Concepts and Methods Data and Formulae -
            Second Edition*,
            Wiley Classics Library Edition, published 2000,
            ISBN-10: 0-471-39918-3,
            page  227.
    .. [8]  *Adobe DNG SDK 1.3.0.0*:
            *dng_sdk_1_3/dng_sdk/source/dng_temperature.cpp*:
            *dng_temperature::xy_coord*.

    Examples
    --------
    >>> CCT = 6500.0081378199056
    >>> Duv = 0.0083333312442250979
    >>> CCT_to_uv_robertson1968(CCT, Duv)  # doctest: +ELLIPSIS
    (0.1937413..., 0.3152210...)
    """

    r = 1.0e6 / CCT

    for i in range(30):
        wr_ruvt = ROBERTSON_ISOTEMPERATURE_LINES[i]
        wr_ruvt_next = ROBERTSON_ISOTEMPERATURE_LINES[i + 1]

        if r < wr_ruvt_next.r or i == 29:
            f = (wr_ruvt_next.r - r) / (wr_ruvt_next.r - wr_ruvt.r)

            u = wr_ruvt.u * f + wr_ruvt_next.u * (1 - f)
            v = wr_ruvt.v * f + wr_ruvt_next.v * (1 - f)

            uu1 = uu2 = 1.0
            vv1, vv2 = wr_ruvt.t, wr_ruvt_next.t

            length1 = math.sqrt(1 + vv1 * vv1)
            length2 = math.sqrt(1 + vv2 * vv2)

            uu1 /= length1
            vv1 /= length1

            uu2 /= length2
            vv2 /= length2

            uu3 = uu1 * f + uu2 * (1 - f)
            vv3 = vv1 * f + vv2 * (1 - f)

            len3 = math.sqrt(uu3 * uu3 + vv3 * vv3)

            uu3 /= len3
            vv3 /= len3

            u += uu3 * -Duv
            v += vv3 * -Duv

            return u, v


UV_TO_CCT_METHODS = CaseInsensitiveMapping(
    {'Ohno 2013': uv_to_CCT_ohno2013,
     'Robertson 1968': uv_to_CCT_robertson1968})
"""
Supported *CIE UCS* colourspace *uv* chromaticity coordinates to correlated
colour temperature :math:`T_{cp}` computation methods.

UV_TO_CCT_METHODS : CaseInsensitiveMapping
    {'Ohno 2013', 'Robertson 1968'}

Aliases:

-   'ohno2013': 'Ohno 2013'
-   'robertson1968': 'Robertson 1968'
"""
UV_TO_CCT_METHODS['ohno2013'] = UV_TO_CCT_METHODS['Ohno 2013']
UV_TO_CCT_METHODS['robertson1968'] = UV_TO_CCT_METHODS['Robertson 1968']


def uv_to_CCT(uv, method='Ohno 2013', **kwargs):
    """
    Returns the correlated colour temperature :math:`T_{cp}` and
    :math:`\Delta_{uv}` from given *CIE UCS* colourspace *uv* chromaticity
    coordinates using given method.

    Parameters
    ----------
    uv : array_like
        *CIE UCS* colourspace *uv* chromaticity coordinates.
    method : unicode, optional
        {'Ohno 2013', 'Robertson 1968'}
        Computation method.
    \*\*kwargs : \*\*
        Keywords arguments.

    Returns
    -------
    tuple
        Correlated colour temperature :math:`T_{cp}`, :math:`\Delta_{uv}`.

    Raises
    ------
    ValueError
        If the computation method is not defined.

    Examples
    --------
    >>> from colour import STANDARD_OBSERVERS_CMFS
    >>> cmfs = 'CIE 1931 2 Degree Standard Observer'
    >>> cmfs = STANDARD_OBSERVERS_CMFS.get(cmfs)
    >>> uv_to_CCT((0.1978, 0.3122), cmfs=cmfs)  # doctest: +ELLIPSIS
    (6507.5470349..., 0.0032236...)
    """

    if method == 'Ohno 2013':
        return UV_TO_CCT_METHODS.get(method)(uv, **kwargs)
    else:
        if 'cmfs' in kwargs:
            if kwargs.get('cmfs').name != (
                    'CIE 1931 2 Degree Standard Observer'):
                raise ValueError(
                    ('"Robertson (1968)" method is only valid for '
                     '"CIE 1931 2 Degree Standard Observer"!'))

        return UV_TO_CCT_METHODS.get(method)(uv)


CCT_TO_UV_METHODS = CaseInsensitiveMapping(
    {'Ohno 2013': CCT_to_uv_ohno2013,
     'Robertson 1968': CCT_to_uv_robertson1968})
"""
Supported correlated colour temperature :math:`T_{cp}` to *CIE UCS* colourspace
*uv* chromaticity coordinates computation methods.

CCT_TO_UV_METHODS : CaseInsensitiveMapping
    {'Ohno 2013', 'Robertson 1968'}

Aliases:

-   'ohno2013': 'Ohno 2013'
-   'robertson1968': 'Robertson 1968'
"""
CCT_TO_UV_METHODS['ohno2013'] = CCT_TO_UV_METHODS['Ohno 2013']
CCT_TO_UV_METHODS['robertson1968'] = CCT_TO_UV_METHODS['Robertson 1968']


def CCT_to_uv(CCT, Duv=0, method='Ohno 2013', **kwargs):
    """
    Returns the *CIE UCS* colourspace *uv* chromaticity coordinates from given
    correlated colour temperature :math:`T_{cp}` and :math:`\Delta_{uv}` using
    given method.

    Parameters
    ----------
    CCT : numeric
        Correlated colour temperature :math:`T_{cp}`.
    Duv : numeric
        :math:`\Delta_{uv}`.
    method : unicode, optional
    {'Ohno 2013', 'Robertson 1968'}
        Computation method.
    \*\*kwargs : \*\*
        Keywords arguments.

    Returns
    -------
    tuple
        *CIE UCS* colourspace *uv* chromaticity coordinates.

    Raises
    ------
    ValueError
        If the computation method is not defined.

    Examples
    --------
    >>> from colour import STANDARD_OBSERVERS_CMFS
    >>> cmfs = 'CIE 1931 2 Degree Standard Observer'
    >>> cmfs = STANDARD_OBSERVERS_CMFS.get(cmfs)
    >>> CCT = 6507.4342201047066
    >>> Duv = 0.003223690901512735
    >>> CCT_to_uv(CCT, Duv, cmfs=cmfs)  # doctest: +ELLIPSIS
    (0.1978003..., 0.3122005...)
    """

    if method == 'Ohno 2013':
        return CCT_TO_UV_METHODS.get(method)(CCT, Duv, **kwargs)
    else:
        if 'cmfs' in kwargs:
            if kwargs.get('cmfs').name != (
                    'CIE 1931 2 Degree Standard Observer'):
                raise ValueError(
                    ('"Robertson (1968)" method is only valid for '
                     '"CIE 1931 2 Degree Standard Observer"!'))

        return CCT_TO_UV_METHODS.get(method)(CCT, Duv)


def xy_to_CCT_mccamy1992(xy):
    """
    Returns the correlated colour temperature :math:`T_{cp}` from given
    *CIE XYZ* colourspace *xy* chromaticity coordinates using
    *C. S. McCamy (1992)* method.

    Parameters
    ----------
    xy : array_like
        *xy* chromaticity coordinates.

    Returns
    -------
    numeric
        Correlated colour temperature :math:`T_{cp}`.

    References
    ----------
    .. [9]  http://en.wikipedia.org/wiki/Color_temperature#Approximation
            (Last accessed 28 June 2014)

    Examples
    --------
    >>> xy_to_CCT_mccamy1992((0.31271, 0.32902))  # doctest: +ELLIPSIS
    6504.3893830...
    """

    x, y = xy

    n = (x - 0.3320) / (y - 0.1858)
    CCT = -449 * math.pow(n, 3) + 3525 * math.pow(n, 2) - 6823.3 * n + 5520.33

    return CCT


def xy_to_CCT_hernandez1999(xy):
    """
    Returns the correlated colour temperature :math:`T_{cp}` from given
    *CIE XYZ* colourspace *xy* chromaticity coordinates using
    *Javier Hernandez-Andres, Raymond L. Lee, Jr., and Javier Romero (1999)*
    method.

    Parameters
    ----------
    xy : array_like
        *xy* chromaticity coordinates.

    Returns
    -------
    numeric
        Correlated colour temperature :math:`T_{cp}`.

    References
    ----------
    .. [10]  `Calculating correlated color temperatures across the entire gamut
            of daylight and skylight chromaticities
            <http://www.ugr.es/~colorimg/pdfs/ao_1999_5703.pdf>`_,
            DOI: http://dx.doi.org/10.1364/AO.38.005703

    Examples
    --------
    >>> xy_to_CCT_hernandez1999((0.31271, 0.32902))  # doctest: +ELLIPSIS
    6500.0421533...
    """

    x, y = xy

    n = (x - 0.3366) / (y - 0.1735)
    CCT = (-949.86315 +
           6253.80338 * math.exp(-n / 0.92159) +
           28.70599 * math.exp(-n / 0.20039) +
           0.00004 * math.exp(-n / 0.07125))

    if CCT > 50000:
        n = (x - 0.3356) / (y - 0.1691)
        CCT = (36284.48953 +
               0.00228 * math.exp(-n / 0.07861) +
               5.4535e-36 * math.exp(-n / 0.01543))

    return CCT


def CCT_to_xy_kang2002(CCT):
    """
    Returns the *CIE XYZ* colourspace *xy* chromaticity coordinates from given
    correlated colour temperature :math:`T_{cp}` using
    *Bongsoon Kang Ohak Moon, Changhee Hong, Honam Lee, Bonghwan Cho and
    Youngsun Kim (2002)* method.

    Parameters
    ----------
    CCT : numeric
        Correlated colour temperature :math:`T_{cp}`.

    Returns
    -------
    tuple
        *xy* chromaticity coordinates.

    Raises
    ------
    ValueError
        If the correlated colour temperature is not in appropriate domain.

    References
    ----------
    .. [11] `Design of Advanced Color -
            Temperature Control System for HDTV Applications
            <http://icpr.snu.ac.kr/resource/wop.pdf/J01/2002/041/R06/J012002041R060865.pdf>`_  # noqa

    Examples
    --------
    >>> CCT_to_xy_kang2002(6504.38938305)  # doctest: +ELLIPSIS
    (0.3134259..., 0.3235959...)
    """

    if 1667 <= CCT <= 4000:
        x = (-0.2661239 * 10 ** 9 / CCT ** 3 -
             0.2343589 * 10 ** 6 / CCT ** 2 +
             0.8776956 * 10 ** 3 / CCT +
             0.179910)
    elif 4000 <= CCT <= 25000:
        x = (-3.0258469 * 10 ** 9 / CCT ** 3 +
             2.1070379 * 10 ** 6 / CCT ** 2 +
             0.2226347 * 10 ** 3 / CCT +
             0.24039)
    else:
        raise ValueError(
            'Correlated colour temperature must be in domain [1667, 25000]!')

    if 1667 <= CCT <= 2222:
        y = (-1.1063814 * x ** 3 -
             1.34811020 * x ** 2 +
             2.18555832 * x -
             0.20219683)
    elif 2222 <= CCT <= 4000:
        y = (-0.9549476 * x ** 3 -
             1.37418593 * x ** 2 +
             2.09137015 * x -
             0.16748867)
    elif 4000 <= CCT <= 25000:
        y = (3.0817580 * x ** 3 -
             5.8733867 * x ** 2 +
             3.75112997 * x -
             0.37001483)

    return x, y


def CCT_to_xy_illuminant_D(CCT):
    """
    Converts from the correlated colour temperature :math:`T_{cp}` of a
    *CIE Illuminant D Series* to the chromaticity of that
    *CIE Illuminant D Series* using *D. B. Judd, D. L. Macadam, G. Wyszecki,
    H. W. Budde, H. R. Condit, S. T. Henderson and J. L. Simonds* method.

    Parameters
    ----------
    CCT : numeric
        Correlated colour temperature :math:`T_{cp}`.

    Returns
    -------
    tuple
        *xy* chromaticity coordinates.

    Raises
    ------
    ValueError
        If the correlated colour temperature is not in appropriate domain.

    References
    ----------
    .. [12] **Wyszecki & Stiles**,
            *Color Science - Concepts and Methods Data and Formulae -
            Second Edition*,
            Wiley Classics Library Edition, published 2000,
            ISBN-10: 0-471-39918-3,
            page  145.

    Examples
    --------
    >>> CCT_to_xy_illuminant_D(6504.38938305)  # doctest: +ELLIPSIS
    (0.3127077..., 0.3291128...)
    """

    if 4000 <= CCT <= 7000:
        x = (-4.607 * 10 ** 9 / CCT ** 3 +
             2.9678 * 10 ** 6 / CCT ** 2 +
             0.09911 * 10 ** 3 / CCT +
             0.244063)
    elif 7000 < CCT <= 25000:
        x = (-2.0064 * 10 ** 9 / CCT ** 3 +
             1.9018 * 10 ** 6 / CCT ** 2 +
             0.24748 * 10 ** 3 / CCT +
             0.23704)
    else:
        raise ValueError(
            'Correlated colour temperature must be in domain [4000, 25000]!')

    y = -3 * x ** 2 + 2.87 * x - 0.275

    return x, y


XY_TO_CCT_METHODS = CaseInsensitiveMapping(
    {'McCamy 1992': xy_to_CCT_mccamy1992,
     'Hernandez 1999': xy_to_CCT_hernandez1999})
"""
Supported *CIE XYZ* colourspace *xy* chromaticity coordinates to correlated
colour temperature :math:`T_{cp}` computation methods.

XY_TO_CCT_METHODS : CaseInsensitiveMapping
    {'McCamy 1992', 'Hernandez 1999'}

Aliases:

-   'mccamy1992': 'McCamy 1992'
-   'hernandez1999': 'Hernandez 1999'
"""
XY_TO_CCT_METHODS['mccamy1992'] = XY_TO_CCT_METHODS['McCamy 1992']
XY_TO_CCT_METHODS['hernandez1999'] = XY_TO_CCT_METHODS['Hernandez 1999']


def xy_to_CCT(xy, method='McCamy 1992', **kwargs):
    """
    Returns the correlated colour temperature :math:`T_{cp}` from given
    *CIE XYZ* colourspace *xy* chromaticity coordinates using given method.

    Parameters
    ----------
    xy : array_like
        *xy* chromaticity coordinates.
    method : unicode, optional
        {'McCamy 1992', 'Hernandez 1999'}
        Computation method.
    \*\*kwargs : \*\*
        Keywords arguments.

    Returns
    -------
    numeric
        Correlated colour temperature :math:`T_{cp}`.
    """

    return XY_TO_CCT_METHODS.get(method)(xy)


CCT_TO_XY_METHODS = CaseInsensitiveMapping(
    {'Kang 2002': CCT_to_xy_kang2002,
     'CIE Illuminant D Series': CCT_to_xy_illuminant_D})
"""
Supported correlated colour temperature :math:`T_{cp}` to *CIE XYZ* colourspace
*xy* chromaticity coordinates computation methods.

CCT_TO_XY_METHODS : CaseInsensitiveMapping
    {'Kang 2002', 'CIE Illuminant D Series'}

Aliases:

-   'kang2002': 'Kang 2002'
-   'cie_d': 'Hernandez 1999'
"""
CCT_TO_XY_METHODS['kang2002'] = CCT_TO_XY_METHODS['Kang 2002']
CCT_TO_XY_METHODS['cie_d'] = CCT_TO_XY_METHODS['CIE Illuminant D Series']


def CCT_to_xy(CCT, method='Kang 2002'):
    """
    Returns the *CIE XYZ* colourspace *xy* chromaticity coordinates from given
    correlated colour temperature :math:`T_{cp}` using given method.

    Parameters
    ----------
    CCT : numeric
        Correlated colour temperature :math:`T_{cp}`.
    method : unicode, optional
        {'Kang 2002', 'CIE Illuminant D Series'}
        Computation method.

    Returns
    -------
    tuple
        *xy* chromaticity coordinates.
    """

    return CCT_TO_XY_METHODS.get(method)(CCT)
