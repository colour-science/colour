# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**temperature.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Color** package *correlated color temperature* manipulation objects.

**Others:**
    :func:`color.temperature.get_planckian_table`, :func:`color.temperature.get_planckian_table_minimal_distance_index`,
    :func:`color.temperature.uv_to_CCT`, :func:`color.temperature.CCT_to_uv`, :func:`color.temperature.XYZ_to_CCT`,
    and :func:`color.temperature.CCT_to_XYZ` definitions implement **Yoshi Ohno**,
    `Practical Use and Calculation of CCT and Duv <http://dx.doi.org/10.1080/15502724.2014.839020>`_ paper.

    :func:`color.temperature.xy_to_CCT`, :func:`color.temperature.CCT_to_xy` definitions are implemented from
    **Adobe DNG SDK 1.3.0.0**, the :attr:`color.temperature.WYSZECKI_ROBERSTON_ISOTEMPERATURE_LINES_DATA` attribute data is
    from **Wyszecki & Stiles**, *Color Science - Concepts and Methods Data and Formulae - Second Edition*, Page 228.
"""

from __future__ import unicode_literals

import math
import numpy
from collections import namedtuple

import color.spectral.blackbody
import color.spectral.cmfs
import color.spectral.transformations
import color.transformations
import color.exceptions
import color.verbose

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["LOGGER",
           "PLANCKIAN_TABLE_TUVD",
           "CCT_MINIMAL",
           "CCT_MAXIMAL",
           "CCT_SAMPLES",
           "CCT_CALCULATION_ITERATIONS",
           "WYSZECKI_ROBERSTON_ISOTEMPERATURE_LINES_DATA",
           "WYSZECKI_ROBERSTON_ISOTEMPERATURE_LINES_RUVT",
           "WYSZECKI_ROBERSTON_ISOTEMPERATURE_LINES",
           "get_planckian_table",
           "get_planckian_table_minimal_distance_index",
           "uv_to_CCT_ohno",
           "CCT_to_uv_ohno",
           "uv_to_CCT_robertson",
           "CCT_to_uv_robertson",
           "uv_to_CCT",
           "CCT_to_uv",
           "D_illuminant_CCT_to_xy"]

LOGGER = color.verbose.install_logger()

PLANCKIAN_TABLE_TUVD = namedtuple("PlanckianTable_Tuvdi", ("Ti", "ui", "vi", "di"))

CCT_MINIMAL = 1000
CCT_MAXIMAL = 100000
CCT_SAMPLES = 10
CCT_CALCULATION_ITERATIONS = 6

# **Wyszecki & Stiles**, *Color Science - Concepts and Methods Data and Formulae - Second Edition*, Page 228.
# (Reciprocal Megakelvin, CIE 1960 Chromaticity Coordinate *u*, CIE 1960 Chromaticity Coordinate *v*, Slope)
WYSZECKI_ROBERSTON_ISOTEMPERATURE_LINES_DATA = ((0, 0.18006, 0.26352, -0.24341),
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
                                                (325, 0.24792, 0.34655, -2.4681),
                                                # 0.24702 ---> 0.24792 Bruce Lindbloom
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

WYSZECKI_ROBERSTON_ISOTEMPERATURE_LINES_RUVT = namedtuple("WyszeckiRoberston_ruvt", ("r", "u", "v", "t"))

WYSZECKI_ROBERSTON_ISOTEMPERATURE_LINES = map(lambda x: WYSZECKI_ROBERSTON_ISOTEMPERATURE_LINES_RUVT(*x),
                                              WYSZECKI_ROBERSTON_ISOTEMPERATURE_LINES_DATA)


def get_planckian_table(uv, cmfs, start, end, count):
    """
    Returns a planckian table from given *CIE UCS* colorspace *uv* chromaticity coordinates, color matching functions and
    temperature range  using *Yoshi Ohno* calculation methods.

    Usage::

        >>> import pprint
        >>> cmfs = color.STANDARD_OBSERVERS_XYZ_COLOR_MATCHING_FUNCTIONS.get("Standard CIE 1931 2 Degree Observer")
        >>> pprint.pprint(get_planckian_table((0.1978, 0.3122), cmfs, 1000, 1010, 10))
        [PlanckianTableTuvdi(Ti=1000.0, ui=0.44800695592713469, vi=0.35462532232761207, di=0.2537783063402483),
         PlanckianTableTuvdi(Ti=1001.1111111111111, ui=0.44774688726773565, vi=0.3546478595072966, di=0.25352567371290297),
         PlanckianTableTuvdi(Ti=1002.2222222222222, ui=0.44748712505363253, vi=0.35467035108531186, di=0.2532733526031864),
         PlanckianTableTuvdi(Ti=1003.3333333333334, ui=0.44722766912561784, vi=0.35469279704978462, di=0.2530213428281355),
         PlanckianTableTuvdi(Ti=1004.4444444444445, ui=0.44696851932239223, vi=0.35471519738915419, di=0.2527696442026852),
         PlanckianTableTuvdi(Ti=1005.5555555555555, ui=0.44670967548058027, vi=0.35473755209217106, di=0.25251825653968457),
         PlanckianTableTuvdi(Ti=1006.6666666666666, ui=0.4464511374347529, vi=0.35475986114789521, di=0.25226717964991896),
         PlanckianTableTuvdi(Ti=1007.7777777777778, ui=0.44619290501744918, vi=0.3547821245456938, di=0.2520164133421324),
         PlanckianTableTuvdi(Ti=1008.8888888888889, ui=0.44593497805919297, vi=0.35480434227524021, di=0.251765957423044),
         PlanckianTableTuvdi(Ti=1010.0, ui=0.4456773563885123, vi=0.35482651432651208, di=0.251515811697368)]

    :param uv: *uv* chromaticity coordinates.
    :type uv: tuple
    :param cmfs: Standard observer color matching functions.
    :type cmfs: XYZ_ColorMatchingFunctions
    :param start: Temperature range start in kelvins.
    :type start: float
    :param end: Temperature range end in kelvins.
    :type end: float
    :param count: Temperatures count in the planckian table.
    :type count: int
    :return: Planckian table.
    :rtype: list
    """

    ux, vx = uv

    planckian_table = []
    for Ti in numpy.linspace(start, end, count):
        spd = color.spectral.blackbody.blackbody_spectral_power_distribution(Ti, *cmfs.shape)
        XYZ = color.spectral.transformations.spectral_to_XYZ(spd, cmfs)
        XYZ *= 1. / numpy.max(XYZ)
        UVW = color.transformations.XYZ_to_UCS(XYZ)
        ui, vi = color.transformations.UCS_to_uv(UVW)
        di = math.sqrt((ux - ui) ** 2 + (vx - vi) ** 2)
        planckian_table.append(PLANCKIAN_TABLE_TUVD(Ti, ui, vi, di))

    return planckian_table


def get_planckian_table_minimal_distance_index(planckian_table):
    """
    Returns the shortest distance index in given planckian table using *Yoshi Ohno* calculation methods.

    Usage::

        >>> cmfs = color.STANDARD_OBSERVERS_XYZ_COLOR_MATCHING_FUNCTIONS.get("Standard CIE 1931 2 Degree Observer")
        >>> get_planckian_table_minimal_distance_index(get_planckian_table((0.1978, 0.3122), cmfs, 1000, 1010, 10)))
        9

    :param planckian_table: Planckian table.
    :type planckian_table: list
    :return: Shortest distance index.
    :rtype: int
    """

    distances = map(lambda x: x.di, planckian_table)
    return distances.index(min(distances))


def uv_to_CCT_ohno(uv,
                   cmfs=color.spectral.cmfs.STANDARD_OBSERVERS_XYZ_COLOR_MATCHING_FUNCTIONS.get(
                       "Standard CIE 1931 2 Degree Observer"),
                   start=CCT_MINIMAL,
                   end=CCT_MAXIMAL,
                   count=CCT_SAMPLES,
                   iterations=CCT_CALCULATION_ITERATIONS):
    """
    | Returns the correlated color temperature and Duv from given *CIE UCS* colorspace *uv* chromaticity coordinates,
        color matching functions and temperature range using *Yoshi Ohno* calculation methods.
    | The iterations parameter defines the calculations precision: The higher its value, the more planckian tables
        will be generated through cascade expansion in order to converge to the exact solution.

    Usage::

        >>> cmfs = color.STANDARD_OBSERVERS_XYZ_COLOR_MATCHING_FUNCTIONS.get("Standard CIE 1931 2 Degree Observer")
        >>> uv_to_CCT_ohno((0.1978, 0.3122), cmfs)
        (6507.4342201047066, 0.003223690901512735)

    :param uv: *uv* chromaticity coordinates.
    :type uv: tuple
    :param cmfs: Standard observer color matching functions.
    :type cmfs: XYZ_ColorMatchingFunctions
    :param start: Temperature range start in kelvins.
    :type start: float
    :param end: Temperature range end in kelvins.
    :type end: float
    :param count: Temperatures count in the planckian tables.
    :type count: int
    :param iterations: Number of planckian tables to generate.
    :type iterations: int
    :return: Correlated color temperature, Duv.
    :rtype: tuple
    """

    # Ensuring we do at least one iteration to initialize variables.
    if iterations <= 0:
        iterations = 1

    # Planckian table creation through cascade expansion.
    for i in range(iterations):
        planckian_table = get_planckian_table(uv, cmfs, start, end, count)
        index = get_planckian_table_minimal_distance_index(planckian_table)
        if index == 0:
            LOGGER.warning(
                "!> {0} | Minimal distance index is on lowest planckian table bound, unpredictable results may occur!".format(
                    __name__))
            index += 1
        elif index == len(planckian_table) - 1:
            LOGGER.warning(
                "!> {0} | Minimal distance index is on highest planckian table bound, unpredictable results may occur!".format(
                    __name__))
            index -= 1

        start = planckian_table[index - 1].Ti
        end = planckian_table[index + 1].Ti

    ux, vx = uv

    Tuvdip, Tuvdi, Tuvdin = planckian_table[index - 1], planckian_table[index], planckian_table[index + 1]
    Tip, uip, vip, dip = Tuvdip.Ti, Tuvdip.ui, Tuvdip.vi, Tuvdip.di
    Ti, ui, vi, di = Tuvdi.Ti, Tuvdi.ui, Tuvdi.vi, Tuvdi.di
    Tin, uin, vin, din = Tuvdin.Ti, Tuvdin.ui, Tuvdin.vi, Tuvdin.di

    # Triangular solution.
    l = math.sqrt((uin - uip) ** 2 + (vin - vip) ** 2)
    x = (dip ** 2 - din ** 2 + l ** 2) / (2 * l)
    T = Tip + (Tin - Tip) * (x / l)

    vtx = vip + (vin - vip) * (x / l)
    sign = 1. if vx - vtx >= 0. else -1.
    Duv = (dip ** 2 - x ** 2) ** (1. / 2.) * sign

    # Parabolic solution.
    if Duv < 0.002:
        X = (Tin - Ti) * (Tip - Tin) * (Ti - Tip)
        a = (Tip * (din - di) + Ti * (dip - din) + Tin * (di - dip)) * X ** - 1
        b = -(Tip ** 2 * (din - di) + Ti ** 2 * (dip - din) + Tin ** 2 * (di - dip)) * X ** - 1
        c = -(dip * (Tin - Ti) * Ti * Tin + di * (Tip - Tin) * Tip * Tin + din * (Ti - Tip) * Tip * Ti) * X ** -1

        T = -b / (2. * a)

        Duv = sign * (a * T ** 2 + b * T + c)

    return T, Duv


def CCT_to_uv_ohno(CCT,
                   Duv=0.,
                   cmfs=color.spectral.cmfs.STANDARD_OBSERVERS_XYZ_COLOR_MATCHING_FUNCTIONS.get(
                       "Standard CIE 1931 2 Degree Observer")):
    """
    Returns the *CIE UCS* colorspace *uv* chromaticity coordinates from given correlated color temperature, Duv and
    color matching functions using *Yoshi Ohno* calculation methods.

    Usage::

        >>> cmfs = color.STANDARD_OBSERVERS_XYZ_COLOR_MATCHING_FUNCTIONS.get("Standard CIE 1931 2 Degree Observer")
        >>> CCT_to_uv_ohno(6507.4342201047066, 0.003223690901512735, cmfs)
        (0.19779977151790701, 0.31219970605380082)

    :param CCT: Correlated color temperature.
    :type CCT: float
    :param Duv: Duv.
    :type Duv: float
    :param cmfs: Standard observer color matching functions.
    :type cmfs: XYZ_ColorMatchingFunctions
    :return: *uv* chromaticity coordinates.
    :rtype: tuple
    """

    delta = 0.01

    spd = color.spectral.blackbody.blackbody_spectral_power_distribution(CCT, *cmfs.shape)
    XYZ = color.spectral.transformations.spectral_to_XYZ(spd, cmfs)
    XYZ *= 1. / numpy.max(XYZ)
    UVW = color.transformations.XYZ_to_UCS(XYZ)
    u0, v0 = color.transformations.UCS_to_uv(UVW)

    if Duv == 0.:
        return u0, v0
    else:
        spd = color.spectral.blackbody.blackbody_spectral_power_distribution(CCT + delta, *cmfs.shape)
        XYZ = color.spectral.transformations.spectral_to_XYZ(spd, cmfs)
        XYZ *= 1. / numpy.max(XYZ)
        UVW = color.transformations.XYZ_to_UCS(XYZ)
        u1, v1 = color.transformations.UCS_to_uv(UVW)

        du = u0 - u1
        dv = v0 - v1

        u = u0 - Duv * (dv / math.sqrt(du ** 2 + dv ** 2))
        v = v0 + Duv * (du / math.sqrt(du ** 2 + dv ** 2))

        return u, v


def uv_to_CCT_robertson(uv):
    """
    Returns the correlated color temperature and Duv from given *CIE UCS* colorspace *uv* chromaticity coordinates using
    *Wyszecki & Roberston* calculation method.
    This implementation is only valid for *Standard CIE 1931 2 Degree Observer*.

    Reference: **Adobe DNG SDK 1.3.0.0**: *dng_sdk_1_3/dng_sdk/source/dng_temperature.cpp*: *dng_temperature::Set_xy_coord*

    Usage::

        >>> uv_to_CCT_robertson((0.19374137599822966, 0.31522104394059397))
        (6500.016287949829, 0.008333328983860189)

    :param uv: *uv* chromaticity coordinates.
    :type uv: tuple
    :return: Correlated color temperature, Duv.
    :rtype: tuple
    """

    u, v = uv

    last_dt = last_dv = last_du = 0.0

    for i in range(1, 31):
        wr_ruvt, wr_ruvt_previous = WYSZECKI_ROBERSTON_ISOTEMPERATURE_LINES[i], \
                                    WYSZECKI_ROBERSTON_ISOTEMPERATURE_LINES[i - 1]

        du = 1.0
        dv = wr_ruvt.t

        len = math.sqrt(1. + dv * dv)

        du /= len
        dv /= len

        uu = u - wr_ruvt.u
        vv = v - wr_ruvt.v

        dt = -uu * dv + vv * du

        if dt <= 0. or i == 30:
            if dt > 0.0:
                dt = 0.0

            dt = -dt

            if i == 1:
                f = 0.0
            else:
                f = dt / (last_dt + dt)

            T = 1.0e6 / (wr_ruvt_previous.r * f + wr_ruvt.r * (1. - f))

            uu = u - (wr_ruvt_previous.u * f + wr_ruvt.u * (1. - f))
            vv = v - (wr_ruvt_previous.v * f + wr_ruvt.v * (1. - f))

            du = du * (1. - f) + last_du * f
            dv = dv * (1. - f) + last_dv * f

            len = math.sqrt(du * du + dv * dv)

            du /= len
            dv /= len

            Duv = uu * du + vv * dv

            break

        last_dt = dt
        last_du = du
        last_dv = dv

    return T, -Duv


def CCT_to_uv_robertson(CCT, Duv=0.):
    """
    Returns the *CIE UCS* colorspace *uv* chromaticity coordinates from given correlated color temperature and Duv using
    *Wyszecki & Roberston* calculation method.
    This implementation is only valid for *Standard CIE 1931 2 Degree Observer*.

    Reference: **Adobe DNG SDK 1.3.0.0**: *dng_sdk_1_3/dng_sdk/source/dng_temperature.cpp*: *dng_temperature::Get_xy_coord*

    Usage::

        >>> CCT_to_uv_robertson(6500.0081378199056, 0.0083333312442250979)
        (0.19374137599822966, 0.31522104394059397)

    :param CCT: Correlated color temperature.
    :type CCT: float
    :param Duv: Duv.
    :type Duv: float
    :return: *uv* chromaticity coordinates.
    :rtype: tuple
    """

    r = 1.0e6 / CCT

    for i in range(30):
        wr_ruvt, wr_ruvt_next = WYSZECKI_ROBERSTON_ISOTEMPERATURE_LINES[i], \
                                WYSZECKI_ROBERSTON_ISOTEMPERATURE_LINES[i + 1]

        if r < wr_ruvt_next.r or i == 29:
            f = (wr_ruvt_next.r - r) / (wr_ruvt_next.r - wr_ruvt.r)

            u = wr_ruvt.u * f + wr_ruvt_next.u * (1. - f)
            v = wr_ruvt.v * f + wr_ruvt_next.v * (1. - f)

            uu1 = uu2 = 1.0
            vv1, vv2 = wr_ruvt.t, wr_ruvt_next.t

            len1, len2 = math.sqrt(1. + vv1 * vv1), math.sqrt(1. + vv2 * vv2)

            uu1 /= len1
            vv1 /= len1

            uu2 /= len2
            vv2 /= len2

            uu3 = uu1 * f + uu2 * (1. - f)
            vv3 = vv1 * f + vv2 * (1. - f)

            len3 = math.sqrt(uu3 * uu3 + vv3 * vv3)

            uu3 /= len3
            vv3 /= len3

            u += uu3 * -Duv
            v += vv3 * -Duv

            return u, v


def uv_to_CCT(uv, method="Yoshi Ohno", **kwargs):
    """
    Returns the correlated color temperature and Duv from given *CIE UCS* colorspace *uv* chromaticity coordinates and method.
    Defines a wrapper for :func:`uv_to_CCT_ohno` and :func:`uv_to_CCT_robertson` definitions.

    :param uv: *uv* chromaticity coordinates.
    :type uv: tuple
    :param method: Calculation method.
    :type method: unicode ("Yoshi Ohno", "Wyszecki Robertson")
    :param \*\*kwargs: Keywords arguments.
    :type \*\*kwargs: \*\*
    :return: Correlated color temperature, Duv.
    :rtype: tuple
    """

    if method == "Yoshi Ohno":
        return uv_to_CCT_ohno(uv, **kwargs)
    else:
        if "cmfs" in kwargs:
            if kwargs.get("cmfs").name != "Standard CIE 1931 2 Degree Observer":
                raise color.exceptions.ProgrammingError(
                    "Wyszecki & Roberston calculation method is only valid for 'Standard CIE 1931 2 Degree Observer'!")

        return uv_to_CCT_robertson(uv)


def CCT_to_uv(CCT, Duv=0., method="Yoshi Ohno", **kwargs):
    """
    Returns the *CIE UCS* colorspace *uv* chromaticity coordinates from given correlated color temperature
    and Duv using given method.
    Defines a wrapper for :func:`CCT_to_uv_ohno` and :func:`CCT_to_uv_robertson` definitions.

    :param CCT: Correlated color temperature.
    :type CCT: float
    :param Duv: Duv.
    :type Duv: float
    :param method: Calculation method.
    :type method: unicode ("Yoshi Ohno", "Wyszecki Robertson")
    :param \*\*kwargs: Keywords arguments.
    :type \*\*kwargs: \*\*
    :return: *uv* chromaticity coordinates.
    :rtype: tuple
    """

    if method == "Yoshi Ohno":
        return CCT_to_uv_ohno(CCT, Duv, **kwargs)
    else:
        if "cmfs" in kwargs:
            if kwargs.get("cmfs").name != "Standard CIE 1931 2 Degree Observer":
                raise color.exceptions.ProgrammingError(
                    "Wyszecki & Roberston calculation method is only valid for 'Standard CIE 1931 2 Degree Observer'!")

        return CCT_to_uv_robertson(CCT, Duv)


def D_illuminant_CCT_to_xy(CCT):
    """
    Converts from the correlated color temperature of a *CIE D-illuminant* to the chromaticity of that *D-illuminant*.

    Reference: http://www.brucelindbloom.com/Eqn_T_to_xy.html

    :param CCT: Correlated color temperature.
    :type CCT: float
    :return: *xy* chromaticity coordinates.
    :rtype: tuple
    """

    if 4000 <= CCT <= 7000:
        x = -4.607 * 10 ** 9 / CCT ** 3 + 2.9678 * 10 ** 6 / CCT ** 2 + 0.09911 * 10 ** 3 / CCT + 0.244063
    elif 7000 < CCT <= 25000:
        x = -2.0064 * 10 ** 9 / CCT ** 3 + 1.9018 * 10 ** 6 / CCT ** 2 + 0.24748 * 10 ** 3 / CCT + 0.23704
    else:
        raise color.exceptions.ProgrammingError("Correlated color temperature must be in domain [4000, 25000]!")

    y = -3 * x ** 2 + 2.87 * x - 0.275

    return x, y