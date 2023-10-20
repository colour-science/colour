"""
Robertson (1968) Correlated Colour Temperature
==============================================

Defines the *Robertson (1968)* correlated colour temperature :math:`T_{cp}`
computations objects:

-   :func:`colour.temperature.mired_to_CCT`: Micro reciprocal degree to
    correlated colour temperature :math:`T_{cp}` computation.
-   :func:`colour.temperature.CCT_to_mired`: Correlated colour
    temperature :math:`T_{cp}` to micro reciprocal degree computation.
-   :func:`colour.temperature.uv_to_CCT_Robertson1968`: Correlated colour
    temperature :math:`T_{cp}` and :math:`\\Delta_{uv}` computation of given
    *CIE UCS* colourspace *uv* chromaticity coordinates using *Robertson
    (1968)* method.
-   :func:`colour.temperature.CCT_to_uv_Robertson1968`: *CIE UCS* colourspace
    *uv* chromaticity coordinates computation of given correlated colour
    temperature :math:`T_{cp}` and :math:`\\Delta_{uv}` using *Robertson
    (1968)* method.

References
----------
-   :cite:`AdobeSystems2013` : Adobe Systems. (2013). Adobe DNG Software
    Development Kit (SDK) - 1.3.0.0 -
    dng_sdk_1_3/dng_sdk/source/dng_temperature.cpp::dng_temperature::\
Set_xy_coord. https://www.adobe.com/support/downloads/dng/dng_sdk.html
-   :cite:`AdobeSystems2013a` : Adobe Systems. (2013). Adobe DNG Software
    Development Kit (SDK) - 1.3.0.0 -
    dng_sdk_1_3/dng_sdk/source/dng_temperature.cpp::dng_temperature::xy_coord.
    https://www.adobe.com/support/downloads/dng/dng_sdk.html
-   :cite:`Wyszecki2000x` : Wyszecki, Günther, & Stiles, W. S. (2000). Table
    1(3.11) Isotemperature Lines. In Color Science: Concepts and Methods,
    Quantitative Data and Formulae (p. 228). Wiley. ISBN:978-0-471-39918-6
-   :cite:`Wyszecki2000y` : Wyszecki, Günther, & Stiles, W. S. (2000).
    DISTRIBUTION TEMPERATURE, COLOR TEMPERATURE, AND CORRELATED COLOR
    TEMPERATURE. In Color Science: Concepts and Methods, Quantitative Data and
    Formulae (pp. 224-229). Wiley. ISBN:978-0-471-39918-6
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from colour.algebra import sdiv, sdiv_mode
from colour.hints import ArrayLike, NDArrayFloat
from colour.utilities import as_float_array, tsplit

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "DATA_ISOTEMPERATURE_LINES_ROBERTSON1968",
    "ISOTemperatureLine_Specification_Robertson1968",
    "ISOTEMPERATURE_LINES_ROBERTSON1968",
    "mired_to_CCT",
    "CCT_to_mired",
    "uv_to_CCT_Robertson1968",
    "CCT_to_uv_Robertson1968",
]

DATA_ISOTEMPERATURE_LINES_ROBERTSON1968: tuple = (
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
    (325, 0.24792, 0.34655, -2.4681),  # 0.24702 --> 0.24792 Bruce Lindbloom
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
    (600, 0.33724, 0.36051, -116.45),
)
"""
*Robertson (1968)* iso-temperature lines as a *tuple* as follows::

    (
        ('Reciprocal Megakelvin', 'CIE 1960 Chromaticity Coordinate *u*',
         'CIE 1960 Chromaticity Coordinate *v*', 'Slope'),
        ...,
        ('Reciprocal Megakelvin', 'CIE 1960 Chromaticity Coordinate *u*',
         'CIE 1960 Chromaticity Coordinate *v*', 'Slope'),
    )

Notes
-----
-   A correction has been done by Lindbloom for *325* Megakelvin
    temperature: 0.24702 --> 0.24792

References
----------
:cite:`Wyszecki2000x`
"""


@dataclass
class ISOTemperatureLine_Specification_Robertson1968:
    """
    Define the data for a *Roberston (1968)* iso-temperature line.

    Parameters
    ----------
    r
        Temperature :math:`r` in reciprocal mega-kelvin degrees.
    u
        *u* chromaticity coordinate of the temperature :math:`r`.
    v
        *v* chromaticity coordinate of the temperature :math:`r`.
    t
        Slope of the *v* chromaticity coordinate.
    """

    r: float
    u: float
    v: float
    t: float


ISOTEMPERATURE_LINES_ROBERTSON1968: list = [
    ISOTemperatureLine_Specification_Robertson1968(*x)
    for x in DATA_ISOTEMPERATURE_LINES_ROBERTSON1968
]


def mired_to_CCT(mired: ArrayLike) -> NDArrayFloat:
    """
    Convert given micro reciprocal degree to correlated colour temperature
    :math:`T_{cp}`.

    Parameters
    ----------
    mired
         Micro reciprocal degree (mired).

    Returns
    -------
    :class:`numpy.ndarray`
        Correlated colour temperature :math:`T_{cp}`.

    Examples
    --------
    >>> CCT_to_mired(153.84615384615384)  # doctest: +ELLIPSIS
    6500.0
    """

    mired = as_float_array(mired)

    with sdiv_mode():
        return sdiv(1.0e6, mired)


def CCT_to_mired(CCT: ArrayLike) -> NDArrayFloat:
    """
    Convert given correlated colour temperature :math:`T_{cp}` to micro
    reciprocal degree (mired).

    Parameters
    ----------
    CCT
         Correlated colour temperature :math:`T_{cp}`.

    Returns
    -------
    :class:`numpy.ndarray`
        Micro reciprocal degree (mired).

    Examples
    --------
    >>> CCT_to_mired(6500)  # doctest: +ELLIPSIS
    153.8461538...
    """

    CCT = as_float_array(CCT)

    with sdiv_mode():
        return sdiv(1.0e6, CCT)


def _uv_to_CCT_Robertson1968(uv: ArrayLike) -> NDArrayFloat:
    """
    Return the correlated colour temperature :math:`T_{cp}` and
    :math:`\\Delta_{uv}` from given *CIE UCS* colourspace *uv* chromaticity
    coordinates using *Roberston (1968)* method.

    Parameters
    ----------
    uv
        *CIE UCS* colourspace *uv* chromaticity coordinates.

    Returns
    -------
    :class:`numpy.ndarray`
        Correlated colour temperature :math:`T_{cp}`, :math:`\\Delta_{uv}`.
    """

    u, v = tsplit(uv)

    last_dt = last_dv = last_du = 0

    D_uv = 0
    for i in range(1, 31):
        wr_ruvt = ISOTEMPERATURE_LINES_ROBERTSON1968[i]
        wr_ruvt_previous = ISOTEMPERATURE_LINES_ROBERTSON1968[i - 1]

        du = 1
        dv = wr_ruvt.t

        length = np.hypot(1, dv)

        du /= length
        dv /= length

        uu = u - wr_ruvt.u
        vv = v - wr_ruvt.v

        dt = -uu * dv + vv * du

        if dt <= 0 or i == 30:
            if dt > 0:
                dt = 0

            dt = -dt

            f = 0 if i == 1 else dt / (last_dt + dt)

            T = mired_to_CCT(wr_ruvt_previous.r * f + wr_ruvt.r * (1 - f))

            uu = u - (wr_ruvt_previous.u * f + wr_ruvt.u * (1 - f))
            vv = v - (wr_ruvt_previous.v * f + wr_ruvt.v * (1 - f))

            du = du * (1 - f) + last_du * f
            dv = dv * (1 - f) + last_dv * f

            length = np.hypot(du, dv)

            du /= length
            dv /= length

            D_uv = uu * du + vv * dv

            break

        last_dt = dt
        last_du = du
        last_dv = dv

    return np.array([T, -D_uv])


def uv_to_CCT_Robertson1968(uv: ArrayLike) -> NDArrayFloat:
    """
    Return the correlated colour temperature :math:`T_{cp}` and
    :math:`\\Delta_{uv}` from given *CIE UCS* colourspace *uv* chromaticity
    coordinates using *Roberston (1968)* method.

    Parameters
    ----------
    uv
        *CIE UCS* colourspace *uv* chromaticity coordinates.

    Returns
    -------
    :class:`numpy.ndarray`
        Correlated colour temperature :math:`T_{cp}`, :math:`\\Delta_{uv}`.

    References
    ----------
    :cite:`AdobeSystems2013`, :cite:`Wyszecki2000y`

    Examples
    --------
    >>> uv = np.array([0.193741375998230, 0.315221043940594])
    >>> uv_to_CCT_Robertson1968(uv)  # doctest: +ELLIPSIS
    array([  6.5000162...e+03,   8.3333289...e-03])
    """

    uv = as_float_array(uv)

    CCT_D_uv = [_uv_to_CCT_Robertson1968(a) for a in np.reshape(uv, (-1, 2))]

    return np.reshape(CCT_D_uv, uv.shape)


def _CCT_to_uv_Robertson1968(CCT_D_uv: ArrayLike) -> NDArrayFloat:
    """
    Return the *CIE UCS* colourspace *uv* chromaticity coordinates from given
    correlated colour temperature :math:`T_{cp}` and :math:`\\Delta_{uv}` using
    *Roberston (1968)* method.

    Parameters
    ----------
    CCT_D_uv
        Correlated colour temperature :math:`T_{cp}`, :math:`\\Delta_{uv}`.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE UCS* colourspace *uv* chromaticity coordinates.
    """

    CCT, D_uv = tsplit(CCT_D_uv)

    r = CCT_to_mired(CCT)

    u, v = 0, 0
    for i in range(30):
        wr_ruvt = ISOTEMPERATURE_LINES_ROBERTSON1968[i]
        wr_ruvt_next = ISOTEMPERATURE_LINES_ROBERTSON1968[i + 1]

        if r < wr_ruvt_next.r or i == 29:
            f = (wr_ruvt_next.r - r) / (wr_ruvt_next.r - wr_ruvt.r)

            u = wr_ruvt.u * f + wr_ruvt_next.u * (1 - f)
            v = wr_ruvt.v * f + wr_ruvt_next.v * (1 - f)

            uu1 = uu2 = 1.0
            vv1, vv2 = wr_ruvt.t, wr_ruvt_next.t

            length1 = np.hypot(1, vv1)
            length2 = np.hypot(1, vv2)

            uu1 /= length1
            vv1 /= length1

            uu2 /= length2
            vv2 /= length2

            uu3 = uu1 * f + uu2 * (1 - f)
            vv3 = vv1 * f + vv2 * (1 - f)

            len3 = np.sqrt(uu3 * uu3 + vv3 * vv3)

            uu3 /= len3
            vv3 /= len3

            u += uu3 * -D_uv
            v += vv3 * -D_uv

            break

    return np.array([u, v])


def CCT_to_uv_Robertson1968(CCT_D_uv: ArrayLike) -> NDArrayFloat:
    """
    Return the *CIE UCS* colourspace *uv* chromaticity coordinates from given
    correlated colour temperature :math:`T_{cp}` and :math:`\\Delta_{uv}` using
    *Roberston (1968)* method.

    Parameters
    ----------
    CCT_D_uv
        Correlated colour temperature :math:`T_{cp}`, :math:`\\Delta_{uv}`.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE UCS* colourspace *uv* chromaticity coordinates.

    References
    ----------
    :cite:`AdobeSystems2013a`, :cite:`Wyszecki2000y`

    Examples
    --------
    >>> CCT_D_uv = np.array([6500.0081378199056, 0.008333331244225])
    >>> CCT_to_uv_Robertson1968(CCT_D_uv)  # doctest: +ELLIPSIS
    array([ 0.1937413...,  0.3152210...])
    """

    CCT_D_uv = as_float_array(CCT_D_uv)

    uv = [_CCT_to_uv_Robertson1968(a) for a in np.reshape(CCT_D_uv, (-1, 2))]

    return np.reshape(uv, CCT_D_uv.shape)
