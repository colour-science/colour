"""
Robertson (1968) Correlated Colour Temperature
==============================================

Define the *Robertson (1968)* correlated colour temperature :math:`T_{cp}`
computation objects.

-   :func:`colour.temperature.mired_to_CCT`: Convert micro reciprocal
    degrees to correlated colour temperature :math:`T_{cp}`.
-   :func:`colour.temperature.CCT_to_mired`: Convert correlated colour
    temperature :math:`T_{cp}` to micro reciprocal degrees.
-   :func:`colour.temperature.uv_to_CCT_Robertson1968`: Compute correlated
    colour temperature :math:`T_{cp}` and :math:`\\Delta_{uv}` from
    specified *CIE UCS* colourspace *uv* chromaticity coordinates using
    the *Robertson (1968)* method.
-   :func:`colour.temperature.CCT_to_uv_Robertson1968`: Compute *CIE UCS*
    colourspace *uv* chromaticity coordinates from specified correlated
    colour temperature :math:`T_{cp}` and :math:`\\Delta_{uv}` using the
    *Robertson (1968)* method.

References
----------
-   :cite:`Wyszecki2000x` : Wyszecki, Günther, & Stiles, W. S. (2000). Table
    1(3.11) Isotemperature Lines. In Color Science: Concepts and Methods,
    Quantitative Data and Formulae (p. 228). Wiley. ISBN:978-0-471-39918-6
-   :cite:`Wyszecki2000y` : Wyszecki, Günther, & Stiles, W. S. (2000).
    DISTRIBUTION TEMPERATURE, COLOR TEMPERATURE, AND CORRELATED COLOR
    TEMPERATURE. In Color Science: Concepts and Methods, Quantitative Data and
    Formulae (pp. 224-229). Wiley. ISBN:978-0-471-39918-6
"""

from __future__ import annotations

import typing
from dataclasses import dataclass

import numpy as np

from colour.algebra import sdiv, sdiv_mode

if typing.TYPE_CHECKING:
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
    Define a data structure for a *Robertson (1968)* iso-temperature line.

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
    Convert specified micro reciprocal degree (mired) to correlated colour
    temperature :math:`T_{cp}`.

    Parameters
    ----------
    mired
         Micro reciprocal degree.

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
    Convert specified correlated colour temperature :math:`T_{cp}` to micro
    reciprocal degree (mired).

    Parameters
    ----------
    CCT
        Correlated colour temperature :math:`T_{cp}`.

    Returns
    -------
    :class:`numpy.ndarray`
        Micro reciprocal degree.

    Examples
    --------
    >>> CCT_to_mired(6500)  # doctest: +ELLIPSIS
    153.8461538...
    """

    CCT = as_float_array(CCT)

    with sdiv_mode():
        return sdiv(1.0e6, CCT)


def uv_to_CCT_Robertson1968(uv: ArrayLike) -> NDArrayFloat:
    """
    Compute the correlated colour temperature :math:`T_{cp}` and
    :math:`\\Delta_{uv}` from the specified *CIE UCS* colourspace *uv*
    chromaticity coordinates using *Robertson (1968)* method.

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
    :cite:`Wyszecki2000y`

    Examples
    --------
    >>> uv = np.array([0.193741375998230, 0.315221043940594])
    >>> uv_to_CCT_Robertson1968(uv)  # doctest: +ELLIPSIS
    array([  6.5000162...e+03,   8.3333289...e-03])
    """

    uv = as_float_array(uv)
    shape = uv.shape
    uv = uv.reshape(-1, 2)

    r_itl, u_itl, v_itl, t_itl = tsplit(
        np.array(DATA_ISOTEMPERATURE_LINES_ROBERTSON1968)
    )

    # Normalized direction vectors
    length = np.hypot(1.0, t_itl)
    du_itl = 1.0 / length
    dv_itl = t_itl / length

    # Vectorized computation for all UV pairs at once
    u, v = tsplit(uv)
    u = u[:, np.newaxis]  # Shape (N, 1)
    v = v[:, np.newaxis]  # Shape (N, 1)

    # Compute distances for all UV pairs against all isotemperature lines
    # Broadcasting: (N, 1) - (30,) = (N, 30)
    uu = u - u_itl[1:]  # Shape (N, 30)
    vv = v - v_itl[1:]  # Shape (N, 30)
    dt = -uu * dv_itl[1:] + vv * du_itl[1:]  # Shape (N, 30)

    # Find the first crossing point for each UV pair
    mask = dt <= 0
    i = np.where(np.any(mask, axis=1), np.argmax(mask, axis=1) + 1, 30)

    # Interpolation factor
    idx = np.arange(len(i))
    dt_current = -np.minimum(dt[idx, i - 1], 0.0)
    dt_previous = dt[idx, i - 2]
    f = np.where(
        i == 1, 0.0, np.where(i > 1, dt_current / (dt_previous + dt_current), 0.0)
    )

    # Interpolate temperature
    T = mired_to_CCT(r_itl[i - 1] * f + r_itl[i] * (1 - f))

    # Interpolate uv position
    u_i = u_itl[i - 1] * f + u_itl[i] * (1 - f)
    v_i = v_itl[i - 1] * f + v_itl[i] * (1 - f)

    # Interpolate direction vectors
    du_i = du_itl[i] * (1 - f) + du_itl[i - 1] * f
    dv_i = dv_itl[i] * (1 - f) + dv_itl[i - 1] * f

    # Normalize interpolated direction
    length_i = np.hypot(du_i, dv_i)
    du_i /= length_i
    dv_i /= length_i

    # Calculate D_uv
    uu = u.ravel() - u_i
    vv = v.ravel() - v_i
    D_uv = uu * du_i + vv * dv_i

    result = np.stack([T, -D_uv], axis=-1)

    return result.reshape(shape)


def CCT_to_uv_Robertson1968(CCT_D_uv: ArrayLike) -> NDArrayFloat:
    """
    Return the *CIE UCS* colourspace *uv* chromaticity coordinates from the
    specified correlated colour temperature :math:`T_{cp}` and
    :math:`\\Delta_{uv}` using *Robertson (1968)* method.

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
    :cite:`Wyszecki2000y`

    Examples
    --------
    >>> CCT_D_uv = np.array([6500.0081378199056, 0.008333331244225])
    >>> CCT_to_uv_Robertson1968(CCT_D_uv)  # doctest: +ELLIPSIS
    array([ 0.1937413...,  0.3152210...])
    """

    CCT_D_uv = as_float_array(CCT_D_uv)
    shape = CCT_D_uv.shape
    CCT_D_uv = CCT_D_uv.reshape(-1, 2)

    r_itl, u_itl, v_itl, t_itl = tsplit(
        np.array(DATA_ISOTEMPERATURE_LINES_ROBERTSON1968)
    )

    # Precompute normalized direction vectors
    length = np.hypot(1.0, t_itl)
    du_itl = 1.0 / length
    dv_itl = t_itl / length

    # Vectorized computation for all CCT/D_uv pairs at once
    CCT, D_uv = tsplit(CCT_D_uv)
    r = CCT_to_mired(CCT)

    # Find the isotemperature range containing r for all values
    mask = r[:, np.newaxis] < r_itl[1:]
    i = np.where(np.any(mask, axis=1), np.argmax(mask, axis=1), 29)

    # Interpolation factor
    f = (r_itl[i + 1] - r) / (r_itl[i + 1] - r_itl[i])

    # Interpolate uv position on Planckian locus
    u = u_itl[i] * f + u_itl[i + 1] * (1 - f)
    v = v_itl[i] * f + v_itl[i + 1] * (1 - f)

    # Interpolate direction vectors
    du_i = du_itl[i] * f + du_itl[i + 1] * (1 - f)
    dv_i = dv_itl[i] * f + dv_itl[i + 1] * (1 - f)

    # Normalize interpolated direction
    length_i = np.hypot(du_i, dv_i)
    du_i /= length_i
    dv_i /= length_i

    # Offset by D_uv along the isotherm
    u += du_i * -D_uv
    v += dv_i * -D_uv

    result = np.stack([u, v], axis=-1)

    return result.reshape(shape)
