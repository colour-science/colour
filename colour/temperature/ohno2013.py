"""
Ohno (2013) Correlated Colour Temperature
=========================================

Defines the *Ohno (2013)* correlated colour temperature :math:`T_{cp}`
computations objects:

-   :func:`colour.temperature.uv_to_CCT_Ohno2013`: Correlated colour
    temperature :math:`T_{cp}` and :math:`\\Delta_{uv}` computation of given
    *CIE UCS* colourspace *uv* chromaticity coordinates using *Ohno (2013)*
    method.
-   :func:`colour.temperature.CCT_to_uv_Ohno2013`: *CIE UCS* colourspace *uv*
    chromaticity coordinates computation of given correlated colour temperature
    :math:`T_{cp}`, :math:`\\Delta_{uv}` using *Ohno (2013)* method.

References
----------
-   :cite:`Ohno2014a` : Ohno, Yoshiro. (2014). Practical Use and Calculation of
    CCT and Duv. LEUKOS, 10(1), 47-55. doi:10.1080/15502724.2014.839020
"""

from __future__ import annotations

import numpy as np

from colour.algebra import sdiv, sdiv_mode
from colour.colorimetry import (
    MultiSpectralDistributions,
    handle_spectral_arguments,
)
from colour.hints import ArrayLike, NDArrayFloat, Optional
from colour.temperature import CCT_to_uv_Planck1900
from colour.utilities import (
    CACHE_REGISTRY,
    as_float_array,
    optional,
    runtime_warning,
    tsplit,
    tstack,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "CCT_MINIMAL_OHNO2013",
    "CCT_MAXIMAL_OHNO2013",
    "CCT_SAMPLES_OHNO2013",
    "CCT_ITERATIONS_OHNO2013",
    "planckian_table",
    "uv_to_CCT_Ohno2013",
    "CCT_to_uv_Ohno2013",
]

CCT_MINIMAL_OHNO2013: float = 1000
CCT_MAXIMAL_OHNO2013: float = 100000
CCT_SAMPLES_OHNO2013: int = 10
CCT_ITERATIONS_OHNO2013: int = 6

_CACHE_PLANCKIAN_TABLE_ROW: dict = CACHE_REGISTRY.register_cache(
    f"{__name__}._CACHE_PLANCKIAN_TABLE_ROW"
)


def planckian_table(
    uv: ArrayLike,
    cmfs: MultiSpectralDistributions,
    start: float,
    end: float,
    count: int,
) -> NDArrayFloat:
    """
    Return a planckian table from given *CIE UCS* colourspace *uv*
    chromaticity coordinates, colour matching functions and temperature range
    using *Ohno (2013)* method.

    Parameters
    ----------
    uv
        *uv* chromaticity coordinates.
    cmfs
        Standard observer colour matching functions.
    start
        Temperature range start in kelvin degrees.
    end
        Temperature range end in kelvin degrees.
    count
        Temperatures count in the planckian table.

    Returns
    -------
    :class:`list`
        Planckian table.

    Examples
    --------
    >>> from colour import MSDS_CMFS, SPECTRAL_SHAPE_DEFAULT
    >>> cmfs = (
    ...     MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    ...     .copy()
    ...     .align(SPECTRAL_SHAPE_DEFAULT)
    ... )
    >>> uv = np.array([0.1978, 0.3122])
    >>> planckian_table(uv, cmfs, 1000, 1010, 10)
    ... # doctest: +ELLIPSIS
    array([[  1.0000000...e+03,   4.4796288...e-01,   3.5462962...e-01,
              2.5373557...e-01],
           [  1.0011111...e+03,   4.4770302...e-01,   3.5465214...e-01,
              2.5348315...e-01],
           [  1.0022222...e+03,   4.4744347...e-01,   3.5467461...e-01,
              2.5323103...e-01],
           [  1.0033333...e+03,   4.4718423...e-01,   3.5469703...e-01,
              2.5297923...e-01],
           [  1.0044444...e+03,   4.4692529...e-01,   3.5471941...e-01,
              2.5272774...e-01],
           [  1.0055555...e+03,   4.4666665...e-01,   3.5474175...e-01,
              2.5247656...e-01],
           [  1.0066666...e+03,   4.4640832...e-01,   3.5476403...e-01,
              2.5222568...e-01],
           [  1.0077777...e+03,   4.4615029...e-01,   3.5478628...e-01,
              2.5197512...e-01],
           [  1.0088888...e+03,   4.4589257...e-01,   3.5480848...e-01,
              2.5172486...e-01],
           [  1.0100000...e+03,   4.4563515...e-01,   3.5483063...e-01,
              2.5147492...e-01]])
    """

    ux, vx = tsplit(uv)

    table_data = []
    for Ti in np.linspace(start, end, count):
        cache_key = Ti
        if cache_key in _CACHE_PLANCKIAN_TABLE_ROW:
            row = _CACHE_PLANCKIAN_TABLE_ROW[cache_key]
        else:
            u_i, v_i = tsplit(CCT_to_uv_Planck1900(Ti, cmfs))
            _CACHE_PLANCKIAN_TABLE_ROW[cache_key] = row = [Ti, u_i, v_i, -1]
        table_data.append(row)

    table = as_float_array(table_data)
    table[..., -1] = np.hypot(ux - table[..., 1], vx - table[..., 2])

    return table


def uv_to_CCT_Ohno2013(
    uv: ArrayLike,
    cmfs: Optional[MultiSpectralDistributions] = None,
    start: Optional[float] = None,
    end: Optional[float] = None,
    count: Optional[int] = None,
    iterations: Optional[int] = None,
) -> NDArrayFloat:
    """
    Return the correlated colour temperature :math:`T_{cp}` and
    :math:`\\Delta_{uv}` from given *CIE UCS* colourspace *uv* chromaticity
    coordinates, colour matching functions and temperature range using
    *Ohno (2013)* method.

    The ``iterations`` parameter defines the calculations' precision: The
    higher its value, the more planckian tables will be generated through
    cascade expansion in order to converge to the exact solution.

    Parameters
    ----------
    uv
        *CIE UCS* colourspace *uv* chromaticity coordinates.
    cmfs
        Standard observer colour matching functions, default to the
        *CIE 1931 2 Degree Standard Observer*.
    start
        Temperature range start in kelvin degrees, default to 1000.
    end
        Temperature range end in kelvin degrees, default to 100000.
    count
        Temperatures count/samples in the planckian tables, default to 10.
    iterations
        Number of planckian tables to generate, default to 6.

    Returns
    -------
    :class:`numpy.ndarray`
        Correlated colour temperature :math:`T_{cp}`, :math:`\\Delta_{uv}`.

    References
    ----------
    :cite:`Ohno2014a`

    Examples
    --------
    >>> from pprint import pprint
    >>> from colour import MSDS_CMFS, SPECTRAL_SHAPE_DEFAULT
    >>> cmfs = (
    ...     MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    ...     .copy()
    ...     .align(SPECTRAL_SHAPE_DEFAULT)
    ... )
    >>> uv = np.array([0.1978, 0.3122])
    >>> uv_to_CCT_Ohno2013(uv, cmfs)  # doctest: +ELLIPSIS
    array([  6.50747...e+03,   3.22334...e-03])
    """

    uv = as_float_array(uv)
    cmfs, _illuminant = handle_spectral_arguments(cmfs)
    start = optional(start, CCT_MINIMAL_OHNO2013)
    end = optional(end, CCT_MAXIMAL_OHNO2013)
    count = optional(count, CCT_SAMPLES_OHNO2013)
    iterations = optional(iterations, CCT_ITERATIONS_OHNO2013)

    shape = uv.shape
    uv = np.reshape(uv, (-1, 2))

    # Planckian tables creation through cascade expansion.
    tables_data = []
    for uv_i in uv:
        start_i, end_r = start, end
        for _i in range(max(int(iterations), 1)):
            table = planckian_table(uv_i, cmfs, start_i, end_r, count)
            index = np.argmin(table[..., -1])
            if index == 0:
                runtime_warning(
                    "Minimal distance index is on lowest planckian table bound, "
                    "unpredictable results may occur!"
                )
                index += 1
            elif index == len(table) - 1:
                runtime_warning(
                    "Minimal distance index is on highest planckian table bound, "
                    "unpredictable results may occur!"
                )
                index -= 1

            start_i = table[index - 1][0]
            end_r = table[index + 1][0]

        tables_data.append(
            np.vstack(
                [
                    table[index - 1, ...],
                    table[index, ...],
                    table[index + 1, ...],
                ]
            )
        )
    tables = as_float_array(tables_data)

    Tip, uip, vip, dip = tsplit(tables[:, 0, :])
    Ti, _ui, _vi, di = tsplit(tables[:, 1, :])
    Tin, uin, vin, din = tsplit(tables[:, 2, :])

    # Triangular solution.
    l = np.hypot(uin - uip, vin - vip)  # noqa
    x = (dip**2 - din**2 + l**2) / (2 * l)
    T_t = Tip + (Tin - Tip) * (x / l)

    vtx = vip + (vin - vip) * (x / l)
    sign = np.sign(uv[..., 1] - vtx)
    D_uv_t = (dip**2 - x**2) ** (1 / 2) * sign

    # Parabolic solution.
    X = (Tin - Ti) * (Tip - Tin) * (Ti - Tip)
    a = (Tip * (din - di) + Ti * (dip - din) + Tin * (di - dip)) * X**-1
    b = (
        -(
            Tip**2 * (din - di)
            + Ti**2 * (dip - din)
            + Tin**2 * (di - dip)
        )
        * X**-1
    )
    c = (
        -(
            dip * (Tin - Ti) * Ti * Tin
            + di * (Tip - Tin) * Tip * Tin
            + din * (Ti - Tip) * Tip * Ti
        )
        * X**-1
    )

    T_p = -b / (2 * a)
    D_uv_p = (a * T_p**2 + b * T_p + c) * sign

    CCT_D_uv = np.where(
        (np.abs(D_uv_t) >= 0.002)[..., None],
        tstack([T_p, D_uv_p]),
        tstack([T_t, D_uv_t]),
    )

    return np.reshape(CCT_D_uv, shape)


def CCT_to_uv_Ohno2013(
    CCT_D_uv: ArrayLike, cmfs: Optional[MultiSpectralDistributions] = None
) -> NDArrayFloat:
    """
    Return the *CIE UCS* colourspace *uv* chromaticity coordinates from given
    correlated colour temperature :math:`T_{cp}`, :math:`\\Delta_{uv}` and
    colour matching functions using *Ohno (2013)* method.

    Parameters
    ----------
    CCT_D_uv
        Correlated colour temperature :math:`T_{cp}`, :math:`\\Delta_{uv}`.
    cmfs
        Standard observer colour matching functions, default to the
        *CIE 1931 2 Degree Standard Observer*.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE UCS* colourspace *uv* chromaticity coordinates.

    References
    ----------
    :cite:`Ohno2014a`

    Examples
    --------
    >>> from pprint import pprint
    >>> from colour import MSDS_CMFS, SPECTRAL_SHAPE_DEFAULT
    >>> cmfs = (
    ...     MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    ...     .copy()
    ...     .align(SPECTRAL_SHAPE_DEFAULT)
    ... )
    >>> CCT_D_uv = np.array([6507.4342201047066, 0.003223690901513])
    >>> CCT_to_uv_Ohno2013(CCT_D_uv, cmfs)  # doctest: +ELLIPSIS
    array([ 0.1977999...,  0.3122004...])
    """

    CCT, D_uv = tsplit(CCT_D_uv)

    cmfs, _illuminant = handle_spectral_arguments(cmfs)

    uv_0 = CCT_to_uv_Planck1900(CCT, cmfs)
    uv_1 = CCT_to_uv_Planck1900(CCT + 0.01, cmfs)

    du, dv = tsplit(uv_0 - uv_1)

    h = np.hypot(du, dv)

    with sdiv_mode():
        uv = tstack(
            [
                uv_0[..., 0] - D_uv * sdiv(dv, h),
                uv_0[..., 1] + D_uv * sdiv(du, h),
            ]
        )

    uv[D_uv == 0] = uv_0[D_uv == 0]

    return uv
