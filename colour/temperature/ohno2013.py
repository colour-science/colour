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
from colour.algebra.common import euclidean_distance

from colour.colorimetry import (
    MultiSpectralDistributions,
    handle_spectral_arguments,
)
from colour.hints import ArrayLike, NDArrayFloat
from colour.models.cie_ucs import UCS_to_XYZ, UCS_to_uv, XYZ_to_UCS, uv_to_UCS
from colour.temperature import CCT_to_uv_Planck1900
from colour.utilities import (
    CACHE_REGISTRY,
    as_float_array,
    optional,
    runtime_warning,
    tsplit,
    tstack,
)
from colour.utilities.common import attest

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "CCT_MINIMAL_OHNO2013",
    "CCT_MAXIMAL_OHNO2013",
    "CCT_DEFAULT_SPACING_OHNO2013",
    "planckian_table",
    "uv_to_CCT_Ohno2013",
    "CCT_to_uv_Ohno2013",
    "XYZ_to_CCT_Ohno2013",
    "CCT_to_XYZ_Ohno2013",
]

CCT_MINIMAL_OHNO2013: float = 1000
CCT_MAXIMAL_OHNO2013: float = 100000
CCT_DEFAULT_SPACING_OHNO2013: float = 1.001

_CACHE_PLANCKIAN_TABLE: dict = CACHE_REGISTRY.register_cache(
    f"{__name__}._CACHE_PLANCKIAN_TABLE"
)


def planckian_table(
    cmfs: MultiSpectralDistributions,
    start: float,
    end: float,
    spacing: float,
) -> NDArrayFloat:
    """
    Return a planckian table from given *CIE UCS* colourspace *uv*
    chromaticity coordinates, colour matching functions and temperature range
    using *Ohno (2013)* method.

    Parameters
    ----------
    cmfs
        Standard observer colour matching functions.
    start
        Temperature range start in kelvin degrees.
    end
        Temperature range end in kelvin degrees.
    spacing
        The spacing between values expressed as a multiplier. Must be greater
        than 1.0

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
    >>> planckian_table(cmfs, 1000, 1010, 1.005)
    ... # doctest: +ELLIPSIS
    array([[  1.00000000e+03,   4.4796...e-01,   3.5462...e-01],
           [  1.00100000e+03,   4.4772...e-01,   3.5464...e-01],
           [  1.00600500e+03,   4.4656...e-01,   3.5475...e-01],
           [  1.00900000e+03,   4.4586...e-01,   3.5481...e-01],
           [  1.01000000e+03,   4.4563...e-01,   3.5483...e-01]])
    """

    cache_key = hash((cmfs, start, end, spacing))
    if cache_key in _CACHE_PLANCKIAN_TABLE:
        table = _CACHE_PLANCKIAN_TABLE[cache_key].copy()
    else:
        attest(spacing > 1.0, "spacing value must be > 1")

        Ti = [start, start + 1]
        next_ti = start + 1
        next_spacing = spacing
        while (next_ti := next_ti * next_spacing) < end:
            Ti.append(next_ti)

            # Slightly decrease stepsize for higher CCT
            D = (next_ti - 1000) / (100_000 - 1000)
            D = min(max(D, 0), 1)
            next_spacing = spacing * (1 - D) + (1 + (spacing - 1) / 10) * D
        Ti = np.concatenate([Ti, [end - 1, end]])

        table = np.concatenate(
            [Ti.reshape((-1, 1)), CCT_to_uv_Planck1900(Ti, cmfs)], axis=1
        )
        _CACHE_PLANCKIAN_TABLE[cache_key] = table.copy()
    return table


def XYZ_to_CCT_Ohno2013(
    XYZ: ArrayLike, cmfs: MultiSpectralDistributions | None = None
):
    """Calculate the CCT of a given XYZ value using the Ohno (2014) CCT
    approximation method. Frequently used in lighting quality calculations

    Parameters
    ----------
    XYZ
        *XYZ* colourspace *uv* chromaticity coordinates.
    cmfs
        Standard observer colour matching functions, default to the
        *CIE 1931 2 Degree Standard Observer*.

    Returns
    -------
    :class:`numpy.ndarray`
        Correlated colour temperature :math:`T_{cp}`, :math:`\\Delta_{uv}`.

    References
    ----------
    :cite:`Ohno2014a`
    """
    return uv_to_CCT_Ohno2013(UCS_to_uv(XYZ_to_UCS(XYZ)), cmfs)


def CCT_to_XYZ_Ohno2013(
    CCT_D_uv: ArrayLike, cmfs: MultiSpectralDistributions | None = None
):
    """Calculate an XYZ for a given CCT_duv. Provided as a convienience function
    for frequent lighting calculations.

    Parameters
    ----------
    CCT_D_uv : ArrayLike
        Target CCT and d_uv values to convert to XYZ
    cmfs
        Standard observer colour matching functions, default to the
        *CIE 1931 2 Degree Standard Observer*.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE UCS* colourspace *uv* chromaticity coordinates.

    """
    return UCS_to_XYZ(uv_to_UCS(CCT_to_uv_Ohno2013(CCT_D_uv, cmfs)))


def uv_to_CCT_Ohno2013(
    uv: ArrayLike,
    cmfs: MultiSpectralDistributions | None = None,
    start: float | None = None,
    end: float | None = None,
    spacing: float | None = None,
) -> NDArrayFloat:
    """
    Return the correlated colour temperature :math:`T_{cp}` and
    :math:`\\Delta_{uv}` from given *CIE UCS* colourspace *uv* chromaticity
    coordinates, colour matching functions and temperature range using
    *Ohno (2013)* method.

    The ``spacing`` parameter defines the calculations' precision: The
    closer to 1.0, the higher the precission of the calculation. The spacing
    value must be greater than 1.0

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
    spacing
        Spacing used for CCT initial LUT. Default = 1.005. 1.01 gives a good
        balance of performance and accuracy.

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
    spacing = optional(spacing, CCT_DEFAULT_SPACING_OHNO2013)

    shape = uv.shape
    uv = np.reshape(uv, (-1, 2))

    # Planckian tables creation through cascade expansion.
    tables_data = []
    for uv_i in uv:
        table = planckian_table(cmfs, start, end, spacing)
        dists = euclidean_distance(table[:, 1:], uv_i)
        index = np.argmin(dists)
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

        tables_data.append(
            np.vstack(
                [
                    [*table[index - 1, ...], dists[index - 1]],
                    [*table[index, ...], dists[index]],
                    [*table[index + 1, ...], dists[index + 1]],
                ]
            )
        )
    tables = as_float_array(tables_data)

    Tip, uip, vip, dip = tsplit(tables[:, 0, :])
    Ti, _ui, _vi, di = tsplit(tables[:, 1, :])
    Tin, uin, vin, din = tsplit(tables[:, 2, :])

    # Triangular solution.
    l = np.hypot(uin - uip, vin - vip)  # noqa: E741
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
    CCT_D_uv: ArrayLike, cmfs: MultiSpectralDistributions | None = None
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
