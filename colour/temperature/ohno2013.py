"""
Ohno (2013) Correlated Colour Temperature
=========================================

Define the *Ohno (2013)* correlated colour temperature :math:`T_{cp}`
computation objects.

-   :func:`colour.temperature.uv_to_CCT_Ohno2013`: Compute correlated colour
    temperature :math:`T_{cp}` and :math:`\\Delta_{uv}` from specified
    *CIE UCS* colourspace *uv* chromaticity coordinates using the
    *Ohno (2013)* method.
-   :func:`colour.temperature.CCT_to_uv_Ohno2013`: Compute *CIE UCS*
    colourspace *uv* chromaticity coordinates from specified correlated
    colour temperature :math:`T_{cp}` and :math:`\\Delta_{uv}` using the
    *Ohno (2013)* method.

References
----------
-   :cite:`Ohno2014a` : Ohno, Yoshiro. (2014). Practical Use and Calculation of
    CCT and Duv. LEUKOS, 10(1), 47-55. doi:10.1080/15502724.2014.839020
"""

from __future__ import annotations

from colour.algebra import sdiv, sdiv_mode
from colour.colorimetry import MultiSpectralDistributions, handle_spectral_arguments
from colour.hints import (  # noqa: TC001
    ArrayLike,
    Domain1,
    NDArrayFloat,
    Range1,
)
from colour.models import UCS_to_uv, UCS_to_XYZ, XYZ_to_UCS, uv_to_UCS
from colour.temperature import CCT_to_uv_Planck1900
from colour.utilities import (
    CACHE_REGISTRY,
    array_namespace,
    as_float_array,
    attest,
    is_caching_enabled,
    optional,
    runtime_warning,
    tsplit,
    tstack,
    xp_as_float_array,
    xp_reshape,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
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

# Finite-difference step :math:`\\Delta T = 0.01\\ K` used by
# :func:`CCT_to_XYZ_Ohno2013` to evaluate the *Planckian* derivative
# numerically. The value is small enough to remain accurate at
# ``CCT_MINIMAL_OHNO2013`` (relative error ~1e-5) and large enough not
# to be lost to float32 round-off at ``CCT_MAXIMAL_OHNO2013``.
_FINITE_DIFFERENCE_STEP_OHNO2013: float = 0.01

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
    Generate a planckian table from the specified *CIE UCS* colourspace
    *uv* chromaticity coordinates, colour matching functions, and
    temperature range using the *Ohno (2013)* method.

    Parameters
    ----------
    cmfs
        Standard observer colour matching functions.
    start
        Temperature range start in kelvin degrees.
    end
        Temperature range end in kelvin degrees.
    spacing
        Spacing between values of the underlying Planckian table expressed
        as a multiplier. Default to 1.001. The closer to 1.0, the higher
        the precision of the returned colour temperature :math:`T_{cp}` and
        :math:`\\Delta_{uv}`. A value of 1.01 provides a good balance
        between performance and accuracy. The ``spacing`` value must be
        greater than 1.

    Returns
    -------
    :class:`list`
        Planckian table.

    Examples
    --------
    >>> import numpy as np
    >>> from colour import MSDS_CMFS, SPECTRAL_SHAPE_DEFAULT
    >>> cmfs = (
    ...     MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    ...     .copy()
    ...     .align(SPECTRAL_SHAPE_DEFAULT)
    ... )
    >>> uv = np.array([0.1978, 0.3122])
    >>> planckian_table(cmfs, 1000, 1010, 1.005)
    ... # doctest: +ELLIPSIS
    array([[1.00000000e+03, 4.4796288...e-01, 3.5462962...e-01],
           [1.00100000e+03, 4.4772900...e-01, 3.5464989...e-01],
           [1.00600500e+03, 4.4656212...e-01, 3.5475077...e-01],
           [1.00900000e+03, 4.4586682...e-01, 3.5481069...e-01],
           [1.01000000e+03, 4.4563515...e-01, 3.5483063...e-01]])
    """

    hash_key = hash((cmfs, start, end, spacing))
    if is_caching_enabled() and hash_key in _CACHE_PLANCKIAN_TABLE:
        table = _CACHE_PLANCKIAN_TABLE[hash_key].copy()
    else:
        attest(spacing > 1, "Spacing value must be greater than 1!")

        Ti = [start, start + 1]
        next_ti = start + 1
        next_spacing = spacing
        while (next_ti := next_ti * next_spacing) < end:
            Ti.append(next_ti)

            # Slightly decrease step-size for higher CCT.
            D = (next_ti - CCT_MINIMAL_OHNO2013) / (
                CCT_MAXIMAL_OHNO2013 - CCT_MINIMAL_OHNO2013
            )
            D = min(max(D, 0), 1)
            next_spacing = spacing * (1 - D) + (1 + (spacing - 1) / 10) * D
        Ti = as_float_array(Ti)

        xp = array_namespace(Ti)

        Ti = xp.concat([Ti, xp_as_float_array([end - 1, end], xp=xp)])

        table = xp.concat(
            [xp_reshape(Ti, (-1, 1), xp=xp), CCT_to_uv_Planck1900(Ti, cmfs)], axis=1
        )
        if is_caching_enabled():
            _CACHE_PLANCKIAN_TABLE[hash_key] = table.copy()

    return table


def uv_to_CCT_Ohno2013(
    uv: ArrayLike,
    cmfs: MultiSpectralDistributions | None = None,
    start: float | None = None,
    end: float | None = None,
    spacing: float | None = None,
) -> NDArrayFloat:
    """
    Compute the correlated colour temperature :math:`T_{cp}` and
    :math:`\\Delta_{uv}` from the specified *CIE UCS* colourspace *uv*
    chromaticity coordinates using the *Ohno (2013)* method.

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
        Spacing between values of the underlying Planckian table expressed
        as a multiplier. Default to 1.001. The closer to 1.0, the higher
        the precision of the returned colour temperature :math:`T_{cp}` and
        :math:`\\Delta_{uv}`. A value of 1.01 provides a good balance
        between performance and accuracy. The ``spacing`` value must be
        greater than 1.

    Returns
    -------
    :class:`numpy.ndarray`
        Correlated colour temperature :math:`T_{cp}`, :math:`\\Delta_{uv}`.

    References
    ----------
    :cite:`Ohno2014a`

    Examples
    --------
    >>> import numpy as np
    >>> from colour import MSDS_CMFS, SPECTRAL_SHAPE_DEFAULT
    >>> cmfs = (
    ...     MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    ...     .copy()
    ...     .align(SPECTRAL_SHAPE_DEFAULT)
    ... )
    >>> uv = np.array([0.1978, 0.3122])
    >>> uv_to_CCT_Ohno2013(uv, cmfs)  # doctest: +ELLIPSIS
    array([6.5074747...e+03, 3.2233463...e-03])
    """

    uv = as_float_array(uv)

    xp = array_namespace(uv)

    cmfs, _illuminant = handle_spectral_arguments(cmfs)
    start = optional(start, CCT_MINIMAL_OHNO2013)
    end = optional(end, CCT_MAXIMAL_OHNO2013)
    spacing = optional(spacing, CCT_DEFAULT_SPACING_OHNO2013)

    shape = uv.shape
    uv = xp_reshape(uv, (-1, 2), xp=xp)

    # The Planckian table depends only on ``cmfs`` / ``start`` / ``end``
    # / ``spacing`` so it is computed once and reused; the per-sample
    # nearest-row lookup is vectorised below.
    table = xp_as_float_array(
        planckian_table(cmfs, start, end, spacing), xp=xp, like=uv
    )
    n_rows = table.shape[0]

    distances = xp.linalg.vector_norm(table[None, :, 1:] - uv[:, None, :], axis=-1)
    indices = xp.argmin(distances, axis=-1)

    # Emit a single warning per boundary irrespective of sample count,
    # then clip so the neighbour gather below stays in range.
    if bool(xp.any(indices == 0)):
        runtime_warning(
            "Minimal distance index is on lowest planckian table bound, "
            "unpredictable results may occur!"
        )
    if bool(xp.any(indices == n_rows - 1)):
        runtime_warning(
            "Minimal distance index is on highest planckian table bound, "
            "unpredictable results may occur!"
        )
    indices = xp.clip(indices, 1, n_rows - 2)

    # ``(N, 3, 4)`` neighbour triple expected by the *Ohno (2014)*
    # triangular and parabolic solutions below.
    sample_indices = xp.arange(uv.shape[0])
    tables = xp.stack(
        [
            xp.concat(
                [
                    table[indices - 1],
                    distances[sample_indices, indices - 1][..., None],
                ],
                axis=-1,
            ),
            xp.concat(
                [table[indices], distances[sample_indices, indices][..., None]],
                axis=-1,
            ),
            xp.concat(
                [
                    table[indices + 1],
                    distances[sample_indices, indices + 1][..., None],
                ],
                axis=-1,
            ),
        ],
        axis=1,
    )

    Tip, uip, vip, dip = tsplit(tables[:, 0, :])
    Ti, _ui, _vi, di = tsplit(tables[:, 1, :])
    Tin, uin, vin, din = tsplit(tables[:, 2, :])

    # *Ohno (2014)* Eqs. (23)-(25), triangular solution.
    l = xp.hypot(uin - uip, vin - vip)  # noqa: E741
    x = (dip**2 - din**2 + l**2) / (2 * l)
    T_t = Tip + (Tin - Tip) * (x / l)

    vtx = vip + (vin - vip) * (x / l)
    sign = xp.sign(uv[..., 1] - vtx)
    radicand = dip**2 - x**2
    D_uv_t = xp.sqrt(xp.where(radicand > 0, radicand, 0.0)) * sign

    # *Ohno (2014)* Eqs. (26)-(28), parabolic solution.
    X = (Tin - Ti) * (Tip - Tin) * (Ti - Tip)
    a = (Tip * (din - di) + Ti * (dip - din) + Tin * (di - dip)) * X**-1
    b = -(Tip**2 * (din - di) + Ti**2 * (dip - din) + Tin**2 * (di - dip)) * X**-1
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

    CCT_D_uv = xp.where(
        (xp.abs(D_uv_t) >= 0.002)[..., None],
        tstack([T_p, D_uv_p]),
        tstack([T_t, D_uv_t]),
    )

    return xp_reshape(CCT_D_uv, shape, xp=xp)


def CCT_to_uv_Ohno2013(
    CCT_D_uv: ArrayLike, cmfs: MultiSpectralDistributions | None = None
) -> NDArrayFloat:
    """
    Compute the *CIE UCS* colourspace *uv* chromaticity coordinates from
    the specified correlated colour temperature :math:`T_{cp}`,
    :math:`\\Delta_{uv}` and colour matching functions using
    *Ohno (2013)* method.

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
    >>> import numpy as np
    >>> from colour import MSDS_CMFS, SPECTRAL_SHAPE_DEFAULT
    >>> cmfs = (
    ...     MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    ...     .copy()
    ...     .align(SPECTRAL_SHAPE_DEFAULT)
    ... )
    >>> CCT_D_uv = np.array([6507.4342201047066, 0.003223690901513])
    >>> CCT_to_uv_Ohno2013(CCT_D_uv, cmfs)  # doctest: +ELLIPSIS
    array([0.1977999..., 0.3122004...])
    """

    xp = array_namespace(CCT_D_uv)

    CCT, D_uv = tsplit(CCT_D_uv)

    cmfs, _illuminant = handle_spectral_arguments(cmfs)

    uv_0 = CCT_to_uv_Planck1900(CCT, cmfs)
    uv_1 = CCT_to_uv_Planck1900(CCT + _FINITE_DIFFERENCE_STEP_OHNO2013, cmfs)

    du, dv = tsplit(uv_0 - uv_1)

    h = xp.hypot(du, dv)

    with sdiv_mode():
        uv = tstack(
            [
                uv_0[..., 0] - D_uv * sdiv(dv, h),
                uv_0[..., 1] + D_uv * sdiv(du, h),
            ]
        )

    return xp.where((D_uv == 0)[..., None], uv_0, uv)


def XYZ_to_CCT_Ohno2013(
    XYZ: Domain1,
    cmfs: MultiSpectralDistributions | None = None,
    start: float | None = None,
    end: float | None = None,
    spacing: float | None = None,
) -> NDArrayFloat:
    """
    Compute the correlated colour temperature :math:`T_{cp}` and
    :math:`\\Delta_{uv}` from the specified *CIE XYZ* tristimulus values
    using the *Ohno (2013)* method.

    The method computes the correlated colour temperature by finding the
    closest point on the Planckian locus to the specified chromaticity
    coordinates using an optimised search algorithm with configurable
    precision through the spacing parameter.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values.
    cmfs
        Standard observer colour matching functions, default to the
        *CIE 1931 2 Degree Standard Observer*.
    start
        Temperature range start in kelvins, default to 1000.
    end
        Temperature range end in kelvins, default to 100000.
    spacing
        Spacing between values of the underlying Planckian table expressed
        as a multiplier. Default to 1.001. The closer to 1.0, the higher
        the precision of the returned colour temperature :math:`T_{cp}` and
        :math:`\\Delta_{uv}`. A value of 1.01 provides a good balance
        between performance and accuracy. The ``spacing`` value must be
        greater than 1.

    Returns
    -------
    :class:`numpy.ndarray`
        Correlated colour temperature :math:`T_{cp}`, :math:`\\Delta_{uv}`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | 1                     | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Ohno2014a`

    Examples
    --------
    >>> import numpy as np
    >>> from colour import MSDS_CMFS, SPECTRAL_SHAPE_DEFAULT
    >>> cmfs = (
    ...     MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    ...     .copy()
    ...     .align(SPECTRAL_SHAPE_DEFAULT)
    ... )
    >>> XYZ = np.array([0.95035049, 1.0, 1.08935705])
    >>> XYZ_to_CCT_Ohno2013(XYZ, cmfs)  # doctest: +ELLIPSIS
    array([6.5074399...e+03, 3.2236914...e-03])
    """

    return uv_to_CCT_Ohno2013(UCS_to_uv(XYZ_to_UCS(XYZ)), cmfs, start, end, spacing)


def CCT_to_XYZ_Ohno2013(
    CCT_D_uv: ArrayLike, cmfs: MultiSpectralDistributions | None = None
) -> Range1:
    """
    Compute the *CIE XYZ* tristimulus values from the specified correlated
    colour temperature :math:`T_{cp}` and :math:`\\Delta_{uv}` using the
    *Ohno (2013)* method.

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
        *CIE XYZ* tristimulus values.

    Notes
    -----
    +-----------+-----------------------+---------------+
    | **Range** | **Scale - Reference** | **Scale - 1** |
    +===========+=======================+===============+
    | ``XYZ``   | 1                     | 1             |
    +-----------+-----------------------+---------------+

    Examples
    --------
    >>> import numpy as np
    >>> from colour import MSDS_CMFS, SPECTRAL_SHAPE_DEFAULT
    >>> cmfs = (
    ...     MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    ...     .copy()
    ...     .align(SPECTRAL_SHAPE_DEFAULT)
    ... )
    >>> CCT_D_uv = np.array([6507.4342201047066, 0.003223690901513])
    >>> CCT_to_XYZ_Ohno2013(CCT_D_uv, cmfs)  # doctest: +ELLIPSIS
    array([0.9503504..., 1.        , 1.0893570...])
    """

    return UCS_to_XYZ(uv_to_UCS(CCT_to_uv_Ohno2013(CCT_D_uv, cmfs)))
