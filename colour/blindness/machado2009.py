"""
Simulation of CVD - Machado, Oliveira and Fernandes (2009)
==========================================================

Define the *Machado et al. (2009)* physiologically-based model for
simulation of colour vision deficiency.

-   :func:`colour.msds_cmfs_anomalous_trichromacy_Machado2009`
-   :func:`colour.matrix_anomalous_trichromacy_Machado2009`
-   :func:`colour.matrix_cvd_Machado2009`

References
----------
-   :cite:`Colblindora` : Colblindor. (n.d.). Deuteranopia - Red-Green Color
    Blindness. Retrieved July 4, 2015, from
    http://www.color-blindness.com/deuteranopia-red-green-color-blindness/
-   :cite:`Colblindorb` : Colblindor. (n.d.). Protanopia - Red-Green Color
    Blindness. Retrieved July 4, 2015, from
    http://www.color-blindness.com/protanopia-red-green-color-blindness/
-   :cite:`Colblindorc` : Colblindor. (n.d.). Tritanopia - Blue-Yellow Color
    Blindness. Retrieved July 4, 2015, from
    http://www.color-blindness.com/tritanopia-blue-yellow-color-blindness/
-   :cite:`Machado2009` : Machado, G.M., Oliveira, M. M., & Fernandes, L.
    (2009). A Physiologically-based Model for Simulation of Color Vision
    Deficiency. IEEE Transactions on Visualization and Computer Graphics,
    15(6), 1291-1298. doi:10.1109/TVCG.2009.113
"""

from __future__ import annotations

import typing

import numpy as np

from colour.algebra import vecmul
from colour.blindness import CVD_MATRICES_MACHADO2010

if typing.TYPE_CHECKING:
    from colour.characterisation import RGB_DisplayPrimaries

from colour.colorimetry import LMS_ConeFundamentals, SpectralShape, reshape_msds

if typing.TYPE_CHECKING:
    from colour.hints import ArrayLike, Literal, NDArrayFloat

from colour.utilities import (
    array_namespace,
    as_ndarray,
    is_non_ndarray,
    tsplit,
    tstack,
    usage_warning,
    xp_as_float_array,
    xp_trapezoid,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "MATRIX_LMS_TO_WSYBRG",
    "matrix_RGB_to_WSYBRG",
    "msds_cmfs_anomalous_trichromacy_Machado2009",
    "matrix_anomalous_trichromacy_Machado2009",
    "matrix_cvd_Machado2009",
]

MATRIX_LMS_TO_WSYBRG: NDArrayFloat = np.array(
    [
        [0.600, 0.400, 0.000],
        [0.240, 0.105, -0.700],
        [1.200, -1.600, 0.400],
    ]
)
"""
Ingling and Tsou (1977) matrix converting from cones responses to
opponent-colour space.
"""


def matrix_RGB_to_WSYBRG(
    cmfs: LMS_ConeFundamentals, primaries: RGB_DisplayPrimaries
) -> NDArrayFloat:
    """
    Compute the matrix for transforming from *RGB* colourspace to
    opponent-colour space using *Machado et al. (2009)* method.

    Parameters
    ----------
    cmfs
        *LMS* cone fundamentals colour matching functions.
    primaries
        *RGB* display primaries tri-spectral distributions.

    Returns
    -------
    :class:`numpy.ndarray`
        Matrix transforming from *RGB* colourspace to opponent-colour
        space.

    Examples
    --------
    >>> from colour.characterisation import MSDS_DISPLAY_PRIMARIES
    >>> from colour.colorimetry import MSDS_CMFS_LMS
    >>> cmfs = MSDS_CMFS_LMS["Stockman & Sharpe 2 Degree Cone Fundamentals"]
    >>> d_LMS = np.array([15, 0, 0])
    >>> primaries = MSDS_DISPLAY_PRIMARIES["Apple Studio Display"]
    >>> matrix_RGB_to_WSYBRG(cmfs, primaries)  # doctest: +ELLIPSIS
    array([[  0.2126535...,   0.6704626...,   0.1168838...],
           [  4.7095295...,  12.4862869..., -16.1958165...],
           [-11.1518474...,  15.2534789...,  -3.1016315...]])
    """

    wavelengths = cmfs.wavelengths
    primaries = reshape_msds(
        primaries,
        cmfs.shape,
        copy=False,
        extrapolator_kwargs={"method": "Constant", "left": 0, "right": 0},
    )

    cmfs_values, primaries_values = cmfs.values, primaries.values
    xp = array_namespace(cmfs_values, primaries_values)
    reference = cmfs_values if is_non_ndarray(cmfs_values) else primaries_values
    cmfs_values = xp_as_float_array(cmfs_values, xp=xp, like=reference)
    primaries_values = xp_as_float_array(primaries_values, xp=xp, like=cmfs_values)

    WS, YB, RG = tsplit(vecmul(MATRIX_LMS_TO_WSYBRG, cmfs_values))
    R, G, B = tsplit(primaries_values)

    WS_R = xp_trapezoid(R * WS, x=wavelengths, xp=xp)
    WS_G = xp_trapezoid(G * WS, x=wavelengths, xp=xp)
    WS_B = xp_trapezoid(B * WS, x=wavelengths, xp=xp)

    YB_R = xp_trapezoid(R * YB, x=wavelengths, xp=xp)
    YB_G = xp_trapezoid(G * YB, x=wavelengths, xp=xp)
    YB_B = xp_trapezoid(B * YB, x=wavelengths, xp=xp)

    RG_R = xp_trapezoid(R * RG, x=wavelengths, xp=xp)
    RG_G = xp_trapezoid(G * RG, x=wavelengths, xp=xp)
    RG_B = xp_trapezoid(B * RG, x=wavelengths, xp=xp)

    M_G = xp.stack(
        [
            xp.stack([WS_R, WS_G, WS_B]),
            xp.stack([YB_R, YB_G, YB_B]),
            xp.stack([RG_R, RG_G, RG_B]),
        ]
    )

    return M_G / xp.sum(M_G, axis=-1)[:, None]


def msds_cmfs_anomalous_trichromacy_Machado2009(
    cmfs: LMS_ConeFundamentals, d_LMS: ArrayLike
) -> LMS_ConeFundamentals:
    """
    Shift the specified *LMS* cone fundamentals colour matching functions
    with the specified :math:`\\Delta_{LMS}` shift amount in nanometers to
    simulate anomalous trichromacy using *Machado et al. (2009)* method.

    Parameters
    ----------
    cmfs
        *LMS* cone fundamentals colour matching functions.
    d_LMS
        :math:`\\Delta_{LMS}` wavelength shift amount in nanometers for each
        cone type.

    Notes
    -----
    -   Input *LMS* cone fundamentals colour matching functions interval is
        expected to be 1 nanometer, incompatible input will be interpolated
        at 1 nanometer interval.
    -   Input :math:`\\Delta_{LMS}` shift amount is in domain [0, 20].

    Returns
    -------
    :class:`colour.LMS_ConeFundamentals`
        Anomalous trichromacy *LMS* cone fundamentals colour matching
        functions.

    Warnings
    --------
    *Machado et al. (2009)* simulation of tritanomaly is based on the shift
    paradigm as an approximation to the actual phenomenon and restrain the
    model from trying to model tritanopia.
    The pre-generated matrices are using a shift value in domain [5, 59]
    contrary to the domain [0, 20] used for protanomaly and deuteranomaly
    simulation.

    References
    ----------
    :cite:`Colblindorb`, :cite:`Colblindora`, :cite:`Colblindorc`,
    :cite:`Machado2009`

    Examples
    --------
    >>> from colour.colorimetry import MSDS_CMFS_LMS
    >>> cmfs = MSDS_CMFS_LMS["Stockman & Sharpe 2 Degree Cone Fundamentals"]
    >>> cmfs[450]
    array([0.0498639, 0.0870524, 0.955393 ])
    >>> msds_cmfs_anomalous_trichromacy_Machado2009(cmfs, np.array([15, 0, 0]))[
    ...     450
    ... ]  # doctest: +ELLIPSIS
    array([0.0891288..., 0.0870524 , 0.955393  ])
    """

    cmfs = cmfs.copy()

    if cmfs.shape.interval != 1:
        cmfs.interpolate(SpectralShape(cmfs.shape.start, cmfs.shape.end, 1))

    cmfs.extrapolator_kwargs = {"method": "Constant", "left": 0, "right": 0}

    xp = array_namespace(cmfs.values)

    L, M, _S = tsplit(cmfs.values)
    # ``cmfs`` is a host-bound :class:`MultiSpectralDistributions` (its
    # ``wavelengths`` are *NumPy*); the shift vector is a per-call host
    # parameter, so it is coerced to *NumPy* here to keep the
    # ``cmfs.wavelengths - d_S`` arithmetic on a single namespace.
    d_L, d_M, d_S = tsplit(as_ndarray(d_LMS))

    if d_S != 0:
        usage_warning(
            '"Machado et al. (2009)" simulation of tritanomaly is based on '
            "the shift paradigm as an approximation to the actual phenomenon "
            "and restrain the model from trying to model tritanopia.\n"
            "The pre-generated matrices are using a shift value in domain "
            "[5, 59] contrary to the domain [0, 20] used for protanomaly and "
            "deuteranomaly simulation."
        )

    area_L = xp_trapezoid(L, x=cmfs.wavelengths, xp=xp)
    area_M = xp_trapezoid(M, x=cmfs.wavelengths, xp=xp)

    def alpha(x: NDArrayFloat) -> NDArrayFloat:
        """Compute :math:`alpha` factor."""

        return (20 - x) / 20

    # Corrected equations as per:
    # http://www.inf.ufrgs.br/~oliveira/pubs_files/
    # CVD_Simulation/CVD_Simulation.html#Errata
    L_a = alpha(d_L) * L + 0.96 * area_L / area_M * (1 - alpha(d_L)) * M
    M_a = alpha(d_M) * M + 1 / 0.96 * area_M / area_L * (1 - alpha(d_M)) * L
    S_a = cmfs[cmfs.wavelengths - d_S][:, 2]

    LMS_a = tstack([L_a, M_a, S_a])
    cmfs[cmfs.wavelengths] = LMS_a

    severity = f"{d_L}, {d_M}, {d_S}"
    template = "{0} - Anomalous Trichromacy ({1})"
    cmfs.name = template.format(cmfs.name, severity)
    cmfs.display_name = template.format(cmfs.display_name, severity)

    return cmfs


def matrix_anomalous_trichromacy_Machado2009(
    cmfs: LMS_ConeFundamentals,
    primaries: RGB_DisplayPrimaries,
    d_LMS: ArrayLike,
) -> NDArrayFloat:
    """
    Compute the *Machado et al. (2009)* colour vision deficiency matrix for
    anomalous trichromacy simulation.
    primaries tri-spectral distributions with the specified :math:`\\Delta_{LMS}` shift
    amount in nanometers to simulate anomalous trichromacy.

    Parameters
    ----------
    cmfs
        *LMS* cone fundamentals colour matching functions.
    primaries
        *RGB* display primaries tri-spectral distributions.
    d_LMS
        :math:`\\Delta_{LMS}` wavelength shift amount in nanometers for each
        cone type.

    Returns
    -------
    :class:`numpy.ndarray`
        Anomalous trichromacy transformation matrix.

    Notes
    -----
    -   Input *LMS* cone fundamentals colour matching functions interval is
        expected to be 1 nanometer, incompatible input will be interpolated
        at 1 nanometer interval.
    -   Input :math:`\\Delta_{LMS}` shift amount is in domain [0, 20].

    References
    ----------
    :cite:`Colblindorb`, :cite:`Colblindora`, :cite:`Colblindorc`,
    :cite:`Machado2009`

    Examples
    --------
    >>> from colour.characterisation import MSDS_DISPLAY_PRIMARIES
    >>> from colour.colorimetry import MSDS_CMFS_LMS
    >>> cmfs = MSDS_CMFS_LMS["Stockman & Sharpe 2 Degree Cone Fundamentals"]
    >>> d_LMS = np.array([15, 0, 0])
    >>> primaries = MSDS_DISPLAY_PRIMARIES["Apple Studio Display"]
    >>> matrix_anomalous_trichromacy_Machado2009(cmfs, primaries, d_LMS)
    ... # doctest: +ELLIPSIS
    array([[-0.2777465...,  2.6515008..., -1.3737543...],
           [ 0.2718936...,  0.2004786...,  0.5276276...],
           [ 0.0064404...,  0.2592157...,  0.7343437...]])
    """

    if cmfs.shape.interval != 1:
        cmfs = reshape_msds(
            cmfs,
            SpectralShape(cmfs.shape.start, cmfs.shape.end, 1),
            "Interpolate",
            copy=False,
        )

    M_n = matrix_RGB_to_WSYBRG(cmfs, primaries)
    cmfs_a = msds_cmfs_anomalous_trichromacy_Machado2009(cmfs, d_LMS)
    M_a = matrix_RGB_to_WSYBRG(cmfs_a, primaries)

    xp = array_namespace(M_n, M_a)

    return xp.matmul(xp.linalg.inv(M_n), M_a)


def matrix_cvd_Machado2009(
    deficiency: Literal["Deuteranomaly", "Protanomaly", "Tritanomaly"] | str,
    severity: ArrayLike,
) -> NDArrayFloat:
    """
    Compute the *Machado et al. (2009)* colour vision deficiency matrix for
    the specified deficiency and severity using pre-computed matrices.

    Parameters
    ----------
    deficiency
        Colour vision deficiency type:

        - *Protanomaly*: Defective long-wavelength cones (L-cones) with
          reduced sensitivity. Complete absence of L-cones is
          *Protanopia* or *red-dichromacy*.
        - *Deuteranomaly*: Defective medium-wavelength cones (M-cones)
          with peak sensitivity shifted towards red-sensitive cones.
          Complete absence of M-cones is *Deuteranopia*.
        - *Tritanomaly*: Defective short-wavelength cones (S-cones),
          representing an alleviated form of blue-yellow colour
          blindness. Complete absence of S-cones is *Tritanopia*.
    severity
        Severity of the colour vision deficiency in domain [0, 1].

    Returns
    -------
    :class:`numpy.ndarray`
        Colour vision deficiency matrix.

    References
    ----------
    :cite:`Colblindorb`, :cite:`Colblindora`, :cite:`Colblindorc`,
    :cite:`Machado2009`, :cite:`Machado2010a`

    Examples
    --------
    >>> matrix_cvd_Machado2009("Protanomaly", 0.15)  # doctest: +ELLIPSIS
    array([[ 0.7954665...,  0.258455 ..., -0.053921 ...],
           [ 0.040591 ...,  0.9371565...,  0.0222535...],
           [-0.003904 ..., -0.002886 ...,  1.00679  ...]])
    """

    if deficiency.lower() == "tritanomaly":
        usage_warning(
            '"Machado et al. (2009)" simulation of tritanomaly is based on '
            "the shift paradigm as an approximation to the actual phenomenon "
            "and restrain the model from trying to model tritanopia.\n"
            "The pre-generated matrices are using a shift value in domain "
            "[5, 59] contrary to the domain [0, 20] used for protanomaly and "
            "deuteranomaly simulation."
        )

    matrices = CVD_MATRICES_MACHADO2010[deficiency]
    keys = sorted(matrices)
    xp = array_namespace(severity)
    severity_array = xp.reshape(xp_as_float_array(severity, xp=xp), ())
    device = getattr(severity_array, "device", None)
    dtype = severity_array.dtype
    samples = xp.asarray(keys, dtype=dtype, device=device)
    table = xp.stack(
        [xp.asarray(matrices[key], dtype=dtype, device=device) for key in keys]
    )

    # Interpolate between the tabulated severities that bound the input.
    last = len(keys) - 1
    index = xp.searchsorted(samples, severity_array, side="right")
    lower = xp.clip(index - 1, 0, last)
    upper = xp.clip(index, 1, last)
    indices = xp.stack([lower, upper])
    bounds = xp.take(samples, indices, axis=0)
    endpoints = xp.take(table, indices, axis=0)
    a, b = bounds[0], bounds[1]
    m_1, m_2 = endpoints[0, ...], endpoints[1, ...]

    at_upper_limit = lower == upper
    offset = xp.where(at_upper_limit, a, severity_array) - a
    interval = xp.where(at_upper_limit, 1, b - a)

    return m_1 + offset * ((m_2 - m_1) / interval)
