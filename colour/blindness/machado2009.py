"""
Simulation of CVD - Machado, Oliveira and Fernandes (2009)
==========================================================

Define the *Machado et al. (2009)* objects for simulation of colour vision
deficiency:

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

import numpy as np

from colour.algebra import vecmul
from colour.blindness import CVD_MATRICES_MACHADO2010
from colour.characterisation import RGB_DisplayPrimaries
from colour.colorimetry import (
    LMS_ConeFundamentals,
    SpectralShape,
    reshape_msds,
)
from colour.hints import ArrayLike, Literal, NDArrayFloat
from colour.utilities import (
    as_float_array,
    as_int_scalar,
    tsplit,
    tstack,
    usage_warning,
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
    Compute the matrix transforming from *RGB* colourspace to opponent-colour
    space using *Machado et al. (2009)* method.

    Parameters
    ----------
    cmfs
        *LMS* cone fundamentals colour matching functions.
    primaries
        *RGB* display primaries tri-spectral distributions.

    Returns
    -------
    :class:`numpy.ndarray`
        Matrix transforming from *RGB* colourspace to opponent-colour space.

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
    WSYBRG = vecmul(MATRIX_LMS_TO_WSYBRG, cmfs.values)
    WS, YB, RG = tsplit(WSYBRG)

    primaries = reshape_msds(
        primaries,
        cmfs.shape,
        copy=False,
        extrapolator_kwargs={"method": "Constant", "left": 0, "right": 0},
    )

    R, G, B = tsplit(primaries.values)

    WS_R = np.trapezoid(R * WS, wavelengths)  # pyright: ignore
    WS_G = np.trapezoid(G * WS, wavelengths)  # pyright: ignore
    WS_B = np.trapezoid(B * WS, wavelengths)  # pyright: ignore

    YB_R = np.trapezoid(R * YB, wavelengths)  # pyright: ignore
    YB_G = np.trapezoid(G * YB, wavelengths)  # pyright: ignore
    YB_B = np.trapezoid(B * YB, wavelengths)  # pyright: ignore

    RG_R = np.trapezoid(R * RG, wavelengths)  # pyright: ignore
    RG_G = np.trapezoid(G * RG, wavelengths)  # pyright: ignore
    RG_B = np.trapezoid(B * RG, wavelengths)  # pyright: ignore

    M_G = as_float_array(
        [
            [WS_R, WS_G, WS_B],
            [YB_R, YB_G, YB_B],
            [RG_R, RG_G, RG_B],
        ]
    )

    M_G /= np.sum(M_G, axis=-1)[:, None]

    return M_G


def msds_cmfs_anomalous_trichromacy_Machado2009(
    cmfs: LMS_ConeFundamentals, d_LMS: ArrayLike
) -> LMS_ConeFundamentals:
    """
    Shift given *LMS* cone fundamentals colour matching functions with given
    :math:`\\Delta_{LMS}` shift amount in nanometers to simulate anomalous
    trichromacy using *Machado et al. (2009)* method.

    Parameters
    ----------
    cmfs
        *LMS* cone fundamentals colour matching functions.
    d_LMS
        :math:`\\Delta_{LMS}` shift amount in nanometers.

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
    array([ 0.0498639,  0.0870524,  0.955393 ])
    >>> msds_cmfs_anomalous_trichromacy_Machado2009(cmfs, np.array([15, 0, 0]))[
    ...     450
    ... ]  # doctest: +ELLIPSIS
    array([ 0.0891288...,  0.0870524 ,  0.955393  ])
    """

    cmfs = cmfs.copy()

    if cmfs.shape.interval != 1:
        cmfs.interpolate(SpectralShape(cmfs.shape.start, cmfs.shape.end, 1))

    cmfs.extrapolator_kwargs = {"method": "Constant", "left": 0, "right": 0}

    L, M, _S = tsplit(cmfs.values)
    d_L, d_M, d_S = tsplit(d_LMS)

    if d_S != 0:
        usage_warning(
            '"Machado et al. (2009)" simulation of tritanomaly is based on '
            "the shift paradigm as an approximation to the actual phenomenon "
            "and restrain the model from trying to model tritanopia.\n"
            "The pre-generated matrices are using a shift value in domain "
            "[5, 59] contrary to the domain [0, 20] used for protanomaly and "
            "deuteranomaly simulation."
        )

    area_L = np.trapezoid(L, cmfs.wavelengths)  # pyright: ignore
    area_M = np.trapezoid(M, cmfs.wavelengths)  # pyright: ignore

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
    Compute the *Machado et al. (2009)* *CVD* matrix for given *LMS* cone
    fundamentals colour matching functions and display primaries tri-spectral
    distributions with given :math:`\\Delta_{LMS}` shift amount in nanometers
    to simulate anomalous trichromacy.

    Parameters
    ----------
    cmfs
        *LMS* cone fundamentals colour matching functions.
    primaries
        *RGB* display primaries tri-spectral distributions.
    d_LMS
        :math:`\\Delta_{LMS}` shift amount in nanometers.

    Notes
    -----
    -   Input *LMS* cone fundamentals colour matching functions interval is
        expected to be 1 nanometer, incompatible input will be interpolated
        at 1 nanometer interval.
    -   Input :math:`\\Delta_{LMS}` shift amount is in domain [0, 20].

    Returns
    -------
    :class:`numpy.ndarray`
        Anomalous trichromacy matrix.

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

    return np.matmul(np.linalg.inv(M_n), M_a)


def matrix_cvd_Machado2009(
    deficiency: Literal["Deuteranomaly", "Protanomaly", "Tritanomaly"] | str,
    severity: float,
) -> NDArrayFloat:
    """
    Compute *Machado et al. (2009)* *CVD* matrix for given deficiency and
    severity using the pre-computed matrices dataset.

    Parameters
    ----------
    deficiency
        Colour blindness / vision deficiency types :
        - *Protanomaly* : defective long-wavelength cones (L-cones). The
        complete absence of L-cones is known as *Protanopia* or
        *red-dichromacy*.
        - *Deuteranomaly* : defective medium-wavelength cones (M-cones) with
        peak of sensitivity moved towards the red sensitive cones. The complete
        absence of M-cones is known as *Deuteranopia*.
        - *Tritanomaly* : defective short-wavelength cones (S-cones), an
        alleviated form of blue-yellow color blindness. The complete absence of
        S-cones is known as *Tritanopia*.
    severity
        Severity of the colour vision deficiency in domain [0, 1].

    Returns
    -------
    :class:`numpy.ndarray`
        *CVD* matrix.

    References
    ----------
    :cite:`Colblindorb`, :cite:`Colblindora`, :cite:`Colblindorc`,
    :cite:`Machado2009`

    Examples
    --------
    >>> matrix_cvd_Machado2009("Protanomaly", 0.15)  # doctest: +ELLIPSIS
    array([[ 0.7869875...,  0.2694875..., -0.0564735...],
           [ 0.0431695...,  0.933774 ...,  0.023058 ...],
           [-0.004238 ..., -0.0024515...,  1.0066895...]])
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
    samples = np.array(sorted(matrices.keys()))
    index = as_int_scalar(
        np.clip(np.searchsorted(samples, severity), 0, len(samples) - 1)
    )

    a = samples[index]
    b = samples[min(index + 1, len(samples) - 1)]

    m1, m2 = matrices[a], matrices[b]

    if a == b:
        # 1.0 severity CVD matrix, returning directly.
        return m1
    else:
        return m1 + (severity - a) * ((m2 - m1) / (b - a))
