"""
Gaussian - Reflectance Recovery
===============================

Define objects for reflectance recovery using Gaussian basis spectra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from colour.characterisation import SDS_COLOURCHECKERS
from colour.colorimetry import (
    CCS_ILLUMINANTS,
    MSDS_CMFS,
    SDS_ILLUMINANTS,
    SPECTRAL_SHAPE_DEFAULT,
    MultiSpectralDistributions,
    SpectralDistribution,
    SpectralShape,
    msds_to_XYZ_integration,
    sd_constant,
    sd_gaussian_super_clamped,
    sd_to_XYZ,
)
from colour.difference import delta_E
from colour.models import RGB_Colourspace, RGB_COLOURSPACE_sRGB, XYZ_to_Lab, XYZ_to_RGB
from colour.recovery.smits1999 import RGB_to_msds_Smits1999, RGB_to_sd_Smits1999
from colour.utilities import as_float, optional, required

if TYPE_CHECKING:
    from colour.hints import ArrayLike, Domain1, DTypeFloat, NDArrayFloat, Range1

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "PRIMARIES_GAUSSIAN",
    "WHITEPOINT_NAME_GAUSSIAN",
    "CCS_WHITEPOINT_GAUSSIAN",
    "RGB_COLOURSPACE_GAUSSIAN",
    "XYZ_to_RGB_Gaussian",
    "optimise_gaussian_basis_parameters",
    "generate_gaussian_basis",
    "MSDS_GAUSSIAN_BASIS",
    "PEAK_WAVELENGTHS_GAUSSIAN_BASIS",
    "FWHM_GAUSSIAN_BASIS",
    "EXPONENT_GAUSSIAN_BASIS",
    "RGB_to_msds_Gaussian",
    "RGB_to_sd_Gaussian",
]

PRIMARIES_GAUSSIAN: NDArrayFloat = RGB_COLOURSPACE_sRGB.primaries
"""*Gaussian* method implementation colourspace primaries."""

WHITEPOINT_NAME_GAUSSIAN: str = "E"
"""*Gaussian* method implementation colourspace whitepoint name."""

CCS_WHITEPOINT_GAUSSIAN: NDArrayFloat = CCS_ILLUMINANTS[
    "CIE 1931 2 Degree Standard Observer"
][WHITEPOINT_NAME_GAUSSIAN]
"""*Gaussian* method implementation colourspace whitepoint."""

RGB_COLOURSPACE_GAUSSIAN: RGB_Colourspace = RGB_Colourspace(
    "Gaussian",
    PRIMARIES_GAUSSIAN,
    CCS_WHITEPOINT_GAUSSIAN,
    WHITEPOINT_NAME_GAUSSIAN,
)
"""*Gaussian* colourspace."""


def XYZ_to_RGB_Gaussian(XYZ: Domain1) -> Range1:
    """
    Convert from *CIE XYZ* tristimulus values to *RGB* colourspace using
    the conditions required by the *Gaussian* method implementation.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values.

    Returns
    -------
    :class:`numpy.ndarray`
        *RGB* colour array.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | 1                     | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``RGB``    | 1                     | 1             |
    +------------+-----------------------+---------------+

    Examples
    --------
    >>> import numpy as np
    >>> XYZ = np.array([0.21781186, 0.12541048, 0.04697113])
    >>> XYZ_to_RGB_Gaussian(XYZ)  # doctest: +ELLIPSIS
    array([ 0.4063959...,  0.0275289...,  0.0398219...])
    """

    return XYZ_to_RGB(XYZ, RGB_COLOURSPACE_GAUSSIAN)


@required("SciPy")
def optimise_gaussian_basis_parameters(
    shape: SpectralShape = SPECTRAL_SHAPE_DEFAULT,
    initial_peak_wavelengths: dict | None = None,
    initial_fwhm: dict | None = None,
    initial_exponent: dict | None = None,
    optimisation_kwargs: dict | None = None,
) -> tuple[dict, dict, dict]:
    """
    Optimise Gaussian basis parameters for colorimetric accuracy.

    This function finds the peak wavelengths, FWHM values, and exponents that
    minimize the colorimetric error between the basis spectra tristimulus
    values and the target *RGB* colourspace primaries, as well as the
    :math:`\\Delta E^*_{ab}` error on *ColorChecker* patches.

    The secondary colours are modeled based on their spectral characteristics:

    -   **Cyan**: Peak in blue-green region (absorbs red), modeled as a
        Gaussian peak clamped left, similar to blue.
    -   **Yellow**: Peak in red-green region (absorbs blue), modeled as a
        Gaussian peak clamped right, similar to red.
    -   **Magenta**: High at red and blue, low at green (absorbs green),
        modeled as an inverted Gaussian (valley at green wavelengths).

    Parameters
    ----------
    shape
        Spectral shape for the distributions.
    initial_peak_wavelengths
        Initial peak wavelengths for optimization. Default includes RGB peaks
        plus cyan, magenta, and yellow peak positions.
    initial_fwhm
        Initial FWHM values for optimization. Default includes RGB FWHM
        plus cyan, magenta, and yellow FWHM values.
    initial_exponent
        Initial exponents for super-Gaussian per basis function. Default 2.0
        gives standard Gaussian. Values > 2 give flatter peaks.
    optimisation_kwargs
        Parameters for :func:`scipy.optimize.minimize` definition.

    Returns
    -------
    :class:`tuple`
        Tuple of (peak_wavelengths, fwhm, exponent) with optimized values.
        All three are dictionaries with entries for red, green, blue, cyan,
        magenta, and yellow.

    References
    ----------
    :cite:`Smits1999a`
    """

    from scipy.optimize import minimize  # noqa: PLC0415

    initial_peak_wavelengths = optional(
        initial_peak_wavelengths,
        {
            "red": 650,
            "green": 536,
            "blue": 405,
            "cyan": 554,
            "magenta": 536,
            "yellow": 550,
        },
    )
    initial_fwhm = optional(
        initial_fwhm,
        {
            "red": 138,
            "green": 137,
            "blue": 192,
            "cyan": 100,
            "magenta": 137,
            "yellow": 170,
        },
    )
    initial_exponent = optional(
        initial_exponent,
        {
            "red": 4.0,
            "green": 3.3,
            "blue": 3.3,
            "cyan": 2.4,
            "magenta": 3.3,
            "yellow": 2.8,
        },
    )

    # CMFs and illuminant for XYZ integration
    cmfs = MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].copy().align(shape)
    illuminant = SDS_ILLUMINANTS["E"].copy().align(shape)

    # XYZ values for round-trip optimization (RGB, CMY, grey)
    M = RGB_COLOURSPACE_GAUSSIAN.matrix_RGB_to_XYZ
    XYZ_c = np.array(
        [
            np.dot(M, [1, 0, 0]),  # Red
            np.dot(M, [0, 1, 0]),  # Green
            np.dot(M, [0, 0, 1]),  # Blue
            np.dot(M, [0, 1, 1]),  # Cyan
            np.dot(M, [1, 0, 1]),  # Magenta
            np.dot(M, [1, 1, 0]),  # Yellow
            np.dot(M, [0.5, 0.5, 0.5]),  # Grey
        ]
    )

    sds_cc_r = list(SDS_COLOURCHECKERS["ISO 17321-1"].values())
    XYZ_cc_r = np.array(
        [sd_to_XYZ(sd, cmfs=cmfs, illuminant=illuminant) / 100 for sd in sds_cc_r]
    )
    RGB_cc_r = XYZ_to_RGB(XYZ_cc_r, RGB_COLOURSPACE_GAUSSIAN)
    Lab_cc_r = XYZ_to_Lab(RGB_cc_r)

    def objective(parameters: NDArrayFloat) -> DTypeFloat:
        """Minimize round-trip XYZ error and ColorChecker Delta E."""

        (
            R_peak,
            G_peak,
            B_peak,
            C_peak,
            M_peak,
            Y_peak,
            R_fwhm,
            G_fwhm,
            B_fwhm,
            C_fwhm,
            M_fwhm,
            Y_fwhm,
            R_exp,
            G_exp,
            B_exp,
            C_exp,
            M_exp,
            Y_exp,
        ) = parameters

        # Primary colours
        sd_R = sd_gaussian_super_clamped(
            R_peak, R_fwhm, shape, clamp="right", exponent=R_exp
        )
        sd_G = sd_gaussian_super_clamped(
            G_peak, G_fwhm, shape, clamp="none", exponent=G_exp
        )
        sd_B = sd_gaussian_super_clamped(
            B_peak, B_fwhm, shape, clamp="left", exponent=B_exp
        )

        # Cyan: peak in blue-green region, clamped left
        sd_cyan = sd_gaussian_super_clamped(
            C_peak, C_fwhm, shape, clamp="left", exponent=C_exp
        )
        sd_cyan.name = "cyan"

        # Yellow: peak in red-green region, clamped right
        sd_yellow = sd_gaussian_super_clamped(
            Y_peak, Y_fwhm, shape, clamp="right", exponent=Y_exp
        )
        sd_yellow.name = "yellow"

        # Magenta: inverted Gaussian (valley) with independent peak and fwhm
        sd_M_gaussian = sd_gaussian_super_clamped(
            M_peak, M_fwhm, shape, clamp="none", exponent=M_exp
        )
        sd_magenta = sd_constant(1, shape) - sd_M_gaussian
        sd_magenta.name = "magenta"

        sd_white = sd_constant(1, shape)

        basis = MultiSpectralDistributions(
            [sd_white, sd_cyan, sd_magenta, sd_yellow, sd_R, sd_G, sd_B],
            labels=["white", "cyan", "magenta", "yellow", "red", "green", "blue"],
            name="Gaussian Basis (Optimisation)",
        )

        msds = MultiSpectralDistributions(
            np.transpose(
                RGB_to_msds_Smits1999(
                    XYZ_to_RGB(XYZ_c, RGB_COLOURSPACE_GAUSSIAN), basis
                )
            ),
            basis.wavelengths,
            labels=[str(i) for i in range(len(XYZ_c))],
        )
        XYZ_t = msds_to_XYZ_integration(msds, cmfs, illuminant) / 100

        # Colorimetric error for primaries/secondaries
        colorimetric_error = np.sum((XYZ_t - XYZ_c) ** 2)

        # ColorChecker Delta E error
        msds_cc_t = MultiSpectralDistributions(
            np.transpose(RGB_to_msds_Smits1999(RGB_cc_r, basis)),
            basis.wavelengths,
            labels=[str(i) for i in range(len(RGB_cc_r))],
        )
        XYZ_cc_t = msds_to_XYZ_integration(msds_cc_t, cmfs, illuminant) / 100
        Lab_cc_t = XYZ_to_Lab(XYZ_cc_t)
        delta_E_cc = delta_E(Lab_cc_r, Lab_cc_t, method="CIE 2000")
        colorchecker_error = np.mean(delta_E_cc)

        # Smoothness penalty: penalize deviation from standard Gaussian (exp=2)
        exponents = np.array([R_exp, G_exp, B_exp, C_exp, M_exp, Y_exp])
        smoothness_penalty = np.sum((exponents - 2.0) ** 2) * 0.1

        # Combined loss: colorimetric error + ColorChecker Delta E + smoothness
        return as_float(colorimetric_error + colorchecker_error + smoothness_penalty)

    x0 = [
        initial_peak_wavelengths["red"],
        initial_peak_wavelengths["green"],
        initial_peak_wavelengths["blue"],
        initial_peak_wavelengths["cyan"],
        initial_peak_wavelengths["magenta"],
        initial_peak_wavelengths["yellow"],
        initial_fwhm["red"],
        initial_fwhm["green"],
        initial_fwhm["blue"],
        initial_fwhm["cyan"],
        initial_fwhm["magenta"],
        initial_fwhm["yellow"],
        initial_exponent["red"],
        initial_exponent["green"],
        initial_exponent["blue"],
        initial_exponent["cyan"],
        initial_exponent["magenta"],
        initial_exponent["yellow"],
    ]

    bounds = [
        (475, 790),  # red peak
        (400, 670),  # green peak
        (360, 510),  # blue peak
        (420, 700),  # cyan peak
        (400, 670),  # magenta peak (valley at green region)
        (400, 670),  # yellow peak
        (75, 175),  # red fwhm
        (75, 175),  # green fwhm
        (110, 240),  # blue fwhm
        (55, 125),  # cyan fwhm
        (75, 175),  # magenta fwhm
        (95, 215),  # yellow fwhm
        (2.0, 5.0),  # red exponent
        (2.0, 5.0),  # green exponent
        (2.0, 5.0),  # blue exponent
        (2.0, 5.0),  # cyan exponent
        (2.0, 5.0),  # magenta exponent
        (2.0, 5.0),  # yellow exponent
    ]

    optimisation_settings = {
        "method": "L-BFGS-B",
        "bounds": bounds,
    }
    if optimisation_kwargs is not None:
        optimisation_settings.update(optimisation_kwargs)

    result = minimize(objective, x0, **optimisation_settings)
    (
        R_peak,
        G_peak,
        B_peak,
        C_peak,
        M_peak,
        Y_peak,
        R_fwhm,
        G_fwhm,
        B_fwhm,
        C_fwhm,
        M_fwhm,
        Y_fwhm,
        R_exp,
        G_exp,
        B_exp,
        C_exp,
        M_exp,
        Y_exp,
    ) = result.x

    peak_wavelengths = {
        "red": float(R_peak),
        "green": float(G_peak),
        "blue": float(B_peak),
        "cyan": float(C_peak),
        "magenta": float(M_peak),
        "yellow": float(Y_peak),
    }

    fwhm = {
        "red": float(R_fwhm),
        "green": float(G_fwhm),
        "blue": float(B_fwhm),
        "cyan": float(C_fwhm),
        "magenta": float(M_fwhm),
        "yellow": float(Y_fwhm),
    }

    exponent = {
        "red": float(R_exp),
        "green": float(G_exp),
        "blue": float(B_exp),
        "cyan": float(C_exp),
        "magenta": float(M_exp),
        "yellow": float(Y_exp),
    }

    return peak_wavelengths, fwhm, exponent


def generate_gaussian_basis(
    shape: SpectralShape = SPECTRAL_SHAPE_DEFAULT,
    peak_wavelengths: dict | None = None,
    fwhm: dict | None = None,
    exponent: dict | None = None,
) -> MultiSpectralDistributions:
    """
    Generate a set of Gaussian basis multi-spectral distributions.

    The secondary colours are modeled based on their spectral characteristics:

    -   **Cyan**: Peak in blue-green region (absorbs red), modeled as a
        Gaussian peak clamped left, similar to blue.
    -   **Yellow**: Peak in red-green region (absorbs blue), modeled as a
        Gaussian peak clamped right, similar to red.
    -   **Magenta**: High at red and blue, low at green (absorbs green),
        modeled as an inverted Gaussian (valley at green wavelengths).

    Parameters
    ----------
    shape
        Spectral shape for the distributions.
    peak_wavelengths
        Dictionary with peak wavelengths for red, green, blue, cyan, magenta,
        and yellow.
    fwhm
        Dictionary with FWHM values for red, green, blue, cyan, magenta, and
        yellow.
    exponent
        Dictionary with exponent values for red, green, blue, cyan, magenta,
        and yellow. Default 2.0 gives a standard Gaussian. Values > 2 give a
        flatter peak (super-Gaussian).

    Returns
    -------
    :class:`colour.MultiSpectralDistributions`
        Gaussian basis multi-spectral distributions with signals: white, cyan,
        magenta, yellow, red, green, blue.

    References
    ----------
    :cite:`Smits1999a`

    Examples
    --------
    >>> basis = generate_gaussian_basis()
    >>> sorted(basis.labels)
    ['blue', 'cyan', 'green', 'magenta', 'red', 'white', 'yellow']
    >>> basis.signals["yellow"].values.max()  # doctest: +ELLIPSIS
    1.0...
    """

    peak_wavelengths = optional(peak_wavelengths, PEAK_WAVELENGTHS_GAUSSIAN_BASIS)
    fwhm = optional(fwhm, FWHM_GAUSSIAN_BASIS)
    exponent = optional(exponent, EXPONENT_GAUSSIAN_BASIS)

    # Primary colours with clamping
    sd_red = sd_gaussian_super_clamped(
        peak_wavelengths["red"],
        fwhm["red"],
        shape,
        clamp="right",
        exponent=exponent["red"],
        name="red",
    )
    sd_green = sd_gaussian_super_clamped(
        peak_wavelengths["green"],
        fwhm["green"],
        shape,
        clamp="none",
        exponent=exponent["green"],
        name="green",
    )
    sd_blue = sd_gaussian_super_clamped(
        peak_wavelengths["blue"],
        fwhm["blue"],
        shape,
        clamp="left",
        exponent=exponent["blue"],
        name="blue",
    )

    # Cyan: peak in blue-green region, clamped left
    sd_cyan = sd_gaussian_super_clamped(
        peak_wavelengths["cyan"],
        fwhm["cyan"],
        shape,
        clamp="left",
        exponent=exponent["cyan"],
        name="cyan",
    )

    # Yellow: peak in red-green region, clamped right
    sd_yellow = sd_gaussian_super_clamped(
        peak_wavelengths["yellow"],
        fwhm["yellow"],
        shape,
        clamp="right",
        exponent=exponent["yellow"],
        name="yellow",
    )

    # Magenta: inverted Gaussian (valley) with independent peak and fwhm
    sd_M_gaussian = sd_gaussian_super_clamped(
        peak_wavelengths["magenta"],
        fwhm["magenta"],
        shape,
        clamp="none",
        exponent=exponent["magenta"],
    )
    sd_magenta = sd_constant(1, shape) - sd_M_gaussian
    sd_magenta.name = "magenta"

    # White as constant
    sd_white = sd_constant(1, shape)
    sd_white.name = "white"

    sds = [sd_white, sd_cyan, sd_magenta, sd_yellow, sd_red, sd_green, sd_blue]

    return MultiSpectralDistributions(
        sds,
        labels=[sd.name for sd in sds],
        name="Gaussian Basis",
    )


PEAK_WAVELENGTHS_GAUSSIAN_BASIS: dict = {
    "red": 633.256,
    "green": 538.296,
    "blue": 429.172,
    "cyan": 560.133,
    "magenta": 539.403,
    "yellow": 536.836,
}
"""
Default peak wavelengths for Gaussian basis spectra.

These values are optimized for round-trip colorimetric accuracy using
:func:`optimise_gaussian_basis_parameters`.
"""

FWHM_GAUSSIAN_BASIS: dict = {
    "red": 103.200,
    "green": 118.477,
    "blue": 144.000,
    "cyan": 75.000,
    "magenta": 124.702,
    "yellow": 128.241,
}
"""
Default full width at half maximum for Gaussian basis spectra.

These values are optimized for round-trip colorimetric accuracy using
:func:`optimise_gaussian_basis_parameters`.
"""

EXPONENT_GAUSSIAN_BASIS: dict = {
    "red": 3.01,
    "green": 2.73,
    "blue": 2.53,
    "cyan": 2.19,
    "magenta": 2.60,
    "yellow": 2.09,
}
"""
Default exponents for Gaussian basis spectra.

A value of 2.0 gives a standard Gaussian. Values > 2 give a flatter peak
(super-Gaussian). These values are optimized for round-trip colorimetric
accuracy using :func:`optimise_gaussian_basis_parameters`.
"""

MSDS_GAUSSIAN_BASIS: MultiSpectralDistributions = generate_gaussian_basis()
MSDS_GAUSSIAN_BASIS.__doc__ = """
Gaussian basis multi-spectral distributions for spectral upsampling.

The basis spectra use clamped Gaussians with parameters optimized for round-trip
colorimetric accuracy with *sRGB* primaries:

- Red: Gaussian clamped flat toward long wavelengths
- Green: Symmetric Gaussian
- Blue: Gaussian clamped flat toward short wavelengths
- Cyan: Independent Gaussian clamped flat toward short wavelengths
- Yellow: Independent Gaussian clamped flat toward long wavelengths
- Magenta: Inverted Gaussian (valley at green wavelengths)
- White: Constant value of 1

Parameters can be recomputed using :func:`optimise_gaussian_basis_parameters`.
"""


def RGB_to_msds_Gaussian(RGB: ArrayLike) -> NDArrayFloat:
    """
    Recover spectral values from *RGB* colourspace array using *Gaussian*
    basis spectra and the *Smits (1999)* decomposition algorithm.

    Parameters
    ----------
    RGB
        *RGB* colourspace array to recover spectral values from. The last
        dimension must be size 3.

    Returns
    -------
    :class:`numpy.ndarray`
        Recovered spectral values with shape ``(*RGB.shape[:-1], wavelengths)``.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``RGB``    | 1                     | 1             |
    +------------+-----------------------+---------------+

    Examples
    --------
    >>> import numpy as np
    >>> RGB = np.array(
    ...     [
    ...         [0.45623196, 0.03080455, 0.04093343],
    ...         [0.05438271, 0.29877169, 0.07188444],
    ...         [0.01863137, 0.05139773, 0.28887675],
    ...     ]
    ... )
    >>> RGB_to_msds_Gaussian(RGB).shape
    (3, 421)
    >>> RGB_to_msds_Gaussian(RGB)[0, 300]  # doctest: +ELLIPSIS
    0.4562...
    """

    return RGB_to_msds_Smits1999(RGB, MSDS_GAUSSIAN_BASIS)


def RGB_to_sd_Gaussian(RGB: Domain1) -> SpectralDistribution:
    """
    Recover the spectral distribution of the specified *RGB* colourspace array
    using Gaussian basis spectra and the *Smits (1999)* decomposition algorithm.

    Parameters
    ----------
    RGB
        *RGB* colourspace array to recover the spectral distribution from.

    Returns
    -------
    :class:`colour.SpectralDistribution`
        Recovered spectral distribution.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``RGB``    | 1                     | 1             |
    +------------+-----------------------+---------------+

    Examples
    --------
    >>> import numpy as np
    >>> from colour import MSDS_CMFS, SDS_ILLUMINANTS, SpectralShape
    >>> from colour.colorimetry import sd_to_XYZ_integration
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> RGB = XYZ_to_RGB_Gaussian(XYZ)
    >>> cmfs = (
    ...     MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    ...     .copy()
    ...     .align(SpectralShape(360, 780, 10))
    ... )
    >>> illuminant = SDS_ILLUMINANTS["E"].copy().align(cmfs.shape)
    >>> sd = RGB_to_sd_Gaussian(RGB)
    >>> sd_to_XYZ_integration(sd, cmfs, illuminant) / 100  # doctest: +ELLIPSIS
    array([ 0.1934227...,  0.1161235...,  0.0435094...])
    """

    return RGB_to_sd_Smits1999(RGB, MSDS_GAUSSIAN_BASIS, f"Gaussian - {RGB!r}")
