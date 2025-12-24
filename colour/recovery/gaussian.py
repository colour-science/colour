"""
Gaussian - Reflectance Recovery
===============================

Define objects for reflectance recovery using Gaussian basis spectra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

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
    sd_gaussian,
)
from colour.models import RGB_Colourspace, RGB_COLOURSPACE_sRGB, XYZ_to_RGB
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
    "sd_gaussian_clamped",
    "optimise_gaussian_basis_parameters",
    "generate_gaussian_basis",
    "MSDS_GAUSSIAN_BASIS",
    "PEAK_WAVELENGTHS_GAUSSIAN_BASIS",
    "FWHM_GAUSSIAN_BASIS",
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


def sd_gaussian_clamped(
    peak_wavelength: float,
    fwhm: float,
    shape: SpectralShape = SPECTRAL_SHAPE_DEFAULT,
    clamp: str = "none",
    name: str | None = None,
) -> SpectralDistribution:
    """
    Generate a Gaussian spectral distribution, optionally clamped flat on one
    side of the peak, with the peak normalized to 1.

    Parameters
    ----------
    peak_wavelength
        Peak wavelength of the Gaussian.
    fwhm
        Full width at half maximum.
    shape
        Spectral shape for the distribution.
    clamp
        Clamping mode: ``"none"`` for symmetric Gaussian, ``"left"`` for flat
        from start to peak, ``"right"`` for flat from peak to end.
    name
        Name for the spectral distribution.

    Returns
    -------
    :class:`colour.SpectralDistribution`
        Clamped Gaussian spectral distribution with peak normalized to 1.

    Examples
    --------
    >>> sd = sd_gaussian_clamped(600, 50, clamp="right")
    >>> sd.name = "Red Gaussian"
    >>> round(sd[600], 5)
    1.0
    >>> round(sd[700], 5)
    1.0
    """

    sd = sd_gaussian(peak_wavelength, fwhm, shape, method="FWHM")
    sd.range = sd.range / sd.range.max()  # Normalize peak to 1

    if clamp == "left":
        sd[sd.wavelengths[sd.wavelengths <= peak_wavelength]] = 1.0
    elif clamp == "right":
        sd[sd.wavelengths[sd.wavelengths >= peak_wavelength]] = 1.0

    sd.name = name or f"Gaussian {peak_wavelength}nm"

    return sd


@required("SciPy")
def optimise_gaussian_basis_parameters(
    shape: SpectralShape = SPECTRAL_SHAPE_DEFAULT,
    initial_peak_wavelengths: dict | None = None,
    initial_fwhm: dict | None = None,
    optimisation_kwargs: dict | None = None,
) -> tuple[dict, dict]:
    """
    Optimise Gaussian basis parameters for colorimetric accuracy.

    This function finds the peak wavelengths and FWHM values that minimize the
    colorimetric error between the basis spectra tristimulus values and the
    target *RGB* colourspace primaries.

    Parameters
    ----------
    shape
        Spectral shape for the distributions.
    initial_peak_wavelengths
        Initial peak wavelengths for optimization. Default is
        ``{"red": 600, "green": 540, "blue": 460}``.
    initial_fwhm
        Initial FWHM values for optimization. Default is
        ``{"red": 65, "green": 65, "blue": 65}``.
    optimisation_kwargs
        Parameters for :func:`scipy.optimize.minimize` definition.

    Returns
    -------
    :class:`tuple`
        Tuple of (peak_wavelengths, fwhm) dictionaries with optimized values.

    Examples
    --------
    >>> peaks, fwhm = optimise_gaussian_basis_parameters()
    >>> print(f"Red: peak={peak_wavelengths['red']:.1f}nm, FWHM={fwhm['red']:.1f}nm")
    ... # doctest: +SKIP
    """

    from scipy.optimize import minimize  # noqa: PLC0415

    initial_peak_wavelengths = optional(
        initial_peak_wavelengths, {"red": 600, "green": 540, "blue": 460}
    )
    initial_fwhm = optional(initial_fwhm, {"red": 70, "green": 70, "blue": 70})

    # CMFs and illuminant for XYZ integration
    cmfs = MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].copy().align(shape)
    illuminant = SDS_ILLUMINANTS["E"].copy().align(shape)

    # Test XYZ values for round-trip optimization (RGB, CMY, grey)
    M = RGB_COLOURSPACE_GAUSSIAN.matrix_RGB_to_XYZ
    test_XYZ = np.array(
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

    def objective(parameters: NDArrayFloat) -> DTypeFloat:
        """Minimize round-trip XYZ error."""

        R_peak, G_peak, B_peak, R_fwhm, G_fwhm, B_fwhm = parameters

        # Generate basis spectra
        sd_R = sd_gaussian_clamped(R_peak, R_fwhm, shape, clamp="right")
        sd_G = sd_gaussian_clamped(G_peak, G_fwhm, shape, clamp="none")
        sd_B = sd_gaussian_clamped(B_peak, B_fwhm, shape, clamp="left")

        # Secondary colours derived from primaries
        sd_cyan = sd_G + sd_B
        sd_cyan.name = "cyan"
        sd_magenta = sd_R + sd_B
        sd_magenta.name = "magenta"
        sd_yellow = sd_R + sd_G
        sd_yellow.name = "yellow"
        sd_white = sd_constant(1, shape)

        basis = MultiSpectralDistributions(
            [sd_white, sd_cyan, sd_magenta, sd_yellow, sd_R, sd_G, sd_B],
            labels=["white", "cyan", "magenta", "yellow", "red", "green", "blue"],
            name="Gaussian Basis (Optimisation)",
        )

        msds = MultiSpectralDistributions(
            np.transpose(
                RGB_to_msds_Smits1999(
                    XYZ_to_RGB(test_XYZ, RGB_COLOURSPACE_GAUSSIAN), basis
                )
            ),
            basis.wavelengths,
            labels=[str(i) for i in range(len(test_XYZ))],
        )
        XYZ_recovered = msds_to_XYZ_integration(msds, cmfs, illuminant) / 100

        return as_float(np.sum((XYZ_recovered - test_XYZ) ** 2))

    x0 = [
        initial_peak_wavelengths["red"],
        initial_peak_wavelengths["green"],
        initial_peak_wavelengths["blue"],
        initial_fwhm["red"],
        initial_fwhm["green"],
        initial_fwhm["blue"],
    ]

    bounds = [
        (580, 650),  # red peak
        (510, 570),  # green peak
        (420, 480),  # blue peak
        (20, 150),  # red fwhm
        (20, 150),  # green fwhm
        (20, 150),  # blue fwhm
    ]

    optimisation_settings = {
        "method": "L-BFGS-B",
        "bounds": bounds,
    }
    if optimisation_kwargs is not None:
        optimisation_settings.update(optimisation_kwargs)

    result = minimize(objective, x0, **optimisation_settings)
    R_peak, G_peak, B_peak, R_fwhm, G_fwhm, B_fwhm = result.x

    peak_wavelengths = {
        "red": float(R_peak),
        "green": float(G_peak),
        "blue": float(B_peak),
    }

    fwhm = {
        "red": float(R_fwhm),
        "green": float(G_fwhm),
        "blue": float(B_fwhm),
    }

    return peak_wavelengths, fwhm


def generate_gaussian_basis(
    shape: SpectralShape = SPECTRAL_SHAPE_DEFAULT,
    peak_wavelengths: dict | None = None,
    fwhm: dict | None = None,
) -> MultiSpectralDistributions:
    """
    Generate a set of Gaussian basis multi-spectral distributions.

    Parameters
    ----------
    shape
        Spectral shape for the distributions.
    peak_wavelengths
        Dictionary with peak wavelengths for red, green, and blue.
    fwhm
        Dictionary with FWHM values for red, green, and blue.

    Returns
    -------
    :class:`colour.MultiSpectralDistributions`
        Gaussian basis multi-spectral distributions with signals: white, cyan,
        magenta, yellow, red, green, blue.

    Examples
    --------
    >>> basis = generate_gaussian_basis()
    >>> sorted(basis.labels)
    ['blue', 'cyan', 'green', 'magenta', 'red', 'white', 'yellow']
    """

    peak_wavelengths = optional(peak_wavelengths, PEAK_WAVELENGTHS_GAUSSIAN_BASIS)
    fwhm = optional(fwhm, FWHM_GAUSSIAN_BASIS)

    # Primary colours with clamping
    sd_red = sd_gaussian_clamped(
        peak_wavelengths["red"], fwhm["red"], shape, clamp="right", name="red"
    )
    sd_green = sd_gaussian_clamped(
        peak_wavelengths["green"], fwhm["green"], shape, clamp="none", name="green"
    )
    sd_blue = sd_gaussian_clamped(
        peak_wavelengths["blue"], fwhm["blue"], shape, clamp="left", name="blue"
    )

    # Secondary colours derived from primaries
    sd_cyan = sd_green + sd_blue
    sd_cyan.name = "cyan"
    sd_magenta = sd_red + sd_blue
    sd_magenta.name = "magenta"
    sd_yellow = sd_red + sd_green
    sd_yellow.name = "yellow"

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
    "red": 619.2,
    "green": 541.8,
    "blue": 445.2,
}
"""
Default peak wavelengths for Gaussian basis spectra.

These values are optimized for round-trip colorimetric accuracy using
:func:`optimise_gaussian_basis_parameters`.
"""

FWHM_GAUSSIAN_BASIS: dict = {
    "red": 58.5,
    "green": 89.4,
    "blue": 85.4,
}
"""
Default full width at half maximum for Gaussian basis spectra.

These values are optimized for round-trip colorimetric accuracy using
:func:`optimise_gaussian_basis_parameters`.
"""

MSDS_GAUSSIAN_BASIS: MultiSpectralDistributions = generate_gaussian_basis()
MSDS_GAUSSIAN_BASIS.__doc__ = """
Gaussian basis multi-spectral distributions for spectral upsampling.

The basis spectra use clamped Gaussians with parameters optimized for round-trip
colorimetric accuracy with *sRGB* primaries:

- Red: Gaussian centered at 619.2nm, FWHM 58.5nm, clamped flat toward long wavelengths
- Green: Symmetric Gaussian centered at 541.8nm, FWHM 89.4nm
- Blue: Gaussian centered at 445.2nm, FWHM 85.4nm, clamped flat toward short wavelengths
- Cyan, Magenta, Yellow: Derived as C=G+B, M=R+B, Y=R+G
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
    array([ 0.2038334...,  0.1254643...,  0.0434193...])
    """

    return RGB_to_sd_Smits1999(RGB, MSDS_GAUSSIAN_BASIS, f"Gaussian - {RGB!r}")
