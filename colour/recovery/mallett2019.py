"""
Mallett and Yuksel (2019) - Reflectance Recovery
================================================

Defines the objects for reflectance recovery, i.e. spectral upsampling, using
*Mallett and Yuksel (2019)* method:

-   :func:`colour.recovery.spectral_primary_decomposition_Mallett2019`
-   :func:`colour.recovery.RGB_to_sd_Mallett2019`

References
----------
-   :cite:`Mallett2019` : Mallett, I., & Yuksel, C. (2019). Spectral Primary
    Decomposition for Rendering with sRGB Reflectance. Eurographics Symposium
    on Rendering - DL-Only and Industry Track, 7 pages. doi:10.2312/SR.20191216
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import Bounds, LinearConstraint, minimize

from colour.colorimetry import (
    MultiSpectralDistributions,
    SpectralDistribution,
    handle_spectral_arguments,
)
from colour.models import RGB_Colourspace
from colour.hints import ArrayLike, Callable, Dict, Optional, Tuple
from colour.recovery import MSDS_BASIS_FUNCTIONS_sRGB_MALLETT2019
from colour.utilities import to_domain_1

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "spectral_primary_decomposition_Mallett2019",
    "RGB_to_sd_Mallett2019",
]


def spectral_primary_decomposition_Mallett2019(
    colourspace: RGB_Colourspace,
    cmfs: Optional[MultiSpectralDistributions] = None,
    illuminant: Optional[SpectralDistribution] = None,
    metric: Callable = np.linalg.norm,
    metric_args: Tuple = tuple(),
    optimisation_kwargs: Optional[Dict] = None,
) -> MultiSpectralDistributions:
    """
    Perform the spectral primary decomposition as described in *Mallett and
    Yuksel (2019)* for given *RGB* colourspace.

    Parameters
    ----------
    colourspace
        *RGB* colourspace.
    cmfs
        Standard observer colour matching functions, default to the
        *CIE 1931 2 Degree Standard Observer*.
    illuminant
        Illuminant spectral distribution, default to
        *CIE Standard Illuminant D65*.
    metric
        Function to be minimised, i.e. the objective function.

            ``metric(basis, *metric_args) -> float``

        where ``basis`` is three reflectances concatenated together, each
        with a shape matching ``shape``.
    metric_args
        Additional arguments passed to ``metric``.
    optimisation_kwargs
        Parameters for :func:`scipy.optimize.minimize` definition.

    Returns
    -------
    :class:`colour.MultiSpectralDistributions`
        Basis functions for given *RGB* colourspace.

    References
    ----------
    :cite:`Mallett2019`

    Notes
    -----
    -   In-addition to the *BT.709* primaries used by the *sRGB* colourspace,
        :cite:`Mallett2019` tried *BT.2020*, *P3 D65*, *Adobe RGB 1998*,
        *NTSC (1987)*, *Pal/Secam*, *ProPhoto RGB*,
        and *Adobe Wide Gamut RGB* primaries, every one of which encompasses a
        larger (albeit not-always-enveloping) set of *CIE L\\*a\\*b\\** colours
        than BT.709. Of these, only *Pal/Secam* produces a feasible basis,
        which is relatively unsurprising since it is very similar to *BT.709*,
        whereas the others are significantly larger.

    Examples
    --------
    >>> from colour import MSDS_CMFS, SDS_ILLUMINANTS, SpectralShape
    >>> from colour.models import RGB_COLOURSPACE_PAL_SECAM
    >>> from colour.utilities import numpy_print_options
    >>> cmfs = (
    ...     MSDS_CMFS['CIE 1931 2 Degree Standard Observer'].
    ...     copy().align(SpectralShape(360, 780, 10))
    ... )
    >>> illuminant = SDS_ILLUMINANTS['D65'].copy().align(cmfs.shape)
    >>> msds = spectral_primary_decomposition_Mallett2019(
    ...     RGB_COLOURSPACE_PAL_SECAM, cmfs, illuminant, optimisation_kwargs={
    ...         'options': {'ftol': 1e-5}
    ...     }
    ... )
    >>> with numpy_print_options(suppress=True):
    ...     print(msds)  # doctest: +SKIP
    [[ 360.            0.3395134...    0.3400214...    0.3204650...]
     [ 370.            0.3355246...    0.3338028...    0.3306724...]
     [ 380.            0.3376707...    0.3185578...    0.3437715...]
     [ 390.            0.3178866...    0.3351754...    0.3469378...]
     [ 400.            0.3045154...    0.3248376...    0.3706469...]
     [ 410.            0.2935652...    0.2919463...    0.4144884...]
     [ 420.            0.1875740...    0.1853729...    0.6270530...]
     [ 430.            0.0167983...    0.054483 ...    0.9287186...]
     [ 440.            0.       ...    0.       ...    1.       ...]
     [ 450.            0.       ...    0.       ...    1.       ...]
     [ 460.            0.       ...    0.       ...    1.       ...]
     [ 470.            0.       ...    0.0458044...    0.9541955...]
     [ 480.            0.       ...    0.2960917...    0.7039082...]
     [ 490.            0.       ...    0.5042592...    0.4957407...]
     [ 500.            0.       ...    0.6655795...    0.3344204...]
     [ 510.            0.       ...    0.8607541...    0.1392458...]
     [ 520.            0.       ...    0.9999998...    0.0000001...]
     [ 530.            0.       ...    1.       ...    0.       ...]
     [ 540.            0.       ...    1.       ...    0.       ...]
     [ 550.            0.       ...    1.       ...    0.       ...]
     [ 560.            0.       ...    0.9924229...    0.       ...]
     [ 570.            0.       ...    0.9970703...    0.0025673...]
     [ 580.            0.0396002...    0.9028231...    0.0575766...]
     [ 590.            0.7058973...    0.2941026...    0.       ...]
     [ 600.            1.       ...    0.       ...    0.       ...]
     [ 610.            1.       ...    0.       ...    0.       ...]
     [ 620.            1.       ...    0.       ...    0.       ...]
     [ 630.            1.       ...    0.       ...    0.       ...]
     [ 640.            0.9835925...    0.0100166...    0.0063908...]
     [ 650.            0.7878949...    0.1265097...    0.0855953...]
     [ 660.            0.5987994...    0.2051062...    0.1960942...]
     [ 670.            0.4724493...    0.2649623...    0.2625883...]
     [ 680.            0.3989806...    0.3007488...    0.3002704...]
     [ 690.            0.3666586...    0.3164003...    0.3169410...]
     [ 700.            0.3497806...    0.3242863...    0.3259329...]
     [ 710.            0.3563736...    0.3232441...    0.3203822...]
     [ 720.            0.3362624...    0.3326209...    0.3311165...]
     [ 730.            0.3245015...    0.3365982...    0.3389002...]
     [ 740.            0.3335520...    0.3320670...    0.3343808...]
     [ 750.            0.3441287...    0.3291168...    0.3267544...]
     [ 760.            0.3343705...    0.3330132...    0.3326162...]
     [ 770.            0.3274633...    0.3305704...    0.3419662...]
     [ 780.            0.3475263...    0.3262331...    0.3262404...]]
    """

    cmfs, illuminant = handle_spectral_arguments(cmfs, illuminant)

    N = len(cmfs.shape)

    R_to_XYZ = np.transpose(
        illuminant.values[..., np.newaxis]
        * cmfs.values
        / (np.sum(cmfs.values[:, 1] * illuminant.values))
    )
    R_to_RGB = np.dot(colourspace.matrix_XYZ_to_RGB, R_to_XYZ)
    basis_to_RGB = block_diag(R_to_RGB, R_to_RGB, R_to_RGB)

    primaries = np.reshape(np.identity(3), 9)

    # Ensure that the reflectances correspond to the correct RGB colours.
    colour_match = LinearConstraint(basis_to_RGB, primaries, primaries)

    # Ensure that the reflectances are bounded by [0, 1].
    energy_conservation = Bounds(np.zeros(3 * N), np.ones(3 * N))

    # Ensure that the sum of the three bases is bounded by [0, 1].
    sum_matrix = np.transpose(np.tile(np.identity(N), (3, 1)))
    sum_constraint = LinearConstraint(sum_matrix, np.zeros(N), np.ones(N))

    optimisation_settings = {
        "method": "SLSQP",
        "constraints": [colour_match, sum_constraint],
        "bounds": energy_conservation,
        "options": {
            "ftol": 1e-10,
        },
    }

    if optimisation_kwargs is not None:
        optimisation_settings.update(optimisation_kwargs)

    result = minimize(
        metric, args=metric_args, x0=np.zeros(3 * N), **optimisation_settings
    )

    basis_functions = np.transpose(np.reshape(result.x, (3, N)))

    return MultiSpectralDistributions(
        basis_functions,
        cmfs.shape.range(),
        name=f"Basis Functions - {colourspace.name} - Mallett (2019)",
        labels=("red", "green", "blue"),
    )


def RGB_to_sd_Mallett2019(
    RGB: ArrayLike,
    basis_functions: MultiSpectralDistributions = MSDS_BASIS_FUNCTIONS_sRGB_MALLETT2019,
) -> SpectralDistribution:
    """
    Recover the spectral distribution of given *RGB* colourspace array using
    *Mallett and Yuksel (2019)* method.

    Parameters
    ----------
    RGB
        *RGB* colourspace array.
    basis_functions
        Basis functions for the method. The default is to use the built-in
        *sRGB* basis functions, i.e.
        :attr:`colour.recovery.MSDS_BASIS_FUNCTIONS_sRGB_MALLETT2019`.

    Returns
    -------
    :class:`colour.SpectralDistribution`
        Recovered reflectance.

    References
    ----------
    :cite:`Mallett2019`

    Notes
    -----
    -   In-addition to the *BT.709* primaries used by the *sRGB* colourspace,
        :cite:`Mallett2019` tried *BT.2020*, *P3 D65*, *Adobe RGB 1998*,
        *NTSC (1987)*, *Pal/Secam*, *ProPhoto RGB*,
        and *Adobe Wide Gamut RGB* primaries, every one of which encompasses a
        larger (albeit not-always-enveloping) set of *CIE L\\*a\\*b\\** colours
        than BT.709. Of these, only *Pal/Secam* produces a feasible basis,
        which is relatively unsurprising since it is very similar to *BT.709*,
        whereas the others are significantly larger.

    Examples
    --------
    >>> from colour import MSDS_CMFS, SDS_ILLUMINANTS, XYZ_to_sRGB
    >>> from colour.colorimetry import  sd_to_XYZ_integration
    >>> from colour.recovery import SPECTRAL_SHAPE_sRGB_MALLETT2019
    >>> from colour.utilities import numpy_print_options
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> RGB = XYZ_to_sRGB(XYZ, apply_cctf_encoding=False)
    >>> cmfs = (
    ...     MSDS_CMFS['CIE 1931 2 Degree Standard Observer'].
    ...     copy().align(SPECTRAL_SHAPE_sRGB_MALLETT2019)
    ... )
    >>> illuminant = SDS_ILLUMINANTS['D65'].copy().align(cmfs.shape)
    >>> sd = RGB_to_sd_Mallett2019(RGB)
    >>> with numpy_print_options(suppress=True):
    ...     sd  # doctest: +ELLIPSIS
    SpectralDistribution([[ 380.        ,    0.1735531...],
                          [ 385.        ,    0.1720357...],
                          [ 390.        ,    0.1677721...],
                          [ 395.        ,    0.1576605...],
                          [ 400.        ,    0.1372829...],
                          [ 405.        ,    0.1170849...],
                          [ 410.        ,    0.0895694...],
                          [ 415.        ,    0.0706232...],
                          [ 420.        ,    0.0585765...],
                          [ 425.        ,    0.0523959...],
                          [ 430.        ,    0.0497598...],
                          [ 435.        ,    0.0476057...],
                          [ 440.        ,    0.0465079...],
                          [ 445.        ,    0.0460337...],
                          [ 450.        ,    0.0455839...],
                          [ 455.        ,    0.0452872...],
                          [ 460.        ,    0.0450981...],
                          [ 465.        ,    0.0448895...],
                          [ 470.        ,    0.0449257...],
                          [ 475.        ,    0.0448987...],
                          [ 480.        ,    0.0446834...],
                          [ 485.        ,    0.0441372...],
                          [ 490.        ,    0.0417137...],
                          [ 495.        ,    0.0373832...],
                          [ 500.        ,    0.0357657...],
                          [ 505.        ,    0.0348263...],
                          [ 510.        ,    0.0341953...],
                          [ 515.        ,    0.0337683...],
                          [ 520.        ,    0.0334979...],
                          [ 525.        ,    0.0332991...],
                          [ 530.        ,    0.0331909...],
                          [ 535.        ,    0.0332181...],
                          [ 540.        ,    0.0333387...],
                          [ 545.        ,    0.0334970...],
                          [ 550.        ,    0.0337381...],
                          [ 555.        ,    0.0341847...],
                          [ 560.        ,    0.0346447...],
                          [ 565.        ,    0.0353993...],
                          [ 570.        ,    0.0367367...],
                          [ 575.        ,    0.0392007...],
                          [ 580.        ,    0.0445902...],
                          [ 585.        ,    0.0625633...],
                          [ 590.        ,    0.2965381...],
                          [ 595.        ,    0.4215576...],
                          [ 600.        ,    0.4347139...],
                          [ 605.        ,    0.4385134...],
                          [ 610.        ,    0.4385184...],
                          [ 615.        ,    0.4385249...],
                          [ 620.        ,    0.4374694...],
                          [ 625.        ,    0.4384672...],
                          [ 630.        ,    0.4368251...],
                          [ 635.        ,    0.4340867...],
                          [ 640.        ,    0.4303219...],
                          [ 645.        ,    0.4243257...],
                          [ 650.        ,    0.4159482...],
                          [ 655.        ,    0.4057443...],
                          [ 660.        ,    0.3919874...],
                          [ 665.        ,    0.3742784...],
                          [ 670.        ,    0.3518421...],
                          [ 675.        ,    0.3240127...],
                          [ 680.        ,    0.2955145...],
                          [ 685.        ,    0.2625658...],
                          [ 690.        ,    0.2343423...],
                          [ 695.        ,    0.2174830...],
                          [ 700.        ,    0.2060461...],
                          [ 705.        ,    0.1977437...],
                          [ 710.        ,    0.1916846...],
                          [ 715.        ,    0.1861020...],
                          [ 720.        ,    0.1823908...],
                          [ 725.        ,    0.1807923...],
                          [ 730.        ,    0.1795571...],
                          [ 735.        ,    0.1785623...],
                          [ 740.        ,    0.1775758...],
                          [ 745.        ,    0.1771614...],
                          [ 750.        ,    0.1767431...],
                          [ 755.        ,    0.1764319...],
                          [ 760.        ,    0.1762597...],
                          [ 765.        ,    0.1762209...],
                          [ 770.        ,    0.1761803...],
                          [ 775.        ,    0.1761195...],
                          [ 780.        ,    0.1760763...]],
                         interpolator=SpragueInterpolator,
                         interpolator_kwargs={},
                         extrapolator=Extrapolator,
                         extrapolator_kwargs={...})
    >>> sd_to_XYZ_integration(sd, cmfs, illuminant) / 100
    ... # doctest: +ELLIPSIS
    array([ 0.2065436...,  0.1219996...,  0.0513764...])
    """

    RGB = to_domain_1(RGB)

    sd = SpectralDistribution(
        np.dot(RGB, np.transpose(basis_functions.values)),
        basis_functions.wavelengths,
    )
    sd.name = f"{RGB} (RGB) - Mallett (2019)"

    return sd
