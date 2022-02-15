"""
Smits (1999) - Reflectance Recovery
===================================

Defines the objects for reflectance recovery using *Smits (1999)* method.

References
----------
-   :cite:`Smits1999a` : Smits, B. (1999). An RGB-to-Spectrum Conversion for
    Reflectances. Journal of Graphics Tools, 4(4), 11-22.
    doi:10.1080/10867651.1999.10487511
"""

from __future__ import annotations

import numpy as np

from colour.colorimetry import CCS_ILLUMINANTS, SpectralDistribution
from colour.hints import ArrayLike, NDArray
from colour.models import (
    XYZ_to_RGB,
    normalised_primary_matrix,
    RGB_COLOURSPACE_sRGB,
)
from colour.recovery import SDS_SMITS1999
from colour.utilities import to_domain_1

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "PRIMARIES_SMITS1999",
    "CCS_WHITEPOINT_SMITS1999",
    "MATRIX_XYZ_TO_RGB_SMITS1999",
    "XYZ_to_RGB_Smits1999",
    "RGB_to_sd_Smits1999",
]

PRIMARIES_SMITS1999: NDArray = RGB_COLOURSPACE_sRGB.primaries
"""Current *Smits (1999)* method implementation colourspace primaries."""

CCS_WHITEPOINT_SMITS1999: NDArray = CCS_ILLUMINANTS[
    "CIE 1931 2 Degree Standard Observer"
]["E"]
"""Current *Smits (1999)* method implementation colourspace whitepoint."""

MATRIX_XYZ_TO_RGB_SMITS1999: NDArray = np.linalg.inv(
    normalised_primary_matrix(PRIMARIES_SMITS1999, CCS_WHITEPOINT_SMITS1999)
)
"""
Current *Smits (1999)* method implementation *RGB* colourspace to
*CIE XYZ* tristimulus values matrix.
"""


def XYZ_to_RGB_Smits1999(XYZ: ArrayLike) -> NDArray:
    """
    Convert from *CIE XYZ* tristimulus values to *RGB* colourspace with
    conditions required by the current *Smits (1999)* method implementation.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values.

    Returns
    -------
    :class:`numpy.ndarray`
        *RGB* colour array.

    Examples
    --------
    >>> XYZ = np.array([0.21781186, 0.12541048, 0.04697113])
    >>> XYZ_to_RGB_Smits1999(XYZ)  # doctest: +ELLIPSIS
    array([ 0.4063959...,  0.0275289...,  0.0398219...])
    """

    return XYZ_to_RGB(
        XYZ,
        CCS_WHITEPOINT_SMITS1999,
        CCS_WHITEPOINT_SMITS1999,
        MATRIX_XYZ_TO_RGB_SMITS1999,
    )


def RGB_to_sd_Smits1999(RGB: ArrayLike) -> SpectralDistribution:
    """
    Recover the spectral distribution of given *RGB* colourspace array using
    *Smits (1999)* method.

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
    | ``RGB``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Smits1999a`

    Examples
    --------
    >>> from colour import MSDS_CMFS, SDS_ILLUMINANTS, SpectralShape
    >>> from colour.colorimetry import sd_to_XYZ_integration
    >>> from colour.utilities import numpy_print_options
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> RGB = XYZ_to_RGB_Smits1999(XYZ)
    >>> cmfs = (
    ...     MSDS_CMFS['CIE 1931 2 Degree Standard Observer'].
    ...     copy().align(SpectralShape(360, 780, 10))
    ... )
    >>> illuminant = SDS_ILLUMINANTS['E'].copy().align(cmfs.shape)
    >>> sd = RGB_to_sd_Smits1999(RGB)
    >>> with numpy_print_options(suppress=True):
    ...     sd # doctest: +ELLIPSIS
    SpectralDistribution([[ 380.        ,    0.0787830...],
                          [ 417.7778    ,    0.0622018...],
                          [ 455.5556    ,    0.0446206...],
                          [ 493.3333    ,    0.0352220...],
                          [ 531.1111    ,    0.0324149...],
                          [ 568.8889    ,    0.0330105...],
                          [ 606.6667    ,    0.3207115...],
                          [ 644.4444    ,    0.3836164...],
                          [ 682.2222    ,    0.3836164...],
                          [ 720.        ,    0.3835649...]],
                         interpolator=LinearInterpolator,
                         interpolator_kwargs={},
                         extrapolator=Extrapolator,
                         extrapolator_kwargs={...})
    >>> sd_to_XYZ_integration(sd, cmfs, illuminant) / 100  # doctest: +ELLIPSIS
    array([ 0.1894770...,  0.1126470...,  0.0474420...])
    """

    white_sd = SDS_SMITS1999["white"].copy()
    cyan_sd = SDS_SMITS1999["cyan"].copy()
    magenta_sd = SDS_SMITS1999["magenta"].copy()
    yellow_sd = SDS_SMITS1999["yellow"].copy()
    red_sd = SDS_SMITS1999["red"].copy()
    green_sd = SDS_SMITS1999["green"].copy()
    blue_sd = SDS_SMITS1999["blue"].copy()

    R, G, B = to_domain_1(RGB)
    sd = white_sd.copy() * 0
    sd.name = f"Smits (1999) - {RGB!r}"

    if R <= G and R <= B:
        sd += white_sd * R
        if G <= B:
            sd += cyan_sd * (G - R)
            sd += blue_sd * (B - G)
        else:
            sd += cyan_sd * (B - R)
            sd += green_sd * (G - B)
    elif G <= R and G <= B:
        sd += white_sd * G
        if R <= B:
            sd += magenta_sd * (R - G)
            sd += blue_sd * (B - R)
        else:
            sd += magenta_sd * (B - G)
            sd += red_sd * (R - B)
    else:
        sd += white_sd * B
        if R <= G:
            sd += yellow_sd * (R - B)
            sd += green_sd * (G - R)
        else:
            sd += yellow_sd * (G - B)
            sd += red_sd * (R - G)

    return sd
