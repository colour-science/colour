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

from colour.colorimetry import CCS_ILLUMINANTS, SpectralDistribution
from colour.hints import ArrayLike, NDArrayFloat
from colour.models import (
    RGB_Colourspace,
    RGB_COLOURSPACE_sRGB,
    XYZ_to_RGB,
)
from colour.recovery import SDS_SMITS1999
from colour.utilities import to_domain_1

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "PRIMARIES_SMITS1999",
    "WHITEPOINT_NAME_SMITS1999",
    "CCS_WHITEPOINT_SMITS1999",
    "RGB_COLOURSPACE_SMITS1999",
    "XYZ_to_RGB_Smits1999",
    "RGB_to_sd_Smits1999",
]

PRIMARIES_SMITS1999: NDArrayFloat = RGB_COLOURSPACE_sRGB.primaries
"""*Smits (1999)* method implementation colourspace primaries."""

WHITEPOINT_NAME_SMITS1999 = "E"
"""*Smits (1999)* method implementation colourspace whitepoint name."""

CCS_WHITEPOINT_SMITS1999: NDArrayFloat = CCS_ILLUMINANTS[
    "CIE 1931 2 Degree Standard Observer"
][WHITEPOINT_NAME_SMITS1999]
"""*Smits (1999)* method implementation colourspace whitepoint."""

RGB_COLOURSPACE_SMITS1999 = RGB_Colourspace(
    "Smits 1999",
    PRIMARIES_SMITS1999,
    CCS_WHITEPOINT_SMITS1999,
    WHITEPOINT_NAME_SMITS1999,
)
RGB_COLOURSPACE_sRGB.__doc__ = """
*Smits (1999)* colourspace.

References
----------
:cite:`Smits1999a`,
"""


def XYZ_to_RGB_Smits1999(XYZ: ArrayLike) -> NDArrayFloat:
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
    >>> import numpy as np
    >>> XYZ = np.array([0.21781186, 0.12541048, 0.04697113])
    >>> XYZ_to_RGB_Smits1999(XYZ)  # doctest: +ELLIPSIS
    array([ 0.4063959...,  0.0275289...,  0.0398219...])
    """

    return XYZ_to_RGB(XYZ, RGB_COLOURSPACE_SMITS1999)


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
    >>> import numpy as np
    >>> from colour import MSDS_CMFS, SDS_ILLUMINANTS, SpectralShape
    >>> from colour.colorimetry import sd_to_XYZ_integration
    >>> from colour.utilities import numpy_print_options
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> RGB = XYZ_to_RGB_Smits1999(XYZ)
    >>> cmfs = (
    ...     MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    ...     .copy()
    ...     .align(SpectralShape(360, 780, 10))
    ... )
    >>> illuminant = SDS_ILLUMINANTS["E"].copy().align(cmfs.shape)
    >>> sd = RGB_to_sd_Smits1999(RGB)
    >>> with numpy_print_options(suppress=True):
    ...     sd  # doctest: +ELLIPSIS
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
                         LinearInterpolator,
                         {},
                         Extrapolator,
                         {'method': 'Constant', 'left': None, 'right': None})
    >>> sd_to_XYZ_integration(sd, cmfs, illuminant) / 100  # doctest: +ELLIPSIS
    array([ 0.1894770...,  0.1126470...,  0.0474420...])
    """

    sd_white = SDS_SMITS1999["white"].copy()
    sd_cyan = SDS_SMITS1999["cyan"].copy()
    sd_magenta = SDS_SMITS1999["magenta"].copy()
    sd_yellow = SDS_SMITS1999["yellow"].copy()
    sd_red = SDS_SMITS1999["red"].copy()
    sd_green = SDS_SMITS1999["green"].copy()
    sd_blue = SDS_SMITS1999["blue"].copy()

    R, G, B = to_domain_1(RGB)
    sd = sd_white.copy() * 0
    sd.name = f"Smits (1999) - {RGB!r}"

    if R <= G and R <= B:
        sd += sd_white * R
        if G <= B:
            sd += sd_cyan * (G - R)
            sd += sd_blue * (B - G)
        else:
            sd += sd_cyan * (B - R)
            sd += sd_green * (G - B)
    elif G <= R and G <= B:
        sd += sd_white * G
        if R <= B:
            sd += sd_magenta * (R - G)
            sd += sd_blue * (B - R)
        else:
            sd += sd_magenta * (B - G)
            sd += sd_red * (R - B)
    else:
        sd += sd_white * B
        if R <= G:
            sd += sd_yellow * (R - B)
            sd += sd_green * (G - R)
        else:
            sd += sd_yellow * (G - B)
            sd += sd_red * (R - G)

    return sd
