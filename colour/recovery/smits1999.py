"""
Smits (1999) - Reflectance Recovery
===================================

Define objects for reflectance recovery using the *Smits (1999)* method.

References
----------
-   :cite:`Smits1999a` : Smits, B. (1999). An RGB-to-Spectrum Conversion for
    Reflectances. Journal of Graphics Tools, 4(4), 11-22.
    doi:10.1080/10867651.1999.10487511
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from colour.colorimetry import (
    CCS_ILLUMINANTS,
    MultiSpectralDistributions,
    SpectralDistribution,
)
from colour.models import RGB_Colourspace, RGB_COLOURSPACE_sRGB, XYZ_to_RGB
from colour.recovery import MSDS_SMITS1999
from colour.utilities import (
    as_float_array,
    optional,
    to_domain_1,
    tsplit,
)

if TYPE_CHECKING:
    from colour.hints import ArrayLike, Domain1, NDArrayFloat, Range1

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
    "RGB_to_msds_Smits1999",
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
RGB_COLOURSPACE_SMITS1999.__doc__ = """
*Smits (1999)* colourspace.

References
----------
:cite:`Smits1999a`,
"""


def XYZ_to_RGB_Smits1999(XYZ: Domain1) -> Range1:
    """
    Convert from *CIE XYZ* tristimulus values to *RGB* colourspace using
    the conditions required by the current *Smits (1999)* method
    implementation.

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
    >>> XYZ_to_RGB_Smits1999(XYZ)  # doctest: +ELLIPSIS
    array([ 0.4063959...,  0.0275289...,  0.0398219...])
    """

    return XYZ_to_RGB(XYZ, RGB_COLOURSPACE_SMITS1999)


def RGB_to_msds_Smits1999(
    RGB: ArrayLike,
    basis: MultiSpectralDistributions | None = None,
) -> NDArrayFloat:
    """
    Recover spectral values from *RGB* colourspace array using the
    *Smits (1999)* decomposition algorithm.

    This is a vectorised implementation supporting multi-dimensional arrays.

    Parameters
    ----------
    RGB
        *RGB* colourspace array to recover spectral values from. The last
        dimension must be size 3.
    basis
        Multi-spectral distributions basis with signals: white, cyan, magenta,
        yellow, red, green, blue. Defaults to :attr:`MSDS_SMITS1999`.

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

    References
    ----------
    :cite:`Smits1999a`

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
    >>> RGB_to_msds_Smits1999(RGB).shape
    (3, 10)
    >>> RGB_to_msds_Smits1999(RGB)[0, 0]  # doctest: +ELLIPSIS
    0.0829...
    """

    basis = optional(basis, MSDS_SMITS1999)

    RGB = to_domain_1(as_float_array(RGB))
    shape = RGB.shape
    RGB = np.atleast_2d(RGB.reshape(-1, 3))

    R, G, B = tsplit(RGB)

    labels = list(basis.labels)
    white = basis.values[:, labels.index("white")]
    cyan = basis.values[:, labels.index("cyan")]
    magenta = basis.values[:, labels.index("magenta")]
    yellow = basis.values[:, labels.index("yellow")]
    red = basis.values[:, labels.index("red")]
    green = basis.values[:, labels.index("green")]
    blue = basis.values[:, labels.index("blue")]

    R = R[..., np.newaxis]
    G = G[..., np.newaxis]
    B = B[..., np.newaxis]

    # if R <= G and R <= B:
    #     sd += white * R
    #     if G <= B:
    #         sd += cyan * (G - R) + blue * (B - G)
    #     else:
    #         sd += cyan * (B - R) + green * (G - B)
    R_le_G_le_B = (R <= G) & (R <= B) & (G <= B)
    R_le_B_lt_G = (R <= G) & (R <= B) & (G > B)

    # elif G <= R and G <= B:
    #     sd += white * G
    #     if R <= B:
    #         sd += magenta * (R - G) + blue * (B - R)
    #     else:
    #         sd += magenta * (B - G) + red * (R - B)
    G_lt_R_le_B = (G < R) & (G <= B) & (R <= B)
    G_le_B_lt_R = (G < R) & (G <= B) & (R > B)

    # else:  # B < R and B < G
    #     sd += white * B
    #     if R <= G:
    #         sd += yellow * (R - B) + green * (G - R)
    #     else:
    #         sd += yellow * (G - B) + red * (R - G)
    B_lt_R_le_G = (B < R) & (B < G) & (R <= G)
    B_lt_G_lt_R = (B < R) & (B < G) & (R > G)

    spectra = np.select(
        [R_le_G_le_B, R_le_B_lt_G, G_lt_R_le_B, G_le_B_lt_R, B_lt_R_le_G, B_lt_G_lt_R],
        [
            white * R + cyan * (G - R) + blue * (B - G),
            white * R + cyan * (B - R) + green * (G - B),
            white * G + magenta * (R - G) + blue * (B - R),
            white * G + magenta * (B - G) + red * (R - B),
            white * B + yellow * (R - B) + green * (G - R),
            white * B + yellow * (G - B) + red * (R - G),
        ],
    )

    return np.reshape(spectra, [*list(shape[:-1]), len(white)])


def RGB_to_sd_Smits1999(
    RGB: Domain1,
    basis: MultiSpectralDistributions | None = None,
    name: str | None = None,
) -> SpectralDistribution:
    """
    Generate a spectral distribution from *RGB* values using the
    *Smits (1999)* decomposition algorithm.

    Parameters
    ----------
    RGB
        *RGB* colourspace array to recover the spectral distribution from.
    basis
        Multi-spectral distributions basis with signals: white, cyan, magenta,
        yellow, red, green, blue. Defaults to :attr:`MSDS_SMITS1999`.
    name
        Name for the resulting spectral distribution.

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

    basis = optional(basis, MSDS_SMITS1999)
    name = optional(name, f"Smits (1999) - {RGB!r}")

    values = RGB_to_msds_Smits1999(RGB, basis)

    return SpectralDistribution(
        values,
        basis.wavelengths,
        name=name,
        interpolator=basis.interpolator,
        interpolator_kwargs=basis.interpolator_kwargs,
    )
