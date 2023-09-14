"""
Common RGB Colour Models Utilities
==================================

Defines various RGB colour models common utilities.
"""

from __future__ import annotations

from colour.colorimetry import CCS_ILLUMINANTS
from colour.hints import (
    ArrayLike,
    LiteralChromaticAdaptationTransform,
    NDArrayFloat,
)
from colour.models.rgb import RGB_COLOURSPACES, RGB_to_XYZ, XYZ_to_RGB

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "XYZ_to_sRGB",
    "sRGB_to_XYZ",
]


def XYZ_to_sRGB(
    XYZ: ArrayLike,
    illuminant: ArrayLike = CCS_ILLUMINANTS[
        "CIE 1931 2 Degree Standard Observer"
    ]["D65"],
    chromatic_adaptation_transform: LiteralChromaticAdaptationTransform
    | str
    | None = "CAT02",
    apply_cctf_encoding: bool = True,
) -> NDArrayFloat:
    """
    Convert from *CIE XYZ* tristimulus values to *sRGB* colourspace.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values.
    illuminant
        Source illuminant chromaticity coordinates.
    chromatic_adaptation_transform
        *Chromatic adaptation* transform.
    apply_cctf_encoding
        Whether to apply the *sRGB* encoding colour component transfer function
        / inverse electro-optical transfer function.

    Returns
    -------
    :class:`numpy.ndarray`
        *sRGB* colour array.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``RGB``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    Examples
    --------
    >>> import numpy as np
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> XYZ_to_sRGB(XYZ)  # doctest: +ELLIPSIS
    array([ 0.7057393...,  0.1924826...,  0.2235416...])
    """

    return XYZ_to_RGB(
        XYZ,
        RGB_COLOURSPACES["sRGB"],
        illuminant,
        chromatic_adaptation_transform,
        apply_cctf_encoding,
    )


def sRGB_to_XYZ(
    RGB: ArrayLike,
    illuminant: ArrayLike = CCS_ILLUMINANTS[
        "CIE 1931 2 Degree Standard Observer"
    ]["D65"],
    chromatic_adaptation_transform: LiteralChromaticAdaptationTransform
    | str
    | None = "CAT02",
    apply_cctf_decoding: bool = True,
) -> NDArrayFloat:
    """
    Convert from *sRGB* colourspace to *CIE XYZ* tristimulus values.

    Parameters
    ----------
    RGB
        *sRGB* colourspace array.
    illuminant
        Source illuminant chromaticity coordinates.
    chromatic_adaptation_transform
        *Chromatic adaptation* transform.
    apply_cctf_decoding
        Whether to apply the *sRGB* decoding colour component transfer function
        / electro-optical transfer function.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ* tristimulus values.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``RGB``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    Examples
    --------
    >>> import numpy as np
    >>> RGB = np.array([0.70573936, 0.19248266, 0.22354169])
    >>> sRGB_to_XYZ(RGB)  # doctest: +ELLIPSIS
    array([ 0.2065429...,  0.1219794...,  0.0513714...])
    """

    return RGB_to_XYZ(
        RGB,
        RGB_COLOURSPACES["sRGB"],
        illuminant,
        chromatic_adaptation_transform,
        apply_cctf_decoding,
    )
