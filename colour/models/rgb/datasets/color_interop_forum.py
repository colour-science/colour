"""
Color Interop Forum - Recommendation
====================================

Define the *Color Interop Forum* (CIF) recommendation *RGB* colourspaces:

-   :attr:`colour.models.RGB_COLOURSPACE_LIN_REC709_SCENE`
-   :attr:`colour.models.RGB_COLOURSPACE_LIN_P3D65_SCENE`
-   :attr:`colour.models.RGB_COLOURSPACE_LIN_REC2020_SCENE`
-   :attr:`colour.models.RGB_COLOURSPACE_LIN_ADOBERGB_SCENE`
-   :attr:`colour.models.RGB_COLOURSPACE_LIN_CIEXYZD65_SCENE`
-   :attr:`colour.models.RGB_COLOURSPACE_SRGB_REC709_SCENE`
-   :attr:`colour.models.RGB_COLOURSPACE_G22_REC709_SCENE`
-   :attr:`colour.models.RGB_COLOURSPACE_G18_REC709_SCENE`
-   :attr:`colour.models.RGB_COLOURSPACE_SRGB_AP1_SCENE`
-   :attr:`colour.models.RGB_COLOURSPACE_G22_AP1_SCENE`
-   :attr:`colour.models.RGB_COLOURSPACE_SRGB_P3D65_SCENE`
-   :attr:`colour.models.RGB_COLOURSPACE_G22_ADOBERGB_SCENE`
-   :attr:`colour.models.RGB_COLOURSPACES_TEXTURE_ASSETS_AND_CG_RENDERING_CIF`

References
----------
-   :cite:`ASWFColorInteropForum2024` : ASWF Color Interop Forum. (2024). Color
    Space Encodings for Texture Assets and CG Rendering.
    https://docs.google.com/document/d/1IV3e_9gpTOS_EFYRv2YGDuhExa4wTaPYHW1HyV36qUU
"""

from __future__ import annotations

from functools import partial

import numpy as np

from colour.colorimetry import CCS_ILLUMINANTS
from colour.models.rgb import (
    RGB_Colourspace,
    eotf_inverse_sRGB,
    eotf_sRGB,
    gamma_function,
    linear_function,
)
from colour.models.rgb.datasets import (
    RGB_COLOURSPACE_ACES2065_1,
    RGB_COLOURSPACE_ACESCG,
    RGB_COLOURSPACE_ADOBE_RGB1998,
    RGB_COLOURSPACE_BT709,
    RGB_COLOURSPACE_BT2020,
    RGB_COLOURSPACE_DISPLAY_P3,
    RGB_COLOURSPACE_P3_D65,
)
from colour.utilities import LazyCanonicalMapping

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "RGB_COLOURSPACE_LIN_REC709_SCENE",
    "RGB_COLOURSPACE_LIN_P3D65_SCENE",
    "RGB_COLOURSPACE_LIN_REC2020_SCENE",
    "RGB_COLOURSPACE_LIN_ADOBERGB_SCENE",
    "RGB_COLOURSPACE_LIN_CIEXYZD65_SCENE",
    "RGB_COLOURSPACE_SRGB_REC709_SCENE",
    "RGB_COLOURSPACE_G22_REC709_SCENE",
    "RGB_COLOURSPACE_G18_REC709_SCENE",
    "RGB_COLOURSPACE_SRGB_AP1_SCENE",
    "RGB_COLOURSPACE_G22_AP1_SCENE",
    "RGB_COLOURSPACE_SRGB_P3D65_SCENE",
    "RGB_COLOURSPACE_G22_ADOBERGB_SCENE",
    "RGB_COLOURSPACES_TEXTURE_ASSETS_AND_CG_RENDERING_CIF",
]

RGB_COLOURSPACE_LIN_REC709_SCENE: RGB_Colourspace = RGB_COLOURSPACE_BT709.copy()
RGB_COLOURSPACE_LIN_REC709_SCENE.cctf_encoding = linear_function
RGB_COLOURSPACE_LIN_REC709_SCENE.cctf_decoding = linear_function
RGB_COLOURSPACE_LIN_REC709_SCENE.name = "Linear Rec.709 (sRGB)"
RGB_COLOURSPACE_LIN_REC709_SCENE.__doc__ = """
*Linear Rec.709 (sRGB)* colourspace.

Still one of the most commonly used computer graphics (CG) rendering space.
However, it is a small gamut which means it is not possible to represent
certain colourful objects with only positive RGB values. This poses challenges
for rendering since negative values cause computational problems. It is
recommended that the industry move to wider gamut colourspaces such as *ACEScg*
for rendering.

References
----------
:cite:`ASWFColorInteropForum2024`
"""

RGB_COLOURSPACE_LIN_P3D65_SCENE: RGB_Colourspace = RGB_COLOURSPACE_P3_D65.copy()
RGB_COLOURSPACE_LIN_P3D65_SCENE.cctf_encoding = linear_function
RGB_COLOURSPACE_LIN_P3D65_SCENE.cctf_decoding = linear_function
RGB_COLOURSPACE_LIN_P3D65_SCENE.name = "Linear P3-D65"
RGB_COLOURSPACE_LIN_P3D65_SCENE.__doc__ = """
*Linear P3-D65* colourspace.

Not as good a rendering space as *ACEScg*, but better than *Linear Rec.709*
since it has a larger gamut. It may be easier for artists to use than *ACEScg*
since most modern monitors are able to show almost all of the *DCI-P3* gamut,
which is not true for *ACEScg*.

References
----------
:cite:`ASWFColorInteropForum2024`
"""

RGB_COLOURSPACE_LIN_REC2020_SCENE: RGB_Colourspace = RGB_COLOURSPACE_BT2020.copy()
RGB_COLOURSPACE_LIN_REC2020_SCENE.cctf_encoding = linear_function
RGB_COLOURSPACE_LIN_REC2020_SCENE.cctf_decoding = linear_function
RGB_COLOURSPACE_LIN_REC2020_SCENE.name = "Linear Rec.2020"
RGB_COLOURSPACE_LIN_REC2020_SCENE.__doc__ = """
*Linear Rec.2020* colourspace.

Its gamut is very similar to *ACEScg* and thus makes a good rendering
colourspace, though it is not often used for that purpose. The primaries are
on the spectrum locus and therefore most monitors are unable to display the
entire gamut.

References
----------
:cite:`ASWFColorInteropForum2024`
"""

RGB_COLOURSPACE_LIN_ADOBERGB_SCENE: RGB_Colourspace = (
    RGB_COLOURSPACE_ADOBE_RGB1998.copy()
)
RGB_COLOURSPACE_LIN_ADOBERGB_SCENE.cctf_encoding = linear_function
RGB_COLOURSPACE_LIN_ADOBERGB_SCENE.cctf_decoding = linear_function
RGB_COLOURSPACE_LIN_ADOBERGB_SCENE.name = "Linear AdobeRGB"
RGB_COLOURSPACE_LIN_ADOBERGB_SCENE.__doc__ = """
*Linear AdobeRGB* colourspace.

References
----------
:cite:`ASWFColorInteropForum2024`
"""

RGB_COLOURSPACE_LIN_CIEXYZD65_SCENE: RGB_Colourspace = RGB_Colourspace(
    "CIE XYZ-D65 - Scene-referred",
    np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    ),
    CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["E"],
    "E",
    cctf_encoding=linear_function,
    cctf_decoding=linear_function,
    use_derived_matrix_RGB_to_XYZ=True,
    use_derived_matrix_XYZ_to_RGB=True,
)
RGB_COLOURSPACE_LIN_CIEXYZD65_SCENE.__doc__ = """
*CIE XYZ-D65 - Scene-referred* colourspace.

This colourspace is not recommended for storing images. It is provided because
there are many situations in computer graphics where the goal is to simulate
some natural phenomena that are described by models or data that involves
*CIE* colorimetry such as physical sky models, daylight or blackbody curves,
spectral material models, diffraction effects, etc. When such data is used,
this colourspace provides a bridge to convert it into one of the other
colourspace encodings.

References
----------
:cite:`ASWFColorInteropForum2024`
"""

RGB_COLOURSPACE_SRGB_REC709_SCENE: RGB_Colourspace = RGB_COLOURSPACE_BT709.copy()
RGB_COLOURSPACE_SRGB_REC709_SCENE.cctf_encoding = eotf_inverse_sRGB
RGB_COLOURSPACE_SRGB_REC709_SCENE.cctf_decoding = eotf_sRGB
RGB_COLOURSPACE_SRGB_REC709_SCENE.name = "sRGB Encoded Rec.709 (sRGB)"
RGB_COLOURSPACE_SRGB_REC709_SCENE.__doc__ = """
*sRGB Encoded Rec.709 (sRGB)* colourspace.

References
----------
:cite:`ASWFColorInteropForum2024`
"""

RGB_COLOURSPACE_G22_REC709_SCENE: RGB_Colourspace = RGB_COLOURSPACE_BT709.copy()
RGB_COLOURSPACE_G22_REC709_SCENE.cctf_encoding = partial(
    gamma_function, exponent=1 / 2.2
)
RGB_COLOURSPACE_G22_REC709_SCENE.cctf_decoding = partial(gamma_function, exponent=2.2)
RGB_COLOURSPACE_G22_REC709_SCENE.name = "Gamma 2.2 Encoded Rec.709"
RGB_COLOURSPACE_G22_REC709_SCENE.__doc__ = """
*Gamma 2.2 Encoded Rec.709* colourspace.

References
----------
:cite:`ASWFColorInteropForum2024`
"""

RGB_COLOURSPACE_G18_REC709_SCENE: RGB_Colourspace = RGB_COLOURSPACE_BT709.copy()
RGB_COLOURSPACE_G18_REC709_SCENE.cctf_encoding = partial(
    gamma_function, exponent=1 / 1.8
)
RGB_COLOURSPACE_G18_REC709_SCENE.cctf_decoding = partial(gamma_function, exponent=1.8)
RGB_COLOURSPACE_G18_REC709_SCENE.name = "Gamma 1.8 Encoded Rec.709"
RGB_COLOURSPACE_G18_REC709_SCENE.__doc__ = """
*Gamma 1.8 Encoded Rec.709* colourspace.

References
----------
:cite:`ASWFColorInteropForum2024`
"""

RGB_COLOURSPACE_SRGB_AP1_SCENE: RGB_Colourspace = RGB_COLOURSPACE_ACESCG.copy()
RGB_COLOURSPACE_SRGB_AP1_SCENE.cctf_encoding = eotf_inverse_sRGB
RGB_COLOURSPACE_SRGB_AP1_SCENE.cctf_decoding = eotf_sRGB
RGB_COLOURSPACE_SRGB_AP1_SCENE.name = "sRGB Encoded AP1"
RGB_COLOURSPACE_SRGB_AP1_SCENE.__doc__ = """
*sRGB Encoded AP1* colourspace.

This colourspace is used in game engines that implement texture decoding on the
GPU using the sRGB piece-wise transfer function and when the working
colourspace is *ACEScg*.

References
----------
:cite:`ASWFColorInteropForum2024`
"""

RGB_COLOURSPACE_G22_AP1_SCENE: RGB_Colourspace = RGB_COLOURSPACE_ACESCG.copy()
RGB_COLOURSPACE_G22_AP1_SCENE.cctf_encoding = partial(gamma_function, exponent=1 / 2.2)
RGB_COLOURSPACE_G22_AP1_SCENE.cctf_decoding = partial(gamma_function, exponent=2.2)
RGB_COLOURSPACE_G22_AP1_SCENE.name = "Gamma 2.2 Encoded AP1"
RGB_COLOURSPACE_G22_AP1_SCENE.__doc__ = """
*Gamma 2.2 Encoded AP1* colourspace.

References
----------
:cite:`ASWFColorInteropForum2024`
"""

RGB_COLOURSPACE_SRGB_P3D65_SCENE: RGB_Colourspace = RGB_COLOURSPACE_DISPLAY_P3.copy()
RGB_COLOURSPACE_SRGB_P3D65_SCENE.name = "sRGB Encoded P3-D65"
RGB_COLOURSPACE_SRGB_P3D65_SCENE.__doc__ = """
*sRGB Encoded P3-D65* colourspace.

References
----------
:cite:`ASWFColorInteropForum2024`
"""

RGB_COLOURSPACE_G22_ADOBERGB_SCENE: RGB_Colourspace = (
    RGB_COLOURSPACE_ADOBE_RGB1998.copy()
)
RGB_COLOURSPACE_G22_ADOBERGB_SCENE.cctf_encoding = partial(
    gamma_function, exponent=1 / 2.2
)
RGB_COLOURSPACE_G22_ADOBERGB_SCENE.cctf_decoding = partial(gamma_function, exponent=2.2)
RGB_COLOURSPACE_G22_ADOBERGB_SCENE.name = "Gamma 2.2 Encoded AdobeRGB"
RGB_COLOURSPACE_G22_ADOBERGB_SCENE.__doc__ = """
*Gamma 2.2 Encoded AdobeRGB* colourspace.

References
----------
:cite:`ASWFColorInteropForum2024`
"""

RGB_COLOURSPACES_TEXTURE_ASSETS_AND_CG_RENDERING_CIF: LazyCanonicalMapping = LazyCanonicalMapping(  # noqa: E501
    {
        "ACEScg": RGB_COLOURSPACE_ACESCG,
        "ACES2065-1": RGB_COLOURSPACE_ACES2065_1,
        RGB_COLOURSPACE_LIN_REC709_SCENE.name: RGB_COLOURSPACE_LIN_REC709_SCENE,
        RGB_COLOURSPACE_LIN_P3D65_SCENE.name: RGB_COLOURSPACE_LIN_P3D65_SCENE,
        RGB_COLOURSPACE_LIN_REC2020_SCENE.name: RGB_COLOURSPACE_LIN_REC2020_SCENE,
        RGB_COLOURSPACE_LIN_ADOBERGB_SCENE.name: RGB_COLOURSPACE_LIN_ADOBERGB_SCENE,
        RGB_COLOURSPACE_LIN_CIEXYZD65_SCENE.name: RGB_COLOURSPACE_LIN_CIEXYZD65_SCENE,
        RGB_COLOURSPACE_SRGB_REC709_SCENE.name: RGB_COLOURSPACE_SRGB_REC709_SCENE,
        RGB_COLOURSPACE_G22_REC709_SCENE.name: RGB_COLOURSPACE_G22_REC709_SCENE,
        RGB_COLOURSPACE_G18_REC709_SCENE.name: RGB_COLOURSPACE_G18_REC709_SCENE,
        RGB_COLOURSPACE_SRGB_AP1_SCENE.name: RGB_COLOURSPACE_SRGB_AP1_SCENE,
        RGB_COLOURSPACE_G22_AP1_SCENE.name: RGB_COLOURSPACE_G22_AP1_SCENE,
        RGB_COLOURSPACE_SRGB_P3D65_SCENE.name: RGB_COLOURSPACE_SRGB_P3D65_SCENE,
        RGB_COLOURSPACE_G22_ADOBERGB_SCENE.name: RGB_COLOURSPACE_G22_ADOBERGB_SCENE,
        # Compact Names
        "lin_ap1_scene": RGB_COLOURSPACE_ACESCG,
        "lin_ap0_scene": RGB_COLOURSPACE_ACES2065_1,
        "lin_rec709_scene": RGB_COLOURSPACE_LIN_REC709_SCENE,
        "lin_p3d65_scene": RGB_COLOURSPACE_LIN_P3D65_SCENE,
        "lin_rec2020_scene": RGB_COLOURSPACE_LIN_REC2020_SCENE,
        "lin_adobergb_scene": RGB_COLOURSPACE_LIN_ADOBERGB_SCENE,
        "lin_ciexyzd65_scene": RGB_COLOURSPACE_LIN_CIEXYZD65_SCENE,
        "srgb_rec709_scene": RGB_COLOURSPACE_SRGB_REC709_SCENE,
        "g22_rec709_scene": RGB_COLOURSPACE_G22_REC709_SCENE,
        "g18_rec709_scene": RGB_COLOURSPACE_G18_REC709_SCENE,
        "srgb_ap1_scene": RGB_COLOURSPACE_SRGB_AP1_SCENE,
        "g22_ap1_scene": RGB_COLOURSPACE_G22_AP1_SCENE,
        "srgb_p3d65_scene": RGB_COLOURSPACE_SRGB_P3D65_SCENE,
        "g22_adobergb_scene": RGB_COLOURSPACE_G22_ADOBERGB_SCENE,
    }
)
RGB_COLOURSPACES_TEXTURE_ASSETS_AND_CG_RENDERING_CIF.__doc__ = """
*RGB* colourspace encodings for texture assets and computer graphics (CG)
rendering.

References
----------
:cite:`ASWFColorInteropForum2024`
"""
