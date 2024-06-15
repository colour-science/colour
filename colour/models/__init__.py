import sys

from colour.utilities import copy_definition
from colour.utilities.deprecation import ModuleAPI, build_API_changes
from colour.utilities.documentation import is_documentation_building

from colour.hints import Any

from .common import (
    COLOURSPACE_MODELS,
    COLOURSPACE_MODELS_AXIS_LABELS,
    COLOURSPACE_MODELS_DOMAIN_RANGE_SCALE_1_TO_REFERENCE,
    Jab_to_JCh,
    JCh_to_Jab,
    XYZ_to_Iab,
    Iab_to_XYZ,
)
from .cam02_ucs import (
    JMh_CIECAM02_to_CAM02LCD,
    CAM02LCD_to_JMh_CIECAM02,
    JMh_CIECAM02_to_CAM02SCD,
    CAM02SCD_to_JMh_CIECAM02,
    JMh_CIECAM02_to_CAM02UCS,
    CAM02UCS_to_JMh_CIECAM02,
    XYZ_to_CAM02LCD,
    CAM02LCD_to_XYZ,
    XYZ_to_CAM02SCD,
    CAM02SCD_to_XYZ,
    XYZ_to_CAM02UCS,
    CAM02UCS_to_XYZ,
)
from .cam16_ucs import (
    JMh_CAM16_to_CAM16LCD,
    CAM16LCD_to_JMh_CAM16,
    JMh_CAM16_to_CAM16SCD,
    CAM16SCD_to_JMh_CAM16,
    JMh_CAM16_to_CAM16UCS,
    CAM16UCS_to_JMh_CAM16,
    XYZ_to_CAM16LCD,
    CAM16LCD_to_XYZ,
    XYZ_to_CAM16SCD,
    CAM16SCD_to_XYZ,
    XYZ_to_CAM16UCS,
    CAM16UCS_to_XYZ,
)
from .cie_xyy import (
    XYZ_to_xyY,
    xyY_to_XYZ,
    xy_to_xyY,
    xyY_to_xy,
    xy_to_XYZ,
    XYZ_to_xy,
)
from .cie_lab import XYZ_to_Lab, Lab_to_XYZ
from .cie_luv import (
    XYZ_to_Luv,
    Luv_to_XYZ,
    Luv_to_uv,
    uv_to_Luv,
    Luv_uv_to_xy,
    xy_to_Luv_uv,
    XYZ_to_CIE1976UCS,
    CIE1976UCS_to_XYZ,
)
from .cie_ucs import (
    XYZ_to_UCS,
    UCS_to_XYZ,
    UCS_to_uv,
    uv_to_UCS,
    UCS_uv_to_xy,
    xy_to_UCS_uv,
    XYZ_to_CIE1960UCS,
    CIE1960UCS_to_XYZ,
)
from .cie_uvw import XYZ_to_UVW, UVW_to_XYZ
from .din99 import Lab_to_DIN99, DIN99_to_Lab, XYZ_to_DIN99, DIN99_to_XYZ
from .hdr_cie_lab import (
    HDR_CIELAB_METHODS,
    XYZ_to_hdr_CIELab,
    hdr_CIELab_to_XYZ,
)
from .hunter_lab import (
    XYZ_to_K_ab_HunterLab1966,
    XYZ_to_Hunter_Lab,
    Hunter_Lab_to_XYZ,
)
from .hunter_rdab import XYZ_to_Hunter_Rdab, Hunter_Rdab_to_XYZ
from .icacb import XYZ_to_ICaCb, ICaCb_to_XYZ
from .igpgtg import XYZ_to_IgPgTg, IgPgTg_to_XYZ
from .ipt import XYZ_to_IPT, IPT_to_XYZ, IPT_hue_angle
from .jzazbz import (
    IZAZBZ_METHODS,
    XYZ_to_Izazbz,
    XYZ_to_Jzazbz,
    Izazbz_to_XYZ,
    Jzazbz_to_XYZ,
)
from .hdr_ipt import HDR_IPT_METHODS, XYZ_to_hdr_IPT, hdr_IPT_to_XYZ
from .oklab import XYZ_to_Oklab, Oklab_to_XYZ
from .osa_ucs import XYZ_to_OSA_UCS, OSA_UCS_to_XYZ
from .prolab import XYZ_to_ProLab, ProLab_to_XYZ
from .ragoo2021 import XYZ_to_IPT_Ragoo2021, IPT_Ragoo2021_to_XYZ
from .yrg import LMS_to_Yrg, Yrg_to_LMS, XYZ_to_Yrg, Yrg_to_XYZ
from .datasets import (
    DATA_MACADAM_1942_ELLIPSES,
    CCS_ILLUMINANT_POINTER_GAMUT,
    DATA_POINTER_GAMUT_VOLUME,
    CCS_POINTER_GAMUT_BOUNDARY,
)
from .rgb import (
    normalised_primary_matrix,
    chromatically_adapted_primaries,
    primaries_whitepoint,
    RGB_luminance_equation,
    RGB_luminance,
)
from .rgb import RGB_Colourspace
from .rgb import XYZ_to_RGB, RGB_to_XYZ
from .rgb import matrix_RGB_to_RGB, RGB_to_RGB
from .rgb import (
    CV_range,
    legal_to_full,
    full_to_legal,
    gamma_function,
    log_encoding_ACESproxy,
    log_decoding_ACESproxy,
    log_encoding_ACEScc,
    log_decoding_ACEScc,
    log_encoding_ACEScct,
    log_decoding_ACEScct,
    log_encoding_AppleLogProfile,
    log_decoding_AppleLogProfile,
    oetf_ARIBSTDB67,
    oetf_inverse_ARIBSTDB67,
    log_encoding_ARRILogC3,
    log_decoding_ARRILogC3,
    log_encoding_ARRILogC4,
    log_decoding_ARRILogC4,
    oetf_BlackmagicFilmGeneration5,
    oetf_inverse_BlackmagicFilmGeneration5,
    log_encoding_CanonLog,
    log_decoding_CanonLog,
    log_encoding_CanonLog2,
    log_decoding_CanonLog2,
    log_encoding_CanonLog3,
    log_decoding_CanonLog3,
    log_encoding_Cineon,
    log_decoding_Cineon,
    oetf_DaVinciIntermediate,
    oetf_inverse_DaVinciIntermediate,
    eotf_inverse_DCDM,
    eotf_DCDM,
    eotf_inverse_DICOMGSDF,
    eotf_DICOMGSDF,
    log_encoding_DJIDLog,
    log_decoding_DJIDLog,
    exponent_function_basic,
    exponent_function_monitor_curve,
    log_encoding_FilmicPro6,
    log_decoding_FilmicPro6,
    log_encoding_FilmLightTLog,
    log_decoding_FilmLightTLog,
    log_encoding_Protune,
    log_decoding_Protune,
    oetf_BT2020,
    oetf_inverse_BT2020,
    oetf_BT601,
    oetf_inverse_BT601,
    oetf_BT709,
    oetf_inverse_BT709,
    oetf_BT1361,
    oetf_inverse_BT1361,
    eotf_inverse_BT1886,
    eotf_BT1886,
    eotf_inverse_ST2084,
    eotf_ST2084,
    oetf_BT2100_PQ,
    oetf_inverse_BT2100_PQ,
    eotf_BT2100_PQ,
    eotf_inverse_BT2100_PQ,
    ootf_BT2100_PQ,
    ootf_inverse_BT2100_PQ,
    oetf_BT2100_HLG,
    oetf_inverse_BT2100_HLG,
    BT2100_HLG_EOTF_METHODS,
    eotf_BT2100_HLG,
    BT2100_HLG_EOTF_INVERSE_METHODS,
    eotf_inverse_BT2100_HLG,
    BT2100_HLG_OOTF_METHODS,
    ootf_BT2100_HLG,
    BT2100_HLG_OOTF_INVERSE_METHODS,
    ootf_inverse_BT2100_HLG,
    oetf_H273_Log,
    oetf_inverse_H273_Log,
    oetf_H273_LogSqrt,
    oetf_inverse_H273_LogSqrt,
    oetf_H273_IEC61966_2,
    oetf_inverse_H273_IEC61966_2,
    eotf_H273_ST428_1,
    eotf_inverse_H273_ST428_1,
    linear_function,
    logarithmic_function_basic,
    logarithmic_function_quasilog,
    logarithmic_function_camera,
    log_encoding_Log2,
    log_decoding_Log2,
    log_encoding_Panalog,
    log_decoding_Panalog,
    log_encoding_VLog,
    log_decoding_VLog,
    log_encoding_FLog,
    log_decoding_FLog,
    log_encoding_FLog2,
    log_decoding_FLog2,
    log_encoding_LLog,
    log_decoding_LLog,
    log_encoding_NLog,
    log_decoding_NLog,
    log_encoding_PivotedLog,
    log_decoding_PivotedLog,
    log_encoding_REDLog,
    log_decoding_REDLog,
    log_encoding_REDLogFilm,
    log_decoding_REDLogFilm,
    LOG3G10_ENCODING_METHODS,
    LOG3G10_DECODING_METHODS,
    log_encoding_Log3G10,
    log_decoding_Log3G10,
    log_encoding_Log3G12,
    log_decoding_Log3G12,
    cctf_encoding_ROMMRGB,
    cctf_decoding_ROMMRGB,
    cctf_encoding_ProPhotoRGB,
    cctf_decoding_ProPhotoRGB,
    cctf_encoding_RIMMRGB,
    cctf_decoding_RIMMRGB,
    log_encoding_ERIMMRGB,
    log_decoding_ERIMMRGB,
    oetf_SMPTE240M,
    eotf_SMPTE240M,
    log_encoding_SLog,
    log_decoding_SLog,
    log_encoding_SLog2,
    log_decoding_SLog2,
    log_encoding_SLog3,
    log_decoding_SLog3,
    eotf_inverse_sRGB,
    eotf_sRGB,
    log_encoding_ViperLog,
    log_decoding_ViperLog,
)
from .rgb import (
    LOG_ENCODINGS,
    log_encoding,
    LOG_DECODINGS,
    log_decoding,
    OETFS,
    oetf,
    OETF_INVERSES,
    oetf_inverse,
    EOTFS,
    eotf,
    EOTF_INVERSES,
    eotf_inverse,
    CCTF_ENCODINGS,
    cctf_encoding,
    CCTF_DECODINGS,
    cctf_decoding,
    OOTFS,
    ootf,
    OOTF_INVERSES,
    ootf_inverse,
)
from .rgb import (
    RGB_COLOURSPACES,
    RGB_COLOURSPACE_ACES2065_1,
    RGB_COLOURSPACE_ACESCC,
    RGB_COLOURSPACE_ACESCCT,
    RGB_COLOURSPACE_ACESPROXY,
    RGB_COLOURSPACE_ACESCG,
    RGB_COLOURSPACE_ADOBE_RGB1998,
    RGB_COLOURSPACE_ADOBE_WIDE_GAMUT_RGB,
    RGB_COLOURSPACE_ARRI_WIDE_GAMUT_3,
    RGB_COLOURSPACE_ARRI_WIDE_GAMUT_4,
    RGB_COLOURSPACE_APPLE_RGB,
    RGB_COLOURSPACE_BEST_RGB,
    RGB_COLOURSPACE_BETA_RGB,
    RGB_COLOURSPACE_BLACKMAGIC_WIDE_GAMUT,
    RGB_COLOURSPACE_BT470_525,
    RGB_COLOURSPACE_BT470_625,
    RGB_COLOURSPACE_BT709,
    RGB_COLOURSPACE_BT2020,
    RGB_COLOURSPACE_CIE_RGB,
    RGB_COLOURSPACE_CINEMA_GAMUT,
    RGB_COLOURSPACE_COLOR_MATCH_RGB,
    RGB_COLOURSPACE_DAVINCI_WIDE_GAMUT,
    RGB_COLOURSPACE_DCDM_XYZ,
    RGB_COLOURSPACE_DCI_P3,
    RGB_COLOURSPACE_DCI_P3_P,
    RGB_COLOURSPACE_DISPLAY_P3,
    RGB_COLOURSPACE_DJI_D_GAMUT,
    RGB_COLOURSPACE_DON_RGB_4,
    RGB_COLOURSPACE_EBU_3213_E,
    RGB_COLOURSPACE_ECI_RGB_V2,
    RGB_COLOURSPACE_EKTA_SPACE_PS_5,
    RGB_COLOURSPACE_FILMLIGHT_E_GAMUT,
    RGB_COLOURSPACE_H273_GENERIC_FILM,
    RGB_COLOURSPACE_H273_22_UNSPECIFIED,
    RGB_COLOURSPACE_PROTUNE_NATIVE,
    RGB_COLOURSPACE_MAX_RGB,
    RGB_COLOURSPACE_N_GAMUT,
    RGB_COLOURSPACE_P3_D65,
    RGB_COLOURSPACE_PAL_SECAM,
    RGB_COLOURSPACE_RED_COLOR,
    RGB_COLOURSPACE_RED_COLOR_2,
    RGB_COLOURSPACE_RED_COLOR_3,
    RGB_COLOURSPACE_RED_COLOR_4,
    RGB_COLOURSPACE_DRAGON_COLOR,
    RGB_COLOURSPACE_DRAGON_COLOR_2,
    RGB_COLOURSPACE_RED_WIDE_GAMUT_RGB,
    RGB_COLOURSPACE_ROMM_RGB,
    RGB_COLOURSPACE_RIMM_RGB,
    RGB_COLOURSPACE_ERIMM_RGB,
    RGB_COLOURSPACE_PROPHOTO_RGB,
    RGB_COLOURSPACE_PLASA_ANSI_E154,
    RGB_COLOURSPACE_RUSSELL_RGB,
    RGB_COLOURSPACE_SHARP_RGB,
    RGB_COLOURSPACE_SMPTE_240M,
    RGB_COLOURSPACE_SMPTE_C,
    RGB_COLOURSPACE_NTSC1953,
    RGB_COLOURSPACE_NTSC1987,
    RGB_COLOURSPACE_S_GAMUT,
    RGB_COLOURSPACE_S_GAMUT3,
    RGB_COLOURSPACE_S_GAMUT3_CINE,
    RGB_COLOURSPACE_VENICE_S_GAMUT3,
    RGB_COLOURSPACE_VENICE_S_GAMUT3_CINE,
    RGB_COLOURSPACE_sRGB,
    RGB_COLOURSPACE_V_GAMUT,
    RGB_COLOURSPACE_XTREME_RGB,
    RGB_COLOURSPACE_F_GAMUT,
)

from .rgb import XYZ_to_sRGB, sRGB_to_XYZ
from .rgb import (
    RGB_to_HSV,
    HSV_to_RGB,
    RGB_to_HSL,
    HSL_to_RGB,
    RGB_to_HCL,
    HCL_to_RGB,
)
from .rgb import RGB_to_CMY, CMY_to_RGB, CMY_to_CMYK, CMYK_to_CMY
from .rgb import RGB_to_IHLS, IHLS_to_RGB
from .rgb import RGB_to_Prismatic, Prismatic_to_RGB
from .rgb import (
    WEIGHTS_YCBCR,
    matrix_YCbCr,
    offset_YCbCr,
    RGB_to_YCbCr,
    YCbCr_to_RGB,
    RGB_to_YcCbcCrc,
    YcCbcCrc_to_RGB,
)
from .rgb import RGB_to_YCoCg, YCoCg_to_RGB
from .rgb import RGB_to_ICtCp, ICtCp_to_RGB, XYZ_to_ICtCp, ICtCp_to_XYZ
from .rgb import (
    COLOUR_PRIMARIES_ITUTH273,
    TRANSFER_CHARACTERISTICS_ITUTH273,
    MATRIX_COEFFICIENTS_ITUTH273,
    describe_video_signal_colour_primaries,
    describe_video_signal_transfer_characteristics,
    describe_video_signal_matrix_coefficients,
)

__all__ = []

# Programmatically defining the colourspace models polar conversions.
COLOURSPACE_MODELS_POLAR_CONVERSIONS = (
    ("Lab", "LCHab"),
    ("Luv", "LCHuv"),
    ("hdr_CIELab", "hdr_CIELCHab"),
    ("Hunter_Lab", "Hunter_LCHab"),
    ("Hunter_Rdab", "Hunter_RdCHab"),
    ("ICaCb", "ICHab"),
    ("ICtCp", "ICHtp"),
    ("IgPgTg", "IgCHpt"),
    ("IPT", "ICH"),
    ("Izazbz", "IzCHab"),
    ("Jzazbz", "JzCHab"),
    ("hdr_IPT", "hdr_ICH"),
    ("Oklab", "Oklch"),
    ("ProLab", "ProLCHab"),
    ("IPT_Ragoo2021", "ICH_Ragoo2021"),
)

_DOCSTRING_JAB_TO_JCH = """
Convert from *{Jab}* colourspace to *{JCh}* colourspace.

This is a convenient definition wrapping :func:`colour.models.Jab_to_JCh`
definition.

Parameters
----------
Jab
    *{Jab}* colourspace array.

Returns
-------
:class:`numpy.ndarray`
    *{JCh}* colourspace array.

Notes
-----
+------------+-----------------------+-----------------+
| **Domain** | **Scale - Reference** | **Scale - 1**   |
+============+=======================+=================+
| ``Jab``    | ``J`` : [0, 100]      | ``J`` : [0, 1]  |
|            |                       |                 |
|            | ``a`` : [-100, 100]   | ``a`` : [-1, 1] |
|            |                       |                 |
|            | ``b`` : [-100, 100]   | ``b`` : [-1, 1] |
+------------+-----------------------+-----------------+

+------------+-----------------------+-----------------+
| **Range**  | **Scale - Reference** | **Scale - 1**   |
+============+=======================+=================+
| ``JCh``    | ``J``  : [0, 100]     | ``J`` : [0, 1]  |
|            |                       |                 |
|            | ``C``  : [0, 100]     | ``C`` : [0, 1]  |
|            |                       |                 |
|            | ``h`` : [0, 360]      | ``h`` : [0, 1]  |
+------------+-----------------------+-----------------+
"""

_DOCSTRING_JCH_TO_JAB = """
Convert from *{JCh}* colourspace to *{Jab}* colourspace.

This is a convenient definition wrapping :func:`colour.models.JCh_to_Jab`
definition.

Parameters
----------
JCh
    *{JCh}* colourspace array.

Returns
-------
:class:`numpy.ndarray`
    *{Jab}* colourspace array.

Notes
-----
+-------------+-----------------------+-----------------+
| **Domain**  | **Scale - Reference** | **Scale - 1**   |
+=============+=======================+=================+
| ``JCh``     | ``J``  : [0, 100]     | ``J``  : [0, 1] |
|             |                       |                 |
|             | ``C``  : [0, 100]     | ``C``  : [0, 1] |
|             |                       |                 |
|             | ``h`` : [0, 360]      | ``h`` : [0, 1]  |
+-------------+-----------------------+-----------------+

+-------------+-----------------------+-----------------+
| **Range**   | **Scale - Reference** | **Scale - 1**   |
+=============+=======================+=================+
| ``Jab``     | ``J`` : [0, 100]      | ``J`` : [0, 1]  |
|             |                       |                 |
|             | ``a`` : [-100, 100]   | ``a`` : [-1, 1] |
|             |                       |                 |
|             | ``b`` : [-100, 100]   | ``b`` : [-1, 1] |
+-------------+-----------------------+-----------------+
"""

for _Jab, _JCh in COLOURSPACE_MODELS_POLAR_CONVERSIONS:
    name = f"{_Jab}_to_{_JCh}"
    _callable = copy_definition(Jab_to_JCh, name)
    _callable.__doc__ = _DOCSTRING_JAB_TO_JCH.format(Jab=_Jab, JCh=_JCh)
    _module = sys.modules["colour.models"]
    setattr(_module, name, _callable)
    __all__.append(name)

    name = f"{_JCh}_to_{_Jab}"
    _callable = copy_definition(JCh_to_Jab, name)
    _callable.__doc__ = _DOCSTRING_JCH_TO_JAB.format(JCh=_JCh, Jab=_Jab)
    _module = sys.modules["colour.models"]
    setattr(_module, name, _callable)
    __all__.append(name)

del _DOCSTRING_JAB_TO_JCH, _DOCSTRING_JCH_TO_JAB, _JCh, _Jab, _callable, _module

__all__ += ["COLOURSPACE_MODELS_POLAR"]

__all__ += [
    "COLOURSPACE_MODELS",
    "COLOURSPACE_MODELS_AXIS_LABELS",
    "COLOURSPACE_MODELS_DOMAIN_RANGE_SCALE_1_TO_REFERENCE",
    "Jab_to_JCh",
    "JCh_to_Jab",
    "XYZ_to_Iab",
    "Iab_to_XYZ",
]
__all__ += [
    "JMh_CIECAM02_to_CAM02LCD",
    "CAM02LCD_to_JMh_CIECAM02",
    "JMh_CIECAM02_to_CAM02SCD",
    "CAM02SCD_to_JMh_CIECAM02",
    "JMh_CIECAM02_to_CAM02UCS",
    "CAM02UCS_to_JMh_CIECAM02",
    "XYZ_to_CAM02LCD",
    "CAM02LCD_to_XYZ",
    "XYZ_to_CAM02SCD",
    "CAM02SCD_to_XYZ",
    "XYZ_to_CAM02UCS",
    "CAM02UCS_to_XYZ",
]
__all__ += [
    "JMh_CAM16_to_CAM16LCD",
    "CAM16LCD_to_JMh_CAM16",
    "JMh_CAM16_to_CAM16SCD",
    "CAM16SCD_to_JMh_CAM16",
    "JMh_CAM16_to_CAM16UCS",
    "CAM16UCS_to_JMh_CAM16",
    "XYZ_to_CAM16LCD",
    "CAM16LCD_to_XYZ",
    "XYZ_to_CAM16SCD",
    "CAM16SCD_to_XYZ",
    "XYZ_to_CAM16UCS",
    "CAM16UCS_to_XYZ",
]
__all__ += [
    "XYZ_to_xyY",
    "xyY_to_XYZ",
    "xy_to_xyY",
    "xyY_to_xy",
    "xy_to_XYZ",
    "XYZ_to_xy",
]
__all__ += [
    "XYZ_to_Lab",
    "Lab_to_XYZ",
]
__all__ += [
    "XYZ_to_Luv",
    "Luv_to_XYZ",
    "Luv_to_uv",
    "uv_to_Luv",
    "Luv_uv_to_xy",
    "xy_to_Luv_uv",
    "XYZ_to_CIE1976UCS",
    "CIE1976UCS_to_XYZ",
]
__all__ += [
    "XYZ_to_UCS",
    "UCS_to_XYZ",
    "UCS_to_uv",
    "uv_to_UCS",
    "UCS_uv_to_xy",
    "xy_to_UCS_uv",
    "XYZ_to_CIE1960UCS",
    "CIE1960UCS_to_XYZ",
]
__all__ += [
    "XYZ_to_UVW",
    "UVW_to_XYZ",
]
__all__ += [
    "Lab_to_DIN99",
    "DIN99_to_Lab",
    "XYZ_to_DIN99",
    "DIN99_to_XYZ",
]
__all__ += [
    "HDR_CIELAB_METHODS",
    "XYZ_to_hdr_CIELab",
    "hdr_CIELab_to_XYZ",
]
__all__ += [
    "XYZ_to_K_ab_HunterLab1966",
    "XYZ_to_Hunter_Lab",
    "Hunter_Lab_to_XYZ",
    "XYZ_to_Hunter_Rdab",
]
__all__ += [
    "XYZ_to_Hunter_Rdab",
    "Hunter_Rdab_to_XYZ",
]
__all__ += [
    "XYZ_to_ICaCb",
    "ICaCb_to_XYZ",
]
__all__ += [
    "XYZ_to_IgPgTg",
    "IgPgTg_to_XYZ",
]
__all__ += [
    "XYZ_to_IPT",
    "IPT_to_XYZ",
    "IPT_hue_angle",
]
__all__ += [
    "IZAZBZ_METHODS",
    "XYZ_to_Izazbz",
    "XYZ_to_Jzazbz",
    "Izazbz_to_XYZ",
    "Jzazbz_to_XYZ",
]
__all__ += [
    "XYZ_to_IPT_Ragoo2021",
    "IPT_Ragoo2021_to_XYZ",
]
__all__ += [
    "LMS_to_Yrg",
    "Yrg_to_LMS",
    "XYZ_to_Yrg",
    "Yrg_to_XYZ",
]
__all__ += [
    "HDR_IPT_METHODS",
    "XYZ_to_hdr_IPT",
    "hdr_IPT_to_XYZ",
]
__all__ += [
    "XYZ_to_Oklab",
    "Oklab_to_XYZ",
]
__all__ += [
    "XYZ_to_OSA_UCS",
    "OSA_UCS_to_XYZ",
]
__all__ += [
    "XYZ_to_ProLab",
    "ProLab_to_XYZ",
]
__all__ += [
    "DATA_MACADAM_1942_ELLIPSES",
    "CCS_ILLUMINANT_POINTER_GAMUT",
    "DATA_POINTER_GAMUT_VOLUME",
    "CCS_POINTER_GAMUT_BOUNDARY",
]
__all__ += [
    "normalised_primary_matrix",
    "chromatically_adapted_primaries",
    "primaries_whitepoint",
    "RGB_luminance_equation",
    "RGB_luminance",
]
__all__ += ["RGB_Colourspace"]
__all__ += ["XYZ_to_RGB", "RGB_to_XYZ"]
__all__ += ["matrix_RGB_to_RGB", "RGB_to_RGB"]
__all__ += [
    "CV_range",
    "legal_to_full",
    "full_to_legal",
    "gamma_function",
    "log_encoding_ACESproxy",
    "log_decoding_ACESproxy",
    "log_encoding_ACEScc",
    "log_decoding_ACEScc",
    "log_encoding_ACEScct",
    "log_decoding_ACEScct",
    "log_encoding_AppleLogProfile",
    "log_decoding_AppleLogProfile",
    "oetf_ARIBSTDB67",
    "oetf_inverse_ARIBSTDB67",
    "log_encoding_ARRILogC3",
    "log_decoding_ARRILogC3",
    "log_encoding_ARRILogC4",
    "log_decoding_ARRILogC4",
    "oetf_BlackmagicFilmGeneration5",
    "oetf_inverse_BlackmagicFilmGeneration5",
    "log_encoding_CanonLog",
    "log_decoding_CanonLog",
    "log_encoding_CanonLog2",
    "log_decoding_CanonLog2",
    "log_encoding_CanonLog3",
    "log_decoding_CanonLog3",
    "log_encoding_Cineon",
    "log_decoding_Cineon",
    "oetf_DaVinciIntermediate",
    "oetf_inverse_DaVinciIntermediate",
    "eotf_inverse_DCDM",
    "eotf_DCDM",
    "eotf_inverse_DICOMGSDF",
    "eotf_DICOMGSDF",
    "log_encoding_DJIDLog",
    "log_decoding_DJIDLog",
    "exponent_function_basic",
    "exponent_function_monitor_curve",
    "log_encoding_FilmicPro6",
    "log_decoding_FilmicPro6",
    "log_encoding_FilmLightTLog",
    "log_decoding_FilmLightTLog",
    "log_encoding_Protune",
    "log_decoding_Protune",
    "oetf_BT2020",
    "oetf_inverse_BT2020",
    "oetf_BT601",
    "oetf_inverse_BT601",
    "oetf_BT709",
    "oetf_inverse_BT709",
    "oetf_BT1361",
    "oetf_inverse_BT1361",
    "eotf_inverse_BT1886",
    "eotf_BT1886",
    "eotf_inverse_ST2084",
    "eotf_ST2084",
    "oetf_BT2100_PQ",
    "oetf_inverse_BT2100_PQ",
    "eotf_BT2100_PQ",
    "eotf_inverse_BT2100_PQ",
    "ootf_BT2100_PQ",
    "ootf_inverse_BT2100_PQ",
    "oetf_BT2100_HLG",
    "oetf_inverse_BT2100_HLG",
    "BT2100_HLG_EOTF_METHODS",
    "eotf_BT2100_HLG",
    "BT2100_HLG_EOTF_INVERSE_METHODS",
    "eotf_inverse_BT2100_HLG",
    "BT2100_HLG_OOTF_METHODS",
    "ootf_BT2100_HLG",
    "BT2100_HLG_OOTF_INVERSE_METHODS",
    "ootf_inverse_BT2100_HLG",
    "oetf_H273_Log",
    "oetf_inverse_H273_Log",
    "oetf_H273_LogSqrt",
    "oetf_inverse_H273_LogSqrt",
    "oetf_H273_IEC61966_2",
    "oetf_inverse_H273_IEC61966_2",
    "eotf_H273_ST428_1",
    "eotf_inverse_H273_ST428_1",
    "linear_function",
    "logarithmic_function_basic",
    "logarithmic_function_quasilog",
    "logarithmic_function_camera",
    "log_encoding_Log2",
    "log_decoding_Log2",
    "log_encoding_Panalog",
    "log_decoding_Panalog",
    "log_encoding_VLog",
    "log_decoding_VLog",
    "log_encoding_FLog",
    "log_decoding_FLog",
    "log_encoding_FLog2",
    "log_decoding_FLog2",
    "log_encoding_LLog",
    "log_decoding_LLog",
    "log_encoding_NLog",
    "log_decoding_NLog",
    "log_encoding_PivotedLog",
    "log_decoding_PivotedLog",
    "log_encoding_REDLog",
    "log_decoding_REDLog",
    "log_encoding_REDLogFilm",
    "log_decoding_REDLogFilm",
    "LOG3G10_ENCODING_METHODS",
    "LOG3G10_DECODING_METHODS",
    "log_encoding_Log3G10",
    "log_decoding_Log3G10",
    "log_encoding_Log3G12",
    "log_decoding_Log3G12",
    "cctf_encoding_ROMMRGB",
    "cctf_decoding_ROMMRGB",
    "cctf_encoding_ProPhotoRGB",
    "cctf_decoding_ProPhotoRGB",
    "cctf_encoding_RIMMRGB",
    "cctf_decoding_RIMMRGB",
    "log_encoding_ERIMMRGB",
    "log_decoding_ERIMMRGB",
    "oetf_SMPTE240M",
    "eotf_SMPTE240M",
    "log_encoding_SLog",
    "log_decoding_SLog",
    "log_encoding_SLog2",
    "log_decoding_SLog2",
    "log_encoding_SLog3",
    "log_decoding_SLog3",
    "eotf_inverse_sRGB",
    "eotf_sRGB",
    "log_encoding_ViperLog",
    "log_decoding_ViperLog",
]
__all__ += [
    "LOG_ENCODINGS",
    "log_encoding",
    "LOG_DECODINGS",
    "log_decoding",
    "OETFS",
    "oetf",
    "OETF_INVERSES",
    "oetf_inverse",
    "EOTFS",
    "eotf",
    "EOTF_INVERSES",
    "eotf_inverse",
    "CCTF_ENCODINGS",
    "cctf_encoding",
    "CCTF_DECODINGS",
    "cctf_decoding",
    "OOTFS",
    "ootf",
    "OOTF_INVERSES",
    "ootf_inverse",
]
__all__ += [
    "RGB_COLOURSPACES",
    "RGB_COLOURSPACE_ACES2065_1",
    "RGB_COLOURSPACE_ACESCC",
    "RGB_COLOURSPACE_ACESCCT",
    "RGB_COLOURSPACE_ACESPROXY",
    "RGB_COLOURSPACE_ACESCG",
    "RGB_COLOURSPACE_ADOBE_RGB1998",
    "RGB_COLOURSPACE_ADOBE_WIDE_GAMUT_RGB",
    "RGB_COLOURSPACE_ARRI_WIDE_GAMUT_3",
    "RGB_COLOURSPACE_ARRI_WIDE_GAMUT_4",
    "RGB_COLOURSPACE_APPLE_RGB",
    "RGB_COLOURSPACE_BEST_RGB",
    "RGB_COLOURSPACE_BETA_RGB",
    "RGB_COLOURSPACE_BLACKMAGIC_WIDE_GAMUT",
    "RGB_COLOURSPACE_BT470_525",
    "RGB_COLOURSPACE_BT470_625",
    "RGB_COLOURSPACE_BT709",
    "RGB_COLOURSPACE_BT2020",
    "RGB_COLOURSPACE_CIE_RGB",
    "RGB_COLOURSPACE_CINEMA_GAMUT",
    "RGB_COLOURSPACE_COLOR_MATCH_RGB",
    "RGB_COLOURSPACE_DAVINCI_WIDE_GAMUT",
    "RGB_COLOURSPACE_DCDM_XYZ",
    "RGB_COLOURSPACE_DCI_P3",
    "RGB_COLOURSPACE_DCI_P3_P",
    "RGB_COLOURSPACE_DISPLAY_P3",
    "RGB_COLOURSPACE_DJI_D_GAMUT",
    "RGB_COLOURSPACE_DON_RGB_4",
    "RGB_COLOURSPACE_EBU_3213_E",
    "RGB_COLOURSPACE_ECI_RGB_V2",
    "RGB_COLOURSPACE_EKTA_SPACE_PS_5",
    "RGB_COLOURSPACE_FILMLIGHT_E_GAMUT",
    "RGB_COLOURSPACE_H273_GENERIC_FILM",
    "RGB_COLOURSPACE_H273_22_UNSPECIFIED",
    "RGB_COLOURSPACE_PROTUNE_NATIVE",
    "RGB_COLOURSPACE_MAX_RGB",
    "RGB_COLOURSPACE_N_GAMUT",
    "RGB_COLOURSPACE_P3_D65",
    "RGB_COLOURSPACE_PAL_SECAM",
    "RGB_COLOURSPACE_RED_COLOR",
    "RGB_COLOURSPACE_RED_COLOR_2",
    "RGB_COLOURSPACE_RED_COLOR_3",
    "RGB_COLOURSPACE_RED_COLOR_4",
    "RGB_COLOURSPACE_DRAGON_COLOR",
    "RGB_COLOURSPACE_DRAGON_COLOR_2",
    "RGB_COLOURSPACE_RED_WIDE_GAMUT_RGB",
    "RGB_COLOURSPACE_ROMM_RGB",
    "RGB_COLOURSPACE_RIMM_RGB",
    "RGB_COLOURSPACE_ERIMM_RGB",
    "RGB_COLOURSPACE_PROPHOTO_RGB",
    "RGB_COLOURSPACE_PLASA_ANSI_E154",
    "RGB_COLOURSPACE_RUSSELL_RGB",
    "RGB_COLOURSPACE_SHARP_RGB",
    "RGB_COLOURSPACE_SMPTE_240M",
    "RGB_COLOURSPACE_SMPTE_C",
    "RGB_COLOURSPACE_NTSC1953",
    "RGB_COLOURSPACE_NTSC1987",
    "RGB_COLOURSPACE_S_GAMUT",
    "RGB_COLOURSPACE_S_GAMUT3",
    "RGB_COLOURSPACE_S_GAMUT3_CINE",
    "RGB_COLOURSPACE_VENICE_S_GAMUT3",
    "RGB_COLOURSPACE_VENICE_S_GAMUT3_CINE",
    "RGB_COLOURSPACE_sRGB",
    "RGB_COLOURSPACE_V_GAMUT",
    "RGB_COLOURSPACE_XTREME_RGB",
    "RGB_COLOURSPACE_F_GAMUT",
]

__all__ += ["XYZ_to_sRGB", "sRGB_to_XYZ"]
__all__ += [
    "RGB_to_HSV",
    "HSV_to_RGB",
    "RGB_to_HSL",
    "HSL_to_RGB",
    "RGB_to_HCL",
    "HCL_to_RGB",
]
__all__ += ["RGB_to_CMY", "CMY_to_RGB", "CMY_to_CMYK", "CMYK_to_CMY"]
__all__ += ["RGB_to_IHLS", "IHLS_to_RGB"]
__all__ += ["RGB_to_Prismatic", "Prismatic_to_RGB"]
__all__ += [
    "WEIGHTS_YCBCR",
    "matrix_YCbCr",
    "offset_YCbCr",
    "RGB_to_YCbCr",
    "YCbCr_to_RGB",
    "RGB_to_YcCbcCrc",
    "YcCbcCrc_to_RGB",
]
__all__ += ["RGB_to_YCoCg", "YCoCg_to_RGB"]
__all__ += ["RGB_to_ICtCp", "ICtCp_to_RGB", "XYZ_to_ICtCp", "ICtCp_to_XYZ"]
__all__ += [
    "COLOUR_PRIMARIES_ITUTH273",
    "TRANSFER_CHARACTERISTICS_ITUTH273",
    "MATRIX_COEFFICIENTS_ITUTH273",
    "describe_video_signal_colour_primaries",
    "describe_video_signal_transfer_characteristics",
    "describe_video_signal_matrix_coefficients",
]


# ----------------------------------------------------------------------------#
# ---                API Changes and Deprecation Management                ---#
# ----------------------------------------------------------------------------#
class models(ModuleAPI):
    """Define a class acting like the *models* module."""

    def __getattr__(self, attribute) -> Any:
        """Return the value from the attribute with given name."""

        return super().__getattr__(attribute)


# v0.4.0
API_CHANGES = {
    "ObjectRenamed": [
        [
            "colour.models.RGB_to_ICTCP",
            "colour.models.RGB_to_ICtCp",
        ],
        [
            "colour.models.ICTCP_to_RGB",
            "colour.models.ICtCp_to_RGB",
        ],
        [
            "colour.models.RGB_to_IGPGTG",
            "colour.models.RGB_to_IgPgTg",
        ],
        [
            "colour.models.IGPGTG_to_RGB",
            "colour.models.IgPgTg_to_RGB",
        ],
        [
            "colour.models.XYZ_to_JzAzBz",
            "colour.models.XYZ_to_Jzazbz",
        ],
        [
            "colour.models.JzAzBz_to_XYZ",
            "colour.models.Jzazbz_to_XYZ",
        ],
    ]
}

# v0.4.2
API_CHANGES["ObjectRenamed"].extend(
    [
        [
            "colour.models.RGB_COLOURSPACE_ALEXA_WIDE_GAMUT",
            "colour.models.RGB_COLOURSPACE_ARRI_WIDE_GAMUT_3",
        ],
        [
            "colour.models.eotf_inverse_BT2020",
            "colour.models.oetf_BT2020",
        ],
        [
            "colour.models.eotf_BT2020",
            "colour.models.oetf_inverse_BT2020",
        ],
        [
            "colour.models.oetf_PQ_BT2100",
            "colour.models.oetf_BT2100_PQ",
        ],
        [
            "colour.models.oetf_inverse_PQ_BT2100",
            "colour.models.oetf_inverse_BT2100_PQ",
        ],
        [
            "colour.models.eotf_PQ_BT2100",
            "colour.models.eotf_BT2100_PQ",
        ],
        [
            "colour.models.eotf_inverse_PQ_BT2100",
            "colour.models.eotf_inverse_BT2100_PQ",
        ],
        [
            "colour.models.ootf_PQ_BT2100",
            "colour.models.ootf_BT2100_PQ",
        ],
        [
            "colour.models.ootf_inverse_PQ_BT2100",
            "colour.models.ootf_inverse_BT2100_PQ",
        ],
        [
            "colour.models.oetf_HLG_BT2100",
            "colour.models.oetf_BT2100_HLG",
        ],
        [
            "colour.models.oetf_inverse_HLG_BT2100",
            "colour.models.oetf_inverse_BT2100_HLG",
        ],
        [
            "colour.models.eotf_HLG_BT2100",
            "colour.models.eotf_BT2100_HLG",
        ],
        [
            "colour.models.eotf_inverse_HLG_BT2100",
            "colour.models.eotf_inverse_BT2100_HLG",
        ],
        [
            "colour.models.ootf_HLG_BT2100",
            "colour.models.ootf_BT2100_HLG",
        ],
        [
            "colour.models.ootf_inverse_HLG_BT2100",
            "colour.models.ootf_inverse_BT2100_HLG",
        ],
        [
            "colour.models.log_decoding_ALEXALogC",
            "colour.models.log_decoding_ARRILogC3",
        ],
        [
            "colour.models.log_encoding_ALEXALogC",
            "colour.models.log_encoding_ARRILogC3",
        ],
    ]
)


# v0.4.3
API_CHANGES["ObjectRenamed"].extend(
    [
        [
            "colour.models.XYZ_to_IPT_Munish2021",
            "colour.models.XYZ_to_IPT_Ragoo2021",
        ],
        [
            "colour.models.IPT_Munish2021_to_XYZ",
            "colour.models.IPT_Ragoo2021_to_XYZ",
        ],
    ]
)
"""Defines the *colour.models* sub-package API changes."""

if not is_documentation_building():
    sys.modules["colour.models"] = models(  # pyright: ignore
        sys.modules["colour.models"], build_API_changes(API_CHANGES)
    )

    del ModuleAPI, is_documentation_building, build_API_changes, sys
