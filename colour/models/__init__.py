import sys
from typing import Annotated

from colour.hints import NDArrayFloat
from colour.utilities import copy_definition, get_domain_range_scale_metadata

# isort: split

from .common import (
    COLOURSPACE_MODELS,
    COLOURSPACE_MODELS_AXIS_LABELS,
    COLOURSPACE_MODELS_DOMAIN_RANGE_SCALE_1_TO_REFERENCE,
    Iab_to_XYZ,
    Jab_to_JCh,
    JCh_to_Jab,
    XYZ_to_Iab,
)

# isort: split

from .cam02_ucs import (
    CAM02LCD_to_JMh_CIECAM02,
    CAM02LCD_to_XYZ,
    CAM02SCD_to_JMh_CIECAM02,
    CAM02SCD_to_XYZ,
    CAM02UCS_to_JMh_CIECAM02,
    CAM02UCS_to_XYZ,
    JMh_CIECAM02_to_CAM02LCD,
    JMh_CIECAM02_to_CAM02SCD,
    JMh_CIECAM02_to_CAM02UCS,
    XYZ_to_CAM02LCD,
    XYZ_to_CAM02SCD,
    XYZ_to_CAM02UCS,
)
from .cam16_ucs import (
    CAM16LCD_to_JMh_CAM16,
    CAM16LCD_to_XYZ,
    CAM16SCD_to_JMh_CAM16,
    CAM16SCD_to_XYZ,
    CAM16UCS_to_JMh_CAM16,
    CAM16UCS_to_XYZ,
    JMh_CAM16_to_CAM16LCD,
    JMh_CAM16_to_CAM16SCD,
    JMh_CAM16_to_CAM16UCS,
    XYZ_to_CAM16LCD,
    XYZ_to_CAM16SCD,
    XYZ_to_CAM16UCS,
)
from .cie_xyy import (
    XYZ_to_xy,
    XYZ_to_xyY,
    xy_to_xyY,
    xy_to_XYZ,
    xyY_to_xy,
    xyY_to_XYZ,
)

# isort: split

from .cie_lab import Lab_to_XYZ, XYZ_to_Lab
from .cie_luv import (
    CIE1976UCS_to_XYZ,
    Luv_to_uv,
    Luv_to_XYZ,
    Luv_uv_to_xy,
    XYZ_to_CIE1976UCS,
    XYZ_to_Luv,
    uv_to_Luv,
    xy_to_Luv_uv,
)
from .cie_ucs import (
    CIE1960UCS_to_XYZ,
    UCS_to_uv,
    UCS_to_XYZ,
    UCS_uv_to_xy,
    XYZ_to_CIE1960UCS,
    XYZ_to_UCS,
    uv_to_UCS,
    xy_to_UCS_uv,
)
from .cie_uvw import UVW_to_XYZ, XYZ_to_UVW
from .din99 import DIN99_to_Lab, DIN99_to_XYZ, Lab_to_DIN99, XYZ_to_DIN99
from .hdr_cie_lab import (
    HDR_CIELAB_METHODS,
    XYZ_to_hdr_CIELab,
    hdr_CIELab_to_XYZ,
)
from .hunter_lab import (
    Hunter_Lab_to_XYZ,
    XYZ_to_Hunter_Lab,
    XYZ_to_K_ab_HunterLab1966,
)
from .hunter_rdab import Hunter_Rdab_to_XYZ, XYZ_to_Hunter_Rdab
from .icacb import ICaCb_to_XYZ, XYZ_to_ICaCb
from .igpgtg import IgPgTg_to_XYZ, XYZ_to_IgPgTg
from .ipt import IPT_hue_angle, IPT_to_XYZ, XYZ_to_IPT
from .jzazbz import (
    IZAZBZ_METHODS,
    Izazbz_to_XYZ,
    Jzazbz_to_XYZ,
    XYZ_to_Izazbz,
    XYZ_to_Jzazbz,
)

# isort: split

from .hdr_ipt import HDR_IPT_METHODS, XYZ_to_hdr_IPT, hdr_IPT_to_XYZ
from .oklab import Oklab_to_XYZ, XYZ_to_Oklab
from .osa_ucs import OSA_UCS_to_XYZ, XYZ_to_OSA_UCS
from .prolab import ProLab_to_XYZ, XYZ_to_ProLab
from .ragoo2021 import IPT_Ragoo2021_to_XYZ, XYZ_to_IPT_Ragoo2021
from .sucs import (
    XYZ_to_sUCS,
    sUCS_chroma,
    sUCS_hue_angle,
    sUCS_Iab_to_sUCS_ICh,
    sUCS_ICh_to_sUCS_Iab,
    sUCS_to_XYZ,
)
from .yrg import LMS_to_Yrg, XYZ_to_Yrg, Yrg_to_LMS, Yrg_to_XYZ

# isort: split

from .datasets import (
    CCS_ILLUMINANT_POINTER_GAMUT,
    CCS_POINTER_GAMUT_BOUNDARY,
    DATA_MACADAM_1942_ELLIPSES,
    DATA_POINTER_GAMUT_VOLUME,
)

# isort: split

from .rgb import (
    BT2100_HLG_EOTF_INVERSE_METHODS,
    BT2100_HLG_EOTF_METHODS,
    BT2100_HLG_OOTF_INVERSE_METHODS,
    BT2100_HLG_OOTF_METHODS,
    CCTF_DECODINGS,
    CCTF_ENCODINGS,
    COLOUR_PRIMARIES_ITUTH273,
    EOTF_INVERSES,
    EOTFS,
    LOG3G10_DECODING_METHODS,
    LOG3G10_ENCODING_METHODS,
    LOG_DECODINGS,
    LOG_ENCODINGS,
    MATRIX_COEFFICIENTS_ITUTH273,
    OETF_INVERSES,
    OETFS,
    OOTF_INVERSES,
    OOTFS,
    RGB_COLOURSPACE_ACES2065_1,
    RGB_COLOURSPACE_ACESCC,
    RGB_COLOURSPACE_ACESCCT,
    RGB_COLOURSPACE_ACESCG,
    RGB_COLOURSPACE_ACESPROXY,
    RGB_COLOURSPACE_ADOBE_RGB1998,
    RGB_COLOURSPACE_ADOBE_WIDE_GAMUT_RGB,
    RGB_COLOURSPACE_APPLE_RGB,
    RGB_COLOURSPACE_ARRI_WIDE_GAMUT_3,
    RGB_COLOURSPACE_ARRI_WIDE_GAMUT_4,
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
    RGB_COLOURSPACE_DRAGON_COLOR,
    RGB_COLOURSPACE_DRAGON_COLOR_2,
    RGB_COLOURSPACE_EBU_3213_E,
    RGB_COLOURSPACE_ECI_RGB_V2,
    RGB_COLOURSPACE_EKTA_SPACE_PS_5,
    RGB_COLOURSPACE_ERIMM_RGB,
    RGB_COLOURSPACE_F_GAMUT,
    RGB_COLOURSPACE_F_GAMUT_C,
    RGB_COLOURSPACE_FILMLIGHT_E_GAMUT,
    RGB_COLOURSPACE_FILMLIGHT_E_GAMUT_2,
    RGB_COLOURSPACE_G18_REC709_SCENE,
    RGB_COLOURSPACE_G22_ADOBERGB_SCENE,
    RGB_COLOURSPACE_G22_AP1_SCENE,
    RGB_COLOURSPACE_G22_REC709_SCENE,
    RGB_COLOURSPACE_H273_22_UNSPECIFIED,
    RGB_COLOURSPACE_H273_GENERIC_FILM,
    RGB_COLOURSPACE_LIN_ADOBERGB_SCENE,
    RGB_COLOURSPACE_LIN_CIEXYZD65_SCENE,
    RGB_COLOURSPACE_LIN_P3D65_SCENE,
    RGB_COLOURSPACE_LIN_REC709_SCENE,
    RGB_COLOURSPACE_LIN_REC2020_SCENE,
    RGB_COLOURSPACE_MAX_RGB,
    RGB_COLOURSPACE_N_GAMUT,
    RGB_COLOURSPACE_NTSC1953,
    RGB_COLOURSPACE_NTSC1987,
    RGB_COLOURSPACE_P3_D65,
    RGB_COLOURSPACE_PAL_SECAM,
    RGB_COLOURSPACE_PLASA_ANSI_E154,
    RGB_COLOURSPACE_PROPHOTO_RGB,
    RGB_COLOURSPACE_PROTUNE_NATIVE,
    RGB_COLOURSPACE_RED_COLOR,
    RGB_COLOURSPACE_RED_COLOR_2,
    RGB_COLOURSPACE_RED_COLOR_3,
    RGB_COLOURSPACE_RED_COLOR_4,
    RGB_COLOURSPACE_RED_WIDE_GAMUT_RGB,
    RGB_COLOURSPACE_RIMM_RGB,
    RGB_COLOURSPACE_ROMM_RGB,
    RGB_COLOURSPACE_RUSSELL_RGB,
    RGB_COLOURSPACE_S_GAMUT,
    RGB_COLOURSPACE_S_GAMUT3,
    RGB_COLOURSPACE_S_GAMUT3_CINE,
    RGB_COLOURSPACE_SHARP_RGB,
    RGB_COLOURSPACE_SMPTE_240M,
    RGB_COLOURSPACE_SMPTE_C,
    RGB_COLOURSPACE_SRGB_AP1_SCENE,
    RGB_COLOURSPACE_SRGB_P3D65_SCENE,
    RGB_COLOURSPACE_SRGB_REC709_SCENE,
    RGB_COLOURSPACE_V_GAMUT,
    RGB_COLOURSPACE_VENICE_S_GAMUT3,
    RGB_COLOURSPACE_VENICE_S_GAMUT3_CINE,
    RGB_COLOURSPACE_XTREME_RGB,
    RGB_COLOURSPACES,
    RGB_COLOURSPACES_TEXTURE_ASSETS_AND_CG_RENDERING_CIF,
    TRANSFER_CHARACTERISTICS_ITUTH273,
    WEIGHTS_YCBCR,
    CMY_to_CMYK,
    CMY_to_RGB,
    CMYK_to_CMY,
    CV_range,
    HCL_to_RGB,
    HSL_to_RGB,
    HSV_to_RGB,
    ICtCp_to_RGB,
    ICtCp_to_XYZ,
    IHLS_to_RGB,
    Prismatic_to_RGB,
    RGB_Colourspace,
    RGB_COLOURSPACE_sRGB,
    RGB_luminance,
    RGB_luminance_equation,
    RGB_to_CMY,
    RGB_to_HCL,
    RGB_to_HSL,
    RGB_to_HSV,
    RGB_to_ICtCp,
    RGB_to_IHLS,
    RGB_to_Prismatic,
    RGB_to_RGB,
    RGB_to_XYZ,
    RGB_to_YCbCr,
    RGB_to_YcCbcCrc,
    RGB_to_YCoCg,
    XYZ_to_ICtCp,
    XYZ_to_RGB,
    XYZ_to_sRGB,
    YCbCr_to_RGB,
    YcCbcCrc_to_RGB,
    YCoCg_to_RGB,
    cctf_decoding,
    cctf_decoding_ProPhotoRGB,
    cctf_decoding_RIMMRGB,
    cctf_decoding_ROMMRGB,
    cctf_encoding,
    cctf_encoding_ProPhotoRGB,
    cctf_encoding_RIMMRGB,
    cctf_encoding_ROMMRGB,
    chromatically_adapted_primaries,
    describe_video_signal_colour_primaries,
    describe_video_signal_matrix_coefficients,
    describe_video_signal_transfer_characteristics,
    eotf,
    eotf_BT1886,
    eotf_BT2100_HLG,
    eotf_BT2100_PQ,
    eotf_DCDM,
    eotf_DICOMGSDF,
    eotf_H273_ST428_1,
    eotf_inverse,
    eotf_inverse_BT1886,
    eotf_inverse_BT2100_HLG,
    eotf_inverse_BT2100_PQ,
    eotf_inverse_DCDM,
    eotf_inverse_DICOMGSDF,
    eotf_inverse_H273_ST428_1,
    eotf_inverse_sRGB,
    eotf_inverse_ST2084,
    eotf_SMPTE240M,
    eotf_sRGB,
    eotf_ST2084,
    exponent_function_basic,
    exponent_function_monitor_curve,
    full_to_legal,
    gamma_function,
    legal_to_full,
    linear_function,
    log_decoding,
    log_decoding_ACEScc,
    log_decoding_ACEScct,
    log_decoding_ACESproxy,
    log_decoding_AppleLogProfile,
    log_decoding_ARRILogC3,
    log_decoding_ARRILogC4,
    log_decoding_CanonLog,
    log_decoding_CanonLog2,
    log_decoding_CanonLog3,
    log_decoding_Cineon,
    log_decoding_DJIDLog,
    log_decoding_ERIMMRGB,
    log_decoding_FilmicPro6,
    log_decoding_FilmLightTLog,
    log_decoding_FLog,
    log_decoding_FLog2,
    log_decoding_LLog,
    log_decoding_Log2,
    log_decoding_Log3G10,
    log_decoding_Log3G12,
    log_decoding_MiLog,
    log_decoding_NLog,
    log_decoding_Panalog,
    log_decoding_PivotedLog,
    log_decoding_Protune,
    log_decoding_REDLog,
    log_decoding_REDLogFilm,
    log_decoding_SLog,
    log_decoding_SLog2,
    log_decoding_SLog3,
    log_decoding_ViperLog,
    log_decoding_VLog,
    log_encoding,
    log_encoding_ACEScc,
    log_encoding_ACEScct,
    log_encoding_ACESproxy,
    log_encoding_AppleLogProfile,
    log_encoding_ARRILogC3,
    log_encoding_ARRILogC4,
    log_encoding_CanonLog,
    log_encoding_CanonLog2,
    log_encoding_CanonLog3,
    log_encoding_Cineon,
    log_encoding_DJIDLog,
    log_encoding_ERIMMRGB,
    log_encoding_FilmicPro6,
    log_encoding_FilmLightTLog,
    log_encoding_FLog,
    log_encoding_FLog2,
    log_encoding_LLog,
    log_encoding_Log2,
    log_encoding_Log3G10,
    log_encoding_Log3G12,
    log_encoding_MiLog,
    log_encoding_NLog,
    log_encoding_Panalog,
    log_encoding_PivotedLog,
    log_encoding_Protune,
    log_encoding_REDLog,
    log_encoding_REDLogFilm,
    log_encoding_SLog,
    log_encoding_SLog2,
    log_encoding_SLog3,
    log_encoding_ViperLog,
    log_encoding_VLog,
    logarithmic_function_basic,
    logarithmic_function_camera,
    logarithmic_function_quasilog,
    matrix_RGB_to_RGB,
    matrix_YCbCr,
    normalised_primary_matrix,
    oetf,
    oetf_ARIBSTDB67,
    oetf_BlackmagicFilmGeneration5,
    oetf_BT601,
    oetf_BT709,
    oetf_BT1361,
    oetf_BT2020,
    oetf_BT2100_HLG,
    oetf_BT2100_PQ,
    oetf_DaVinciIntermediate,
    oetf_H273_IEC61966_2,
    oetf_H273_Log,
    oetf_H273_LogSqrt,
    oetf_inverse,
    oetf_inverse_ARIBSTDB67,
    oetf_inverse_BlackmagicFilmGeneration5,
    oetf_inverse_BT601,
    oetf_inverse_BT709,
    oetf_inverse_BT1361,
    oetf_inverse_BT2020,
    oetf_inverse_BT2100_HLG,
    oetf_inverse_BT2100_PQ,
    oetf_inverse_DaVinciIntermediate,
    oetf_inverse_H273_IEC61966_2,
    oetf_inverse_H273_Log,
    oetf_inverse_H273_LogSqrt,
    oetf_SMPTE240M,
    offset_YCbCr,
    ootf,
    ootf_BT2100_HLG,
    ootf_BT2100_PQ,
    ootf_inverse,
    ootf_inverse_BT2100_HLG,
    ootf_inverse_BT2100_PQ,
    primaries_whitepoint,
    sRGB_to_XYZ,
)

__all__ = [
    "COLOURSPACE_MODELS",
    "COLOURSPACE_MODELS_AXIS_LABELS",
    "COLOURSPACE_MODELS_DOMAIN_RANGE_SCALE_1_TO_REFERENCE",
    "Iab_to_XYZ",
    "Jab_to_JCh",
    "JCh_to_Jab",
    "XYZ_to_Iab",
]
__all__ += [
    "CAM02LCD_to_JMh_CIECAM02",
    "CAM02LCD_to_XYZ",
    "CAM02SCD_to_JMh_CIECAM02",
    "CAM02SCD_to_XYZ",
    "CAM02UCS_to_JMh_CIECAM02",
    "CAM02UCS_to_XYZ",
    "JMh_CIECAM02_to_CAM02LCD",
    "JMh_CIECAM02_to_CAM02SCD",
    "JMh_CIECAM02_to_CAM02UCS",
    "XYZ_to_CAM02LCD",
    "XYZ_to_CAM02SCD",
    "XYZ_to_CAM02UCS",
]
__all__ += [
    "CAM16LCD_to_JMh_CAM16",
    "CAM16LCD_to_XYZ",
    "CAM16SCD_to_JMh_CAM16",
    "CAM16SCD_to_XYZ",
    "CAM16UCS_to_JMh_CAM16",
    "CAM16UCS_to_XYZ",
    "JMh_CAM16_to_CAM16LCD",
    "JMh_CAM16_to_CAM16SCD",
    "JMh_CAM16_to_CAM16UCS",
    "XYZ_to_CAM16LCD",
    "XYZ_to_CAM16SCD",
    "XYZ_to_CAM16UCS",
]
__all__ += [
    "XYZ_to_xy",
    "XYZ_to_xyY",
    "xy_to_xyY",
    "xy_to_XYZ",
    "xyY_to_xy",
    "xyY_to_XYZ",
]
__all__ += [
    "Lab_to_XYZ",
    "XYZ_to_Lab",
]
__all__ += [
    "CIE1976UCS_to_XYZ",
    "Luv_to_uv",
    "Luv_to_XYZ",
    "Luv_uv_to_xy",
    "XYZ_to_CIE1976UCS",
    "XYZ_to_Luv",
    "uv_to_Luv",
    "xy_to_Luv_uv",
]
__all__ += [
    "CIE1960UCS_to_XYZ",
    "UCS_to_uv",
    "UCS_to_XYZ",
    "UCS_uv_to_xy",
    "XYZ_to_CIE1960UCS",
    "XYZ_to_UCS",
    "uv_to_UCS",
    "xy_to_UCS_uv",
]
__all__ += [
    "UVW_to_XYZ",
    "XYZ_to_UVW",
]
__all__ += [
    "DIN99_to_Lab",
    "DIN99_to_XYZ",
    "Lab_to_DIN99",
    "XYZ_to_DIN99",
]
__all__ += [
    "HDR_CIELAB_METHODS",
    "XYZ_to_hdr_CIELab",
    "hdr_CIELab_to_XYZ",
]
__all__ += [
    "Hunter_Lab_to_XYZ",
    "XYZ_to_Hunter_Lab",
    "XYZ_to_K_ab_HunterLab1966",
]
__all__ += [
    "Hunter_Rdab_to_XYZ",
    "XYZ_to_Hunter_Rdab",
]
__all__ += [
    "ICaCb_to_XYZ",
    "XYZ_to_ICaCb",
]
__all__ += [
    "IgPgTg_to_XYZ",
    "XYZ_to_IgPgTg",
]
__all__ += [
    "IPT_hue_angle",
    "IPT_to_XYZ",
    "XYZ_to_IPT",
]
__all__ += [
    "IZAZBZ_METHODS",
    "Izazbz_to_XYZ",
    "Jzazbz_to_XYZ",
    "XYZ_to_Izazbz",
    "XYZ_to_Jzazbz",
]
__all__ += [
    "HDR_IPT_METHODS",
    "XYZ_to_hdr_IPT",
    "hdr_IPT_to_XYZ",
]
__all__ += [
    "Oklab_to_XYZ",
    "XYZ_to_Oklab",
]
__all__ += [
    "OSA_UCS_to_XYZ",
    "XYZ_to_OSA_UCS",
]
__all__ += [
    "ProLab_to_XYZ",
    "XYZ_to_ProLab",
]
__all__ += [
    "IPT_Ragoo2021_to_XYZ",
    "XYZ_to_IPT_Ragoo2021",
]
__all__ += [
    "XYZ_to_sUCS",
    "sUCS_chroma",
    "sUCS_hue_angle",
    "sUCS_Iab_to_sUCS_ICh",
    "sUCS_ICh_to_sUCS_Iab",
    "sUCS_to_XYZ",
]
__all__ += [
    "LMS_to_Yrg",
    "XYZ_to_Yrg",
    "Yrg_to_LMS",
    "Yrg_to_XYZ",
]
__all__ += [
    "CCS_ILLUMINANT_POINTER_GAMUT",
    "CCS_POINTER_GAMUT_BOUNDARY",
    "DATA_MACADAM_1942_ELLIPSES",
    "DATA_POINTER_GAMUT_VOLUME",
]
__all__ += [
    "BT2100_HLG_EOTF_INVERSE_METHODS",
    "BT2100_HLG_EOTF_METHODS",
    "BT2100_HLG_OOTF_INVERSE_METHODS",
    "BT2100_HLG_OOTF_METHODS",
    "CCTF_DECODINGS",
    "CCTF_ENCODINGS",
    "COLOUR_PRIMARIES_ITUTH273",
    "EOTF_INVERSES",
    "EOTFS",
    "LOG3G10_DECODING_METHODS",
    "LOG3G10_ENCODING_METHODS",
    "LOG_DECODINGS",
    "LOG_ENCODINGS",
    "MATRIX_COEFFICIENTS_ITUTH273",
    "OETF_INVERSES",
    "OETFS",
    "OOTF_INVERSES",
    "OOTFS",
    "RGB_COLOURSPACE_ACES2065_1",
    "RGB_COLOURSPACE_ACESCC",
    "RGB_COLOURSPACE_ACESCCT",
    "RGB_COLOURSPACE_ACESCG",
    "RGB_COLOURSPACE_ACESPROXY",
    "RGB_COLOURSPACE_ADOBE_RGB1998",
    "RGB_COLOURSPACE_ADOBE_WIDE_GAMUT_RGB",
    "RGB_COLOURSPACE_APPLE_RGB",
    "RGB_COLOURSPACE_ARRI_WIDE_GAMUT_3",
    "RGB_COLOURSPACE_ARRI_WIDE_GAMUT_4",
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
    "RGB_COLOURSPACE_DRAGON_COLOR",
    "RGB_COLOURSPACE_DRAGON_COLOR_2",
    "RGB_COLOURSPACE_EBU_3213_E",
    "RGB_COLOURSPACE_ECI_RGB_V2",
    "RGB_COLOURSPACE_EKTA_SPACE_PS_5",
    "RGB_COLOURSPACE_ERIMM_RGB",
    "RGB_COLOURSPACE_F_GAMUT",
    "RGB_COLOURSPACE_F_GAMUT_C",
    "RGB_COLOURSPACE_FILMLIGHT_E_GAMUT",
    "RGB_COLOURSPACE_FILMLIGHT_E_GAMUT_2",
    "RGB_COLOURSPACE_G18_REC709_SCENE",
    "RGB_COLOURSPACE_G22_ADOBERGB_SCENE",
    "RGB_COLOURSPACE_G22_AP1_SCENE",
    "RGB_COLOURSPACE_G22_REC709_SCENE",
    "RGB_COLOURSPACE_H273_22_UNSPECIFIED",
    "RGB_COLOURSPACE_H273_GENERIC_FILM",
    "RGB_COLOURSPACE_LIN_ADOBERGB_SCENE",
    "RGB_COLOURSPACE_LIN_CIEXYZD65_SCENE",
    "RGB_COLOURSPACE_LIN_P3D65_SCENE",
    "RGB_COLOURSPACE_LIN_REC709_SCENE",
    "RGB_COLOURSPACE_LIN_REC2020_SCENE",
    "RGB_COLOURSPACE_MAX_RGB",
    "RGB_COLOURSPACE_N_GAMUT",
    "RGB_COLOURSPACE_NTSC1953",
    "RGB_COLOURSPACE_NTSC1987",
    "RGB_COLOURSPACE_P3_D65",
    "RGB_COLOURSPACE_PAL_SECAM",
    "RGB_COLOURSPACE_PLASA_ANSI_E154",
    "RGB_COLOURSPACE_PROPHOTO_RGB",
    "RGB_COLOURSPACE_PROTUNE_NATIVE",
    "RGB_COLOURSPACE_RED_COLOR",
    "RGB_COLOURSPACE_RED_COLOR_2",
    "RGB_COLOURSPACE_RED_COLOR_3",
    "RGB_COLOURSPACE_RED_COLOR_4",
    "RGB_COLOURSPACE_RED_WIDE_GAMUT_RGB",
    "RGB_COLOURSPACE_RIMM_RGB",
    "RGB_COLOURSPACE_ROMM_RGB",
    "RGB_COLOURSPACE_RUSSELL_RGB",
    "RGB_COLOURSPACE_S_GAMUT",
    "RGB_COLOURSPACE_S_GAMUT3",
    "RGB_COLOURSPACE_S_GAMUT3_CINE",
    "RGB_COLOURSPACE_SHARP_RGB",
    "RGB_COLOURSPACE_SMPTE_240M",
    "RGB_COLOURSPACE_SMPTE_C",
    "RGB_COLOURSPACE_SRGB_AP1_SCENE",
    "RGB_COLOURSPACE_SRGB_P3D65_SCENE",
    "RGB_COLOURSPACE_SRGB_REC709_SCENE",
    "RGB_COLOURSPACE_V_GAMUT",
    "RGB_COLOURSPACE_VENICE_S_GAMUT3",
    "RGB_COLOURSPACE_VENICE_S_GAMUT3_CINE",
    "RGB_COLOURSPACE_XTREME_RGB",
    "RGB_COLOURSPACES",
    "RGB_COLOURSPACES_TEXTURE_ASSETS_AND_CG_RENDERING_CIF",
    "TRANSFER_CHARACTERISTICS_ITUTH273",
    "WEIGHTS_YCBCR",
    "CMY_to_CMYK",
    "CMY_to_RGB",
    "CMYK_to_CMY",
    "CV_range",
    "HCL_to_RGB",
    "HSL_to_RGB",
    "HSV_to_RGB",
    "ICtCp_to_RGB",
    "ICtCp_to_XYZ",
    "IHLS_to_RGB",
    "Prismatic_to_RGB",
    "RGB_Colourspace",
    "RGB_COLOURSPACE_sRGB",
    "RGB_luminance",
    "RGB_luminance_equation",
    "RGB_to_CMY",
    "RGB_to_HCL",
    "RGB_to_HSL",
    "RGB_to_HSV",
    "RGB_to_ICtCp",
    "RGB_to_IHLS",
    "RGB_to_Prismatic",
    "RGB_to_RGB",
    "RGB_to_XYZ",
    "RGB_to_YCbCr",
    "RGB_to_YcCbcCrc",
    "RGB_to_YCoCg",
    "XYZ_to_ICtCp",
    "XYZ_to_RGB",
    "XYZ_to_sRGB",
    "YCbCr_to_RGB",
    "YcCbcCrc_to_RGB",
    "YCoCg_to_RGB",
    "cctf_decoding",
    "cctf_decoding_ProPhotoRGB",
    "cctf_decoding_RIMMRGB",
    "cctf_decoding_ROMMRGB",
    "cctf_encoding",
    "cctf_encoding_ProPhotoRGB",
    "cctf_encoding_RIMMRGB",
    "cctf_encoding_ROMMRGB",
    "chromatically_adapted_primaries",
    "describe_video_signal_colour_primaries",
    "describe_video_signal_matrix_coefficients",
    "describe_video_signal_transfer_characteristics",
    "eotf",
    "eotf_BT1886",
    "eotf_BT2100_HLG",
    "eotf_BT2100_PQ",
    "eotf_DCDM",
    "eotf_DICOMGSDF",
    "eotf_H273_ST428_1",
    "eotf_inverse",
    "eotf_inverse_BT1886",
    "eotf_inverse_BT2100_HLG",
    "eotf_inverse_BT2100_PQ",
    "eotf_inverse_DCDM",
    "eotf_inverse_DICOMGSDF",
    "eotf_inverse_H273_ST428_1",
    "eotf_inverse_sRGB",
    "eotf_inverse_ST2084",
    "eotf_SMPTE240M",
    "eotf_sRGB",
    "eotf_ST2084",
    "exponent_function_basic",
    "exponent_function_monitor_curve",
    "full_to_legal",
    "gamma_function",
    "legal_to_full",
    "linear_function",
    "log_decoding",
    "log_decoding_ACEScc",
    "log_decoding_ACEScct",
    "log_decoding_ACESproxy",
    "log_decoding_AppleLogProfile",
    "log_decoding_ARRILogC3",
    "log_decoding_ARRILogC4",
    "log_decoding_CanonLog",
    "log_decoding_CanonLog2",
    "log_decoding_CanonLog3",
    "log_decoding_Cineon",
    "log_decoding_DJIDLog",
    "log_decoding_ERIMMRGB",
    "log_decoding_FilmicPro6",
    "log_decoding_FilmLightTLog",
    "log_decoding_FLog",
    "log_decoding_FLog2",
    "log_decoding_LLog",
    "log_decoding_Log2",
    "log_decoding_Log3G10",
    "log_decoding_Log3G12",
    "log_decoding_MiLog",
    "log_decoding_NLog",
    "log_decoding_Panalog",
    "log_decoding_PivotedLog",
    "log_decoding_Protune",
    "log_decoding_REDLog",
    "log_decoding_REDLogFilm",
    "log_decoding_SLog",
    "log_decoding_SLog2",
    "log_decoding_SLog3",
    "log_decoding_ViperLog",
    "log_decoding_VLog",
    "log_encoding",
    "log_encoding_ACEScc",
    "log_encoding_ACEScct",
    "log_encoding_ACESproxy",
    "log_encoding_AppleLogProfile",
    "log_encoding_ARRILogC3",
    "log_encoding_ARRILogC4",
    "log_encoding_CanonLog",
    "log_encoding_CanonLog2",
    "log_encoding_CanonLog3",
    "log_encoding_Cineon",
    "log_encoding_DJIDLog",
    "log_encoding_ERIMMRGB",
    "log_encoding_FilmicPro6",
    "log_encoding_FilmLightTLog",
    "log_encoding_FLog",
    "log_encoding_FLog2",
    "log_encoding_LLog",
    "log_encoding_Log2",
    "log_encoding_Log3G10",
    "log_encoding_Log3G12",
    "log_encoding_MiLog",
    "log_encoding_NLog",
    "log_encoding_Panalog",
    "log_encoding_PivotedLog",
    "log_encoding_Protune",
    "log_encoding_REDLog",
    "log_encoding_REDLogFilm",
    "log_encoding_SLog",
    "log_encoding_SLog2",
    "log_encoding_SLog3",
    "log_encoding_ViperLog",
    "log_encoding_VLog",
    "logarithmic_function_basic",
    "logarithmic_function_camera",
    "logarithmic_function_quasilog",
    "matrix_RGB_to_RGB",
    "matrix_YCbCr",
    "normalised_primary_matrix",
    "oetf",
    "oetf_ARIBSTDB67",
    "oetf_BlackmagicFilmGeneration5",
    "oetf_BT601",
    "oetf_BT709",
    "oetf_BT1361",
    "oetf_BT2020",
    "oetf_BT2100_HLG",
    "oetf_BT2100_PQ",
    "oetf_DaVinciIntermediate",
    "oetf_H273_IEC61966_2",
    "oetf_H273_Log",
    "oetf_H273_LogSqrt",
    "oetf_inverse",
    "oetf_inverse_ARIBSTDB67",
    "oetf_inverse_BlackmagicFilmGeneration5",
    "oetf_inverse_BT601",
    "oetf_inverse_BT709",
    "oetf_inverse_BT1361",
    "oetf_inverse_BT2020",
    "oetf_inverse_BT2100_HLG",
    "oetf_inverse_BT2100_PQ",
    "oetf_inverse_DaVinciIntermediate",
    "oetf_inverse_H273_IEC61966_2",
    "oetf_inverse_H273_Log",
    "oetf_inverse_H273_LogSqrt",
    "oetf_SMPTE240M",
    "offset_YCbCr",
    "ootf",
    "ootf_BT2100_HLG",
    "ootf_BT2100_PQ",
    "ootf_inverse",
    "ootf_inverse_BT2100_HLG",
    "ootf_inverse_BT2100_PQ",
    "primaries_whitepoint",
    "sRGB_to_XYZ",
]

# Programmatically defining the colourspace models polar conversions.
COLOURSPACE_MODELS_POLAR_CONVERSIONS = (
    ("hdr_CIELab", "hdr_CIELCHab"),
    ("hdr_IPT", "hdr_ICH"),
    ("Hunter_Lab", "Hunter_LCHab"),
    ("Hunter_Rdab", "Hunter_RdCHab"),
    ("ICaCb", "ICHab"),
    ("ICtCp", "ICHtp"),
    ("IPT", "ICH"),
    ("IPT_Ragoo2021", "ICH_Ragoo2021"),
    ("IgPgTg", "IgCHpt"),
    ("Izazbz", "IzCHab"),
    ("Jzazbz", "JzCHab"),
    ("Lab", "LCHab"),
    ("Luv", "LCHuv"),
    ("Oklab", "Oklch"),
    ("ProLab", "ProLCHab"),
    ("sUCS", "sUCSICH"),
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
"""

for _Jab, _JCh in COLOURSPACE_MODELS_POLAR_CONVERSIONS:
    # Derive the correct annotation scale from the source model's XYZ_to_Jab function
    _scale = 1  # Default scale for most models
    _XYZ_to_Jab_name = f"XYZ_to_{_Jab}"
    _module = sys.modules["colour.models"]

    if hasattr(_module, _XYZ_to_Jab_name):
        _XYZ_to_Jab_callable = getattr(_module, _XYZ_to_Jab_name)
        _metadata = get_domain_range_scale_metadata(_XYZ_to_Jab_callable)
        _range_scale = _metadata.get("range")

        # If the source model uses scale 100, the polar form should too
        if _range_scale == 100:
            _scale = 100

    # Create Jab_to_JCh wrapper with correct annotation
    name = f"{_Jab}_to_{_JCh}"
    _callable = copy_definition(Jab_to_JCh, name)
    _callable.__doc__ = _DOCSTRING_JAB_TO_JCH.format(Jab=_Jab, JCh=_JCh)
    # Update the return annotation with the derived scale
    _callable.__annotations__["return"] = Annotated[NDArrayFloat, (_scale, _scale, 360)]
    setattr(_module, name, _callable)
    __all__.append(name)

    # Create JCh_to_Jab wrapper with correct annotation
    name = f"{_JCh}_to_{_Jab}"
    _callable = copy_definition(JCh_to_Jab, name)
    _callable.__doc__ = _DOCSTRING_JCH_TO_JAB.format(JCh=_JCh, Jab=_Jab)
    # Update the parameter annotation with the derived scale
    _parameter = next(iter(_callable.__annotations__.keys()))
    _callable.__annotations__[_parameter] = Annotated[
        NDArrayFloat, (_scale, _scale, 360)
    ]
    setattr(_module, name, _callable)
    __all__.append(name)

del (
    _DOCSTRING_JAB_TO_JCH,
    _DOCSTRING_JCH_TO_JAB,
    _JCh,
    _Jab,
    _callable,
    _module,
    _scale,
    _XYZ_to_Jab_name,
    _metadata,
    _range_scale,
    _XYZ_to_Jab_callable,
    _parameter,
)

__all__ += ["COLOURSPACE_MODELS_POLAR"]
