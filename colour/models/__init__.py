# -*- coding: utf-8 -*-

from __future__ import absolute_import

import sys

from colour.utilities.deprecation import ModuleAPI, build_API_changes
from colour.utilities.documentation import is_documentation_building

from .common import (Jab_to_JCh, JCh_to_Jab, COLOURSPACE_MODELS,
                     COLOURSPACE_MODELS_AXIS_LABELS, XYZ_to_colourspace_model)
from .cam02_ucs import (JMh_CIECAM02_to_CAM02LCD, CAM02LCD_to_JMh_CIECAM02,
                        JMh_CIECAM02_to_CAM02SCD, CAM02SCD_to_JMh_CIECAM02,
                        JMh_CIECAM02_to_CAM02UCS, CAM02UCS_to_JMh_CIECAM02)
from .cam16_ucs import (JMh_CAM16_to_CAM16LCD, CAM16LCD_to_JMh_CAM16,
                        JMh_CAM16_to_CAM16SCD, CAM16SCD_to_JMh_CAM16,
                        JMh_CAM16_to_CAM16UCS, CAM16UCS_to_JMh_CAM16)
from .cie_xyy import (XYZ_to_xyY, xyY_to_XYZ, xy_to_xyY, xyY_to_xy, xy_to_XYZ,
                      XYZ_to_xy)
from .cie_lab import XYZ_to_Lab, Lab_to_XYZ, Lab_to_LCHab, LCHab_to_Lab
from .cie_luv import (XYZ_to_Luv, Luv_to_XYZ, Luv_to_uv, uv_to_Luv,
                      Luv_uv_to_xy, xy_to_Luv_uv, Luv_to_LCHuv, LCHuv_to_Luv)
from .cie_ucs import (XYZ_to_UCS, UCS_to_XYZ, UCS_to_uv, uv_to_UCS,
                      UCS_uv_to_xy, xy_to_UCS_uv)
from .cie_uvw import XYZ_to_UVW, UVW_to_XYZ
from .din99 import Lab_to_DIN99, DIN99_to_Lab
from .hdr_cie_lab import (HDR_CIELAB_METHODS, XYZ_to_hdr_CIELab,
                          hdr_CIELab_to_XYZ)
from .hunter_lab import (XYZ_to_K_ab_HunterLab1966, XYZ_to_Hunter_Lab,
                         Hunter_Lab_to_XYZ)
from .hunter_rdab import XYZ_to_Hunter_Rdab, Hunter_Rdab_to_XYZ
from .igpgtg import XYZ_to_IGPGTG, IGPGTG_to_XYZ
from .ipt import XYZ_to_IPT, IPT_to_XYZ, IPT_hue_angle
from .jzazbz import XYZ_to_JzAzBz, JzAzBz_to_XYZ
from .hdr_ipt import HDR_IPT_METHODS, XYZ_to_hdr_IPT, hdr_IPT_to_XYZ
from .osa_ucs import XYZ_to_OSA_UCS, OSA_UCS_to_XYZ
from .datasets import *  # noqa
from . import datasets
from .rgb import *  # noqa
from . import rgb

__all__ = [
    'Jab_to_JCh', 'JCh_to_Jab', 'COLOURSPACE_MODELS',
    'COLOURSPACE_MODELS_AXIS_LABELS', 'XYZ_to_colourspace_model'
]
__all__ += [
    'JMh_CIECAM02_to_CAM02LCD', 'CAM02LCD_to_JMh_CIECAM02',
    'JMh_CIECAM02_to_CAM02SCD', 'CAM02SCD_to_JMh_CIECAM02',
    'JMh_CIECAM02_to_CAM02UCS', 'CAM02UCS_to_JMh_CIECAM02'
]
__all__ += [
    'JMh_CAM16_to_CAM16LCD', 'CAM16LCD_to_JMh_CAM16', 'JMh_CAM16_to_CAM16SCD',
    'CAM16SCD_to_JMh_CAM16', 'JMh_CAM16_to_CAM16UCS', 'CAM16UCS_to_JMh_CAM16'
]
__all__ += [
    'XYZ_to_xyY', 'xyY_to_XYZ', 'xy_to_xyY', 'xyY_to_xy', 'xy_to_XYZ',
    'XYZ_to_xy'
]
__all__ += ['XYZ_to_Lab', 'Lab_to_XYZ', 'Lab_to_LCHab', 'LCHab_to_Lab']
__all__ += [
    'XYZ_to_Luv', 'Luv_to_XYZ', 'Luv_to_uv', 'uv_to_Luv', 'Luv_uv_to_xy',
    'xy_to_Luv_uv', 'Luv_to_LCHuv', 'LCHuv_to_Luv'
]
__all__ += [
    'XYZ_to_UCS', 'UCS_to_XYZ', 'UCS_to_uv', 'uv_to_UCS', 'UCS_uv_to_xy',
    'xy_to_UCS_uv'
]
__all__ += ['XYZ_to_UVW', 'UVW_to_XYZ']
__all__ += ['Lab_to_DIN99', 'DIN99_to_Lab']
__all__ += ['HDR_CIELAB_METHODS', 'XYZ_to_hdr_CIELab', 'hdr_CIELab_to_XYZ']
__all__ += [
    'XYZ_to_K_ab_HunterLab1966', 'XYZ_to_Hunter_Lab', 'Hunter_Lab_to_XYZ',
    'XYZ_to_Hunter_Rdab'
]
__all__ += ['XYZ_to_Hunter_Rdab', 'Hunter_Rdab_to_XYZ']
__all__ += ['XYZ_to_IGPGTG', 'IGPGTG_to_XYZ']
__all__ += ['XYZ_to_IPT', 'IPT_to_XYZ', 'IPT_hue_angle']
__all__ += ['XYZ_to_JzAzBz', 'JzAzBz_to_XYZ']
__all__ += ['HDR_IPT_METHODS', 'XYZ_to_hdr_IPT', 'hdr_IPT_to_XYZ']
__all__ += ['XYZ_to_OSA_UCS', 'OSA_UCS_to_XYZ']
__all__ += datasets.__all__
__all__ += rgb.__all__


# ----------------------------------------------------------------------------#
# ---                API Changes and Deprecation Management                ---#
# ----------------------------------------------------------------------------#
class models(ModuleAPI):
    def __getattr__(self, attribute):
        return super(models, self).__getattr__(attribute)


# v0.3.14
API_CHANGES = {
    'ObjectFutureRemove': ['colour.models.XYZ_to_colourspace_model', ],
    'ObjectRenamed': [
        [
            'colour.models.decoding_cctf',
            'colour.models.cctf_decoding',
        ],
        [
            'colour.models.DECODING_CCTFS',
            'colour.models.CCTF_DECODINGS',
        ],
        [
            'colour.models.encoding_cctf',
            'colour.models.cctf_encoding',
        ],
        [
            'colour.models.ENCODING_CCTFS',
            'colour.models.CCTF_ENCODINGS',
        ],
        [
            'colour.models.log_decoding_curve',
            'colour.models.log_decoding',
        ],
        [
            'colour.models.LOG_DECODING_CURVES',
            'colour.models.LOG_DECODINGS',
        ],
        [
            'colour.models.log_encoding_curve',
            'colour.models.log_encoding',
        ],
        [
            'colour.models.LOG_ENCODING_CURVES',
            'colour.models.LOG_ENCODINGS',
        ],
        [
            'colour.models.oetf_ROMMRGB',
            'colour.models.cctf_encoding_ROMMRGB',
        ],
        [
            'colour.models.oetf_RIMMRGB',
            'colour.models.cctf_encoding_RIMMRGB',
        ],
        [
            'colour.models.oetf_ProPhotoRGB',
            'colour.models.cctf_encoding_ProPhotoRGB',
        ],
        [
            'colour.models.oetf_ST2084',
            'colour.models.eotf_inverse_ST2084',
        ],
        [
            'colour.models.oetf_sRGB',
            'colour.models.eotf_inverse_sRGB',
        ],
        [
            'colour.models.oetf_inverse_sRGB',
            'colour.models.eotf_sRGB',
        ],
        [
            'colour.models.oetf_BT2100_HLG',
            'colour.models.oetf_HLG_BT2100',
        ],
        [
            'colour.models.oetf_reverse_ARIBSTDB67',
            'colour.models.oetf_inverse_ARIBSTDB67'
        ],
        [
            'colour.models.oetf_reverse_BT2100_HLG',
            'colour.models.oetf_inverse_HLG_BT2100',
        ],
        [
            'colour.models.oetf_reverse_BT601',
            'colour.models.oetf_inverse_BT601',
        ],
        [
            'colour.models.oetf_reverse_BT709',
            'colour.models.oetf_inverse_BT709',
        ],
        [
            'colour.models.eotf_ROMMRGB',
            'colour.models.cctf_decoding_ROMMRGB',
        ],
        [
            'colour.models.eotf_RIMMRGB',
            'colour.models.cctf_decoding_RIMMRGB',
        ],
        [
            'colour.models.eotf_ProPhotoRGB',
            'colour.models.cctf_decoding_ProPhotoRGB',
        ],
        [
            'colour.models.eotf_reverse_BT1886',
            'colour.models.eotf_inverse_BT1886',
        ],
        [
            'colour.models.eotf_BT2100_HLG',
            'colour.models.eotf_HLG_BT2100',
        ],
        [
            'colour.models.eotf_reverse_BT2100_HLG',
            'colour.models.eotf_inverse_HLG_BT2100',
        ],
        [
            'colour.models.eotf_reverse_DCDM',
            'colour.models.eotf_inverse_DCDM',
        ],
        [
            'colour.models.eotf_reverse_sRGB',
            'colour.models.eotf_inverse_sRGB',
        ],
        [
            'colour.models.eotf_reverse_ST2084',
            'colour.models.eotf_inverse_ST2084',
        ],
        [
            'colour.models.ootf_BT2100_HLG',
            'colour.models.ootf_HLG_BT2100',
        ],
        [
            'colour.models.ootf_reverse_BT2100_HLG',
            'colour.models.ootf_inverse_HLG_BT2100',
        ],
        [
            'colour.models.oetf_BT2100_PQ',
            'colour.models.oetf_PQ_BT2100',
        ],
        [
            'colour.models.oetf_reverse_BT2100_PQ',
            'colour.models.oetf_inverse_PQ_BT2100',
        ],
        [
            'colour.models.eotf_BT2100_PQ',
            'colour.models.eotf_PQ_BT2100',
        ],
        [
            'colour.models.eotf_reverse_BT2100_PQ',
            'colour.models.eotf_inverse_PQ_BT2100',
        ],
        [
            'colour.models.ootf_BT2100_PQ',
            'colour.models.ootf_PQ_BT2100',
        ],
        [
            'colour.models.ootf_reverse_BT2100_PQ',
            'colour.models.ootf_inverse_PQ_BT2100',
        ],
    ]
}
"""
Defines *colour.models* sub-package API changes.

API_CHANGES : dict
"""

# v0.3.16
API_CHANGES['ObjectRenamed'] = API_CHANGES['ObjectRenamed'] + [
    [
        'colour.models.ACES_2065_1_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_ACES2065_1',
    ],
    [
        'colour.models.ACES_CC_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_ACESCC',
    ],
    [
        'colour.models.ACES_CCT_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_ACESCCT',
    ],
    [
        'colour.models.ACES_CG_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_ACESCG',
    ],
    [
        'colour.models.ACES_PROXY_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_ACESPROXY',
    ],
    [
        'colour.models.ADOBE_RGB_1998_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_ADOBE_RGB1998',
    ],
    [
        'colour.models.ADOBE_WIDE_GAMUT_RGB_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_ADOBE_WIDE_GAMUT_RGB',
    ],
    [
        'colour.models.APPLE_RGB_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_APPLE_RGB',
    ],
    [
        'colour.models.ALEXA_WIDE_GAMUT_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_ALEXA_WIDE_GAMUT',
    ],
    [
        'colour.models.BEST_RGB_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_BEST_RGB',
    ],
    [
        'colour.models.BETA_RGB_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_BETA_RGB',
    ],
    [
        'colour.models.CIE_RGB_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_CIE_RGB',
    ],
    [
        'colour.models.CINEMA_GAMUT_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_CINEMA_GAMUT',
    ],
    [
        'colour.models.COLOR_MATCH_RGB_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_COLOR_MATCH_RGB',
    ],
    [
        'colour.models.DCDM_XYZ_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_DCDM_XYZ',
    ],
    [
        'colour.models.DCI_P3_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_DCI_P3_P',
    ],
    [
        'colour.models.DCI_P3_P_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_DCI_P3',
    ],
    [
        'colour.models.DISPLAY_P3_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_DISPLAY_P3',
    ],
    [
        'colour.models.P3_D65_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_P3_D65',
    ],
    [
        'colour.models.DON_RGB_4_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_DON_RGB_4',
    ],
    [
        'colour.models.DJI_D_GAMUT_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_DJI_D_GAMUT',
    ],
    [
        'colour.models.ECI_RGB_V2_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_ECI_RGB_V2',
    ],
    [
        'colour.models.EKTA_SPACE_PS_5_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_EKTA_SPACE_PS_5',
    ],
    [
        'colour.models.F_GAMUT_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_F_GAMUT',
    ],
    [
        'colour.models.FILMLIGHT_E_GAMUT_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_FILMLIGHT_E_GAMUT',
    ],
    [
        'colour.models.PROTUNE_NATIVE_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_PROTUNE_NATIVE',
    ],
    [
        'colour.models.BT470_525_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_BT470_525',
    ],
    [
        'colour.models.BT470_625_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_BT470_625',
    ],
    [
        'colour.models.BT709_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_BT709',
    ],
    [
        'colour.models.BT2020_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_BT2020',
    ],
    [
        'colour.models.MAX_RGB_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_MAX_RGB',
    ],
    [
        'colour.models.PAL_SECAM_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_PAL_SECAM',
    ],
    [
        'colour.models.RED_COLOR_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_RED_COLOR',
    ],
    [
        'colour.models.RED_COLOR_2_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_RED_COLOR_2',
    ],
    [
        'colour.models.RED_COLOR_3_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_RED_COLOR_3',
    ],
    [
        'colour.models.RED_COLOR_4_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_RED_COLOR_4',
    ],
    [
        'colour.models.DRAGON_COLOR_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_DRAGON_COLOR',
    ],
    [
        'colour.models.DRAGON_COLOR_2_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_DRAGON_COLOR_2',
    ],
    [
        'colour.models.RED_WIDE_GAMUT_RGB_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_RED_WIDE_GAMUT_RGB',
    ],
    [
        'colour.models.ROMM_RGB_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_ROMM_RGB',
    ],
    [
        'colour.models.RIMM_RGB_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_RIMM_RGB',
    ],
    [
        'colour.models.ERIMM_RGB_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_ERIMM_RGB',
    ],
    [
        'colour.models.PROPHOTO_RGB_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_PROPHOTO_RGB',
    ],
    [
        'colour.models.RUSSELL_RGB_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_RUSSELL_RGB',
    ],
    [
        'colour.models.SHARP_RGB_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_SHARP_RGB',
    ],
    [
        'colour.models.SMPTE_240M_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_SMPTE_240M',
    ],
    [
        'colour.models.SMPTE_C_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_SMPTE_C',
    ],
    [
        'colour.models.NTSC_1953_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_NTSC1953',
    ],
    [
        'colour.models.NTSC_1987_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_NTSC1987',
    ],
    [
        'colour.models.S_GAMUT_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_S_GAMUT',
    ],
    [
        'colour.models.S_GAMUT3_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_S_GAMUT3',
    ],
    [
        'colour.models.S_GAMUT3_CINE_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_S_GAMUT3_CINE',
    ],
    [
        'colour.models.VENICE_S_GAMUT3_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_VENICE_S_GAMUT3',
    ],
    [
        'colour.models.VENICE_S_GAMUT3_CINE_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_VENICE_S_GAMUT3_CINE',
    ],
    [
        'colour.models.sRGB_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_sRGB',
    ],
    [
        'colour.models.V_GAMUT_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_V_GAMUT',
    ],
    [
        'colour.models.XTREME_RGB_COLOURSPACE',
        'colour.models.RGB_COLOURSPACE_XTREME_RGB',
    ],
    [
        'colour.models.ACES_RICD',
        'colour.characterisation.MSDS_ACES_RICD',
    ],
    [
        'colour.models.oetf_BT2020',
        'colour.models.eotf_inverse_BT2020',
    ],
    [
        'colour.models.POINTER_GAMUT_BOUNDARIES',
        'colour.models.CCS_POINTER_GAMUT_BOUNDARY',
    ],
    [
        'colour.models.POINTER_GAMUT_DATA',
        'colour.models.DATA_POINTER_GAMUT_VOLUME',
    ],
    [
        'colour.models.POINTER_GAMUT_ILLUMINANT',
        'colour.models.CCS_ILLUMINANT_POINTER_GAMUT',
    ],
    [
        'colour.models.YCBCR_WEIGHTS',
        'colour.models.WEIGHTS_YCBCR',
    ],
    [
        'colour.models.RGB_to_RGB_matrix',
        'colour.models.matrix_RGB_to_RGB',
    ],
]

if not is_documentation_building():
    sys.modules['colour.models'] = models(sys.modules['colour.models'],
                                          build_API_changes(API_CHANGES))

    del ModuleAPI, is_documentation_building, build_API_changes, sys
