# -*- coding: utf-8 -*-

from __future__ import absolute_import

import sys

from colour.utilities.deprecation import ModuleAPI, Renamed
from colour.utilities.documentation import is_documentation_building

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
from .ipt import XYZ_to_IPT, IPT_to_XYZ, IPT_hue_angle
from .jzazbz import XYZ_to_JzAzBz, JzAzBz_to_XYZ
from .hdr_ipt import HDR_IPT_METHODS, XYZ_to_hdr_IPT, hdr_IPT_to_XYZ
from .osa_ucs import XYZ_to_OSA_UCS, OSA_UCS_to_XYZ
from .common import (COLOURSPACE_MODELS, COLOURSPACE_MODELS_AXIS_LABELS,
                     XYZ_to_colourspace_model)
from .datasets import *  # noqa
from . import datasets
from .rgb import *  # noqa
from . import rgb

__all__ = [
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
__all__ += ['XYZ_to_IPT', 'IPT_to_XYZ', 'IPT_hue_angle']
__all__ += ['XYZ_to_JzAzBz', 'JzAzBz_to_XYZ']
__all__ += ['HDR_IPT_METHODS', 'XYZ_to_hdr_IPT', 'hdr_IPT_to_XYZ']
__all__ += ['XYZ_to_OSA_UCS', 'OSA_UCS_to_XYZ']
__all__ += [
    'COLOURSPACE_MODELS', 'COLOURSPACE_MODELS_AXIS_LABELS',
    'XYZ_to_colourspace_model'
]
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
    'FutureRemove': [
        'colour.models.XYZ_to_colourspace_model',
    ],
    'Renamed': [
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


def _setup_api_changes():
    """
    Setups *Colour* API changes.
    """

    global API_CHANGES

    for renamed in API_CHANGES['Renamed']:
        name, access = renamed
        API_CHANGES[name.split('.')[-1]] = Renamed(name, access)  # noqa
    API_CHANGES.pop('Renamed')


if not is_documentation_building():
    _setup_api_changes()

    del ModuleAPI
    del Renamed
    del is_documentation_building
    del _setup_api_changes

    sys.modules['colour.models'] = models(sys.modules['colour.models'],
                                          API_CHANGES)

    del sys
