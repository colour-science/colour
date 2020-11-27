# -*- coding: utf-8 -*-

from __future__ import absolute_import

import sys

from colour.utilities.deprecation import ModuleAPI, build_API_changes
from colour.utilities.documentation import is_documentation_building

from .hunt import (InductionFactors_Hunt, VIEWING_CONDITIONS_HUNT,
                   CAM_Specification_Hunt, XYZ_to_Hunt)
from .atd95 import CAM_Specification_ATD95, XYZ_to_ATD95
from .ciecam02 import (InductionFactors_CIECAM02, VIEWING_CONDITIONS_CIECAM02,
                       CAM_Specification_CIECAM02, XYZ_to_CIECAM02,
                       CIECAM02_to_XYZ)
from .cam16 import (InductionFactors_CAM16, VIEWING_CONDITIONS_CAM16,
                    CAM_Specification_CAM16, XYZ_to_CAM16, CAM16_to_XYZ)
from .llab import (InductionFactors_LLAB, VIEWING_CONDITIONS_LLAB,
                   CAM_Specification_LLAB, XYZ_to_LLAB)
from .nayatani95 import CAM_Specification_Nayatani95, XYZ_to_Nayatani95
from .rlab import (VIEWING_CONDITIONS_RLAB, D_FACTOR_RLAB,
                   CAM_Specification_RLAB, XYZ_to_RLAB)

__all__ = [
    'InductionFactors_Hunt', 'VIEWING_CONDITIONS_HUNT',
    'CAM_Specification_Hunt', 'XYZ_to_Hunt'
]
__all__ += ['CAM_Specification_ATD95', 'XYZ_to_ATD95']
__all__ += [
    'InductionFactors_CIECAM02', 'VIEWING_CONDITIONS_CIECAM02',
    'CAM_Specification_CIECAM02', 'XYZ_to_CIECAM02', 'CIECAM02_to_XYZ'
]
__all__ += [
    'InductionFactors_CAM16', 'VIEWING_CONDITIONS_CAM16',
    'CAM_Specification_CAM16', 'XYZ_to_CAM16', 'CAM16_to_XYZ'
]
__all__ += [
    'InductionFactors_LLAB', 'VIEWING_CONDITIONS_LLAB',
    'CAM_Specification_LLAB', 'XYZ_to_LLAB'
]
__all__ += ['CAM_Specification_Nayatani95', 'XYZ_to_Nayatani95']
__all__ += [
    'VIEWING_CONDITIONS_RLAB', 'D_FACTOR_RLAB', 'CAM_Specification_RLAB',
    'XYZ_to_RLAB'
]


# ----------------------------------------------------------------------------#
# ---                API Changes and Deprecation Management                ---#
# ----------------------------------------------------------------------------#
class appearance(ModuleAPI):
    def __getattr__(self, attribute):
        return super(appearance, self).__getattr__(attribute)


# v0.3.16
API_CHANGES = {
    'ObjectRenamed': [
        [
            'colour.appearance.ATD95_Specification',
            'colour.appearance.CAM_Specification_ATD95',
        ],
        [
            'colour.appearance.CAM16_InductionFactors',
            'colour.appearance.InductionFactors_CAM16',
        ],
        [
            'colour.appearance.CAM16_VIEWING_CONDITIONS',
            'colour.appearance.VIEWING_CONDITIONS_CAM16',
        ],
        [
            'colour.appearance.CAM16_Specification',
            'colour.appearance.CAM_Specification_CAM16',
        ],
        [
            'colour.appearance.CIECAM02_InductionFactors',
            'colour.appearance.InductionFactors_CIECAM02',
        ],
        [
            'colour.appearance.CIECAM02_VIEWING_CONDITIONS',
            'colour.appearance.VIEWING_CONDITIONS_CIECAM02',
        ],
        [
            'colour.appearance.CIECAM02_Specification',
            'colour.appearance.CAM_Specification_CIECAM02',
        ],
        [
            'colour.appearance.Hunt_InductionFactors',
            'colour.appearance.InductionFactors_Hunt',
        ],
        [
            'colour.appearance.HUNT_VIEWING_CONDITIONS',
            'colour.appearance.VIEWING_CONDITIONS_HUNT',
        ],
        [
            'colour.appearance.Hunt_Specification',
            'colour.appearance.CAM_Specification_Hunt',
        ],
        [
            'colour.appearance.LLAB_InductionFactors',
            'colour.appearance.InductionFactors_LLAB',
        ],
        [
            'colour.appearance.LLAB_VIEWING_CONDITIONS',
            'colour.appearance.VIEWING_CONDITIONS_LLAB',
        ],
        [
            'colour.appearance.LLAB_Specification',
            'colour.appearance.CAM_Specification_LLAB',
        ],
        [
            'colour.appearance.Nayatani95_Specification',
            'colour.appearance.CAM_Specification_Nayatani95',
        ],
        [
            'colour.appearance.RLAB_VIEWING_CONDITIONS',
            'colour.appearance.VIEWING_CONDITIONS_RLAB',
        ],
        [
            'colour.appearance.RLAB_D_FACTOR',
            'colour.appearance.D_FACTOR_RLAB',
        ],
        [
            'colour.appearance.RLAB_Specification',
            'colour.appearance.CAM_Specification_RLAB',
        ],
    ]
}
"""
Defines *colour.appearance* sub-package API changes.

API_CHANGES : dict
"""

if not is_documentation_building():
    sys.modules['colour.appearance'] = appearance(
        sys.modules['colour.appearance'], build_API_changes(API_CHANGES))

    del ModuleAPI, is_documentation_building, build_API_changes, sys
