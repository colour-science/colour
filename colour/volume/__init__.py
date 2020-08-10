# -*- coding: utf-8 -*-

from __future__ import absolute_import

import sys

from colour.utilities.deprecation import ModuleAPI, build_API_changes
from colour.utilities.documentation import is_documentation_building

from .datasets import *  # noqa
from . import datasets
from .macadam_limits import is_within_macadam_limits
from .mesh import is_within_mesh_volume
from .pointer_gamut import is_within_pointer_gamut
from .spectrum import (generate_pulse_waves, XYZ_outer_surface,
                       is_within_visible_spectrum)
from .rgb import (RGB_colourspace_limits, RGB_colourspace_volume_MonteCarlo,
                  RGB_colourspace_volume_coverage_MonteCarlo,
                  RGB_colourspace_pointer_gamut_coverage_MonteCarlo,
                  RGB_colourspace_visible_spectrum_coverage_MonteCarlo)

__all__ = []
__all__ += datasets.__all__
__all__ += ['is_within_macadam_limits']
__all__ += ['is_within_mesh_volume']
__all__ += ['is_within_pointer_gamut']
__all__ += [
    'generate_pulse_waves', 'XYZ_outer_surface', 'is_within_visible_spectrum'
]
__all__ += [
    'RGB_colourspace_limits', 'RGB_colourspace_volume_MonteCarlo',
    'RGB_colourspace_volume_coverage_MonteCarlo',
    'RGB_colourspace_pointer_gamut_coverage_MonteCarlo',
    'RGB_colourspace_visible_spectrum_coverage_MonteCarlo'
]


# ----------------------------------------------------------------------------#
# ---                API Changes and Deprecation Management                ---#
# ----------------------------------------------------------------------------#
class volume(ModuleAPI):
    def __getattr__(self, attribute):
        return super(volume, self).__getattr__(attribute)


# v0.3.16
API_CHANGES = {
    'ObjectRenamed': [[
        'colour.volume.ILLUMINANT_OPTIMAL_COLOUR_STIMULI',
        'colour.volume.OPTIMAL_COLOUR_STIMULI_ILLUMINANTS',
    ], ]
}
"""
Defines *colour.volume* sub-package API changes.

API_CHANGES : dict
"""

if not is_documentation_building():
    sys.modules['colour.volume'] = volume(sys.modules['colour.volume'],
                                          build_API_changes(API_CHANGES))

    del ModuleAPI, is_documentation_building, build_API_changes, sys
