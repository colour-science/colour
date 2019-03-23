# -*- coding: utf-8 -*-

from __future__ import absolute_import

from .dataset import *  # noqa
from . import dataset
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
__all__ += dataset.__all__
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
