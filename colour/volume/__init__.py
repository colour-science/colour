from . import datasets
from .datasets import *  # noqa: F403
from .macadam_limits import is_within_macadam_limits
from .mesh import is_within_mesh_volume
from .pointer_gamut import is_within_pointer_gamut
from .spectrum import (
    XYZ_outer_surface,
    generate_pulse_waves,
    is_within_visible_spectrum,
    solid_RoschMacAdam,
)

# isort: split

from .rgb import (
    RGB_colourspace_limits,
    RGB_colourspace_pointer_gamut_coverage_MonteCarlo,
    RGB_colourspace_visible_spectrum_coverage_MonteCarlo,
    RGB_colourspace_volume_coverage_MonteCarlo,
    RGB_colourspace_volume_MonteCarlo,
)

__all__ = datasets.__all__
__all__ += [
    "is_within_macadam_limits",
]
__all__ += [
    "is_within_mesh_volume",
]
__all__ += [
    "is_within_pointer_gamut",
]
__all__ += [
    "XYZ_outer_surface",
    "generate_pulse_waves",
    "is_within_visible_spectrum",
    "solid_RoschMacAdam",
]
__all__ += [
    "RGB_colourspace_limits",
    "RGB_colourspace_pointer_gamut_coverage_MonteCarlo",
    "RGB_colourspace_visible_spectrum_coverage_MonteCarlo",
    "RGB_colourspace_volume_coverage_MonteCarlo",
    "RGB_colourspace_volume_MonteCarlo",
]
