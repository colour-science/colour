# -*- coding: utf-8 -*-

from __future__ import absolute_import

import sys

from colour.utilities.deprecation import ModuleAPI, Renamed
from colour.utilities.documentation import is_documentation_building

from .dataset import *  # noqa
from . import dataset
from .common import (
    PLOTTING_RESOURCES_DIRECTORY, DEFAULT_FIGURE_ASPECT_RATIO,
    DEFAULT_FIGURE_WIDTH, DEFAULT_FIGURE_HEIGHT, DEFAULT_FIGURE_SIZE,
    DEFAULT_FONT_SIZE, DEFAULT_COLOUR_CYCLE, DEFAULT_HATCH_PATTERNS,
    DEFAULT_PARAMETERS, DEFAULT_PLOTTING_ILLUMINANT,
    DEFAULT_PLOTTING_ENCODING_CCTF, colour_plotting_defaults, ColourSwatch,
    colour_cycle, canvas, camera, boundaries, decorate, display, render,
    label_rectangles, equal_axes3d, get_RGB_colourspace, get_cmfs,
    get_illuminant, single_colour_swatch_plot, multi_colour_swatches_plot,
    image_plot)
from .colorimetry import (
    single_spd_plot, multi_spd_plot, single_cmfs_plot, multi_cmfs_plot,
    single_illuminant_relative_spd_plot, multi_illuminants_relative_spd_plot,
    visible_spectrum_plot, single_lightness_function_plot,
    multi_lightness_function_plot, blackbody_spectral_radiance_plot,
    blackbody_colours_plot)
from .characterisation import colour_checker_plot
from .diagrams import (chromaticity_diagram_plot_CIE1931,
                       chromaticity_diagram_plot_CIE1960UCS,
                       chromaticity_diagram_plot_CIE1976UCS,
                       spds_chromaticity_diagram_plot_CIE1931,
                       spds_chromaticity_diagram_plot_CIE1960UCS,
                       spds_chromaticity_diagram_plot_CIE1976UCS)
from .corresponding import corresponding_chromaticities_prediction_plot
from .geometry import (quad, grid, cube)
from .models import (
    RGB_colourspaces_chromaticity_diagram_plot_CIE1931,
    RGB_colourspaces_chromaticity_diagram_plot_CIE1960UCS,
    RGB_colourspaces_chromaticity_diagram_plot_CIE1976UCS,
    RGB_chromaticity_coordinates_chromaticity_diagram_plot_CIE1931,
    RGB_chromaticity_coordinates_chromaticity_diagram_plot_CIE1960UCS,
    RGB_chromaticity_coordinates_chromaticity_diagram_plot_CIE1976UCS,
    single_cctf_plot, multi_cctf_plot)
from .notation import (single_munsell_value_function_plot,
                       multi_munsell_value_function_plot)
from .phenomena import single_rayleigh_scattering_spd_plot, the_blue_sky_plot
from .quality import (single_spd_colour_rendering_index_bars_plot,
                      multi_spd_colour_rendering_index_bars_plot,
                      single_spd_colour_quality_scale_bars_plot,
                      multi_spd_colour_quality_scale_bars_plot)
from .temperature import (planckian_locus_chromaticity_diagram_plot_CIE1931,
                          planckian_locus_chromaticity_diagram_plot_CIE1960UCS)
from .volume import RGB_colourspaces_gamuts_plot, RGB_scatter_plot

__all__ = []
__all__ += dataset.__all__
__all__ += [
    'PLOTTING_RESOURCES_DIRECTORY', 'DEFAULT_FIGURE_ASPECT_RATIO',
    'DEFAULT_FIGURE_WIDTH', 'DEFAULT_FIGURE_HEIGHT', 'DEFAULT_FIGURE_SIZE',
    'DEFAULT_FONT_SIZE', 'DEFAULT_COLOUR_CYCLE', 'DEFAULT_HATCH_PATTERNS',
    'DEFAULT_PARAMETERS', 'DEFAULT_PLOTTING_ILLUMINANT',
    'DEFAULT_PLOTTING_ENCODING_CCTF', 'colour_plotting_defaults',
    'ColourSwatch', 'colour_cycle', 'canvas', 'camera', 'boundaries',
    'decorate', 'display', 'render', 'label_rectangles', 'equal_axes3d',
    'get_RGB_colourspace', 'get_cmfs', 'get_illuminant',
    'single_colour_swatch_plot', 'multi_colour_swatches_plot', 'image_plot'
]
__all__ += [
    'single_spd_plot', 'multi_spd_plot', 'single_cmfs_plot', 'multi_cmfs_plot',
    'single_illuminant_relative_spd_plot',
    'multi_illuminants_relative_spd_plot', 'visible_spectrum_plot',
    'single_lightness_function_plot', 'multi_lightness_function_plot',
    'blackbody_spectral_radiance_plot', 'blackbody_colours_plot'
]
__all__ += ['colour_checker_plot']
__all__ += [
    'chromaticity_diagram_plot_CIE1931',
    'chromaticity_diagram_plot_CIE1960UCS',
    'chromaticity_diagram_plot_CIE1976UCS',
    'spds_chromaticity_diagram_plot_CIE1931',
    'spds_chromaticity_diagram_plot_CIE1960UCS',
    'spds_chromaticity_diagram_plot_CIE1976UCS'
]
__all__ += ['corresponding_chromaticities_prediction_plot']
__all__ += ['quad', 'grid', 'cube']
__all__ += [
    'RGB_colourspaces_chromaticity_diagram_plot_CIE1931',
    'RGB_colourspaces_chromaticity_diagram_plot_CIE1960UCS',
    'RGB_colourspaces_chromaticity_diagram_plot_CIE1976UCS',
    'RGB_chromaticity_coordinates_chromaticity_diagram_plot_CIE1931',
    'RGB_chromaticity_coordinates_chromaticity_diagram_plot_CIE1960UCS',
    'RGB_chromaticity_coordinates_chromaticity_diagram_plot_CIE1976UCS',
    'single_cctf_plot', 'multi_cctf_plot'
]
__all__ += [
    'single_munsell_value_function_plot', 'multi_munsell_value_function_plot'
]
__all__ += ['single_rayleigh_scattering_spd_plot', 'the_blue_sky_plot']
__all__ += [
    'single_spd_colour_rendering_index_bars_plot',
    'multi_spd_colour_rendering_index_bars_plot',
    'single_spd_colour_quality_scale_bars_plot',
    'multi_spd_colour_quality_scale_bars_plot'
]
__all__ += [
    'planckian_locus_chromaticity_diagram_plot_CIE1931',
    'planckian_locus_chromaticity_diagram_plot_CIE1960UCS'
]
__all__ += ['RGB_colourspaces_gamuts_plot', 'RGB_scatter_plot']


# ----------------------------------------------------------------------------#
# ---                API Changes and Deprecation Management                ---#
# ----------------------------------------------------------------------------#
class plotting(ModuleAPI):
    def __getattr__(self, attribute):
        return super(plotting, self).__getattr__(attribute)


API_CHANGES = {
    'Renamed': [
        [
            'colour.plotting.CIE_1931_chromaticity_diagram_plot',
            'colour.plotting.chromaticity_diagram_plot_CIE1931',
        ],
        [
            'colour.plotting.CIE_1960_UCS_chromaticity_diagram_plot',
            'colour.plotting.chromaticity_diagram_plot_CIE1960UCS',
        ],
        [
            'colour.plotting.CIE_1976_UCS_chromaticity_diagram_plot',
            'colour.plotting.chromaticity_diagram_plot_CIE1976UCS',
        ],
        [
            'colour.plotting.spds_CIE_1931_chromaticity_diagram_plot',
            'colour.plotting.spds_chromaticity_diagram_plot_CIE1931',
        ],
        [
            'colour.plotting.spds_CIE_1960_UCS_chromaticity_diagram_plot',
            'colour.plotting.spds_chromaticity_diagram_plot_CIE1960UCS',
        ],
        [
            'colour.plotting.spds_CIE_1976_UCS_chromaticity_diagram_plot',
            'colour.plotting.spds_chromaticity_diagram_plot_CIE1976UCS',
        ],
        [
            'colour.plotting.'
            'RGB_colourspaces_CIE_1931_chromaticity_diagram_plot',
            'colour.plotting.'
            'RGB_colourspaces_chromaticity_diagram_plot_CIE1931',
        ],
        [
            'colour.plotting.'
            'RGB_colourspaces_CIE_1960_UCS_chromaticity_diagram_plot',
            'colour.plotting.'
            'RGB_colourspaces_chromaticity_diagram_plot_CIE1960UCS',
        ],
        [
            'colour.plotting.'
            'RGB_colourspaces_CIE_1976_UCS_chromaticity_diagram_plot',
            'colour.plotting.'
            'RGB_colourspaces_chromaticity_diagram_plot_CIE1976UCS',
        ],
        [
            'colour.plotting.'
            'RGB_chromaticity_coordinates_CIE_1931_chromaticity_diagram_plot',
            'colour.plotting.'
            'RGB_chromaticity_coordinates_chromaticity_diagram_plot_CIE1931',
        ],
        [
            'colour.plotting.RGB_chromaticity_coordinates_CIE_1960_UCS_chromaticity_diagram_plot',  # noqa
            'colour.plotting.'
            'RGB_chromaticity_coordinates_chromaticity_diagram_plot_CIE1960UCS',  # noqa
        ],
        [
            'colour.plotting.RGB_chromaticity_coordinates_CIE_1976_UCS_chromaticity_diagram_plot',  # noqa
            'colour.plotting.'
            'RGB_chromaticity_coordinates_chromaticity_diagram_plot_CIE1976UCS',  # noqa
        ],
        [
            'colour.plotting.'
            'planckian_locus_CIE_1931_chromaticity_diagram_plot',
            'colour.plotting.'
            'planckian_locus_chromaticity_diagram_plot_CIE1931',
        ],
        [
            'colour.plotting.'
            'planckian_locus_CIE_1960_UCS_chromaticity_diagram_plot',
            'colour.plotting.'
            'planckian_locus_chromaticity_diagram_plot_CIE1960UCS',
        ],
    ]
}
"""
Defines *colour.plotting* sub-package API changes.

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
    del ModuleAPI
    del Renamed
    del is_documentation_building
    del _setup_api_changes

    sys.modules['colour.plotting'] = plotting(sys.modules['colour.plotting'],
                                              API_CHANGES)

    del sys
