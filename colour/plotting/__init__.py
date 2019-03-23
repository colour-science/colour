# -*- coding: utf-8 -*-

from __future__ import absolute_import

import sys

from colour.utilities.deprecation import ModuleAPI, Renamed
from colour.utilities.documentation import is_documentation_building

from .dataset import *  # noqa
from . import dataset
from .common import (COLOUR_STYLE_CONSTANTS, COLOUR_ARROW_STYLE, colour_style,
                     override_style, XYZ_to_plotting_colourspace, ColourSwatch,
                     colour_cycle, artist, camera, render, wrap_label,
                     label_rectangles, uniform_axes3d, filter_passthrough,
                     filter_RGB_colourspaces, filter_cmfs, filter_illuminants,
                     filter_colour_checkers, plot_single_colour_swatch,
                     plot_multi_colour_swatches, plot_single_function,
                     plot_multi_functions, plot_image)
from .blindness import plot_cvd_simulation_Machado2009
from .colorimetry import (
    plot_single_sd, plot_multi_sds, plot_single_cmfs, plot_multi_cmfs,
    plot_single_illuminant_sd, plot_multi_illuminant_sds,
    plot_visible_spectrum, plot_single_lightness_function,
    plot_multi_lightness_functions, plot_single_luminance_function,
    plot_multi_luminance_functions, plot_blackbody_spectral_radiance,
    plot_blackbody_colours)
from .characterisation import (plot_single_colour_checker,
                               plot_multi_colour_checkers)
from .diagrams import (plot_chromaticity_diagram_CIE1931,
                       plot_chromaticity_diagram_CIE1960UCS,
                       plot_chromaticity_diagram_CIE1976UCS,
                       plot_sds_in_chromaticity_diagram_CIE1931,
                       plot_sds_in_chromaticity_diagram_CIE1960UCS,
                       plot_sds_in_chromaticity_diagram_CIE1976UCS)
from .corresponding import plot_corresponding_chromaticities_prediction
from .geometry import quad, grid, cube
from .models import (
    plot_pointer_gamut, plot_RGB_colourspaces_in_chromaticity_diagram_CIE1931,
    plot_RGB_colourspaces_in_chromaticity_diagram_CIE1960UCS,
    plot_RGB_colourspaces_in_chromaticity_diagram_CIE1976UCS,
    plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931,
    plot_RGB_chromaticities_in_chromaticity_diagram_CIE1960UCS,
    plot_RGB_chromaticities_in_chromaticity_diagram_CIE1976UCS,
    plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1931,
    plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1960UCS,
    plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1976UCS,
    plot_single_cctf, plot_multi_cctfs)
from .notation import (plot_single_munsell_value_function,
                       plot_multi_munsell_value_functions)
from .phenomena import plot_single_sd_rayleigh_scattering, plot_the_blue_sky
from .quality import (plot_single_sd_colour_rendering_index_bars,
                      plot_multi_sds_colour_rendering_indexes_bars,
                      plot_single_sd_colour_quality_scale_bars,
                      plot_multi_sds_colour_quality_scales_bars)
from .temperature import (
    plot_planckian_locus_in_chromaticity_diagram_CIE1931,
    plot_planckian_locus_in_chromaticity_diagram_CIE1960UCS)
from .volume import plot_RGB_colourspaces_gamuts, plot_RGB_scatter

__all__ = []
__all__ += dataset.__all__
__all__ += [
    'COLOUR_STYLE_CONSTANTS', 'COLOUR_ARROW_STYLE', 'colour_style',
    'override_style', 'XYZ_to_plotting_colourspace', 'ColourSwatch',
    'colour_cycle', 'artist', 'camera', 'render', 'wrap_label',
    'label_rectangles', 'uniform_axes3d', 'filter_passthrough',
    'filter_RGB_colourspaces', 'filter_cmfs', 'filter_illuminants',
    'filter_colour_checkers', 'plot_single_colour_swatch',
    'plot_multi_colour_swatches', 'plot_single_function',
    'plot_multi_functions', 'plot_image'
]
__all__ += ['plot_cvd_simulation_Machado2009']
__all__ += [
    'plot_single_sd', 'plot_multi_sds', 'plot_single_cmfs', 'plot_multi_cmfs',
    'plot_single_illuminant_sd', 'plot_multi_illuminant_sds',
    'plot_visible_spectrum', 'plot_single_lightness_function',
    'plot_multi_lightness_functions', 'plot_single_luminance_function',
    'plot_multi_luminance_functions', 'plot_blackbody_spectral_radiance',
    'plot_blackbody_colours'
]
__all__ += ['plot_single_colour_checker', 'plot_multi_colour_checkers']
__all__ += [
    'plot_chromaticity_diagram_CIE1931',
    'plot_chromaticity_diagram_CIE1960UCS',
    'plot_chromaticity_diagram_CIE1976UCS',
    'plot_sds_in_chromaticity_diagram_CIE1931',
    'plot_sds_in_chromaticity_diagram_CIE1960UCS',
    'plot_sds_in_chromaticity_diagram_CIE1976UCS'
]
__all__ += ['plot_corresponding_chromaticities_prediction']
__all__ += ['quad', 'grid', 'cube']
__all__ += [
    'plot_pointer_gamut',
    'plot_RGB_colourspaces_in_chromaticity_diagram_CIE1931',
    'plot_RGB_colourspaces_in_chromaticity_diagram_CIE1960UCS',
    'plot_RGB_colourspaces_in_chromaticity_diagram_CIE1976UCS',
    'plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931',
    'plot_RGB_chromaticities_in_chromaticity_diagram_CIE1960UCS',
    'plot_RGB_chromaticities_in_chromaticity_diagram_CIE1976UCS',
    'plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1931',
    'plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1960UCS',
    'plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1976UCS',
    'plot_single_cctf', 'plot_multi_cctfs'
]
__all__ += [
    'plot_single_munsell_value_function', 'plot_multi_munsell_value_functions'
]
__all__ += ['plot_single_sd_rayleigh_scattering', 'plot_the_blue_sky']
__all__ += [
    'plot_single_sd_colour_rendering_index_bars',
    'plot_multi_sds_colour_rendering_indexes_bars',
    'plot_single_sd_colour_quality_scale_bars',
    'plot_multi_sds_colour_quality_scales_bars'
]
__all__ += [
    'plot_planckian_locus_in_chromaticity_diagram_CIE1931',
    'plot_planckian_locus_in_chromaticity_diagram_CIE1960UCS'
]
__all__ += ['plot_RGB_colourspaces_gamuts', 'plot_RGB_scatter']


# ----------------------------------------------------------------------------#
# ---                API Changes and Deprecation Management                ---#
# ----------------------------------------------------------------------------#
class plotting(ModuleAPI):
    def __getattr__(self, attribute):
        return super(plotting, self).__getattr__(attribute)


# v0.3.11
API_CHANGES = {
    'Renamed': [
        [
            'colour.plotting.CIE_1931_chromaticity_diagram_plot',
            'colour.plotting.plot_chromaticity_diagram_CIE1931',
        ],
        [
            'colour.plotting.CIE_1960_UCS_chromaticity_diagram_plot',
            'colour.plotting.plot_chromaticity_diagram_CIE1960UCS',
        ],
        [
            'colour.plotting.CIE_1976_UCS_chromaticity_diagram_plot',
            'colour.plotting.plot_chromaticity_diagram_CIE1976UCS',
        ],
        [
            'colour.plotting.RGB_chromaticity_coordinates_CIE_1931_chromaticity_diagram_plot',  # noqa
            'colour.plotting.plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931',  # noqa
        ],
        [
            'colour.plotting.RGB_chromaticity_coordinates_CIE_1960_UCS_chromaticity_diagram_plot',  # noqa
            'colour.plotting.plot_RGB_chromaticities_in_chromaticity_diagram_CIE1960UCS',  # noqa
        ],
        [
            'colour.plotting.RGB_chromaticity_coordinates_CIE_1976_UCS_chromaticity_diagram_plot',  # noqa
            'colour.plotting.plot_RGB_chromaticities_in_chromaticity_diagram_CIE1976UCS',  # noqa
        ],
        [
            'colour.plotting.RGB_colourspaces_CIE_1931_chromaticity_diagram_plot',  # noqa
            'colour.plotting.plot_RGB_colourspaces_in_chromaticity_diagram_CIE1931',  # noqa
        ],
        [
            'colour.plotting.RGB_colourspaces_CIE_1960_UCS_chromaticity_diagram_plot',  # noqa
            'colour.plotting.plot_RGB_colourspaces_in_chromaticity_diagram_CIE1960UCS',  # noqa
        ],
        [
            'colour.plotting.RGB_colourspaces_CIE_1976_UCS_chromaticity_diagram_plot',  # noqa
            'colour.plotting.plot_RGB_colourspaces_in_chromaticity_diagram_CIE1976UCS',  # noqa
        ],
        [
            'colour.plotting.planckian_locus_CIE_1931_chromaticity_diagram_plot',  # noqa
            'colour.plotting.plot_planckian_locus_in_chromaticity_diagram_CIE1931',  # noqa
        ],
        [
            'colour.plotting.planckian_locus_CIE_1960_UCS_chromaticity_diagram_plot',  # noqa
            'colour.plotting.plot_planckian_locus_in_chromaticity_diagram_CIE1960UCS',  # noqa
        ],
        [
            'colour.plotting.spds_CIE_1931_chromaticity_diagram_plot',
            'colour.plotting.plot_sds_in_chromaticity_diagram_CIE1931',
        ],
        [
            'colour.plotting.spds_CIE_1960_UCS_chromaticity_diagram_plot',
            'colour.plotting.plot_sds_in_chromaticity_diagram_CIE1960UCS',
        ],
        [
            'colour.plotting.spds_CIE_1976_UCS_chromaticity_diagram_plot',
            'colour.plotting.plot_sds_in_chromaticity_diagram_CIE1976UCS'
        ],
    ]
}
"""
Defines *colour.plotting* sub-package API changes.

API_CHANGES : dict
"""

# v0.3.12
API_CHANGES['Renamed'] = API_CHANGES['Renamed'] + [
    [
        'colour.plotting.RGB_chromaticity_coordinates_chromaticity_diagram_plot',  # noqa
        'colour.plotting.plot_RGB_chromaticities_in_chromaticity_diagram',
    ],
    [
        'colour.plotting.RGB_chromaticity_coordinates_chromaticity_diagram_plot_CIE1931',  # noqa
        'colour.plotting.plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931',  # noqa
    ],
    [
        'colour.plotting.RGB_chromaticity_coordinates_chromaticity_diagram_plot_CIE1960UCS',  # noqa
        'colour.plotting.plot_RGB_chromaticities_in_chromaticity_diagram_CIE1960UCS',  # noqa
    ],
    [
        'colour.plotting.RGB_chromaticity_coordinates_chromaticity_diagram_plot_CIE1976UCS',  # noqa
        'colour.plotting.plot_RGB_chromaticities_in_chromaticity_diagram_CIE1976UCS',  # noqa
    ],
    [
        'colour.plotting.RGB_colourspaces_chromaticity_diagram_plot',
        'colour.plotting.plot_RGB_colourspaces_in_chromaticity_diagram',
    ],
    [
        'colour.plotting.RGB_colourspaces_chromaticity_diagram_plot_CIE1931',
        'colour.plotting.plot_RGB_colourspaces_in_chromaticity_diagram_CIE1931',  # noqa
    ],
    [
        'colour.plotting.RGB_colourspaces_chromaticity_diagram_plot_CIE1960UCS',  # noqa
        'colour.plotting.plot_RGB_colourspaces_in_chromaticity_diagram_CIE1960UCS',  # noqa
    ],
    [
        'colour.plotting.RGB_colourspaces_chromaticity_diagram_plot_CIE1976UCS',  # noqa
        'colour.plotting.plot_RGB_colourspaces_in_chromaticity_diagram_CIE1976UCS',  # noqa
    ],
    [
        'colour.plotting.RGB_colourspaces_gamuts_plot',
        'colour.plotting.plot_RGB_colourspaces_gamuts',
    ],
    [
        'colour.plotting.RGB_scatter_plot',
        'colour.plotting.plot_RGB_scatter',
    ],
    [
        'colour.plotting.blackbody_colours_plot',
        'colour.plotting.plot_blackbody_colours',
    ],
    [
        'colour.plotting.blackbody_spectral_radiance_plot',
        'colour.plotting.plot_blackbody_spectral_radiance',
    ],
    [
        'colour.plotting.chromaticity_diagram_colours_plot',
        'colour.plotting.plot_chromaticity_diagram_colours',
    ],
    [
        'colour.plotting.chromaticity_diagram_plot',
        'colour.plotting.plot_chromaticity_diagram',
    ],
    [
        'colour.plotting.chromaticity_diagram_plot_CIE1931',
        'colour.plotting.plot_chromaticity_diagram_CIE1931',
    ],
    [
        'colour.plotting.chromaticity_diagram_plot_CIE1960UCS',
        'colour.plotting.plot_chromaticity_diagram_CIE1960UCS',
    ],
    [
        'colour.plotting.chromaticity_diagram_plot_CIE1976UCS',
        'colour.plotting.plot_chromaticity_diagram_CIE1976UCS',
    ],
    [
        'colour.plotting.colour_plotting_defaults',
        'colour.plotting.colour_style',
    ],
    [
        'colour.plotting.colour_quality_bars_plot',
        'colour.plotting.plot_colour_quality_bars',
    ],
    [
        'colour.plotting.corresponding_chromaticities_prediction_plot',
        'colour.plotting.plot_corresponding_chromaticities_prediction',
    ],
    [
        'colour.plotting.cvd_simulation_Machado2009_plot',
        'colour.plotting.plot_cvd_simulation_Machado2009',
    ],
    [
        'colour.plotting.equal_axes3d',
        'colour.plotting.uniform_axes3d',
    ],
    [
        'colour.plotting.get_RGB_colourspace',
        'colour.plotting.filter_RGB_colourspaces',
    ],
    [
        'colour.plotting.get_cmfs',
        'colour.plotting.filter_cmfs',
    ],
    [
        'colour.plotting.get_illuminant',
        'colour.plotting.filter_illuminants',
    ],
    [
        'colour.plotting.image_plot',
        'colour.plotting.plot_image',
    ],
    [
        'colour.plotting.multi_cctf_plot',
        'colour.plotting.plot_multi_cctfs',
    ],
    [
        'colour.plotting.multi_cmfs_plot',
        'colour.plotting.plot_multi_cmfs',
    ],
    [
        'colour.plotting.multi_colour_checker_plot',
        'colour.plotting.plot_multi_colour_checkers',
    ],
    [
        'colour.plotting.multi_colour_swatch_plot',
        'colour.plotting.plot_multi_colour_swatches',
    ],
    [
        'colour.plotting.multi_illuminant_spd_plot',
        'colour.plotting.plot_multi_illuminant_sds',
    ],
    [
        'colour.plotting.multi_lightness_function_plot',
        'colour.plotting.plot_multi_lightness_functions',
    ],
    [
        'colour.plotting.multi_munsell_value_function_plot',
        'colour.plotting.plot_multi_munsell_value_functions',
    ],
    [
        'colour.plotting.multi_spd_colour_quality_scale_bars_plot',
        'colour.plotting.plot_multi_sds_colour_quality_scales_bars',
    ],
    [
        'colour.plotting.multi_spd_colour_rendering_index_bars_plot',
        'colour.plotting.plot_multi_sds_colour_rendering_indexes_bars',
    ],
    [
        'colour.plotting.multi_spd_plot',
        'colour.plotting.plot_multi_sds',
    ],
    [
        'colour.plotting.planckian_locus_chromaticity_diagram_plot',
        'colour.plotting.plot_planckian_locus_in_chromaticity_diagram',
    ],
    [
        'colour.plotting.planckian_locus_chromaticity_diagram_plot_CIE1931',
        'colour.plotting.plot_planckian_locus_in_chromaticity_diagram_CIE1931',
    ],
    [
        'colour.plotting.planckian_locus_chromaticity_diagram_plot_CIE1960UCS',
        'colour.plotting.plot_planckian_locus_in_chromaticity_diagram_CIE1960UCS',  # noqa
    ],
    [
        'colour.plotting.planckian_locus_plot',
        'colour.plotting.plot_planckian_locus',
    ],
    [
        'colour.plotting.single_cctf_plot',
        'colour.plotting.plot_single_cctf',
    ],
    [
        'colour.plotting.single_cmfs_plot',
        'colour.plotting.plot_single_cmfs',
    ],
    [
        'colour.plotting.single_colour_checker_plot',
        'colour.plotting.plot_single_colour_checker',
    ],
    [
        'colour.plotting.single_colour_swatch_plot',
        'colour.plotting.plot_single_colour_swatch',
    ],
    [
        'colour.plotting.single_illuminant_spd_plot',
        'colour.plotting.plot_single_illuminant_sd',
    ],
    [
        'colour.plotting.single_lightness_function_plot',
        'colour.plotting.plot_single_lightness_function',
    ],
    [
        'colour.plotting.single_munsell_value_function_plot',
        'colour.plotting.plot_single_munsell_value_function',
    ],
    [
        'colour.plotting.single_spd_colour_quality_scale_bars_plot',
        'colour.plotting.plot_single_sd_colour_quality_scale_bars',
    ],
    [
        'colour.plotting.single_spd_colour_rendering_index_bars_plot',
        'colour.plotting.plot_single_sd_colour_rendering_index_bars',
    ],
    [
        'colour.plotting.single_spd_plot',
        'colour.plotting.plot_single_sd',
    ],
    [
        'colour.plotting.single_spd_rayleigh_scattering_plot',
        'colour.plotting.plot_single_sd_rayleigh_scattering',
    ],
    [
        'colour.plotting.spds_chromaticity_diagram_plot',
        'colour.plotting.plot_sds_in_chromaticity_diagram',
    ],
    [
        'colour.plotting.spds_chromaticity_diagram_plot_CIE1931',
        'colour.plotting.plot_sds_in_chromaticity_diagram_CIE1931',
    ],
    [
        'colour.plotting.spds_chromaticity_diagram_plot_CIE1960UCS',
        'colour.plotting.plot_sds_in_chromaticity_diagram_CIE1960UCS',
    ],
    [
        'colour.plotting.spds_chromaticity_diagram_plot_CIE1976UCS',
        'colour.plotting.plot_sds_in_chromaticity_diagram_CIE1976UCS',
    ],
    [
        'colour.plotting.spectral_locus_plot',
        'colour.plotting.plot_spectral_locus',
    ],
    [
        'colour.plotting.the_blue_sky_plot',
        'colour.plotting.plot_the_blue_sky',
    ],
    [
        'colour.plotting.visible_spectrum_plot',
        'colour.plotting.plot_visible_spectrum'
    ],
]


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

    sys.modules['colour.plotting'] = plotting(sys.modules['colour.plotting'],
                                              API_CHANGES)

    del sys
