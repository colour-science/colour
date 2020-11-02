# -*- coding: utf-8 -*-

from __future__ import absolute_import

from colour.utilities import is_matplotlib_installed

is_matplotlib_installed(raise_exception=True)

import sys  # noqa

from colour.utilities.deprecation import ModuleAPI, build_API_changes  # noqa
from colour.utilities.documentation import is_documentation_building  # noqa

from .datasets import *  # noqa
from . import datasets  # noqa
from .common import (  # noqa
    CONSTANTS_COLOUR_STYLE, CONSTANTS_ARROW_STYLE, colour_style,
    override_style, XYZ_to_plotting_colourspace, ColourSwatch, colour_cycle,
    artist, camera, render, label_rectangles, uniform_axes3d,
    filter_passthrough, filter_RGB_colourspaces, filter_cmfs,
    filter_illuminants, filter_colour_checkers, update_settings_collection,
    plot_single_colour_swatch, plot_multi_colour_swatches,
    plot_single_function, plot_multi_functions, plot_image)
from .blindness import plot_cvd_simulation_Machado2009  # noqa
from .colorimetry import (  # noqa
    plot_single_sd, plot_multi_sds, plot_single_cmfs, plot_multi_cmfs,
    plot_single_illuminant_sd, plot_multi_illuminant_sds,
    plot_visible_spectrum, plot_single_lightness_function,
    plot_multi_lightness_functions, plot_single_luminance_function,
    plot_multi_luminance_functions, plot_blackbody_spectral_radiance,
    plot_blackbody_colours)
from .characterisation import (  # noqa
    plot_single_colour_checker, plot_multi_colour_checkers)
from .diagrams import (  # noqa
    plot_chromaticity_diagram_CIE1931, plot_chromaticity_diagram_CIE1960UCS,
    plot_chromaticity_diagram_CIE1976UCS,
    plot_sds_in_chromaticity_diagram_CIE1931,
    plot_sds_in_chromaticity_diagram_CIE1960UCS,
    plot_sds_in_chromaticity_diagram_CIE1976UCS)
from .corresponding import plot_corresponding_chromaticities_prediction  # noqa
from .graph import plot_automatic_colour_conversion_graph  # noqa
from .models import (  # noqa
    common_colourspace_model_axis_reorder, plot_pointer_gamut,
    plot_RGB_colourspaces_in_chromaticity_diagram_CIE1931,
    plot_RGB_colourspaces_in_chromaticity_diagram_CIE1960UCS,
    plot_RGB_colourspaces_in_chromaticity_diagram_CIE1976UCS,
    plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931,
    plot_RGB_chromaticities_in_chromaticity_diagram_CIE1960UCS,
    plot_RGB_chromaticities_in_chromaticity_diagram_CIE1976UCS,
    plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1931,
    plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1960UCS,
    plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1976UCS,
    plot_single_cctf, plot_multi_cctfs, plot_constant_hue_loci)
from .notation import (  # noqa
    plot_single_munsell_value_function, plot_multi_munsell_value_functions)
from .phenomena import (  # noqa
    plot_single_sd_rayleigh_scattering, plot_the_blue_sky)
from .quality import (  # noqa
    plot_single_sd_colour_rendering_index_bars,
    plot_multi_sds_colour_rendering_indexes_bars,
    plot_single_sd_colour_quality_scale_bars,
    plot_multi_sds_colour_quality_scales_bars)
from .temperature import (  # noqa
    plot_planckian_locus_in_chromaticity_diagram_CIE1931,
    plot_planckian_locus_in_chromaticity_diagram_CIE1960UCS)
from .tm3018 import plot_single_sd_colour_rendition_report  # noqa
from .volume import plot_RGB_colourspaces_gamuts, plot_RGB_scatter  # noqa

__all__ = []
__all__ += datasets.__all__
__all__ += [
    'CONSTANTS_COLOUR_STYLE', 'CONSTANTS_ARROW_STYLE', 'colour_style',
    'override_style', 'XYZ_to_plotting_colourspace', 'ColourSwatch',
    'colour_cycle', 'artist', 'camera', 'render', 'label_rectangles',
    'uniform_axes3d', 'filter_passthrough', 'filter_RGB_colourspaces',
    'filter_cmfs', 'filter_illuminants', 'filter_colour_checkers',
    'update_settings_collection', 'plot_single_colour_swatch',
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
__all__ += ['plot_automatic_colour_conversion_graph']
__all__ += [
    'common_colourspace_model_axis_reorder', 'plot_pointer_gamut',
    'plot_RGB_colourspaces_in_chromaticity_diagram_CIE1931',
    'plot_RGB_colourspaces_in_chromaticity_diagram_CIE1960UCS',
    'plot_RGB_colourspaces_in_chromaticity_diagram_CIE1976UCS',
    'plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931',
    'plot_RGB_chromaticities_in_chromaticity_diagram_CIE1960UCS',
    'plot_RGB_chromaticities_in_chromaticity_diagram_CIE1976UCS',
    'plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1931',
    'plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1960UCS',
    'plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1976UCS',
    'plot_single_cctf', 'plot_multi_cctfs', 'plot_constant_hue_loci'
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
__all__ += ['plot_single_sd_colour_rendition_report']
__all__ += ['plot_RGB_colourspaces_gamuts', 'plot_RGB_scatter']


# ----------------------------------------------------------------------------#
# ---                API Changes and Deprecation Management                ---#
# ----------------------------------------------------------------------------#
class plotting(ModuleAPI):
    def __getattr__(self, attribute):
        return super(plotting, self).__getattr__(attribute)


# v0.3.11
API_CHANGES = {
    'ObjectRenamed': [
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
API_CHANGES['ObjectRenamed'] = API_CHANGES['ObjectRenamed'] + [
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

# v0.3.14
API_CHANGES['ObjectRenamed'] = API_CHANGES['ObjectRenamed'] + [
    [
        'colour.plotting.ASTM_G_173_DIRECT_CIRCUMSOLAR',
        'colour.plotting.SD_ASTMG173_DIRECT_CIRCUMSOLAR',
    ],
    [
        'colour.plotting.ASTM_G_173_ETR',
        'colour.plotting.SD_ASTMG173_ETR',
    ],
    [
        'colour.plotting.ASTM_G_173_GLOBAL_TILT',
        'colour.plotting.SD_ASTMG173_GLOBAL_TILT',
    ],
]

# v0.3.16
API_CHANGES['ObjectRenamed'] = API_CHANGES['ObjectRenamed'] + [
    [
        'colour.plotting.COLOUR_STYLE_CONSTANTS',
        'colour.plotting.CONSTANTS_COLOUR_STYLE',
    ],
    [
        'colour.plotting.COLOUR_ARROW_STYLE',
        'colour.plotting.CONSTANTS_ARROW_STYLE',
    ],
    [
        'colour.plotting.quad',
        'colour.geometry.primitive_vertices_quad_mpl',
    ],
    [
        'colour.plotting.grid',
        'colour.geometry.primitive_vertices_grid_mpl',
    ],
    [
        'colour.plotting.cube',
        'colour.geometry.primitive_vertices_cube_mpl',
    ],
]

if not is_documentation_building():
    sys.modules['colour.plotting'] = plotting(sys.modules['colour.plotting'],
                                              build_API_changes(API_CHANGES))

    del ModuleAPI, is_documentation_building, build_API_changes, sys
