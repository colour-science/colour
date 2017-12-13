#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

from .dataset import *  # noqa
from . import dataset
from .common import (
    PLOTTING_RESOURCES_DIRECTORY, DEFAULT_FIGURE_ASPECT_RATIO,
    DEFAULT_FIGURE_WIDTH, DEFAULT_FIGURE_HEIGHT, DEFAULT_FIGURE_SIZE,
    DEFAULT_FONT_SIZE, DEFAULT_COLOUR_CYCLE, DEFAULT_HATCH_PATTERNS,
    DEFAULT_PARAMETERS, DEFAULT_PLOTTING_ILLUMINANT,
    DEFAULT_PLOTTING_ENCODING_CCTF, colour_plotting_defaults, ColourParameter,
    colour_cycle, canvas, camera, decorate, boundaries, display,
    label_rectangles, equal_axes3d, get_RGB_colourspace, get_cmfs,
    get_illuminant, colour_parameters_plot, single_colour_plot,
    multi_colour_plot, image_plot)
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
from .phenomenon import single_rayleigh_scattering_spd_plot, the_blue_sky_plot
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
    'ColourParameter', 'colour_cycle', 'canvas', 'camera', 'decorate',
    'boundaries', 'display', 'label_rectangles', 'equal_axes3d',
    'get_RGB_colourspace', 'get_cmfs', 'get_illuminant',
    'colour_parameters_plot', 'single_colour_plot', 'multi_colour_plot',
    'image_plot'
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
