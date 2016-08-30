#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

from .dataset import *  # noqa
from . import dataset
from .common import (
    PLOTTING_RESOURCES_DIRECTORY,
    DEFAULT_FIGURE_ASPECT_RATIO,
    DEFAULT_FIGURE_WIDTH,
    DEFAULT_FIGURE_HEIGHT,
    DEFAULT_FIGURE_SIZE,
    DEFAULT_FONT_SIZE,
    DEFAULT_COLOUR_CYCLE,
    DEFAULT_HATCH_PATTERNS,
    DEFAULT_PARAMETERS,
    DEFAULT_PLOTTING_ILLUMINANT,
    DEFAULT_PLOTTING_ENCODING_CCTF,
    colour_plotting_defaults,
    ColourParameter,
    colour_cycle,
    canvas,
    camera,
    decorate,
    boundaries,
    display,
    label_rectangles,
    equal_axes3d,
    get_RGB_colourspace,
    get_cmfs,
    get_illuminant,
    colour_parameters_plot,
    single_colour_plot,
    multi_colour_plot,
    image_plot)
from .colorimetry import (
    single_spd_plot,
    multi_spd_plot,
    single_cmfs_plot,
    multi_cmfs_plot,
    single_illuminant_relative_spd_plot,
    multi_illuminants_relative_spd_plot,
    visible_spectrum_plot,
    single_lightness_function_plot,
    multi_lightness_function_plot,
    blackbody_spectral_radiance_plot,
    blackbody_colours_plot)
from .characterisation import colour_checker_plot
from .diagrams import (
    CIE_1931_chromaticity_diagram_plot,
    CIE_1960_UCS_chromaticity_diagram_plot,
    CIE_1976_UCS_chromaticity_diagram_plot,
    spds_CIE_1931_chromaticity_diagram_plot,
    spds_CIE_1960_UCS_chromaticity_diagram_plot,
    spds_CIE_1976_UCS_chromaticity_diagram_plot)
from .corresponding import corresponding_chromaticities_prediction_plot
from .geometry import (
    quad,
    grid,
    cube)
from .models import (
    RGB_colourspaces_CIE_1931_chromaticity_diagram_plot,
    RGB_colourspaces_CIE_1960_UCS_chromaticity_diagram_plot,
    RGB_colourspaces_CIE_1976_UCS_chromaticity_diagram_plot,
    RGB_chromaticity_coordinates_CIE_1931_chromaticity_diagram_plot,
    RGB_chromaticity_coordinates_CIE_1960_UCS_chromaticity_diagram_plot,
    RGB_chromaticity_coordinates_CIE_1976_UCS_chromaticity_diagram_plot,
    single_cctf_plot,
    multi_cctf_plot)
from .notation import (
    single_munsell_value_function_plot,
    multi_munsell_value_function_plot)
from .phenomenon import single_rayleigh_scattering_spd_plot, the_blue_sky_plot
from .quality import (
    single_spd_colour_rendering_index_bars_plot,
    multi_spd_colour_rendering_index_bars_plot,
    single_spd_colour_quality_scale_bars_plot,
    multi_spd_colour_quality_scale_bars_plot)
from .temperature import (
    planckian_locus_CIE_1931_chromaticity_diagram_plot,
    planckian_locus_CIE_1960_UCS_chromaticity_diagram_plot)
from .volume import RGB_colourspaces_gamuts_plot, RGB_scatter_plot

__all__ = []
__all__ += dataset.__all__
__all__ += [
    'PLOTTING_RESOURCES_DIRECTORY',
    'DEFAULT_FIGURE_ASPECT_RATIO',
    'DEFAULT_FIGURE_WIDTH',
    'DEFAULT_FIGURE_HEIGHT',
    'DEFAULT_FIGURE_SIZE',
    'DEFAULT_FONT_SIZE',
    'DEFAULT_COLOUR_CYCLE',
    'DEFAULT_HATCH_PATTERNS',
    'DEFAULT_PARAMETERS',
    'DEFAULT_PLOTTING_ILLUMINANT',
    'DEFAULT_PLOTTING_ENCODING_CCTF',
    'colour_plotting_defaults',
    'ColourParameter',
    'colour_cycle',
    'canvas',
    'camera',
    'decorate',
    'boundaries',
    'display',
    'label_rectangles',
    'equal_axes3d',
    'get_RGB_colourspace',
    'get_cmfs',
    'get_illuminant',
    'colour_parameters_plot',
    'single_colour_plot',
    'multi_colour_plot',
    'image_plot']
__all__ += [
    'single_spd_plot',
    'multi_spd_plot',
    'single_cmfs_plot',
    'multi_cmfs_plot',
    'single_illuminant_relative_spd_plot',
    'multi_illuminants_relative_spd_plot',
    'visible_spectrum_plot',
    'single_lightness_function_plot',
    'multi_lightness_function_plot',
    'blackbody_spectral_radiance_plot',
    'blackbody_colours_plot']
__all__ += [
    'colour_checker_plot']
__all__ += [
    'CIE_1931_chromaticity_diagram_plot',
    'CIE_1960_UCS_chromaticity_diagram_plot',
    'CIE_1976_UCS_chromaticity_diagram_plot',
    'spds_CIE_1931_chromaticity_diagram_plot',
    'spds_CIE_1960_UCS_chromaticity_diagram_plot',
    'spds_CIE_1976_UCS_chromaticity_diagram_plot']
__all__ += [
    'corresponding_chromaticities_prediction_plot']
__all__ += [
    'quad',
    'grid',
    'cube']
__all__ += [
    'RGB_colourspaces_CIE_1931_chromaticity_diagram_plot',
    'RGB_colourspaces_CIE_1960_UCS_chromaticity_diagram_plot',
    'RGB_colourspaces_CIE_1976_UCS_chromaticity_diagram_plot',
    'RGB_chromaticity_coordinates_CIE_1931_chromaticity_diagram_plot',
    'RGB_chromaticity_coordinates_CIE_1960_UCS_chromaticity_diagram_plot',
    'RGB_chromaticity_coordinates_CIE_1976_UCS_chromaticity_diagram_plot',
    'single_cctf_plot',
    'multi_cctf_plot']
__all__ += [
    'single_munsell_value_function_plot',
    'multi_munsell_value_function_plot']
__all__ += ['single_rayleigh_scattering_spd_plot', 'the_blue_sky_plot']
__all__ += [
    'single_spd_colour_rendering_index_bars_plot',
    'multi_spd_colour_rendering_index_bars_plot',
    'single_spd_colour_quality_scale_bars_plot',
    'multi_spd_colour_quality_scale_bars_plot']
__all__ += [
    'planckian_locus_CIE_1931_chromaticity_diagram_plot',
    'planckian_locus_CIE_1960_UCS_chromaticity_diagram_plot']
__all__ += ['RGB_colourspaces_gamuts_plot', 'RGB_scatter_plot']
