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
    DEFAULT_PARAMETERS,
    DEFAULT_COLOUR_CYCLE,
    ColourParameter,
    colour_cycle,
    canvas,
    decorate,
    boundaries,
    display,
    colour_parameter,
    colour_parameters_plot,
    single_colour_plot,
    multi_colour_plot,
    image_plot)
from .colorimetry import (
    get_cmfs,
    get_illuminant,
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
    CHROMATICITY_DIAGRAM_DEFAULT_ILLUMINANT,
    CIE_1931_chromaticity_diagram_plot,
    CIE_1960_UCS_chromaticity_diagram_plot,
    CIE_1976_UCS_chromaticity_diagram_plot,
    spds_CIE_1931_chromaticity_diagram_plot,
    spds_CIE_1960_UCS_chromaticity_diagram_plot,
    spds_CIE_1976_UCS_chromaticity_diagram_plot)
from .corresponding import corresponding_chromaticities_prediction_plot
from .models import (
    get_RGB_colourspace,
    RGB_colourspaces_CIE_1931_chromaticity_diagram_plot,
    RGB_colourspaces_CIE_1960_UCS_chromaticity_diagram_plot,
    RGB_colourspaces_CIE_1976_UCS_chromaticity_diagram_plot,
    RGB_chromaticity_coordinates_CIE_1931_chromaticity_diagram_plot,
    RGB_chromaticity_coordinates_CIE_1960_UCS_chromaticity_diagram_plot,
    RGB_chromaticity_coordinates_CIE_1976_UCS_chromaticity_diagram_plot,
    single_transfer_function_plot,
    multi_transfer_function_plot)
from .notation import (
    single_munsell_value_function_plot,
    multi_munsell_value_function_plot)
from .phenomenon import single_rayleigh_scattering_spd_plot, the_blue_sky_plot
from .quality import (colour_quality_scale_bars_plot,
                      colour_rendering_index_bars_plot)
from .temperature import (
    planckian_locus_CIE_1931_chromaticity_diagram_plot,
    planckian_locus_CIE_1960_UCS_chromaticity_diagram_plot)

__all__ = []
__all__ += dataset.__all__
__all__ += [
    'PLOTTING_RESOURCES_DIRECTORY',
    'DEFAULT_FIGURE_ASPECT_RATIO',
    'DEFAULT_FIGURE_WIDTH',
    'DEFAULT_FIGURE_HEIGHT',
    'DEFAULT_FIGURE_SIZE',
    'DEFAULT_FONT_SIZE',
    'DEFAULT_PARAMETERS',
    'DEFAULT_COLOUR_CYCLE',
    'ColourParameter',
    'colour_cycle',
    'canvas',
    'decorate',
    'boundaries',
    'display',
    'colour_parameter',
    'colour_parameters_plot',
    'single_colour_plot',
    'multi_colour_plot',
    'image_plot']
__all__ += [
    'get_cmfs',
    'get_illuminant',
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
    'CHROMATICITY_DIAGRAM_DEFAULT_ILLUMINANT',
    'CIE_1931_chromaticity_diagram_plot',
    'CIE_1960_UCS_chromaticity_diagram_plot',
    'CIE_1976_UCS_chromaticity_diagram_plot',
    'spds_CIE_1931_chromaticity_diagram_plot',
    'spds_CIE_1960_UCS_chromaticity_diagram_plot',
    'spds_CIE_1976_UCS_chromaticity_diagram_plot']
__all__ += [
    'corresponding_chromaticities_prediction_plot']
__all__ += [
    'get_RGB_colourspace',
    'RGB_colourspaces_CIE_1931_chromaticity_diagram_plot',
    'RGB_colourspaces_CIE_1960_UCS_chromaticity_diagram_plot',
    'RGB_colourspaces_CIE_1976_UCS_chromaticity_diagram_plot',
    'RGB_chromaticity_coordinates_CIE_1931_chromaticity_diagram_plot',
    'RGB_chromaticity_coordinates_CIE_1960_UCS_chromaticity_diagram_plot',
    'RGB_chromaticity_coordinates_CIE_1976_UCS_chromaticity_diagram_plot',
    'single_transfer_function_plot',
    'multi_transfer_function_plot']
__all__ += [
    'single_munsell_value_function_plot',
    'multi_munsell_value_function_plot']
__all__ += ['single_rayleigh_scattering_spd_plot', 'the_blue_sky_plot']
__all__ += ['colour_quality_scale_bars_plot',
            'colour_rendering_index_bars_plot']
__all__ += [
    'planckian_locus_CIE_1931_chromaticity_diagram_plot',
    'planckian_locus_CIE_1960_UCS_chromaticity_diagram_plot']
