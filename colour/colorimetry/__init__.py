#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

from .spectrum import (
    SpectralShape,
    DEFAULT_SPECTRAL_SHAPE,
    SpectralPowerDistribution,
    TriSpectralPowerDistribution,
    constant_spd,
    zeros_spd,
    ones_spd)
from .blackbody import (
    blackbody_spd,
    blackbody_spectral_radiance,
    planck_law)
from .cmfs import (
    LMS_ConeFundamentals,
    RGB_ColourMatchingFunctions,
    XYZ_ColourMatchingFunctions)
from .dataset import *  # noqa
from . import dataset
from .correction import BANDPASS_CORRECTION_METHODS
from .correction import bandpass_correction
from .correction import bandpass_correction_stearns1988
from .illuminants import D_illuminant_relative_spd
from .lefs import (
    mesopic_luminous_efficiency_function,
    mesopic_weighting_function,
    luminous_flux)
from .lightness import LIGHTNESS_METHODS
from .lightness import lightness
from .lightness import (
    lightness_glasser1958,
    lightness_wyszecki1964,
    lightness_1976)
from .luminance import LUMINANCE_METHODS
from .luminance import luminance
from .luminance import (
    luminance_newhall1943,
    luminance_ASTM_D1535_08,
    luminance_1976)
from .transformations import RGB_10_degree_cmfs_to_LMS_10_degree_cmfs
from .transformations import RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs
from .transformations import RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs
from .transformations import LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs
from .transformations import LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs
from .tristimulus import spectral_to_XYZ, wavelength_to_XYZ
from .whiteness import WHITENESS_METHODS
from .whiteness import whiteness
from .whiteness import (
    whiteness_berger1959,
    whiteness_taube1960,
    whiteness_stensby1968,
    whiteness_ASTM_313,
    whiteness_ganz1979,
    whiteness_cie2004)

__all__ = ['SpectralShape',
           'DEFAULT_SPECTRAL_SHAPE',
           'SpectralPowerDistribution',
           'TriSpectralPowerDistribution',
           'constant_spd',
           'zeros_spd',
           'ones_spd']
__all__ += ['blackbody_spd',
            'blackbody_spectral_radiance',
            'planck_law']
__all__ += ['LMS_ConeFundamentals',
            'RGB_ColourMatchingFunctions',
            'XYZ_ColourMatchingFunctions']
__all__ += dataset.__all__
__all__ += ['BANDPASS_CORRECTION_METHODS']
__all__ += ['bandpass_correction']
__all__ += ['bandpass_correction_stearns1988']
__all__ += ['D_illuminant_relative_spd']
__all__ += ['mesopic_luminous_efficiency_function',
            'mesopic_weighting_function',
            'luminous_flux']
__all__ += ['LIGHTNESS_METHODS']
__all__ += ['lightness']
__all__ += ['lightness_glasser1958',
            'lightness_wyszecki1964',
            'lightness_1976']
__all__ += ['LUMINANCE_METHODS']
__all__ += ['luminance']
__all__ += ['luminance_newhall1943',
            'luminance_ASTM_D1535_08',
            'luminance_1976']
__all__ += ['RGB_10_degree_cmfs_to_LMS_10_degree_cmfs']
__all__ += ['RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs']
__all__ += ['RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs']
__all__ += ['LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs']
__all__ += ['LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs']
__all__ += ['spectral_to_XYZ', 'wavelength_to_XYZ']
__all__ += ['WHITENESS_METHODS']
__all__ += ['whiteness']
__all__ += ['whiteness_berger1959',
            'whiteness_taube1960',
            'whiteness_stensby1968',
            'whiteness_ASTM_313',
            'whiteness_ganz1979',
            'whiteness_cie2004']
