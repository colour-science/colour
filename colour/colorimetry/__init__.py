# -*- coding: utf-8 -*-

from __future__ import absolute_import

import sys

from colour.utilities.deprecation import ModuleAPI, build_API_changes
from colour.utilities.documentation import is_documentation_building

from .spectrum import (SpectralShape, SPECTRAL_SHAPE_DEFAULT,
                       SpectralDistribution, MultiSpectralDistributions,
                       sds_and_msds_to_sds, sds_and_msds_to_msds)
from .blackbody import sd_blackbody, blackbody_spectral_radiance, planck_law
from .cmfs import (LMS_ConeFundamentals, RGB_ColourMatchingFunctions,
                   XYZ_ColourMatchingFunctions)
from .datasets import *  # noqa
from . import datasets
from .generation import sd_constant, sd_zeros, sd_ones
from .generation import msds_constant, msds_zeros, msds_ones
from .generation import SD_GAUSSIAN_METHODS
from .generation import sd_gaussian, sd_gaussian_normal, sd_gaussian_fwhm
from .generation import SD_SINGLE_LED_METHODS
from .generation import sd_single_led, sd_single_led_Ohno2005
from .generation import SD_MULTI_LEDS_METHODS
from .generation import sd_multi_leds, sd_multi_leds_Ohno2005
from .tristimulus import SD_TO_XYZ_METHODS, MSDS_TO_XYZ_METHODS
from .tristimulus import sd_to_XYZ, msds_to_XYZ
from .tristimulus import (
    SPECTRAL_SHAPE_ASTME308, lagrange_coefficients_ASTME2022,
    tristimulus_weighting_factors_ASTME2022,
    adjust_tristimulus_weighting_factors_ASTME308, sd_to_XYZ_integration,
    sd_to_XYZ_tristimulus_weighting_factors_ASTME308, sd_to_XYZ_ASTME308,
    msds_to_XYZ_integration, msds_to_XYZ_ASTME308, wavelength_to_XYZ)
from .correction import BANDPASS_CORRECTION_METHODS
from .correction import bandpass_correction
from .correction import bandpass_correction_Stearns1988
from .illuminants import (sd_CIE_standard_illuminant_A,
                          sd_CIE_illuminant_D_series, daylight_locus_function)
from .lefs import (sd_mesopic_luminous_efficiency_function,
                   mesopic_weighting_function)
from .lightness import LIGHTNESS_METHODS
from .lightness import lightness
from .lightness import (lightness_Glasser1958, lightness_Wyszecki1963,
                        lightness_CIE1976, lightness_Fairchild2010,
                        lightness_Fairchild2011)
from .lightness import intermediate_lightness_function_CIE1976
from .luminance import LUMINANCE_METHODS
from .luminance import luminance
from .luminance import (luminance_Newhall1943, luminance_ASTMD1535,
                        luminance_CIE1976, luminance_Fairchild2010,
                        luminance_Fairchild2011)
from .luminance import intermediate_luminance_function_CIE1976
from .dominant import (dominant_wavelength, complementary_wavelength,
                       excitation_purity, colorimetric_purity)
from .photometry import luminous_flux, luminous_efficiency, luminous_efficacy
from .transformations import RGB_10_degree_cmfs_to_LMS_10_degree_cmfs
from .transformations import RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs
from .transformations import RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs
from .transformations import LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs
from .transformations import LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs
from .whiteness import WHITENESS_METHODS
from .whiteness import whiteness
from .whiteness import (whiteness_Berger1959, whiteness_Taube1960,
                        whiteness_Stensby1968, whiteness_ASTME313,
                        whiteness_Ganz1979, whiteness_CIE2004)
from .yellowness import YELLOWNESS_METHODS
from .yellowness import yellowness
from .yellowness import yellowness_ASTMD1925, yellowness_ASTME313

__all__ = [
    'SpectralShape', 'SPECTRAL_SHAPE_DEFAULT', 'SpectralDistribution',
    'MultiSpectralDistributions', 'sds_and_msds_to_sds', 'sds_and_msds_to_msds'
]
__all__ += ['sd_blackbody', 'blackbody_spectral_radiance', 'planck_law']
__all__ += [
    'LMS_ConeFundamentals', 'RGB_ColourMatchingFunctions',
    'XYZ_ColourMatchingFunctions'
]
__all__ += datasets.__all__
__all__ += ['sd_constant', 'sd_zeros', 'sd_ones']
__all__ += ['msds_constant', 'msds_zeros', 'msds_ones']
__all__ += ['SD_GAUSSIAN_METHODS']
__all__ += ['sd_gaussian', 'sd_gaussian_normal', 'sd_gaussian_fwhm']
__all__ += ['SD_SINGLE_LED_METHODS']
__all__ += ['sd_single_led', 'sd_single_led_Ohno2005']
__all__ += ['SD_MULTI_LEDS_METHODS']
__all__ += ['sd_multi_leds', 'sd_multi_leds_Ohno2005']
__all__ += ['SD_TO_XYZ_METHODS', 'MSDS_TO_XYZ_METHODS']
__all__ += ['sd_to_XYZ', 'msds_to_XYZ']
__all__ += [
    'SPECTRAL_SHAPE_ASTME308', 'lagrange_coefficients_ASTME2022',
    'tristimulus_weighting_factors_ASTME2022',
    'adjust_tristimulus_weighting_factors_ASTME308', 'sd_to_XYZ_integration',
    'sd_to_XYZ_tristimulus_weighting_factors_ASTME308', 'sd_to_XYZ_ASTME308',
    'msds_to_XYZ_integration', 'msds_to_XYZ_ASTME308', 'wavelength_to_XYZ'
]
__all__ += ['BANDPASS_CORRECTION_METHODS']
__all__ += ['bandpass_correction']
__all__ += ['bandpass_correction_Stearns1988']
__all__ += [
    'sd_CIE_standard_illuminant_A', 'sd_CIE_illuminant_D_series',
    'daylight_locus_function'
]
__all__ += [
    'sd_mesopic_luminous_efficiency_function', 'mesopic_weighting_function'
]
__all__ += ['LIGHTNESS_METHODS']
__all__ += ['lightness']
__all__ += [
    'lightness_Glasser1958', 'lightness_Wyszecki1963', 'lightness_CIE1976',
    'lightness_Fairchild2010', 'lightness_Fairchild2011'
]
__all__ += ['intermediate_lightness_function_CIE1976']
__all__ += ['LUMINANCE_METHODS']
__all__ += ['luminance']
__all__ += [
    'luminance_Newhall1943', 'luminance_ASTMD1535', 'luminance_CIE1976',
    'luminance_Fairchild2010', 'luminance_Fairchild2011'
]
__all__ += ['intermediate_luminance_function_CIE1976']
__all__ += [
    'dominant_wavelength', 'complementary_wavelength', 'excitation_purity',
    'colorimetric_purity'
]
__all__ += ['luminous_flux', 'luminous_efficiency', 'luminous_efficacy']
__all__ += ['RGB_10_degree_cmfs_to_LMS_10_degree_cmfs']
__all__ += ['RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs']
__all__ += ['RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs']
__all__ += ['LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs']
__all__ += ['LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs']
__all__ += ['WHITENESS_METHODS']
__all__ += ['whiteness']
__all__ += [
    'whiteness_Berger1959', 'whiteness_Taube1960', 'whiteness_Stensby1968',
    'whiteness_ASTME313', 'whiteness_Ganz1979', 'whiteness_CIE2004'
]
__all__ += ['YELLOWNESS_METHODS']
__all__ += ['yellowness']
__all__ += ['yellowness_ASTMD1925', 'yellowness_ASTME313']


# ----------------------------------------------------------------------------#
# ---                API Changes and Deprecation Management                ---#
# ----------------------------------------------------------------------------#
class colorimetry(ModuleAPI):
    def __getattr__(self, attribute):
        return super(colorimetry, self).__getattr__(attribute)


# v0.3.12
API_CHANGES = {
    'ObjectRenamed': [
        [
            'colour.colorimetry.spectral_to_XYZ_ASTME30815',
            'colour.colorimetry.sd_to_XYZ_ASTME308',
        ],
        [
            'colour.colorimetry.spectral_to_XYZ_integration',
            'colour.colorimetry.sd_to_XYZ_integration',
        ],
        [
            'colour.colorimetry.spectral_to_XYZ_tristimulus_weighting_factors_ASTME30815',  # noqa
            'colour.colorimetry.sd_to_XYZ_tristimulus_weighting_factors_ASTME308',  # noqa
        ],
    ]
}
"""
Defines *colour.colorimetry* sub-package API changes.

API_CHANGES : dict
"""

# v0.3.14
API_CHANGES['ObjectRenamed'] = API_CHANGES['ObjectRenamed'] + [
    [
        'colour.colorimetry.adjust_tristimulus_weighting_factors_ASTME30815',  # noqa
        'colour.colorimetry.adjust_tristimulus_weighting_factors_ASTME308',  # noqa
    ],
    [
        'colour.colorimetry.lagrange_coefficients_ASTME202211',
        'colour.colorimetry.lagrange_coefficients_ASTME2022',
    ],
    [
        'colour.colorimetry.luminance_ASTMD153508',
        'colour.colorimetry.luminance_ASTMD1535',
    ],
    [
        'colour.colorimetry.sd_to_XYZ_ASTME30815',
        'colour.colorimetry.sd_to_XYZ_ASTME308',
    ],
    [
        'colour.colorimetry.sd_to_XYZ_tristimulus_weighting_factors_ASTME30815',  # noqa
        'colour.colorimetry.sd_to_XYZ_tristimulus_weighting_factors_ASTME308',  # noqa
    ],
    [
        'colour.colorimetry.tristimulus_weighting_factors_ASTME202211',
        'colour.colorimetry.tristimulus_weighting_factors_ASTME2022',
    ],
]

# v0.3.16
API_CHANGES['ObjectRenamed'] = API_CHANGES['ObjectRenamed'] + [
    [
        'colour.colorimetry.ASTME308_PRACTISE_SHAPE',
        'colour.colorimetry.SPECTRAL_SHAPE_ASTME308',
    ],
    [
        'colour.colorimetry.CMFS',
        'colour.colorimetry.MSDS_CMFS',
    ],
    [
        'colour.colorimetry.D_ILLUMINANTS_S_SDS',
        'colour.colorimetry.SDS_BASIS_FUNCTIONS_CIE_ILLUMINANT_D_SERIES',
    ],
    [
        'colour.colorimetry.DEFAULT_SPECTRAL_SHAPE',
        'colour.colorimetry.SPECTRAL_SHAPE_DEFAULT',
    ],
    [
        'colour.colorimetry.HUNTERLAB_ILLUMINANTS',
        'colour.colorimetry.TVS_ILLUMINANTS_HUNTERLAB',
    ],
    [
        'colour.colorimetry.ILLUMINANTS',
        'colour.colorimetry.CCS_ILLUMINANTS',
    ],
    [
        'colour.colorimetry.ILLUMINANTS_SDS',
        'colour.colorimetry.SDS_ILLUMINANTS',
    ],
    [
        'colour.colorimetry.LEFS',
        'colour.colorimetry.SDS_LEFS',
    ],
    [
        'colour.colorimetry.LIGHT_SOURCES',
        'colour.colorimetry.CCS_LIGHT_SOURCES',
    ],
    [
        'colour.colorimetry.LIGHT_SOURCES_SDS',
        'colour.colorimetry.SDS_LIGHT_SOURCES',
    ],
    [
        'colour.colorimetry.LMS_CMFS',
        'colour.colorimetry.MSDS_CMFS_LMS',
    ],
    [
        'colour.colorimetry.MULTI_SD_TO_XYZ_METHODS',
        'colour.colorimetry.MSDS_TO_XYZ_METHODS',
    ],
    [
        'colour.colorimetry.multi_sds_to_XYZ_integration',
        'colour.colorimetry.msds_to_XYZ_integration',
    ],
    [
        'colour.colorimetry.multi_sds_to_XYZ_ASTME308',
        'colour.colorimetry.msds_to_XYZ_ASTME308',
    ],
    [
        'colour.colorimetry.multi_sds_to_XYZ',
        'colour.colorimetry.msds_to_XYZ',
    ],
    [
        'colour.colorimetry.PHOTOPIC_LEFS',
        'colour.colorimetry.SDS_LEFS_PHOTOPIC',
    ],
    [
        'colour.colorimetry.RGB_CMFS',
        'colour.colorimetry.MSDS_CMFS_RGB',
    ],
    [
        'colour.colorimetry.SCOTOPIC_LEFS',
        'colour.colorimetry.SDS_LEFS_SCOTOPIC',
    ],
    [
        'colour.colorimetry.STANDARD_OBSERVERS_CMFS',
        'colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER',
    ],
    [
        'colour.colorimetry.sds_and_multi_sds_to_sds',
        'colour.colorimetry.sds_and_msds_to_sds',
    ],
    [
        'colour.colorimetry.sds_and_multi_sds_to_multi_sds',
        'colour.colorimetry.sds_and_msds_to_msds',
    ],
]

if not is_documentation_building():
    sys.modules['colour.colorimetry'] = colorimetry(
        sys.modules['colour.colorimetry'], build_API_changes(API_CHANGES))

    del ModuleAPI, is_documentation_building, build_API_changes, sys
