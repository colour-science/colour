# -*- coding: utf-8 -*-

from .spectrum import (
    SpectralShape,
    SPECTRAL_SHAPE_DEFAULT,
    SpectralDistribution,
    MultiSpectralDistributions,
    reshape_sd,
    reshape_msds,
    sds_and_msds_to_sds,
    sds_and_msds_to_msds,
)
from .blackbody import sd_blackbody, blackbody_spectral_radiance, planck_law
from .cmfs import (
    LMS_ConeFundamentals,
    RGB_ColourMatchingFunctions,
    XYZ_ColourMatchingFunctions,
)
from .datasets import (
    CCS_ILLUMINANTS,
    CCS_LIGHT_SOURCES,
    MSDS_CMFS,
    MSDS_CMFS_LMS,
    MSDS_CMFS_RGB,
    MSDS_CMFS_STANDARD_OBSERVER,
    SDS_BASIS_FUNCTIONS_CIE_ILLUMINANT_D_SERIES,
    SDS_ILLUMINANTS,
    SDS_LEFS,
    SDS_LEFS_PHOTOPIC,
    SDS_LEFS_SCOTOPIC,
    SDS_LIGHT_SOURCES,
    TVS_ILLUMINANTS,
    TVS_ILLUMINANTS_HUNTERLAB,
)
from .generation import sd_constant, sd_zeros, sd_ones
from .generation import msds_constant, msds_zeros, msds_ones
from .generation import SD_GAUSSIAN_METHODS
from .generation import sd_gaussian, sd_gaussian_normal, sd_gaussian_fwhm
from .generation import SD_SINGLE_LED_METHODS
from .generation import sd_single_led, sd_single_led_Ohno2005
from .generation import SD_MULTI_LEDS_METHODS
from .generation import sd_multi_leds, sd_multi_leds_Ohno2005
from .tristimulus_values import SD_TO_XYZ_METHODS, MSDS_TO_XYZ_METHODS
from .tristimulus_values import sd_to_XYZ, msds_to_XYZ
from .tristimulus_values import (
    SPECTRAL_SHAPE_ASTME308,
    handle_spectral_arguments,
    lagrange_coefficients_ASTME2022,
    tristimulus_weighting_factors_ASTME2022,
    adjust_tristimulus_weighting_factors_ASTME308,
    sd_to_XYZ_integration,
    sd_to_XYZ_tristimulus_weighting_factors_ASTME308,
    sd_to_XYZ_ASTME308,
    msds_to_XYZ_integration,
    msds_to_XYZ_ASTME308,
    wavelength_to_XYZ,
)
from .uniformity import spectral_uniformity
from .correction import BANDPASS_CORRECTION_METHODS
from .correction import bandpass_correction
from .correction import bandpass_correction_Stearns1988
from .illuminants import (
    sd_CIE_standard_illuminant_A,
    sd_CIE_illuminant_D_series,
    daylight_locus_function,
)
from .lefs import (
    sd_mesopic_luminous_efficiency_function,
    mesopic_weighting_function,
)
from .lightness import LIGHTNESS_METHODS
from .lightness import lightness
from .lightness import (
    lightness_Glasser1958,
    lightness_Wyszecki1963,
    lightness_CIE1976,
    lightness_Fairchild2010,
    lightness_Fairchild2011,
    lightness_Abebe2017,
)
from .lightness import intermediate_lightness_function_CIE1976
from .luminance import LUMINANCE_METHODS
from .luminance import luminance
from .luminance import (
    luminance_Newhall1943,
    luminance_ASTMD1535,
    luminance_CIE1976,
    luminance_Fairchild2010,
    luminance_Fairchild2011,
    luminance_Abebe2017,
)
from .luminance import intermediate_luminance_function_CIE1976
from .dominant import (
    dominant_wavelength,
    complementary_wavelength,
    excitation_purity,
    colorimetric_purity,
)
from .photometry import luminous_flux, luminous_efficiency, luminous_efficacy
from .transformations import RGB_10_degree_cmfs_to_LMS_10_degree_cmfs
from .transformations import RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs
from .transformations import RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs
from .transformations import LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs
from .transformations import LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs
from .whiteness import WHITENESS_METHODS
from .whiteness import whiteness
from .whiteness import (
    whiteness_Berger1959,
    whiteness_Taube1960,
    whiteness_Stensby1968,
    whiteness_ASTME313,
    whiteness_Ganz1979,
    whiteness_CIE2004,
)
from .yellowness import YELLOWNESS_METHODS
from .yellowness import yellowness
from .yellowness import (
    yellowness_ASTMD1925,
    yellowness_ASTME313_alternative,
    YELLOWNESS_COEFFICIENTS_ASTME313,
    yellowness_ASTME313,
)

__all__ = [
    'SpectralShape',
    'SPECTRAL_SHAPE_DEFAULT',
    'SpectralDistribution',
    'MultiSpectralDistributions',
    'reshape_sd',
    'reshape_msds',
    'sds_and_msds_to_sds',
    'sds_and_msds_to_msds',
]
__all__ += [
    'sd_blackbody',
    'blackbody_spectral_radiance',
    'planck_law',
]
__all__ += [
    'LMS_ConeFundamentals',
    'RGB_ColourMatchingFunctions',
    'XYZ_ColourMatchingFunctions',
]
__all__ += [
    'CCS_ILLUMINANTS',
    'MSDS_CMFS',
    'MSDS_CMFS_LMS',
    'MSDS_CMFS_RGB',
    'MSDS_CMFS_STANDARD_OBSERVER',
    'SDS_BASIS_FUNCTIONS_CIE_ILLUMINANT_D_SERIES',
    'SDS_ILLUMINANTS',
    'SDS_LEFS',
    'SDS_LEFS_PHOTOPIC',
    'SDS_LEFS_SCOTOPIC',
    'TVS_ILLUMINANTS',
    'TVS_ILLUMINANTS_HUNTERLAB',
    'CCS_LIGHT_SOURCES',
    'SDS_LIGHT_SOURCES',
]
__all__ += [
    'sd_constant',
    'sd_zeros',
    'sd_ones',
]
__all__ += [
    'msds_constant',
    'msds_zeros',
    'msds_ones',
]
__all__ += [
    'SD_GAUSSIAN_METHODS',
]
__all__ += [
    'sd_gaussian',
    'sd_gaussian_normal',
    'sd_gaussian_fwhm',
]
__all__ += [
    'SD_SINGLE_LED_METHODS',
]
__all__ += [
    'sd_single_led',
    'sd_single_led_Ohno2005',
]
__all__ += [
    'SD_MULTI_LEDS_METHODS',
]
__all__ += [
    'sd_multi_leds',
    'sd_multi_leds_Ohno2005',
]
__all__ += [
    'SD_TO_XYZ_METHODS',
    'MSDS_TO_XYZ_METHODS',
]
__all__ += [
    'sd_to_XYZ',
    'msds_to_XYZ',
]
__all__ += [
    'SPECTRAL_SHAPE_ASTME308',
    'handle_spectral_arguments',
    'lagrange_coefficients_ASTME2022',
    'tristimulus_weighting_factors_ASTME2022',
    'adjust_tristimulus_weighting_factors_ASTME308',
    'sd_to_XYZ_integration',
    'sd_to_XYZ_tristimulus_weighting_factors_ASTME308',
    'sd_to_XYZ_ASTME308',
    'msds_to_XYZ_integration',
    'msds_to_XYZ_ASTME308',
    'wavelength_to_XYZ',
]
__all__ += [
    'spectral_uniformity',
]
__all__ += [
    'BANDPASS_CORRECTION_METHODS',
]
__all__ += [
    'bandpass_correction',
]
__all__ += [
    'bandpass_correction_Stearns1988',
]
__all__ += [
    'sd_CIE_standard_illuminant_A',
    'sd_CIE_illuminant_D_series',
    'daylight_locus_function',
]
__all__ += [
    'sd_mesopic_luminous_efficiency_function',
    'mesopic_weighting_function',
]
__all__ += [
    'LIGHTNESS_METHODS',
]
__all__ += [
    'lightness',
]
__all__ += [
    'lightness_Glasser1958',
    'lightness_Wyszecki1963',
    'lightness_CIE1976',
    'lightness_Fairchild2010',
    'lightness_Fairchild2011',
    'lightness_Abebe2017',
]
__all__ += [
    'intermediate_lightness_function_CIE1976',
]
__all__ += [
    'LUMINANCE_METHODS',
]
__all__ += [
    'luminance',
]
__all__ += [
    'luminance_Newhall1943',
    'luminance_ASTMD1535',
    'luminance_CIE1976',
    'luminance_Fairchild2010',
    'luminance_Fairchild2011',
    'luminance_Abebe2017',
]
__all__ += [
    'intermediate_luminance_function_CIE1976',
]
__all__ += [
    'dominant_wavelength',
    'complementary_wavelength',
    'excitation_purity',
    'colorimetric_purity',
]
__all__ += [
    'luminous_flux',
    'luminous_efficiency',
    'luminous_efficacy',
]
__all__ += [
    'RGB_10_degree_cmfs_to_LMS_10_degree_cmfs',
]
__all__ += [
    'RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs',
]
__all__ += [
    'RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs',
]
__all__ += [
    'LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs',
]
__all__ += [
    'LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs',
]
__all__ += [
    'WHITENESS_METHODS',
]
__all__ += [
    'whiteness',
]
__all__ += [
    'whiteness_Berger1959',
    'whiteness_Taube1960',
    'whiteness_Stensby1968',
    'whiteness_ASTME313',
    'whiteness_Ganz1979',
    'whiteness_CIE2004',
]
__all__ += [
    'YELLOWNESS_METHODS',
]
__all__ += [
    'yellowness',
]
__all__ += [
    'yellowness_ASTMD1925',
    'yellowness_ASTME313_alternative',
    'YELLOWNESS_COEFFICIENTS_ASTME313',
    'yellowness_ASTME313',
]
