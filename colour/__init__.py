# -*- coding: utf-8 -*-
"""
Colour
======

*Colour* is a *Python* colour science package implementing a comprehensive
number of colour theory transformations and algorithms.

Sub-packages
------------
-   adaptation: Chromatic adaptation models and transformations.
-   algebra: Algebra utilities.
-   appearance: Colour appearance models.
-   biochemistry: Biochemistry computations.
-   blindness: Colour vision deficiency models.
-   characterisation: Colour fitting and camera characterisation.
-   colorimetry: Core objects for colour computations.
-   constants: *CIE* and *CODATA* constants.
-   continuous: Base objects for continuous data representation.
-   contrast: Objects for contrast sensitivity computation.
-   corresponding: Corresponding colour chromaticities computations.
-   difference: Colour difference computations.
-   examples: Examples for the sub-packages.
-   io: Input / output objects for reading and writing data.
-   models: Colour models.
-   notation: Colour notation systems.
-   phenomena: Computation of various optical phenomena.
-   plotting: Diagrams, figures, etc...
-   quality: Colour quality computation.
-   recovery: Reflectance recovery.
-   temperature: Colour temperature and correlated colour temperature
    computation.
-   utilities: Various utilities and data structures.
-   volume: Colourspace volumes computation and optimal colour stimuli.
"""

from __future__ import absolute_import

import numpy as np
import sys

from .utilities.deprecation import (FutureAccessChange, FutureAccessRemove,
                                    ModuleAPI, Removed, Renamed)
from .utilities.documentation import is_documentation_building
from .utilities.common import (domain_range_scale, get_domain_range_scale,
                               set_domain_range_scale)

from .adaptation import (CHROMATIC_ADAPTATION_METHODS,
                         CHROMATIC_ADAPTATION_TRANSFORMS,
                         CMCCAT2000_VIEWING_CONDITIONS, chromatic_adaptation)
from .algebra import (CubicSplineInterpolator, Extrapolator,
                      KernelInterpolator, NearestNeighbourInterpolator,
                      LinearInterpolator, NullInterpolator, PchipInterpolator,
                      SpragueInterpolator, TABLE_INTERPOLATION_METHODS,
                      kernel_cardinal_spline, kernel_lanczos, kernel_linear,
                      kernel_nearest_neighbour, kernel_sinc,
                      table_interpolation, lagrange_coefficients)
from .colorimetry import (
    ASTME30815_PRACTISE_SHAPE, BANDPASS_CORRECTION_METHODS, CMFS,
    DEFAULT_SPECTRAL_SHAPE, HUNTERLAB_ILLUMINANTS, ILLUMINANTS,
    ILLUMINANTS_SDS, LEFS, LIGHTNESS_METHODS, LIGHT_SOURCES, LIGHT_SOURCES_SDS,
    LMS_CMFS, LUMINANCE_METHODS, MULTI_SD_TO_XYZ_METHODS,
    MultiSpectralDistribution, PHOTOPIC_LEFS, RGB_CMFS, SCOTOPIC_LEFS,
    SD_GAUSSIAN_METHODS, SD_MULTI_LEDS_METHODS, SD_SINGLE_LED_METHODS,
    SD_TO_XYZ_METHODS, STANDARD_OBSERVERS_CMFS, SpectralDistribution,
    SpectralShape, WHITENESS_METHODS, YELLOWNESS_METHODS, bandpass_correction,
    colorimetric_purity, complementary_wavelength, dominant_wavelength,
    excitation_purity, lightness, luminance, luminous_efficacy,
    luminous_efficiency, luminous_flux, multi_sds_to_XYZ,
    sd_CIE_standard_illuminant_A, sd_CIE_illuminant_D_series, sd_blackbody,
    sd_constant, sd_gaussian, sd_mesopic_luminous_efficiency_function,
    sd_multi_leds, sd_ones, sd_single_led, sd_zeros, sd_to_XYZ,
    wavelength_to_XYZ, whiteness, yellowness)
from .blindness import (
    CVD_MATRICES_MACHADO2010, anomalous_trichromacy_cmfs_Machado2009,
    anomalous_trichromacy_matrix_Machado2009, cvd_matrix_Machado2009)
from .appearance import (
    ATD95_Specification, CAM16_Specification, CAM16_VIEWING_CONDITIONS,
    CAM16_to_XYZ, CIECAM02_Specification, CIECAM02_VIEWING_CONDITIONS,
    CIECAM02_to_XYZ, HUNT_VIEWING_CONDITIONS, Hunt_Specification,
    LLAB_Specification, LLAB_VIEWING_CONDITIONS, Nayatani95_Specification,
    RLAB_D_FACTOR, RLAB_Specification, RLAB_VIEWING_CONDITIONS, XYZ_to_ATD95,
    XYZ_to_CAM16, XYZ_to_CIECAM02, XYZ_to_Hunt, XYZ_to_LLAB, XYZ_to_Nayatani95,
    XYZ_to_RLAB)
from .difference import DELTA_E_METHODS, delta_E
from .characterisation import (
    CAMERAS_RGB_SPECTRAL_SENSITIVITIES, COLOURCHECKERS, COLOURCHECKERS_SDS,
    DISPLAYS_RGB_PRIMARIES, POLYNOMIAL_EXPANSION_METHODS, polynomial_expansion,
    COLOUR_CORRECTION_MATRIX_METHODS, colour_correction_matrix,
    COLOUR_CORRECTION_METHODS, colour_correction)
from .io import (LUT1D, LUT3x1D, LUT3D, LUTSequence,
                 SpectralDistribution_IESTM2714, read_image, read_LUT,
                 read_sds_from_csv_file, read_sds_from_xrite_file,
                 read_spectral_data_from_csv_file, write_image, write_LUT,
                 write_sds_to_csv_file)
from .models import (
    CAM02LCD_to_JMh_CIECAM02, CAM02SCD_to_JMh_CIECAM02,
    CAM02UCS_to_JMh_CIECAM02, CAM16LCD_to_JMh_CAM16, CAM16SCD_to_JMh_CAM16,
    CAM16UCS_to_JMh_CAM16, CMYK_to_CMY, CMY_to_CMYK, CMY_to_RGB, CV_range,
    DECODING_CCTFS, DIN99_to_Lab, ENCODING_CCTFS, EOTFS, EOTFS_REVERSE,
    HDR_CIELAB_METHODS, HDR_IPT_METHODS, HSL_to_RGB, HSV_to_RGB,
    Hunter_Lab_to_XYZ, Hunter_Rdab_to_XYZ, ICTCP_to_RGB, IPT_hue_angle,
    IPT_to_XYZ, JMh_CAM16_to_CAM16LCD, JMh_CAM16_to_CAM16SCD,
    JMh_CAM16_to_CAM16UCS, JMh_CIECAM02_to_CAM02LCD, JMh_CIECAM02_to_CAM02SCD,
    JMh_CIECAM02_to_CAM02UCS, JzAzBz_to_XYZ, LCHab_to_Lab, LCHuv_to_Luv,
    LOG_DECODING_CURVES, LOG_ENCODING_CURVES, Lab_to_DIN99, Lab_to_LCHab,
    Lab_to_XYZ, Luv_to_LCHuv, Luv_to_XYZ, Luv_to_uv, Luv_uv_to_xy,
    MACADAM_1942_ELLIPSES_DATA, OETFS, OETFS_REVERSE, OOTFS, OOTFS_REVERSE,
    OSA_UCS_to_XYZ, POINTER_GAMUT_BOUNDARIES, POINTER_GAMUT_DATA,
    POINTER_GAMUT_ILLUMINANT, Prismatic_to_RGB, RGB_COLOURSPACES,
    RGB_Colourspace, RGB_luminance, RGB_luminance_equation, RGB_to_CMY,
    RGB_to_HSL, RGB_to_HSV, RGB_to_ICTCP, RGB_to_Prismatic, RGB_to_RGB,
    RGB_to_RGB_matrix, RGB_to_XYZ, RGB_to_YCbCr, RGB_to_YcCbcCrc, RGB_to_YCoCg,
    UCS_to_XYZ, UCS_to_uv, UCS_uv_to_xy, UVW_to_XYZ, XYZ_to_Hunter_Lab,
    XYZ_to_Hunter_Rdab, XYZ_to_IPT, XYZ_to_JzAzBz, XYZ_to_K_ab_HunterLab1966,
    XYZ_to_Lab, XYZ_to_Luv, XYZ_to_OSA_UCS, XYZ_to_RGB, XYZ_to_UCS, XYZ_to_UVW,
    XYZ_to_hdr_CIELab, XYZ_to_hdr_IPT, XYZ_to_sRGB, XYZ_to_xy, XYZ_to_xyY,
    YCBCR_WEIGHTS, YCbCr_to_RGB, YcCbcCrc_to_RGB, YCoCg_to_RGB,
    chromatically_adapted_primaries, decoding_cctf, encoding_cctf, eotf,
    eotf_reverse, full_to_legal, gamma_function, hdr_CIELab_to_XYZ,
    hdr_IPT_to_XYZ, legal_to_full, linear_function, log_decoding_curve,
    log_encoding_curve, normalised_primary_matrix, oetf, oetf_reverse, ootf,
    ootf_reverse, primaries_whitepoint, sd_to_aces_relative_exposure_values,
    sRGB_to_XYZ, xyY_to_XYZ, xyY_to_xy, xy_to_Luv_uv, xy_to_UCS_uv, xy_to_XYZ,
    xy_to_xyY)
from .corresponding import (BRENEMAN_EXPERIMENTS,
                            BRENEMAN_EXPERIMENTS_PRIMARIES_CHROMATICITIES,
                            CORRESPONDING_CHROMATICITIES_PREDICTION_MODELS,
                            corresponding_chromaticities_prediction)
from .contrast import (CONTRAST_SENSITIVITY_METHODS,
                       contrast_sensitivity_function)
from .phenomena import (rayleigh_scattering, scattering_cross_section,
                        sd_rayleigh_scattering)
from .notation import (MUNSELL_COLOURS, MUNSELL_VALUE_METHODS,
                       munsell_colour_to_xyY, munsell_value,
                       xyY_to_munsell_colour)
from .quality import colour_quality_scale, colour_rendering_index
from .recovery import XYZ_TO_SD_METHODS, XYZ_to_sd
from .temperature import (CCT_TO_UV_METHODS, CCT_TO_XY_METHODS, CCT_to_uv,
                          CCT_to_xy, UV_TO_CCT_METHODS, XY_TO_CCT_METHODS,
                          uv_to_CCT, xy_to_CCT)
from .volume import (
    ILLUMINANTS_OPTIMAL_COLOUR_STIMULI, RGB_colourspace_limits,
    RGB_colourspace_pointer_gamut_coverage_MonteCarlo,
    RGB_colourspace_visible_spectrum_coverage_MonteCarlo,
    RGB_colourspace_volume_MonteCarlo,
    RGB_colourspace_volume_coverage_MonteCarlo, is_within_macadam_limits,
    is_within_mesh_volume, is_within_pointer_gamut, is_within_visible_spectrum)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'domain_range_scale', 'get_domain_range_scale', 'set_domain_range_scale'
]
__all__ += [
    'CHROMATIC_ADAPTATION_METHODS', 'CHROMATIC_ADAPTATION_TRANSFORMS',
    'CMCCAT2000_VIEWING_CONDITIONS', 'chromatic_adaptation'
]
__all__ += [
    'CubicSplineInterpolator', 'Extrapolator', 'KernelInterpolator',
    'NearestNeighbourInterpolator', 'LinearInterpolator', 'NullInterpolator',
    'PchipInterpolator', 'SpragueInterpolator', 'TABLE_INTERPOLATION_METHODS',
    'kernel_cardinal_spline', 'kernel_lanczos', 'kernel_linear',
    'kernel_nearest_neighbour', 'kernel_sinc', 'table_interpolation',
    'lagrange_coefficients'
]
__all__ += [
    'ASTME30815_PRACTISE_SHAPE', 'BANDPASS_CORRECTION_METHODS', 'CMFS',
    'DEFAULT_SPECTRAL_SHAPE', 'HUNTERLAB_ILLUMINANTS', 'ILLUMINANTS',
    'ILLUMINANTS_SDS', 'LEFS', 'LIGHTNESS_METHODS', 'LIGHT_SOURCES',
    'LIGHT_SOURCES_SDS', 'LMS_CMFS', 'LUMINANCE_METHODS',
    'MULTI_SD_TO_XYZ_METHODS', 'MultiSpectralDistribution', 'PHOTOPIC_LEFS',
    'RGB_CMFS', 'SCOTOPIC_LEFS', 'SD_GAUSSIAN_METHODS',
    'SD_MULTI_LEDS_METHODS', 'SD_SINGLE_LED_METHODS', 'SD_TO_XYZ_METHODS',
    'STANDARD_OBSERVERS_CMFS', 'SpectralDistribution', 'SpectralShape',
    'WHITENESS_METHODS', 'YELLOWNESS_METHODS', 'bandpass_correction',
    'colorimetric_purity', 'complementary_wavelength', 'dominant_wavelength',
    'excitation_purity', 'lightness', 'luminance', 'luminous_efficacy',
    'luminous_efficiency', 'luminous_flux', 'multi_sds_to_XYZ',
    'sd_CIE_standard_illuminant_A', 'sd_CIE_illuminant_D_series',
    'sd_blackbody', 'sd_constant', 'sd_gaussian',
    'sd_mesopic_luminous_efficiency_function', 'sd_multi_leds', 'sd_ones',
    'sd_zeros', 'sd_single_led', 'sd_to_XYZ', 'wavelength_to_XYZ', 'whiteness',
    'yellowness'
]
__all__ += [
    'CVD_MATRICES_MACHADO2010', 'anomalous_trichromacy_cmfs_Machado2009',
    'anomalous_trichromacy_matrix_Machado2009', 'cvd_matrix_Machado2009'
]
__all__ += [
    'ATD95_Specification', 'CAM16_Specification', 'CAM16_VIEWING_CONDITIONS',
    'CAM16_to_XYZ', 'CIECAM02_Specification', 'CIECAM02_VIEWING_CONDITIONS',
    'CIECAM02_to_XYZ', 'HUNT_VIEWING_CONDITIONS', 'Hunt_Specification',
    'LLAB_Specification', 'LLAB_VIEWING_CONDITIONS',
    'Nayatani95_Specification', 'RLAB_D_FACTOR', 'RLAB_Specification',
    'RLAB_VIEWING_CONDITIONS', 'XYZ_to_ATD95', 'XYZ_to_CAM16',
    'XYZ_to_CIECAM02', 'XYZ_to_Hunt', 'XYZ_to_LLAB', 'XYZ_to_Nayatani95',
    'XYZ_to_RLAB'
]
__all__ += ['DELTA_E_METHODS', 'delta_E']
__all__ += [
    'CAMERAS_RGB_SPECTRAL_SENSITIVITIES', 'COLOURCHECKERS',
    'COLOURCHECKERS_SDS', 'DISPLAYS_RGB_PRIMARIES',
    'POLYNOMIAL_EXPANSION_METHODS', 'polynomial_expansion',
    'COLOUR_CORRECTION_MATRIX_METHODS', 'colour_correction_matrix',
    'COLOUR_CORRECTION_METHODS', 'colour_correction'
]
__all__ += [
    'LUT1D', 'LUT3x1D', 'LUT3D', 'LUTSequence',
    'SpectralDistribution_IESTM2714', 'read_image', 'read_LUT',
    'read_sds_from_csv_file', 'read_sds_from_xrite_file',
    'read_spectral_data_from_csv_file', 'write_image', 'write_LUT',
    'write_sds_to_csv_file'
]
__all__ += [
    'CAM02LCD_to_JMh_CIECAM02', 'CAM02SCD_to_JMh_CIECAM02',
    'CAM02UCS_to_JMh_CIECAM02', 'CAM16LCD_to_JMh_CAM16',
    'CAM16SCD_to_JMh_CAM16', 'CAM16UCS_to_JMh_CAM16', 'CMYK_to_CMY',
    'CMY_to_CMYK', 'CMY_to_RGB', 'CV_range', 'DECODING_CCTFS', 'DIN99_to_Lab',
    'ENCODING_CCTFS', 'EOTFS', 'EOTFS_REVERSE', 'HDR_CIELAB_METHODS',
    'HDR_IPT_METHODS', 'HSL_to_RGB', 'HSV_to_RGB', 'Hunter_Lab_to_XYZ',
    'Hunter_Rdab_to_XYZ', 'ICTCP_to_RGB', 'IPT_hue_angle', 'IPT_to_XYZ',
    'JMh_CAM16_to_CAM16LCD', 'JMh_CAM16_to_CAM16SCD', 'JMh_CAM16_to_CAM16UCS',
    'JMh_CIECAM02_to_CAM02LCD', 'JMh_CIECAM02_to_CAM02SCD',
    'JMh_CIECAM02_to_CAM02UCS', 'JzAzBz_to_XYZ', 'LCHab_to_Lab',
    'LCHuv_to_Luv', 'LOG_DECODING_CURVES', 'LOG_ENCODING_CURVES',
    'Lab_to_DIN99', 'Lab_to_LCHab', 'Lab_to_XYZ', 'Luv_to_LCHuv', 'Luv_to_XYZ',
    'Luv_to_uv', 'Luv_uv_to_xy', 'OETFS', 'OETFS_REVERSE', 'OOTFS',
    'MACADAM_1942_ELLIPSES_DATA', 'OOTFS_REVERSE', 'OSA_UCS_to_XYZ',
    'POINTER_GAMUT_BOUNDARIES', 'POINTER_GAMUT_DATA',
    'POINTER_GAMUT_ILLUMINANT', 'Prismatic_to_RGB', 'RGB_COLOURSPACES',
    'RGB_Colourspace', 'RGB_luminance', 'RGB_luminance_equation', 'RGB_to_CMY',
    'RGB_to_HSL', 'RGB_to_HSV', 'RGB_to_ICTCP', 'RGB_to_Prismatic',
    'RGB_to_RGB', 'RGB_to_RGB_matrix', 'RGB_to_XYZ', 'RGB_to_YCbCr',
    'RGB_to_YcCbcCrc', 'RGB_to_YCoCg', 'UCS_to_XYZ', 'UCS_to_uv',
    'UCS_uv_to_xy', 'UVW_to_XYZ', 'XYZ_to_Hunter_Lab', 'XYZ_to_Hunter_Rdab',
    'XYZ_to_IPT', 'XYZ_to_JzAzBz', 'XYZ_to_K_ab_HunterLab1966', 'XYZ_to_Lab',
    'XYZ_to_Luv', 'XYZ_to_OSA_UCS', 'XYZ_to_RGB', 'XYZ_to_UCS', 'XYZ_to_UVW',
    'XYZ_to_hdr_CIELab', 'XYZ_to_hdr_IPT', 'XYZ_to_sRGB', 'XYZ_to_xy',
    'XYZ_to_xyY', 'YCBCR_WEIGHTS', 'YCbCr_to_RGB', 'YcCbcCrc_to_RGB',
    'YCoCg_to_RGB', 'chromatically_adapted_primaries', 'decoding_cctf',
    'encoding_cctf', 'eotf', 'eotf_reverse', 'full_to_legal', 'gamma_function',
    'hdr_CIELab_to_XYZ', 'hdr_IPT_to_XYZ', 'legal_to_full', 'linear_function',
    'log_decoding_curve', 'log_encoding_curve', 'normalised_primary_matrix',
    'oetf', 'oetf_reverse', 'ootf', 'ootf_reverse', 'primaries_whitepoint',
    'sd_to_aces_relative_exposure_values', 'sRGB_to_XYZ', 'xyY_to_XYZ',
    'xyY_to_xy', 'xy_to_Luv_uv', 'xy_to_UCS_uv', 'xy_to_XYZ', 'xy_to_xyY'
]
__all__ += [
    'BRENEMAN_EXPERIMENTS', 'BRENEMAN_EXPERIMENTS_PRIMARIES_CHROMATICITIES',
    'CORRESPONDING_CHROMATICITIES_PREDICTION_MODELS',
    'corresponding_chromaticities_prediction'
]
__all__ += ['CONTRAST_SENSITIVITY_METHODS', 'contrast_sensitivity_function']
__all__ += [
    'rayleigh_scattering', 'scattering_cross_section', 'sd_rayleigh_scattering'
]
__all__ += [
    'MUNSELL_COLOURS', 'MUNSELL_VALUE_METHODS', 'munsell_colour_to_xyY',
    'munsell_value', 'xyY_to_munsell_colour'
]
__all__ += ['colour_quality_scale', 'colour_rendering_index']
__all__ += ['XYZ_TO_SD_METHODS', 'XYZ_to_sd']
__all__ += [
    'CCT_TO_UV_METHODS', 'CCT_TO_XY_METHODS', 'CCT_to_uv', 'CCT_to_xy',
    'UV_TO_CCT_METHODS', 'XY_TO_CCT_METHODS', 'uv_to_CCT', 'xy_to_CCT'
]
__all__ += [
    'ILLUMINANTS_OPTIMAL_COLOUR_STIMULI', 'RGB_colourspace_limits',
    'RGB_colourspace_pointer_gamut_coverage_MonteCarlo',
    'RGB_colourspace_visible_spectrum_coverage_MonteCarlo',
    'RGB_colourspace_volume_MonteCarlo',
    'RGB_colourspace_volume_coverage_MonteCarlo', 'is_within_macadam_limits',
    'is_within_mesh_volume', 'is_within_pointer_gamut',
    'is_within_visible_spectrum'
]
__application_name__ = 'Colour'

__major_version__ = '0'
__minor_version__ = '3'
__change_version__ = '12'
__version__ = '.'.join(
    (__major_version__,
     __minor_version__,
     __change_version__))  # yapf: disable

# TODO: Remove legacy printing support when deemed appropriate.
try:
    np.set_printoptions(legacy='1.13')
except TypeError:
    pass


# ----------------------------------------------------------------------------#
# ---                API Changes and Deprecation Management                ---#
# ----------------------------------------------------------------------------#
class colour(ModuleAPI):
    def __getattr__(self, attribute):
        return super(colour, self).__getattr__(attribute)


colour.__application_name__ = __application_name__

colour.__major_version__ = __major_version__
colour.__minor_version__ = __minor_version__
colour.__change_version__ = __change_version__
colour.__version__ = __version__

# v0.3.11
API_CHANGES = {
    'Future Access Change': [
        [
            'colour.ACES_2065_1_COLOURSPACE',
            'colour.models.ACES_2065_1_COLOURSPACE',
        ],
        [
            'colour.ACES_CC_COLOURSPACE',
            'colour.models.ACES_CC_COLOURSPACE',
        ],
        [
            'colour.ACES_CCT_COLOURSPACE',
            'colour.models.ACES_CCT_COLOURSPACE',
        ],
        [
            'colour.ACES_CG_COLOURSPACE',
            'colour.models.ACES_CG_COLOURSPACE',
        ],
        [
            'colour.ACES_PROXY_COLOURSPACE',
            'colour.models.ACES_PROXY_COLOURSPACE',
        ],
        [
            'colour.ACES_RICD',
            'colour.models.ACES_RICD',
        ],
        [
            'colour.ADOBE_RGB_1998_COLOURSPACE',
            'colour.models.ADOBE_RGB_1998_COLOURSPACE',
        ],
        [
            'colour.ADOBE_WIDE_GAMUT_RGB_COLOURSPACE',
            'colour.models.ADOBE_WIDE_GAMUT_RGB_COLOURSPACE',
        ],
        [
            'colour.ALEXA_WIDE_GAMUT_COLOURSPACE',
            'colour.models.ALEXA_WIDE_GAMUT_COLOURSPACE',
        ],
        [
            'colour.APPLE_RGB_COLOURSPACE',
            'colour.models.APPLE_RGB_COLOURSPACE',
        ],
        [
            'colour.as_namedtuple',
            'colour.utilities.as_namedtuple',
        ],
        [
            'colour.as_numeric',
            'colour.utilities.as_numeric',
        ],
        [
            'colour.AVOGADRO_CONSTANT',
            'colour.constants.AVOGADRO_CONSTANT',
        ],
        [
            'colour.bandpass_correction_Stearns1988',
            'colour.colorimetry.bandpass_correction_Stearns1988',
        ],
        [
            'colour.batch',
            'colour.utilities.batch',
        ],
        [
            'colour.BEST_RGB_COLOURSPACE',
            'colour.models.BEST_RGB_COLOURSPACE',
        ],
        [
            'colour.BETA_RGB_COLOURSPACE',
            'colour.models.BETA_RGB_COLOURSPACE',
        ],
        [
            'colour.blackbody_spectral_radiance',
            'colour.colorimetry.blackbody_spectral_radiance',
        ],
        [
            'colour.BOLTZMANN_CONSTANT',
            'colour.constants.BOLTZMANN_CONSTANT',
        ],
        [
            'colour.BRADFORD_CAT',
            'colour.adaptation.BRADFORD_CAT',
        ],
        [
            'colour.BS_CAT',
            'colour.adaptation.BS_CAT',
        ],
        [
            'colour.BS_PC_CAT',
            'colour.adaptation.BS_PC_CAT',
        ],
        [
            'colour.BT2020_COLOURSPACE',
            'colour.models.BT2020_COLOURSPACE',
        ],
        [
            'colour.BT470_525_COLOURSPACE',
            'colour.models.BT470_525_COLOURSPACE',
        ],
        [
            'colour.BT470_625_COLOURSPACE',
            'colour.models.BT470_625_COLOURSPACE',
        ],
        [
            'colour.BT709_COLOURSPACE',
            'colour.models.BT709_COLOURSPACE',
        ],
        [
            'colour.cartesian_to_cylindrical',
            'colour.algebra.cartesian_to_cylindrical',
        ],
        [
            'colour.cartesian_to_polar',
            'colour.algebra.cartesian_to_polar',
        ],
        [
            'colour.cartesian_to_spherical',
            'colour.algebra.cartesian_to_spherical',
        ],
        [
            'colour.CaseInsensitiveMapping',
            'colour.utilities.CaseInsensitiveMapping',
        ],
        [
            'colour.CAT02_BRILL_CAT',
            'colour.adaptation.CAT02_BRILL_CAT',
        ],
        [
            'colour.CAT02_CAT',
            'colour.adaptation.CAT02_CAT',
        ],
        [
            'colour.CCT_to_uv_Krystek1985',
            'colour.temperature.CCT_to_uv_Krystek1985',
        ],
        [
            'colour.CCT_to_uv_Ohno2013',
            'colour.temperature.CCT_to_uv_Ohno2013',
        ],
        [
            'colour.CCT_to_uv_Robertson1968',
            'colour.temperature.CCT_to_uv_Robertson1968',
        ],
        [
            'colour.CCT_to_xy_CIE_D',
            'colour.temperature.CCT_to_xy_CIE_D',
        ],
        [
            'colour.CCT_to_xy_Kang2002',
            'colour.temperature.CCT_to_xy_Kang2002',
        ],
        [
            'colour.centroid',
            'colour.utilities.centroid',
        ],
        [
            'colour.chromatic_adaptation_CIE1994',
            'colour.adaptation.chromatic_adaptation_CIE1994',
        ],
        [
            'colour.chromatic_adaptation_CMCCAT2000',
            'colour.adaptation.chromatic_adaptation_CMCCAT2000',
        ],
        [
            'colour.chromatic_adaptation_Fairchild1990',
            'colour.adaptation.chromatic_adaptation_Fairchild1990',
        ],
        [
            'colour.chromatic_adaptation_matrix_VonKries',
            'colour.adaptation.chromatic_adaptation_matrix_VonKries',
        ],
        [
            'colour.chromatic_adaptation_VonKries',
            'colour.adaptation.chromatic_adaptation_VonKries',
        ],
        [
            'colour.CIE_RGB_COLOURSPACE',
            'colour.models.CIE_RGB_COLOURSPACE',
        ],
        [
            'colour.CINEMA_GAMUT_COLOURSPACE',
            'colour.models.CINEMA_GAMUT_COLOURSPACE',
        ],
        [
            'colour.closest',
            'colour.utilities.closest',
        ],
        [
            'colour.closest_indexes',
            'colour.utilities.closest_indexes',
        ],
        [
            'colour.CMCCAT2000_CAT',
            'colour.adaptation.CMCCAT2000_CAT',
        ],
        [
            'colour.CMCCAT97_CAT',
            'colour.adaptation.CMCCAT97_CAT',
        ],
        [
            'colour.COLOR_MATCH_RGB_COLOURSPACE',
            'colour.models.COLOR_MATCH_RGB_COLOURSPACE',
        ],
        [
            'colour.COLOURCHECKER_INDEXES_TO_NAMES_MAPPING',
            'colour.characterisation.COLOURCHECKER_INDEXES_TO_NAMES_MAPPING',
        ],
        [
            'colour.COLOURSPACE_MODELS',
            'colour.models.COLOURSPACE_MODELS',
        ],
        [
            'colour.COLOURSPACE_MODELS_LABELS',
            'colour.models.COLOURSPACE_MODELS_LABELS',
        ],
        [
            'colour.corresponding_chromaticities_prediction_CIE1994',
            'colour.corresponding.'
            'corresponding_chromaticities_prediction_CIE1994',
        ],
        [
            'colour.corresponding_chromaticities_prediction_CMCCAT2000',
            'colour.corresponding.'
            'corresponding_chromaticities_prediction_CMCCAT2000',
        ],
        [
            'colour.corresponding_chromaticities_prediction_Fairchild1990',
            'colour.corresponding.'
            'corresponding_chromaticities_prediction_Fairchild1990',
        ],
        [
            'colour.corresponding_chromaticities_prediction_VonKries',
            'colour.corresponding.'
            'corresponding_chromaticities_prediction_VonKries',
        ],
        [
            'colour.cylindrical_to_cartesian',
            'colour.algebra.cylindrical_to_cartesian',
        ],
        [
            'colour.D_ILLUMINANTS_S_SPDS',
            'colour.colorimetry.D_ILLUMINANTS_S_SDS',
        ],
        [
            'colour.DCI_P3_COLOURSPACE',
            'colour.models.DCI_P3_COLOURSPACE',
        ],
        [
            'colour.DCI_P3_P_COLOURSPACE',
            'colour.models.DCI_P3_P_COLOURSPACE',
        ],
        [
            'colour.DEFAULT_FLOAT_DTYPE',
            'colour.constants.DEFAULT_FLOAT_DTYPE',
        ],
        [
            'colour.delta_E_CAM02LCD',
            'colour.difference.delta_E_CAM02LCD',
        ],
        [
            'colour.delta_E_CAM02SCD',
            'colour.difference.delta_E_CAM02SCD',
        ],
        [
            'colour.delta_E_CAM02UCS',
            'colour.difference.delta_E_CAM02UCS',
        ],
        [
            'colour.delta_E_CAM16LCD',
            'colour.difference.delta_E_CAM16LCD',
        ],
        [
            'colour.delta_E_CAM16SCD',
            'colour.difference.delta_E_CAM16SCD',
        ],
        [
            'colour.delta_E_CAM16UCS',
            'colour.difference.delta_E_CAM16UCS',
        ],
        [
            'colour.delta_E_CIE1976',
            'colour.difference.delta_E_CIE1976',
        ],
        [
            'colour.delta_E_CIE1994',
            'colour.difference.delta_E_CIE1994',
        ],
        [
            'colour.delta_E_CIE2000',
            'colour.difference.delta_E_CIE2000',
        ],
        [
            'colour.delta_E_CMC',
            'colour.difference.delta_E_CMC',
        ],
        [
            'colour.DON_RGB_4_COLOURSPACE',
            'colour.models.DON_RGB_4_COLOURSPACE',
        ],
        [
            'colour.dot_matrix',
            'colour.utilities.dot_matrix',
        ],
        [
            'colour.dot_vector',
            'colour.utilities.dot_vector',
        ],
        [
            'colour.DRAGON_COLOR_2_COLOURSPACE',
            'colour.models.DRAGON_COLOR_2_COLOURSPACE',
        ],
        [
            'colour.DRAGON_COLOR_COLOURSPACE',
            'colour.models.DRAGON_COLOR_COLOURSPACE',
        ],
        [
            'colour.ECI_RGB_V2_COLOURSPACE',
            'colour.models.ECI_RGB_V2_COLOURSPACE',
        ],
        [
            'colour.EKTA_SPACE_PS_5_COLOURSPACE',
            'colour.models.EKTA_SPACE_PS_5_COLOURSPACE',
        ],
        [
            'colour.eotf_BT1886',
            'colour.models.eotf_BT1886',
        ],
        [
            'colour.eotf_BT2020',
            'colour.models.eotf_BT2020',
        ],
        [
            'colour.eotf_BT2100_HLG',
            'colour.models.eotf_BT2100_HLG',
        ],
        [
            'colour.eotf_BT2100_PQ',
            'colour.models.eotf_BT2100_PQ',
        ],
        [
            'colour.eotf_DCIP3',
            'colour.models.eotf_DCDM',
        ],
        [
            'colour.eotf_DICOMGSDF',
            'colour.models.eotf_DICOMGSDF',
        ],
        [
            'colour.eotf_ProPhotoRGB',
            'colour.models.eotf_ProPhotoRGB',
        ],
        [
            'colour.eotf_reverse_BT1886',
            'colour.models.eotf_reverse_BT1886',
        ],
        [
            'colour.eotf_reverse_BT2100_HLG',
            'colour.models.eotf_reverse_BT2100_HLG',
        ],
        [
            'colour.eotf_reverse_BT2100_PQ',
            'colour.models.eotf_reverse_BT2100_PQ',
        ],
        [
            'colour.eotf_RIMMRGB',
            'colour.models.eotf_RIMMRGB',
        ],
        [
            'colour.eotf_ROMMRGB',
            'colour.models.eotf_ROMMRGB',
        ],
        [
            'colour.eotf_SMPTE240M',
            'colour.models.eotf_SMPTE240M',
        ],
        [
            'colour.eotf_ST2084',
            'colour.models.eotf_ST2084',
        ],
        [
            'colour.EPSILON',
            'colour.constants.EPSILON',
        ],
        [
            'colour.ERIMM_RGB_COLOURSPACE',
            'colour.models.ERIMM_RGB_COLOURSPACE',
        ],
        [
            'colour.euclidean_distance',
            'colour.algebra.euclidean_distance',
        ],
        [
            'colour.FAIRCHILD_CAT',
            'colour.adaptation.FAIRCHILD_CAT',
        ],
        [
            'colour.fill_nan',
            'colour.utilities.fill_nan',
        ],
        [
            'colour.filter_kwargs',
            'colour.utilities.filter_kwargs',
        ],
        [
            'colour.filter_warnings',
            'colour.utilities.filter_warnings',
        ],
        [
            'colour.first_item',
            'colour.utilities.first_item',
        ],
        [
            'colour.handle_numpy_errors',
            'colour.utilities.handle_numpy_errors',
        ],
        [
            'colour.ignore_numpy_errors',
            'colour.utilities.ignore_numpy_errors',
        ],
        [
            'colour.ignore_python_warnings',
            'colour.utilities.ignore_python_warnings',
        ],
        [
            'colour.in_array',
            'colour.utilities.in_array',
        ],
        [
            'colour.interval',
            'colour.utilities.interval',
        ],
        [
            'colour.is_identity',
            'colour.algebra.is_identity',
        ],
        [
            'colour.is_integer',
            'colour.utilities.is_integer',
        ],
        [
            'colour.is_iterable',
            'colour.utilities.is_iterable',
        ],
        [
            'colour.is_numeric',
            'colour.utilities.is_numeric',
        ],
        [
            'colour.is_openimageio_installed',
            'colour.utilities.is_openimageio_installed',
        ],
        [
            'colour.is_pandas_installed',
            'colour.utilities.is_pandas_installed',
        ],
        [
            'colour.is_string',
            'colour.utilities.is_string',
        ],
        [
            'colour.is_uniform',
            'colour.utilities.is_uniform',
        ],
        [
            'colour.K_M',
            'colour.constants.K_M',
        ],
        [
            'colour.KP_M',
            'colour.constants.KP_M',
        ],
        [
            'colour.LIGHT_SPEED',
            'colour.constants.LIGHT_SPEED',
        ],
        [
            'colour.lightness_CIE1976',
            'colour.colorimetry.lightness_CIE1976',
        ],
        [
            'colour.lightness_Fairchild2010',
            'colour.colorimetry.lightness_Fairchild2010',
        ],
        [
            'colour.lightness_Fairchild2011',
            'colour.colorimetry.lightness_Fairchild2011',
        ],
        [
            'colour.lightness_Glasser1958',
            'colour.colorimetry.lightness_Glasser1958',
        ],
        [
            'colour.lightness_Wyszecki1963',
            'colour.colorimetry.lightness_Wyszecki1963',
        ],
        [
            'colour.linear_conversion',
            'colour.utilities.linear_conversion',
        ],
        [
            'colour.LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs',
            'colour.colorimetry.LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs',
        ],
        [
            'colour.LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs',
            'colour.colorimetry.LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs',
        ],
        [
            'colour.LMS_ConeFundamentals',
            'colour.colorimetry.LMS_ConeFundamentals',
        ],
        [
            'colour.log_decoding_ACEScc',
            'colour.models.log_decoding_ACEScc',
        ],
        [
            'colour.log_decoding_ACEScct',
            'colour.models.log_decoding_ACEScct',
        ],
        [
            'colour.log_decoding_ACESproxy',
            'colour.models.log_decoding_ACESproxy',
        ],
        [
            'colour.log_decoding_ALEXALogC',
            'colour.models.log_decoding_ALEXALogC',
        ],
        [
            'colour.log_decoding_CanonLog',
            'colour.models.log_decoding_CanonLog',
        ],
        [
            'colour.log_decoding_CanonLog2',
            'colour.models.log_decoding_CanonLog2',
        ],
        [
            'colour.log_decoding_CanonLog3',
            'colour.models.log_decoding_CanonLog3',
        ],
        [
            'colour.log_decoding_Cineon',
            'colour.models.log_decoding_Cineon',
        ],
        [
            'colour.log_decoding_ERIMMRGB',
            'colour.models.log_decoding_ERIMMRGB',
        ],
        [
            'colour.log_decoding_Log3G10',
            'colour.models.log_decoding_Log3G10',
        ],
        [
            'colour.log_decoding_Log3G12',
            'colour.models.log_decoding_Log3G12',
        ],
        [
            'colour.log_decoding_Panalog',
            'colour.models.log_decoding_Panalog',
        ],
        [
            'colour.log_decoding_PivotedLog',
            'colour.models.log_decoding_PivotedLog',
        ],
        [
            'colour.log_decoding_Protune',
            'colour.models.log_decoding_Protune',
        ],
        [
            'colour.log_decoding_REDLog',
            'colour.models.log_decoding_REDLog',
        ],
        [
            'colour.log_decoding_REDLogFilm',
            'colour.models.log_decoding_REDLogFilm',
        ],
        [
            'colour.log_decoding_SLog',
            'colour.models.log_decoding_SLog',
        ],
        [
            'colour.log_decoding_SLog2',
            'colour.models.log_decoding_SLog2',
        ],
        [
            'colour.log_decoding_SLog3',
            'colour.models.log_decoding_SLog3',
        ],
        [
            'colour.log_decoding_ViperLog',
            'colour.models.log_decoding_ViperLog',
        ],
        [
            'colour.log_decoding_VLog',
            'colour.models.log_decoding_VLog',
        ],
        [
            'colour.log_encoding_ACEScc',
            'colour.models.log_encoding_ACEScc',
        ],
        [
            'colour.log_encoding_ACEScct',
            'colour.models.log_encoding_ACEScct',
        ],
        [
            'colour.log_encoding_ACESproxy',
            'colour.models.log_encoding_ACESproxy',
        ],
        [
            'colour.log_encoding_ALEXALogC',
            'colour.models.log_encoding_ALEXALogC',
        ],
        [
            'colour.log_encoding_CanonLog',
            'colour.models.log_encoding_CanonLog',
        ],
        [
            'colour.log_encoding_CanonLog2',
            'colour.models.log_encoding_CanonLog2',
        ],
        [
            'colour.log_encoding_CanonLog3',
            'colour.models.log_encoding_CanonLog3',
        ],
        [
            'colour.log_encoding_Cineon',
            'colour.models.log_encoding_Cineon',
        ],
        [
            'colour.log_encoding_ERIMMRGB',
            'colour.models.log_encoding_ERIMMRGB',
        ],
        [
            'colour.log_encoding_Log3G10',
            'colour.models.log_encoding_Log3G10',
        ],
        [
            'colour.log_encoding_Log3G12',
            'colour.models.log_encoding_Log3G12',
        ],
        [
            'colour.log_encoding_Panalog',
            'colour.models.log_encoding_Panalog',
        ],
        [
            'colour.log_encoding_PivotedLog',
            'colour.models.log_encoding_PivotedLog',
        ],
        [
            'colour.log_encoding_Protune',
            'colour.models.log_encoding_Protune',
        ],
        [
            'colour.log_encoding_REDLog',
            'colour.models.log_encoding_REDLog',
        ],
        [
            'colour.log_encoding_REDLogFilm',
            'colour.models.log_encoding_REDLogFilm',
        ],
        [
            'colour.log_encoding_SLog',
            'colour.models.log_encoding_SLog',
        ],
        [
            'colour.log_encoding_SLog2',
            'colour.models.log_encoding_SLog2',
        ],
        [
            'colour.log_encoding_SLog3',
            'colour.models.log_encoding_SLog3',
        ],
        [
            'colour.log_encoding_VLog',
            'colour.models.log_encoding_VLog',
        ],
        [
            'colour.Lookup',
            'colour.utilities.Lookup',
        ],
        [
            'colour.luminance_ASTMD153508',
            'colour.colorimetry.luminance_ASTMD153508',
        ],
        [
            'colour.luminance_CIE1976',
            'colour.colorimetry.luminance_CIE1976',
        ],
        [
            'colour.luminance_Fairchild2010',
            'colour.colorimetry.luminance_Fairchild2010',
        ],
        [
            'colour.luminance_Fairchild2011',
            'colour.colorimetry.luminance_Fairchild2011',
        ],
        [
            'colour.luminance_Newhall1943',
            'colour.colorimetry.luminance_Newhall1943',
        ],
        [
            'colour.MAX_RGB_COLOURSPACE',
            'colour.models.MAX_RGB_COLOURSPACE',
        ],
        [
            'colour.mesopic_weighting_function',
            'colour.colorimetry.mesopic_weighting_function',
        ],
        [
            'colour.message_box',
            'colour.utilities.message_box',
        ],
        [
            'colour.MultiSignal',
            'colour.continuous.MultiSignal',
        ],
        [
            'colour.MUNSELL_COLOURS_1929',
            'colour.notation.MUNSELL_COLOURS_1929',
        ],
        [
            'colour.MUNSELL_COLOURS_ALL',
            'colour.notation.MUNSELL_COLOURS_ALL',
        ],
        [
            'colour.MUNSELL_COLOURS_REAL',
            'colour.notation.MUNSELL_COLOURS_REAL',
        ],
        [
            'colour.munsell_value_ASTMD153508',
            'colour.notation.munsell_value_ASTMD153508',
        ],
        [
            'colour.munsell_value_Ladd1955',
            'colour.notation.munsell_value_Ladd1955',
        ],
        [
            'colour.munsell_value_McCamy1987',
            'colour.notation.munsell_value_McCamy1987',
        ],
        [
            'colour.munsell_value_Moon1943',
            'colour.notation.munsell_value_Moon1943',
        ],
        [
            'colour.munsell_value_Munsell1933',
            'colour.notation.munsell_value_Munsell1933',
        ],
        [
            'colour.munsell_value_Priest1920',
            'colour.notation.munsell_value_Priest1920',
        ],
        [
            'colour.munsell_value_Saunderson1944',
            'colour.notation.munsell_value_Saunderson1944',
        ],
        [
            'colour.ndarray_write',
            'colour.utilities.ndarray_write',
        ],
        [
            'colour.normalise_maximum',
            'colour.utilities.normalise_maximum',
        ],
        [
            'colour.normalise_vector',
            'colour.algebra.normalise_vector',
        ],
        [
            'colour.NTSC_COLOURSPACE',
            'colour.models.NTSC_COLOURSPACE',
        ],
        [
            'colour.numpy_print_options',
            'colour.utilities.numpy_print_options',
        ],
        [
            'colour.oetf_ARIBSTDB67',
            'colour.models.oetf_ARIBSTDB67',
        ],
        [
            'colour.oetf_BT2020',
            'colour.models.oetf_BT2020',
        ],
        [
            'colour.oetf_BT2100_HLG',
            'colour.models.oetf_BT2100_HLG',
        ],
        [
            'colour.oetf_BT2100_PQ',
            'colour.models.oetf_BT2100_PQ',
        ],
        [
            'colour.oetf_BT601',
            'colour.models.oetf_BT601',
        ],
        [
            'colour.oetf_BT709',
            'colour.models.oetf_BT709',
        ],
        [
            'colour.oetf_DCIP3',
            'colour.models.eotf_reverse_DCIP3',
        ],
        [
            'colour.oetf_DICOMGSDF',
            'colour.models.oetf_DICOMGSDF',
        ],
        [
            'colour.oetf_ProPhotoRGB',
            'colour.models.oetf_ProPhotoRGB',
        ],
        [
            'colour.oetf_reverse_ARIBSTDB67',
            'colour.models.oetf_reverse_ARIBSTDB67',
        ],
        [
            'colour.oetf_reverse_BT2100_HLG',
            'colour.models.oetf_reverse_BT2100_HLG',
        ],
        [
            'colour.oetf_reverse_BT2100_PQ',
            'colour.models.oetf_reverse_BT2100_PQ',
        ],
        [
            'colour.oetf_reverse_BT601',
            'colour.models.oetf_reverse_BT601',
        ],
        [
            'colour.oetf_reverse_BT709',
            'colour.models.oetf_reverse_BT709',
        ],
        [
            'colour.oetf_reverse_sRGB',
            'colour.models.oetf_reverse_sRGB',
        ],
        [
            'colour.oetf_RIMMRGB',
            'colour.models.oetf_RIMMRGB',
        ],
        [
            'colour.oetf_ROMMRGB',
            'colour.models.oetf_ROMMRGB',
        ],
        [
            'colour.oetf_SMPTE240M',
            'colour.models.oetf_SMPTE240M',
        ],
        [
            'colour.oetf_sRGB',
            'colour.models.oetf_sRGB',
        ],
        [
            'colour.oetf_ST2084',
            'colour.models.oetf_ST2084',
        ],
        [
            'colour.ootf_BT2100_HLG',
            'colour.models.ootf_BT2100_HLG',
        ],
        [
            'colour.ootf_BT2100_PQ',
            'colour.models.ootf_BT2100_PQ',
        ],
        [
            'colour.ootf_reverse_BT2100_HLG',
            'colour.models.ootf_reverse_BT2100_HLG',
        ],
        [
            'colour.ootf_reverse_BT2100_PQ',
            'colour.models.ootf_reverse_BT2100_PQ',
        ],
        [
            'colour.orient',
            'colour.utilities.orient',
        ],
        [
            'colour.PAL_SECAM_COLOURSPACE',
            'colour.models.PAL_SECAM_COLOURSPACE',
        ],
        [
            'colour.PLANCK_CONSTANT',
            'colour.constants.PLANCK_CONSTANT',
        ],
        [
            'colour.planck_law',
            'colour.colorimetry.planck_law',
        ],
        [
            'colour.polar_to_cartesian',
            'colour.algebra.polar_to_cartesian',
        ],
        [
            'colour.print_numpy_errors',
            'colour.utilities.print_numpy_errors',
        ],
        [
            'colour.PROPHOTO_RGB_COLOURSPACE',
            'colour.models.PROPHOTO_RGB_COLOURSPACE',
        ],
        [
            'colour.PROTUNE_NATIVE_COLOURSPACE',
            'colour.models.PROTUNE_NATIVE_COLOURSPACE',
        ],
        [
            'colour.raise_numpy_errors',
            'colour.utilities.raise_numpy_errors',
        ],
        [
            'colour.random_triplet_generator',
            'colour.algebra.random_triplet_generator',
        ],
        [
            'colour.rayleigh_optical_depth',
            'colour.phenomena.rayleigh_optical_depth',
        ],
        [
            'colour.reaction_rate_MichealisMenten',
            'colour.biochemistry.reaction_rate_MichealisMenten',
        ],
        [
            'colour.RED_COLOR_2_COLOURSPACE',
            'colour.models.RED_COLOR_2_COLOURSPACE',
        ],
        [
            'colour.RED_COLOR_3_COLOURSPACE',
            'colour.models.RED_COLOR_3_COLOURSPACE',
        ],
        [
            'colour.RED_COLOR_4_COLOURSPACE',
            'colour.models.RED_COLOR_4_COLOURSPACE',
        ],
        [
            'colour.RED_COLOR_COLOURSPACE',
            'colour.models.RED_COLOR_COLOURSPACE',
        ],
        [
            'colour.RED_WIDE_GAMUT_RGB_COLOURSPACE',
            'colour.models.RED_WIDE_GAMUT_RGB_COLOURSPACE',
        ],
        [
            'colour.RGB_10_degree_cmfs_to_LMS_10_degree_cmfs',
            'colour.colorimetry.RGB_10_degree_cmfs_to_LMS_10_degree_cmfs',
        ],
        [
            'colour.RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs',
            'colour.colorimetry.RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs',
        ],
        [
            'colour.RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs',
            'colour.colorimetry.RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs',
        ],
        [
            'colour.RGB_ColourMatchingFunctions',
            'colour.colorimetry.RGB_ColourMatchingFunctions',
        ],
        [
            'colour.RGB_DisplayPrimaries',
            'colour.characterisation.RGB_DisplayPrimaries',
        ],
        [
            'colour.RGB_SpectralSensitivities',
            'colour.characterisation.RGB_SpectralSensitivities',
        ],
        [
            'colour.RGB_to_sd_Smits1999',
            'colour.recovery.RGB_to_sd_Smits1999',
        ],
        [
            'colour.RIMM_RGB_COLOURSPACE',
            'colour.models.RIMM_RGB_COLOURSPACE',
        ],
        [
            'colour.ROMM_RGB_COLOURSPACE',
            'colour.models.ROMM_RGB_COLOURSPACE',
        ],
        [
            'colour.row_as_diagonal',
            'colour.utilities.row_as_diagonal',
        ],
        [
            'colour.RUSSELL_RGB_COLOURSPACE',
            'colour.models.RUSSELL_RGB_COLOURSPACE',
        ],
        [
            'colour.S_GAMUT_COLOURSPACE',
            'colour.models.S_GAMUT_COLOURSPACE',
        ],
        [
            'colour.S_GAMUT3_CINE_COLOURSPACE',
            'colour.models.S_GAMUT3_CINE_COLOURSPACE',
        ],
        [
            'colour.S_GAMUT3_COLOURSPACE',
            'colour.models.S_GAMUT3_COLOURSPACE',
        ],
        [
            'colour.SHARP_CAT',
            'colour.adaptation.SHARP_CAT',
        ],
        [
            'colour.Signal',
            'colour.continuous.Signal',
        ],
        [
            'colour.SMITS_1999_SPDS',
            'colour.recovery.SMITS_1999_SDS',
        ],
        [
            'colour.SMPTE_240M_COLOURSPACE',
            'colour.models.SMPTE_240M_COLOURSPACE',
        ],
        [
            'colour.sd_to_XYZ_ASTME30815',
            'colour.colorimetry.sd_to_XYZ_ASTME30815',
        ],
        [
            'colour.sd_to_XYZ_integration',
            'colour.colorimetry.sd_to_XYZ_integration',
        ],
        [
            'colour.spherical_to_cartesian',
            'colour.algebra.spherical_to_cartesian',
        ],
        [
            'colour.sRGB_COLOURSPACE',
            'colour.models.sRGB_COLOURSPACE',
        ],
        [
            'colour.Structure',
            'colour.utilities.Structure',
        ],
        [
            'colour.substrate_concentration_MichealisMenten',
            'colour.biochemistry.substrate_concentration_MichealisMenten',
        ],
        [
            'colour.TCS_SPDS',
            'colour.quality.TCS_SDS',
        ],
        [
            'colour.tsplit',
            'colour.utilities.tsplit',
        ],
        [
            'colour.tstack',
            'colour.utilities.tstack',
        ],
        [
            'colour.uv_to_CCT_Ohno2013',
            'colour.temperature.uv_to_CCT_Ohno2013',
        ],
        [
            'colour.uv_to_CCT_Robertson1968',
            'colour.temperature.uv_to_CCT_Robertson1968',
        ],
        [
            'colour.V_GAMUT_COLOURSPACE',
            'colour.models.V_GAMUT_COLOURSPACE',
        ],
        [
            'colour.VON_KRIES_CAT',
            'colour.adaptation.VON_KRIES_CAT',
        ],
        [
            'colour.VS_SPDS',
            'colour.quality.VS_SDS',
        ],
        [
            'colour.warn_numpy_errors',
            'colour.utilities.warn_numpy_errors',
        ],
        [
            'colour.warning',
            'colour.utilities.warning',
        ],
        [
            'colour.whiteness_ASTME313',
            'colour.colorimetry.whiteness_ASTME313',
        ],
        [
            'colour.whiteness_Berger1959',
            'colour.colorimetry.whiteness_Berger1959',
        ],
        [
            'colour.whiteness_CIE2004',
            'colour.colorimetry.whiteness_CIE2004',
        ],
        [
            'colour.whiteness_Ganz1979',
            'colour.colorimetry.whiteness_Ganz1979',
        ],
        [
            'colour.whiteness_Stensby1968',
            'colour.colorimetry.whiteness_Stensby1968',
        ],
        [
            'colour.whiteness_Taube1960',
            'colour.colorimetry.whiteness_Taube1960',
        ],
        [
            'colour.XTREME_RGB_COLOURSPACE',
            'colour.models.XTREME_RGB_COLOURSPACE',
        ],
        [
            'colour.xy_to_CCT_Hernandez1999',
            'colour.temperature.xy_to_CCT_Hernandez1999',
        ],
        [
            'colour.xy_to_CCT_McCamy1992',
            'colour.temperature.xy_to_CCT_McCamy1992',
        ],
        [
            'colour.XYZ_ColourMatchingFunctions',
            'colour.colorimetry.XYZ_ColourMatchingFunctions',
        ],
        [
            'colour.XYZ_SCALING_CAT',
            'colour.adaptation.XYZ_SCALING_CAT',
        ],
        [
            'colour.XYZ_to_sd_Meng2015',
            'colour.recovery.XYZ_to_sd_Meng2015',
        ],
        [
            'colour.yellowness_ASTMD1925',
            'colour.colorimetry.yellowness_ASTMD1925',
        ],
        [
            'colour.yellowness_ASTME313',
            'colour.colorimetry.yellowness_ASTME313'
        ],
    ],
    'Future Access Remove': [
        [
            'colour.AbstractContinuousFunction',
            'colour.continuous.AbstractContinuousFunction',
        ],
        [
            'colour.CAM16_InductionFactors',
            'colour.appearance.CAM16_InductionFactors',
        ],
        [
            'colour.CIECAM02_InductionFactors',
            'colour.appearance.CIECAM02_InductionFactors',
        ],
        [
            'colour.CMCCAT2000_InductionFactors',
            'colour.adaptation.CMCCAT2000_InductionFactors',
        ],
        [
            'colour.CQS_Specification',
            'colour.quality.CQS_Specification',
        ],
        [
            'colour.CRI_Specification',
            'colour.quality.CRI_Specification',
        ],
        [
            'colour.ColourWarning',
            'colour.utilities.ColourWarning',
        ],
        [
            'colour.FLOATING_POINT_NUMBER_PATTERN',
            'colour.constants.FLOATING_POINT_NUMBER_PATTERN',
        ],
        [
            'colour.Hunt_InductionFactors',
            'colour.appearance.Hunt_InductionFactors',
        ],
        [
            'colour.INTEGER_THRESHOLD',
            'colour.constants.INTEGER_THRESHOLD',
        ],
        [
            'colour.LLAB_InductionFactors',
            'colour.appearance.LLAB_InductionFactors',
        ],
        [
            'colour.LineSegmentsIntersections_Specification',
            'colour.algebra.LineSegmentsIntersections_Specification',
        ],
        [
            'colour.XYZ_to_colourspace_model',
            'colour.models.XYZ_to_colourspace_model',
        ],
        [
            'colour.adjust_tristimulus_weighting_factors_ASTME30815',
            'colour.colorimetry.'
            'adjust_tristimulus_weighting_factors_ASTME30815',
        ],
        [
            'colour.chromatic_adaptation_forward_CMCCAT2000',
            'colour.adaptation.chromatic_adaptation_forward_CMCCAT2000',
        ],
        [
            'colour.chromatic_adaptation_reverse_CMCCAT2000',
            'colour.adaptation.chromatic_adaptation_reverse_CMCCAT2000',
        ],
        [
            'colour.extend_line_segment',
            'colour.algebra.extend_line_segment',
        ],
        [
            'colour.intersect_line_segments',
            'colour.algebra.intersect_line_segments',
        ],
        [
            'colour.lagrange_coefficients_ASTME202211',
            'colour.colorimetry.lagrange_coefficients_ASTME202211',
        ],
        [
            'colour.sd_to_XYZ_tristimulus_weighting_factors_ASTME30815',
            'colour.colorimetry.'
            'sd_to_XYZ_tristimulus_weighting_factors_ASTME30815',
        ],
        [
            'colour.tristimulus_weighting_factors_ASTME202211',
            'colour.colorimetry.tristimulus_weighting_factors_ASTME202211'
        ],
    ]
}
"""
Defines *colour* package API changes.

API_CHANGES : dict
"""

API_CHANGES.update({
    'Removed': [
        'colour.DEFAULT_WAVELENGTH_DECIMALS',
        'colour.ArbitraryPrecisionMapping',
        'colour.SpectralMapping',
    ],
    'Renamed': [
        [
            'colour.eotf_ARIBSTDB67',
            'colour.models.oetf_reverse_ARIBSTDB67',
        ],
        [
            'colour.eotf_BT709',
            'colour.models.oetf_reverse_BT709',
        ],
        [
            'colour.oetf_BT1886',
            'colour.models.eotf_reverse_BT1886',
        ],
        [
            'colour.eotf_sRGB',
            'colour.models.oetf_reverse_sRGB',
        ],
        [
            'colour.ALEXA_WIDE_GAMUT_RGB_COLOURSPACE',
            'colour.models.ALEXA_WIDE_GAMUT_COLOURSPACE',
        ],
        [
            'colour.NTSC_RGB_COLOURSPACE',
            'colour.models.NTSC_COLOURSPACE',
        ],
        [
            'colour.PAL_SECAM_RGB_COLOURSPACE',
            'colour.models.PAL_SECAM_COLOURSPACE',
        ],
        [
            'colour.REC_709_COLOURSPACE',
            'colour.models.BT709_COLOURSPACE',
        ],
        [
            'colour.REC_2020_COLOURSPACE',
            'colour.models.BT2020_COLOURSPACE',
        ],
        [
            'colour.SMPTE_C_RGB_COLOURSPACE',
            'colour.models.SMPTE_240M_COLOURSPACE',
        ],
        [
            'colour.TriSpectralPowerDistribution',
            'colour.MultiSpectralDistribution',
        ],
    ]
})

# v0.3.12
API_CHANGES['Renamed'] = API_CHANGES['Renamed'] + [
    [
        'colour.CIE_standard_illuminant_A_function',
        'colour.sd_CIE_standard_illuminant_A',
    ],
    [
        'colour.COLOURCHECKERS_SPDS',
        'colour.COLOURCHECKERS_SDS',
    ],
    [
        'colour.D_illuminant_relative_spd',
        'colour.sd_CIE_illuminant_D_series',
    ],
    [
        'colour.ILLUMINANTS_RELATIVE_SPDS',
        'colour.ILLUMINANTS_SDS',
    ],
    [
        'colour.LIGHT_SOURCES_RELATIVE_SPDS',
        'colour.LIGHT_SOURCES_SDS',
    ],
    [
        'colour.MultiSpectralPowerDistribution',
        'colour.MultiSpectralDistribution',
    ],
    [
        'colour.REFLECTANCE_RECOVERY_METHODS',
        'colour.XYZ_TO_SD_METHODS',
    ],
    [
        'colour.SPECTRAL_TO_XYZ_METHODS',
        'colour.SD_TO_XYZ_METHODS',
    ],
    [
        'colour.SpectralPowerDistribution',
        'colour.SpectralDistribution',
    ],
    [
        'colour.blackbody_spd',
        'colour.sd_blackbody',
    ],
    [
        'colour.constant_spd',
        'colour.sd_constant',
    ],
    [
        'colour.first_order_colour_fit',
        'colour.colour_correction_matrix',
    ],
    [
        'colour.IES_TM2714_Spd',
        'colour.SpectralDistribution_IESTM2714',
    ],
    [
        'colour.function_gamma',
        'colour.gamma_function',
    ],
    [
        'colour.function_linear',
        'colour.linear_function',
    ],
    [
        'colour.mesopic_luminous_efficiency_function',
        'colour.sd_mesopic_luminous_efficiency_function',
    ],
    [
        'colour.ones_spd',
        'colour.sd_ones',
    ],
    [
        'colour.rayleigh_scattering_spd',
        'colour.sd_rayleigh_scattering',
    ],
    [
        'colour.read_spds_from_csv_file',
        'colour.read_sds_from_csv_file',
    ],
    [
        'colour.read_spds_from_xrite_file',
        'colour.read_sds_from_xrite_file',
    ],
    [
        'colour.spectral_to_aces_relative_exposure_values',
        'colour.sd_to_aces_relative_exposure_values',
    ],
    [
        'colour.spectral_to_XYZ',
        'colour.sd_to_XYZ',
    ],
    [
        'colour.write_spds_to_csv_file',
        'colour.write_sds_to_csv_file',
    ],
    [
        'colour.XYZ_to_spectral',
        'colour.XYZ_to_sd',
    ],
    [
        'colour.zeros_spd',
        'colour.sd_zeros',
    ],
]


def _setup_api_changes():
    """
    Setups *Colour* API changes.
    """

    global API_CHANGES

    for access_change in API_CHANGES['Future Access Change']:
        old_access, new_access = access_change
        API_CHANGES[old_access.split('.')[-1]] = (
            FutureAccessChange(  # noqa
                old_access, new_access))
    API_CHANGES.pop('Future Access Change')

    for access_remove in API_CHANGES['Future Access Remove']:
        name, access = access_remove
        API_CHANGES[name.split('.')[-1]] = (
            FutureAccessRemove(  # noqa
                name, access))
    API_CHANGES.pop('Future Access Remove')

    for removed in API_CHANGES['Removed']:
        API_CHANGES[removed.split('.')[-1]] = Removed(removed)  # noqa
    API_CHANGES.pop('Removed')

    for renamed in API_CHANGES['Renamed']:
        name, access = renamed
        API_CHANGES[name.split('.')[-1]] = Renamed(name, access)  # noqa
    API_CHANGES.pop('Renamed')


if not is_documentation_building():
    _setup_api_changes()

    del FutureAccessChange
    del FutureAccessRemove
    del ModuleAPI
    del Removed
    del Renamed
    del is_documentation_building
    del _setup_api_changes

    sys.modules['colour'] = colour(sys.modules['colour'], API_CHANGES)

    del sys
