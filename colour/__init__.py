# -*- coding: utf-8 -*-
"""
Colour
======

`Colour <https://github.com/colour-science/colour>`__ is an open-source
`Python <https://www.python.org/>`__ package providing a comprehensive number
of algorithms and datasets for colour science.

It is freely available under the
`New BSD License <https://opensource.org/licenses/BSD-3-Clause>`__ terms.

`Colour <https://github.com/colour-science/colour>`__ is an affiliated project
of `NumFOCUS <https://numfocus.org/>`__, a 501(c)(3) nonprofit in the United
States.

Sub-packages
------------
-   adaptation: Chromatic adaptation models and transformations.
-   algebra: Algebra utilities.
-   appearance: Colour appearance models.
-   biochemistry: Biochemistry computations.
-   blindness: Colour vision deficiency models.
-   characterisation: Colour correction, camera and display characterisation.
-   colorimetry: Core objects for colour computations.
-   constants: *CIE* and *CODATA* constants.
-   continuous: Base objects for continuous data representation.
-   contrast: Objects for contrast sensitivity computation.
-   corresponding: Corresponding colour chromaticities computations.
-   difference: Colour difference computations.
-   examples: Examples for the sub-packages.
-   geometry: Geometry primitives generation.
-   graph: Graph for automatic colour conversions.
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

from .utilities.deprecation import ModuleAPI, build_API_changes
from .utilities.documentation import is_documentation_building
from .utilities.common import (domain_range_scale, get_domain_range_scale,
                               set_domain_range_scale)

from .adaptation import (CHROMATIC_ADAPTATION_METHODS,
                         CHROMATIC_ADAPTATION_TRANSFORMS,
                         VIEWING_CONDITIONS_CMCCAT2000, chromatic_adaptation)
from .algebra import (CubicSplineInterpolator, Extrapolator,
                      KernelInterpolator, NearestNeighbourInterpolator,
                      LinearInterpolator, NullInterpolator, PchipInterpolator,
                      SpragueInterpolator, TABLE_INTERPOLATION_METHODS,
                      kernel_cardinal_spline, kernel_lanczos, kernel_linear,
                      kernel_nearest_neighbour, kernel_sinc,
                      table_interpolation, lagrange_coefficients)
from .colorimetry import (
    BANDPASS_CORRECTION_METHODS, CCS_ILLUMINANTS, CCS_LIGHT_SOURCES,
    LIGHTNESS_METHODS, LUMINANCE_METHODS, MSDS_CMFS, MSDS_TO_XYZ_METHODS,
    MultiSpectralDistributions, SDS_ILLUMINANTS, SDS_LEFS, SDS_LIGHT_SOURCES,
    SD_GAUSSIAN_METHODS, SD_MULTI_LEDS_METHODS, SD_SINGLE_LED_METHODS,
    SD_TO_XYZ_METHODS, SPECTRAL_SHAPE_ASTME308, SPECTRAL_SHAPE_DEFAULT,
    SpectralDistribution, SpectralShape, TVS_ILLUMINANTS_HUNTERLAB,
    WHITENESS_METHODS, YELLOWNESS_METHODS, bandpass_correction,
    colorimetric_purity, complementary_wavelength, dominant_wavelength,
    excitation_purity, lightness, luminance, luminous_efficacy,
    luminous_efficiency, luminous_flux, msds_constant, msds_ones, msds_zeros,
    msds_to_XYZ, sd_CIE_illuminant_D_series, sd_CIE_standard_illuminant_A,
    sd_blackbody, sd_constant, sd_gaussian,
    sd_mesopic_luminous_efficiency_function, sd_multi_leds, sd_ones,
    sd_single_led, sd_to_XYZ, sd_zeros, wavelength_to_XYZ, whiteness,
    yellowness)
from .blindness import (
    CVD_MATRICES_MACHADO2010, matrix_anomalous_trichromacy_Machado2009,
    matrix_cvd_Machado2009, msds_cmfs_anomalous_trichromacy_Machado2009)
from .appearance import (
    CAM_Specification_ATD95, CAM_Specification_CAM16,
    CAM_Specification_CIECAM02, CAM_Specification_Hunt, CAM_Specification_LLAB,
    CAM_Specification_Nayatani95, CAM_Specification_RLAB, CAM16_to_XYZ,
    CIECAM02_to_XYZ, VIEWING_CONDITIONS_CAM16, VIEWING_CONDITIONS_CIECAM02,
    VIEWING_CONDITIONS_HUNT, VIEWING_CONDITIONS_LLAB, VIEWING_CONDITIONS_RLAB,
    XYZ_to_ATD95, XYZ_to_CAM16, XYZ_to_CIECAM02, XYZ_to_Hunt, XYZ_to_LLAB,
    XYZ_to_Nayatani95, XYZ_to_RLAB)
from .difference import DELTA_E_METHODS, delta_E
from .geometry import (PRIMITIVE_METHODS, primitive,
                       PRIMITIVE_VERTICES_METHODS, primitive_vertices)
from .io import (LUT1D, LUT3x1D, LUT3D, LUTSequence, READ_IMAGE_METHODS,
                 SpectralDistribution_IESTM2714, WRITE_IMAGE_METHODS,
                 read_image, read_LUT, read_sds_from_csv_file,
                 read_sds_from_xrite_file, read_spectral_data_from_csv_file,
                 write_image, write_LUT, write_sds_to_csv_file)
from .models import (
    CAM02LCD_to_JMh_CIECAM02, CAM02SCD_to_JMh_CIECAM02,
    CAM02UCS_to_JMh_CIECAM02, CAM16LCD_to_JMh_CAM16, CAM16SCD_to_JMh_CAM16,
    CAM16UCS_to_JMh_CAM16, CCTF_DECODINGS, CCTF_ENCODINGS, CMYK_to_CMY,
    CMY_to_CMYK, CMY_to_RGB, CV_range, DATA_MACADAM_1942_ELLIPSES,
    DIN99_to_Lab, EOTFS, EOTF_INVERSES, HDR_CIELAB_METHODS, HDR_IPT_METHODS,
    HSL_to_RGB, HSV_to_RGB, Hunter_Lab_to_XYZ, Hunter_Rdab_to_XYZ,
    ICTCP_to_RGB, IGPGTG_to_XYZ, IPT_hue_angle, IPT_to_XYZ,
    JMh_CAM16_to_CAM16LCD, JMh_CAM16_to_CAM16SCD, JMh_CAM16_to_CAM16UCS,
    JMh_CIECAM02_to_CAM02LCD, JMh_CIECAM02_to_CAM02SCD,
    JMh_CIECAM02_to_CAM02UCS, JzAzBz_to_XYZ, LCHab_to_Lab, LCHuv_to_Luv,
    LOG_DECODINGS, LOG_ENCODINGS, Lab_to_DIN99, Lab_to_LCHab, Lab_to_XYZ,
    Luv_to_LCHuv, Luv_to_XYZ, Luv_to_uv, Luv_uv_to_xy, OETFS, OETF_INVERSES,
    OOTFS, OOTF_INVERSES, OSA_UCS_to_XYZ, Prismatic_to_RGB, RGB_COLOURSPACES,
    RGB_Colourspace, RGB_luminance, RGB_luminance_equation, RGB_to_CMY,
    RGB_to_HSL, RGB_to_HSV, RGB_to_ICTCP, RGB_to_Prismatic, RGB_to_RGB,
    RGB_to_XYZ, RGB_to_YCbCr, RGB_to_YCoCg, RGB_to_YcCbcCrc, UCS_to_XYZ,
    UCS_to_uv, UCS_uv_to_xy, UVW_to_XYZ, WEIGHTS_YCBCR, XYZ_to_Hunter_Lab,
    XYZ_to_Hunter_Rdab, XYZ_to_IGPGTG, XYZ_to_IPT, XYZ_to_JzAzBz,
    XYZ_to_K_ab_HunterLab1966, XYZ_to_Lab, XYZ_to_Luv, XYZ_to_OSA_UCS,
    XYZ_to_RGB, XYZ_to_UCS, XYZ_to_UVW, XYZ_to_hdr_CIELab, XYZ_to_hdr_IPT,
    XYZ_to_sRGB, XYZ_to_xy, XYZ_to_xyY, YCbCr_to_RGB, YCoCg_to_RGB,
    YcCbcCrc_to_RGB, cctf_decoding, cctf_encoding,
    chromatically_adapted_primaries, eotf, eotf_inverse, full_to_legal,
    gamma_function, hdr_CIELab_to_XYZ, hdr_IPT_to_XYZ, legal_to_full,
    linear_function, log_decoding, log_encoding, matrix_RGB_to_RGB,
    normalised_primary_matrix, oetf, oetf_inverse, ootf, ootf_inverse,
    primaries_whitepoint, sRGB_to_XYZ, uv_to_Luv, uv_to_UCS, xyY_to_XYZ,
    xyY_to_xy, xy_to_Luv_uv, xy_to_UCS_uv, xy_to_XYZ, xy_to_xyY)
from .corresponding import (
    BRENEMAN_EXPERIMENTS, BRENEMAN_EXPERIMENT_PRIMARIES_CHROMATICITIES,
    CORRESPONDING_CHROMATICITIES_PREDICTION_MODELS, CorrespondingColourDataset,
    CorrespondingChromaticitiesPrediction,
    corresponding_chromaticities_prediction)
from .contrast import (CONTRAST_SENSITIVITY_METHODS,
                       contrast_sensitivity_function)
from .phenomena import (rayleigh_scattering, scattering_cross_section,
                        sd_rayleigh_scattering)
from .notation import (MUNSELL_COLOURS, MUNSELL_VALUE_METHODS,
                       munsell_colour_to_xyY, munsell_value,
                       xyY_to_munsell_colour)
from .quality import (COLOUR_FIDELITY_INDEX_METHODS,
                      COLOUR_QUALITY_SCALE_METHODS, colour_fidelity_index,
                      colour_quality_scale, colour_rendering_index,
                      spectral_similarity_index)
from .recovery import XYZ_TO_SD_METHODS, XYZ_to_sd
from .temperature import (CCT_TO_UV_METHODS, CCT_TO_XY_METHODS, CCT_to_uv,
                          CCT_to_xy, UV_TO_CCT_METHODS, XY_TO_CCT_METHODS,
                          uv_to_CCT, xy_to_CCT)
from .characterisation import (
    CCS_COLOURCHECKERS, MATRIX_COLOUR_CORRECTION_METHODS,
    COLOUR_CORRECTION_METHODS, MSDS_CAMERA_SENSITIVITIES,
    MSDS_DISPLAY_PRIMARIES, POLYNOMIAL_EXPANSION_METHODS, SDS_COLOURCHECKERS,
    SDS_FILTERS, SDS_LENSES, colour_correction, matrix_colour_correction,
    matrix_idt, polynomial_expansion, sd_to_aces_relative_exposure_values)
from .volume import (
    OPTIMAL_COLOUR_STIMULI_ILLUMINANTS, RGB_colourspace_limits,
    RGB_colourspace_pointer_gamut_coverage_MonteCarlo,
    RGB_colourspace_visible_spectrum_coverage_MonteCarlo,
    RGB_colourspace_volume_MonteCarlo,
    RGB_colourspace_volume_coverage_MonteCarlo, is_within_macadam_limits,
    is_within_mesh_volume, is_within_pointer_gamut, is_within_visible_spectrum)
from .graph import describe_conversion_path, convert

from colour.utilities import is_matplotlib_installed

# Exposing "colour.plotting" sub-package if "Matplotlib" is available.
if is_matplotlib_installed():
    import colour.plotting as plotting  # noqa
else:

    class MockPlotting(object):
        """
        Mock object for :mod:`colour.plotting` sub-package raising an exception
        if the sub-package is accessed but *Matplotlib* is not installed.
        """

        def __getattr__(self, attribute):
            is_matplotlib_installed(raise_exception=True)

    globals()['plotting'] = MockPlotting()

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'domain_range_scale', 'get_domain_range_scale', 'set_domain_range_scale'
]
__all__ += [
    'CHROMATIC_ADAPTATION_METHODS', 'CHROMATIC_ADAPTATION_TRANSFORMS',
    'VIEWING_CONDITIONS_CMCCAT2000', 'chromatic_adaptation'
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
    'BANDPASS_CORRECTION_METHODS', 'CCS_ILLUMINANTS', 'CCS_LIGHT_SOURCES',
    'LIGHTNESS_METHODS', 'LUMINANCE_METHODS', 'MSDS_CMFS',
    'MSDS_TO_XYZ_METHODS', 'MultiSpectralDistributions', 'SDS_ILLUMINANTS',
    'SDS_LEFS', 'SDS_LIGHT_SOURCES', 'SD_GAUSSIAN_METHODS',
    'SD_MULTI_LEDS_METHODS', 'SD_SINGLE_LED_METHODS', 'SD_TO_XYZ_METHODS',
    'SPECTRAL_SHAPE_ASTME308', 'SPECTRAL_SHAPE_DEFAULT',
    'SpectralDistribution', 'SpectralShape', 'TVS_ILLUMINANTS_HUNTERLAB',
    'WHITENESS_METHODS', 'YELLOWNESS_METHODS', 'bandpass_correction',
    'colorimetric_purity', 'complementary_wavelength', 'dominant_wavelength',
    'excitation_purity', 'lightness', 'luminance', 'luminous_efficacy',
    'luminous_efficiency', 'luminous_flux', 'msds_constant', 'msds_ones',
    'msds_zeros', 'msds_to_XYZ', 'sd_CIE_illuminant_D_series',
    'sd_CIE_standard_illuminant_A', 'sd_blackbody', 'sd_constant',
    'sd_gaussian', 'sd_mesopic_luminous_efficiency_function', 'sd_multi_leds',
    'sd_ones', 'sd_single_led', 'sd_to_XYZ', 'sd_zeros', 'wavelength_to_XYZ',
    'whiteness', 'yellowness'
]
__all__ += [
    'CVD_MATRICES_MACHADO2010', 'matrix_anomalous_trichromacy_Machado2009',
    'matrix_cvd_Machado2009', 'msds_cmfs_anomalous_trichromacy_Machado2009'
]
__all__ += [
    'CAM_Specification_ATD95', 'CAM_Specification_CAM16',
    'CAM_Specification_CIECAM02', 'CAM_Specification_Hunt',
    'CAM_Specification_LLAB', 'CAM_Specification_Nayatani95',
    'CAM_Specification_RLAB', 'CAM16_to_XYZ', 'CIECAM02_to_XYZ',
    'VIEWING_CONDITIONS_CAM16', 'VIEWING_CONDITIONS_CIECAM02',
    'VIEWING_CONDITIONS_HUNT', 'VIEWING_CONDITIONS_LLAB',
    'VIEWING_CONDITIONS_RLAB', 'XYZ_to_ATD95', 'XYZ_to_CAM16',
    'XYZ_to_CIECAM02', 'XYZ_to_Hunt', 'XYZ_to_LLAB', 'XYZ_to_Nayatani95',
    'XYZ_to_RLAB'
]
__all__ += ['DELTA_E_METHODS', 'delta_E']
__all__ += [
    'PRIMITIVE_METHODS', 'primitive', 'PRIMITIVE_VERTICES_METHODS',
    'primitive_vertices'
]
__all__ += [
    'LUT1D', 'LUT3x1D', 'LUT3D', 'LUTSequence', 'READ_IMAGE_METHODS',
    'SpectralDistribution_IESTM2714', 'WRITE_IMAGE_METHODS', 'read_image',
    'read_LUT', 'read_sds_from_csv_file', 'read_sds_from_xrite_file',
    'read_spectral_data_from_csv_file', 'write_image', 'write_LUT',
    'write_sds_to_csv_file'
]
__all__ += [
    'CAM02LCD_to_JMh_CIECAM02', 'CAM02SCD_to_JMh_CIECAM02',
    'CAM02UCS_to_JMh_CIECAM02', 'CAM16LCD_to_JMh_CAM16',
    'CAM16SCD_to_JMh_CAM16', 'CAM16UCS_to_JMh_CAM16', 'CCTF_DECODINGS',
    'CCTF_ENCODINGS', 'CMYK_to_CMY', 'CMY_to_CMYK', 'CMY_to_RGB', 'CV_range',
    'DATA_MACADAM_1942_ELLIPSES', 'DIN99_to_Lab', 'EOTFS', 'EOTF_INVERSES',
    'HDR_CIELAB_METHODS', 'HDR_IPT_METHODS', 'HSL_to_RGB', 'HSV_to_RGB',
    'Hunter_Lab_to_XYZ', 'Hunter_Rdab_to_XYZ', 'ICTCP_to_RGB', 'IGPGTG_to_XYZ',
    'IPT_hue_angle', 'IPT_to_XYZ', 'JMh_CAM16_to_CAM16LCD',
    'JMh_CAM16_to_CAM16SCD', 'JMh_CAM16_to_CAM16UCS',
    'JMh_CIECAM02_to_CAM02LCD', 'JMh_CIECAM02_to_CAM02SCD',
    'JMh_CIECAM02_to_CAM02UCS', 'JzAzBz_to_XYZ', 'LCHab_to_Lab',
    'LCHuv_to_Luv', 'LOG_DECODINGS', 'LOG_ENCODINGS', 'Lab_to_DIN99',
    'Lab_to_LCHab', 'Lab_to_XYZ', 'Luv_to_LCHuv', 'Luv_to_XYZ', 'Luv_to_uv',
    'Luv_uv_to_xy', 'OETFS', 'OETF_INVERSES', 'OOTFS', 'OOTF_INVERSES',
    'OSA_UCS_to_XYZ', 'Prismatic_to_RGB', 'RGB_COLOURSPACES',
    'RGB_Colourspace', 'RGB_luminance', 'RGB_luminance_equation', 'RGB_to_CMY',
    'RGB_to_HSL', 'RGB_to_HSV', 'RGB_to_ICTCP', 'RGB_to_Prismatic',
    'RGB_to_RGB', 'RGB_to_XYZ', 'RGB_to_YCbCr', 'RGB_to_YCoCg',
    'RGB_to_YcCbcCrc', 'UCS_to_XYZ', 'UCS_to_uv', 'UCS_uv_to_xy', 'UVW_to_XYZ',
    'WEIGHTS_YCBCR', 'XYZ_to_Hunter_Lab', 'XYZ_to_Hunter_Rdab',
    'XYZ_to_IGPGTG', 'XYZ_to_IPT', 'XYZ_to_JzAzBz',
    'XYZ_to_K_ab_HunterLab1966', 'XYZ_to_Lab', 'XYZ_to_Luv', 'XYZ_to_OSA_UCS',
    'XYZ_to_RGB', 'XYZ_to_UCS', 'XYZ_to_UVW', 'XYZ_to_hdr_CIELab',
    'XYZ_to_hdr_IPT', 'XYZ_to_sRGB', 'XYZ_to_xy', 'XYZ_to_xyY', 'YCbCr_to_RGB',
    'YCoCg_to_RGB', 'YcCbcCrc_to_RGB', 'cctf_decoding', 'cctf_encoding',
    'chromatically_adapted_primaries', 'eotf', 'eotf_inverse', 'full_to_legal',
    'gamma_function', 'hdr_CIELab_to_XYZ', 'hdr_IPT_to_XYZ', 'legal_to_full',
    'linear_function', 'log_decoding', 'log_encoding', 'matrix_RGB_to_RGB',
    'normalised_primary_matrix', 'oetf', 'oetf_inverse', 'ootf',
    'ootf_inverse', 'primaries_whitepoint', 'sRGB_to_XYZ', 'uv_to_Luv',
    'uv_to_UCS', 'xyY_to_XYZ', 'xyY_to_xy', 'xy_to_Luv_uv', 'xy_to_UCS_uv',
    'xy_to_XYZ', 'xy_to_xyY'
]
__all__ += [
    'BRENEMAN_EXPERIMENTS', 'BRENEMAN_EXPERIMENT_PRIMARIES_CHROMATICITIES',
    'CORRESPONDING_CHROMATICITIES_PREDICTION_MODELS',
    'CorrespondingColourDataset', 'CorrespondingChromaticitiesPrediction',
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
__all__ += [
    'COLOUR_FIDELITY_INDEX_METHODS', 'COLOUR_QUALITY_SCALE_METHODS',
    'colour_fidelity_index', 'colour_quality_scale', 'colour_rendering_index',
    'spectral_similarity_index'
]
__all__ += ['XYZ_TO_SD_METHODS', 'XYZ_to_sd']
__all__ += [
    'CCT_TO_UV_METHODS', 'CCT_TO_XY_METHODS', 'CCT_to_uv', 'CCT_to_xy',
    'UV_TO_CCT_METHODS', 'XY_TO_CCT_METHODS', 'uv_to_CCT', 'xy_to_CCT'
]
__all__ += [
    'CCS_COLOURCHECKERS', 'MATRIX_COLOUR_CORRECTION_METHODS',
    'COLOUR_CORRECTION_METHODS', 'MSDS_CAMERA_SENSITIVITIES',
    'MSDS_DISPLAY_PRIMARIES', 'POLYNOMIAL_EXPANSION_METHODS',
    'SDS_COLOURCHECKERS', 'SDS_FILTERS', 'SDS_LENSES', 'colour_correction',
    'matrix_colour_correction', 'matrix_idt', 'polynomial_expansion',
    'sd_to_aces_relative_exposure_values'
]
__all__ += [
    'OPTIMAL_COLOUR_STIMULI_ILLUMINANTS', 'RGB_colourspace_limits',
    'RGB_colourspace_pointer_gamut_coverage_MonteCarlo',
    'RGB_colourspace_visible_spectrum_coverage_MonteCarlo',
    'RGB_colourspace_volume_MonteCarlo',
    'RGB_colourspace_volume_coverage_MonteCarlo', 'is_within_macadam_limits',
    'is_within_mesh_volume', 'is_within_pointer_gamut',
    'is_within_visible_spectrum'
]
__all__ += ['describe_conversion_path', 'convert']

__application_name__ = 'Colour'

__major_version__ = '0'
__minor_version__ = '3'
__change_version__ = '16'
__version__ = '.'.join(
    (__major_version__,
     __minor_version__,
     __change_version__))  # yapf: disable

# TODO: Remove legacy printing support when deemed appropriate.
try:
    np.set_printoptions(legacy='1.13')
except TypeError:  # pragma: no cover
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
    'ObjectFutureAccessChange': [
        [
            'colour.ACES_2065_1_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_ACES2065_1',
        ],
        [
            'colour.ACES_CCT_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_ACESCCT',
        ],
        [
            'colour.ACES_CC_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_ACESCC',
        ],
        [
            'colour.ACES_CG_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_ACESCG',
        ],
        [
            'colour.ACES_PROXY_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_ACESPROXY',
        ],
        [
            'colour.ACES_RICD',
            'colour.models.MSDS_ACES_RICD',
        ],
        [
            'colour.ADOBE_RGB1998_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_ADOBE_RGB1998',
        ],
        [
            'colour.ADOBE_WIDE_GAMUT_RGB_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_ADOBE_WIDE_GAMUT_RGB',
        ],
        [
            'colour.ALEXA_WIDE_GAMUT_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_ALEXA_WIDE_GAMUT',
        ],
        [
            'colour.APPLE_RGB_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_APPLE_RGB',
        ],
        [
            'colour.AVOGADRO_CONSTANT',
            'colour.constants.CONSTANT_AVOGADRO',
        ],
        [
            'colour.AbstractContinuousFunction',
            'colour.continuous.AbstractContinuousFunction',
        ],
        [
            'colour.BEST_RGB_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_BEST_RGB',
        ],
        [
            'colour.BETA_RGB_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_BETA_RGB',
        ],
        [
            'colour.BOLTZMANN_CONSTANT',
            'colour.constants.CONSTANT_BOLTZMANN',
        ],
        [
            'colour.BRADFORD_CAT',
            'colour.adaptation.CAT_BRADFORD',
        ],
        [
            'colour.BS_CAT',
            'colour.adaptation.CAT_BIANCO2010',
        ],
        [
            'colour.BS_PC_CAT',
            'colour.adaptation.CAT_PC_BIANCO2010',
        ],
        [
            'colour.BT2020_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_BT2020',
        ],
        [
            'colour.BT470_525_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_BT470_525',
        ],
        [
            'colour.BT470_625_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_BT470_625',
        ],
        [
            'colour.BT709_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_BT709',
        ],
        [
            'colour.CAM16_InductionFactors',
            'colour.appearance.InductionFactors_CAM16',
        ],
        [
            'colour.CAT02_BRILL_CAT',
            'colour.adaptation.CAT_CAT02_BRILL2008',
        ],
        [
            'colour.CAT02_CAT',
            'colour.adaptation.CAT_CAT02',
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
            'colour.CIECAM02_InductionFactors',
            'colour.appearance.InductionFactors_CIECAM02',
        ],
        [
            'colour.CIE_RGB_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_CIE_RGB',
        ],
        [
            'colour.CINEMA_GAMUT_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_CINEMA_GAMUT',
        ],
        [
            'colour.CMCCAT2000_CAT',
            'colour.adaptation.CAT_CMCCAT2000',
        ],
        [
            'colour.InductionFactors_CMCCAT2000',
            'colour.adaptation.InductionFactors_CMCCAT2000',
        ],
        [
            'colour.CMCCAT97_CAT',
            'colour.adaptation.CAT_CMCCAT97',
        ],
        [
            'colour.COLOR_MATCH_RGB_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_COLOR_MATCH_RGB',
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
            'colour.models.COLOURSPACE_MODELS_AXIS_LABELS',
        ],
        [
            'colour.CQS_Specification',
            'colour.quality.ColourRendering_Specification_CQS',
        ],
        [
            'colour.CRI_Specification',
            'colour.quality.ColourRendering_Specification_CRI',
        ],
        [
            'colour.CaseInsensitiveMapping',
            'colour.utilities.CaseInsensitiveMapping',
        ],
        [
            'colour.ColourWarning',
            'colour.utilities.ColourWarning',
        ],
        [
            'colour.DCI_P3_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_DCI_P3',
        ],
        [
            'colour.DCI_P3_P_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_DCI_P3_P',
        ],
        [
            'colour.DEFAULT_FLOAT_DTYPE',
            'colour.constants.DEFAULT_FLOAT_DTYPE',
        ],
        [
            'colour.DON_RGB_4_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_DON_RGB_4',
        ],
        [
            'colour.DRAGON_COLOR_2_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_DRAGON_COLOR_2',
        ],
        [
            'colour.DRAGON_COLOR_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_DRAGON_COLOR',
        ],
        [
            'colour.D_ILLUMINANTS_S_SPDS',
            'colour.colorimetry.SDS_BASIS_FUNCTIONS_CIE_ILLUMINANT_D_SERIES',
        ],
        [
            'colour.ECI_RGB_V2_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_ECI_RGB_V2',
        ],
        [
            'colour.EKTA_SPACE_PS_5_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_EKTA_SPACE_PS_5',
        ],
        [
            'colour.EPSILON',
            'colour.constants.EPSILON',
        ],
        [
            'colour.ERIMM_RGB_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_ERIMM_RGB',
        ],
        [
            'colour.FAIRCHILD_CAT',
            'colour.adaptation.CAT_FAIRCHILD',
        ],
        [
            'colour.FLOATING_POINT_NUMBER_PATTERN',
            'colour.constants.FLOATING_POINT_NUMBER_PATTERN',
        ],
        [
            'colour.Hunt_InductionFactors',
            'colour.appearance.InductionFactors_Hunt',
        ],
        [
            'colour.INTEGER_THRESHOLD',
            'colour.constants.INTEGER_THRESHOLD',
        ],
        [
            'colour.KP_M',
            'colour.constants.CONSTANT_KP_M',
        ],
        [
            'colour.K_M',
            'colour.constants.CONSTANT_K_M',
        ],
        [
            'colour.LIGHT_SPEED',
            'colour.constants.CONSTANT_LIGHT_SPEED',
        ],
        [
            'colour.LLAB_InductionFactors',
            'colour.appearance.InductionFactors_LLAB',
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
            'colour.LineSegmentsIntersections_Specification',
            'colour.algebra.LineSegmentsIntersections_Specification',
        ],
        [
            'colour.Lookup',
            'colour.utilities.Lookup',
        ],
        [
            'colour.MAX_RGB_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_MAX_RGB',
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
            'colour.MultiSignal',
            'colour.continuous.MultiSignals',
        ],
        [
            'colour.NTSC1953_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_NTSC1953',
        ],
        [
            'colour.PAL_SECAM_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_PAL_SECAM',
        ],
        [
            'colour.PLANCK_CONSTANT',
            'colour.constants.CONSTANT_PLANCK',
        ],
        [
            'colour.PROPHOTO_RGB_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_PROPHOTO_RGB',
        ],
        [
            'colour.PROTUNE_NATIVE_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_PROTUNE_NATIVE',
        ],
        [
            'colour.RED_COLOR_2_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_RED_COLOR_2',
        ],
        [
            'colour.RED_COLOR_3_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_RED_COLOR_3',
        ],
        [
            'colour.RED_COLOR_4_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_RED_COLOR_4',
        ],
        [
            'colour.RED_COLOR_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_RED_COLOR',
        ],
        [
            'colour.RED_WIDE_GAMUT_RGB_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_RED_WIDE_GAMUT_RGB',
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
            'colour.characterisation.RGB_CameraSensitivities',
        ],
        [
            'colour.RGB_to_sd_Smits1999',
            'colour.recovery.RGB_to_sd_Smits1999',
        ],
        [
            'colour.RIMM_RGB_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_RIMM_RGB',
        ],
        [
            'colour.ROMM_RGB_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_ROMM_RGB',
        ],
        [
            'colour.RUSSELL_RGB_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_RUSSELL_RGB',
        ],
        [
            'colour.SHARP_CAT',
            'colour.adaptation.CAT_SHARP',
        ],
        [
            'colour.SMITS_1999_SPDS',
            'colour.recovery.SDS_SMITS1999',
        ],
        [
            'colour.SMPTE_240M_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_SMPTE_240M',
        ],
        [
            'colour.S_GAMUT3_CINE_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_S_GAMUT3_CINE',
        ],
        [
            'colour.S_GAMUT3_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_S_GAMUT3',
        ],
        [
            'colour.S_GAMUT_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_S_GAMUT',
        ],
        [
            'colour.Signal',
            'colour.continuous.Signal',
        ],
        [
            'colour.Structure',
            'colour.utilities.Structure',
        ],
        [
            'colour.TCS_SPDS',
            'colour.quality.SDS_TCS',
        ],
        [
            'colour.VON_KRIES_CAT',
            'colour.adaptation.CAT_VON_KRIES',
        ],
        [
            'colour.VS_SPDS',
            'colour.quality.SDS_VS',
        ],
        [
            'colour.V_GAMUT_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_V_GAMUT',
        ],
        [
            'colour.XTREME_RGB_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_XTREME_RGB',
        ],
        [
            'colour.XYZ_ColourMatchingFunctions',
            'colour.colorimetry.XYZ_ColourMatchingFunctions',
        ],
        [
            'colour.XYZ_SCALING_CAT',
            'colour.adaptation.CAT_XYZ_SCALING',
        ],
        [
            'colour.XYZ_to_colourspace_model',
            'colour.models.XYZ_to_colourspace_model',
        ],
        [
            'colour.XYZ_to_sd_Meng2015',
            'colour.recovery.XYZ_to_sd_Meng2015',
        ],
        [
            'colour.adjust_tristimulus_weighting_factors_ASTME30815',
            'colour.colorimetry.adjust_tristimulus_weighting_factors_ASTME308',
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
            'colour.bandpass_correction_Stearns1988',
            'colour.colorimetry.bandpass_correction_Stearns1988',
        ],
        [
            'colour.batch',
            'colour.utilities.batch',
        ],
        [
            'colour.blackbody_spectral_radiance',
            'colour.colorimetry.blackbody_spectral_radiance',
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
            'colour.chromatic_adaptation_VonKries',
            'colour.adaptation.chromatic_adaptation_VonKries',
        ],
        [
            'colour.chromatic_adaptation_forward_CMCCAT2000',
            'colour.adaptation.chromatic_adaptation_forward_CMCCAT2000',
        ],
        [
            'colour.matrix_chromatic_adaptation_VonKries',
            'colour.adaptation.matrix_chromatic_adaptation_VonKries',
        ],
        [
            'colour.chromatic_adaptation_reverse_CMCCAT2000',
            'colour.adaptation.chromatic_adaptation_inverse_CMCCAT2000',
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
            'colour.corresponding_chromaticities_prediction_CIE1994',
            'colour.corresponding.corresponding_chromaticities_prediction_CIE1994',  # noqa
        ],
        [
            'colour.corresponding_chromaticities_prediction_CMCCAT2000',
            'colour.corresponding.corresponding_chromaticities_prediction_CMCCAT2000',  # noqa
        ],
        [
            'colour.corresponding_chromaticities_prediction_Fairchild1990',
            'colour.corresponding.corresponding_chromaticities_prediction_Fairchild1990',  # noqa
        ],
        [
            'colour.corresponding_chromaticities_prediction_VonKries',
            'colour.corresponding.corresponding_chromaticities_prediction_VonKries',  # noqa
        ],
        [
            'colour.cylindrical_to_cartesian',
            'colour.algebra.cylindrical_to_cartesian',
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
            'colour.dot_matrix',
            'colour.utilities.matrix_dot',
        ],
        [
            'colour.dot_vector',
            'colour.utilities.vector_dot',
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
            'colour.models.eotf_HLG_BT2100',
        ],
        [
            'colour.eotf_BT2100_PQ',
            'colour.models.eotf_PQ_BT2100',
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
            'colour.cctf_decoding_ProPhotoRGB',
            'colour.models.cctf_decoding_ProPhotoRGB',
        ],
        [
            'colour.cctf_decoding_RIMMRGB',
            'colour.models.cctf_decoding_RIMMRGB',
        ],
        [
            'colour.cctf_decoding_ROMMRGB',
            'colour.models.cctf_decoding_ROMMRGB',
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
            'colour.eotf_reverse_BT1886',
            'colour.models.eotf_inverse_BT1886',
        ],
        [
            'colour.eotf_reverse_BT2100_HLG',
            'colour.models.eotf_inverse_HLG_BT2100',
        ],
        [
            'colour.eotf_reverse_BT2100_PQ',
            'colour.models.eotf_inverse_PQ_BT2100',
        ],
        [
            'colour.eotf_reverse_ST2084',
            'colour.models.eotf_inverse_ST2084',
        ],
        [
            'colour.eotf_reverse_sRGB',
            'colour.models.eotf_inverse_sRGB',
        ],
        [
            'colour.eotf_sRGB',
            'colour.models.eotf_sRGB',
        ],
        [
            'colour.euclidean_distance',
            'colour.algebra.euclidean_distance',
        ],
        [
            'colour.extend_line_segment',
            'colour.algebra.extend_line_segment',
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
            'colour.intersect_line_segments',
            'colour.algebra.intersect_line_segments',
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
            'colour.lagrange_coefficients_ASTME2022',
            'colour.colorimetry.lagrange_coefficients_ASTME2022',
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
            'colour.log_decoding_VLog',
            'colour.models.log_decoding_VLog',
        ],
        [
            'colour.log_decoding_ViperLog',
            'colour.models.log_decoding_ViperLog',
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
            'colour.luminance_ASTMD153508',
            'colour.colorimetry.luminance_ASTMD1535',
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
            'colour.mesopic_weighting_function',
            'colour.colorimetry.mesopic_weighting_function',
        ],
        [
            'colour.message_box',
            'colour.utilities.message_box',
        ],
        [
            'colour.munsell_value_ASTMD153508',
            'colour.notation.munsell_value_ASTMD1535',
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
            'colour.numpy_print_options',
            'colour.utilities.numpy_print_options',
        ],
        [
            'colour.oetf_ARIBSTDB67',
            'colour.models.oetf_ARIBSTDB67',
        ],
        [
            'colour.oetf_BT2020',
            'colour.models.eotf_inverse_BT2020',
        ],
        [
            'colour.oetf_BT2100_HLG',
            'colour.models.oetf_HLG_BT2100',
        ],
        [
            'colour.oetf_BT2100_PQ',
            'colour.models.oetf_PQ_BT2100',
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
            'colour.models.eotf_inverse_DCIP3',
        ],
        [
            'colour.oetf_DICOMGSDF',
            'colour.models.eotf_inverse_DICOMGSDF',
        ],
        [
            'colour.cctf_encoding_ProPhotoRGB',
            'colour.models.cctf_encoding_ProPhotoRGB',
        ],
        [
            'colour.cctf_encoding_RIMMRGB',
            'colour.models.cctf_encoding_RIMMRGB',
        ],
        [
            'colour.cctf_encoding_ROMMRGB',
            'colour.models.cctf_encoding_ROMMRGB',
        ],
        [
            'colour.oetf_SMPTE240M',
            'colour.models.oetf_SMPTE240M',
        ],
        [
            'colour.oetf_reverse_ARIBSTDB67',
            'colour.models.oetf_inverse_ARIBSTDB67',
        ],
        [
            'colour.oetf_reverse_BT2100_HLG',
            'colour.models.oetf_inverse_HLG_BT2100',
        ],
        [
            'colour.oetf_reverse_BT2100_PQ',
            'colour.models.oetf_inverse_PQ_BT2100',
        ],
        [
            'colour.oetf_reverse_BT601',
            'colour.models.oetf_inverse_BT601',
        ],
        [
            'colour.oetf_reverse_BT709',
            'colour.models.oetf_inverse_BT709',
        ],
        [
            'colour.ootf_BT2100_HLG',
            'colour.models.ootf_HLG_BT2100',
        ],
        [
            'colour.ootf_BT2100_PQ',
            'colour.models.ootf_PQ_BT2100',
        ],
        [
            'colour.ootf_reverse_BT2100_HLG',
            'colour.models.ootf_inverse_HLG_BT2100',
        ],
        [
            'colour.ootf_reverse_BT2100_PQ',
            'colour.models.ootf_inverse_PQ_BT2100',
        ],
        [
            'colour.orient',
            'colour.utilities.orient',
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
            'colour.row_as_diagonal',
            'colour.utilities.row_as_diagonal',
        ],
        [
            'colour.sRGB_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_sRGB',
        ],
        [
            'colour.sd_to_XYZ_ASTME30815',
            'colour.colorimetry.sd_to_XYZ_ASTME308',
        ],
        [
            'colour.sd_to_XYZ_integration',
            'colour.colorimetry.sd_to_XYZ_integration',
        ],
        [
            'colour.sd_to_XYZ_tristimulus_weighting_factors_ASTME30815',
            'colour.colorimetry.sd_to_XYZ_tristimulus_weighting_factors_ASTME308',  # noqa
        ],
        [
            'colour.spherical_to_cartesian',
            'colour.algebra.spherical_to_cartesian',
        ],
        [
            'colour.substrate_concentration_MichealisMenten',
            'colour.biochemistry.substrate_concentration_MichealisMenten',
        ],
        [
            'colour.tristimulus_weighting_factors_ASTME2022',
            'colour.colorimetry.tristimulus_weighting_factors_ASTME2022',
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
            'colour.xy_to_CCT_Hernandez1999',
            'colour.temperature.xy_to_CCT_Hernandez1999',
        ],
        [
            'colour.xy_to_CCT_McCamy1992',
            'colour.temperature.xy_to_CCT_McCamy1992',
        ],
        [
            'colour.yellowness_ASTMD1925',
            'colour.colorimetry.yellowness_ASTMD1925',
        ],
        [
            'colour.yellowness_ASTME313',
            'colour.colorimetry.yellowness_ASTME313',
        ],
    ]
}
"""
Defines *colour* package API changes.

API_CHANGES : dict
"""

API_CHANGES.update({
    'ObjectRemoved': [
        'colour.DEFAULT_WAVELENGTH_DECIMALS',
        'colour.ArbitraryPrecisionMapping',
        'colour.SpectralMapping',
    ],
    'ObjectRenamed': [
        [
            'colour.eotf_ARIBSTDB67',
            'colour.models.oetf_inverse_ARIBSTDB67',
        ],
        [
            'colour.eotf_BT709',
            'colour.models.oetf_inverse_BT709',
        ],
        [
            'colour.oetf_BT1886',
            'colour.models.eotf_inverse_BT1886',
        ],
        [
            'colour.eotf_sRGB',
            'colour.models.eotf_sRGB',
        ],
        [
            'colour.ALEXA_WIDE_GAMUT_RGB_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_ALEXA_WIDE_GAMUT',
        ],
        [
            'colour.NTSC_1953_RGB_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_NTSC1953',
        ],
        [
            'colour.PAL_SECAM_RGB_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_PAL_SECAM',
        ],
        [
            'colour.REC_709_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_BT709',
        ],
        [
            'colour.REC_2020_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_BT2020',
        ],
        [
            'colour.SMPTE_C_RGB_COLOURSPACE',
            'colour.models.RGB_COLOURSPACE_SMPTE_240M',
        ],
        [
            'colour.TriSpectralPowerDistribution',
            'colour.MultiSpectralDistributions',
        ],
    ]
})

# v0.3.12
API_CHANGES['ObjectRenamed'] = API_CHANGES['ObjectRenamed'] + [
    [
        'colour.CIE_standard_illuminant_A_function',
        'colour.sd_CIE_standard_illuminant_A',
    ],
    [
        'colour.COLOURCHECKERS_SPDS',
        'colour.SDS_COLOURCHECKERS',
    ],
    [
        'colour.D_illuminant_relative_spd',
        'colour.sd_CIE_illuminant_D_series',
    ],
    [
        'colour.ILLUMINANTS_RELATIVE_SPDS',
        'colour.SDS_ILLUMINANTS',
    ],
    [
        'colour.LIGHT_SOURCES_RELATIVE_SPDS',
        'colour.SDS_LIGHT_SOURCES',
    ],
    [
        'colour.MultiSpectralPowerDistribution',
        'colour.MultiSpectralDistributions',
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
        'colour.matrix_colour_correction',
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

# v0.3.14
API_CHANGES['ObjectRenamed'] = API_CHANGES['ObjectRenamed'] + [
    [
        'colour.ASTME30815_PRACTISE_SHAPE',
        'colour.SPECTRAL_SHAPE_ASTME308',
    ],
    [
        'colour.decoding_cctf',
        'colour.cctf_decoding',
    ],
    [
        'colour.DECODING_CCTFS',
        'colour.CCTF_DECODINGS',
    ],
    [
        'colour.encoding_cctf',
        'colour.cctf_encoding',
    ],
    [
        'colour.ENCODING_CCTFS',
        'colour.CCTF_ENCODINGS',
    ],
    [
        'colour.EOTFS_REVERSE',
        'colour.EOTF_INVERSES',
    ],
    [
        'colour.eotf_reverse',
        'colour.eotf_inverse',
    ],
    [
        'colour.log_decoding_curve',
        'colour.log_decoding',
    ],
    [
        'colour.LOG_DECODING_CURVES',
        'colour.LOG_DECODINGS',
    ],
    [
        'colour.log_encoding_curve',
        'colour.log_encoding',
    ],
    [
        'colour.LOG_ENCODING_CURVES',
        'colour.LOG_ENCODINGS',
    ],
    [
        'colour.OETFS_REVERSE',
        'colour.OETF_INVERSES',
    ],
    [
        'colour.oetf_reverse',
        'colour.oetf_inverse',
    ],
    [
        'colour.OOTFS_REVERSE',
        'colour.OOTF_INVERSES',
    ],
    [
        'colour.ootf_reverse',
        'colour.ootf_inverse',
    ],
    [
        'colour.MultiSpectralDistribution',
        'colour.MultiSpectralDistributions',
    ],
]

# v0.3.16
API_CHANGES['ObjectRenamed'] = API_CHANGES['ObjectRenamed'] + [
    [
        'colour.ASTME308_PRACTISE_SHAPE',
        'colour.SPECTRAL_SHAPE_ASTME308',
    ],
    [
        'colour.ATD95_Specification',
        'colour.CAM_Specification_ATD95',
    ],
    [
        'colour.BRENEMAN_EXPERIMENTS_PRIMARIES_CHROMATICITIES',
        'colour.BRENEMAN_EXPERIMENT_PRIMARIES_CHROMATICITIES',
    ],
    [
        'colour.CAM16_Specification',
        'colour.CAM_Specification_CAM16',
    ],
    [
        'colour.CIECAM02_Specification',
        'colour.CAM_Specification_CIECAM02',
    ],
    [
        'colour.CAMERAS_RGB_SPECTRAL_SENSITIVITIES',
        'colour.MSDS_CAMERA_SENSITIVITIES',
    ],
    [
        'colour.CMCCAT2000_VIEWING_CONDITIONS',
        'colour.VIEWING_CONDITIONS_CMCCAT2000',
    ],
    [
        'colour.CMFS',
        'colour.colorimetry.MSDS_CMFS',
    ],
    [
        'colour.COLOURCHECKERS',
        'colour.CCS_COLOURCHECKERS',
    ],
    [
        'colour.CMCCAT2000_VIEWING_CONDITIONS',
        'colour.VIEWING_CONDITIONS_CMCCAT2000',
    ],
    [
        'colour.COLOURCHECKERS',
        'colour.CCS_COLOURCHECKERS',
    ],
    [
        'colour.COLOURCHECKERS_SDS',
        'colour.SDS_COLOURCHECKERS',
    ],
    [
        'colour.COLOUR_CORRECTION_MATRIX_METHODS',
        'colour.MATRIX_COLOUR_CORRECTION_METHODS',
    ],
    [
        'colour.DEFAULT_SPECTRAL_SHAPE',
        'colour.SPECTRAL_SHAPE_DEFAULT',
    ],
    [
        'colour.DISPLAYS_RGB_PRIMARIES',
        'colour.MSDS_DISPLAY_PRIMARIES',
    ],
    [
        'colour.Hunt_Specification',
        'colour.CAM_Specification_Hunt',
    ],
    [
        'colour.HUNTERLAB_ILLUMINANTS',
        'colour.TVS_ILLUMINANTS_HUNTERLAB',
    ],
    [
        'colour.ILLUMINANTS',
        'colour.CCS_ILLUMINANTS',
    ],
    [
        'colour.ILLUMINANTS_OPTIMAL_COLOUR_STIMULI',
        'colour.OPTIMAL_COLOUR_STIMULI_ILLUMINANTS',
    ],
    [
        'colour.ILLUMINANTS_SDS',
        'colour.SDS_ILLUMINANTS',
    ],
    [
        'colour.LEFS',
        'colour.SDS_LEFS',
    ],
    [
        'colour.LIGHT_SOURCES',
        'colour.CCS_LIGHT_SOURCES',
    ],
    [
        'colour.LIGHT_SOURCES_SDS',
        'colour.SDS_LIGHT_SOURCES',
    ],
    [
        'colour.LLAB_Specification',
        'colour.CAM_Specification_LLAB',
    ],
    [
        'colour.LMS_CMFS',
        'colour.colorimetry.MSDS_CMFS_LMS',
    ],
    [
        'colour.MULTI_SD_TO_XYZ_METHODS',
        'colour.MSDS_TO_XYZ_METHODS',
    ],
    [
        'colour.multi_sds_to_XYZ',
        'colour.msds_to_XYZ',
    ],
    [
        'colour.Nayatani95_Specification',
        'colour.CAM_Specification_Nayatani95',
    ],
    [
        'colour.PHOTOPIC_LEFS',
        'colour.colorimetry.SDS_LEFS_PHOTOPIC',
    ],
    [
        'colour.POINTER_GAMUT_BOUNDARIES',
        'colour.models.CCS_POINTER_GAMUT_BOUNDARY',
    ],
    [
        'colour.POINTER_GAMUT_DATA',
        'colour.models.DATA_POINTER_GAMUT_VOLUME',
    ],
    [
        'colour.POINTER_GAMUT_ILLUMINANT',
        'colour.models.CCS_ILLUMINANT_POINTER_GAMUT',
    ],
    [
        'colour.RGB_CMFS',
        'colour.colorimetry.MSDS_CMFS_RGB',
    ],
    [
        'colour.RGB_to_RGB_matrix',
        'colour.matrix_RGB_to_RGB',
    ],
    [
        'colour.RLAB_D_FACTOR',
        'colour.appearance.D_FACTOR_RLAB',
    ],
    [
        'colour.RLAB_Specification',
        'colour.CAM_Specification_RLAB',
    ],
    [
        'colour.SCOTOPIC_LEFS',
        'colour.colorimetry.SDS_LEFS_SCOTOPIC',
    ],
    [
        'colour.STANDARD_OBSERVERS_CMFS',
        'colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER',
    ],
    [
        'colour.YCBCR_WEIGHTS',
        'colour.WEIGHTS_YCBCR',
    ],
    [
        'colour.characterisation.colour_correction_matrix',
        'colour.characterisation.matrix_colour_correction',
    ],
    # Not strictly needed but in use by A.M.P.A.S.
    [
        'colour.characterisation.idt_matrix',
        'colour.characterisation.matrix_idt',
    ],
]

if not is_documentation_building():
    sys.modules['colour'] = colour(sys.modules['colour'],
                                   build_API_changes(API_CHANGES))

    del ModuleAPI, is_documentation_building, build_API_changes, sys
