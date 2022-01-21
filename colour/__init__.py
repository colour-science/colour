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
-   hints: Type hints for annotations.
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

import numpy as np
import sys

from .utilities.deprecation import ModuleAPI, build_API_changes
from .utilities.documentation import is_documentation_building
from .utilities.common import (
    domain_range_scale,
    get_domain_range_scale,
    set_domain_range_scale,
)

from .adaptation import (
    CHROMATIC_ADAPTATION_METHODS,
    CHROMATIC_ADAPTATION_TRANSFORMS,
    VIEWING_CONDITIONS_CMCCAT2000,
    chromatic_adaptation,
)
from .algebra import (
    CubicSplineInterpolator,
    Extrapolator,
    KernelInterpolator,
    NearestNeighbourInterpolator,
    LinearInterpolator,
    NullInterpolator,
    PchipInterpolator,
    SpragueInterpolator,
    TABLE_INTERPOLATION_METHODS,
    kernel_cardinal_spline,
    kernel_lanczos,
    kernel_linear,
    kernel_nearest_neighbour,
    kernel_sinc,
    table_interpolation,
    lagrange_coefficients,
)
from .colorimetry import (
    BANDPASS_CORRECTION_METHODS,
    CCS_ILLUMINANTS,
    CCS_LIGHT_SOURCES,
    LIGHTNESS_METHODS,
    LUMINANCE_METHODS,
    MSDS_CMFS,
    MSDS_TO_XYZ_METHODS,
    MultiSpectralDistributions,
    SDS_ILLUMINANTS,
    SDS_LEFS,
    SDS_LIGHT_SOURCES,
    SD_GAUSSIAN_METHODS,
    SD_MULTI_LEDS_METHODS,
    SD_SINGLE_LED_METHODS,
    SD_TO_XYZ_METHODS,
    SPECTRAL_SHAPE_ASTME308,
    SPECTRAL_SHAPE_DEFAULT,
    SpectralDistribution,
    SpectralShape,
    TVS_ILLUMINANTS,
    TVS_ILLUMINANTS_HUNTERLAB,
    WHITENESS_METHODS,
    YELLOWNESS_METHODS,
    bandpass_correction,
    colorimetric_purity,
    complementary_wavelength,
    dominant_wavelength,
    excitation_purity,
    lightness,
    luminance,
    luminous_efficacy,
    luminous_efficiency,
    luminous_flux,
    msds_constant,
    msds_ones,
    msds_zeros,
    msds_to_XYZ,
    sd_CIE_illuminant_D_series,
    sd_CIE_standard_illuminant_A,
    sd_blackbody,
    sd_constant,
    sd_gaussian,
    sd_mesopic_luminous_efficiency_function,
    sd_multi_leds,
    sd_ones,
    sd_single_led,
    sd_to_XYZ,
    sd_zeros,
    spectral_uniformity,
    wavelength_to_XYZ,
    whiteness,
    yellowness,
)
from .blindness import (
    CVD_MATRICES_MACHADO2010,
    matrix_anomalous_trichromacy_Machado2009,
    matrix_cvd_Machado2009,
    msds_cmfs_anomalous_trichromacy_Machado2009,
)
from .appearance import (
    CAM_Specification_ATD95,
    CAM_Specification_CAM16,
    CAM_Specification_CIECAM02,
    CAM_Specification_Hunt,
    CAM_Specification_Kim2009,
    CAM_Specification_LLAB,
    CAM_Specification_Nayatani95,
    CAM_Specification_RLAB,
    CAM_Specification_ZCAM,
    CAM16_to_XYZ,
    CIECAM02_to_XYZ,
    HKE_NAYATANI1997_METHODS,
    HelmholtzKohlrausch_effect_object_Nayatani1997,
    HelmholtzKohlrausch_effect_luminous_Nayatani1997,
    Kim2009_to_XYZ,
    MEDIA_PARAMETERS_KIM2009,
    VIEWING_CONDITIONS_CAM16,
    VIEWING_CONDITIONS_CIECAM02,
    VIEWING_CONDITIONS_HUNT,
    VIEWING_CONDITIONS_KIM2009,
    VIEWING_CONDITIONS_LLAB,
    VIEWING_CONDITIONS_RLAB,
    VIEWING_CONDITIONS_ZCAM,
    XYZ_to_ATD95,
    XYZ_to_CAM16,
    XYZ_to_CIECAM02,
    XYZ_to_Kim2009,
    XYZ_to_Hunt,
    XYZ_to_LLAB,
    XYZ_to_Nayatani95,
    XYZ_to_RLAB,
    XYZ_to_ZCAM,
    ZCAM_to_XYZ,
)
from .difference import (
    DELTA_E_METHODS,
    delta_E,
    INDEX_STRESS_METHODS,
    index_stress,
)
from .geometry import (
    PRIMITIVE_METHODS,
    primitive,
    PRIMITIVE_VERTICES_METHODS,
    primitive_vertices,
)
from .io import (
    LUT1D,
    LUT3x1D,
    LUT3D,
    LUTOperatorMatrix,
    LUTSequence,
    READ_IMAGE_METHODS,
    SpectralDistribution_IESTM2714,
    WRITE_IMAGE_METHODS,
    read_image,
    read_LUT,
    read_sds_from_csv_file,
    read_sds_from_xrite_file,
    read_spectral_data_from_csv_file,
    SpectralDistribution_Sekonic,
    SpectralDistribution_UPRTek,
    write_image,
    write_LUT,
    write_sds_to_csv_file,
)
from .models import (
    CAM02LCD_to_JMh_CIECAM02,
    CAM02SCD_to_JMh_CIECAM02,
    CAM02UCS_to_JMh_CIECAM02,
    CAM02LCD_to_XYZ,
    CAM02SCD_to_XYZ,
    CAM02UCS_to_XYZ,
    CAM16LCD_to_JMh_CAM16,
    CAM16SCD_to_JMh_CAM16,
    CAM16UCS_to_JMh_CAM16,
    CAM16LCD_to_XYZ,
    CAM16SCD_to_XYZ,
    CAM16UCS_to_XYZ,
    CCTF_DECODINGS,
    CCTF_ENCODINGS,
    CMYK_to_CMY,
    CMY_to_CMYK,
    CMY_to_RGB,
    COLOURSPACE_MODELS,
    CV_range,
    DATA_MACADAM_1942_ELLIPSES,
    DIN99_to_Lab,
    DIN99_to_XYZ,
    EOTFS,
    EOTF_INVERSES,
    HCL_to_RGB,
    HDR_CIELAB_METHODS,
    HDR_IPT_METHODS,
    HSL_to_RGB,
    HSV_to_RGB,
    Hunter_Lab_to_XYZ,
    Hunter_Rdab_to_XYZ,
    ICaCb_to_XYZ,
    ICtCp_to_RGB,
    ICtCp_to_XYZ,
    IHLS_to_RGB,
    IgPgTg_to_XYZ,
    IPT_hue_angle,
    IPT_to_XYZ,
    JMh_CAM16_to_CAM16LCD,
    JMh_CAM16_to_CAM16SCD,
    JMh_CAM16_to_CAM16UCS,
    JMh_CIECAM02_to_CAM02LCD,
    JMh_CIECAM02_to_CAM02SCD,
    JMh_CIECAM02_to_CAM02UCS,
    Jzazbz_to_XYZ,
    LCHab_to_Lab,
    LCHuv_to_Luv,
    LOG_DECODINGS,
    LOG_ENCODINGS,
    Lab_to_DIN99,
    Lab_to_LCHab,
    Lab_to_XYZ,
    Luv_to_LCHuv,
    Luv_to_XYZ,
    Luv_to_uv,
    Luv_uv_to_xy,
    OETFS,
    OETF_INVERSES,
    OOTFS,
    OOTF_INVERSES,
    OSA_UCS_to_XYZ,
    Oklab_to_XYZ,
    Prismatic_to_RGB,
    RGB_COLOURSPACES,
    RGB_Colourspace,
    RGB_luminance,
    RGB_luminance_equation,
    RGB_to_CMY,
    RGB_to_HCL,
    RGB_to_HSL,
    RGB_to_HSV,
    RGB_to_ICtCp,
    RGB_to_IHLS,
    RGB_to_Prismatic,
    RGB_to_RGB,
    RGB_to_XYZ,
    RGB_to_YCbCr,
    RGB_to_YCoCg,
    RGB_to_YcCbcCrc,
    UCS_to_XYZ,
    UCS_to_uv,
    UCS_uv_to_xy,
    UVW_to_XYZ,
    WEIGHTS_YCBCR,
    XYZ_to_CAM02LCD,
    XYZ_to_CAM02SCD,
    XYZ_to_CAM02UCS,
    XYZ_to_CAM16LCD,
    XYZ_to_CAM16SCD,
    XYZ_to_CAM16UCS,
    XYZ_to_DIN99,
    XYZ_to_Hunter_Lab,
    XYZ_to_Hunter_Rdab,
    XYZ_to_ICaCb,
    XYZ_to_ICtCp,
    XYZ_to_IgPgTg,
    XYZ_to_IPT,
    XYZ_to_Jzazbz,
    XYZ_to_K_ab_HunterLab1966,
    XYZ_to_Lab,
    XYZ_to_Luv,
    XYZ_to_OSA_UCS,
    XYZ_to_Oklab,
    XYZ_to_RGB,
    XYZ_to_UCS,
    XYZ_to_UVW,
    XYZ_to_hdr_CIELab,
    XYZ_to_hdr_IPT,
    XYZ_to_sRGB,
    XYZ_to_xy,
    XYZ_to_xyY,
    YCbCr_to_RGB,
    YCoCg_to_RGB,
    YcCbcCrc_to_RGB,
    cctf_decoding,
    cctf_encoding,
    chromatically_adapted_primaries,
    eotf,
    eotf_inverse,
    full_to_legal,
    gamma_function,
    hdr_CIELab_to_XYZ,
    hdr_IPT_to_XYZ,
    legal_to_full,
    linear_function,
    log_decoding,
    log_encoding,
    matrix_RGB_to_RGB,
    matrix_YCbCr,
    normalised_primary_matrix,
    oetf,
    oetf_inverse,
    offset_YCbCr,
    ootf,
    ootf_inverse,
    primaries_whitepoint,
    sRGB_to_XYZ,
    uv_to_Luv,
    uv_to_UCS,
    xyY_to_XYZ,
    xyY_to_xy,
    xy_to_Luv_uv,
    xy_to_UCS_uv,
    xy_to_XYZ,
    xy_to_xyY,
)
from .corresponding import (
    BRENEMAN_EXPERIMENTS,
    BRENEMAN_EXPERIMENT_PRIMARIES_CHROMATICITIES,
    CORRESPONDING_CHROMATICITIES_PREDICTION_MODELS,
    CorrespondingColourDataset,
    CorrespondingChromaticitiesPrediction,
    corresponding_chromaticities_prediction,
)
from .contrast import (
    CONTRAST_SENSITIVITY_METHODS,
    contrast_sensitivity_function,
)
from .phenomena import (
    rayleigh_scattering,
    scattering_cross_section,
    sd_rayleigh_scattering,
)
from .notation import (
    MUNSELL_COLOURS,
    MUNSELL_VALUE_METHODS,
    munsell_colour_to_xyY,
    munsell_value,
    xyY_to_munsell_colour,
)
from .quality import (
    COLOUR_FIDELITY_INDEX_METHODS,
    COLOUR_QUALITY_SCALE_METHODS,
    colour_fidelity_index,
    colour_quality_scale,
    colour_rendering_index,
    spectral_similarity_index,
)
from .recovery import XYZ_TO_SD_METHODS, XYZ_to_sd
from .temperature import (
    CCT_TO_UV_METHODS,
    CCT_TO_XY_METHODS,
    CCT_to_uv,
    CCT_to_xy,
    UV_TO_CCT_METHODS,
    XY_TO_CCT_METHODS,
    uv_to_CCT,
    xy_to_CCT,
)
from .characterisation import (
    CCS_COLOURCHECKERS,
    MATRIX_COLOUR_CORRECTION_METHODS,
    COLOUR_CORRECTION_METHODS,
    MSDS_CAMERA_SENSITIVITIES,
    MSDS_DISPLAY_PRIMARIES,
    POLYNOMIAL_EXPANSION_METHODS,
    SDS_COLOURCHECKERS,
    SDS_FILTERS,
    SDS_LENSES,
    camera_RGB_to_ACES2065_1,
    colour_correction,
    matrix_colour_correction,
    matrix_idt,
    polynomial_expansion,
    sd_to_ACES2065_1,
    sd_to_aces_relative_exposure_values,
)
from .volume import (
    OPTIMAL_COLOUR_STIMULI_ILLUMINANTS,
    RGB_colourspace_limits,
    RGB_colourspace_pointer_gamut_coverage_MonteCarlo,
    RGB_colourspace_visible_spectrum_coverage_MonteCarlo,
    RGB_colourspace_volume_MonteCarlo,
    RGB_colourspace_volume_coverage_MonteCarlo,
    is_within_macadam_limits,
    is_within_mesh_volume,
    is_within_pointer_gamut,
    is_within_visible_spectrum,
)
from .graph import describe_conversion_path, convert

from colour.utilities import is_matplotlib_installed

# Exposing "colour.plotting" sub-package if "Matplotlib" is available.
if is_matplotlib_installed():
    import colour.plotting as plotting  # noqa
else:

    class MockPlotting:  # pragma: no cover
        """
        Mock object for :mod:`colour.plotting` sub-package raising an exception
        if the sub-package is accessed but *Matplotlib* is not installed.
        """

        def __getattr__(self, attribute):
            is_matplotlib_installed(raise_exception=True)

    globals()['plotting'] = MockPlotting()  # pragma: no cover

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'domain_range_scale',
    'get_domain_range_scale',
    'set_domain_range_scale',
]
__all__ += [
    'CHROMATIC_ADAPTATION_METHODS',
    'CHROMATIC_ADAPTATION_TRANSFORMS',
    'VIEWING_CONDITIONS_CMCCAT2000',
    'chromatic_adaptation',
]
__all__ += [
    'CubicSplineInterpolator',
    'Extrapolator',
    'KernelInterpolator',
    'NearestNeighbourInterpolator',
    'LinearInterpolator',
    'NullInterpolator',
    'PchipInterpolator',
    'SpragueInterpolator',
    'TABLE_INTERPOLATION_METHODS',
    'kernel_cardinal_spline',
    'kernel_lanczos',
    'kernel_linear',
    'kernel_nearest_neighbour',
    'kernel_sinc',
    'table_interpolation',
    'lagrange_coefficients',
]
__all__ += [
    'BANDPASS_CORRECTION_METHODS',
    'CCS_ILLUMINANTS',
    'CCS_LIGHT_SOURCES',
    'LIGHTNESS_METHODS',
    'LUMINANCE_METHODS',
    'MSDS_CMFS',
    'MSDS_TO_XYZ_METHODS',
    'MultiSpectralDistributions',
    'SDS_ILLUMINANTS',
    'SDS_LEFS',
    'SDS_LIGHT_SOURCES',
    'SD_GAUSSIAN_METHODS',
    'SD_MULTI_LEDS_METHODS',
    'SD_SINGLE_LED_METHODS',
    'SD_TO_XYZ_METHODS',
    'SPECTRAL_SHAPE_ASTME308',
    'SPECTRAL_SHAPE_DEFAULT',
    'SpectralDistribution',
    'SpectralShape',
    'TVS_ILLUMINANTS',
    'TVS_ILLUMINANTS_HUNTERLAB',
    'WHITENESS_METHODS',
    'YELLOWNESS_METHODS',
    'bandpass_correction',
    'colorimetric_purity',
    'complementary_wavelength',
    'dominant_wavelength',
    'excitation_purity',
    'lightness',
    'luminance',
    'luminous_efficacy',
    'luminous_efficiency',
    'luminous_flux',
    'msds_constant',
    'msds_ones',
    'msds_zeros',
    'msds_to_XYZ',
    'sd_CIE_illuminant_D_series',
    'sd_CIE_standard_illuminant_A',
    'sd_blackbody',
    'sd_constant',
    'sd_gaussian',
    'sd_mesopic_luminous_efficiency_function',
    'sd_multi_leds',
    'sd_ones',
    'sd_single_led',
    'sd_to_XYZ',
    'sd_zeros',
    'spectral_uniformity',
    'wavelength_to_XYZ',
    'whiteness',
    'yellowness',
]
__all__ += [
    'CVD_MATRICES_MACHADO2010',
    'matrix_anomalous_trichromacy_Machado2009',
    'matrix_cvd_Machado2009',
    'msds_cmfs_anomalous_trichromacy_Machado2009',
]
__all__ += [
    'CAM_Specification_ATD95',
    'CAM_Specification_CAM16',
    'CAM_Specification_CIECAM02',
    'CAM_Specification_Hunt',
    'CAM_Specification_Kim2009',
    'CAM_Specification_LLAB',
    'CAM_Specification_Nayatani95',
    'CAM_Specification_RLAB',
    'CAM_Specification_ZCAM',
    'CAM16_to_XYZ',
    'CIECAM02_to_XYZ',
    'HKE_NAYATANI1997_METHODS',
    'HelmholtzKohlrausch_effect_object_Nayatani1997',
    'HelmholtzKohlrausch_effect_luminous_Nayatani1997',
    'Kim2009_to_XYZ',
    'MEDIA_PARAMETERS_KIM2009',
    'VIEWING_CONDITIONS_CAM16',
    'VIEWING_CONDITIONS_CIECAM02',
    'VIEWING_CONDITIONS_HUNT',
    'VIEWING_CONDITIONS_KIM2009',
    'VIEWING_CONDITIONS_LLAB',
    'VIEWING_CONDITIONS_RLAB',
    'VIEWING_CONDITIONS_ZCAM',
    'XYZ_to_ATD95',
    'XYZ_to_CAM16',
    'XYZ_to_CIECAM02',
    'XYZ_to_Kim2009',
    'XYZ_to_Hunt',
    'XYZ_to_LLAB',
    'XYZ_to_Nayatani95',
    'XYZ_to_RLAB',
    'XYZ_to_ZCAM',
    'ZCAM_to_XYZ',
]
__all__ += [
    'DELTA_E_METHODS',
    'delta_E',
    'INDEX_STRESS_METHODS',
    'index_stress',
]
__all__ += [
    'PRIMITIVE_METHODS',
    'primitive',
    'PRIMITIVE_VERTICES_METHODS',
    'primitive_vertices',
]
__all__ += [
    'LUT1D',
    'LUT3x1D',
    'LUT3D',
    'LUTOperatorMatrix',
    'LUTSequence',
    'READ_IMAGE_METHODS',
    'SpectralDistribution_IESTM2714',
    'WRITE_IMAGE_METHODS',
    'read_image',
    'read_LUT',
    'read_sds_from_csv_file',
    'read_sds_from_xrite_file',
    'read_spectral_data_from_csv_file',
    'SpectralDistribution_UPRTek',
    'SpectralDistribution_Sekonic',
    'write_image',
    'write_LUT',
    'write_sds_to_csv_file',
]
__all__ += [
    'CAM02LCD_to_JMh_CIECAM02',
    'CAM02SCD_to_JMh_CIECAM02',
    'CAM02UCS_to_JMh_CIECAM02',
    'CAM02LCD_to_XYZ',
    'CAM02SCD_to_XYZ',
    'CAM02UCS_to_XYZ',
    'CAM16LCD_to_JMh_CAM16',
    'CAM16SCD_to_JMh_CAM16',
    'CAM16UCS_to_JMh_CAM16',
    'CAM16LCD_to_XYZ',
    'CAM16SCD_to_XYZ',
    'CAM16UCS_to_XYZ',
    'CCTF_DECODINGS',
    'CCTF_ENCODINGS',
    'CMYK_to_CMY',
    'CMY_to_CMYK',
    'CMY_to_RGB',
    'COLOURSPACE_MODELS',
    'CV_range',
    'DATA_MACADAM_1942_ELLIPSES',
    'DIN99_to_Lab',
    'DIN99_to_XYZ',
    'EOTFS',
    'EOTF_INVERSES',
    'HCL_to_RGB',
    'HDR_CIELAB_METHODS',
    'HDR_IPT_METHODS',
    'HSL_to_RGB',
    'HSV_to_RGB',
    'Hunter_Lab_to_XYZ',
    'Hunter_Rdab_to_XYZ',
    'ICaCb_to_XYZ',
    'ICtCp_to_RGB',
    'ICtCp_to_XYZ',
    'IHLS_to_RGB',
    'IgPgTg_to_XYZ',
    'IPT_hue_angle',
    'IPT_to_XYZ',
    'JMh_CAM16_to_CAM16LCD',
    'JMh_CAM16_to_CAM16SCD',
    'JMh_CAM16_to_CAM16UCS',
    'JMh_CIECAM02_to_CAM02LCD',
    'JMh_CIECAM02_to_CAM02SCD',
    'JMh_CIECAM02_to_CAM02UCS',
    'Jzazbz_to_XYZ',
    'LCHab_to_Lab',
    'LCHuv_to_Luv',
    'LOG_DECODINGS',
    'LOG_ENCODINGS',
    'Lab_to_DIN99',
    'Lab_to_LCHab',
    'Lab_to_XYZ',
    'Luv_to_LCHuv',
    'Luv_to_XYZ',
    'Luv_to_uv',
    'Luv_uv_to_xy',
    'OETFS',
    'OETF_INVERSES',
    'OOTFS',
    'OOTF_INVERSES',
    'OSA_UCS_to_XYZ',
    'Oklab_to_XYZ',
    'Prismatic_to_RGB',
    'RGB_COLOURSPACES',
    'RGB_Colourspace',
    'RGB_luminance',
    'RGB_luminance_equation',
    'RGB_to_CMY',
    'RGB_to_HCL',
    'RGB_to_HSL',
    'RGB_to_HSV',
    'RGB_to_ICtCp',
    'RGB_to_IHLS',
    'RGB_to_Prismatic',
    'RGB_to_RGB',
    'RGB_to_XYZ',
    'RGB_to_YCbCr',
    'RGB_to_YCoCg',
    'RGB_to_YcCbcCrc',
    'UCS_to_XYZ',
    'UCS_to_uv',
    'UCS_uv_to_xy',
    'UVW_to_XYZ',
    'WEIGHTS_YCBCR',
    'XYZ_to_CAM02LCD',
    'XYZ_to_CAM02SCD',
    'XYZ_to_CAM02UCS',
    'XYZ_to_CAM16LCD',
    'XYZ_to_CAM16SCD',
    'XYZ_to_CAM16UCS',
    'XYZ_to_DIN99',
    'XYZ_to_Hunter_Lab',
    'XYZ_to_Hunter_Rdab',
    'XYZ_to_ICaCb',
    'XYZ_to_ICtCp',
    'XYZ_to_IgPgTg',
    'XYZ_to_IPT',
    'XYZ_to_Jzazbz',
    'XYZ_to_K_ab_HunterLab1966',
    'XYZ_to_Lab',
    'XYZ_to_Luv',
    'XYZ_to_OSA_UCS',
    'XYZ_to_Oklab',
    'XYZ_to_RGB',
    'XYZ_to_UCS',
    'XYZ_to_UVW',
    'XYZ_to_hdr_CIELab',
    'XYZ_to_hdr_IPT',
    'XYZ_to_sRGB',
    'XYZ_to_xy',
    'XYZ_to_xyY',
    'YCbCr_to_RGB',
    'YCoCg_to_RGB',
    'YcCbcCrc_to_RGB',
    'cctf_decoding',
    'cctf_encoding',
    'chromatically_adapted_primaries',
    'eotf',
    'eotf_inverse',
    'full_to_legal',
    'gamma_function',
    'hdr_CIELab_to_XYZ',
    'hdr_IPT_to_XYZ',
    'legal_to_full',
    'linear_function',
    'log_decoding',
    'log_encoding',
    'matrix_RGB_to_RGB',
    'matrix_YCbCr',
    'normalised_primary_matrix',
    'oetf',
    'oetf_inverse',
    'offset_YCbCr',
    'ootf',
    'ootf_inverse',
    'primaries_whitepoint',
    'sRGB_to_XYZ',
    'uv_to_Luv',
    'uv_to_UCS',
    'xyY_to_XYZ',
    'xyY_to_xy',
    'xy_to_Luv_uv',
    'xy_to_UCS_uv',
    'xy_to_XYZ',
    'xy_to_xyY',
]
__all__ += [
    'BRENEMAN_EXPERIMENTS',
    'BRENEMAN_EXPERIMENT_PRIMARIES_CHROMATICITIES',
    'CORRESPONDING_CHROMATICITIES_PREDICTION_MODELS',
    'CorrespondingColourDataset',
    'CorrespondingChromaticitiesPrediction',
    'corresponding_chromaticities_prediction',
]
__all__ += [
    'CONTRAST_SENSITIVITY_METHODS',
    'contrast_sensitivity_function',
]
__all__ += [
    'rayleigh_scattering',
    'scattering_cross_section',
    'sd_rayleigh_scattering',
]
__all__ += [
    'MUNSELL_COLOURS',
    'MUNSELL_VALUE_METHODS',
    'munsell_colour_to_xyY',
    'munsell_value',
    'xyY_to_munsell_colour',
]
__all__ += [
    'COLOUR_FIDELITY_INDEX_METHODS',
    'COLOUR_QUALITY_SCALE_METHODS',
    'colour_fidelity_index',
    'colour_quality_scale',
    'colour_rendering_index',
    'spectral_similarity_index',
]
__all__ += [
    'XYZ_TO_SD_METHODS',
    'XYZ_to_sd',
]
__all__ += [
    'CCT_TO_UV_METHODS',
    'CCT_TO_XY_METHODS',
    'CCT_to_uv',
    'CCT_to_xy',
    'UV_TO_CCT_METHODS',
    'XY_TO_CCT_METHODS',
    'uv_to_CCT',
    'xy_to_CCT',
]
__all__ += [
    'CCS_COLOURCHECKERS',
    'MATRIX_COLOUR_CORRECTION_METHODS',
    'COLOUR_CORRECTION_METHODS',
    'MSDS_CAMERA_SENSITIVITIES',
    'MSDS_DISPLAY_PRIMARIES',
    'POLYNOMIAL_EXPANSION_METHODS',
    'SDS_COLOURCHECKERS',
    'SDS_FILTERS',
    'SDS_LENSES',
    'camera_RGB_to_ACES2065_1',
    'colour_correction',
    'matrix_colour_correction',
    'matrix_idt',
    'polynomial_expansion',
    'sd_to_ACES2065_1',
    'sd_to_aces_relative_exposure_values',
]
__all__ += [
    'OPTIMAL_COLOUR_STIMULI_ILLUMINANTS',
    'RGB_colourspace_limits',
    'RGB_colourspace_pointer_gamut_coverage_MonteCarlo',
    'RGB_colourspace_visible_spectrum_coverage_MonteCarlo',
    'RGB_colourspace_volume_MonteCarlo',
    'RGB_colourspace_volume_coverage_MonteCarlo',
    'is_within_macadam_limits',
    'is_within_mesh_volume',
    'is_within_pointer_gamut',
    'is_within_visible_spectrum',
]
__all__ += [
    'describe_conversion_path',
    'convert',
]

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

# v0.4.0
API_CHANGES = {
    'ObjectRenamed': [
        [
            'colour.RGB_to_ICTCP',
            'colour.RGB_to_ICtCp',
        ],
        [
            'colour.ICTCP_to_RGB',
            'colour.ICtCp_to_RGB',
        ],
        [
            'colour.RGB_to_IGPGTG',
            'colour.RGB_to_IgPgTg',
        ],
        [
            'colour.IGPGTG_to_RGB',
            'colour.IgPgTg_to_RGB',
        ],
        [
            'colour.XYZ_to_JzAzBz',
            'colour.XYZ_to_Jzazbz',
        ],
        [
            'colour.JzAzBz_to_XYZ',
            'colour.Jzazbz_to_XYZ',
        ],
    ]
}
"""
Defines the *colour.models* sub-package API changes.

API_CHANGES : dict
"""

if not is_documentation_building():
    sys.modules['colour'] = colour(sys.modules['colour'],
                                   build_API_changes(API_CHANGES))

    del ModuleAPI, is_documentation_building, build_API_changes, sys

__disable_lazy_load__ = colour.__disable_lazy_load__ = True
"""
Ensures that the lazy loaded datasets are not transformed during import.
See :class:`colour.utilities.LazyCaseInsensitiveMapping` for more information.
"""
