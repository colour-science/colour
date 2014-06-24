# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**__init__.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Color** package initialisation.

**Others:**

"""

from __future__ import unicode_literals

import foundations.globals.constants

from color.globals.constants import Constants

foundations.globals.constants.Constants.__dict__.update(Constants.__dict__)

from color.algebra.common import get_closest, get_steps, is_uniform
from color.algebra.interpolation import SpragueInterpolator
from color.algebra.matrix import is_identity, linear_interpolate_matrices
from color.algebra.regression import linear_regression

from color.computation.blackbody import blackbody_spectral_power_distribution, blackbody_spectral_radiance, planck_law
from color.computation.chromatic_adaptation import CHROMATIC_ADAPTATION_METHODS
from color.computation.chromatic_adaptation import get_chromatic_adaptation_matrix
from color.computation.cmfs import LMS_ConeFundamentals, RGB_ColorMatchingFunctions, XYZ_ColorMatchingFunctions
from color.computation.cmfs import RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs, RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs
from color.computation.colorspace import Colorspace
from color.computation.correction import bandpass_correction, bandpass_correction_stearns
from color.computation.cri import get_color_rendering_index
from color.computation.derivation import get_normalized_primary_matrix
from color.computation.difference import delta_E_CIE_1976, delta_E_CIE_1994, delta_E_CIE_2000, delta_E_CMC
from color.computation.illuminants import D_illuminant_relative_spd
from color.computation.lefs import mesopic_luminous_efficiency_function, mesopic_weighting_function
from color.computation.lightness import get_lightness, get_luminance, get_luminance_equation, get_munsell_value
from color.computation.lightness import lightness_1958, lightness_1964, lightness_1976
from color.computation.lightness import luminance_1943, luminance_1976
from color.computation.lightness import munsell_value_1920, munsell_value_1933, munsell_value_1943, munsell_value_1944, munsell_value_1955
from color.computation.lightness import LIGHTNESS_FUNCTIONS, MUNSELL_VALUE_FUNCTIONS
from color.computation.spectrum import SpectralPowerDistribution, SpectralPowerDistributionTriad
from color.computation.temperature import CCT_to_uv, CCT_to_uv_ohno, CCT_to_uv_robertson
from color.computation.temperature import uv_to_CCT, uv_to_CCT_ohno, uv_to_CCT_robertson
from color.computation.temperature import D_illuminant_CCT_to_xy
from color.computation.transformations import XYZ_to_xyY, xyY_to_XYZ, xyY_to_RGB, RGB_to_xyY
from color.computation.transformations import xy_to_XYZ, XYZ_to_xy
from color.computation.transformations import XYZ_to_RGB, RGB_to_XYZ
from color.computation.transformations import XYZ_to_UCS, UCS_to_XYZ, UCS_to_uv, UCS_uv_to_xy
from color.computation.transformations import XYZ_to_UVW
from color.computation.transformations import XYZ_to_Luv, Luv_to_XYZ, Luv_to_uv, Luv_uv_to_xy, Luv_to_LCHuv, LCHuv_to_Luv
from color.computation.transformations import XYZ_to_Lab, Lab_to_XYZ, Lab_to_LCHab, LCHab_to_Lab
from color.computation.tristimulus import spectral_to_XYZ, wavelength_to_XYZ

from color.dataset.cmfs import CMFS, LMS_ConeFundamentals, RGB_CMFS, STANDARD_OBSERVERS_CMFS
from color.dataset.color_checkers.chromaticity_coordinates import COLORCHECKERS
from color.dataset.color_checkers.spds import COLORCHECKERS_SPDS
from color.dataset.colorspaces.aces_rgb import ACES_RGB_COLORSPACE, ACES_RGB_LOG_COLORSPACE
from color.dataset.colorspaces.aces_rgb import ACES_RGB_PROXY_10_COLORSPACE, ACES_RGB_PROXY_12_COLORSPACE
from color.dataset.colorspaces.adobe_rgb_1998 import ADOBE_RGB_1998_COLORSPACE
from color.dataset.colorspaces.adobe_wide_gamut_rgb import ADOBE_WIDE_GAMUT_RGB_COLORSPACE
from color.dataset.colorspaces.alexa_wide_gamut_rgb import ALEXA_WIDE_GAMUT_RGB_COLORSPACE
from color.dataset.colorspaces.apple_rgb import APPLE_RGB_COLORSPACE
from color.dataset.colorspaces.best_rgb import BEST_RGB_COLORSPACE
from color.dataset.colorspaces.best_rgb import BEST_RGB_COLORSPACE
from color.dataset.colorspaces.beta_rgb import BETA_RGB_COLORSPACE
from color.dataset.colorspaces.c_log import C_LOG_COLORSPACE
from color.dataset.colorspaces.cie_rgb import CIE_RGB_COLORSPACE
from color.dataset.colorspaces.color_match_rgb import COLOR_MATCH_RGB_COLORSPACE
from color.dataset.colorspaces.dci_p3 import DCI_P3_COLORSPACE
from color.dataset.colorspaces.don_rgb_4 import DON_RGB_4_COLORSPACE
from color.dataset.colorspaces.eci_rgb_v2 import ECI_RGB_V2_COLORSPACE
from color.dataset.colorspaces.ekta_space_ps5 import EKTA_SPACE_PS_5_COLORSPACE
from color.dataset.colorspaces.max_rgb import MAX_RGB_COLORSPACE
from color.dataset.colorspaces.ntsc_rgb import NTSC_RGB_COLORSPACE
from color.dataset.colorspaces.pal_secam_rgb import PAL_SECAM_RGB_COLORSPACE
from color.dataset.colorspaces.pointer_gamut import POINTER_GAMUT_DATA
from color.dataset.colorspaces.prophoto_rgb import PROPHOTO_RGB_COLORSPACE
from color.dataset.colorspaces.rec_709 import REC_709_COLORSPACE
from color.dataset.colorspaces.rec_2020 import REC_2020_COLORSPACE
from color.dataset.colorspaces.russell_rgb import RUSSELL_RGB_COLORSPACE
from color.dataset.colorspaces.s_log import S_LOG_COLORSPACE
from color.dataset.colorspaces.smptec_rgb import SMPTE_C_RGB_COLORSPACE
from color.dataset.colorspaces.srgb import sRGB_COLORSPACE
from color.dataset.colorspaces.xtreme_rgb import XTREME_RGB_COLORSPACE
from color.dataset.illuminants.chromaticity_coordinates import ILLUMINANTS
from color.dataset.illuminants.d_illuminants_s_spds import D_ILLUMINANTS_S_SPDS
from color.dataset.illuminants.spds import ILLUMINANTS_RELATIVE_SPDS
from color.dataset.lefs import LEFS, PHOTOPIC_LEFS, SCOTOPIC_LEFS
from color.dataset.tcs import TCS_SPDS

from color.implementation.fitting import first_order_color_fit

from color.utilities.verbose import get_logging_console_handler, install_logger, set_verbosity_level

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["Constants"]

# *color.algebra* objects.
__all__.extend(["get_closest", "get_steps", "is_uniform",
                "is_identity", "linear_interpolate_matrices",
                "linear_regression",
                "SpragueInterpolator"])

# *color.computation.blackbody* objects.
__all__.extend(["blackbody_spectral_power_distribution", "blackbody_spectral_radiance", "planck_law"])

# *color.computation.chromatic_adaptation* objects.
__all__.extend(["CHROMATIC_ADAPTATION_METHODS", "get_chromatic_adaptation_matrix"])

# *color.computation.cmfs* objects.
__all__.extend(["LMS_ConeFundamentals", "RGB_ColorMatchingFunctions", "XYZ_ColorMatchingFunctions"])
__all__.extend(["RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs", "RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs"])

# *color.computation.colorspaces* objects.
__all__.extend(["Colorspace"])

# *color.computation.correction* objects.
__all__.extend(["bandpass_correction", "bandpass_correction_stearns"])

# *color.computation.cri* objects.
__all__.extend(["get_color_rendering_index"])

# *color.computation.derivation* objects.
__all__.extend(["get_normalized_primary_matrix"])

# *color.computation.difference* objects.
__all__.extend(["delta_E_CIE_1976", "delta_E_CIE_1994", "delta_E_CIE_2000", "delta_E_CMC"])

# *color.computation.illuminants* objects.
__all__.extend(["D_illuminant_relative_spd"])

# *color.computation.lefs* objects.
__all__.extend(["mesopic_luminous_efficiency_function", "mesopic_weighting_function"])

# *color.computation.lightness* objects.
__all__.extend(["get_lightness", "get_luminance", "get_luminance_equation", "get_munsell_value"])
__all__.extend(["lightness_1958", "lightness_1964", "lightness_1976"])
__all__.extend(["luminance_1943", "luminance_1976"])
__all__.extend(["munsell_value_1920", "munsell_value_1933", "munsell_value_1943", "munsell_value_1944", "munsell_value_1955"])
__all__.extend(["LIGHTNESS_FUNCTIONS", "MUNSELL_VALUE_FUNCTIONS"])

# *color.computation.spectrum* objects.
__all__.extend(["SpectralPowerDistribution", "SpectralPowerDistributionTriad"])

# *color.computation.temperature* objects.
__all__.extend(["CCT_to_uv", "CCT_to_uv_ohno", "CCT_to_uv_robertson",
                "uv_to_CCT", "uv_to_CCT_ohno", "uv_to_CCT_robertson",
                "D_illuminant_CCT_to_xy", ])

# *color.computation.transformations* objects.
__all__.extend(["XYZ_to_xyY", "xyY_to_XYZ", "xyY_to_RGB", "RGB_to_xyY",
                "xy_to_XYZ", "XYZ_to_xy",
                "XYZ_to_RGB", "RGB_to_XYZ",
                "XYZ_to_UCS", "UCS_to_XYZ", "UCS_to_uv", "UCS_uv_to_xy",
                "XYZ_to_UVW",
                "XYZ_to_Luv", "Luv_to_XYZ", "Luv_to_uv", "Luv_uv_to_xy", "Luv_to_LCHuv", "LCHuv_to_Luv",
                "XYZ_to_Lab", "Lab_to_XYZ", "Lab_to_LCHab", "LCHab_to_Lab"])

# *color.computation.tristimulus* objects.
__all__.extend(["spectral_to_XYZ", "wavelength_to_XYZ"])

# *color.dataset.cmfs* objects.
__all__.extend(["CMFS", "LMS_ConeFundamentals", "RGB_CMFS", "STANDARD_OBSERVERS_CMFS"])

# *color.dataset.color_checkers.chromaticity_coordinates* objects.
__all__.extend(["COLORCHECKERS"])

# *color.dataset.color_checkers.spds* objects.
__all__.extend(["COLORCHECKERS_SPDS"])

# *color.dataset.colorspaces* objects.
COLORSPACES = {ACES_RGB_COLORSPACE.name: ACES_RGB_COLORSPACE,
               ACES_RGB_LOG_COLORSPACE.name: ACES_RGB_LOG_COLORSPACE,
               ACES_RGB_PROXY_10_COLORSPACE.name: ACES_RGB_PROXY_10_COLORSPACE,
               ACES_RGB_PROXY_12_COLORSPACE.name: ACES_RGB_PROXY_12_COLORSPACE,
               ADOBE_RGB_1998_COLORSPACE.name: ADOBE_RGB_1998_COLORSPACE,
               ADOBE_WIDE_GAMUT_RGB_COLORSPACE.name: ADOBE_WIDE_GAMUT_RGB_COLORSPACE,
               ALEXA_WIDE_GAMUT_RGB_COLORSPACE.name: ALEXA_WIDE_GAMUT_RGB_COLORSPACE,
               APPLE_RGB_COLORSPACE.name: APPLE_RGB_COLORSPACE,
               BEST_RGB_COLORSPACE.name: BEST_RGB_COLORSPACE,
               BETA_RGB_COLORSPACE.name: BETA_RGB_COLORSPACE,
               CIE_RGB_COLORSPACE.name: CIE_RGB_COLORSPACE,
               C_LOG_COLORSPACE.name: C_LOG_COLORSPACE,
               COLOR_MATCH_RGB_COLORSPACE.name: COLOR_MATCH_RGB_COLORSPACE,
               DCI_P3_COLORSPACE.name: DCI_P3_COLORSPACE,
               DON_RGB_4_COLORSPACE.name: DON_RGB_4_COLORSPACE,
               ECI_RGB_V2_COLORSPACE.name: ECI_RGB_V2_COLORSPACE,
               EKTA_SPACE_PS_5_COLORSPACE.name: EKTA_SPACE_PS_5_COLORSPACE,
               MAX_RGB_COLORSPACE.name: MAX_RGB_COLORSPACE,
               NTSC_RGB_COLORSPACE.name: NTSC_RGB_COLORSPACE,
               PAL_SECAM_RGB_COLORSPACE.name: PAL_SECAM_RGB_COLORSPACE,
               PROPHOTO_RGB_COLORSPACE.name: PROPHOTO_RGB_COLORSPACE,
               REC_709_COLORSPACE.name: REC_709_COLORSPACE,
               REC_2020_COLORSPACE.name: REC_2020_COLORSPACE,
               RUSSELL_RGB_COLORSPACE.name: RUSSELL_RGB_COLORSPACE,
               S_LOG_COLORSPACE.name: S_LOG_COLORSPACE,
               SMPTE_C_RGB_COLORSPACE.name: SMPTE_C_RGB_COLORSPACE,
               sRGB_COLORSPACE.name: sRGB_COLORSPACE,
               XTREME_RGB_COLORSPACE.name: XTREME_RGB_COLORSPACE}

__all__.extend(["ACES_RGB_COLORSPACE", "ACES_RGB_LOG_COLORSPACE",
                "ACES_RGB_PROXY_10_COLORSPACE", "ACES_RGB_PROXY_12_COLORSPACE",
                "ADOBE_RGB_1998_COLORSPACE",
                "ADOBE_WIDE_GAMUT_RGB_COLORSPACE",
                "ALEXA_WIDE_GAMUT_RGB_COLORSPACE",
                "APPLE_RGB_COLORSPACE",
                "BEST_RGB_COLORSPACE",
                "BETA_RGB_COLORSPACE",
                "C_LOG_COLORSPACE",
                "CIE_RGB_COLORSPACE",
                "COLOR_MATCH_RGB_COLORSPACE",
                "DCI_P3_COLORSPACE",
                "DON_RGB_4_COLORSPACE",
                "ECI_RGB_V2_COLORSPACE",
                "EKTA_SPACE_PS_5_COLORSPACE",
                "MAX_RGB_COLORSPACE",
                "NTSC_RGB_COLORSPACE",
                "PAL_SECAM_RGB_COLORSPACE",
                "POINTER_GAMUT_DATA",
                "PROPHOTO_RGB_COLORSPACE",
                "REC_709_COLORSPACE",
                "REC_2020_COLORSPACE",
                "RUSSELL_RGB_COLORSPACE",
                "S_LOG_COLORSPACE",
                "SMPTE_C_RGB_COLORSPACE",
                "sRGB_COLORSPACE",
                "XTREME_RGB_COLORSPACE",
                "COLORSPACES"])

# *color.dataset.illuminants.chromaticity_coordinates* objects.
__all__.extend(["ILLUMINANTS"])

# *color.dataset.illuminants.d_illuminants_s_spds* objects.
__all__.extend(["D_ILLUMINANTS_S_SPDS"])

# *color.dataset.illuminants.spds* objects.
__all__.extend(["ILLUMINANTS_RELATIVE_SPDS"])

# *color.dataset.lefs* objects.
__all__.extend(["LEFS", "PHOTOPIC_LEFS", "SCOTOPIC_LEFS"])

# *color.dataset.tcs* objects.
__all__.extend(["TCS_SPDS"])

# *color.implementation* objects.
__all__.extend(["first_order_color_fit"])

LOGGER = install_logger()

get_logging_console_handler()
set_verbosity_level(Constants.verbosity_level)

__all__.extend("LOGGER")

__all__ = map(lambda x: x.encode("ascii"), __all__)

__version__ = Constants.version
