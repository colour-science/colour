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

from color.algebra.interpolation import SpragueInterpolator
from color.algebra.matrix import is_identity, linear_interpolate_matrices
from color.algebra.regression import linear_regression
from color.chromatic_adaptation import CHROMATIC_ADAPTATION_METHODS
from color.chromatic_adaptation import get_chromatic_adaptation_matrix
from color.color_checkers import COLORCHECKERS
from color.colorspaces.colorspace import Colorspace
from color.colorspaces.aces_rgb import ACES_RGB_COLORSPACE, ACES_RGB_LOG_COLORSPACE
from color.colorspaces.aces_rgb import ACES_RGB_PROXY_10_COLORSPACE, ACES_RGB_PROXY_12_COLORSPACE
from color.colorspaces.adobe_rgb_1998 import ADOBE_RGB_1998_COLORSPACE
from color.colorspaces.adobe_wide_gamut_rgb import ADOBE_WIDE_GAMUT_RGB_COLORSPACE
from color.colorspaces.alexa_wide_gamut_rgb import ALEXA_WIDE_GAMUT_RGB_COLORSPACE
from color.colorspaces.apple_rgb import APPLE_RGB_COLORSPACE
from color.colorspaces.best_rgb import BEST_RGB_COLORSPACE
from color.colorspaces.best_rgb import BEST_RGB_COLORSPACE
from color.colorspaces.beta_rgb import BETA_RGB_COLORSPACE
from color.colorspaces.c_log import C_LOG_COLORSPACE
from color.colorspaces.cie_rgb import CIE_RGB_COLORSPACE
from color.colorspaces.color_match_rgb import COLOR_MATCH_RGB_COLORSPACE
from color.colorspaces.dci_p3 import DCI_P3_COLORSPACE
from color.colorspaces.don_rgb_4 import DON_RGB_4_COLORSPACE
from color.colorspaces.eci_rgb_v2 import ECI_RGB_V2_COLORSPACE
from color.colorspaces.ekta_space_ps5 import EKTA_SPACE_PS_5_COLORSPACE
from color.colorspaces.max_rgb import MAX_RGB_COLORSPACE
from color.colorspaces.ntsc_rgb import NTSC_RGB_COLORSPACE
from color.colorspaces.pal_secam_rgb import PAL_SECAM_RGB_COLORSPACE
from color.colorspaces.pointer_gamut import POINTER_GAMUT_DATA
from color.colorspaces.prophoto_rgb import PROPHOTO_RGB_COLORSPACE
from color.colorspaces.rec_709 import REC_709_COLORSPACE
from color.colorspaces.rec_2020 import REC_2020_COLORSPACE
from color.colorspaces.russell_rgb import RUSSELL_RGB_COLORSPACE
from color.colorspaces.s_log import S_LOG_COLORSPACE
from color.colorspaces.smptec_rgb import SMPTE_C_RGB_COLORSPACE
from color.colorspaces.srgb import sRGB_COLORSPACE
from color.colorspaces.xtreme_rgb import XTREME_RGB_COLORSPACE
from color.cri import get_color_rendering_index
from color.derivation import get_normalized_primary_matrix
from color.difference import delta_E_CIE_1976, delta_E_CIE_1994, delta_E_CIE_2000, delta_E_CMC
from color.illuminants import ILLUMINANTS
from color.implementations.fitting import first_order_color_fit
from color.lightness import get_lightness, get_luminance, get_luminance_equation, get_munsell_value
from color.lightness import lightness_1958, lightness_1964, lightness_1976
from color.lightness import luminance_1943, luminance_1976
from color.lightness import munsell_value_1920, munsell_value_1933, munsell_value_1943, munsell_value_1944, munsell_value_1955
from color.lightness import LIGHTNESS_FUNCTIONS, MUNSELL_VALUE_FUNCTIONS
from color.spectrum.blackbody import blackbody_spectral_power_distribution, blackbody_spectral_radiance, planck_law
from color.spectrum.cmfs import CIE_RGB_CMFS, CMFS, STANDARD_OBSERVERS_CMFS
from color.spectrum.color_checkers import BABELCOLOR_AVERAGE_SPDS, COLORCHECKER_N_OHTA_SPDS, COLORCHECKERS_SPDS
from color.spectrum.correction import bandpass_correction, bandpass_correction_stearns
from color.spectrum.illuminants import D_illuminant_relative_spectral_power_distribution
from color.spectrum.illuminants import D_ILLUMINANTS_S_SPDS, ILLUMINANTS_RELATIVE_SPDS
from color.spectrum.lefs import PHOTOPIC_LEFS, SCOTOPIC_LEFS
from color.spectrum.spd import SpectralPowerDistribution
from color.spectrum.spd import AbstractColorMatchingFunctions, RGB_ColorMatchingFunctions, XYZ_ColorMatchingFunctions
from color.spectrum.tcs import TCS_SPDS
from color.spectrum.transformations import spectral_to_XYZ, wavelength_to_XYZ
from color.temperature import CCT_to_uv, CCT_to_uv_ohno, CCT_to_uv_robertson
from color.temperature import uv_to_CCT, uv_to_CCT_ohno, uv_to_CCT_robertson
from color.temperature import D_illuminant_CCT_to_xy
from color.transformations import XYZ_to_xyY, xyY_to_XYZ, xyY_to_RGB, RGB_to_xyY
from color.transformations import xy_to_XYZ, XYZ_to_xy
from color.transformations import XYZ_to_RGB, RGB_to_XYZ
from color.transformations import XYZ_to_UCS, UCS_to_XYZ, UCS_to_uv, UCS_uv_to_xy
from color.transformations import XYZ_to_UVW
from color.transformations import XYZ_to_Luv, Luv_to_XYZ, Luv_to_uv, Luv_uv_to_xy, Luv_to_LCHuv, LCHuv_to_Luv
from color.transformations import XYZ_to_Lab, Lab_to_XYZ, Lab_to_LCHab, LCHab_to_Lab
from color.utilities.verbose import get_logging_console_handler, install_logger, set_verbosity_level

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["Constants"]

# *color.algebra* objects.
__all__.extend(["is_identity",
                "linear_interpolate_matrices",
                "linear_regression",
                "SpragueInterpolator"])

# *color.chromatic_adaptation* objects.
__all__.extend(["CHROMATIC_ADAPTATION_METHODS", "get_chromatic_adaptation_matrix"])

# *color.color_checkers* objects.
__all__.extend(["COLORCHECKERS"])

# *color.colorspaces* objects.
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

__all__.extend(["Colorspace",
                "ACES_RGB_COLORSPACE", "ACES_RGB_LOG_COLORSPACE",
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

# *color.cri* objects.
__all__.extend(["get_color_rendering_index"])

# *color.derivation* objects.
__all__.extend(["get_normalized_primary_matrix"])

# *color.difference* objects.
__all__.extend(["delta_E_CIE_1976", "delta_E_CIE_1994", "delta_E_CIE_2000", "delta_E_CMC"])

# *color.illuminants* objects.
__all__.extend(["ILLUMINANTS"])

# *color.implementations* objects.
__all__.extend(["first_order_color_fit"])

# *color.lightness* objects.
__all__.extend(["get_lightness", "get_luminance", "get_luminance_equation", "get_munsell_value",
                "lightness_1958", "lightness_1964", "lightness_1976",
                "luminance_1943", "luminance_1976",
                "munsell_value_1920", "munsell_value_1933", "munsell_value_1943", "munsell_value_1944", "munsell_value_1955",
                "LIGHTNESS_FUNCTIONS", "MUNSELL_VALUE_FUNCTIONS"])

# *color.spectrum* objects.
__all__.extend(["blackbody_spectral_power_distribution", "blackbody_spectral_radiance", "planck_law",
                "CIE_RGB_CMFS", "CMFS", "STANDARD_OBSERVERS_CMFS",
                "BABELCOLOR_AVERAGE_SPDS", "COLORCHECKER_N_OHTA_SPDS", "COLORCHECKERS_SPDS",
                "bandpass_correction", "bandpass_correction_stearns",
                "D_illuminant_relative_spectral_power_distribution",
                "D_ILLUMINANTS_S_SPDS", "ILLUMINANTS_RELATIVE_SPDS",
                "PHOTOPIC_LEFS", "SCOTOPIC_LEFS",
                "SpectralPowerDistribution",
                "AbstractColorMatchingFunctions", "RGB_ColorMatchingFunctions", "XYZ_ColorMatchingFunctions",
                "TCS_SPDS",
                "spectral_to_XYZ", "wavelength_to_XYZ"])

# *color.temperature* objects.
__all__.extend(["CCT_to_uv", "CCT_to_uv_ohno", "CCT_to_uv_robertson",
                "uv_to_CCT", "uv_to_CCT_ohno", "uv_to_CCT_robertson",
                "D_illuminant_CCT_to_xy", ])

# *color.transformations* objects.
__all__.extend(["XYZ_to_xyY", "xyY_to_XYZ", "xyY_to_RGB", "RGB_to_xyY",
                "xy_to_XYZ", "XYZ_to_xy",
                "XYZ_to_RGB", "RGB_to_XYZ",
                "XYZ_to_UCS", "UCS_to_XYZ", "UCS_to_uv", "UCS_uv_to_xy",
                "XYZ_to_UVW",
                "XYZ_to_Luv", "Luv_to_XYZ", "Luv_to_uv", "Luv_uv_to_xy", "Luv_to_LCHuv", "LCHuv_to_Luv",
                "XYZ_to_Lab", "Lab_to_XYZ", "Lab_to_LCHab", "LCHab_to_Lab"])

LOGGER = install_logger()

get_logging_console_handler()
set_verbosity_level(Constants.verbosity_level)

__all__.extend("LOGGER")

__all__ = map(lambda x: x.encode("ascii"), __all__)

__version__ = Constants.version
