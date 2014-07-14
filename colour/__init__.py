# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**__init__.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package initialisation.

**Others:**

"""

from __future__ import unicode_literals

import foundations.globals.constants

from colour.globals.constants import Constants


foundations.globals.constants.Constants.__dict__.update(Constants.__dict__)

from colour.algebra.common import get_closest, get_steps, is_uniform, is_iterable, to_ndarray
from colour.algebra.extrapolation import Extrapolator1d
from colour.algebra.interpolation import LinearInterpolator, SpragueInterpolator
from colour.algebra.matrix import is_identity, linear_interpolate_matrices
from colour.algebra.regression import linear_regression

from colour.computation.blackbody import blackbody_spectral_power_distribution, blackbody_spectral_radiance, planck_law
from colour.computation.chromatic_adaptation import CHROMATIC_ADAPTATION_METHODS
from colour.computation.chromatic_adaptation import get_chromatic_adaptation_matrix
from colour.computation.cmfs import LMS_ConeFundamentals, RGB_ColourMatchingFunctions, XYZ_ColourMatchingFunctions
from colour.computation.cmfs import RGB_10_degree_cmfs_to_LMS_10_degree_cmfs
from colour.computation.cmfs import RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs
from colour.computation.cmfs import RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs
from colour.computation.cmfs import LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs
from colour.computation.cmfs import LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs
from colour.computation.colourspaces.rgb.colourspace import Colourspace
from colour.computation.colourspaces.rgb.derivation import get_normalised_primary_matrix
from colour.computation.correction import bandpass_correction, bandpass_correction_stearns
from colour.computation.cri import get_colour_rendering_index
from colour.computation.difference import delta_E_CIE_1976, delta_E_CIE_1994, delta_E_CIE_2000, delta_E_CMC
from colour.computation.illuminants import D_illuminant_relative_spd
from colour.computation.lefs import mesopic_luminous_efficiency_function, mesopic_weighting_function
from colour.computation.lightness import get_lightness, get_luminance, get_luminance_equation, get_munsell_value
from colour.computation.lightness import lightness_1958, lightness_1964, lightness_1976
from colour.computation.lightness import luminance_1943, luminance_1976
from colour.computation.lightness import munsell_value_1920, munsell_value_1933, munsell_value_1943, munsell_value_1944, \
    munsell_value_1955
from colour.computation.lightness import LIGHTNESS_FUNCTIONS, MUNSELL_VALUE_FUNCTIONS
from colour.computation.spectrum import SpectralPowerDistribution, TriSpectralPowerDistribution
from colour.computation.temperature import CCT_to_uv, CCT_to_uv_ohno, CCT_to_uv_robertson
from colour.computation.temperature import uv_to_CCT, uv_to_CCT_ohno, uv_to_CCT_robertson
from colour.computation.temperature import CCT_to_xy, CCT_to_xy_kim, CCT_to_xy_illuminant_D
from colour.computation.temperature import xy_to_CCT, xy_to_CCT_mccamy, xy_to_CCT_romero
from colour.computation.transformations import XYZ_to_xyY, xyY_to_XYZ, xyY_to_RGB, RGB_to_xyY
from colour.computation.transformations import xy_to_XYZ, XYZ_to_xy
from colour.computation.transformations import XYZ_to_RGB, RGB_to_XYZ
from colour.computation.transformations import XYZ_to_UCS, UCS_to_XYZ, UCS_to_uv, UCS_uv_to_xy
from colour.computation.transformations import XYZ_to_UVW
from colour.computation.transformations import XYZ_to_Luv, Luv_to_XYZ, Luv_to_uv, Luv_uv_to_xy, Luv_to_LCHuv, \
    LCHuv_to_Luv
from colour.computation.transformations import XYZ_to_Lab, Lab_to_XYZ, Lab_to_LCHab, LCHab_to_Lab
from colour.computation.tristimulus import spectral_to_XYZ, wavelength_to_XYZ

from colour.dataset.cmfs import CMFS, LMS_CMFS, RGB_CMFS, STANDARD_OBSERVERS_CMFS
from colour.dataset.colour_checkers.chromaticity_coordinates import COLOURCHECKERS
from colour.dataset.colour_checkers.spds import COLOURCHECKERS_SPDS
from colour.dataset.colourspaces.rgb.aces_rgb import ACES_RGB_COLOURSPACE, ACES_RGB_LOG_COLOURSPACE
from colour.dataset.colourspaces.rgb.aces_rgb import ACES_RGB_PROXY_10_COLOURSPACE, ACES_RGB_PROXY_12_COLOURSPACE
from colour.dataset.colourspaces.rgb.adobe_rgb_1998 import ADOBE_RGB_1998_COLOURSPACE
from colour.dataset.colourspaces.rgb.adobe_wide_gamut_rgb import ADOBE_WIDE_GAMUT_RGB_COLOURSPACE
from colour.dataset.colourspaces.rgb.alexa_wide_gamut_rgb import ALEXA_WIDE_GAMUT_RGB_COLOURSPACE
from colour.dataset.colourspaces.rgb.apple_rgb import APPLE_RGB_COLOURSPACE
from colour.dataset.colourspaces.rgb.best_rgb import BEST_RGB_COLOURSPACE
from colour.dataset.colourspaces.rgb.best_rgb import BEST_RGB_COLOURSPACE
from colour.dataset.colourspaces.rgb.beta_rgb import BETA_RGB_COLOURSPACE
from colour.dataset.colourspaces.rgb.c_log import C_LOG_COLOURSPACE
from colour.dataset.colourspaces.rgb.cie_rgb import CIE_RGB_COLOURSPACE
from colour.dataset.colourspaces.rgb.color_match_rgb import COLOR_MATCH_RGB_COLOURSPACE
from colour.dataset.colourspaces.rgb.dci_p3 import DCI_P3_COLOURSPACE
from colour.dataset.colourspaces.rgb.don_rgb_4 import DON_RGB_4_COLOURSPACE
from colour.dataset.colourspaces.rgb.eci_rgb_v2 import ECI_RGB_V2_COLOURSPACE
from colour.dataset.colourspaces.rgb.ekta_space_ps5 import EKTA_SPACE_PS_5_COLOURSPACE
from colour.dataset.colourspaces.rgb.max_rgb import MAX_RGB_COLOURSPACE
from colour.dataset.colourspaces.rgb.ntsc_rgb import NTSC_RGB_COLOURSPACE
from colour.dataset.colourspaces.rgb.pal_secam_rgb import PAL_SECAM_RGB_COLOURSPACE
from colour.dataset.colourspaces.rgb.pointer_gamut import POINTER_GAMUT_DATA
from colour.dataset.colourspaces.rgb.prophoto_rgb import PROPHOTO_RGB_COLOURSPACE
from colour.dataset.colourspaces.rgb.rec_709 import REC_709_COLOURSPACE
from colour.dataset.colourspaces.rgb.rec_2020 import REC_2020_COLOURSPACE
from colour.dataset.colourspaces.rgb.russell_rgb import RUSSELL_RGB_COLOURSPACE
from colour.dataset.colourspaces.rgb.s_log import S_LOG_COLOURSPACE
from colour.dataset.colourspaces.rgb.smptec_rgb import SMPTE_C_RGB_COLOURSPACE
from colour.dataset.colourspaces.rgb.srgb import sRGB_COLOURSPACE
from colour.dataset.colourspaces.rgb.xtreme_rgb import XTREME_RGB_COLOURSPACE
from colour.dataset.illuminants.chromaticity_coordinates import ILLUMINANTS
from colour.dataset.illuminants.d_illuminants_s_spds import D_ILLUMINANTS_S_SPDS
from colour.dataset.illuminants.spds import ILLUMINANTS_RELATIVE_SPDS
from colour.dataset.lefs import LEFS, PHOTOPIC_LEFS, SCOTOPIC_LEFS
from colour.dataset.tcs import TCS_SPDS

from colour.implementation.fitting import first_order_colour_fit

from colour.utilities.verbose import get_logging_console_handler, install_logger, set_verbosity_level

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["Constants"]

# *colour.algebra* objects.
__all__.extend(["get_closest", "get_steps", "is_uniform", "is_iterable", "to_ndarray",
                "Extrapolator1d",
                "LinearInterpolator", "SpragueInterpolator",
                "is_identity", "linear_interpolate_matrices",
                "linear_regression"])

# *colour.computation.blackbody* objects.
__all__.extend(["blackbody_spectral_power_distribution", "blackbody_spectral_radiance", "planck_law"])

# *colour.computation.chromatic_adaptation* objects.
__all__.extend(["CHROMATIC_ADAPTATION_METHODS", "get_chromatic_adaptation_matrix"])

# *colour.computation.cmfs* objects.
__all__.extend(["LMS_ConeFundamentals", "RGB_ColourMatchingFunctions", "XYZ_ColourMatchingFunctions"])
__all__.extend(["RGB_10_degree_cmfs_to_LMS_10_degree_cmfs",
                "RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs",
                "RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs",
                "LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs",
                "LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs"])

# *colour.computation.colourspaces.rgb.colourspace* objects.
__all__.extend(["Colourspace"])

# *colour.computation.colourspaces.rgb.derivation* objects.
__all__.extend(["get_normalised_primary_matrix"])

# *colour.computation.correction* objects.
__all__.extend(["bandpass_correction", "bandpass_correction_stearns"])

# *colour.computation.cri* objects.
__all__.extend(["get_colour_rendering_index"])

# *colour.computation.difference* objects.
__all__.extend(["delta_E_CIE_1976", "delta_E_CIE_1994", "delta_E_CIE_2000", "delta_E_CMC"])

# *colour.computation.illuminants* objects.
__all__.extend(["D_illuminant_relative_spd"])

# *colour.computation.lefs* objects.
__all__.extend(["mesopic_luminous_efficiency_function", "mesopic_weighting_function"])

# *colour.computation.lightness* objects.
__all__.extend(["get_lightness", "get_luminance", "get_luminance_equation", "get_munsell_value"])
__all__.extend(["lightness_1958", "lightness_1964", "lightness_1976"])
__all__.extend(["luminance_1943", "luminance_1976"])
__all__.extend(
    ["munsell_value_1920", "munsell_value_1933", "munsell_value_1943", "munsell_value_1944", "munsell_value_1955"])
__all__.extend(["LIGHTNESS_FUNCTIONS", "MUNSELL_VALUE_FUNCTIONS"])

# *colour.computation.spectrum* objects.
__all__.extend(["SpectralPowerDistribution", "TriSpectralPowerDistribution"])

# *colour.computation.temperature* objects.
__all__.extend(["CCT_to_uv", "CCT_to_uv_ohno", "CCT_to_uv_robertson",
                "uv_to_CCT", "uv_to_CCT_ohno", "uv_to_CCT_robertson",
                "CCT_to_xy", "CCT_to_xy_kim", "CCT_to_xy_illuminant_D",
                "xy_to_CCT", "xy_to_CCT_mccamy", "xy_to_CCT_romero"])

# *colour.computation.transformations* objects.
__all__.extend(["XYZ_to_xyY", "xyY_to_XYZ", "xyY_to_RGB", "RGB_to_xyY",
                "xy_to_XYZ", "XYZ_to_xy",
                "XYZ_to_RGB", "RGB_to_XYZ",
                "XYZ_to_UCS", "UCS_to_XYZ", "UCS_to_uv", "UCS_uv_to_xy",
                "XYZ_to_UVW",
                "XYZ_to_Luv", "Luv_to_XYZ", "Luv_to_uv", "Luv_uv_to_xy", "Luv_to_LCHuv", "LCHuv_to_Luv",
                "XYZ_to_Lab", "Lab_to_XYZ", "Lab_to_LCHab", "LCHab_to_Lab"])

# *colour.computation.tristimulus* objects.
__all__.extend(["spectral_to_XYZ", "wavelength_to_XYZ"])

# *colour.dataset.cmfs* objects.
__all__.extend(["CMFS", "LMS_CMFS", "RGB_CMFS", "STANDARD_OBSERVERS_CMFS"])

# *colour.dataset.colour_checkers.chromaticity_coordinates* objects.
__all__.extend(["COLORCHECKERS"])

# *colour.dataset.colour_checkers.spds* objects.
__all__.extend(["COLORCHECKERS_SPDS"])

# *colour.dataset.rgb.colourspaces* objects.
COLOURSPACES = {ACES_RGB_COLOURSPACE.name: ACES_RGB_COLOURSPACE,
                ACES_RGB_LOG_COLOURSPACE.name: ACES_RGB_LOG_COLOURSPACE,
                ACES_RGB_PROXY_10_COLOURSPACE.name: ACES_RGB_PROXY_10_COLOURSPACE,
                ACES_RGB_PROXY_12_COLOURSPACE.name: ACES_RGB_PROXY_12_COLOURSPACE,
                ADOBE_RGB_1998_COLOURSPACE.name: ADOBE_RGB_1998_COLOURSPACE,
                ADOBE_WIDE_GAMUT_RGB_COLOURSPACE.name: ADOBE_WIDE_GAMUT_RGB_COLOURSPACE,
                ALEXA_WIDE_GAMUT_RGB_COLOURSPACE.name: ALEXA_WIDE_GAMUT_RGB_COLOURSPACE,
                APPLE_RGB_COLOURSPACE.name: APPLE_RGB_COLOURSPACE,
                BEST_RGB_COLOURSPACE.name: BEST_RGB_COLOURSPACE,
                BETA_RGB_COLOURSPACE.name: BETA_RGB_COLOURSPACE,
                CIE_RGB_COLOURSPACE.name: CIE_RGB_COLOURSPACE,
                C_LOG_COLOURSPACE.name: C_LOG_COLOURSPACE,
                COLOR_MATCH_RGB_COLOURSPACE.name: COLOR_MATCH_RGB_COLOURSPACE,
                DCI_P3_COLOURSPACE.name: DCI_P3_COLOURSPACE,
                DON_RGB_4_COLOURSPACE.name: DON_RGB_4_COLOURSPACE,
                ECI_RGB_V2_COLOURSPACE.name: ECI_RGB_V2_COLOURSPACE,
                EKTA_SPACE_PS_5_COLOURSPACE.name: EKTA_SPACE_PS_5_COLOURSPACE,
                MAX_RGB_COLOURSPACE.name: MAX_RGB_COLOURSPACE,
                NTSC_RGB_COLOURSPACE.name: NTSC_RGB_COLOURSPACE,
                PAL_SECAM_RGB_COLOURSPACE.name: PAL_SECAM_RGB_COLOURSPACE,
                PROPHOTO_RGB_COLOURSPACE.name: PROPHOTO_RGB_COLOURSPACE,
                REC_709_COLOURSPACE.name: REC_709_COLOURSPACE,
                REC_2020_COLOURSPACE.name: REC_2020_COLOURSPACE,
                RUSSELL_RGB_COLOURSPACE.name: RUSSELL_RGB_COLOURSPACE,
                S_LOG_COLOURSPACE.name: S_LOG_COLOURSPACE,
                SMPTE_C_RGB_COLOURSPACE.name: SMPTE_C_RGB_COLOURSPACE,
                sRGB_COLOURSPACE.name: sRGB_COLOURSPACE,
                XTREME_RGB_COLOURSPACE.name: XTREME_RGB_COLOURSPACE}

__all__.extend(["ACES_RGB_COLOURSPACE", "ACES_RGB_LOG_COLOURSPACE",
                "ACES_RGB_PROXY_10_COLOURSPACE", "ACES_RGB_PROXY_12_COLOURSPACE",
                "ADOBE_RGB_1998_COLOURSPACE",
                "ADOBE_WIDE_GAMUT_RGB_COLOURSPACE",
                "ALEXA_WIDE_GAMUT_RGB_COLOURSPACE",
                "APPLE_RGB_COLOURSPACE",
                "BEST_RGB_COLOURSPACE",
                "BETA_RGB_COLOURSPACE",
                "C_LOG_COLOURSPACE",
                "CIE_RGB_COLOURSPACE",
                "COLOR_MATCH_RGB_COLOURSPACE",
                "DCI_P3_COLOURSPACE",
                "DON_RGB_4_COLOURSPACE",
                "ECI_RGB_V2_COLOURSPACE",
                "EKTA_SPACE_PS_5_COLOURSPACE",
                "MAX_RGB_COLOURSPACE",
                "NTSC_RGB_COLOURSPACE",
                "PAL_SECAM_RGB_COLOURSPACE",
                "POINTER_GAMUT_DATA",
                "PROPHOTO_RGB_COLOURSPACE",
                "REC_709_COLOURSPACE",
                "REC_2020_COLOURSPACE",
                "RUSSELL_RGB_COLOURSPACE",
                "S_LOG_COLOURSPACE",
                "SMPTE_C_RGB_COLOURSPACE",
                "sRGB_COLOURSPACE",
                "XTREME_RGB_COLOURSPACE",
                "COLOURSPACES"])

# *colour.dataset.illuminants.chromaticity_coordinates* objects.
__all__.extend(["ILLUMINANTS"])

# *colour.dataset.illuminants.d_illuminants_s_spds* objects.
__all__.extend(["D_ILLUMINANTS_S_SPDS"])

# *colour.dataset.illuminants.spds* objects.
__all__.extend(["ILLUMINANTS_RELATIVE_SPDS"])

# *colour.dataset.lefs* objects.
__all__.extend(["LEFS", "PHOTOPIC_LEFS", "SCOTOPIC_LEFS"])

# *colour.dataset.tcs* objects.
__all__.extend(["TCS_SPDS"])

# *colour.implementation* objects.
__all__.extend(["first_order_colour_fit"])

LOGGER = install_logger()

get_logging_console_handler()
set_verbosity_level(Constants.verbosity_level)

__all__.extend("LOGGER")

__all__ = map(lambda x: x.encode("ascii"), __all__)

__version__ = Constants.version