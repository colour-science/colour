#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**__init__.py**

**Platform:**
	Windows, Linux, Mac Os X.

**Description:**
	Defines **Color** package *colorspaces* initialisation.

**Others:**

"""

#**********************************************************************************************************************
#***	Future imports.
#**********************************************************************************************************************
from __future__ import unicode_literals

#**********************************************************************************************************************
#***	Internal imports.
#**********************************************************************************************************************
import color.verbose

#**********************************************************************************************************************
#***	Module attributes.
#**********************************************************************************************************************
__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["LOGGER",
		   "COLORSPACES"]

LOGGER = color.verbose.install_logger()

#**********************************************************************************************************************
#***	Internal imports.
#**********************************************************************************************************************
from color.colorspaces.colorspace import *
from color.colorspaces.aces_rgb import *
from color.colorspaces.adobe_rgb_1998 import *
from color.colorspaces.adobe_wide_gamut_rgb import *
from color.colorspaces.alexa_wide_gamut_rgb import *
from color.colorspaces.apple_rgb import *
from color.colorspaces.best_rgb import *
from color.colorspaces.beta_rgb import *
from color.colorspaces.cie_rgb import *
from color.colorspaces.c_log import *
from color.colorspaces.color_match_rgb import *
from color.colorspaces.dci_p3 import *
from color.colorspaces.don_rgb_4 import *
from color.colorspaces.eci_rgb_v2 import *
from color.colorspaces.ekta_space_ps5 import *
from color.colorspaces.max_rgb import *
from color.colorspaces.ntsc_rgb import *
from color.colorspaces.pal_secam_rgb import *
from color.colorspaces.pointer_gamut import *
from color.colorspaces.prophoto_rgb import *
from color.colorspaces.rec_709 import *
from color.colorspaces.rec_2020 import *
from color.colorspaces.russell_rgb import *
from color.colorspaces.s_log import *
from color.colorspaces.smptec_rgb import *
from color.colorspaces.srgb import *
from color.colorspaces.xtreme_rgb import *

from color.colorspaces import colorspace
from color.colorspaces import aces_rgb
from color.colorspaces import adobe_rgb_1998
from color.colorspaces import adobe_wide_gamut_rgb
from color.colorspaces import alexa_wide_gamut_rgb
from color.colorspaces import apple_rgb
from color.colorspaces import best_rgb
from color.colorspaces import beta_rgb
from color.colorspaces import cie_rgb
from color.colorspaces import c_log
from color.colorspaces import color_match_rgb
from color.colorspaces import dci_p3
from color.colorspaces import don_rgb_4
from color.colorspaces import eci_rgb_v2
from color.colorspaces import ekta_space_ps5
from color.colorspaces import max_rgb
from color.colorspaces import ntsc_rgb
from color.colorspaces import pal_secam_rgb
from color.colorspaces import pointer_gamut
from color.colorspaces import prophoto_rgb
from color.colorspaces import rec_709
from color.colorspaces import rec_2020
from color.colorspaces import russell_rgb
from color.colorspaces import s_log
from color.colorspaces import smptec_rgb
from color.colorspaces import srgb
from color.colorspaces import xtreme_rgb

__all__.extend(colorspace.__all__)
__all__.extend(aces_rgb.__all__)
__all__.extend(adobe_rgb_1998.__all__)
__all__.extend(adobe_wide_gamut_rgb.__all__)
__all__.extend(alexa_wide_gamut_rgb.__all__)
__all__.extend(apple_rgb.__all__)
__all__.extend(best_rgb.__all__)
__all__.extend(beta_rgb.__all__)
__all__.extend(cie_rgb.__all__)
__all__.extend(c_log.__all__)
__all__.extend(color_match_rgb.__all__)
__all__.extend(dci_p3.__all__)
__all__.extend(don_rgb_4.__all__)
__all__.extend(eci_rgb_v2.__all__)
__all__.extend(ekta_space_ps5.__all__)
__all__.extend(max_rgb.__all__)
__all__.extend(ntsc_rgb.__all__)
__all__.extend(pal_secam_rgb.__all__)
__all__.extend(pointer_gamut.__all__)
__all__.extend(prophoto_rgb.__all__)
__all__.extend(rec_709.__all__)
__all__.extend(rec_2020.__all__)
__all__.extend(russell_rgb.__all__)
__all__.extend(s_log.__all__)
__all__.extend(smptec_rgb.__all__)
__all__.extend(srgb.__all__)
__all__.extend(xtreme_rgb.__all__)

__all__ = map(str, __all__)

COLORSPACES = {ACES_RGB_COLORSPACE.name: ACES_RGB_COLORSPACE,
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