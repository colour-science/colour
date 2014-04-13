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

LOGGER = color.verbose.installLogger()

#**********************************************************************************************************************
#***	Internal imports.
#**********************************************************************************************************************
from color.colorspaces.colorspace import *
from color.colorspaces.acesRgb import *
from color.colorspaces.adobeRgb1998 import *
from color.colorspaces.adobeWideGamutRgb import *
from color.colorspaces.appleRgb import *
from color.colorspaces.bestRgb import *
from color.colorspaces.betaRgb import *
from color.colorspaces.cieRgb import *
from color.colorspaces.colorMatchRgb import *
from color.colorspaces.dciP3 import *
from color.colorspaces.donRgb4 import *
from color.colorspaces.eciRgbV2 import *
from color.colorspaces.ektaSpacePs5 import *
from color.colorspaces.maxRgb import *
from color.colorspaces.ntscRgb import *
from color.colorspaces.palSecamRgb import *
from color.colorspaces.pointerGamut import *
from color.colorspaces.proPhotoRgb import *
from color.colorspaces.rec709 import *
from color.colorspaces.russellRgb import *
from color.colorspaces.smptecRgb import *
from color.colorspaces.sRgb import *
from color.colorspaces.xtremeRgb import *

from color.colorspaces import colorspace
from color.colorspaces import acesRgb
from color.colorspaces import adobeRgb1998
from color.colorspaces import adobeWideGamutRgb
from color.colorspaces import appleRgb
from color.colorspaces import bestRgb
from color.colorspaces import betaRgb
from color.colorspaces import cieRgb
from color.colorspaces import colorMatchRgb
from color.colorspaces import dciP3
from color.colorspaces import donRgb4
from color.colorspaces import eciRgbV2
from color.colorspaces import ektaSpacePs5
from color.colorspaces import maxRgb
from color.colorspaces import ntscRgb
from color.colorspaces import palSecamRgb
from color.colorspaces import pointerGamut
from color.colorspaces import proPhotoRgb
from color.colorspaces import rec709
from color.colorspaces import russellRgb
from color.colorspaces import smptecRgb
from color.colorspaces import sRgb
from color.colorspaces import xtremeRgb

__all__.extend(colorspace.__all__)
__all__.extend(acesRgb.__all__)
__all__.extend(adobeRgb1998.__all__)
__all__.extend(adobeWideGamutRgb.__all__)
__all__.extend(appleRgb.__all__)
__all__.extend(bestRgb.__all__)
__all__.extend(betaRgb.__all__)
__all__.extend(cieRgb.__all__)
__all__.extend(colorMatchRgb.__all__)
__all__.extend(dciP3.__all__)
__all__.extend(donRgb4.__all__)
__all__.extend(eciRgbV2.__all__)
__all__.extend(ektaSpacePs5.__all__)
__all__.extend(maxRgb.__all__)
__all__.extend(ntscRgb.__all__)
__all__.extend(palSecamRgb.__all__)
__all__.extend(pointerGamut.__all__)
__all__.extend(proPhotoRgb.__all__)
__all__.extend(rec709.__all__)
__all__.extend(russellRgb.__all__)
__all__.extend(smptecRgb.__all__)
__all__.extend(sRgb.__all__)
__all__.extend(xtremeRgb.__all__)

__all__ = map(str, __all__)

COLORSPACES = {ACES_RGB_COLORSPACE.name: ACES_RGB_COLORSPACE,
			   ADOBE_RGB_1998_COLORSPACE.name: ADOBE_RGB_1998_COLORSPACE,
			   ADOBE_WIDE_GAMUT_RGB_COLORSPACE.name: ADOBE_WIDE_GAMUT_RGB_COLORSPACE,
			   APPLE_RGB_COLORSPACE.name: APPLE_RGB_COLORSPACE,
			   BEST_RGB_COLORSPACE.name: BEST_RGB_COLORSPACE,
			   BETA_RGB_COLORSPACE.name: BETA_RGB_COLORSPACE,
			   CIE_RGB_COLORSPACE.name: CIE_RGB_COLORSPACE,
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
			   RUSSELL_RGB_COLORSPACE.name: RUSSELL_RGB_COLORSPACE,
			   SMPTE_C_RGB_COLORSPACE.name: SMPTE_C_RGB_COLORSPACE,
			   sRGB_COLORSPACE.name: sRGB_COLORSPACE,
			   XTREME_RGB_COLORSPACE.name: XTREME_RGB_COLORSPACE}