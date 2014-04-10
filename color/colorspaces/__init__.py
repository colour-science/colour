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
#***	External imports.
#**********************************************************************************************************************
import inspect

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
from color.colorspaces.cieRgb import *
from color.colorspaces.dciP3 import *
from color.colorspaces.pointerGamut import *
from color.colorspaces.proPhotoRgb import *
from color.colorspaces.rec709 import *
from color.colorspaces.sRgb import *

from color.colorspaces import colorspace
from color.colorspaces import acesRgb
from color.colorspaces import adobeRgb1998
from color.colorspaces import cieRgb
from color.colorspaces import dciP3
from color.colorspaces import pointerGamut
from color.colorspaces import proPhotoRgb
from color.colorspaces import rec709
from color.colorspaces import sRgb

__all__.extend(colorspace.__all__)
__all__.extend(acesRgb.__all__)
__all__.extend(adobeRgb1998.__all__)
__all__.extend(cieRgb.__all__)
__all__.extend(dciP3.__all__)
__all__.extend(pointerGamut.__all__)
__all__.extend(proPhotoRgb.__all__)
__all__.extend(rec709.__all__)
__all__.extend(sRgb.__all__)

# Oddly need to convert from *unicode* to *str* here.
__all__ = map(str, __all__)

COLORSPACES = {"ACES RGB": ACES_RGB_COLORSPACE,
			   "Adobe RGB 1998": ADOBE_RGB_1998_COLORSPACE,
			   "CIE RGB": CIE_RGB_COLORSPACE,
			   "DCI-P3": DCI_P3_COLORSPACE,
			   "ProPhoto RGB": PROPHOTO_RGB_COLORSPACE,
			   "sRGB": sRGB_COLORSPACE,
			   "Rec. 709": REC_709_COLORSPACE}