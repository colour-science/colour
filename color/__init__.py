#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**__init__.py**

**Platform:**
	Windows, Linux, Mac Os X.

**Description:**
	Defines **Color** package initialisation.

**Others:**

"""

#**********************************************************************************************************************
#***	Future imports.
#**********************************************************************************************************************
from __future__ import unicode_literals

#**********************************************************************************************************************
#***	Internal imports.
#**********************************************************************************************************************
import foundations.globals.constants
from globals.constants import Constants

#**********************************************************************************************************************
#***	Dependencies globals manipulation.
#**********************************************************************************************************************
foundations.globals.constants.Constants.__dict__.update(Constants.__dict__)

#**********************************************************************************************************************
#***	Internal imports.
#**********************************************************************************************************************
from color.verbose import *

#**********************************************************************************************************************
#***	Module attributes.
#**********************************************************************************************************************
__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["LOGGER"]

LOGGER = installLogger()

getLoggingConsoleHandler()
setVerbosityLevel(Constants.verbosityLevel)

#**********************************************************************************************************************
#***	Internal imports.
#**********************************************************************************************************************
from blackbody import *
from chromaticAdaptation import *
from colorCheckers import *
from colorspaces import *
from derivation import *
from difference import *
from illuminants import *
from matrix import *
from spectral import *
from temperature import *
from transformations import *