#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**verbose.py**

**Platform:**
	Windows, Linux, Mac Os X.

**Description:**
	Defines **Color** package verbose and logging objects.

**Others:**

"""

#**********************************************************************************************************************
#***	Future imports.
#**********************************************************************************************************************
from __future__ import unicode_literals

#**********************************************************************************************************************
#***    Internal imports.
#**********************************************************************************************************************
import foundations.verbose

#**********************************************************************************************************************
#***	Module attributes.
#**********************************************************************************************************************
__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["installLogger",
		   "getLoggingConsoleHandler",
		   "setVerbosityLevel"]

#**********************************************************************************************************************
#***    Module classes and definitions.
#**********************************************************************************************************************
installLogger = foundations.verbose.installLogger
getLoggingConsoleHandler = foundations.verbose.getLoggingConsoleHandler
setVerbosityLevel = foundations.verbose.setVerbosityLevel