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

__all__ = ["install_logger",
		   "get_logging_console_handler",
		   "set_verbosity_level"]

#**********************************************************************************************************************
#***    Module classes and definitions.
#**********************************************************************************************************************
install_logger = foundations.verbose.install_logger
get_logging_console_handler = foundations.verbose.get_logging_console_handler
set_verbosity_level = foundations.verbose.set_verbosity_level