#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**constants.py**

**Platform:**
	Windows, Linux, Mac Os X.

**Description:**
	Defines **Color** package default constants through the :class:`Constants` class.

**Others:**

"""

#**********************************************************************************************************************
#***	Future imports.
#**********************************************************************************************************************
from __future__ import unicode_literals

#**********************************************************************************************************************
#***	Module attributes.
#**********************************************************************************************************************
__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["Constants"]

#**********************************************************************************************************************
#***	Module classes and definitions.
#**********************************************************************************************************************
class Constants():
	"""
	Defines **Color** package default constants.
	"""

	applicationName = "Color"
	"""
	:param applicationName: Package Application name.
	:type applicationName: unicode
	"""
	majorVersion = "0"
	"""
	:param majorVersion: Package major version.
	:type majorVersion: unicode
	"""
	minorVersion = "1"
	"""
	:param minorVersion: Package minor version.
	:type minorVersion: unicode
	"""
	changeVersion = "0"
	"""
	:param changeVersion: Package change version.
	:type changeVersion: unicode
	"""
	version = ".".join((majorVersion, minorVersion, changeVersion))
	"""
	:param version: Package version.
	:type version: unicode
	"""

	logger = "Color_Logger"
	"""
	:param logger: Package logger name.
	:type logger: unicode
	"""
	verbosityLevel = 3
	"""
	:param verbosityLevel: Default logging verbosity level.
	:type verbosityLevel: int
	"""
	verbosityLabels = ("Critical", "Error", "Warning", "Info", "Debug")
	"""
	:param verbosityLabels: Logging verbosity labels.
	:type verbosityLabels: tuple
	"""
	loggingDefaultFormatter = "Default"
	"""
	:param loggingDefaultFormatter: Default logging formatter name.
	:type loggingDefaultFormatter: unicode
	"""
	loggingSeparators = "*" * 96
	"""
	:param loggingSeparators: Logging separators.
	:type loggingSeparators: unicode
	"""

	encodingCodec = "utf-8"
	"""
	:param encodingCodec: Default encoding codec.
	:type encodingCodec: unicode
	"""
	encodingError = "ignore"
	"""
	:param encodingError: Default encoding error behavior.
	:type encodingError: unicode
	"""
