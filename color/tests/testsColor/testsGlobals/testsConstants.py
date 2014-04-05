#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**testsConstants.py**

**Platform:**
	Windows, Linux, Mac Os X.

**Description:**
	Defines units tests for :mod:`color.globals.constants` module.

**Others:**

"""

#**********************************************************************************************************************
#***	Future imports.
#**********************************************************************************************************************
from __future__ import unicode_literals

#**********************************************************************************************************************
#***	External imports.
#**********************************************************************************************************************
import sys
if sys.version_info[:2] <= (2, 6):
	import unittest2 as unittest
else:
	import unittest

#**********************************************************************************************************************
#***	Internal imports.
#**********************************************************************************************************************
from color.globals.constants import Constants

#**********************************************************************************************************************
#***	Module attributes.
#**********************************************************************************************************************
__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["ConstantsTestCase"]

#**********************************************************************************************************************
#***	Module classes and definitions.
#**********************************************************************************************************************
class ConstantsTestCase(unittest.TestCase):
	"""
	Defines :class:`color.globals.constants.Constants` class units tests methods.
	"""

	def testRequiredAttributes(self):
		"""
		Tests presence of required attributes.
		"""

		requiredAttributes = ("applicationName",
								"majorVersion",
								"minorVersion",
								"changeVersion",
								"version",
								"logger",
								"verbosityLevel",
								"verbosityLabels",
								"loggingDefaultFormatter",
								"loggingSeparators",
								"encodingCodec",
								"encodingError")

		for attribute in requiredAttributes:
			self.assertIn(attribute, Constants.__dict__)

	def testApplicationNameAttribute(self):
		"""
		Tests :attr:`color.globals.constants.Constants.applicationName` attribute.
		"""

		self.assertRegexpMatches(Constants.applicationName, "\w+")

	def testMajorVersionAttribute(self):
		"""
		Tests :attr:`color.globals.constants.Constants.majorVersion` attribute.
		"""

		self.assertRegexpMatches(Constants.version, "\d")

	def testMinorVersionAttribute(self):
		"""
		Tests :attr:`color.globals.constants.Constants.minorVersion` attribute.
		"""

		self.assertRegexpMatches(Constants.version, "\d")

	def testChangeVersionAttribute(self):
		"""
		Tests :attr:`color.globals.constants.Constants.changeVersion` attribute.
		"""

		self.assertRegexpMatches(Constants.version, "\d")

	def testversionAttribute(self):
		"""
		Tests :attr:`color.globals.constants.Constants.version` attribute.
		"""

		self.assertRegexpMatches(Constants.version, "\d\.\d\.\d")

	def testLoggerAttribute(self):
		"""
		Tests :attr:`color.globals.constants.Constants.logger` attribute.
		"""

		self.assertRegexpMatches(Constants.logger, "\w+")

	def testVerbosityLevelAttribute(self):
		"""
		Tests :attr:`color.globals.constants.Constants.verbosityLevel` attribute.
		"""

		self.assertIsInstance(Constants.verbosityLevel, int)
		self.assertGreaterEqual(Constants.verbosityLevel, 0)
		self.assertLessEqual(Constants.verbosityLevel, 4)

	def testVerbosityLabelsAttribute(self):
		"""
		Tests :attr:`color.globals.constants.Constants.verbosityLabels` attribute.
		"""

		self.assertIsInstance(Constants.verbosityLabels, tuple)
		for label in Constants.verbosityLabels:
			self.assertIsInstance(label, unicode)

	def testLoggingDefaultFormatterAttribute(self):
		"""
		Tests :attr:`color.globals.constants.Constants.loggingDefaultFormatter` attribute.
		"""

		self.assertIsInstance(Constants.loggingDefaultFormatter, unicode)

	def testLoggingSeparatorsAttribute(self):
		"""
		Tests :attr:`color.globals.constants.Constants.loggingSeparators` attribute.
		"""

		self.assertIsInstance(Constants.loggingSeparators, unicode)

	def testEncodingCodecAttribute(self):
		"""
		Tests :attr:`color.globals.constants.Constants.encodingCodec` attribute.
		"""

		validEncodings = ("ascii",
						"utf-8",
						"cp1252")

		self.assertIn(Constants.encodingCodec, validEncodings)

	def testEncodingErrorAttribute(self):
		"""
		Tests :attr:`color.globals.constants.Constants.encodingError` attribute.
		"""

		validEncodings = ("strict",
						"ignore",
						"replace",
						"xmlcharrefreplace")

		self.assertIn(Constants.encodingError, validEncodings)

if __name__ == "__main__":
	unittest.main()
