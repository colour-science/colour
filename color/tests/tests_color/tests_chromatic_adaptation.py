#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**tests_chromatic_adaptation.py**

**Platform:**
	Windows, Linux, Mac Os X.

**Description:**
	Defines units tests for :mod:`color.chromatic_adaptation` module.

**Others:**

"""

#**********************************************************************************************************************
#***	Future imports.
#**********************************************************************************************************************
from __future__ import unicode_literals

#**********************************************************************************************************************
#***	External imports.
#**********************************************************************************************************************
import numpy
import sys

if sys.version_info[:2] <= (2, 6):
	import unittest2 as unittest
else:
	import unittest

#**********************************************************************************************************************
#***	Internal imports.
#**********************************************************************************************************************
import color.chromatic_adaptation

#**********************************************************************************************************************
#***	Module attributes.
#**********************************************************************************************************************
__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["TestGetChromaticAdaptationMatrix"]

#**********************************************************************************************************************
#***	Module classes and definitions.
#**********************************************************************************************************************
class TestGetChromaticAdaptationMatrix(unittest.TestCase):
	"""
	Defines :func:`color.chromatic_adaptation.get_chromatic_adaptation_matrix` definition units tests methods.
	"""

	def test_get_chromatic_adaptation_matrix(self):
		"""
		Tests :func:`color.chromatic_adaptation.get_chromatic_adaptation_matrix` definition.
		"""

		numpy.testing.assert_almost_equal(
			color.chromatic_adaptation.get_chromatic_adaptation_matrix(numpy.matrix([1.09923822,
																				 1.000,
																				 0.35445412]).reshape((3, 1)),
																   numpy.matrix([0.96907232,
																				 1.000,
																				 1.121792157]).reshape((3, 1))),
			numpy.matrix([0.87145615, -0.13204674, 0.40394832,
						  -0.09638805, 1.04909781, 0.1604033,
						  0.0080207, 0.02826367, 3.06023194]).reshape((3, 3)),
			decimal=7)

		numpy.testing.assert_almost_equal(
			color.chromatic_adaptation.get_chromatic_adaptation_matrix(numpy.matrix([1.92001986,
																				 1.,
																				 -0.1241347]).reshape((3, 1)),
																   numpy.matrix([1.0131677,
																				 1.000,
																				 2.11217686]).reshape((3, 1))),
			numpy.matrix([0.91344833, -1.20588903, -3.74768526,
						  -0.81680514, 2.3858187, -1.46988227,
						  -0.05367575, -0.31122239, -20.35255049]).reshape((3, 3)),
			decimal=7)

		numpy.testing.assert_almost_equal(
			color.chromatic_adaptation.get_chromatic_adaptation_matrix(numpy.matrix([1.92001986,
																				 1.,
																				 -0.1241347]).reshape((3, 1)),
																   numpy.matrix([1.0131677,
																				 1.000,
																				 2.11217686]).reshape((3, 1)), ),
			color.chromatic_adaptation.get_chromatic_adaptation_matrix(numpy.matrix([1.0131677,
																				 1.000,
																				 2.11217686]).reshape((3, 1)),
																   numpy.matrix([1.92001986,
																				 1.,
																				 -0.1241347]).reshape((3, 1))).getI())

	numpy.testing.assert_almost_equal(
		color.chromatic_adaptation.get_chromatic_adaptation_matrix(numpy.matrix([1.09850,
																			 1.00000,
																			 0.35585]).reshape((3, 1)),
															   numpy.matrix([0.99072,
																			 1.00000,
																			 0.85223]).reshape((3, 1)),
															   method="XYZ Scaling"),
		numpy.matrix([0.90188439, 0., 0.,
					  0., 1., 0.,
					  0., 0., 2.39491359]).reshape((3, 3)),
		decimal=7)

	numpy.testing.assert_almost_equal(
		color.chromatic_adaptation.get_chromatic_adaptation_matrix(numpy.matrix([1.09850,
																			 1.00000,
																			 0.35585]).reshape((3, 1)),
															   numpy.matrix([0.99072,
																			 1.00000,
																			 0.85223]).reshape((3, 1)),
															   method="Bradford"),
		numpy.matrix([0.89051629, -0.08291357, 0.26809449,
					  -0.09715236, 1.07542618, 0.08794629,
					  0.05389701, -0.09085576, 2.48385527]).reshape((3, 3)),
		decimal=7)

	numpy.testing.assert_almost_equal(
		color.chromatic_adaptation.get_chromatic_adaptation_matrix(numpy.matrix([1.09850,
																			 1.00000,
																			 0.35585]).reshape((3, 1)),
															   numpy.matrix([0.99072,
																			 1.00000,
																			 0.85223]).reshape((3, 1)),
															   method="Von Kries"),
		numpy.matrix([0.9574884, -0.16436134, 0.29023559,
					  -0.01805393, 1.01853791, 0.00363729,
					  0., 0., 2.39491359]).reshape((3, 3)),
		decimal=7)

if __name__ == "__main__":
	unittest.main()
