# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**tests_interpolation.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines units tests for :mod:`color.algebra.interpolation` module.

**Others:**

"""

from __future__ import unicode_literals

import sys
import numpy

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

from color.algebra.interpolation import SpragueInterpolator

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["POINTS_DATA_A",
           "INTERPOLATED_POINTS_DATA_A_10_SAMPLES",
           "TestSpragueInterpolator"]

POINTS_DATA_A = numpy.array([
    9.3700,
    12.3200,
    12.4600,
    9.5100,
    5.9200,
    4.3300,
    4.2900,
    3.8800,
    4.5100,
    10.9200,
    27.5000,
    49.6700,
    69.5900,
    81.7300,
    88.1900,
    86.0500])

INTERPOLATED_POINTS_DATA_A_10_SAMPLES = [
    9.37,
    9.72075073,
    10.06936191,
    10.4114757,
    10.7430227,
    11.06022653,
    11.35960827,
    11.637991,
    11.89250427,
    12.1205886,
    12.32,
    12.48873542,
    12.62489669,
    12.7270653,
    12.79433478,
    12.82623598,
    12.82266243,
    12.78379557,
    12.71003009,
    12.60189921,
    12.46,
    12.28440225,
    12.074048,
    11.829765,
    11.554432,
    11.25234375,
    10.928576,
    10.5883505,
    10.2364,
    9.87633325,
    9.51,
    9.13692962,
    8.756208,
    8.36954763,
    7.980976,
    7.59601562,
    7.220864,
    6.86157362,
    6.523232,
    6.20914162,
    5.92,
    5.654602,
    5.414496,
    5.20073875,
    5.012944,
    4.8496875,
    4.708912,
    4.58833225,
    4.48584,
    4.399909,
    4.33,
    4.27757887,
    4.245952,
    4.23497388,
    4.240992,
    4.25804688,
    4.279072,
    4.29709387,
    4.306432,
    4.30389887,
    4.29,
    4.26848387,
    4.240432,
    4.20608887,
    4.166032,
    4.12117188,
    4.072752,
    4.02234887,
    3.971872,
    3.92356387,
    3.88,
    3.84319188,
    3.813184,
    3.79258487,
    3.786912,
    3.80367187,
    3.85144,
    3.93894087,
    4.074128,
    4.26326387,
    4.51,
    4.81362075,
    5.170288,
    5.5822515,
    6.05776,
    6.60890625,
    7.249472,
    7.992773,
    8.849504,
    9.82558375,
    10.92,
    12.12700944,
    13.448928,
    14.88581406,
    16.432832,
    18.08167969,
    19.822016,
    21.64288831,
    23.53416,
    25.48793794,
    27.5,
    29.57061744,
    31.699648,
    33.88185481,
    36.107776,
    38.36511719,
    40.640144,
    42.91907456,
    45.189472,
    47.44163694,
    49.67,
    51.87389638,
    54.052736,
    56.20157688,
    58.311984,
    60.37335938,
    62.374272,
    64.30378787,
    66.1528,
    67.91535838,
    69.59,
    71.17616669,
    72.662832,
    74.04610481,
    75.331712,
    76.53183594,
    77.661952,
    78.73766606,
    79.771552,
    80.76998919,
    81.73,
    82.64375688,
    83.51935227,
    84.35919976,
    85.15567334,
    85.89451368,
    86.55823441,
    87.12952842,
    87.59467414,
    87.94694187,
    88.19,
    88.33345751,
    88.37111372,
    88.30221714,
    88.13600972,
    87.88846516,
    87.57902706,
    87.2273472,
    86.85002373,
    86.45733945,
    86.05]


class TestSpragueInterpolator(unittest.TestCase):
    """
    Defines :func:`color.algebra.interpolation.SpragueInterpolator` class units tests methods.
    """

    def test_required_attributes(self):
        """
        Tests presence of required attributes.
        """

        required_attributes = ("x",
                               "y")

        for attribute in required_attributes:
            self.assertIn(attribute, dir(SpragueInterpolator))

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ()

        for method in required_methods:
            self.assertIn(method, dir(SpragueInterpolator))

    def test___call__(self):
        """
        Tests :func:`color.algebra.interpolation.SpragueInterpolator.__call__` method.
        """

        steps = 0.1
        x = numpy.arange(len(POINTS_DATA_A))
        sprague_interpolator = SpragueInterpolator(x, POINTS_DATA_A)

        for i, value in enumerate(numpy.arange(0, len(POINTS_DATA_A) - 1 + steps, steps)):
            self.assertAlmostEqual(INTERPOLATED_POINTS_DATA_A_10_SAMPLES[i], sprague_interpolator(value), places=7)

        numpy.testing.assert_almost_equal(sprague_interpolator(numpy.arange(0, len(POINTS_DATA_A) - 1 + steps, steps)),
                                          INTERPOLATED_POINTS_DATA_A_10_SAMPLES)


if __name__ == "__main__":
    unittest.main()
