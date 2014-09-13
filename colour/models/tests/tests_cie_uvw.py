#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.models.cie_uvw` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

from colour.models import XYZ_to_UVW

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestXYZ_to_UVW']


class TestXYZ_to_UVW(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_uvw.XYZ_to_UVW` definition unit tests
    methods.
    """

    def test_XYZ_to_UVW(self):
        """
        Tests :func:`colour.models.cie_uvw.XYZ_to_UVW` definition.
        """

        np.testing.assert_almost_equal(
            XYZ_to_UVW(np.array([0.07049534, 0.1008, 0.09558313]) * 100),
            np.array([-28.0483277, -0.88052424, 37.00411491]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_UVW(np.array([0.4709771, 0.3495, 0.11301649]) * 100),
            np.array([85.92692334, 17.74352405, 64.73769793]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_UVW(np.array([0.25506814, 0.1915, 0.08849752]) * 100),
            np.array([59.36088697, 8.5919153, 49.88513399]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_UVW(np.array([0.07049534, 0.1008, 0.09558313]) * 100,
                       (0.44757, 0.40745)),
            np.array([-50.56405108, -12.4960054, 37.00411491]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_UVW(np.array([0.07049534, 0.1008, 0.09558313]) * 100,
                       (0.31271, 0.32902)),
            np.array([-22.59813763, 5.45115077, 37.00411491]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_UVW(np.array([0.07049534, 0.1008, 0.09558313]) * 100,
                       (0.37208, 0.37529)),
            np.array([-33.35371445, -4.60753245, 37.00411491]),
            decimal=7)


if __name__ == '__main__':
    unittest.main()
