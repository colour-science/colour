#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines units tests for :mod:`colour.adaptation.cat` module.
"""

from __future__ import unicode_literals

import sys
import numpy as np

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

from colour.adaptation import get_chromatic_adaptation_matrix

__author__ = "Colour Developers"
__copyright__ = "Copyright (C) 2013 - 2014 - Colour Developers"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Colour Developers"
__email__ = "colour-science@googlegroups.com"
__status__ = "Production"

__all__ = ["TestGetChromaticAdaptationMatrix"]


class TestGetChromaticAdaptationMatrix(unittest.TestCase):
    """
    Defines :func:`colour.adaptation.cat.get_chromatic_adaptation_matrix`
    definition units tests methods.
    """

    def test_get_chromatic_adaptation_matrix(self):
        """
        Tests :func:`colour.adaptation.cat.get_chromatic_adaptation_matrix`
        definition.
        """

        np.testing.assert_almost_equal(
            get_chromatic_adaptation_matrix(
                np.array([1.09923822, 1.000, 0.35445412]),
                np.array([0.96907232, 1.000, 1.121792157])),
            np.array([0.87145615, -0.13204674, 0.40394832,
                      -0.09638805, 1.04909781, 0.1604033,
                      0.0080207, 0.02826367, 3.06023194]).reshape((3, 3)),
            decimal=7)

        np.testing.assert_almost_equal(
            get_chromatic_adaptation_matrix(
                np.array([1.92001986, 1., -0.1241347]),
                np.array([1.0131677, 1.000, 2.11217686])),
            np.array([0.91344833, -1.20588903, -3.74768526,
                      -0.81680514, 2.3858187, -1.46988227,
                      -0.05367575, -0.31122239, -20.35255049]).reshape((3, 3)),
            decimal=7)

        np.testing.assert_almost_equal(
            get_chromatic_adaptation_matrix(
                np.array([1.92001986, 1., -0.1241347]),
                np.array([1.0131677, 1.000, 2.11217686])),
            np.linalg.inv(get_chromatic_adaptation_matrix(
                np.array([1.0131677, 1.000, 2.11217686]),
                np.array([1.92001986, 1., -0.1241347]))))

        np.testing.assert_almost_equal(
            get_chromatic_adaptation_matrix(
                np.array([1.09850, 1.00000, 0.35585]),
                np.array([0.99072, 1.00000, 0.85223]),
                method="XYZ Scaling"),
            np.array([0.90188439, 0., 0.,
                      0., 1., 0.,
                      0., 0., 2.39491359]).reshape((3, 3)),
            decimal=7)

        np.testing.assert_almost_equal(
            get_chromatic_adaptation_matrix(
                np.array([1.09850, 1.00000, 0.35585]),
                np.array([0.99072, 1.00000, 0.85223]),
                method="Bradford"),
            np.array([0.89051629, -0.08291357, 0.26809449,
                      -0.09715236, 1.07542618, 0.08794629,
                      0.05389701, -0.09085576, 2.48385527]).reshape((3, 3)),
            decimal=7)

        np.testing.assert_almost_equal(
            get_chromatic_adaptation_matrix(
                np.array([1.09850, 1.00000, 0.35585]),
                np.array([0.99072, 1.00000, 0.85223]),
                method="Von Kries"),
            np.array([0.9574884, -0.16436134, 0.29023559,
                      -0.01805393, 1.01853791, 0.00363729,
                      0., 0., 2.39491359]).reshape((3, 3)),
            decimal=7)


if __name__ == "__main__":
    unittest.main()
