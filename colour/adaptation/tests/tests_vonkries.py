# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.adaptation.vonkries` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

from colour.adaptation import (
    chromatic_adaptation_matrix_VonKries,
    chromatic_adaptation_VonKries)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestChromaticAdaptationMatrixVonKries',
           'TestChromaticAdaptationVonKries']


class TestChromaticAdaptationMatrixVonKries(unittest.TestCase):
    """
    Defines
    :func:`colour.adaptation.vonkries.chromatic_adaptation_matrix_VonKries`
    definition unit tests methods.
    """

    def test_chromatic_adaptation_matrix_VonKries(self):
        """
        Tests
        :func:`colour.adaptation.vonkries.chromatic_adaptation_matrix_VonKries`
        definition.
        """

        np.testing.assert_almost_equal(
            chromatic_adaptation_matrix_VonKries(
                np.array([1.09846607, 1., 0.3558228]),
                np.array([0.95042855, 1., 1.08890037])),
            np.array([[0.86876537, -0.14165393, 0.38719611],
                      [-0.10300724, 1.05840142, 0.15386462],
                      [0.00781674, 0.0267875, 2.96081771]]),
            decimal=7)

        np.testing.assert_almost_equal(
            chromatic_adaptation_matrix_VonKries(
                np.array([0.99092745, 1., 0.85313273]),
                np.array([1.01679082, 1., 0.67610122])),
            np.array([[1.03379528e+00, 3.06532172e-02, -4.48681876e-02],
                      [2.19582633e-02, 9.93543483e-01, -1.79368671e-02],
                      [-1.02726253e-03, -2.81711777e-03, 7.96987686e-01]]),
            decimal=7)

        np.testing.assert_almost_equal(
            chromatic_adaptation_matrix_VonKries(
                np.array([0.98070597, 1., 1.18224949]),
                np.array([0.92833635, 1., 1.0366472])),
            np.linalg.inv(chromatic_adaptation_matrix_VonKries(
                np.array([0.92833635, 1., 1.0366472]),
                np.array([0.98070597, 1., 1.18224949]))))

        np.testing.assert_almost_equal(
            chromatic_adaptation_matrix_VonKries(
                np.array([1.09846607, 1., 0.3558228]),
                np.array([0.95042855, 1., 1.08890037]),
                transform='XYZ Scaling'),
            np.array([[0.86523251, 0., 0.],
                      [0., 1., 0.],
                      [0., 0., 3.06023214]]),
            decimal=7)

        np.testing.assert_almost_equal(
            chromatic_adaptation_matrix_VonKries(
                np.array([1.09846607, 1., 0.3558228]),
                np.array([0.95042855, 1., 1.08890037]),
                transform='Bradford'),
            np.array([[0.84467949, -0.11793553, 0.39489408],
                      [-0.13664085, 1.10412369, 0.12919812],
                      [0.07986716, -0.13493155, 3.19288296]]),
            decimal=7)

        np.testing.assert_almost_equal(
            chromatic_adaptation_matrix_VonKries(
                np.array([1.09846607, 1., 0.3558228]),
                np.array([0.95042855, 1., 1.08890037]),
                transform='Von Kries'),
            np.array([[0.93949221, -0.23393727, 0.42820614],
                      [-0.02569635, 1.02638463, 0.00517656],
                      [0., 0., 3.06023214]]),
            decimal=7)


class TestChromaticAdaptationVonKries(unittest.TestCase):
    """
    Defines :func:`colour.adaptation.vonrkies.chromatic_adaptation_VonKries`
    definition unit tests methods.
    """

    def test_chromatic_adaptation_VonKries(self):
        """
        Tests :func:`colour.adaptation.vonrkies.chromatic_adaptation_VonKries`
        definition.
        """

        np.testing.assert_almost_equal(
            chromatic_adaptation_VonKries(
                np.array([0.07049534, 0.1008, 0.09558313]),
                np.array([1.09846607, 1., 0.3558228]),
                np.array([0.95042855, 1., 1.08890037])),
            np.array([0.08397461, 0.11413219, 0.28625545]),
            decimal=7)

        np.testing.assert_almost_equal(
            chromatic_adaptation_VonKries(
                np.array([0.4709771, 0.3495, 0.11301649]),
                np.array([0.99092745, 1., 0.85313273]),
                np.array([1.01679082, 1., 0.67610122])),
            np.array([0.49253636, 0.35555812, 0.08860435]),
            decimal=7)

        np.testing.assert_almost_equal(
            chromatic_adaptation_VonKries(
                np.array([0.25506814, 0.1915, 0.08849752]),
                np.array([0.98070597, 1., 1.18224949]),
                np.array([0.92833635, 1., 1.0366472])),
            np.array([0.24731314, 0.19137674, 0.07734837]),
            decimal=7)

        np.testing.assert_almost_equal(
            chromatic_adaptation_VonKries(
                np.array([0.07049534, 0.1008, 0.09558313]),
                np.array([1.09846607, 1., 0.3558228]),
                np.array([0.95042855, 1., 1.08890037]),
                transform='XYZ Scaling'),
            np.array([0.06099486, 0.1008, 0.29250657]),
            decimal=7)

        np.testing.assert_almost_equal(
            chromatic_adaptation_VonKries(
                np.array([0.07049534, 0.1008, 0.09558313]),
                np.array([1.09846607, 1., 0.3558228]),
                np.array([0.95042855, 1., 1.08890037]),
                transform='Bradford'),
            np.array([0.08540328, 0.11401229, 0.29721491]),
            decimal=7)

        np.testing.assert_almost_equal(
            chromatic_adaptation_VonKries(
                np.array([0.07049534, 0.1008, 0.09558313]),
                np.array([1.09846607, 1., 0.3558228]),
                np.array([0.95042855, 1., 1.08890037]),
                transform='Von Kries'),
            np.array([0.08357823, 0.10214289, 0.29250657]),
            decimal=7)


if __name__ == '__main__':
    unittest.main()
