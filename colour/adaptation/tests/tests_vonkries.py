# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.adaptation.vonkries` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from itertools import permutations

from colour.adaptation import (
    chromatic_adaptation_matrix_VonKries,
    chromatic_adaptation_VonKries)
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestChromaticAdaptationMatrixVonKries',
           'TestChromaticAdaptationVonKries']


class TestChromaticAdaptationMatrixVonKries(unittest.TestCase):
    """
    Defines :func:`colour.adaptation.vonkries.\
chromatic_adaptation_matrix_VonKries` definition unit tests methods.
    """

    def test_chromatic_adaptation_matrix_VonKries(self):
        """
        Tests :func:`colour.adaptation.vonkries.\
chromatic_adaptation_matrix_VonKries` definition.
        """

        np.testing.assert_almost_equal(
            chromatic_adaptation_matrix_VonKries(
                np.array([1.09846607, 1.00000000, 0.35582280]),
                np.array([0.95042855, 1.00000000, 1.08890037])),
            np.array([[0.86876537, -0.14165393, 0.38719611],
                      [-0.10300724, 1.05840142, 0.15386462],
                      [0.00781674, 0.02678750, 2.96081771]]),
            decimal=7)

        np.testing.assert_almost_equal(
            chromatic_adaptation_matrix_VonKries(
                np.array([0.99092745, 1.00000000, 0.85313273]),
                np.array([1.01679082, 1.00000000, 0.67610122])),
            np.array([[1.03379528, 0.03065322, -0.04486819],
                      [0.02195826, 0.99354348, -0.01793687],
                      [-0.00102726, -0.00281712, 0.79698769]]),
            decimal=7)

        np.testing.assert_almost_equal(
            chromatic_adaptation_matrix_VonKries(
                np.array([0.98070597, 1.00000000, 1.18224949]),
                np.array([0.92833635, 1.00000000, 1.03664720])),
            np.linalg.inv(chromatic_adaptation_matrix_VonKries(
                np.array([0.92833635, 1.00000000, 1.03664720]),
                np.array([0.98070597, 1.00000000, 1.18224949]))))

        np.testing.assert_almost_equal(
            chromatic_adaptation_matrix_VonKries(
                np.array([1.09846607, 1.00000000, 0.35582280]),
                np.array([0.95042855, 1.00000000, 1.08890037]),
                transform='XYZ Scaling'),
            np.array([[0.86523251, 0.00000000, 0.00000000],
                      [0.00000000, 1.00000000, 0.00000000],
                      [0.00000000, 0.00000000, 3.06023214]]),
            decimal=7)

        np.testing.assert_almost_equal(
            chromatic_adaptation_matrix_VonKries(
                np.array([1.09846607, 1.00000000, 0.35582280]),
                np.array([0.95042855, 1.00000000, 1.08890037]),
                transform='Bradford'),
            np.array([[0.84467949, -0.11793553, 0.39489408],
                      [-0.13664085, 1.10412369, 0.12919812],
                      [0.07986716, -0.13493155, 3.19288296]]),
            decimal=7)

        np.testing.assert_almost_equal(
            chromatic_adaptation_matrix_VonKries(
                np.array([1.09846607, 1.00000000, 0.35582280]),
                np.array([0.95042855, 1.00000000, 1.08890037]),
                transform='Von Kries'),
            np.array([[0.93949221, -0.23393727, 0.42820614],
                      [-0.02569635, 1.02638463, 0.00517656],
                      [0.00000000, 0.00000000, 3.06023214]]),
            decimal=7)

    def test_n_dimensional_chromatic_adaptation_matrix_VonKries(self):
        """
        Tests :func:`colour.adaptation.vonkries.\
chromatic_adaptation_matrix_VonKries` definition n-dimensional arrays support.
        """

        XYZ_w = np.array([1.09846607, 1.00000000, 0.35582280])
        XYZ_wr = np.array([0.95042855, 1.00000000, 1.08890037])
        M = np.array([[0.86876537, -0.14165393, 0.38719611],
                      [-0.10300724, 1.05840142, 0.15386462],
                      [0.00781674, 0.02678750, 2.96081771]])
        np.testing.assert_almost_equal(
            chromatic_adaptation_matrix_VonKries(XYZ_w, XYZ_wr),
            M,
            decimal=7)

        XYZ_w = np.tile(XYZ_w, (6, 1))
        XYZ_wr = np.tile(XYZ_wr, (6, 1))
        M = np.reshape(np.tile(M, (6, 1)), (6, 3, 3))
        np.testing.assert_almost_equal(
            chromatic_adaptation_matrix_VonKries(XYZ_w, XYZ_wr),
            M,
            decimal=7)

        XYZ_w = np.reshape(XYZ_w, (2, 3, 3))
        XYZ_wr = np.reshape(XYZ_wr, (2, 3, 3))
        M = np.reshape(M, (2, 3, 3, 3))
        np.testing.assert_almost_equal(
            chromatic_adaptation_matrix_VonKries(XYZ_w, XYZ_wr),
            M,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_chromatic_adaptation_matrix_VonKries(self):
        """
        Tests :func:`colour.adaptation.vonkries.\
chromatic_adaptation_matrix_VonKries` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ_w = np.array(case)
            XYZ_wr = np.array(case)
            chromatic_adaptation_matrix_VonKries(XYZ_w, XYZ_wr)


class TestChromaticAdaptationVonKries(unittest.TestCase):
    """
    Defines :func:`colour.adaptation.vonkries.chromatic_adaptation_VonKries`
    definition unit tests methods.
    """

    def test_chromatic_adaptation_VonKries(self):
        """
        Tests :func:`colour.adaptation.vonkries.chromatic_adaptation_VonKries`
        definition.
        """

        np.testing.assert_almost_equal(
            chromatic_adaptation_VonKries(
                np.array([0.07049534, 0.10080000, 0.09558313]),
                np.array([1.09846607, 1.00000000, 0.35582280]),
                np.array([0.95042855, 1.00000000, 1.08890037])),
            np.array([0.08397461, 0.11413219, 0.28625545]),
            decimal=7)

        np.testing.assert_almost_equal(
            chromatic_adaptation_VonKries(
                np.array([0.47097710, 0.34950000, 0.11301649]),
                np.array([0.99092745, 1.00000000, 0.85313273]),
                np.array([1.01679082, 1.00000000, 0.67610122])),
            np.array([0.49253636, 0.35555812, 0.08860435]),
            decimal=7)

        np.testing.assert_almost_equal(
            chromatic_adaptation_VonKries(
                np.array([0.25506814, 0.19150000, 0.08849752]),
                np.array([0.98070597, 1.00000000, 1.18224949]),
                np.array([0.92833635, 1.00000000, 1.03664720])),
            np.array([0.24731314, 0.19137674, 0.07734837]),
            decimal=7)

        np.testing.assert_almost_equal(
            chromatic_adaptation_VonKries(
                np.array([0.07049534, 0.10080000, 0.09558313]),
                np.array([1.09846607, 1.00000000, 0.35582280]),
                np.array([0.95042855, 1.00000000, 1.08890037]),
                transform='XYZ Scaling'),
            np.array([0.06099486, 0.10080000, 0.29250657]),
            decimal=7)

        np.testing.assert_almost_equal(
            chromatic_adaptation_VonKries(
                np.array([0.07049534, 0.10080000, 0.09558313]),
                np.array([1.09846607, 1.00000000, 0.35582280]),
                np.array([0.95042855, 1.00000000, 1.08890037]),
                transform='Bradford'),
            np.array([0.08540328, 0.11401229, 0.29721491]),
            decimal=7)

        np.testing.assert_almost_equal(
            chromatic_adaptation_VonKries(
                np.array([0.07049534, 0.10080000, 0.09558313]),
                np.array([1.09846607, 1.00000000, 0.35582280]),
                np.array([0.95042855, 1.00000000, 1.08890037]),
                transform='Von Kries'),
            np.array([0.08357823, 0.10214289, 0.29250657]),
            decimal=7)

    def test_n_dimensional_chromatic_adaptation_VonKries(self):
        """
        Tests :func:`colour.adaptation.vonkries.chromatic_adaptation_VonKries`
        definition n-dimensional arrays support.
        """

        XYZ = np.array([0.07049534, 0.10080000, 0.09558313])
        XYZ_w = np.array([1.09846607, 1.00000000, 0.35582280])
        XYZ_wr = np.array([0.95042855, 1.00000000, 1.08890037])
        XYZ_a = np.array([0.08397461, 0.11413219, 0.28625545])
        np.testing.assert_almost_equal(
            chromatic_adaptation_VonKries(XYZ, XYZ_w, XYZ_wr),
            XYZ_a,
            decimal=7)

        XYZ = np.tile(XYZ, (6, 1))
        XYZ_w = np.tile(XYZ_w, (6, 1))
        XYZ_wr = np.tile(XYZ_wr, (6, 1))
        XYZ_a = np.tile(XYZ_a, (6, 1))
        np.testing.assert_almost_equal(
            chromatic_adaptation_VonKries(XYZ, XYZ_w, XYZ_wr),
            XYZ_a,
            decimal=7)

        XYZ = np.reshape(XYZ, (2, 3, 3))
        XYZ_w = np.reshape(XYZ_w, (2, 3, 3))
        XYZ_wr = np.reshape(XYZ_wr, (2, 3, 3))
        XYZ_a = np.reshape(XYZ_a, (2, 3, 3))
        np.testing.assert_almost_equal(
            chromatic_adaptation_VonKries(XYZ, XYZ_w, XYZ_wr),
            XYZ_a,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_chromatic_adaptation_VonKries(self):
        """
        Tests :func:`colour.adaptation.vonkries.chromatic_adaptation_VonKries`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ = np.array(case)
            XYZ_w = np.array(case)
            XYZ_wr = np.array(case)
            chromatic_adaptation_VonKries(XYZ, XYZ_w, XYZ_wr)


if __name__ == '__main__':
    unittest.main()
