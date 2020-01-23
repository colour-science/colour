# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.contrast.barten1999` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from itertools import permutations

from colour.contrast import (optical_MTF_Barten1999, pupil_diameter_Barten1999,
                             sigma_Barten1999, retinal_illuminance_Barten1999,
                             maximum_angular_size_Barten1999,
                             contrast_sensitivity_function_Barten1999)
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestOpticalMTFBarten1999', 'TestPupilDiameterBarten1999',
    'TestSigmaBarten1999', 'TestRetinalIlluminanceBarten1999',
    'TestMaximumAngularSizeBarten1999',
    'TestContrastSensitivityFunctionBarten1999'
]


class TestOpticalMTFBarten1999(unittest.TestCase):
    """
    Defines :func:`colour.contrast.barten1999.optical_MTF_Barten1999`
    definition unit tests methods.
    """

    def test_optical_MTF_Barten1999(self):
        """
        Tests :func:`colour.contrast.barten1999.optical_MTF_Barten1999`
        definition.
        """

        np.testing.assert_almost_equal(
            optical_MTF_Barten1999(4, 0.01), 0.968910791191297, decimal=7)

        np.testing.assert_almost_equal(
            optical_MTF_Barten1999(8, 0.01), 0.881323136669471, decimal=7)

        np.testing.assert_almost_equal(
            optical_MTF_Barten1999(4, 0.05), 0.454040738727245, decimal=7)

    def test_n_dimensional_optical_MTF_Barten1999(self):
        """
        Tests :func:`colour.contrast.barten1999.optical_MTF_Barten1999`
        definition n-dimensional support.
        """

        u = np.array([4, 8, 12])
        sigma = np.array([0.01, 0.05, 0.1])
        M_opt = optical_MTF_Barten1999(u, sigma)

        u = np.tile(u, (6, 1))
        sigma = np.tile(sigma, (6, 1))
        M_opt = np.tile(M_opt, (6, 1))
        np.testing.assert_almost_equal(
            optical_MTF_Barten1999(u, sigma), M_opt, decimal=7)

        u = np.reshape(u, (2, 3, 3))
        sigma = np.reshape(sigma, (2, 3, 3))
        M_opt = np.reshape(M_opt, (2, 3, 3))
        np.testing.assert_almost_equal(
            optical_MTF_Barten1999(u, sigma), M_opt, decimal=7)

    @ignore_numpy_errors
    def test_nan_optical_MTF_Barten1999(self):
        """
        Tests :func:`colour.contrast.barten1999.optical_MTF_Barten1999`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            optical_MTF_Barten1999(np.array(case), np.array(case))


class TestPupilDiameterBarten1999(unittest.TestCase):
    """
    Defines :func:`colour.contrast.barten1999.pupil_diameter_Barten1999`
    definition unit tests methods.
    """

    def test_pupil_diameter_Barten1999(self):
        """
        Tests :func:`colour.contrast.barten1999.pupil_diameter_Barten1999`
        definition.
        """

        np.testing.assert_almost_equal(
            pupil_diameter_Barten1999(20, 60), 2.272517118855717, decimal=7)

        np.testing.assert_almost_equal(
            pupil_diameter_Barten1999(0.2, 600), 2.272517118855717, decimal=7)

        np.testing.assert_almost_equal(
            pupil_diameter_Barten1999(20, 60, 30),
            2.459028745178825,
            decimal=7)

    def test_n_dimensional_pupil_diameter_Barten1999(self):
        """
        Tests :func:`colour.contrast.barten1999.pupil_diameter_Barten1999`
        definition n-dimensional support.
        """

        L = np.array([0.2, 20, 100])
        X_0 = np.array([60, 120, 240])
        Y_0 = np.array([60, 30, 15])
        d = pupil_diameter_Barten1999(L, X_0, Y_0)

        L = np.tile(L, (6, 1))
        X_0 = np.tile(X_0, (6, 1))
        d = np.tile(d, (6, 1))
        np.testing.assert_almost_equal(
            pupil_diameter_Barten1999(L, X_0, Y_0), d, decimal=7)

        L = np.reshape(L, (2, 3, 3))
        X_0 = np.reshape(X_0, (2, 3, 3))
        d = np.reshape(d, (2, 3, 3))
        np.testing.assert_almost_equal(
            pupil_diameter_Barten1999(L, X_0, Y_0), d, decimal=7)

    @ignore_numpy_errors
    def test_nan_pupil_diameter_Barten1999(self):
        """
        Tests :func:`colour.contrast.barten1999.pupil_diameter_Barten1999`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            pupil_diameter_Barten1999(
                np.array(case), np.array(case), np.array(case))


class TestSigmaBarten1999(unittest.TestCase):
    """
    Defines :func:`colour.contrast.barten1999.sigma_Barten1999` definition unit
    tests methods.
    """

    def test_sigma_Barten1999(self):
        """
        Tests :func:`colour.contrast.barten1999.sigma_Barten1999` definition.
        """

        np.testing.assert_almost_equal(
            sigma_Barten1999(0.5 / 60, 0.08 / 60, 2.1),
            0.008791157173231,
            decimal=7)

        np.testing.assert_almost_equal(
            sigma_Barten1999(0.75 / 60, 0.08 / 60, 2.1),
            0.012809761902549,
            decimal=7)

        np.testing.assert_almost_equal(
            sigma_Barten1999(0.5 / 60, 0.16 / 60, 2.1),
            0.010040141654601,
            decimal=7)

        np.testing.assert_almost_equal(
            sigma_Barten1999(0.5 / 60, 0.08 / 60, 2.5),
            0.008975274678558,
            decimal=7)

    def test_n_dimensional_sigma_Barten1999(self):
        """
        Tests :func:`colour.contrast.barten1999.sigma_Barten1999` definition
        n-dimensional support.
        """

        sigma_0 = np.array([0.25 / 60, 0.5 / 60, 0.75 / 60])
        C_ab = np.array([0.04 / 60, 0.08 / 60, 0.16 / 60])
        d = np.array([2.1, 2.5, 5.0])
        sigma = sigma_Barten1999(sigma_0, C_ab, d)

        sigma_0 = np.tile(sigma_0, (6, 1))
        C_ab = np.tile(C_ab, (6, 1))
        sigma = np.tile(sigma, (6, 1))
        np.testing.assert_almost_equal(
            sigma_Barten1999(sigma_0, C_ab, d), sigma, decimal=7)

        sigma_0 = np.reshape(sigma_0, (2, 3, 3))
        C_ab = np.reshape(C_ab, (2, 3, 3))
        sigma = np.reshape(sigma, (2, 3, 3))
        np.testing.assert_almost_equal(
            sigma_Barten1999(sigma_0, C_ab, d), sigma, decimal=7)

    @ignore_numpy_errors
    def test_nan_sigma_Barten1999(self):
        """
        Tests :func:`colour.contrast.barten1999.sigma_Barten1999`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            sigma_Barten1999(np.array(case), np.array(case), np.array(case))


class TestRetinalIlluminanceBarten1999(unittest.TestCase):
    """
    Defines :func:`colour.contrast.barten1999.retinal_illuminance_Barten1999`
    definition unit tests methods.
    """

    def test_retinal_illuminance_Barten1999(self):
        """
        Tests :func:`colour.contrast.barten1999.retinal_illuminance_Barten1999`
        definition.
        """

        np.testing.assert_almost_equal(
            retinal_illuminance_Barten1999(20, 2.1, True),
            66.082316060529919,
            decimal=7)

        np.testing.assert_almost_equal(
            retinal_illuminance_Barten1999(20, 2.5, True),
            91.815644777503664,
            decimal=7)

        np.testing.assert_almost_equal(
            retinal_illuminance_Barten1999(20, 2.1, False),
            69.272118011654939,
            decimal=7)

    def test_n_dimensional_retinal_illuminance_Barten1999(self):
        """
        Tests :func:`colour.contrast.barten1999.retinal_illuminance_Barten1999`
        definition n-dimensional support.
        """

        L = np.array([0.2, 20, 100])
        d = np.array([2.1, 2.5, 5.0])
        E = retinal_illuminance_Barten1999(L, d)

        L = np.tile(L, (6, 1))
        d = np.tile(d, (6, 1))
        E = np.tile(E, (6, 1))
        np.testing.assert_almost_equal(
            retinal_illuminance_Barten1999(L, d), E, decimal=7)

        L = np.reshape(L, (2, 3, 3))
        d = np.reshape(d, (2, 3, 3))
        E = np.reshape(E, (2, 3, 3))
        np.testing.assert_almost_equal(
            retinal_illuminance_Barten1999(L, d), E, decimal=7)

    @ignore_numpy_errors
    def test_nan_retinal_illuminance_Barten1999(self):
        """
        Tests :func:`colour.contrast.barten1999.retinal_illuminance_Barten1999`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            retinal_illuminance_Barten1999(np.array(case), np.array(case))


class TestMaximumAngularSizeBarten1999(unittest.TestCase):
    """
    Defines :func:`colour.contrast.barten1999.maximum_angular_size_Barten1999`
    definition unit tests methods.
    """

    def test_maximum_angular_size_Barten1999(self):
        """
        Tests :func:`colour.contrast.barten1999.\
maximum_angular_size_Barten1999` definition.
        """

        np.testing.assert_almost_equal(
            maximum_angular_size_Barten1999(4, 60, 12, 15),
            3.572948005052482,
            decimal=7)

        np.testing.assert_almost_equal(
            maximum_angular_size_Barten1999(8, 60, 12, 15),
            1.851640199545103,
            decimal=7)

        np.testing.assert_almost_equal(
            maximum_angular_size_Barten1999(4, 120, 12, 15),
            3.577708763999663,
            decimal=7)

        np.testing.assert_almost_equal(
            maximum_angular_size_Barten1999(4, 60, 24, 15),
            3.698001308168194,
            decimal=7)

        np.testing.assert_almost_equal(
            maximum_angular_size_Barten1999(4, 60, 12, 30),
            6.324555320336758,
            decimal=7)

    def test_n_dimensional_maximum_angular_size_Barten1999(self):
        """
        Tests :func:`colour.contrast.barten1999.\
maximum_angular_size_Barten1999` definition n-dimensional support.
        """

        u = np.array([4, 8, 12])
        X_0 = np.array([60, 120, 240])
        X_max = np.array([12, 14, 16])
        N_max = np.array([15, 20, 25])
        X = maximum_angular_size_Barten1999(u, X_0, X_max, N_max)

        u = np.tile(u, (6, 1))
        X_0 = np.tile(X_0, (6, 1))
        X = np.tile(X, (6, 1))
        np.testing.assert_almost_equal(
            maximum_angular_size_Barten1999(u, X_0, X_max, N_max),
            X,
            decimal=7)

        u = np.reshape(u, (2, 3, 3))
        X_0 = np.reshape(X_0, (2, 3, 3))
        X = np.reshape(X, (2, 3, 3))
        np.testing.assert_almost_equal(
            maximum_angular_size_Barten1999(u, X_0, X_max, N_max),
            X,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_maximum_angular_size_Barten1999(self):
        """
        Tests :func:`colour.contrast.barten1999.\
maximum_angular_size_Barten1999` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            maximum_angular_size_Barten1999(
                np.array(case), np.array(case), np.array(case), np.array(case))


class TestContrastSensitivityFunctionBarten1999(unittest.TestCase):
    """
    Defines :func:`colour.contrast.barten1999.\
contrast_sensitivity_function_Barten1999` definition unit tests methods.
    """

    def test_contrast_sensitivity_function_Barten1999(self):
        """
        Tests :func:`colour.contrast.barten1999.\
contrast_sensitivity_function_Barten1999` definition.
        """

        np.testing.assert_almost_equal(
            contrast_sensitivity_function_Barten1999(
                u=4,
                sigma=0.01,
                E=65,
                X_0=60,
                X_max=12,
                Y_0=60,
                Y_max=12,
                p=1.2 * 10 ** 6),
            352.761342126727020,
            decimal=7)

        np.testing.assert_almost_equal(
            contrast_sensitivity_function_Barten1999(
                u=8,
                sigma=0.01,
                E=65,
                X_0=60,
                X_max=12,
                Y_0=60,
                Y_max=12,
                p=1.2 * 10 ** 6),
            177.706338840717340,
            decimal=7)

        np.testing.assert_almost_equal(
            contrast_sensitivity_function_Barten1999(
                u=4,
                sigma=0.02,
                E=65,
                X_0=60,
                X_max=12,
                Y_0=60,
                Y_max=12,
                p=1.2 * 10 ** 6),
            320.872401634215750,
            decimal=7)

        np.testing.assert_almost_equal(
            contrast_sensitivity_function_Barten1999(
                u=4,
                sigma=0.01,
                E=130,
                X_0=60,
                X_max=12,
                Y_0=60,
                Y_max=12,
                p=1.2 * 10 ** 6),
            455.171315756946400,
            decimal=7)

        np.testing.assert_almost_equal(
            contrast_sensitivity_function_Barten1999(
                u=4,
                sigma=0.01,
                E=65,
                X_0=120,
                X_max=12,
                Y_0=60,
                Y_max=12,
                p=1.2 * 10 ** 6),
            352.996281545740660,
            decimal=7)

        np.testing.assert_almost_equal(
            contrast_sensitivity_function_Barten1999(
                u=4,
                sigma=0.01,
                E=65,
                X_0=60,
                X_max=24,
                Y_0=60,
                Y_max=12,
                p=1.2 * 10 ** 6),
            358.881580104493650,
            decimal=7)

        np.testing.assert_almost_equal(
            contrast_sensitivity_function_Barten1999(
                u=4,
                sigma=0.01,
                E=65,
                X_0=240,
                X_max=12,
                Y_0=60,
                Y_max=12,
                p=1.2 * 10 ** 6),
            contrast_sensitivity_function_Barten1999(
                u=4,
                sigma=0.01,
                E=65,
                X_0=60,
                X_max=12,
                Y_0=240,
                Y_max=12,
                p=1.2 * 10 ** 6),
            decimal=7)

        np.testing.assert_almost_equal(
            contrast_sensitivity_function_Barten1999(
                u=4,
                sigma=0.01,
                E=65,
                X_0=60,
                X_max=12,
                Y_0=60,
                Y_max=12,
                p=1.4 * 10 ** 6),
            374.791328640476140,
            decimal=7)

    def test_n_dimensional_contrast_sensitivity_function_Barten1999(self):
        """
        Tests :func:`colour.contrast.barten1999.\
contrast_sensitivity_function_Barten1999` definition n-dimensional support.
        """

        u = np.array([4, 8, 12])
        sigma = np.array([0.01, 0.02, 0.04])
        E = np.array([0.65, 90, 1500])
        X_0 = np.array([60, 120, 240])
        S = contrast_sensitivity_function_Barten1999(
            u=u, sigma=sigma, E=E, X_0=X_0)

        u = np.tile(u, (6, 1))
        E = np.tile(E, (6, 1))
        S = np.tile(S, (6, 1))
        np.testing.assert_almost_equal(
            contrast_sensitivity_function_Barten1999(
                u=u, sigma=sigma, E=E, X_0=X_0),
            S,
            decimal=7)

        u = np.reshape(u, (2, 3, 3))
        E = np.reshape(E, (2, 3, 3))
        S = np.reshape(S, (2, 3, 3))
        np.testing.assert_almost_equal(
            contrast_sensitivity_function_Barten1999(
                u=u, sigma=sigma, E=E, X_0=X_0),
            S,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_contrast_sensitivity_function_Barten1999(self):
        """
        Tests :func:`colour.contrast.barten1999.\
contrast_sensitivity_function_Barten1999` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            contrast_sensitivity_function_Barten1999(
                u=np.array(case),
                sigma=np.array(case),
                E=np.array(case),
                X_0=np.array(case))


if __name__ == '__main__':
    unittest.main()
