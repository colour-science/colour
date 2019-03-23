# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.colorimetry.luminance` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.colorimetry import (
    luminance_Newhall1943, intermediate_luminance_function_CIE1976,
    luminance_CIE1976, luminance_ASTMD153508, luminance_Fairchild2010,
    luminance_Fairchild2011)
from colour.colorimetry.luminance import luminance
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'TestLuminanceNewhall1943', 'TestLuminanceASTMD153508',
    'TestIntermediateLuminanceFunctionCIE1976', 'TestLuminanceCIE1976',
    'TestLuminanceFairchild2010', 'TestLuminanceFairchild2011', 'TestLuminance'
]


class TestLuminanceNewhall1943(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.luminance.luminance_Newhall1943`
    definition unit tests methods.
    """

    def test_luminance_Newhall1943(self):
        """
        Tests :func:`colour.colorimetry.luminance.luminance_Newhall1943`
        definition.
        """

        self.assertAlmostEqual(
            luminance_Newhall1943(4.08244375), 12.550078816731881, places=7)

        self.assertAlmostEqual(
            luminance_Newhall1943(5.39132685), 23.481252371310738, places=7)

        self.assertAlmostEqual(
            luminance_Newhall1943(2.97619312), 6.4514266875601924, places=7)

    def test_n_dimensional_luminance_Newhall1943(self):
        """
        Tests :func:`colour.colorimetry.luminance.luminance_Newhall1943`
        definition n-dimensional arrays support.
        """

        V = 4.08244375
        Y = 12.550078816731881
        np.testing.assert_almost_equal(luminance_Newhall1943(V), Y, decimal=7)

        V = np.tile(V, 6)
        Y = np.tile(Y, 6)
        np.testing.assert_almost_equal(luminance_Newhall1943(V), Y, decimal=7)

        V = np.reshape(V, (2, 3))
        Y = np.reshape(Y, (2, 3))
        np.testing.assert_almost_equal(luminance_Newhall1943(V), Y, decimal=7)

        V = np.reshape(V, (2, 3, 1))
        Y = np.reshape(Y, (2, 3, 1))
        np.testing.assert_almost_equal(luminance_Newhall1943(V), Y, decimal=7)

    def test_domain_range_scale_luminance_Newhall1943(self):
        """
        Tests :func:`colour.colorimetry.luminance.luminance_Newhall1943`
        definition domain and range scale support.
        """

        Y = luminance_Newhall1943(4.08244375)

        d_r = (('reference', 1, 1), (1, 0.1, 0.01), (100, 10, 1))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    luminance_Newhall1943(4.08244375 * factor_a),
                    Y * factor_b,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_luminance_Newhall1943(self):
        """
        Tests :func:`colour.colorimetry.luminance.luminance_Newhall1943`
        definition nan support.
        """

        luminance_Newhall1943(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLuminanceASTMD153508(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.luminance.luminance_ASTMD153508`
    definition unit tests methods.
    """

    def test_luminance_ASTMD153508(self):
        """
        Tests :func:`colour.colorimetry.luminance.luminance_ASTMD153508`
        definition.
        """

        self.assertAlmostEqual(
            luminance_ASTMD153508(4.08244375), 12.236342675366036, places=7)

        self.assertAlmostEqual(
            luminance_ASTMD153508(5.39132685), 22.893999867280378, places=7)

        self.assertAlmostEqual(
            luminance_ASTMD153508(2.97619312), 6.2902253509053132, places=7)

    def test_n_dimensional_luminance_ASTMD153508(self):
        """
        Tests :func:`colour.colorimetry.luminance.luminance_ASTMD153508`
        definition n-dimensional arrays support.
        """

        V = 4.08244375
        Y = 12.236342675366036
        np.testing.assert_almost_equal(luminance_ASTMD153508(V), Y, decimal=7)

        V = np.tile(V, 6)
        Y = np.tile(Y, 6)
        np.testing.assert_almost_equal(luminance_ASTMD153508(V), Y, decimal=7)

        V = np.reshape(V, (2, 3))
        Y = np.reshape(Y, (2, 3))
        np.testing.assert_almost_equal(luminance_ASTMD153508(V), Y, decimal=7)

        V = np.reshape(V, (2, 3, 1))
        Y = np.reshape(Y, (2, 3, 1))
        np.testing.assert_almost_equal(luminance_ASTMD153508(V), Y, decimal=7)

    def test_domain_range_scale_luminance_ASTMD153508(self):
        """
        Tests :func:`colour.colorimetry.luminance.luminance_ASTMD153508`
        definition domain and range scale support.
        """

        Y = luminance_ASTMD153508(4.08244375)

        d_r = (('reference', 1, 1), (1, 0.1, 0.01), (100, 10, 1))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    luminance_ASTMD153508(4.08244375 * factor_a),
                    Y * factor_b,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_luminance_ASTMD153508(self):
        """
        Tests :func:`colour.colorimetry.luminance.luminance_ASTMD153508`
        definition nan support.
        """

        luminance_ASTMD153508(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestIntermediateLuminanceFunctionCIE1976(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.luminance.\
intermediate_luminance_function_CIE1976` definition unit tests methods.
    """

    def test_intermediate_luminance_function_CIE1976(self):
        """
        Tests :func:`colour.colorimetry.luminance.\
intermediate_luminance_function_CIE1976` definition.
        """

        self.assertAlmostEqual(
            intermediate_luminance_function_CIE1976(0.495929964178047),
            12.197225350000002,
            places=7)

        self.assertAlmostEqual(
            intermediate_luminance_function_CIE1976(0.613072093530391),
            23.042767810000004,
            places=7)

        self.assertAlmostEqual(
            intermediate_luminance_function_CIE1976(0.394876333449113),
            6.157200790000001,
            places=7)

    def test_n_dimensional_intermediate_luminance_function_CIE1976(self):
        """
        Tests :func:`colour.colorimetry.luminance.\
intermediate_luminance_function_CIE1976` definition n-dimensional arrays
support.
        """

        f_Y_Y_n = 0.495929964178047
        Y = 12.197225350000002
        np.testing.assert_almost_equal(
            intermediate_luminance_function_CIE1976(f_Y_Y_n), Y, decimal=7)

        f_Y_Y_n = np.tile(f_Y_Y_n, 6)
        Y = np.tile(Y, 6)
        np.testing.assert_almost_equal(
            intermediate_luminance_function_CIE1976(f_Y_Y_n), Y, decimal=7)

        f_Y_Y_n = np.reshape(f_Y_Y_n, (2, 3))
        Y = np.reshape(Y, (2, 3))
        np.testing.assert_almost_equal(
            intermediate_luminance_function_CIE1976(f_Y_Y_n), Y, decimal=7)

        f_Y_Y_n = np.reshape(f_Y_Y_n, (2, 3, 1))
        Y = np.reshape(Y, (2, 3, 1))
        np.testing.assert_almost_equal(
            intermediate_luminance_function_CIE1976(f_Y_Y_n), Y, decimal=7)

    def test_domain_range_scale_intermediate_luminance_function_CIE1976(self):
        """
        Tests :func:`colour.colorimetry.luminance.\
intermediate_luminance_function_CIE1976` definition domain and range scale
support.
        """

        Y = intermediate_luminance_function_CIE1976(41.527875844653451, 100)

        for scale in ('reference', 1, 100):
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    intermediate_luminance_function_CIE1976(
                        41.527875844653451, 100),
                    Y,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_intermediate_luminance_function_CIE1976(self):
        """
        Tests :func:`colour.colorimetry.luminance.\
    intermediate_luminance_function_CIE1976` definition nan support.
        """

        intermediate_luminance_function_CIE1976(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLuminanceCIE1976(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.luminance.luminance_CIE1976` definition
    unit tests methods.
    """

    def test_luminance_CIE1976(self):
        """
        Tests :func:`colour.colorimetry.luminance.luminance_CIE1976`
        definition.
        """

        self.assertAlmostEqual(
            luminance_CIE1976(41.527875844653451),
            12.197225350000002,
            places=7)

        self.assertAlmostEqual(
            luminance_CIE1976(55.116362849525402),
            23.042767810000004,
            places=7)

        self.assertAlmostEqual(
            luminance_CIE1976(29.805654680097106), 6.157200790000001, places=7)

        self.assertAlmostEqual(
            luminance_CIE1976(56.480581732417676, 50),
            12.197225349999998,
            places=7)

        self.assertAlmostEqual(
            luminance_CIE1976(47.317620274162735, 75),
            12.197225350000002,
            places=7)

        self.assertAlmostEqual(
            luminance_CIE1976(42.519930728120940, 95),
            12.197225350000005,
            places=7)

    def test_n_dimensional_luminance_CIE1976(self):
        """
        Tests :func:`colour.colorimetry.luminance.luminance_CIE1976`
        definition n-dimensional arrays support.
        """

        L_star = 41.527875844653451
        Y = 12.197225350000002
        np.testing.assert_almost_equal(luminance_CIE1976(L_star), Y, decimal=7)

        L_star = np.tile(L_star, 6)
        Y = np.tile(Y, 6)
        np.testing.assert_almost_equal(luminance_CIE1976(L_star), Y, decimal=7)

        L_star = np.reshape(L_star, (2, 3))
        Y = np.reshape(Y, (2, 3))
        np.testing.assert_almost_equal(luminance_CIE1976(L_star), Y, decimal=7)

        L_star = np.reshape(L_star, (2, 3, 1))
        Y = np.reshape(Y, (2, 3, 1))
        np.testing.assert_almost_equal(luminance_CIE1976(L_star), Y, decimal=7)

    def test_domain_range_scale_luminance_CIE1976(self):
        """
        Tests :func:`colour.colorimetry.luminance.luminance_CIE1976`
        definition domain and range scale support.
        """

        Y = luminance_CIE1976(41.527875844653451, 100)

        d_r = (('reference', 1), (1, 0.01), (100, 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    luminance_CIE1976(41.527875844653451 * factor, 100),
                    Y * factor,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_luminance_CIE1976(self):
        """
        Tests :func:`colour.colorimetry.luminance.luminance_CIE1976`
        definition nan support.
        """

        luminance_CIE1976(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLuminanceFairchild2010(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.luminance.luminance_Fairchild2010`
    definition unit tests methods.
    """

    def test_luminance_Fairchild2010(self):
        """
        Tests :func:`colour.colorimetry.luminance.luminance_Fairchild2010`
        definition.
        """

        self.assertAlmostEqual(
            luminance_Fairchild2010(31.996390226262736),
            0.12197225350000002,
            places=7)

        self.assertAlmostEqual(
            luminance_Fairchild2010(60.203153682783302),
            0.23042767809999998,
            places=7)

        self.assertAlmostEqual(
            luminance_Fairchild2010(11.836517240976489),
            0.06157200790000001,
            places=7)

        self.assertAlmostEqual(
            luminance_Fairchild2010(24.424283249379986, 2.75),
            0.12197225350000002,
            places=7)

        self.assertAlmostEqual(
            luminance_Fairchild2010(100.019986327374240),
            1008.00000024,
            places=7)

        self.assertAlmostEqual(
            luminance_Fairchild2010(100.019999997090270),
            100799.92312466,
            places=7)

    def test_n_dimensional_luminance_Fairchild2010(self):
        """
        Tests :func:`colour.colorimetry.luminance.luminance_Fairchild2010`
        definition n-dimensional arrays support.
        """

        L_hdr = 31.996390226262736
        Y = 0.12197225350000002
        np.testing.assert_almost_equal(
            luminance_Fairchild2010(L_hdr), Y, decimal=7)

        L_hdr = np.tile(L_hdr, 6)
        Y = np.tile(Y, 6)
        np.testing.assert_almost_equal(
            luminance_Fairchild2010(L_hdr), Y, decimal=7)

        L_hdr = np.reshape(L_hdr, (2, 3))
        Y = np.reshape(Y, (2, 3))
        np.testing.assert_almost_equal(
            luminance_Fairchild2010(L_hdr), Y, decimal=7)

        L_hdr = np.reshape(L_hdr, (2, 3, 1))
        Y = np.reshape(Y, (2, 3, 1))
        np.testing.assert_almost_equal(
            luminance_Fairchild2010(L_hdr), Y, decimal=7)

    def test_domain_range_scale_luminance_Fairchild2010(self):
        """
        Tests :func:`colour.colorimetry.luminance.luminance_Fairchild2010`
        definition domain and range scale support.
        """

        Y = luminance_Fairchild2010(31.996390226262736)

        d_r = (('reference', 1, 1), (1, 0.01, 1), (100, 1, 100))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    luminance_Fairchild2010(31.996390226262736 * factor_a),
                    Y * factor_b,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_luminance_Fairchild2010(self):
        """
        Tests :func:`colour.colorimetry.luminance.luminance_Fairchild2010`
        definition nan support.
        """

        luminance_Fairchild2010(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLuminanceFairchild2011(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.luminance.luminance_Fairchild2011`
    definition unit tests methods.
    """

    def test_luminance_Fairchild2011(self):
        """
        Tests :func:`colour.colorimetry.luminance.luminance_Fairchild2011`
        definition.
        """

        self.assertAlmostEqual(
            luminance_Fairchild2011(51.852958445912506),
            0.12197225350000007,
            places=7)

        self.assertAlmostEqual(
            luminance_Fairchild2011(65.275207956353853),
            0.23042767809999998,
            places=7)

        self.assertAlmostEqual(
            luminance_Fairchild2011(39.818935510715917),
            0.061572007900000038,
            places=7)

        self.assertAlmostEqual(
            luminance_Fairchild2011(0.13268968410139345, 2.75),
            0.12197225350000002,
            places=7)

        self.assertAlmostEqual(
            luminance_Fairchild2011(234.72925681957565),
            1008.00000000,
            places=7)

        self.assertAlmostEqual(
            luminance_Fairchild2011(245.57059778237573),
            100800.00000000,
            places=7)

    def test_n_dimensional_luminance_Fairchild2011(self):
        """
        Tests :func:`colour.colorimetry.luminance.luminance_Fairchild2011`
        definition n-dimensional arrays support.
        """

        L_hdr = 51.852958445912506
        Y = 0.12197225350000007
        np.testing.assert_almost_equal(
            luminance_Fairchild2011(L_hdr), Y, decimal=7)

        L_hdr = np.tile(L_hdr, 6)
        Y = np.tile(Y, 6)
        np.testing.assert_almost_equal(
            luminance_Fairchild2011(L_hdr), Y, decimal=7)

        L_hdr = np.reshape(L_hdr, (2, 3))
        Y = np.reshape(Y, (2, 3))
        np.testing.assert_almost_equal(
            luminance_Fairchild2011(L_hdr), Y, decimal=7)

        L_hdr = np.reshape(L_hdr, (2, 3, 1))
        Y = np.reshape(Y, (2, 3, 1))
        np.testing.assert_almost_equal(
            luminance_Fairchild2011(L_hdr), Y, decimal=7)

    def test_domain_range_scale_luminance_Fairchild2011(self):
        """
        Tests :func:`colour.colorimetry.luminance.luminance_Fairchild2011`
        definition domain and range scale support.
        """

        Y = luminance_Fairchild2011(26.459509817572265)

        d_r = (('reference', 1, 1), (1, 0.01, 1), (100, 1, 100))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    luminance_Fairchild2011(26.459509817572265 * factor_a),
                    Y * factor_b,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_luminance_Fairchild2011(self):
        """
        Tests :func:`colour.colorimetry.luminance.luminance_Fairchild2011`
        definition nan support.
        """

        luminance_Fairchild2011(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLuminance(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.luminance.luminance` definition unit
    tests methods.
    """

    def test_domain_range_scale_luminance(self):
        """
        Tests :func:`colour.colorimetry.luminance.luminance` definition
        domain and range scale support.
        """

        m = ('Newhall 1943', 'ASTM D1535-08', 'CIE 1976', 'Fairchild 2010',
             'Fairchild 2011')
        v = [luminance(41.527875844653451, method, Y_n=100) for method in m]

        d_r = (('reference', 1), (1, 0.01), (100, 1))
        for method, value in zip(m, v):
            for scale, factor in d_r:
                with domain_range_scale(scale):
                    np.testing.assert_almost_equal(
                        luminance(
                            41.527875844653451 * factor, method, Y_n=100),
                        value * factor,
                        decimal=7)


if __name__ == '__main__':
    unittest.main()
