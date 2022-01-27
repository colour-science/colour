# -*- coding: utf-8 -*-
"""
Defines the unit tests for the :mod:`colour.colorimetry.lightness` module.
"""

import numpy as np
import unittest

from colour.colorimetry import (
    lightness_Glasser1958,
    lightness_Wyszecki1963,
    intermediate_lightness_function_CIE1976,
    lightness_CIE1976,
    lightness_Fairchild2010,
    lightness_Fairchild2011,
    lightness_Abebe2017,
)
from colour.colorimetry.lightness import lightness
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestLightnessGlasser1958',
    'TestLightnessWyszecki1963',
    'TestIntermediateLightnessFunctionCIE1976',
    'TestLightnessCIE1976',
    'TestLightnessFairchild2010',
    'TestLightnessFairchild2011',
    'TestLightnessAbebe2017',
    'TestLightness',
]


class TestLightnessGlasser1958(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.lightness.lightness_Glasser1958`
    definition unit tests methods.
    """

    def test_lightness_Glasser1958(self):
        """
        Tests :func:`colour.colorimetry.lightness.lightness_Glasser1958`
        definition.
        """

        self.assertAlmostEqual(
            lightness_Glasser1958(12.19722535), 39.83512646492521, places=7)

        self.assertAlmostEqual(
            lightness_Glasser1958(23.04276781), 53.585946877480623, places=7)

        self.assertAlmostEqual(
            lightness_Glasser1958(6.15720079), 27.972867038082629, places=7)

    def test_n_dimensional_lightness_Glasser1958(self):
        """
        Tests :func:`colour.colorimetry.lightness.lightness_Glasser1958`
        definition n-dimensional arrays support.
        """

        Y = 12.19722535
        L = lightness_Glasser1958(Y)

        Y = np.tile(Y, 6)
        L = np.tile(L, 6)
        np.testing.assert_almost_equal(lightness_Glasser1958(Y), L, decimal=7)

        Y = np.reshape(Y, (2, 3))
        L = np.reshape(L, (2, 3))
        np.testing.assert_almost_equal(lightness_Glasser1958(Y), L, decimal=7)

        Y = np.reshape(Y, (2, 3, 1))
        L = np.reshape(L, (2, 3, 1))
        np.testing.assert_almost_equal(lightness_Glasser1958(Y), L, decimal=7)

    def test_domain_range_scale_lightness_Glasser1958(self):
        """
        Tests :func:`colour.colorimetry.lightness.lightness_Glasser1958`
        definition domain and range scale support.
        """

        L = lightness_Glasser1958(12.19722535)

        d_r = (('reference', 1), ('1', 0.01), ('100', 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    lightness_Glasser1958(12.19722535 * factor),
                    L * factor,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_lightness_Glasser1958(self):
        """
        Tests :func:`colour.colorimetry.lightness.lightness_Glasser1958`
        definition nan support.
        """

        lightness_Glasser1958(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLightnessWyszecki1963(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.lightness.lightness_Wyszecki1963`
    definition unit tests methods.
    """

    def test_lightness_Wyszecki1963(self):
        """
        Tests :func:`colour.colorimetry.lightness.lightness_Wyszecki1963`
        definition.
        """

        self.assertAlmostEqual(
            lightness_Wyszecki1963(12.19722535), 40.547574599570197, places=7)

        self.assertAlmostEqual(
            lightness_Wyszecki1963(23.04276781), 54.140714588256841, places=7)

        self.assertAlmostEqual(
            lightness_Wyszecki1963(6.15720079), 28.821339499883976, places=7)

    def test_n_dimensional_lightness_Wyszecki1963(self):
        """
        Tests :func:`colour.colorimetry.lightness.lightness_Wyszecki1963`
        definition n-dimensional arrays support.
        """

        Y = 12.19722535
        W = lightness_Wyszecki1963(Y)

        Y = np.tile(Y, 6)
        W = np.tile(W, 6)
        np.testing.assert_almost_equal(lightness_Wyszecki1963(Y), W, decimal=7)

        Y = np.reshape(Y, (2, 3))
        W = np.reshape(W, (2, 3))
        np.testing.assert_almost_equal(lightness_Wyszecki1963(Y), W, decimal=7)

        Y = np.reshape(Y, (2, 3, 1))
        W = np.reshape(W, (2, 3, 1))
        np.testing.assert_almost_equal(lightness_Wyszecki1963(Y), W, decimal=7)

    def test_domain_range_scale_lightness_Wyszecki1963(self):
        """
        Tests :func:`colour.colorimetry.lightness.lightness_Wyszecki1963`
        definition domain and range scale support.
        """

        W = lightness_Wyszecki1963(12.19722535)

        d_r = (('reference', 1), ('1', 0.01), ('100', 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    lightness_Wyszecki1963(12.19722535 * factor),
                    W * factor,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_lightness_Wyszecki1963(self):
        """
        Tests :func:`colour.colorimetry.lightness.lightness_Wyszecki1963`
        definition nan support.
        """

        lightness_Wyszecki1963(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestIntermediateLightnessFunctionCIE1976(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.lightness.\
intermediate_lightness_function_CIE1976` definition unit tests methods.
    """

    def test_intermediate_lightness_function_CIE1976(self):
        """
        Tests :func:`colour.colorimetry.lightness.\
intermediate_lightness_function_CIE1976` definition.
        """

        self.assertAlmostEqual(
            intermediate_lightness_function_CIE1976(12.19722535),
            0.495929964178047,
            places=7)

        self.assertAlmostEqual(
            intermediate_lightness_function_CIE1976(23.04276781),
            0.613072093530391,
            places=7)

        self.assertAlmostEqual(
            intermediate_lightness_function_CIE1976(6.15720079),
            0.394876333449113,
            places=7)

    def test_n_dimensional_intermediate_lightness_function_CIE1976(self):
        """
        Tests :func:`colour.colorimetry.lightness.\
intermediate_lightness_function_CIE1976` definition n-dimensional arrays
        support.
        """

        Y = 12.19722535
        f_Y_Y_n = intermediate_lightness_function_CIE1976(Y)

        Y = np.tile(Y, 6)
        f_Y_Y_n = np.tile(f_Y_Y_n, 6)
        np.testing.assert_almost_equal(
            intermediate_lightness_function_CIE1976(Y), f_Y_Y_n, decimal=7)

        Y = np.reshape(Y, (2, 3))
        f_Y_Y_n = np.reshape(f_Y_Y_n, (2, 3))
        np.testing.assert_almost_equal(
            intermediate_lightness_function_CIE1976(Y), f_Y_Y_n, decimal=7)

        Y = np.reshape(Y, (2, 3, 1))
        f_Y_Y_n = np.reshape(f_Y_Y_n, (2, 3, 1))
        np.testing.assert_almost_equal(
            intermediate_lightness_function_CIE1976(Y), f_Y_Y_n, decimal=7)

    def test_domain_range_scale_intermediate_lightness_function_CIE1976(self):
        """
        Tests :func:`colour.colorimetry.lightness.\
intermediate_lightness_function_CIE1976` definition domain and range scale
        support.
        """

        f_Y_Y_n = intermediate_lightness_function_CIE1976(12.19722535, 100)

        for scale in ('reference', '1', '100'):
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    intermediate_lightness_function_CIE1976(12.19722535, 100),
                    f_Y_Y_n,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_intermediate_lightness_function_CIE1976(self):
        """
        Tests :func:`colour.colorimetry.lightness.\
intermediate_lightness_function_CIE1976` definition nan support.
        """

        intermediate_lightness_function_CIE1976(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLightnessCIE1976(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.lightness.lightness_CIE1976` definition
    unit tests methods.
    """

    def test_lightness_CIE1976(self):
        """
        Tests :func:`colour.colorimetry.lightness.lightness_CIE1976`
        definition.
        """

        self.assertAlmostEqual(
            lightness_CIE1976(12.19722535), 41.527875844653451, places=7)

        self.assertAlmostEqual(
            lightness_CIE1976(23.04276781), 55.116362849525402, places=7)

        self.assertAlmostEqual(
            lightness_CIE1976(6.15720079), 29.805654680097106, places=7)

        self.assertAlmostEqual(
            lightness_CIE1976(12.19722535, 50), 56.480581732417676, places=7)

        self.assertAlmostEqual(
            lightness_CIE1976(12.19722535, 75), 47.317620274162735, places=7)

        self.assertAlmostEqual(
            lightness_CIE1976(12.19722535, 95), 42.519930728120940, places=7)

    def test_n_dimensional_lightness_CIE1976(self):
        """
        Tests :func:`colour.colorimetry.lightness.lightness_CIE1976`
        definition n-dimensional arrays support.
        """

        Y = 12.19722535
        L_star = lightness_CIE1976(Y)

        Y = np.tile(Y, 6)
        L_star = np.tile(L_star, 6)
        np.testing.assert_almost_equal(lightness_CIE1976(Y), L_star, decimal=7)

        Y = np.reshape(Y, (2, 3))
        L_star = np.reshape(L_star, (2, 3))
        np.testing.assert_almost_equal(lightness_CIE1976(Y), L_star, decimal=7)

        Y = np.reshape(Y, (2, 3, 1))
        L_star = np.reshape(L_star, (2, 3, 1))
        np.testing.assert_almost_equal(lightness_CIE1976(Y), L_star, decimal=7)

    def test_domain_range_scale_lightness_CIE1976(self):
        """
        Tests :func:`colour.colorimetry.lightness.lightness_CIE1976`
        definition domain and range scale support.
        """

        L_star = lightness_CIE1976(12.19722535, 100)

        d_r = (('reference', 1), ('1', 0.01), ('100', 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    lightness_CIE1976(12.19722535 * factor, 100),
                    L_star * factor,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_lightness_CIE1976(self):
        """
        Tests :func:`colour.colorimetry.lightness.lightness_CIE1976`
        definition nan support.
        """

        lightness_CIE1976(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLightnessFairchild2010(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.lightness.lightness_Fairchild2010`
    definition unit tests methods.
    """

    def test_lightness_Fairchild2010(self):
        """
        Tests :func:`colour.colorimetry.lightness.lightness_Fairchild2010`
        definition.
        """

        self.assertAlmostEqual(
            lightness_Fairchild2010(12.19722535 / 100),
            31.996390226262736,
            places=7)

        self.assertAlmostEqual(
            lightness_Fairchild2010(23.04276781 / 100),
            60.203153682783302,
            places=7)

        self.assertAlmostEqual(
            lightness_Fairchild2010(6.15720079 / 100),
            11.836517240976489,
            places=7)

        self.assertAlmostEqual(
            lightness_Fairchild2010(12.19722535 / 100, 2.75),
            24.424283249379986,
            places=7)

        self.assertAlmostEqual(
            lightness_Fairchild2010(1008), 100.019986327374240, places=7)

        self.assertAlmostEqual(
            lightness_Fairchild2010(100800), 100.019999997090270, places=7)

    def test_n_dimensional_lightness_Fairchild2010(self):
        """
        Tests :func:`colour.colorimetry.lightness.lightness_Fairchild2010`
        definition n-dimensional arrays support.
        """

        Y = 12.19722535 / 100
        L_hdr = lightness_Fairchild2010(Y)

        Y = np.tile(Y, 6)
        L_hdr = np.tile(L_hdr, 6)
        np.testing.assert_almost_equal(
            lightness_Fairchild2010(Y), L_hdr, decimal=7)

        Y = np.reshape(Y, (2, 3))
        L_hdr = np.reshape(L_hdr, (2, 3))
        np.testing.assert_almost_equal(
            lightness_Fairchild2010(Y), L_hdr, decimal=7)

        Y = np.reshape(Y, (2, 3, 1))
        L_hdr = np.reshape(L_hdr, (2, 3, 1))
        np.testing.assert_almost_equal(
            lightness_Fairchild2010(Y), L_hdr, decimal=7)

    def test_domain_range_scale_lightness_Fairchild2010(self):
        """
        Tests :func:`colour.colorimetry.lightness.lightness_Fairchild2010`
        definition domain and range scale support.
        """

        L_hdr = lightness_Fairchild2010(12.19722535 / 100)

        d_r = (('reference', 1, 1), ('1', 1, 0.01), ('100', 100, 1))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    lightness_Fairchild2010(12.19722535 / 100 * factor_a),
                    L_hdr * factor_b,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_lightness_Fairchild2010(self):
        """
        Tests :func:`colour.colorimetry.lightness.lightness_Fairchild2010`
        definition nan support.
        """

        lightness_Fairchild2010(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLightnessFairchild2011(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.lightness.lightness_Fairchild2011`
    definition unit tests methods.
    """

    def test_lightness_Fairchild2011(self):
        """
        Tests :func:`colour.colorimetry.lightness.lightness_Fairchild2011`
        definition.
        """

        self.assertAlmostEqual(
            lightness_Fairchild2011(12.19722535 / 100),
            51.852958445912506,
            places=7)

        self.assertAlmostEqual(
            lightness_Fairchild2011(23.04276781 / 100),
            65.275207956353853,
            places=7)

        self.assertAlmostEqual(
            lightness_Fairchild2011(6.15720079 / 100),
            39.818935510715917,
            places=7)

        self.assertAlmostEqual(
            lightness_Fairchild2011(12.19722535 / 100, 2.75),
            0.13268968410139345,
            places=7)

        self.assertAlmostEqual(
            lightness_Fairchild2011(1008), 234.72925682, places=7)

        self.assertAlmostEqual(
            lightness_Fairchild2011(100800), 245.5705978, places=7)

    def test_n_dimensional_lightness_Fairchild2011(self):
        """
        Tests :func:`colour.colorimetry.lightness.lightness_Fairchild2011`
        definition n-dimensional arrays support.
        """

        Y = 12.19722535 / 100
        L_hdr = lightness_Fairchild2011(Y)

        Y = np.tile(Y, 6)
        L_hdr = np.tile(L_hdr, 6)
        np.testing.assert_almost_equal(
            lightness_Fairchild2011(Y), L_hdr, decimal=7)

        Y = np.reshape(Y, (2, 3))
        L_hdr = np.reshape(L_hdr, (2, 3))
        np.testing.assert_almost_equal(
            lightness_Fairchild2011(Y), L_hdr, decimal=7)

        Y = np.reshape(Y, (2, 3, 1))
        L_hdr = np.reshape(L_hdr, (2, 3, 1))
        np.testing.assert_almost_equal(
            lightness_Fairchild2011(Y), L_hdr, decimal=7)

    def test_domain_range_scale_lightness_Fairchild2011(self):
        """
        Tests :func:`colour.colorimetry.lightness.lightness_Fairchild2011`
        definition domain and range scale support.
        """

        L_hdr = lightness_Fairchild2011(12.19722535 / 100)

        d_r = (('reference', 1, 1), ('1', 1, 0.01), ('100', 100, 1))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    lightness_Fairchild2011(12.19722535 / 100 * factor_a),
                    L_hdr * factor_b,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_lightness_Fairchild2011(self):
        """
        Tests :func:`colour.colorimetry.lightness.lightness_Fairchild2011`
        definition nan support.
        """

        lightness_Fairchild2011(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLightnessAbebe2017(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.lightness.lightness_Abebe2017`
    definition unit tests methods.
    """

    def test_lightness_Abebe2017(self):
        """
        Tests :func:`colour.colorimetry.lightness.lightness_Abebe2017`
        definition.
        """

        self.assertAlmostEqual(
            lightness_Abebe2017(12.19722535), 0.486955571109229, places=7)

        self.assertAlmostEqual(
            lightness_Abebe2017(12.19722535, method='Stevens'),
            0.474544792145434,
            places=7)

        self.assertAlmostEqual(
            lightness_Abebe2017(12.19722535, 1000),
            0.286847428534793,
            places=7)

        self.assertAlmostEqual(
            lightness_Abebe2017(12.19722535, 4000),
            0.192145492588158,
            places=7)

        self.assertAlmostEqual(
            lightness_Abebe2017(12.19722535, 4000, method='Stevens'),
            0.170365211220992,
            places=7)

    def test_n_dimensional_lightness_Abebe2017(self):
        """
        Tests :func:`colour.colorimetry.lightness.lightness_Abebe2017`
        definition n-dimensional arrays support.
        """

        Y = 12.19722535
        L = lightness_Abebe2017(Y)

        Y = np.tile(Y, 6)
        L = np.tile(L, 6)
        np.testing.assert_almost_equal(lightness_Abebe2017(Y), L, decimal=7)

        Y = np.reshape(Y, (2, 3))
        L = np.reshape(L, (2, 3))
        np.testing.assert_almost_equal(lightness_Abebe2017(Y), L, decimal=7)

        Y = np.reshape(Y, (2, 3, 1))
        L = np.reshape(L, (2, 3, 1))
        np.testing.assert_almost_equal(lightness_Abebe2017(Y), L, decimal=7)

    def test_domain_range_scale_lightness_Abebe2017(self):
        """
        Tests :func:`colour.colorimetry.lightness.lightness_Abebe2017`
        definition domain and range scale support.
        """

        L = lightness_Abebe2017(12.19722535)

        d_r = (('reference', 1), ('1', 1), ('100', 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    lightness_Abebe2017(12.19722535 * factor, 100 * factor),
                    L * factor,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_lightness_Abebe2017(self):
        """
        Tests :func:`colour.colorimetry.lightness.lightness_Abebe2017`
        definition nan support.
        """

        lightness_Abebe2017(
            *[np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])] * 2)


class TestLightness(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.lightness.lightness` definition unit
    tests methods.
    """

    def test_domain_range_scale_lightness(self):
        """
        Tests :func:`colour.colorimetry.lightness.lightness` definition domain
        and range scale support.
        """

        m = ('Glasser 1958', 'Wyszecki 1963', 'CIE 1976', 'Fairchild 2010',
             'Fairchild 2011', 'Abebe 2017')
        v = [lightness(12.19722535, method, Y_n=100) for method in m]

        d_r = (('reference', 1), ('1', 0.01), ('100', 1))
        for method, value in zip(m, v):
            for scale, factor in d_r:
                with domain_range_scale(scale):
                    np.testing.assert_almost_equal(
                        lightness(12.19722535 * factor, method, Y_n=100),
                        value * factor,
                        decimal=7)


if __name__ == '__main__':
    unittest.main()
