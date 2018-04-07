# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.colorimetry.luminance` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.colorimetry import (luminance_Newhall1943, luminance_CIE1976,
                                luminance_ASTMD153508, luminance_Fairchild2010,
                                luminance_Fairchild2011)
from colour.colorimetry.luminance import luminance
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'TestLuminanceNewhall1943', 'TestLuminanceASTMD153508',
    'TestLuminanceCIE1976', 'TestLuminanceFairchild2010',
    'TestLuminanceFairchild2011', 'TestLuminance'
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
            luminance_Newhall1943(3.74629715), 10.40898746, places=7)

        self.assertAlmostEqual(
            luminance_Newhall1943(8.64728711), 71.31748010, places=7)

        self.assertAlmostEqual(
            luminance_Newhall1943(1.52569022), 2.06998750, places=7)

    def test_n_dimensional_luminance_Newhall1943(self):
        """
        Tests :func:`colour.colorimetry.luminance.luminance_Newhall1943`
        definition n-dimensional arrays support.
        """

        V = 3.74629715
        Y = 10.408987457743208
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

        Y = luminance_Newhall1943(3.74629715)

        d_r = (('reference', 1, 1), (1, 0.1, 0.01), (100, 10, 1))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    luminance_Newhall1943(3.74629715 * factor_a),
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
            luminance_ASTMD153508(3.74629715), 10.14880968, places=7)

        self.assertAlmostEqual(
            luminance_ASTMD153508(8.64728711), 69.53240916, places=7)

        self.assertAlmostEqual(
            luminance_ASTMD153508(1.52569022), 2.01830631, places=7)

    def test_n_dimensional_luminance_ASTMD153508(self):
        """
        Tests :func:`colour.colorimetry.luminance.luminance_ASTMD153508`
        definition n-dimensional arrays support.
        """

        V = 3.74629715
        Y = 10.148809678226682
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

        Y = luminance_ASTMD153508(3.74629715)

        d_r = (('reference', 1, 1), (1, 0.1, 0.01), (100, 10, 1))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    luminance_ASTMD153508(3.74629715 * factor_a),
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
            luminance_CIE1976(37.98562910), 10.08000000, places=7)

        self.assertAlmostEqual(
            luminance_CIE1976(80.04441556), 56.76000000, places=7)

        self.assertAlmostEqual(
            luminance_CIE1976(99.34672790), 98.32000000, places=7)

        self.assertAlmostEqual(
            luminance_CIE1976(37.98562910, 50), 5.04000000, places=7)

        self.assertAlmostEqual(
            luminance_CIE1976(37.98562910, 75), 7.56000000, places=7)

        self.assertAlmostEqual(
            luminance_CIE1976(37.98562910, 95), 9.57600000, places=7)

    def test_n_dimensional_luminance_CIE1976(self):
        """
        Tests :func:`colour.colorimetry.luminance.luminance_CIE1976`
        definition n-dimensional arrays support.
        """

        Lstar = 37.98562910
        Y = 10.080000000026304
        np.testing.assert_almost_equal(luminance_CIE1976(Lstar), Y, decimal=7)

        Lstar = np.tile(Lstar, 6)
        Y = np.tile(Y, 6)
        np.testing.assert_almost_equal(luminance_CIE1976(Lstar), Y, decimal=7)

        Lstar = np.reshape(Lstar, (2, 3))
        Y = np.reshape(Y, (2, 3))
        np.testing.assert_almost_equal(luminance_CIE1976(Lstar), Y, decimal=7)

        Lstar = np.reshape(Lstar, (2, 3, 1))
        Y = np.reshape(Y, (2, 3, 1))
        np.testing.assert_almost_equal(luminance_CIE1976(Lstar), Y, decimal=7)

    def test_domain_range_scale_luminance_CIE1976(self):
        """
        Tests :func:`colour.colorimetry.luminance.luminance_CIE1976`
        definition domain and range scale support.
        """

        Y = luminance_CIE1976(37.98562910, 100)

        d_r = (('reference', 1), (1, 0.01), (100, 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    luminance_CIE1976(37.98562910 * factor, 100 * factor),
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
            luminance_Fairchild2010(24.902290269546651), 0.10079999, places=7)

        self.assertAlmostEqual(
            luminance_Fairchild2010(88.797568871771162), 0.56759999, places=7)

        self.assertAlmostEqual(
            luminance_Fairchild2010(95.613018520289828), 0.98319999, places=7)

        self.assertAlmostEqual(
            luminance_Fairchild2010(16.064202706248068, 2.75),
            0.10079999,
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

        L_hdr = 24.902290269546651
        Y = 10.08 / 100
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

        Y = luminance_Fairchild2010(24.902290269546651)

        d_r = (('reference', 1, 1), (1, 0.01, 1), (100, 1, 100))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    luminance_Fairchild2010(24.902290269546651 * factor_a),
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
            luminance_Fairchild2011(26.459509817572265), 0.10079999, places=7)

        self.assertAlmostEqual(
            luminance_Fairchild2011(71.708466023819625), 0.56759999, places=7)

        self.assertAlmostEqual(
            luminance_Fairchild2011(93.030975393206475), 0.98319999, places=7)

        self.assertAlmostEqual(
            luminance_Fairchild2011(0.08672116154998, 2.75),
            0.10079999,
            places=7)

        self.assertAlmostEqual(
            luminance_Fairchild2011(244.07716520973938),
            1008.00000000,
            places=7)

        self.assertAlmostEqual(
            luminance_Fairchild2011(246.90681933957006),
            100800.00000000,
            places=7)

    def test_n_dimensional_luminance_Fairchild2011(self):
        """
        Tests :func:`colour.colorimetry.luminance.luminance_Fairchild2011`
        definition n-dimensional arrays support.
        """

        L_hdr = 26.459509817572265
        Y = 10.08 / 100
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
        v = [luminance(37.98562910, method, Y_n=100) for method in m]

        d_r = (('reference', 1), (1, 0.01), (100, 1))
        for method, value in zip(m, v):
            for scale, factor in d_r:
                with domain_range_scale(scale):
                    np.testing.assert_almost_equal(
                        luminance(
                            37.98562910 * factor, method, Y_n=100 * factor),
                        value * factor,
                        decimal=7)


if __name__ == '__main__':
    unittest.main()
