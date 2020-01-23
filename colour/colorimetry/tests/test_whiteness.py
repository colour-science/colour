# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.colorimetry.whiteness` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from itertools import permutations

from colour.colorimetry import (whiteness_Berger1959, whiteness_Taube1960,
                                whiteness_Stensby1968, whiteness_ASTME313,
                                whiteness_Ganz1979, whiteness_CIE2004)
from colour.colorimetry.whiteness import whiteness
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestWhitenessBerger1959', 'TestWhitenessTaube1960',
    'TestWhitenessStensby1968', 'TestWhitenessASTM313',
    'TestWhitenessGanz1979', 'TestWhitenessCIE2004', 'TestWhiteness'
]


class TestWhitenessBerger1959(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.whiteness.whiteness_Berger1959`
    definition unit tests methods.
    """

    def test_whiteness_Berger1959(self):
        """
        Tests :func:`colour.colorimetry.whiteness.whiteness_Berger1959`
        definition.
        """

        self.assertAlmostEqual(
            whiteness_Berger1959(
                np.array([95.00000000, 100.00000000, 105.00000000]),
                np.array([94.80966767, 100.00000000, 107.30513595])),
            30.36380179,
            places=7)

        self.assertAlmostEqual(
            whiteness_Berger1959(
                np.array([105.00000000, 100.00000000, 95.00000000]),
                np.array([94.80966767, 100.00000000, 107.30513595])),
            5.530469280673941,
            places=7)

        self.assertAlmostEqual(
            whiteness_Berger1959(
                np.array([100.00000000, 100.00000000, 100.00000000]),
                np.array([100.00000000, 100.00000000, 100.00000000])),
            33.300000000000011,
            places=7)

    def test_n_dimensional_whiteness_Berger1959(self):
        """
        Tests :func:`colour.colorimetry.whiteness.whiteness_Berger1959`
        definition n_dimensional arrays support.
        """

        XYZ = np.array([95.00000000, 100.00000000, 105.00000000])
        XYZ_0 = np.array([94.80966767, 100.00000000, 107.30513595])
        W = whiteness_Berger1959(XYZ, XYZ_0)

        XYZ = np.tile(XYZ, (6, 1))
        XYZ_0 = np.tile(XYZ_0, (6, 1))
        W = np.tile(W, 6)
        np.testing.assert_almost_equal(
            whiteness_Berger1959(XYZ, XYZ_0), W, decimal=7)

        XYZ = np.reshape(XYZ, (2, 3, 3))
        XYZ_0 = np.reshape(XYZ_0, (2, 3, 3))
        W = np.reshape(W, (2, 3))
        np.testing.assert_almost_equal(
            whiteness_Berger1959(XYZ, XYZ_0), W, decimal=7)

    def test_domain_range_scale_whiteness_Berger1959(self):
        """
        Tests :func:`colour.colorimetry.whiteness.whiteness_Berger1959`
        definition domain and range scale support.
        """

        XYZ = np.array([95.00000000, 100.00000000, 105.00000000])
        XYZ_0 = np.array([94.80966767, 100.00000000, 107.30513595])
        W = whiteness_Berger1959(XYZ, XYZ_0)

        d_r = (('reference', 1), (1, 0.01), (100, 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    whiteness_Berger1959(XYZ * factor, XYZ_0 * factor),
                    W * factor,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_whiteness_Berger1959(self):
        """
        Tests :func:`colour.colorimetry.whiteness.whiteness_Berger1959`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ = np.array(case)
            XYZ_0 = np.array(case)
            whiteness_Berger1959(XYZ, XYZ_0)


class TestWhitenessTaube1960(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.whiteness.whiteness_Taube1960`
    definition unit tests methods.
    """

    def test_whiteness_Taube1960(self):
        """
        Tests :func:`colour.colorimetry.whiteness.whiteness_Taube1960`
        definition.
        """

        self.assertAlmostEqual(
            whiteness_Taube1960(
                np.array([95.00000000, 100.00000000, 105.00000000]),
                np.array([94.80966767, 100.00000000, 107.30513595])),
            91.407173833416152,
            places=7)

        self.assertAlmostEqual(
            whiteness_Taube1960(
                np.array([105.00000000, 100.00000000, 95.00000000]),
                np.array([94.80966767, 100.00000000, 107.30513595])),
            54.130300134995593,
            places=7)

        self.assertAlmostEqual(
            whiteness_Taube1960(
                np.array([100.00000000, 100.00000000, 100.00000000]),
                np.array([100.00000000, 100.00000000, 100.00000000])),
            100.0,
            places=7)

    def test_n_dimensional_whiteness_Taube1960(self):
        """
        Tests :func:`colour.colorimetry.whiteness.whiteness_Taube1960`
        definition n_dimensional arrays support.
        """

        XYZ = np.array([95.00000000, 100.00000000, 105.00000000])
        XYZ_0 = np.array([94.80966767, 100.00000000, 107.30513595])
        WI = whiteness_Taube1960(XYZ, XYZ_0)

        XYZ = np.tile(XYZ, (6, 1))
        XYZ_0 = np.tile(XYZ_0, (6, 1))
        WI = np.tile(WI, 6)
        np.testing.assert_almost_equal(
            whiteness_Taube1960(XYZ, XYZ_0), WI, decimal=7)

        XYZ = np.reshape(XYZ, (2, 3, 3))
        XYZ_0 = np.reshape(XYZ_0, (2, 3, 3))
        WI = np.reshape(WI, (2, 3))
        np.testing.assert_almost_equal(
            whiteness_Taube1960(XYZ, XYZ_0), WI, decimal=7)

    def test_domain_range_scale_whiteness_Taube1960(self):
        """
        Tests :func:`colour.colorimetry.whiteness.whiteness_Taube1960`
        definition domain and range scale support.
        """

        XYZ = np.array([95.00000000, 100.00000000, 105.00000000])
        XYZ_0 = np.array([94.80966767, 100.00000000, 107.30513595])
        WI = whiteness_Taube1960(XYZ, XYZ_0)

        d_r = (('reference', 1), (1, 0.01), (100, 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    whiteness_Taube1960(XYZ * factor, XYZ_0 * factor),
                    WI * factor,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_whiteness_Berger1959(self):
        """
        Tests :func:`colour.colorimetry.whiteness.whiteness_Berger1959`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ = np.array(case)
            XYZ_0 = np.array(case)
            whiteness_Berger1959(XYZ, XYZ_0)


class TestWhitenessStensby1968(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.whiteness.whiteness_Stensby1968`
    definition unit tests methods.
    """

    def test_whiteness_Stensby1968(self):
        """
        Tests :func:`colour.colorimetry.whiteness.whiteness_Stensby1968`
        definition.
        """

        self.assertAlmostEqual(
            whiteness_Stensby1968(
                np.array([100.00000000, -2.46875131, -16.72486654])),
            142.76834569,
            places=7)

        self.assertAlmostEqual(
            whiteness_Stensby1968(
                np.array([100.00000000, 14.40943727, -9.61394885])),
            172.07015836,
            places=7)

        self.assertAlmostEqual(
            whiteness_Stensby1968(np.array([1, 1, 1])), 1.00000000, places=7)

    def test_n_dimensional_whiteness_Stensby1968(self):
        """
        Tests :func:`colour.colorimetry.whiteness.whiteness_Stensby1968`
        definition n_dimensional arrays support.
        """

        Lab = np.array([100.00000000, -2.46875131, -16.72486654])
        WI = whiteness_Stensby1968(Lab)

        Lab = np.tile(Lab, (6, 1))
        WI = np.tile(WI, 6)
        np.testing.assert_almost_equal(
            whiteness_Stensby1968(Lab), WI, decimal=7)

        Lab = np.reshape(Lab, (2, 3, 3))
        WI = np.reshape(WI, (2, 3))
        np.testing.assert_almost_equal(
            whiteness_Stensby1968(Lab), WI, decimal=7)

    def test_domain_range_scale_whiteness_Stensby1968(self):
        """
        Tests :func:`colour.colorimetry.whiteness.whiteness_Stensby1968`
        definition domain and range scale support.
        """

        Lab = np.array([100.00000000, -2.46875131, -16.72486654])
        WI = whiteness_Stensby1968(Lab)

        d_r = (('reference', 1), (1, 0.01), (100, 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    whiteness_Stensby1968(Lab * factor),
                    WI * factor,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_whiteness_Stensby1968(self):
        """
        Tests :func:`colour.colorimetry.whiteness.whiteness_Stensby1968`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            Lab = np.array(case)
            whiteness_Stensby1968(Lab)


class TestWhitenessASTM313(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.whiteness.whiteness_ASTME313`
    definition unit tests methods.
    """

    def test_whiteness_ASTME313(self):
        """
        Tests :func:`colour.colorimetry.whiteness.whiteness_ASTME313`
        definition.
        """

        self.assertAlmostEqual(
            whiteness_ASTME313(
                np.array([95.00000000, 100.00000000, 105.00000000])),
            55.740000000000009,
            places=7)

        self.assertAlmostEqual(
            whiteness_ASTME313(
                np.array([105.00000000, 100.00000000, 95.00000000])),
            21.860000000000014,
            places=7)

        self.assertAlmostEqual(
            whiteness_ASTME313(
                np.array([100.00000000, 100.00000000, 100.00000000])),
            38.800000000000011,
            places=7)

    def test_n_dimensional_whiteness_ASTME313(self):
        """
        Tests :func:`colour.colorimetry.whiteness.whiteness_ASTME313`
        definition n_dimensional arrays support.
        """

        XYZ = np.array([95.00000000, 100.00000000, 105.00000000])
        WI = whiteness_ASTME313(XYZ)

        XYZ = np.tile(XYZ, (6, 1))
        WI = np.tile(WI, 6)
        np.testing.assert_almost_equal(whiteness_ASTME313(XYZ), WI, decimal=7)

        XYZ = np.reshape(XYZ, (2, 3, 3))
        WI = np.reshape(WI, (2, 3))
        np.testing.assert_almost_equal(whiteness_ASTME313(XYZ), WI, decimal=7)

    def test_domain_range_scale_whiteness_ASTME313(self):
        """
        Tests :func:`colour.colorimetry.whiteness.whiteness_ASTME313`
        definition domain and range scale support.
        """

        XYZ = np.array([95.00000000, 100.00000000, 105.00000000])
        WI = whiteness_ASTME313(XYZ)

        d_r = (('reference', 1), (1, 0.01), (100, 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    whiteness_ASTME313(XYZ * factor), WI * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_whiteness_ASTME313(self):
        """
        Tests :func:`colour.colorimetry.whiteness.whiteness_ASTME313`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ = np.array(case)
            whiteness_ASTME313(XYZ)


class TestWhitenessGanz1979(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.whiteness.whiteness_Ganz1979`
    definition unit tests methods.
    """

    def test_whiteness_Ganz1979(self):
        """
        Tests :func:`colour.colorimetry.whiteness.whiteness_Ganz1979`
        definition.
        """

        np.testing.assert_almost_equal(
            whiteness_Ganz1979(np.array([0.3139, 0.3311]), 100),
            np.array([99.33176520, 1.76108290]),
            decimal=7)

        np.testing.assert_almost_equal(
            whiteness_Ganz1979(np.array([0.3500, 0.3334]), 100),
            np.array([23.38525400, -32.66182560]),
            decimal=7)

        np.testing.assert_almost_equal(
            whiteness_Ganz1979(np.array([0.3334, 0.3334]), 100),
            np.array([54.39939920, -16.04152380]),
            decimal=7)

    def test_n_dimensional_whiteness_Ganz1979(self):
        """
        Tests :func:`colour.colorimetry.whiteness.whiteness_Ganz1979`
        definition n_dimensional arrays support.
        """

        xy = np.array([0.3167, 0.3334])
        Y = 100
        WT = whiteness_Ganz1979(xy, Y)

        xy = np.tile(xy, (6, 1))
        WT = np.tile(WT, (6, 1))
        np.testing.assert_almost_equal(
            whiteness_Ganz1979(xy, Y), WT, decimal=7)

        Y = np.tile(Y, 6)
        np.testing.assert_almost_equal(
            whiteness_Ganz1979(xy, Y), WT, decimal=7)

        xy = np.reshape(xy, (2, 3, 2))
        Y = np.reshape(Y, (2, 3))
        WT = np.reshape(WT, (2, 3, 2))
        np.testing.assert_almost_equal(
            whiteness_Ganz1979(xy, Y), WT, decimal=7)

    def test_domain_range_scale_whiteness_Ganz1979(self):
        """
        Tests :func:`colour.colorimetry.whiteness.whiteness_Ganz1979`
        definition domain and range scale support.
        """

        xy = np.array([0.3167, 0.3334])
        Y = 100
        WT = whiteness_Ganz1979(xy, Y)

        d_r = (('reference', 1), (1, 0.01), (100, 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    whiteness_Ganz1979(xy, Y * factor), WT * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_whiteness_Ganz1979(self):
        """
        Tests :func:`colour.colorimetry.whiteness.whiteness_Ganz1979`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            xy = np.array(case[0:2])
            Y = np.array(case[0])
            whiteness_Ganz1979(xy, Y)


class TestWhitenessCIE2004(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.whiteness.whiteness_CIE2004`
    definition unit tests methods.
    """

    def test_whiteness_CIE2004(self):
        """
        Tests :func:`colour.colorimetry.whiteness.whiteness_CIE2004`
        definition.
        """

        np.testing.assert_almost_equal(
            whiteness_CIE2004(
                np.array([0.3139, 0.3311]), 100, np.array([0.3139, 0.3311])),
            np.array([100.00000000, 0.00000000]),
            decimal=7)

        np.testing.assert_almost_equal(
            whiteness_CIE2004(
                np.array([0.3500, 0.3334]), 100, np.array([0.3139, 0.3311])),
            np.array([67.21000000, -34.60500000]),
            decimal=7)

        np.testing.assert_almost_equal(
            whiteness_CIE2004(
                np.array([0.3334, 0.3334]), 100, np.array([0.3139, 0.3311])),
            np.array([80.49000000, -18.00500000]),
            decimal=7)

    def test_n_dimensional_whiteness_CIE2004(self):
        """
        Tests :func:`colour.colorimetry.whiteness.whiteness_CIE2004`
        definition n_dimensional arrays support.
        """

        xy = np.array([0.3167, 0.3334])
        Y = 100
        xy_n = np.array([0.3139, 0.3311])
        WT = whiteness_CIE2004(xy, Y, xy_n)

        xy = np.tile(xy, (6, 1))
        WT = np.tile(WT, (6, 1))
        np.testing.assert_almost_equal(
            whiteness_CIE2004(xy, Y, xy_n), WT, decimal=7)

        Y = np.tile(Y, 6)
        xy_n = np.tile(xy_n, (6, 1))
        np.testing.assert_almost_equal(
            whiteness_CIE2004(xy, Y, xy_n), WT, decimal=7)

        xy = np.reshape(xy, (2, 3, 2))
        Y = np.reshape(Y, (2, 3))
        xy_n = np.reshape(xy_n, (2, 3, 2))
        WT = np.reshape(WT, (2, 3, 2))
        np.testing.assert_almost_equal(
            whiteness_CIE2004(xy, Y, xy_n), WT, decimal=7)

    def test_domain_range_scale_whiteness_CIE2004(self):
        """
        Tests :func:`colour.colorimetry.whiteness.whiteness_CIE2004`
        definition domain and range scale support.
        """

        xy = np.array([0.3167, 0.3334])
        Y = 100
        xy_n = np.array([0.3139, 0.3311])
        WT = whiteness_CIE2004(xy, Y, xy_n)

        d_r = (('reference', 1), (1, 0.01), (100, 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    whiteness_CIE2004(xy, Y * factor, xy_n),
                    WT * factor,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_whiteness_CIE2004(self):
        """
        Tests :func:`colour.colorimetry.whiteness.whiteness_CIE2004`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            xy = np.array(case[0:2])
            Y = np.array(case[0])
            xy_n = np.array(case[0:2])
            whiteness_CIE2004(xy, Y, xy_n)


class TestWhiteness(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.whiteness.whiteness` definition unit
    tests methods.
    """

    def test_domain_range_scale_whiteness(self):
        """
        Tests :func:`colour.colorimetry.whiteness.whiteness` definition domain
        and range scale support.
        """

        XYZ = np.array([95.00000000, 100.00000000, 105.00000000])
        XYZ_0 = np.array([94.80966767, 100.00000000, 107.30513595])

        m = ('Berger 1959', 'Taube 1960', 'Stensby 1968', 'ASTM E313',
             'Ganz 1979', 'CIE 2004')
        v = [whiteness(XYZ, XYZ_0, method) for method in m]

        d_r = (('reference', 1), (1, 0.01), (100, 1))
        for method, value in zip(m, v):
            for scale, factor in d_r:
                with domain_range_scale(scale):
                    np.testing.assert_almost_equal(
                        whiteness(XYZ * factor, XYZ_0 * factor, method),
                        value * factor,
                        decimal=7)


if __name__ == '__main__':
    unittest.main()
