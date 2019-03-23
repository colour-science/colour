# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.models.cie_xyy` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from itertools import permutations

from colour.models import (XYZ_to_xyY, xyY_to_XYZ, xy_to_xyY, xyY_to_xy,
                           xy_to_XYZ, XYZ_to_xy)
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'TestXYZ_to_xyY', 'TestxyY_to_XYZ', 'TestxyY_to_xy', 'Testxy_to_xyY',
    'TestXYZ_to_xy', 'Testxy_to_XYZ'
]


class TestXYZ_to_xyY(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_xyy.XYZ_to_xyY` definition unit tests
    methods.
    """

    def test_XYZ_to_xyY(self):
        """
        Tests :func:`colour.models.cie_xyy.XYZ_to_xyY` definition.
        """

        np.testing.assert_almost_equal(
            XYZ_to_xyY(np.array([0.20654008, 0.12197225, 0.05136952])),
            np.array([0.54369557, 0.32107944, 0.12197225]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_xyY(np.array([0.14222010, 0.23042768, 0.10495772])),
            np.array([0.29777735, 0.48246446, 0.23042768]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_xyY(np.array([0.07818780, 0.06157201, 0.28099326])),
            np.array([0.18582823, 0.14633764, 0.06157201]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_xyY(np.array([0.00000000, 0.00000000, 1.00000000])),
            np.array([0.00000000, 0.00000000, 0.00000000]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_xyY(
                np.array([
                    [0.20654008, 0.12197225, 0.05136952],
                    [0.00000000, 0.00000000, 0.00000000],
                    [0.00000000, 1.00000000, 0.00000000],
                ])),
            np.array([
                [0.54369557, 0.32107944, 0.12197225],
                [0.31270000, 0.32900000, 0.00000000],
                [0.00000000, 1.00000000, 1.00000000],
            ]),
            decimal=7)

    def test_n_dimensional_XYZ_to_xyY(self):
        """
        Tests :func:`colour.models.cie_xyy.XYZ_to_xyY` definition n-dimensions
        support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        illuminant = np.array([0.31270, 0.32900])
        xyY = np.array([0.54369557, 0.32107944, 0.12197225])
        np.testing.assert_almost_equal(
            XYZ_to_xyY(XYZ, illuminant), xyY, decimal=7)

        XYZ = np.tile(XYZ, (6, 1))
        xyY = np.tile(xyY, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_xyY(XYZ, illuminant), xyY, decimal=7)

        illuminant = np.tile(illuminant, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_xyY(XYZ, illuminant), xyY, decimal=7)

        XYZ = np.reshape(XYZ, (2, 3, 3))
        illuminant = np.reshape(illuminant, (2, 3, 2))
        xyY = np.reshape(xyY, (2, 3, 3))
        np.testing.assert_almost_equal(
            XYZ_to_xyY(XYZ, illuminant), xyY, decimal=7)

    def test_domain_range_scale_XYZ_to_xyY(self):
        """
        Tests :func:`colour.models.cie_xyy.XYZ_to_xyY` definition domain and
        range scale support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        xyY = XYZ_to_xyY(XYZ)
        XYZ = np.tile(XYZ, (6, 1)).reshape(2, 3, 3)
        xyY = np.tile(xyY, (6, 1)).reshape(2, 3, 3)

        d_r = (('reference', 1, 1), (1, 1, 1), (
            100,
            100,
            np.array([1, 1, 100]),
        ))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    XYZ_to_xyY(XYZ * factor_a), xyY * factor_b, decimal=7)

    @ignore_numpy_errors
    def test_nan_XYZ_to_xyY(self):
        """
        Tests :func:`colour.models.cie_xyy.XYZ_to_xyY` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ = np.array(case)
            illuminant = np.array(case[0:2])
            XYZ_to_xyY(XYZ, illuminant)


class TestxyY_to_XYZ(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_xyy.xyY_to_XYZ` definition unit tests
    methods.
    """

    def test_xyY_to_XYZ(self):
        """
        Tests :func:`colour.models.cie_xyy.xyY_to_XYZ` definition.
        """

        np.testing.assert_almost_equal(
            xyY_to_XYZ(np.array([0.54369557, 0.32107944, 0.12197225])),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            decimal=7)

        np.testing.assert_almost_equal(
            xyY_to_XYZ(np.array([0.29777735, 0.48246446, 0.23042768])),
            np.array([0.14222010, 0.23042768, 0.10495772]),
            decimal=7)

        np.testing.assert_almost_equal(
            xyY_to_XYZ(np.array([0.18582823, 0.14633764, 0.06157201])),
            np.array([0.07818780, 0.06157201, 0.28099326]),
            decimal=7)

        np.testing.assert_almost_equal(
            xyY_to_XYZ(np.array([0.34567, 0.3585, 0.00000000])),
            np.array([0.00000000, 0.00000000, 0.00000000]),
            decimal=7)

        np.testing.assert_almost_equal(
            xyY_to_XYZ(
                np.array([
                    [0.54369557, 0.32107944, 0.12197225],
                    [0.31270000, 0.32900000, 0.00000000],
                    [0.00000000, 1.00000000, 1.00000000],
                ])),
            np.array([
                [0.20654008, 0.12197225, 0.05136952],
                [0.00000000, 0.00000000, 0.00000000],
                [0.00000000, 1.00000000, 0.00000000],
            ]),
            decimal=7)

    def test_n_dimensional_xyY_to_XYZ(self):
        """
        Tests :func:`colour.models.cie_xyy.xyY_to_XYZ` definition n-dimensions
        support.
        """

        xyY = np.array([0.54369557, 0.32107944, 0.12197225])
        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        np.testing.assert_almost_equal(xyY_to_XYZ(xyY), XYZ, decimal=7)

        xyY = np.tile(xyY, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_almost_equal(xyY_to_XYZ(xyY), XYZ, decimal=7)

        xyY = np.reshape(xyY, (2, 3, 3))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_almost_equal(xyY_to_XYZ(xyY), XYZ, decimal=7)

    def test_domain_range_scale_xyY_to_XYZ(self):
        """
        Tests :func:`colour.models.cie_xyy.xyY_to_XYZ` definition domain and
        range scale support.
        """

        xyY = np.array([0.54369557, 0.32107944, 0.12197225])
        XYZ = xyY_to_XYZ(xyY)
        xyY = np.tile(xyY, (6, 1)).reshape(2, 3, 3)
        XYZ = np.tile(XYZ, (6, 1)).reshape(2, 3, 3)

        d_r = (('reference', 1, 1), (1, 1, 1), (
            100,
            np.array([1, 1, 100]),
            100,
        ))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    xyY_to_XYZ(xyY * factor_a), XYZ * factor_b, decimal=7)

    @ignore_numpy_errors
    def test_nan_xyY_to_XYZ(self):
        """
        Tests :func:`colour.models.cie_xyy.xyY_to_XYZ` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            xyY = np.array(case)
            xyY_to_XYZ(xyY)


class TestxyY_to_xy(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_xyy.xyY_to_xy` definition unit tests
    methods.
    """

    def test_xyY_to_xy(self):
        """
        Tests :func:`colour.models.cie_xyy.xyY_to_xy` definition.
        """

        np.testing.assert_almost_equal(
            xyY_to_xy(np.array([0.54369557, 0.32107944, 0.12197225])),
            np.array([0.54369557, 0.32107944]),
            decimal=7)

        np.testing.assert_almost_equal(
            xyY_to_xy(np.array([0.29777735, 0.48246446, 0.23042768])),
            np.array([0.29777735, 0.48246446]),
            decimal=7)

        np.testing.assert_almost_equal(
            xyY_to_xy(np.array([0.18582823, 0.14633764, 0.06157201])),
            np.array([0.18582823, 0.14633764]),
            decimal=7)

        np.testing.assert_almost_equal(
            xyY_to_xy(np.array([0.31270, 0.32900])),
            np.array([0.31270000, 0.32900000]),
            decimal=7)

    def test_n_dimensional_xyY_to_xy(self):
        """
        Tests :func:`colour.models.cie_xyy.xyY_to_xy` definition n-dimensions
        support.
        """

        xyY = np.array([0.54369557, 0.32107944, 0.12197225])
        xy = np.array([0.54369557, 0.32107944])
        np.testing.assert_almost_equal(xyY_to_xy(xyY), xy, decimal=7)

        xyY = np.tile(xyY, (6, 1))
        xy = np.tile(xy, (6, 1))
        np.testing.assert_almost_equal(xyY_to_xy(xyY), xy, decimal=7)

        xyY = np.reshape(xyY, (2, 3, 3))
        xy = np.reshape(xy, (2, 3, 2))
        np.testing.assert_almost_equal(xyY_to_xy(xyY), xy, decimal=7)

    def test_domain_range_scale_xyY_to_xy(self):
        """
        Tests :func:`colour.models.cie_xyy.xyY_to_xy` definition domain and
        range scale support.
        """

        xyY = np.array([0.54369557, 0.32107944, 0.12197225])
        xy = xyY_to_xy(xyY)
        xyY = np.tile(xyY, (6, 1)).reshape(2, 3, 3)
        xy = np.tile(xy, (6, 1)).reshape(2, 3, 2)

        d_r = (('reference', 1, 1), (1, 1, 1), (100, np.array([1, 1, 100]), 1))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    xyY_to_xy(xyY * factor_a), xy * factor_b, decimal=7)

    @ignore_numpy_errors
    def test_nan_xyY_to_xy(self):
        """
        Tests :func:`colour.models.cie_xyy.xyY_to_xy` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=2))
        for case in cases:
            xyY = np.array(case)
            xyY_to_xy(xyY)


class Testxy_to_xyY(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_xyy.xy_to_xyY` definition unit tests
    methods.
    """

    def test_xy_to_xyY(self):
        """
        Tests :func:`colour.models.cie_xyy.xy_to_xyY` definition.
        """

        np.testing.assert_almost_equal(
            xy_to_xyY(np.array([0.54369557, 0.32107944])),
            np.array([0.54369557, 0.32107944, 1.00000000]),
            decimal=7)

        np.testing.assert_almost_equal(
            xy_to_xyY(np.array([0.29777735, 0.48246446])),
            np.array([0.29777735, 0.48246446, 1.00000000]),
            decimal=7)

        np.testing.assert_almost_equal(
            xy_to_xyY(np.array([0.18582823, 0.14633764])),
            np.array([0.18582823, 0.14633764, 1.00000000]),
            decimal=7)

        np.testing.assert_almost_equal(
            xy_to_xyY(np.array([0.31270000, 0.32900000, 1.00000000])),
            np.array([0.31270000, 0.32900000, 1.00000000]),
            decimal=7)

        np.testing.assert_almost_equal(
            xy_to_xyY(np.array([0.31270000, 0.32900000]), 100),
            np.array([0.31270000, 0.32900000, 100.00000000]),
            decimal=7)

    def test_n_dimensional_xy_to_xyY(self):
        """
        Tests :func:`colour.models.cie_xyy.xy_to_xyY` definition n-dimensions
        support.
        """

        xy = np.array([0.54369557, 0.32107944])
        XYZ = np.array([0.54369557, 0.32107944, 1.00000000])
        np.testing.assert_almost_equal(xy_to_xyY(xy), XYZ, decimal=7)

        xy = np.tile(xy, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_almost_equal(xy_to_xyY(xy), XYZ, decimal=7)

        xy = np.reshape(xy, (2, 3, 2))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_almost_equal(xy_to_xyY(xy), XYZ, decimal=7)

    def test_domain_range_scale_xy_to_xyY(self):
        """
        Tests :func:`colour.models.cie_xyy.xy_to_xyY` definition domain and
        range scale support.
        """

        xy = np.array([0.54369557, 0.32107944, 0.12197225])
        xyY = xy_to_xyY(xy)
        xy = np.tile(xy, (6, 1)).reshape(2, 3, 3)
        xyY = np.tile(xyY, (6, 1)).reshape(2, 3, 3)

        d_r = (('reference', 1, 1), (1, 1, 1), (
            100,
            np.array([1, 1, 100]),
            np.array([1, 1, 100]),
        ))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    xy_to_xyY(xy * factor_a), xyY * factor_b, decimal=7)

    @ignore_numpy_errors
    def test_nan_xy_to_xyY(self):
        """
        Tests :func:`colour.models.cie_xyy.xy_to_xyY` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=2))
        for case in cases:
            xy = np.array(case)
            xy_to_xyY(xy)


class TestXYZ_to_xy(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_xyy.XYZ_to_xy` definition unit tests
    methods.
    """

    def test_XYZ_to_xy(self):
        """
        Tests :func:`colour.models.cie_xyy.XYZ_to_xy` definition.
        """

        np.testing.assert_almost_equal(
            XYZ_to_xy(np.array([0.20654008, 0.12197225, 0.05136952])),
            np.array([0.54369557, 0.32107944]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_xy(np.array([0.14222010, 0.23042768, 0.10495772])),
            np.array([0.29777735, 0.48246446]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_xy(np.array([0.07818780, 0.06157201, 0.28099326])),
            np.array([0.18582823, 0.14633764]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_xy(np.array([0.00000000, 0.00000000, 0.00000000])),
            np.array([0.31270000, 0.32900000]),
            decimal=7)

    def test_n_dimensional_XYZ_to_xy(self):
        """
        Tests :func:`colour.models.cie_xyy.XYZ_to_xy` definition n-dimensions
        support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        illuminant = np.array([0.31270, 0.32900])
        xy = np.array([0.54369557, 0.32107944])
        np.testing.assert_almost_equal(
            XYZ_to_xy(XYZ, illuminant), xy, decimal=7)

        XYZ = np.tile(XYZ, (6, 1))
        xy = np.tile(xy, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_xy(XYZ, illuminant), xy, decimal=7)

        illuminant = np.tile(illuminant, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_xy(XYZ, illuminant), xy, decimal=7)

        XYZ = np.reshape(XYZ, (2, 3, 3))
        illuminant = np.reshape(xy, (2, 3, 2))
        xy = np.reshape(xy, (2, 3, 2))
        np.testing.assert_almost_equal(
            XYZ_to_xy(XYZ, illuminant), xy, decimal=7)

    def test_domain_range_scale_XYZ_to_xy(self):
        """
        Tests :func:`colour.models.cie_xyy.XYZ_to_xy` definition domain and
        range scale support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        xy = XYZ_to_xy(XYZ)
        XYZ = np.tile(XYZ, (6, 1)).reshape(2, 3, 3)
        xy = np.tile(xy, (6, 1)).reshape(2, 3, 2)

        d_r = (('reference', 1), (1, 1), (100, 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    XYZ_to_xy(XYZ * factor), xy, decimal=7)

    @ignore_numpy_errors
    def test_nan_XYZ_to_xy(self):
        """
        Tests :func:`colour.models.cie_xyy.XYZ_to_xy` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ = np.array(case)
            illuminant = np.array(case[0:2])
            XYZ_to_xy(XYZ, illuminant)


class Testxy_to_XYZ(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_xyy.xy_to_XYZ` definition unit tests
    methods.
    """

    def test_xy_to_XYZ(self):
        """
        Tests :func:`colour.models.cie_xyy.xy_to_XYZ` definition.
        """

        np.testing.assert_almost_equal(
            xy_to_XYZ(np.array([0.54369557, 0.32107944])),
            np.array([1.69333661, 1.00000000, 0.42115742]),
            decimal=7)

        np.testing.assert_almost_equal(
            xy_to_XYZ(np.array([0.29777735, 0.48246446])),
            np.array([0.61720059, 1.00000000, 0.45549094]),
            decimal=7)

        np.testing.assert_almost_equal(
            xy_to_XYZ(np.array([0.18582823, 0.14633764])),
            np.array([1.26985942, 1.00000000, 4.56365245]),
            decimal=7)

        np.testing.assert_almost_equal(
            xy_to_XYZ(np.array([0.31270000, 0.32900000])),
            np.array([0.95045593, 1.00000000, 1.08905775]),
            decimal=7)

    def test_n_dimensional_xy_to_XYZ(self):
        """
        Tests :func:`colour.models.cie_xyy.xy_to_XYZ` definition n-dimensions
        support.
        """

        xy = np.array([0.54369557, 0.32107944])
        XYZ = np.array([1.69333661, 1.00000000, 0.42115742])
        np.testing.assert_almost_equal(xy_to_XYZ(xy), XYZ, decimal=7)

        xy = np.tile(xy, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_almost_equal(xy_to_XYZ(xy), XYZ, decimal=7)

        xy = np.reshape(xy, (2, 3, 2))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_almost_equal(xy_to_XYZ(xy), XYZ, decimal=7)

    def test_domain_range_scale_xy_to_XYZ(self):
        """
        Tests :func:`colour.models.cie_xyy.xy_to_XYZ` definition domain and
        range scale support.
        """

        xy = np.array([0.54369557, 0.32107944, 0.12197225])
        XYZ = xy_to_XYZ(xy)
        xy = np.tile(xy, (6, 1)).reshape(2, 3, 3)
        XYZ = np.tile(XYZ, (6, 1)).reshape(2, 3, 3)

        d_r = (('reference', 1, 1), (1, 1, 1), (
            100,
            np.array([1, 1, 100]),
            100,
        ))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    xy_to_XYZ(xy * factor_a), XYZ * factor_b, decimal=7)

    @ignore_numpy_errors
    def test_nan_xy_to_XYZ(self):
        """
        Tests :func:`colour.models.cie_xyy.xy_to_XYZ` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=2))
        for case in cases:
            xy = np.array(case)
            xy_to_XYZ(xy)


if __name__ == '__main__':
    unittest.main()
