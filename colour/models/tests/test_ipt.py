# -*- coding: utf-8 -*-
"""
Defines the unit tests for the :mod:`colour.models.ipt` module.
"""

import numpy as np
import unittest
from itertools import permutations

from colour.models import XYZ_to_IPT, IPT_to_XYZ, IPT_hue_angle
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestXYZ_to_IPT',
    'TestIPT_to_XYZ',
    'TestIPTHueAngle',
]


class TestXYZ_to_IPT(unittest.TestCase):
    """
    Defines :func:`colour.models.ipt.XYZ_to_IPT` definition unit tests methods.
    """

    def test_XYZ_to_IPT(self):
        """
        Tests :func:`colour.models.ipt.XYZ_to_IPT` definition.
        """

        np.testing.assert_almost_equal(
            XYZ_to_IPT(np.array([0.20654008, 0.12197225, 0.05136952])),
            np.array([0.38426191, 0.38487306, 0.18886838]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_IPT(np.array([0.14222010, 0.23042768, 0.10495772])),
            np.array([0.49437481, -0.19251742, 0.18080304]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_IPT(np.array([0.07818780, 0.06157201, 0.28099326])),
            np.array([0.35167774, -0.07525627, -0.30921279]),
            decimal=7)

    def test_n_dimensional_XYZ_to_IPT(self):
        """
        Tests :func:`colour.models.ipt.XYZ_to_IPT` definition n-dimensional
        support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        IPT = XYZ_to_IPT(XYZ)

        XYZ = np.tile(XYZ, (6, 1))
        IPT = np.tile(IPT, (6, 1))
        np.testing.assert_almost_equal(XYZ_to_IPT(XYZ), IPT, decimal=7)

        XYZ = np.reshape(XYZ, (2, 3, 3))
        IPT = np.reshape(IPT, (2, 3, 3))
        np.testing.assert_almost_equal(XYZ_to_IPT(XYZ), IPT, decimal=7)

    def test_domain_range_scale_XYZ_to_IPT(self):
        """
        Tests :func:`colour.models.ipt.XYZ_to_IPT` definition domain and
        range scale support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        IPT = XYZ_to_IPT(XYZ)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    XYZ_to_IPT(XYZ * factor), IPT * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_XYZ_to_IPT(self):
        """
        Tests :func:`colour.models.ipt.XYZ_to_IPT` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ = np.array(case)
            XYZ_to_IPT(XYZ)


class TestIPT_to_XYZ(unittest.TestCase):
    """
    Defines :func:`colour.models.ipt.IPT_to_XYZ` definition unit tests
    methods.
    """

    def test_IPT_to_XYZ(self):
        """
        Tests :func:`colour.models.ipt.IPT_to_XYZ` definition.
        """

        np.testing.assert_almost_equal(
            IPT_to_XYZ(np.array([0.38426191, 0.38487306, 0.18886838])),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            decimal=7)

        np.testing.assert_almost_equal(
            IPT_to_XYZ(np.array([0.49437481, -0.19251742, 0.18080304])),
            np.array([0.14222010, 0.23042768, 0.10495772]),
            decimal=7)

        np.testing.assert_almost_equal(
            IPT_to_XYZ(np.array([0.35167774, -0.07525627, -0.30921279])),
            np.array([0.07818780, 0.06157201, 0.28099326]),
            decimal=7)

    def test_n_dimensional_IPT_to_XYZ(self):
        """
        Tests :func:`colour.models.ipt.IPT_to_XYZ` definition n-dimensional
        support.
        """

        IPT = np.array([0.38426191, 0.38487306, 0.18886838])
        XYZ = IPT_to_XYZ(IPT)

        IPT = np.tile(IPT, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_almost_equal(IPT_to_XYZ(IPT), XYZ, decimal=7)

        IPT = np.reshape(IPT, (2, 3, 3))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_almost_equal(IPT_to_XYZ(IPT), XYZ, decimal=7)

    def test_domain_range_scale_IPT_to_XYZ(self):
        """
        Tests :func:`colour.models.ipt.IPT_to_XYZ` definition domain and
        range scale support.
        """

        IPT = np.array([0.38426191, 0.38487306, 0.18886838])
        XYZ = IPT_to_XYZ(IPT)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    IPT_to_XYZ(IPT * factor), XYZ * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_IPT_to_XYZ(self):
        """
        Tests :func:`colour.models.ipt.IPT_to_XYZ` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            IPT = np.array(case)
            IPT_to_XYZ(IPT)


class TestIPTHueAngle(unittest.TestCase):
    """
    Defines :func:`colour.models.ipt.IPT_hue_angle` definition unit tests
    methods.
    """

    def test_IPT_hue_angle(self):
        """
        Tests :func:`colour.models.ipt.IPT_hue_angle` definition.
        """

        np.testing.assert_almost_equal(
            IPT_hue_angle(np.array([0.20654008, 0.12197225, 0.05136952])),
            22.838754548625527,
            decimal=7)

        np.testing.assert_almost_equal(
            IPT_hue_angle(np.array([0.14222010, 0.23042768, 0.10495772])),
            24.488834912466245,
            decimal=7)

        np.testing.assert_almost_equal(
            IPT_hue_angle(np.array([0.07818780, 0.06157201, 0.28099326])),
            77.640533743711813,
            decimal=7)

    def test_n_dimensional_IPT_hue_angle(self):
        """
        Tests :func:`colour.models.ipt.IPT_hue_angle` definition n-dimensional
        support.
        """

        IPT = np.array([0.20654008, 0.12197225, 0.05136952])
        hue = IPT_hue_angle(IPT)

        IPT = np.tile(IPT, (6, 1))
        hue = np.tile(hue, 6)
        np.testing.assert_almost_equal(IPT_hue_angle(IPT), hue, decimal=7)

        IPT = np.reshape(IPT, (2, 3, 3))
        hue = np.reshape(hue, (2, 3))
        np.testing.assert_almost_equal(IPT_hue_angle(IPT), hue, decimal=7)

    def test_domain_range_scale_IPT_hue_angle(self):
        """
        Tests :func:`colour.models.ipt.IPT_hue_angle` definition domain and
        range scale support.
        """

        IPT = np.array([0.20654008, 0.12197225, 0.05136952])
        hue = IPT_hue_angle(IPT)

        d_r = (('reference', 1, 1), ('1', 1, 1 / 360), ('100', 100, 1 / 3.6))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    IPT_hue_angle(IPT * factor_a), hue * factor_b, decimal=7)

    @ignore_numpy_errors
    def test_nan_IPT_hue_angle(self):
        """
        Tests :func:`colour.models.ipt.IPT_hue_angle` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            IPT = np.array(case)
            IPT_hue_angle(IPT)


if __name__ == '__main__':
    unittest.main()
