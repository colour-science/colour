# -*- coding: utf-8 -*-
"""
Defines the unit tests for the :mod:`colour.models.rgb.ictcp` module.
"""

import numpy as np
import unittest
from itertools import permutations

from colour.models.rgb import (
    RGB_to_ICtCp,
    ICtCp_to_RGB,
    XYZ_to_ICtCp,
    ICtCp_to_XYZ,
)
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestRGB_to_ICtCp',
    'TestICtCp_to_RGB',
    'TestXYZ_to_ICtCp',
    'TestICtCp_to_XYZ',
]


class TestRGB_to_ICtCp(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.ictcp.TestRGB_to_ICtCp` definition unit
    tests methods.
    """

    def test_RGB_to_ICtCp(self):
        """
        Tests :func:`colour.models.rgb.ictcp.RGB_to_ICtCp` definition.
        """

        np.testing.assert_almost_equal(
            RGB_to_ICtCp(np.array([0.45620519, 0.03081071, 0.04091952])),
            np.array([0.07351364, 0.00475253, 0.09351596]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_ICtCp(
                np.array([0.45620519, 0.03081071, 0.04091952]), L_p=4000),
            np.array([0.10516931, 0.00514031, 0.12318730]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_ICtCp(
                np.array([0.45620519, 0.03081071, 0.04091952]), L_p=1000),
            np.array([0.17079612, 0.00485580, 0.17431356]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_ICtCp(
                np.array([0.45620519, 0.03081071, 0.04091952]),
                method='ITU-R BT.2100-1 PQ'),
            np.array([0.07351364, 0.00475253, 0.09351596]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_ICtCp(
                np.array([0.45620519, 0.03081071, 0.04091952]),
                method='ITU-R BT.2100-2 PQ'),
            np.array([0.07351364, 0.00475253, 0.09351596]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_ICtCp(
                np.array([0.45620519, 0.03081071, 0.04091952]),
                method='ITU-R BT.2100-1 HLG'),
            np.array([0.62567899, -0.03622422, 0.67786522]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_ICtCp(
                np.array([0.45620519, 0.03081071, 0.04091952]),
                method='ITU-R BT.2100-2 HLG'),
            np.array([0.62567899, -0.01984490, 0.35911259]),
            decimal=7)

    def test_n_dimensional_RGB_to_ICtCp(self):
        """
        Tests :func:`colour.models.rgb.ictcp.RGB_to_ICtCp` definition
        n-dimensional support.
        """

        RGB = np.array([0.45620519, 0.03081071, 0.04091952])
        ICtCp = RGB_to_ICtCp(RGB)

        RGB = np.tile(RGB, (6, 1))
        ICtCp = np.tile(ICtCp, (6, 1))
        np.testing.assert_almost_equal(RGB_to_ICtCp(RGB), ICtCp, decimal=7)

        RGB = np.reshape(RGB, (2, 3, 3))
        ICtCp = np.reshape(ICtCp, (2, 3, 3))
        np.testing.assert_almost_equal(RGB_to_ICtCp(RGB), ICtCp, decimal=7)

    def test_domain_range_scale_RGB_to_ICtCp(self):
        """
        Tests :func:`colour.models.rgb.ictcp.RGB_to_ICtCp` definition domain
        and range scale support.
        """

        RGB = np.array([0.45620519, 0.03081071, 0.04091952])
        ICtCp = RGB_to_ICtCp(RGB)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    RGB_to_ICtCp(RGB * factor), ICtCp * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_RGB_to_ICtCp(self):
        """
        Tests :func:`colour.models.rgb.ictcp.RGB_to_ICtCp` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            RGB = np.array(case)
            RGB_to_ICtCp(RGB)


class TestICtCp_to_RGB(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.ictcp.ICtCp_to_RGB` definition unit tests
    methods.
    """

    def test_ICtCp_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.ictcp.ICtCp_to_RGB` definition.
        """

        np.testing.assert_almost_equal(
            ICtCp_to_RGB(np.array([0.07351364, 0.00475253, 0.09351596])),
            np.array([0.45620519, 0.03081071, 0.04091952]),
            decimal=7)

        np.testing.assert_almost_equal(
            ICtCp_to_RGB(
                np.array([0.10516931, 0.00514031, 0.12318730]), L_p=4000),
            np.array([0.45620519, 0.03081071, 0.04091952]),
            decimal=7)

        np.testing.assert_almost_equal(
            ICtCp_to_RGB(
                np.array([0.17079612, 0.00485580, 0.17431356]), L_p=1000),
            np.array([0.45620519, 0.03081071, 0.04091952]),
            decimal=7)

        np.testing.assert_almost_equal(
            ICtCp_to_RGB(
                np.array([0.07351364, 0.00475253, 0.09351596]),
                method='ITU-R BT.2100-1 PQ'),
            np.array([0.45620519, 0.03081071, 0.04091952]),
            decimal=7)

        np.testing.assert_almost_equal(
            ICtCp_to_RGB(
                np.array([0.07351364, 0.00475253, 0.09351596]),
                method='ITU-R BT.2100-2 PQ'),
            np.array([0.45620519, 0.03081071, 0.04091952]),
            decimal=7)

        np.testing.assert_almost_equal(
            ICtCp_to_RGB(
                np.array([0.62567899, -0.03622422, 0.67786522]),
                method='ITU-R BT.2100-1 HLG'),
            np.array([0.45620519, 0.03081071, 0.04091952]),
            decimal=7)

        np.testing.assert_almost_equal(
            ICtCp_to_RGB(
                np.array([0.62567899, -0.01984490, 0.35911259]),
                method='ITU-R BT.2100-2 HLG'),
            np.array([0.45620519, 0.03081071, 0.04091952]),
            decimal=7)

    def test_n_dimensional_ICtCp_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.ictcp.ICtCp_to_RGB` definition
        n-dimensional support.
        """

        ICtCp = np.array([0.07351364, 0.00475253, 0.09351596])
        RGB = ICtCp_to_RGB(ICtCp)

        ICtCp = np.tile(ICtCp, (6, 1))
        RGB = np.tile(RGB, (6, 1))
        np.testing.assert_almost_equal(ICtCp_to_RGB(ICtCp), RGB, decimal=7)

        ICtCp = np.reshape(ICtCp, (2, 3, 3))
        RGB = np.reshape(RGB, (2, 3, 3))
        np.testing.assert_almost_equal(ICtCp_to_RGB(ICtCp), RGB, decimal=7)

    def test_domain_range_scale_ICtCp_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.ictcp.ICtCp_to_RGB` definition domain
        and range scale support.
        """

        ICtCp = np.array([0.07351364, 0.00475253, 0.09351596])
        RGB = ICtCp_to_RGB(ICtCp)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    ICtCp_to_RGB(ICtCp * factor), RGB * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_ICtCp_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.ictcp.ICtCp_to_RGB` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            ICtCp = np.array(case)
            ICtCp_to_RGB(ICtCp)


class TestXYZ_to_ICtCp(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.ictcp.TestXYZ_to_ICtCp` definition unit
    tests methods.
    """

    def test_XYZ_to_ICtCp(self):
        """
        Tests :func:`colour.models.rgb.ictcp.XYZ_to_ICtCp` definition.
        """

        np.testing.assert_almost_equal(
            XYZ_to_ICtCp(np.array([0.20654008, 0.12197225, 0.05136952])),
            np.array([0.06858097, -0.00283842, 0.06020983]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_ICtCp(
                np.array([0.20654008, 0.12197225, 0.05136952]),
                np.array([0.34570, 0.35850])),
            np.array([0.06792437, 0.00452089, 0.05514480]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_ICtCp(
                np.array([0.20654008, 0.12197225, 0.05136952]),
                np.array([0.34570, 0.35850]),
                chromatic_adaptation_transform='Bradford'),
            np.array([0.06783951, 0.00476111, 0.05523093]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_ICtCp(
                np.array([0.20654008, 0.12197225, 0.05136952]), L_p=4000),
            np.array([0.09871102, -0.00447247, 0.07984812]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_ICtCp(
                np.array([0.20654008, 0.12197225, 0.05136952]), L_p=1000),
            np.array([0.16173872, -0.00792543, 0.11409458]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_ICtCp(
                np.array([0.20654008, 0.12197225, 0.05136952]),
                method='ITU-R BT.2100-1 PQ'),
            np.array([0.06858097, -0.00283842, 0.06020983]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_ICtCp(
                np.array([0.20654008, 0.12197225, 0.05136952]),
                method='ITU-R BT.2100-2 PQ'),
            np.array([0.06858097, -0.00283842, 0.06020983]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_ICtCp(
                np.array([0.20654008, 0.12197225, 0.05136952]),
                method='ITU-R BT.2100-1 HLG'),
            np.array([0.59242792, -0.06824263, 0.47421473]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_ICtCp(
                np.array([0.20654008, 0.12197225, 0.05136952]),
                method='ITU-R BT.2100-2 HLG'),
            np.array([0.59242792, -0.03740730, 0.25122675]),
            decimal=7)

    def test_n_dimensional_XYZ_to_ICtCp(self):
        """
        Tests :func:`colour.models.rgb.ictcp.XYZ_to_ICtCp` definition
        n-dimensional support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        ICtCp = XYZ_to_ICtCp(XYZ)

        XYZ = np.tile(XYZ, (6, 1))
        ICtCp = np.tile(ICtCp, (6, 1))
        np.testing.assert_almost_equal(XYZ_to_ICtCp(XYZ), ICtCp, decimal=7)

        XYZ = np.reshape(XYZ, (2, 3, 3))
        ICtCp = np.reshape(ICtCp, (2, 3, 3))
        np.testing.assert_almost_equal(XYZ_to_ICtCp(XYZ), ICtCp, decimal=7)

    def test_domain_range_scale_XYZ_to_ICtCp(self):
        """
        Tests :func:`colour.models.rgb.ictcp.XYZ_to_ICtCp` definition domain
        and range scale support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        ICtCp = XYZ_to_ICtCp(XYZ)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    XYZ_to_ICtCp(XYZ * factor), ICtCp * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_XYZ_to_ICtCp(self):
        """
        Tests :func:`colour.models.rgb.ictcp.XYZ_to_ICtCp` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ = np.array(case)
            XYZ_to_ICtCp(XYZ)


class TestICtCp_to_XYZ(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.ictcp.ICtCp_to_XYZ` definition unit tests
    methods.
    """

    def test_ICtCp_to_XYZ(self):
        """
        Tests :func:`colour.models.rgb.ictcp.ICtCp_to_XYZ` definition.
        """

        np.testing.assert_almost_equal(
            ICtCp_to_XYZ(np.array([0.06858097, -0.00283842, 0.06020983])),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            decimal=7)

        np.testing.assert_almost_equal(
            ICtCp_to_XYZ(
                np.array([0.06792437, 0.00452089, 0.05514480]),
                np.array([0.34570, 0.35850])),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            decimal=7)

        np.testing.assert_almost_equal(
            ICtCp_to_XYZ(
                np.array([0.06783951, 0.00476111, 0.05523093]),
                np.array([0.34570, 0.35850]),
                chromatic_adaptation_transform='Bradford'),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            decimal=7)

        np.testing.assert_almost_equal(
            ICtCp_to_XYZ(
                np.array([0.09871102, -0.00447247, 0.07984812]), L_p=4000),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            decimal=7)

        np.testing.assert_almost_equal(
            ICtCp_to_XYZ(
                np.array([0.16173872, -0.00792543, 0.11409458]), L_p=1000),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            decimal=7)

        np.testing.assert_almost_equal(
            ICtCp_to_XYZ(
                np.array([0.06858097, -0.00283842, 0.06020983]),
                method='ITU-R BT.2100-1 PQ'),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            decimal=7)

        np.testing.assert_almost_equal(
            ICtCp_to_XYZ(
                np.array([0.06858097, -0.00283842, 0.06020983]),
                method='ITU-R BT.2100-2 PQ'),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            decimal=7)

        np.testing.assert_almost_equal(
            ICtCp_to_XYZ(
                np.array([0.59242792, -0.06824263, 0.47421473]),
                method='ITU-R BT.2100-1 HLG'),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            decimal=7)

        np.testing.assert_almost_equal(
            ICtCp_to_XYZ(
                np.array([0.59242792, -0.03740730, 0.25122675]),
                method='ITU-R BT.2100-2 HLG'),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            decimal=7)

    def test_n_dimensional_ICtCp_to_XYZ(self):
        """
        Tests :func:`colour.models.rgb.ictcp.ICtCp_to_XYZ` definition
        n-dimensional support.
        """

        ICtCp = np.array([0.06858097, -0.00283842, 0.06020983])
        XYZ = ICtCp_to_XYZ(ICtCp)

        ICtCp = np.tile(ICtCp, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_almost_equal(ICtCp_to_XYZ(ICtCp), XYZ, decimal=7)

        ICtCp = np.reshape(ICtCp, (2, 3, 3))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_almost_equal(ICtCp_to_XYZ(ICtCp), XYZ, decimal=7)

    def test_domain_range_scale_ICtCp_to_XYZ(self):
        """
        Tests :func:`colour.models.rgb.ictcp.ICtCp_to_XYZ` definition domain
        and range scale support.
        """

        ICtCp = np.array([0.06858097, -0.00283842, 0.06020983])
        XYZ = ICtCp_to_XYZ(ICtCp)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    ICtCp_to_XYZ(ICtCp * factor), XYZ * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_ICtCp_to_XYZ(self):
        """
        Tests :func:`colour.models.rgb.ictcp.ICtCp_to_XYZ` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            ICtCp = np.array(case)
            ICtCp_to_XYZ(ICtCp)


if __name__ == '__main__':
    unittest.main()
