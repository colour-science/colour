# -*- coding: utf-8 -*-
"""
Defines the unit tests for the :mod:`colour.models.osa_ucs` module.
"""

import numpy as np
import unittest
from itertools import permutations

from colour.models import XYZ_to_OSA_UCS, OSA_UCS_to_XYZ
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestXYZ_to_OSA_UCS',
    'TestOSA_UCS_to_XYZ',
]


class TestXYZ_to_OSA_UCS(unittest.TestCase):
    """
    Defines :func:`colour.models.osa_ucs.XYZ_to_OSA_UCS` definition unit tests
    methods.
    """

    def test_XYZ_to_OSA_UCS(self):
        """
        Tests :func:`colour.models.osa_ucs.XYZ_to_OSA_UCS` definition.
        """

        np.testing.assert_almost_equal(
            XYZ_to_OSA_UCS(
                np.array([0.20654008, 0.12197225, 0.05136952]) * 100),
            np.array([-3.00499790, 2.99713697, -9.66784231]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_OSA_UCS(
                np.array([0.14222010, 0.23042768, 0.10495772]) * 100),
            np.array([-1.64657491, 4.59201565, 5.31738757]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_OSA_UCS(
                np.array([0.07818780, 0.06157201, 0.28099326]) * 100),
            np.array([-5.08589672, -7.91062749, 0.98107575]),
            decimal=7)

    def test_n_dimensional_XYZ_to_OSA_UCS(self):
        """
        Tests :func:`colour.models.osa_ucs.XYZ_to_OSA_UCS` definition
        n-dimensional support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952]) * 100
        Ljg = XYZ_to_OSA_UCS(XYZ)

        XYZ = np.tile(XYZ, (6, 1))
        Ljg = np.tile(Ljg, (6, 1))
        np.testing.assert_almost_equal(XYZ_to_OSA_UCS(XYZ), Ljg, decimal=7)

        XYZ = np.reshape(XYZ, (2, 3, 3))
        Ljg = np.reshape(Ljg, (2, 3, 3))
        np.testing.assert_almost_equal(XYZ_to_OSA_UCS(XYZ), Ljg, decimal=7)

    def test_domain_range_scale_XYZ_to_OSA_UCS(self):
        """
        Tests :func:`colour.models.osa_ucs.XYZ_to_OSA_UCS` definition domain
        and range scale support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952]) * 100
        Ljg = XYZ_to_OSA_UCS(XYZ)

        d_r = (('reference', 1), ('1', 0.01), ('100', 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    XYZ_to_OSA_UCS(XYZ * factor), Ljg * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_XYZ_to_OSA_UCS(self):
        """
        Tests :func:`colour.models.osa_ucs.XYZ_to_OSA_UCS` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ_to_OSA_UCS(np.array(case))


class TestOSA_UCS_to_XYZ(unittest.TestCase):
    """
    Defines :func:`colour.models.osa_ucs.OSA_UCS_to_XYZ` definition unit tests
    methods.
    """

    def test_OSA_UCS_to_XYZ(self):
        """
        Tests :func:`colour.models.osa_ucs.OSA_UCS_to_XYZ` definition.
        """

        np.testing.assert_allclose(
            OSA_UCS_to_XYZ(
                np.array([-3.00499790, 2.99713697, -9.66784231]),
                {'disp': False}),
            np.array([0.20654008, 0.12197225, 0.05136952]) * 100,
            rtol=0.00001,
            atol=0.00001)

        np.testing.assert_allclose(
            OSA_UCS_to_XYZ(
                np.array([-1.64657491, 4.59201565, 5.31738757]),
                {'disp': False}),
            np.array([0.14222010, 0.23042768, 0.10495772]) * 100,
            rtol=0.00001,
            atol=0.00001)

        np.testing.assert_allclose(
            OSA_UCS_to_XYZ(
                np.array([-5.08589672, -7.91062749, 0.98107575]),
                {'disp': False}),
            np.array([0.07818780, 0.06157201, 0.28099326]) * 100,
            rtol=0.00001,
            atol=0.00001)

    def test_n_dimensional_OSA_UCS_to_XYZ(self):
        """
        Tests :func:`colour.models.osa_ucs.OSA_UCS_to_XYZ` definition
        n-dimensional support.
        """

        Ljg = np.array([-3.00499790, 2.99713697, -9.66784231])
        XYZ = OSA_UCS_to_XYZ(Ljg)

        Ljg = np.tile(Ljg, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_allclose(
            OSA_UCS_to_XYZ(Ljg), XYZ, rtol=0.00001, atol=0.00001)

        Ljg = np.reshape(Ljg, (2, 3, 3))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_allclose(
            OSA_UCS_to_XYZ(Ljg), XYZ, rtol=0.00001, atol=0.00001)

    def test_domain_range_scale_OSA_UCS_to_XYZ(self):
        """
        Tests :func:`colour.models.osa_ucs.OSA_UCS_to_XYZ` definition domain
        and range scale support.
        """

        Ljg = np.array([-3.00499790, 2.99713697, -9.66784231])
        XYZ = OSA_UCS_to_XYZ(Ljg)

        d_r = (('reference', 1), ('1', 0.01), ('100', 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    OSA_UCS_to_XYZ(Ljg * factor), XYZ * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_OSA_UCS_to_XYZ(self):
        """
        Tests :func:`colour.models.osa_ucs.OSA_UCS_to_XYZ` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            OSA_UCS_to_XYZ(np.array(case))


if __name__ == '__main__':
    unittest.main()
