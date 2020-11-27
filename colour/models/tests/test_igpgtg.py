# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.models.igpgtg` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from itertools import permutations

from colour.models import XYZ_to_IGPGTG, IGPGTG_to_XYZ
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['TestXYZ_to_IGPGTG', 'TestIGPGTG_to_XYZ']


class TestXYZ_to_IGPGTG(unittest.TestCase):
    """
    Defines :func:`colour.models.igpgtg.XYZ_to_IGPGTG` definition unit tests
    methods.
    """

    def test_XYZ_to_IGPGTG(self):
        """
        Tests :func:`colour.models.igpgtg.XYZ_to_IGPGTG` definition.
        """

        np.testing.assert_almost_equal(
            XYZ_to_IGPGTG(np.array([0.20654008, 0.12197225, 0.05136952])),
            np.array([0.42421258, 0.18632491, 0.10689223]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_IGPGTG(np.array([0.14222010, 0.23042768, 0.10495772])),
            np.array([0.50912820, -0.14804331, 0.11921472]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_IGPGTG(np.array([0.07818780, 0.06157201, 0.28099326])),
            np.array([0.29095152, -0.04057508, -0.18220795]),
            decimal=7)

    def test_n_dimensional_XYZ_to_IGPGTG(self):
        """
        Tests :func:`colour.models.igpgtg.XYZ_to_IGPGTG` definition
        n-dimensional support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        IGPGTG = XYZ_to_IGPGTG(XYZ)

        XYZ = np.tile(XYZ, (6, 1))
        IGPGTG = np.tile(IGPGTG, (6, 1))
        np.testing.assert_almost_equal(XYZ_to_IGPGTG(XYZ), IGPGTG, decimal=7)

        XYZ = np.reshape(XYZ, (2, 3, 3))
        IGPGTG = np.reshape(IGPGTG, (2, 3, 3))
        np.testing.assert_almost_equal(XYZ_to_IGPGTG(XYZ), IGPGTG, decimal=7)

    def test_domain_range_scale_XYZ_to_IGPGTG(self):
        """
        Tests :func:`colour.models.igpgtg.XYZ_to_IGPGTG` definition domain and
        range scale support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        IGPGTG = XYZ_to_IGPGTG(XYZ)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    XYZ_to_IGPGTG(XYZ * factor), IGPGTG * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_XYZ_to_IGPGTG(self):
        """
        Tests :func:`colour.models.igpgtg.XYZ_to_IGPGTG` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ = np.array(case)
            XYZ_to_IGPGTG(XYZ)


class TestIGPGTG_to_XYZ(unittest.TestCase):
    """
    Defines :func:`colour.models.igpgtg.IGPGTG_to_XYZ` definition unit tests
    methods.
    """

    def test_IGPGTG_to_XYZ(self):
        """
        Tests :func:`colour.models.igpgtg.IGPGTG_to_XYZ` definition.
        """

        np.testing.assert_almost_equal(
            IGPGTG_to_XYZ(np.array([0.42421258, 0.18632491, 0.10689223])),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            decimal=7)

        np.testing.assert_almost_equal(
            IGPGTG_to_XYZ(np.array([0.50912820, -0.14804331, 0.11921472])),
            np.array([0.14222010, 0.23042768, 0.10495772]),
            decimal=7)

        np.testing.assert_almost_equal(
            IGPGTG_to_XYZ(np.array([0.29095152, -0.04057508, -0.18220795])),
            np.array([0.07818780, 0.06157201, 0.28099326]),
            decimal=7)

    def test_n_dimensional_IGPGTG_to_XYZ(self):
        """
        Tests :func:`colour.models.igpgtg.IGPGTG_to_XYZ` definition
        n-dimensional support.
        """

        IGPGTG = np.array([0.42421258, 0.18632491, 0.10689223])
        XYZ = IGPGTG_to_XYZ(IGPGTG)

        IGPGTG = np.tile(IGPGTG, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_almost_equal(IGPGTG_to_XYZ(IGPGTG), XYZ, decimal=7)

        IGPGTG = np.reshape(IGPGTG, (2, 3, 3))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_almost_equal(IGPGTG_to_XYZ(IGPGTG), XYZ, decimal=7)

    def test_domain_range_scale_IGPGTG_to_XYZ(self):
        """
        Tests :func:`colour.models.igpgtg.IGPGTG_to_XYZ` definition domain and
        range scale support.
        """

        IGPGTG = np.array([0.42421258, 0.18632491, 0.10689223])
        XYZ = IGPGTG_to_XYZ(IGPGTG)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    IGPGTG_to_XYZ(IGPGTG * factor), XYZ * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_IGPGTG_to_XYZ(self):
        """
        Tests :func:`colour.models.igpgtg.IGPGTG_to_XYZ` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            IGPGTG = np.array(case)
            IGPGTG_to_XYZ(IGPGTG)


if __name__ == '__main__':
    unittest.main()
