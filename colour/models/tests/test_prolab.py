# -*- coding: utf-8 -*-
"""
Defines the unit tests for the :mod:`colour.models.prolab` module.
"""

import numpy as np
import unittest
from itertools import permutations

from colour.models import XYZ_to_ProLab, ProLab_to_XYZ
from colour.utilities import ignore_numpy_errors, domain_range_scale

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['TestXYZ_to_ProLab', 'TestProLab_to_XYZ']


class TestXYZ_to_ProLab(unittest.TestCase):
    """
    Defines :func:`colour.models.ProLab.TestXYZ_to_ProLab` definition unit
    tests methods.
    """

    def test_XYZ_to_ProLab(self):
        """
        Tests :func:`colour.models.ProLab.XYZ_to_ProLab` definition.
        """

        np.testing.assert_almost_equal(
            XYZ_to_ProLab(np.array([0.20654008, 0.12197225, 0.05136952])),
            np.array([48.7948929, 35.31503175, 13.30044932]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_ProLab(np.array([0.14222010, 0.23042768, 0.10495772])),
            np.array([64.45929636, -21.67007419, 13.25749056]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_ProLab(np.array([0.96907232, 1.00000000, 0.12179215])),
            np.array([100., 5.47367608, 37.26313098]),
            decimal=7)

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        ProLab = XYZ_to_ProLab(XYZ)

        d_r = (('reference', 1), (1, 1), (100, 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    XYZ_to_ProLab(XYZ * factor), ProLab * factor, decimal=7)

    def test_n_dimensional_XYZ_to_ProLab(self):
        """
        Tests :func:`colour.models.prolab.XYZ_to_ProLab` definition
        n-dimensional support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        ProLab = XYZ_to_ProLab(XYZ)

        XYZ = np.tile(XYZ, (6, 1))
        ProLab = np.tile(ProLab, (6, 1))
        np.testing.assert_almost_equal(XYZ_to_ProLab(XYZ), ProLab, decimal=7)

        XYZ = np.reshape(XYZ, (2, 3, 3))
        ProLab = np.reshape(ProLab, (2, 3, 3))
        np.testing.assert_almost_equal(XYZ_to_ProLab(XYZ), ProLab, decimal=7)

    @ignore_numpy_errors
    def test_nan_XYZ_to_ProLab(self):
        """
        Tests :func:`colour.models.ProLab.XYZ_to_ProLab` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ = np.array(case)
            XYZ_to_ProLab(XYZ)


class TestProLab_to_XYZ(unittest.TestCase):
    """
    Defines :func:`colour.models.ProLab.ProLab_to_XYZ` definition unit tests
    methods.
    """

    def test_ProLab_to_XYZ(self):
        """
        Tests :func:`colour.models.ProLab.ProLab_to_XYZ` definition.
        """

        np.testing.assert_almost_equal(
            ProLab_to_XYZ(np.array([0.83076696, 0.60122873, 0.22642636])),
            np.array([0.00206535, 0.00121972, 0.00051378]))

        np.testing.assert_almost_equal(
            ProLab_to_XYZ(np.array([0.37970163, -0.46384071, 0.28371029])),
            np.array([0.00010336, 0.00090972, -0.00045632]))

        np.testing.assert_almost_equal(
            ProLab_to_XYZ(np.array([6.92102743, 0.10732891, -0.0603479])),
            np.array([0.0096905, 0.00999991, 0.01122017]))

    def test_n_dimensional_XYZ_to_ProLab(self):
        """
        Tests :func:`colour.models.prolab.XYZ_to_ProLab` definition
        n-dimensional support.
        """

        ProLab = np.array([0.20654008, 0.12197225, 0.05136952])
        XYZ = ProLab_to_XYZ(ProLab)

        ProLab = np.tile(ProLab, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_almost_equal(ProLab_to_XYZ(ProLab), XYZ, decimal=7)

        ProLab = np.reshape(ProLab, (2, 3, 3))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_almost_equal(ProLab_to_XYZ(ProLab), XYZ, decimal=7)

    @ignore_numpy_errors
    def test_nan_ProLab_to_XYZ(self):
        """
        Tests :func:`colour.models.ProLab.ProLab_to_XYZ` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            ProLab = np.array(case)
            ProLab_to_XYZ(ProLab)


if __name__ == '__main__':
    unittest.main()
