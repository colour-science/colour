# -*- coding: utf-8 -*-
"""
Defines the unit tests for the :mod:`colour.models.prolab` module.
"""

import numpy as np
import unittest
from itertools import permutations

from colour.models import XYZ_to_ProLab, ProLab_to_XYZ
from colour.utilities import ignore_numpy_errors

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
            np.array([0.83076696, 0.60122873, 0.22642636]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_ProLab(np.array([0.14222010, 0.23042768, 0.10495772])),
            np.array([1.37970163, -0.46384071, 0.28371029]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_ProLab(np.array([0.96907232, 1.00000000, 1.12179215])),
            np.array([6.92102743, 0.10732891, -0.0603479]),
            decimal=7)

    @ignore_numpy_errors
    def test_nan_XYZ_to_ProLab(self):
        """
        Tests :func:`colour.models.ProLab.XYZ_to_ProLab` definition nan
        support.
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

        np.testing.assert_allclose(
            ProLab_to_XYZ(np.array([0.83076696, 0.60122873, 0.22642636])),
            np.array([0.20654008, 0.12197225, 0.05136952]))

        np.testing.assert_allclose(
            ProLab_to_XYZ(np.array([1.37970163, -0.46384071, 0.28371029])),
            np.array([0.14222010, 0.23042768, 0.10495772]))

        np.testing.assert_allclose(
            ProLab_to_XYZ(np.array([6.92102743, 0.10732891, -0.0603479])),
            np.array([0.96907232, 1.00000000, 1.12179215]))

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
