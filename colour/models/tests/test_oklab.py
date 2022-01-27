# -*- coding: utf-8 -*-
"""
Defines the unit tests for the :mod:`colour.models.oklab` module.
"""

import numpy as np
import unittest
from itertools import permutations

from colour.models import XYZ_to_Oklab, Oklab_to_XYZ
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestXYZ_to_Oklab',
    'TestOklab_to_XYZ',
]


class TestXYZ_to_Oklab(unittest.TestCase):
    """
    Defines :func:`colour.models.oklab.TestXYZ_to_Oklab` definition unit
    tests methods.
    """

    def test_XYZ_to_Oklab(self):
        """
        Tests :func:`colour.models.oklab.XYZ_to_Oklab` definition.
        """

        np.testing.assert_almost_equal(
            XYZ_to_Oklab(np.array([0.20654008, 0.12197225, 0.05136952])),
            np.array([0.51634019, 0.15469500, 0.06289579]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Oklab(np.array([0.14222010, 0.23042768, 0.10495772])),
            np.array([0.59910746, -0.11139207, 0.07508465]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Oklab(np.array([0.96907232, 1.00000000, 1.12179215])),
            np.array([1.00121561, 0.00899591, -0.00535107]),
            decimal=7)

    def test_n_dimensional_XYZ_to_Oklab(self):
        """
        Tests :func:`colour.models.oklab.XYZ_to_Oklab` definition
        n-dimensional support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        Oklab = XYZ_to_Oklab(XYZ)

        XYZ = np.tile(XYZ, (6, 1))
        Oklab = np.tile(Oklab, (6, 1))
        np.testing.assert_almost_equal(XYZ_to_Oklab(XYZ), Oklab, decimal=7)

        XYZ = np.reshape(XYZ, (2, 3, 3))
        Oklab = np.reshape(Oklab, (2, 3, 3))
        np.testing.assert_almost_equal(XYZ_to_Oklab(XYZ), Oklab, decimal=7)

    def test_domain_range_scale_XYZ_to_Oklab(self):
        """
        Tests :func:`colour.models.oklab.XYZ_to_Oklab` definition domain and
        range scale support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        Oklab = XYZ_to_Oklab(XYZ)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    XYZ_to_Oklab(XYZ * factor), Oklab * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_XYZ_to_Oklab(self):
        """
        Tests :func:`colour.models.oklab.XYZ_to_Oklab` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ = np.array(case)
            XYZ_to_Oklab(XYZ)


class TestOklab_to_XYZ(unittest.TestCase):
    """
    Defines :func:`colour.models.oklab.Oklab_to_XYZ` definition unit tests
    methods.
    """

    def test_Oklab_to_XYZ(self):
        """
        Tests :func:`colour.models.oklab.Oklab_to_XYZ` definition.
        """

        np.testing.assert_allclose(
            Oklab_to_XYZ(np.array([0.51634019, 0.15469500, 0.06289579])),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            rtol=0.000001,
            atol=0.000001)

        np.testing.assert_allclose(
            Oklab_to_XYZ(np.array([0.59910746, -0.11139207, 0.07508465])),
            np.array([0.14222010, 0.23042768, 0.10495772]),
            rtol=0.000001,
            atol=0.000001)

        np.testing.assert_allclose(
            Oklab_to_XYZ(np.array([1.00121561, 0.00899591, -0.00535107])),
            np.array([0.96907232, 1.00000000, 1.12179215]),
            rtol=0.000001,
            atol=0.000001)

    def test_n_dimensional_Oklab_to_XYZ(self):
        """
        Tests :func:`colour.models.oklab.Oklab_to_XYZ` definition
        n-dimensional support.
        """

        Oklab = np.array([0.51634019, 0.15469500, 0.06289579])
        XYZ = Oklab_to_XYZ(Oklab)

        Oklab = np.tile(Oklab, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_allclose(
            Oklab_to_XYZ(Oklab), XYZ, rtol=0.000001, atol=0.000001)

        Oklab = np.reshape(Oklab, (2, 3, 3))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_allclose(
            Oklab_to_XYZ(Oklab), XYZ, rtol=0.000001, atol=0.000001)

    def test_domain_range_scale_Oklab_to_XYZ(self):
        """
        Tests :func:`colour.models.oklab.Oklab_to_XYZ` definition domain and
        range scale support.
        """

        Oklab = np.array([0.51634019, 0.15469500, 0.06289579])
        XYZ = Oklab_to_XYZ(Oklab)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    Oklab_to_XYZ(Oklab * factor),
                    XYZ * factor,
                    rtol=0.000001,
                    atol=0.000001)

    @ignore_numpy_errors
    def test_nan_Oklab_to_XYZ(self):
        """
        Tests :func:`colour.models.oklab.Oklab_to_XYZ` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            Oklab = np.array(case)
            Oklab_to_XYZ(Oklab)


if __name__ == '__main__':
    unittest.main()
