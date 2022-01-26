# -*- coding: utf-8 -*-
"""
Defines the unit tests for the :mod:`colour.models.cie_ucs` module.
"""

import numpy as np
import unittest
from itertools import permutations

from colour.models import (
    XYZ_to_UCS,
    UCS_to_XYZ,
    UCS_to_uv,
    uv_to_UCS,
    UCS_uv_to_xy,
    xy_to_UCS_uv,
)
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestXYZ_to_UCS',
    'TestUCS_to_XYZ',
    'TestUCS_to_uv',
    'Testuv_to_UCS',
    'TestUCS_uv_to_xy',
    'TestXy_to_UCS_uv',
]


class TestXYZ_to_UCS(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_ucs.XYZ_to_UCS` definition unit tests
    methods.
    """

    def test_XYZ_to_UCS(self):
        """
        Tests :func:`colour.models.cie_ucs.XYZ_to_UCS` definition.
        """

        np.testing.assert_almost_equal(
            XYZ_to_UCS(np.array([0.20654008, 0.12197225, 0.05136952])),
            np.array([0.13769339, 0.12197225, 0.10537310]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_UCS(np.array([0.14222010, 0.23042768, 0.10495772])),
            np.array([0.09481340, 0.23042768, 0.32701033]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_UCS(np.array([0.07818780, 0.06157201, 0.28099326])),
            np.array([0.05212520, 0.06157201, 0.19376075]),
            decimal=7)

    def test_n_dimensional_XYZ_to_UCS(self):
        """
        Tests :func:`colour.models.cie_ucs.XYZ_to_UCS` definition n-dimensional
        support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        UCS = XYZ_to_UCS(XYZ)

        UCS = np.tile(UCS, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_almost_equal(XYZ_to_UCS(XYZ), UCS, decimal=7)

        UCS = np.reshape(UCS, (2, 3, 3))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_almost_equal(XYZ_to_UCS(XYZ), UCS, decimal=7)

    def test_domain_range_scale_XYZ_to_UCS(self):
        """
        Tests :func:`colour.models.cie_ucs.XYZ_to_UCS` definition domain and
        range scale support.
        """

        XYZ = np.array([0.0704953400, 0.1008000000, 0.0955831300])
        UCS = XYZ_to_UCS(XYZ)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    XYZ_to_UCS(XYZ * factor), UCS * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_XYZ_to_UCS(self):
        """
        Tests :func:`colour.models.cie_ucs.XYZ_to_UCS` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ = np.array(case)
            XYZ_to_UCS(XYZ)


class TestUCS_to_XYZ(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_ucs.UCS_to_XYZ` definition unit tests
    methods.
    """

    def test_UCS_to_XYZ(self):
        """
        Tests :func:`colour.models.cie_ucs.UCS_to_XYZ` definition.
        """

        np.testing.assert_almost_equal(
            UCS_to_XYZ(np.array([0.13769339, 0.12197225, 0.10537310])),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            decimal=7)

        np.testing.assert_almost_equal(
            UCS_to_XYZ(np.array([0.09481340, 0.23042768, 0.32701033])),
            np.array([0.14222010, 0.23042768, 0.10495772]),
            decimal=7)

        np.testing.assert_almost_equal(
            UCS_to_XYZ(np.array([0.05212520, 0.06157201, 0.19376075])),
            np.array([0.07818780, 0.06157201, 0.28099326]),
            decimal=7)

    def test_n_dimensional_UCS_to_XYZ(self):
        """
        Tests :func:`colour.models.cie_ucs.UCS_to_XYZ` definition n-dimensional
        support.
        """

        UCS = np.array([0.13769339, 0.12197225, 0.10537310])
        XYZ = UCS_to_XYZ(UCS)

        UCS = np.tile(UCS, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_almost_equal(UCS_to_XYZ(UCS), XYZ, decimal=7)

        UCS = np.reshape(UCS, (2, 3, 3))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_almost_equal(UCS_to_XYZ(UCS), XYZ, decimal=7)

    def test_domain_range_scale_UCS_to_XYZ(self):
        """
        Tests :func:`colour.models.cie_ucs.UCS_to_XYZ` definition domain and
        range scale support.
        """

        UCS = np.array([0.0469968933, 0.1008000000, 0.1637438950])
        XYZ = UCS_to_XYZ(UCS)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    UCS_to_XYZ(UCS * factor), XYZ * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_UCS_to_XYZ(self):
        """
        Tests :func:`colour.models.cie_ucs.UCS_to_XYZ` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            UCS = np.array(case)
            UCS_to_XYZ(UCS)


class TestUCS_to_uv(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_ucs.UCS_to_uv` definition unit tests
    methods.
    """

    def test_UCS_to_uv(self):
        """
        Tests :func:`colour.models.cie_ucs.UCS_to_uv` definition.
        """

        np.testing.assert_almost_equal(
            UCS_to_uv(np.array([0.13769339, 0.12197225, 0.10537310])),
            np.array([0.37720213, 0.33413508]),
            decimal=7)

        np.testing.assert_almost_equal(
            UCS_to_uv(np.array([0.09481340, 0.23042768, 0.32701033])),
            np.array([0.14536327, 0.35328046]),
            decimal=7)

        np.testing.assert_almost_equal(
            UCS_to_uv(np.array([0.05212520, 0.06157201, 0.19376075])),
            np.array([0.16953602, 0.20026156]),
            decimal=7)

    def test_n_dimensional_UCS_to_uv(self):
        """
        Tests :func:`colour.models.cie_ucs.UCS_to_uv` definition n-dimensional
        support.
        """

        UCS = np.array([0.13769339, 0.12197225, 0.10537310])
        uv = UCS_to_uv(UCS)

        UCS = np.tile(UCS, (6, 1))
        uv = np.tile(uv, (6, 1))
        np.testing.assert_almost_equal(UCS_to_uv(UCS), uv, decimal=7)

        UCS = np.reshape(UCS, (2, 3, 3))
        uv = np.reshape(uv, (2, 3, 2))
        np.testing.assert_almost_equal(UCS_to_uv(UCS), uv, decimal=7)

    def test_domain_range_scale_UCS_to_uv(self):
        """
        Tests :func:`colour.models.cie_ucs.UCS_to_uv` definition domain and
        range scale support.
        """

        UCS = np.array([0.0469968933, 0.1008000000, 0.1637438950])
        uv = UCS_to_uv(UCS)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    UCS_to_uv(UCS * factor), uv, decimal=7)

    @ignore_numpy_errors
    def test_nan_UCS_to_uv(self):
        """
        Tests :func:`colour.models.cie_ucs.UCS_to_uv` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            UCS = np.array(case)
            UCS_to_uv(UCS)


class Testuv_to_UCS(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_ucs.uv_to_UCS` definition unit tests
    methods.
    """

    def test_uv_to_UCS(self):
        """
        Tests :func:`colour.models.cie_ucs.uv_to_UCS` definition.
        """

        np.testing.assert_almost_equal(
            uv_to_UCS(np.array([0.37720213, 0.33413508])),
            np.array([1.12889114, 1.00000000, 0.86391046]),
            decimal=7)

        np.testing.assert_almost_equal(
            uv_to_UCS(np.array([0.14536327, 0.35328046])),
            np.array([0.41146705, 1.00000000, 1.41914520]),
            decimal=7)

        np.testing.assert_almost_equal(
            uv_to_UCS(np.array([0.16953602, 0.20026156])),
            np.array([0.84657295, 1.00000000, 3.14689659]),
            decimal=7)

        np.testing.assert_almost_equal(
            uv_to_UCS(np.array([0.37720213, 0.33413508]), V=0.18),
            np.array([0.20320040, 0.18000000, 0.15550388]),
            decimal=7)

    def test_n_dimensional_uv_to_UCS(self):
        """
        Tests :func:`colour.models.cie_ucs.uv_to_UCS` definition n-dimensional
        support.
        """

        uv = np.array([0.37720213, 0.33413508])
        UCS = uv_to_UCS(uv)

        uv = np.tile(uv, (6, 1))
        UCS = np.tile(UCS, (6, 1))
        np.testing.assert_almost_equal(uv_to_UCS(uv), UCS, decimal=7)

        uv = np.reshape(uv, (2, 3, 2))
        UCS = np.reshape(UCS, (2, 3, 3))
        np.testing.assert_almost_equal(uv_to_UCS(uv), UCS, decimal=7)

    def test_domain_range_scale_uv_to_UCS(self):
        """
        Tests :func:`colour.models.cie_ucs.uv_to_UCS` definition domain and
        range scale support.
        """

        uv = np.array([0.37720213, 0.33413508])
        V = 1
        UCS = uv_to_UCS(uv, V)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    uv_to_UCS(uv, V * factor), UCS * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_uv_to_UCS(self):
        """
        Tests :func:`colour.models.cie_ucs.uv_to_UCS` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=2))
        for case in cases:
            uv = np.array(case)
            uv_to_UCS(uv)


class TestUCS_uv_to_xy(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_ucs.UCS_uv_to_xy` definition unit tests
    methods.
    """

    def test_UCS_uv_to_xy(self):
        """
        Tests :func:`colour.models.cie_ucs.UCS_uv_to_xy` definition.
        """

        np.testing.assert_almost_equal(
            UCS_uv_to_xy(np.array([0.37720213, 0.33413508])),
            np.array([0.54369555, 0.32107941]),
            decimal=7)

        np.testing.assert_almost_equal(
            UCS_uv_to_xy(np.array([0.14536327, 0.35328046])),
            np.array([0.29777734, 0.48246445]),
            decimal=7)

        np.testing.assert_almost_equal(
            UCS_uv_to_xy(np.array([0.16953602, 0.20026156])),
            np.array([0.18582823, 0.14633764]),
            decimal=7)

    def test_n_dimensional_UCS_uv_to_xy(self):
        """
        Tests :func:`colour.models.cie_ucs.UCS_uv_to_xy` definition
        n-dimensional arrays support.
        """

        uv = np.array([0.37720213, 0.33413508])
        xy = UCS_uv_to_xy(uv)

        uv = np.tile(uv, (6, 1))
        xy = np.tile(xy, (6, 1))
        np.testing.assert_almost_equal(UCS_uv_to_xy(uv), xy, decimal=7)

        uv = np.reshape(uv, (2, 3, 2))
        xy = np.reshape(xy, (2, 3, 2))
        np.testing.assert_almost_equal(UCS_uv_to_xy(uv), xy, decimal=7)

    @ignore_numpy_errors
    def test_nan_UCS_uv_to_xy(self):
        """
        Tests :func:`colour.models.cie_ucs.UCS_uv_to_xy` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=2))
        for case in cases:
            uv = np.array(case)
            UCS_uv_to_xy(uv)


class TestXy_to_UCS_uv(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_ucs.xy_to_UCS_uv` definition unit tests
    methods.
    """

    def test_xy_to_UCS_uv(self):
        """
        Tests :func:`colour.models.cie_ucs.xy_to_UCS_uv` definition.
        """

        np.testing.assert_almost_equal(
            xy_to_UCS_uv(np.array([0.54369555, 0.32107941])),
            np.array([0.37720213, 0.33413508]),
            decimal=7)

        np.testing.assert_almost_equal(
            xy_to_UCS_uv(np.array([0.29777734, 0.48246445])),
            np.array([0.14536327, 0.35328046]),
            decimal=7)

        np.testing.assert_almost_equal(
            xy_to_UCS_uv(np.array([0.18582823, 0.14633764])),
            np.array([0.16953602, 0.20026156]),
            decimal=7)

    def test_n_dimensional_xy_to_UCS_uv(self):
        """
        Tests :func:`colour.models.cie_ucs.xy_to_UCS_uv` definition
        n-dimensional arrays support.
        """

        xy = np.array([0.54369555, 0.32107941])
        uv = xy_to_UCS_uv(xy)

        xy = np.tile(xy, (6, 1))
        uv = np.tile(uv, (6, 1))
        np.testing.assert_almost_equal(xy_to_UCS_uv(xy), uv, decimal=7)

        xy = np.reshape(xy, (2, 3, 2))
        uv = np.reshape(uv, (2, 3, 2))
        np.testing.assert_almost_equal(xy_to_UCS_uv(xy), uv, decimal=7)

    @ignore_numpy_errors
    def test_nan_xy_to_UCS_uv(self):
        """
        Tests :func:`colour.models.cie_ucs.xy_to_UCS_uv` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=2))
        for case in cases:
            xy = np.array(case)
            xy_to_UCS_uv(xy)


if __name__ == '__main__':
    unittest.main()
