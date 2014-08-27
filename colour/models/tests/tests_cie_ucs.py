#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.models.cie_ucs` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

from colour.models import XYZ_to_UCS, UCS_to_XYZ, UCS_to_uv, UCS_uv_to_xy

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestXYZ_to_UCS',
           'TestUCS_to_XYZ',
           'TestUCS_to_uv',
           'TestUCS_uv_to_xy']


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
            XYZ_to_UCS(np.array([11.80583421, 10.34, 5.15089229])),
            np.array([7.87055614, 10.34, 12.18252904]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_UCS(np.array([3.08690042, 3.2, 2.68925666])),
            np.array([2.05793361, 3.2, 4.60117812]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_UCS(np.array([0.96907232, 1, 1.12179215])),
            np.array([0.64604821, 1., 1.57635992]),
            decimal=7)


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
            UCS_to_XYZ(np.array([7.87055614, 10.34, 12.18252904])),
            np.array([11.80583421, 10.34, 5.15089229]),
            decimal=7)

        np.testing.assert_almost_equal(
            UCS_to_XYZ(np.array([2.05793361, 3.2, 4.60117812])),
            np.array([3.08690042, 3.2, 2.68925666]),
            decimal=7)

        np.testing.assert_almost_equal(
            UCS_to_XYZ(np.array([0.64604821, 1, 1.57635992])),
            np.array([0.96907232, 1., 1.12179215]),
            decimal=7)


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
            UCS_to_uv(np.array([7.87055614, 10.34, 12.18252904])),
            (0.25895877609618834, 0.34020896328103534),
            decimal=7)

        np.testing.assert_almost_equal(
            UCS_to_uv(np.array([2.05793361, 3.2, 4.60117812])),
            (0.20873418076173886, 0.32457285074301517),
            decimal=7)

        np.testing.assert_almost_equal(
            UCS_to_uv(np.array([0.64604821, 1, 1.57635992])),
            (0.20048615319251942, 0.31032692311386395),
            decimal=7)


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
            UCS_uv_to_xy((0.2033733344733139, 0.3140500001549052)),
            (0.32207410281368043, 0.33156550013623537),
            decimal=7)

        np.testing.assert_almost_equal(
            UCS_uv_to_xy((0.20873418102926322, 0.32457285063327812)),
            (0.3439000000209443, 0.35650000010917804),
            decimal=7)

        np.testing.assert_almost_equal(
            UCS_uv_to_xy((0.25585459629500179, 0.34952813701502972)),
            (0.4474327628361858, 0.40749796251018744),
            decimal=7)


if __name__ == '__main__':
    unittest.main()
