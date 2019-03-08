# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.models.rgb.ictpt` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from itertools import permutations

from colour.models.rgb import RGB_to_ICTCP, ICTCP_to_RGB
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestRGB_to_ICTCP', 'TestICTCP_to_RGB']


class TestRGB_to_ICTCP(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.ictpt.TestRGB_to_ICTCP` definition unit
    tests methods.
    """

    def test_RGB_to_ICTCP(self):
        """
        Tests :func:`colour.models.rgb.ictpt.RGB_to_ICTCP` definition.
        """

        np.testing.assert_almost_equal(
            RGB_to_ICTCP(np.array([0.45620519, 0.03081071, 0.04091952])),
            np.array([0.07351364, 0.00475253, 0.09351596]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_ICTCP(np.array([0.45620519, 0.03081071, 0.04091952]), 4000),
            np.array([0.10516931, 0.00514031, 0.12318730]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_ICTCP(np.array([0.45620519, 0.03081071, 0.04091952]), 1000),
            np.array([0.17079612, 0.00485580, 0.17431356]),
            decimal=7)

    def test_n_dimensional_RGB_to_ICTCP(self):
        """
        Tests :func:`colour.models.rgb.ictpt.RGB_to_ICTCP` definition
        n-dimensions support.
        """

        RGB = np.array([0.45620519, 0.03081071, 0.04091952])
        ICTCP = np.array([0.07351364, 0.00475253, 0.09351596])
        np.testing.assert_almost_equal(RGB_to_ICTCP(RGB), ICTCP, decimal=7)

        RGB = np.tile(RGB, (6, 1))
        ICTCP = np.tile(ICTCP, (6, 1))
        np.testing.assert_almost_equal(RGB_to_ICTCP(RGB), ICTCP, decimal=7)

        RGB = np.reshape(RGB, (2, 3, 3))
        ICTCP = np.reshape(ICTCP, (2, 3, 3))
        np.testing.assert_almost_equal(RGB_to_ICTCP(RGB), ICTCP, decimal=7)

    def test_domain_range_scale_RGB_to_ICTCP(self):
        """
        Tests :func:`colour.models.rgb.ictpt.RGB_to_ICTCP` definition domain
        and range scale support.
        """

        RGB = np.array([0.45620519, 0.03081071, 0.04091952])
        ICTCP = RGB_to_ICTCP(RGB)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    RGB_to_ICTCP(RGB * factor), ICTCP * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_RGB_to_ICTCP(self):
        """
        Tests :func:`colour.models.rgb.ictpt.RGB_to_ICTCP` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            RGB = np.array(case)
            RGB_to_ICTCP(RGB)


class TestICTCP_to_RGB(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.ictpt.ICTCP_to_RGB` definition unit tests
    methods.
    """

    def test_ICTCP_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.ictpt.ICTCP_to_RGB` definition.
        """

        np.testing.assert_almost_equal(
            ICTCP_to_RGB(np.array([0.07351364, 0.00475253, 0.09351596])),
            np.array([0.45620519, 0.03081071, 0.04091952]),
            decimal=7)

        np.testing.assert_almost_equal(
            ICTCP_to_RGB(np.array([0.10516931, 0.00514031, 0.12318730]), 4000),
            np.array([0.45620519, 0.03081071, 0.04091952]),
            decimal=7)

        np.testing.assert_almost_equal(
            ICTCP_to_RGB(np.array([0.17079612, 0.00485580, 0.17431356]), 1000),
            np.array([0.45620519, 0.03081071, 0.04091952]),
            decimal=7)

    def test_n_dimensional_ICTCP_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.ictpt.ICTCP_to_RGB` definition
        n-dimensions support.
        """

        ICTCP = np.array([0.07351364, 0.00475253, 0.09351596])
        RGB = np.array([0.45620519, 0.03081071, 0.04091952])
        np.testing.assert_almost_equal(ICTCP_to_RGB(ICTCP), RGB, decimal=7)

        ICTCP = np.tile(ICTCP, (6, 1))
        RGB = np.tile(RGB, (6, 1))
        np.testing.assert_almost_equal(ICTCP_to_RGB(ICTCP), RGB, decimal=7)

        ICTCP = np.reshape(ICTCP, (2, 3, 3))
        RGB = np.reshape(RGB, (2, 3, 3))
        np.testing.assert_almost_equal(ICTCP_to_RGB(ICTCP), RGB, decimal=7)

    def test_domain_range_scale_ICTCP_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.ictpt.ICTCP_to_RGB` definition domain
        and range scale support.
        """

        ICTCP = np.array([0.07351364, 0.00475253, 0.09351596])
        RGB = ICTCP_to_RGB(ICTCP)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    ICTCP_to_RGB(ICTCP * factor), RGB * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_ICTCP_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.ictpt.ICTCP_to_RGB` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            ICTCP = np.array(case)
            ICTCP_to_RGB(ICTCP)


if __name__ == '__main__':
    unittest.main()
