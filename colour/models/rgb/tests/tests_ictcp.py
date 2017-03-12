#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.models.rgb.ictpt` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from itertools import permutations

from colour.models.rgb import RGB_to_ICTCP, ICTCP_to_RGB
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestRGB_to_ICTCP',
           'TestICTCP_to_RGB']


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
            RGB_to_ICTCP(np.array([0.35181454, 0.26934757, 0.21288023])),
            np.array([0.09554079, -0.00890639, 0.01389286]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_ICTCP(
                np.array([0.35181454, 0.26934757, 0.21288023]),
                4000),
            np.array([0.13385341, -0.01151831, 0.01780776]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_ICTCP(
                np.array([0.35181454, 0.26934757, 0.21288023]),
                1000),
            np.array([0.21071460, -0.01586417, 0.02421400]),
            decimal=7)

    def test_n_dimensional_RGB_to_ICTCP(self):
        """
        Tests :func:`colour.models.rgb.ictpt.RGB_to_ICTCP` definition
        n-dimensions support.
        """

        RGB = np.array([0.35181454, 0.26934757, 0.21288023])
        ICTCP = np.array([0.09554079, -0.00890639, 0.01389286])
        np.testing.assert_almost_equal(
            RGB_to_ICTCP(RGB),
            ICTCP,
            decimal=7)

        RGB = np.tile(RGB, (6, 1))
        ICTCP = np.tile(ICTCP, (6, 1))
        np.testing.assert_almost_equal(
            RGB_to_ICTCP(RGB),
            ICTCP,
            decimal=7)

        RGB = np.reshape(RGB, (2, 3, 3))
        ICTCP = np.reshape(ICTCP, (2, 3, 3))
        np.testing.assert_almost_equal(
            RGB_to_ICTCP(RGB),
            ICTCP,
            decimal=7)

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
            ICTCP_to_RGB(np.array([0.09554079, -0.00890639, 0.01389286])),
            np.array([0.35181454, 0.26934757, 0.21288023]),
            decimal=7)

        np.testing.assert_almost_equal(
            ICTCP_to_RGB(
                np.array([0.13385341, -0.01151831, 0.01780776]),
                4000),
            np.array([0.35181454, 0.26934757, 0.21288023]),
            decimal=7)

        np.testing.assert_almost_equal(
            ICTCP_to_RGB(
                np.array([0.21071460, -0.01586417, 0.02421400]),
                1000),
            np.array([0.35181454, 0.26934757, 0.21288023]),
            decimal=7)

    def test_n_dimensional_ICTCP_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.ictpt.ICTCP_to_RGB` definition
        n-dimensions support.
        """

        ICTCP = np.array([0.09554079, -0.00890639, 0.01389286])
        RGB = np.array([0.35181454, 0.26934757, 0.21288023])
        np.testing.assert_almost_equal(
            ICTCP_to_RGB(ICTCP),
            RGB,
            decimal=7)

        ICTCP = np.tile(ICTCP, (6, 1))
        RGB = np.tile(RGB, (6, 1))
        np.testing.assert_almost_equal(
            ICTCP_to_RGB(ICTCP),
            RGB,
            decimal=7)

        ICTCP = np.reshape(ICTCP, (2, 3, 3))
        RGB = np.reshape(RGB, (2, 3, 3))
        np.testing.assert_almost_equal(
            ICTCP_to_RGB(ICTCP),
            RGB,
            decimal=7)

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
