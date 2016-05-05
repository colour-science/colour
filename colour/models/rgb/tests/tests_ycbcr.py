#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.models.rgb.ycbcr` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.models.rgb.ycbcr import (RGB_to_YCbCr,
                                     YCbCr_to_RGB,
                                     RGB_to_YcCbcCrc,
                                     YcCbcCrc_to_RGB,
                                     YCBCR_WEIGHTS)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2016 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Development'

__all__ = ['TestRGB_to_YCbCr',
           'TestYCbCr_to_RGB',
           'TestRGB_to_YcCbcCrc',
           'TestYcCbcCrc_to_RGB']


class TestRGB_to_YCbCr(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.ycbcr.RGB_to_YCbCr` definition unit tests
    methods.
    """

    def test_RGB_to_YCbCr(self):
        """
        Tests :func:`colour.models.rgb.ycbcr.RGB_to_YCbCr` definition.
        """

        np.testing.assert_almost_equal(
            RGB_to_YCbCr(np.array([0.75, 0.75, 0.0])),
            np.array([0.66035745, 0.17254902, 0.53216593]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_YCbCr(np.array([0.25, 0.5, 0.75]),
                         K=YCBCR_WEIGHTS['Rec. 601'],
                         out_int=True,
                         out_legal=True,
                         out_bits=10),
            np.array([461, 662, 382]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_YCbCr(np.array([0.0, 0.75, 0.75]),
                         K=YCBCR_WEIGHTS['Rec. 2020'],
                         out_int=False,
                         out_legal=False),
            np.array([0.552975, 0.10472255, -0.375]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_YCbCr(np.array([0.75, 0.0, 0.75]),
                         K=YCBCR_WEIGHTS['Rec. 709'],
                         out_range=(16./255, 235./255, 15.5/255, 239.5/255)),
            np.array([0.2461898 , 0.75392897, 0.79920662]),
            decimal=7)


class TestYCbCr_to_RGB(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.ycbcr.YCbCr_to_RGB` definition unit tests
    methods.
    """

    def test_YCbCr_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.ycbcr.YCbCr_to_RGB` definition.
        """

        np.testing.assert_almost_equal(
            YCbCr_to_RGB(np.array([0.66035745, 0.17254902, 0.53216593])),
            np.array([0.75, 0.75, 0.0]),
            decimal=7)

        np.testing.assert_almost_equal(
            YCbCr_to_RGB(np.array([471, 650, 390]),
                         in_bits=10,
                         in_legal=True,
                         in_int=True),
            np.array([0.25018598, 0.49950072, 0.75040741]),
            decimal=7)


class TestRGB_to_YcCbcCrc(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.ycbcr.RGB_to_YcCbcCrc` definition unit
    tests methods.
    """

    def test_RGB_to_YcCbcCrc(self):
        """
        Tests :func:`colour.models.rgb.ycbcr.RGB_to_YcCbcCrc` definition.
        """

        np.testing.assert_almost_equal(
            RGB_to_YcCbcCrc(np.array([0.123, 0.456, 0.789])),
            np.array([0.59433183, 0.65184256, 0.35373582]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_YcCbcCrc(np.array([0.18, 0.18, 0.18]),
                            out_bits=10,
                            out_legal=True,
                            out_int=True,
                            is_10_bits_system=True),
            np.array([422, 512, 512]),
            decimal=7)


class TestYcCbcCrc_to_RGB(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.ycbcr.YCbCr_to_RGB` definition unit tests
    methods.
    """

    def test_YcCbcCrc_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.ycbcr.YCbCr_to_RGB` definition.
        """

        np.testing.assert_almost_equal(
            YcCbcCrc_to_RGB(np.array([1689, 2048, 2048]),
                            in_bits=12,
                            in_legal=True,
                            in_int=True,
                            is_10_bits_system=False),
            np.array([0.18009037, 0.18009037, 0.18009037]),
            decimal=7)

        np.testing.assert_almost_equal(
            YcCbcCrc_to_RGB(np.array([0.678, 0.4, 0.6])),
            np.array([0.68390184, 0.47285022, 0.25116003]),
            decimal=7)


if __name__ == '__main__':
    unittest.main()
