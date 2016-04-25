#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.models.rgb.ycbcr` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.models.rgb.ycbcr import rgb_to_YCbCr, YCbCr_to_rgb, rgb_to_YcCbcCrc,
        YcCbcCrc_to_rgb, RANGE, WEIGHT

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2016 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Development'

__all__ = ['Testrgb_to_YCbCr', 'TestYCbCr_to_rgb', 'Testrgb_to_YcCbcCrc', 'TestYcCbcCrc_to_rgb']


class Testrgb_to_YCbCr(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.ycbcr.rgb_to_YCbCr` definition unit tests
    methods.
    """

    def test_rgb_to_YCbCr(self):
        """
        Tests :func:`colour.models.rgb.ycbcr.rgb_to_YCbCr` definition.
        """

        np.testing.assert_almost_equal(
            rgb_to_YCbCr(np.array([0.75, 0.75, 0.0])),
            np.array([ 0.65842092,  0.17204301,  0.53060532]),
            decimal=7)
            
        np.testing.assert_almost_equal(
            rgb_to_YCbCr(np.array([0.25, 0.5, 0.75]), outRange=RANGE['legal_10_YC_int']),
            np.array([471, 650, 390]),
            decimal=7)


class TestYCbCr_to_rgb(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.ycbcr.YCbCr_to_rgb` definition unit tests
    methods.
    """

    def test_YCbCr_to_rgb(self):
        """
        Tests :func:`colour.models.rgb.ycbcr.YCbCr_to_rgb` definition.
        """

        np.testing.assert_almost_equal(
            YCbCr_to_rgb(np.array([ 0.65842092,  0.17204301,  0.53060532])),
            np.array([0.75, 0.75, 0.0]),
            decimal=7)
            
        np.testing.assert_almost_equal(
            YCbCr_to_rgb(np.array([ 471, 650, 390]), inRange=RANGE['legal_10_YC_int']),
            np.array([ 0.25018598,  0.49950072,  0.75040741]),
            decimal=7)
            

class Testrgb_to_YcCbcCrc(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.ycbcr.rgb_to_YcCbcCrc` definition unit tests
    methods.
    """

    def test_rgb_to_YcCbcCrc(self):
        """
        Tests :func:`colour.models.rgb.ycbcr.rgb_to_YcCbcCrc` definition.
        """

        np.testing.assert_almost_equal(
            rgb_to_YcCbcCrc(np.array([0.123, 0.456, 0.789])),
            np.array([ 0.59258892,  0.64993099,  0.35269847]),
            decimal=7)
            
        np.testing.assert_almost_equal(
            rgb_to_YcCbcCrc(np.array([0.18, 0.18, 0.18]), outRange=RANGE['legal_10_YC_int'], is_10_bits_system = True),
            np.array([422, 512, 512]),
            decimal=7)


class TestYcCbcCrc_to_rgb(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.ycbcr.YCbCr_to_rgb` definition unit tests
    methods.
    """

    def test_YcCbcCrc_to_rgb(self):
        """
        Tests :func:`colour.models.rgb.ycbcr.YCbCr_to_rgb` definition.
        """

        np.testing.assert_almost_equal(
            YcCbcCrc_to_rgb(np.array([1689, 2048, 2048]), inRange=RANGE['legal_12_YC_int'], is_10_bits_system = False),
            np.array([ 0.18009037,  0.18009037,  0.18009037]),
            decimal=7)
            
        np.testing.assert_almost_equal(
            YcCbcCrc_to_rgb(np.array([0.678, 0.4, 0.6])),
            np.array([ 0.69100667,  0.47450469,  0.25583733]),
            decimal=7)


if __name__ == '__main__':
    unittest.main()
