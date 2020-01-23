# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.graph.conversion` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import six
import unittest

from colour import COLOURCHECKERS_SDS, ILLUMINANTS, ILLUMINANTS_SDS
from colour.graph import describe_conversion_path, convert

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['TestDescribeConversionPath', 'TestConvert']


class TestDescribeConversionPath(unittest.TestCase):
    """
    Defines :func:`colour.graph.conversion.describe_conversion_path` definition
    unit tests methods.
    """

    def test_describe_conversion_path(self):
        """
        Tests :func:`colour.graph.conversion.describe_conversion_path`
        definition.
        """

        describe_conversion_path('Spectral Distribution', 'sRGB')

        describe_conversion_path('Spectral Distribution', 'sRGB', mode='Long')

        describe_conversion_path(
            'Spectral Distribution',
            'sRGB',
            mode='Extended',
            sd_to_XYZ={
                'illuminant': ILLUMINANTS_SDS['FL2'],
                'return': np.array([0.47924575, 0.31676968, 0.17362725])
            })


class TestConvert(unittest.TestCase):
    """
    Defines :func:`colour.graph.conversion.convert` definition unit tests
    methods.
    """

    def test_convert(self):
        """
        Tests :func:`colour.graph.conversion.convert` definition.
        """

        RGB_a = convert(COLOURCHECKERS_SDS['ColorChecker N Ohta']['dark skin'],
                        'Spectral Distribution', 'sRGB')
        np.testing.assert_almost_equal(
            RGB_a, np.array([0.45675795, 0.30986982, 0.24861924]), decimal=7)

        Jpapbp = convert(RGB_a, 'Output-Referred RGB', 'CAM16UCS')
        np.testing.assert_almost_equal(
            Jpapbp, np.array([0.39994810, 0.09206557, 0.08127526]), decimal=7)

        RGB_b = convert(
            Jpapbp, 'CAM16UCS', 'sRGB', verbose={'mode': 'Extended'})
        # NOTE: The "CIE XYZ" tristimulus values to "sRGB" matrix is given
        # rounded at 4 decimals as per "IEC 61966-2-1:1999" and thus preventing
        # exact roundtrip.
        np.testing.assert_allclose(RGB_a, RGB_b, rtol=1e-5, atol=1e-5)

        np.testing.assert_almost_equal(
            convert('#808080', 'Hexadecimal', 'Scene-Referred RGB'),
            np.array([0.21586050, 0.21586050, 0.21586050]),
            decimal=7)

        self.assertAlmostEqual(
            convert('#808080', 'Hexadecimal', 'RGB Luminance'),
            0.21586050,
            places=7)

        np.testing.assert_almost_equal(
            convert(
                convert(
                    np.array([0.5, 0.5, 0.5]), 'Output-Referred RGB',
                    'Scene-Referred RGB'), 'RGB', 'YCbCr'),
            np.array([0.49215686, 0.50196078, 0.50196078]),
            decimal=7)

    def test_convert_direct_keyword_argument_passing(self):
        """
        Tests :func:`colour.graph.conversion.convert` definition behaviour when
        direct keyword arguments are passed.
        """

        a = np.array([0.20654008, 0.12197225, 0.05136952])
        illuminant = ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D50']
        np.testing.assert_almost_equal(
            convert(
                a, 'CIE XYZ', 'CIE xyY',
                XYZ_to_xyY={'illuminant': illuminant}),
            convert(a, 'CIE XYZ', 'CIE xyY', illuminant=illuminant),
            decimal=7)

        if six.PY3:  # pragma: no cover
            self.assertRaises(AttributeError, lambda: convert(
                COLOURCHECKERS_SDS['ColorChecker N Ohta']['dark skin'],
                'Spectral Distribution', 'sRGB',
                illuminant=illuminant))


if __name__ == '__main__':
    unittest.main()
