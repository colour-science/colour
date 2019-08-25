# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.graph.conversion` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour import COLOURCHECKERS_SDS
from colour.graph import convert

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestConvert']


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
            Jpapbp,
            'CAM16UCS',
            'sRGB',
            verbose_parameters={'describe': 'extended'})
        # NOTE: The "CIE XYZ" tristimulus values to "sRGB" matrix is given
        # rounded at 4 decimals as per "IEC 61966-2-1:1999" and thus preventing
        # exact roundtrip.
        np.testing.assert_allclose(RGB_a, RGB_b, rtol=1e-5, atol=1e-5)

    def test_raise_exception_convert(self):
        """
        Tests :func:`colour.graph.conversion.convert` definition raised
        exception.
        """

        self.assertRaises(
            Exception,
            convert,
            np.array([0.45675795, 0.30986982, 0.24861924]),
            'Spectral Distribution',
            'sRGB',
            verbose_parameters={'describe': 'extended'})


if __name__ == '__main__':
    unittest.main()
