# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.recovery.smits1999` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.colorimetry import spectral_to_XYZ_integration
from colour.recovery import RGB_to_spectral_Smits1999
from colour.recovery.smits1999 import XYZ_to_RGB_Smits1999
from colour.utilities import domain_range_scale

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestRGB_to_spectral_Smits1999']


class TestRGB_to_spectral_Smits1999(unittest.TestCase):
    """
    Defines :func:`colour.recovery.smits1999.RGB_to_spectral_Smits1999`
    definition unit tests methods.
    """

    def test_RGB_to_spectral_Smits1999(self):
        """
        Tests :func:`colour.recovery.smits1999.RGB_to_spectral_Smits1999`
        definition.
        """

        np.testing.assert_almost_equal(
            RGB_to_spectral_Smits1999(
                np.array([0.45293517, 0.31732158, 0.26414773])).values,
            np.array([
                0.27787714, 0.27113183, 0.26990663, 0.29932875, 0.31711026,
                0.31726875, 0.43019862, 0.45275442, 0.45328084, 0.45410503
            ]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_spectral_Smits1999(
                np.array([0.77875824, 0.57726450, 0.50453169])).values,
            np.array([
                0.52493013, 0.51490862, 0.51239457, 0.55255311, 0.57686087,
                0.57716359, 0.74497895, 0.77874936, 0.77946941, 0.78059677
            ]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_spectral_Smits1999(
                np.array([0.35505307, 0.47995567, 0.61088035])).values,
            np.array([
                0.60725817, 0.60371094, 0.59674004, 0.52330084, 0.47975906,
                0.47997209, 0.37462711, 0.35988419, 0.36137673, 0.36154693
            ]),
            decimal=7)

    def test_domain_range_scale_RGB_to_spectral_Smits1999(self):
        """
        Tests :func:`colour.recovery.smits1999.RGB_to_spectral_Smits1999`
        definition domain and range scale support.
        """

        RGB_i = XYZ_to_RGB_Smits1999(
            np.array([0.07049534, 0.10080000, 0.09558313]))
        XYZ_o = spectral_to_XYZ_integration(RGB_to_spectral_Smits1999(RGB_i))

        d_r = (('reference', 1, 1), (1, 1, 0.01), (100, 100, 1))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    spectral_to_XYZ_integration(
                        RGB_to_spectral_Smits1999(RGB_i * factor_a)),
                    XYZ_o * factor_b,
                    decimal=7)


if __name__ == '__main__':
    unittest.main()
