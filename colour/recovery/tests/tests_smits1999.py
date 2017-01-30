#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.recovery.smits1999` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.recovery import RGB_to_spectral_Smits1999

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
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
            np.array([0.27787714,
                      0.27113183,
                      0.26990663,
                      0.29932875,
                      0.31711026,
                      0.31726875,
                      0.43019862,
                      0.45275442,
                      0.45328084,
                      0.45410503]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_spectral_Smits1999(
                np.array([0.77875824, 0.57726450, 0.50453169])).values,
            np.array([0.52493013,
                      0.51490862,
                      0.51239457,
                      0.55255311,
                      0.57686087,
                      0.57716359,
                      0.74497895,
                      0.77874936,
                      0.77946941,
                      0.78059677]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_spectral_Smits1999(
                np.array([0.35505307, 0.47995567, 0.61088035])).values,
            np.array([0.60725817,
                      0.60371094,
                      0.59674004,
                      0.52330084,
                      0.47975906,
                      0.47997209,
                      0.37462711,
                      0.35988419,
                      0.36137673,
                      0.36154693]),
            decimal=7)


if __name__ == '__main__':
    unittest.main()
