#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for
:mod:`colour.models.rgb.transfer_functions.prophoto_rgb` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.models.rgb.transfer_functions import (
    oetf_ProPhotoRGB,
    eotf_ProPhotoRGB)
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestOetf_ProPhotoRGB',
           'TestEotf_ProPhotoRGB']


class TestOetf_ProPhotoRGB(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.prophoto_rgb.\
oetf_ProPhotoRGB` definition unit tests methods.
    """

    def test_oetf_ProPhotoRGB(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.prophoto_rgb.\
oetf_ProPhotoRGB` definition.
        """

        self.assertAlmostEqual(
            oetf_ProPhotoRGB(0.0),
            0.0,
            places=7)

        self.assertAlmostEqual(
            oetf_ProPhotoRGB(0.18),
            0.3857114247511376,
            places=7)

        self.assertAlmostEqual(
            oetf_ProPhotoRGB(1.0),
            1.0,
            places=7)

    def test_n_dimensional_oetf_ProPhotoRGB(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.prophoto_rgb.\
oetf_ProPhotoRGB` definition n-dimensional arrays support.
        """

        L = 0.18
        V = 0.3857114247511376
        np.testing.assert_almost_equal(
            oetf_ProPhotoRGB(L),
            V,
            decimal=7)

        L = np.tile(L, 6)
        V = np.tile(V, 6)
        np.testing.assert_almost_equal(
            oetf_ProPhotoRGB(L),
            V,
            decimal=7)

        L = np.reshape(L, (2, 3))
        V = np.reshape(V, (2, 3))
        np.testing.assert_almost_equal(
            oetf_ProPhotoRGB(L),
            V,
            decimal=7)

        L = np.reshape(L, (2, 3, 1))
        V = np.reshape(V, (2, 3, 1))
        np.testing.assert_almost_equal(
            oetf_ProPhotoRGB(L),
            V,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_oetf_ProPhotoRGB(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.prophoto_rgb.\
oetf_ProPhotoRGB` definition nan support.
        """

        oetf_ProPhotoRGB(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestEotf_ProPhotoRGB(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.prophoto_rgb.
eotf_ProPhotoRGB` definition unit tests methods.
    """

    def test_eotf_ProPhotoRGB(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.prophoto_rgb.\
eotf_ProPhotoRGB` definition.
        """

        self.assertAlmostEqual(
            eotf_ProPhotoRGB(0.0),
            0.0,
            places=7)

        self.assertAlmostEqual(
            eotf_ProPhotoRGB(0.3857114247511376),
            0.18,
            places=7)

        self.assertAlmostEqual(
            eotf_ProPhotoRGB(1.0),
            1.0,
            places=7)

    def test_n_dimensional_eotf_ProPhotoRGB(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.prophoto_rgb.\
eotf_ProPhotoRGB` definition n-dimensional arrays support.
        """

        L = 0.3857114247511376
        V = 0.18
        np.testing.assert_almost_equal(
            eotf_ProPhotoRGB(L),
            V,
            decimal=7)

        L = np.tile(L, 6)
        V = np.tile(V, 6)
        np.testing.assert_almost_equal(
            eotf_ProPhotoRGB(L),
            V,
            decimal=7)

        L = np.reshape(L, (2, 3))
        V = np.reshape(V, (2, 3))
        np.testing.assert_almost_equal(
            eotf_ProPhotoRGB(L),
            V,
            decimal=7)

        L = np.reshape(L, (2, 3, 1))
        V = np.reshape(V, (2, 3, 1))
        np.testing.assert_almost_equal(
            eotf_ProPhotoRGB(L),
            V,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_eotf_ProPhotoRGB(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.prophoto_rgb.\
eotf_ProPhotoRGB` definition nan support.
        """

        eotf_ProPhotoRGB(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


if __name__ == '__main__':
    unittest.main()
