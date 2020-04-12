# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.characterisation.aces_it` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.characterisation import (ACES_RICD, COLOURCHECKER_SDS,
                                     sd_to_aces_relative_exposure_values)
from colour.colorimetry import ILLUMINANT_SDS, sd_constant, sd_ones
from colour.utilities import domain_range_scale

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['TestSpectralToAcesRelativeExposureValues']


class TestSpectralToAcesRelativeExposureValues(unittest.TestCase):
    """
    Defines :func:`colour.characterisation.aces_it.\
sd_to_aces_relative_exposure_values` definition unit tests methods.
    """

    def test_spectral_to_aces_relative_exposure_values(self):
        """
        Tests :func:`colour.characterisation.aces_it.
sd_to_aces_relative_exposure_values` definition.
        """

        shape = ACES_RICD.shape
        grey_reflector = sd_constant(0.18, shape)
        np.testing.assert_almost_equal(
            sd_to_aces_relative_exposure_values(grey_reflector),
            np.array([0.18, 0.18, 0.18]),
            decimal=7)

        perfect_reflector = sd_ones(shape)
        np.testing.assert_almost_equal(
            sd_to_aces_relative_exposure_values(perfect_reflector),
            np.array([0.97783784, 0.97783784, 0.97783784]),
            decimal=7)

        dark_skin = COLOURCHECKER_SDS['ColorChecker N Ohta']['dark skin']
        np.testing.assert_almost_equal(
            sd_to_aces_relative_exposure_values(dark_skin),
            np.array([0.11718149, 0.08663609, 0.05897268]),
            decimal=7)

        dark_skin = COLOURCHECKER_SDS['ColorChecker N Ohta']['dark skin']
        np.testing.assert_almost_equal(
            sd_to_aces_relative_exposure_values(dark_skin,
                                                ILLUMINANT_SDS['A']),
            np.array([0.13583991, 0.09431845, 0.05928214]),
            decimal=7)

        dark_skin = COLOURCHECKER_SDS['ColorChecker N Ohta']['dark skin']
        np.testing.assert_almost_equal(
            sd_to_aces_relative_exposure_values(
                dark_skin, apply_chromatic_adaptation=True),
            np.array([0.11807796, 0.08690312, 0.05891252]),
            decimal=7)

        dark_skin = COLOURCHECKER_SDS['ColorChecker N Ohta']['dark skin']
        np.testing.assert_almost_equal(
            sd_to_aces_relative_exposure_values(
                dark_skin,
                apply_chromatic_adaptation=True,
                chromatic_adaptation_transform='Bradford'),
            np.array([0.11805993, 0.08689013, 0.05900396]),
            decimal=7)

    def test_domain_range_scale_spectral_to_aces_relative_exposure_values(
            self):
        """
        Tests :func:`colour.characterisation.aces_it.
sd_to_aces_relative_exposure_values`  definition domain and range scale
        support.
        """

        shape = ACES_RICD.shape
        grey_reflector = sd_constant(0.18, shape)
        RGB = sd_to_aces_relative_exposure_values(grey_reflector)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    sd_to_aces_relative_exposure_values(grey_reflector),
                    RGB * factor,
                    decimal=7)


if __name__ == '__main__':
    unittest.main()
