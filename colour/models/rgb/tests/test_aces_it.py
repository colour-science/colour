# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.models.rgb.aces_it` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.characterisation import COLOURCHECKERS_SDS
from colour.colorimetry import (ILLUMINANTS_SDS, sd_constant, sd_ones)
from colour.models import ACES_RICD, sd_to_aces_relative_exposure_values
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
    Defines :func:`colour.models.rgb.aces_it.\
sd_to_aces_relative_exposure_values` definition unit tests methods.
    """

    def test_spectral_to_aces_relative_exposure_values(self):
        """
        Tests :func:`colour.models.rgb.aces_it.
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

        dark_skin = COLOURCHECKERS_SDS['ColorChecker N Ohta']['dark skin']
        np.testing.assert_almost_equal(
            sd_to_aces_relative_exposure_values(dark_skin),
            np.array([0.11717855, 0.08663479, 0.05897071]),
            decimal=7)

        dark_skin = COLOURCHECKERS_SDS['ColorChecker N Ohta']['dark skin']
        np.testing.assert_almost_equal(
            sd_to_aces_relative_exposure_values(dark_skin,
                                                ILLUMINANTS_SDS['A']),
            np.array([0.13584109, 0.09431910, 0.05928216]),
            decimal=7)

        dark_skin = COLOURCHECKERS_SDS['ColorChecker N Ohta']['dark skin']
        np.testing.assert_almost_equal(
            sd_to_aces_relative_exposure_values(
                dark_skin, apply_chromatic_adaptation=True),
            np.array([0.11807662, 0.0869023, 0.05891045]),
            decimal=7)

        dark_skin = COLOURCHECKERS_SDS['ColorChecker N Ohta']['dark skin']
        np.testing.assert_almost_equal(
            sd_to_aces_relative_exposure_values(
                dark_skin,
                apply_chromatic_adaptation=True,
                chromatic_adaptation_transform='Bradford'),
            np.array([0.11805856, 0.08688928, 0.05900204]),
            decimal=7)

    def test_domain_range_scale_spectral_to_aces_relative_exposure_values(
            self):
        """
        Tests :func:`colour.models.rgb.aces_it.
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
