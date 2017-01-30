#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.models.rgb.aces_it` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.characterisation import COLOURCHECKERS_SPDS
from colour.colorimetry import constant_spd, ones_spd
from colour.models import ACES_RICD, spectral_to_aces_relative_exposure_values

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestSpectralToAcesRelativeExposureValues']


class TestSpectralToAcesRelativeExposureValues(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.aces_it.\
spectral_to_aces_relative_exposure_values` definition unit tests methods.
    """

    def test_spectral_to_aces_relative_exposure_values(self):
        """
        Tests :func:`colour.models.rgb.aces_it.
spectral_to_aces_relative_exposure_values` definition.
        """

        shape = ACES_RICD.shape
        grey_reflector = constant_spd(0.18, shape)
        np.testing.assert_almost_equal(
            spectral_to_aces_relative_exposure_values(grey_reflector),
            np.array([0.18, 0.18, 0.18]))

        perfect_reflector = ones_spd(shape)
        np.testing.assert_almost_equal(
            spectral_to_aces_relative_exposure_values(perfect_reflector),
            np.array([0.97783784, 0.97783784, 0.97783784]))

        dark_skin = (
            COLOURCHECKERS_SPDS['ColorChecker N Ohta']['dark skin'])
        np.testing.assert_almost_equal(
            spectral_to_aces_relative_exposure_values(dark_skin),
            np.array([0.11876978, 0.08708666, 0.0589442]))


if __name__ == '__main__':
    unittest.main()
