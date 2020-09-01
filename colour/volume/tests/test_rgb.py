# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.volume.rgb` module.

Notes
-----
The MonteCarlo sampling based unit tests are assuming that
:func:`np.random.RandomState` definition will return the same sequence no
matter which *OS* or *Python* version is used. There is however no formal
promise about the *prng* sequence reproducibility of either *Python* or *Numpy*
implementations:

References
----------
-   :cite:`Laurent2012a` : Laurent. (2012). Reproducibility of python
    pseudo-random numbers across systems and versions? Retrieved January 20,
    2015, from
    http://stackoverflow.com/questions/8786084/\
reproducibility-of-python-pseudo-random-numbers-across-systems-and-versions
"""

from __future__ import division, unicode_literals

import colour.ndarray as np
import unittest

from colour.models import (RGB_COLOURSPACE_ACES2065_1, RGB_COLOURSPACE_BT2020,
                           RGB_COLOURSPACE_BT709)
from colour.volume import (
    RGB_colourspace_limits, RGB_colourspace_volume_MonteCarlo,
    RGB_colourspace_volume_coverage_MonteCarlo,
    RGB_colourspace_pointer_gamut_coverage_MonteCarlo,
    RGB_colourspace_visible_spectrum_coverage_MonteCarlo,
    is_within_pointer_gamut)
from colour.utilities import disable_multiprocessing

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestRGB_colourspaceLimits', 'TestRGB_colourspaceVolumeMonteCarlo',
    'TestRGB_colourspace_volume_coverage_MonteCarlo',
    'TestRGB_colourspacePointerGamutCoverageMonteCarlo',
    'TestRGB_colourspaceVisibleSpectrumCoverageMonteCarlo'
]


class TestRGB_colourspaceLimits(unittest.TestCase):
    """
    Defines :func:`colour.volume.rgb.RGB_colourspace_limits` definition unit
    tests methods.
    """

    def test_RGB_colourspace_limits(self):
        """
        Tests :func:`colour.volume.rgb.RGB_colourspace_limits` definition.
        """

        np.testing.assert_array_almost_equal(
            RGB_colourspace_limits(RGB_COLOURSPACE_BT709),
            np.array([
                [0.00000000, 100.00000000],
                [-86.18159689, 98.23744381],
                [-107.85546554, 94.48384002],
            ]),
            decimal=7)

        np.testing.assert_array_almost_equal(
            RGB_colourspace_limits(RGB_COLOURSPACE_BT2020),
            np.array([
                [0.00000000, 100.00000000],
                [-172.32005590, 130.52657313],
                [-120.27412558, 136.88564561],
            ]),
            decimal=7)

        np.testing.assert_array_almost_equal(
            RGB_colourspace_limits(RGB_COLOURSPACE_ACES2065_1),
            np.array([
                [-58.9920208, 102.4721629],
                [-404.1883039, 317.5082799],
                [-274.0297625, 174.4716296],
            ]),
            decimal=7)


class TestRGB_colourspaceVolumeMonteCarlo(unittest.TestCase):
    """
    Defines :func:`colour.volume.rgb.RGB_colourspace_volume_MonteCarlo`
    definition unit tests methods.

    References
    ----------
    :cite:`Laurent2012a`
    """

    @disable_multiprocessing()
    def test_RGB_colourspace_volume_MonteCarlo(self):
        """
        Tests :func:`colour.volume.rgb.RGB_colourspace_volume_MonteCarlo`
        definition.
        """

        self.assertAlmostEqual(
            RGB_colourspace_volume_MonteCarlo(
                RGB_COLOURSPACE_BT709,
                10e3,
                random_state=np.random.RandomState(2)) * 1e-6,
            821700.0 * 1e-6,
            places=1)


class TestRGB_colourspace_volume_coverage_MonteCarlo(unittest.TestCase):
    """
    Defines :func:`colour.volume.rgb.\
RGB_colourspace_volume_coverage_MonteCarlo` definition unit tests methods.

    References
    ----------
    :cite:`Laurent2012a`
    """

    def test_RGB_colourspace_volume_coverage_MonteCarlo(self):
        """
        Tests :func:`colour.volume.rgb.\
RGB_colourspace_volume_coverage_MonteCarlo` definition.
        """

        np.testing.assert_array_almost_equal(
            RGB_colourspace_volume_coverage_MonteCarlo(
                RGB_COLOURSPACE_BT709,
                is_within_pointer_gamut,
                10e3,
                random_state=np.random.RandomState(2)),
            81.044349070100140,
            decimal=7)


class TestRGB_colourspacePointerGamutCoverageMonteCarlo(unittest.TestCase):
    """
    Defines :func:`colour.volume.rgb.\
RGB_colourspace_pointer_gamut_coverage_MonteCarlo` definition unit tests
    methods.

    References
    ----------
    :cite:`Laurent2012a`
    """

    def test_RGB_colourspace_pointer_gamut_coverage_MonteCarlo(self):
        """
        Tests :func:`colour.volume.rgb.\
RGB_colourspace_pointer_gamut_coverage_MonteCarlo` definition.
        """

        np.testing.assert_array_almost_equal(
            RGB_colourspace_pointer_gamut_coverage_MonteCarlo(
                RGB_COLOURSPACE_BT709,
                10e3,
                random_state=np.random.RandomState(2)),
            81.044349070100140,
            decimal=7)


class TestRGB_colourspaceVisibleSpectrumCoverageMonteCarlo(unittest.TestCase):
    """
    Defines :func:`colour.volume.rgb.\
RGB_colourspace_visible_spectrum_coverage_MonteCarlo` definition unit tests
    methods.

    References
    ----------
    :cite:`Laurent2012a`
    """

    def test_RGB_colourspace_visible_spectrum_coverage_MonteCarlo(self):
        """
        Tests :func:`colour.volume.rgb.\
RGB_colourspace_visible_spectrum_coverage_MonteCarlo` definition.
        """

        np.testing.assert_array_almost_equal(
            RGB_colourspace_visible_spectrum_coverage_MonteCarlo(
                RGB_COLOURSPACE_BT709,
                10e3,
                random_state=np.random.RandomState(2)),
            46.931407942238266,
            decimal=7)


if __name__ == '__main__':
    unittest.main()
