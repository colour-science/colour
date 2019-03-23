# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.volume.spectrum` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from itertools import permutations

from colour.volume import (generate_pulse_waves, XYZ_outer_surface,
                           is_within_visible_spectrum)
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'TestGeneratePulseWaves', 'TestXYZOuterSurface',
    'TestIsWithinVisibleSpectrum'
]


class TestGeneratePulseWaves(unittest.TestCase):
    """
    Defines :func:`colour.volume.spectrum.generate_pulse_waves`
    definition unit tests methods.
    """

    def test_generate_pulse_waves(self):
        """
        Tests :func:`colour.volume.spectrum.generate_pulse_waves`
        definition.
        """

        np.testing.assert_array_equal(
            generate_pulse_waves(5),
            np.array([
                [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000],
                [1.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000],
                [0.00000000, 1.00000000, 0.00000000, 0.00000000, 0.00000000],
                [0.00000000, 0.00000000, 1.00000000, 0.00000000, 0.00000000],
                [0.00000000, 0.00000000, 0.00000000, 1.00000000, 0.00000000],
                [0.00000000, 0.00000000, 0.00000000, 0.00000000, 1.00000000],
                [1.00000000, 1.00000000, 0.00000000, 0.00000000, 0.00000000],
                [0.00000000, 1.00000000, 1.00000000, 0.00000000, 0.00000000],
                [0.00000000, 0.00000000, 1.00000000, 1.00000000, 0.00000000],
                [0.00000000, 0.00000000, 0.00000000, 1.00000000, 1.00000000],
                [1.00000000, 0.00000000, 0.00000000, 0.00000000, 1.00000000],
                [1.00000000, 1.00000000, 1.00000000, 0.00000000, 0.00000000],
                [0.00000000, 1.00000000, 1.00000000, 1.00000000, 0.00000000],
                [0.00000000, 0.00000000, 1.00000000, 1.00000000, 1.00000000],
                [1.00000000, 0.00000000, 0.00000000, 1.00000000, 1.00000000],
                [1.00000000, 1.00000000, 0.00000000, 0.00000000, 1.00000000],
                [1.00000000, 1.00000000, 1.00000000, 1.00000000, 0.00000000],
                [0.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000],
                [1.00000000, 0.00000000, 1.00000000, 1.00000000, 1.00000000],
                [1.00000000, 1.00000000, 0.00000000, 1.00000000, 1.00000000],
                [1.00000000, 1.00000000, 1.00000000, 0.00000000, 1.00000000],
                [1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000],
            ]))


class TestXYZOuterSurface(unittest.TestCase):
    """
    Defines :func:`colour.volume.spectrum.XYZ_outer_surface`
    definition unit tests methods.
    """

    def test_XYZ_outer_surface(self):
        """
        Tests :func:`colour.volume.spectrum.XYZ_outer_surface`
        definition.
        """

        np.testing.assert_array_almost_equal(
            XYZ_outer_surface(84),
            np.array([
                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                [1.47669249e-03, 4.15303476e-05, 6.98843624e-03],
                [1.62812757e-01, 3.71143871e-02, 9.01514713e-01],
                [1.86508941e-01, 5.66174645e-01, 9.13551791e-02],
                [6.15553478e-01, 3.84277759e-01, 4.74220708e-04],
                [3.36220454e-02, 1.23545569e-02, 0.00000000e+00],
                [1.02795008e-04, 3.71211582e-05, 0.00000000e+00],
                [1.64289450e-01, 3.71559174e-02, 9.08503149e-01],
                [3.49321699e-01, 6.03289032e-01, 9.92869892e-01],
                [8.02062419e-01, 9.50452405e-01, 9.18293998e-02],
                [6.49175523e-01, 3.96632316e-01, 4.74220708e-04],
                [3.37248404e-02, 1.23916780e-02, 0.00000000e+00],
                [1.57948749e-03, 7.86515058e-05, 6.98843624e-03],
                [3.50798391e-01, 6.03330563e-01, 9.99858328e-01],
                [9.64875177e-01, 9.87566792e-01, 9.93344113e-01],
                [8.35684465e-01, 9.62806961e-01, 9.18293998e-02],
                [6.49278318e-01, 3.96669437e-01, 4.74220708e-04],
                [3.52015329e-02, 1.24332084e-02, 6.98843624e-03],
                [1.64392245e-01, 3.71930386e-02, 9.08503149e-01],
                [9.66351869e-01, 9.87608322e-01, 1.00033255e+00],
                [9.98497222e-01, 9.99921348e-01, 9.93344113e-01],
                [8.35787260e-01, 9.62844083e-01, 9.18293998e-02],
                [6.50755011e-01, 3.96710968e-01, 7.46265695e-03],
                [1.98014290e-01, 4.95475954e-02, 9.08503149e-01],
                [3.50901186e-01, 6.03367684e-01, 9.99858328e-01],
                [9.99973915e-01, 9.99962879e-01, 1.00033255e+00],
                [9.98600017e-01, 9.99958470e-01, 9.93344113e-01],
                [8.37263952e-01, 9.62885613e-01, 9.88178360e-02],
                [8.13567768e-01, 4.33825355e-01, 9.08977370e-01],
                [3.84523232e-01, 6.15722241e-01, 9.99858328e-01],
                [9.66454664e-01, 9.87645443e-01, 1.00033255e+00],
                [1.00007671e+00, 1.00000000e+00, 1.00033255e+00],
            ]),
            decimal=7)


class TestIsWithinVisibleSpectrum(unittest.TestCase):
    """
    Defines :func:`colour.volume.spectrum.is_within_visible_spectrum`
    definition unit tests methods.
    """

    def test_is_within_visible_spectrum(self):
        """
        Tests :func:`colour.volume.spectrum.is_within_visible_spectrum`
        definition.
        """

        self.assertTrue(
            is_within_visible_spectrum(np.array([0.3205, 0.4131, 0.5100])))

        self.assertFalse(
            is_within_visible_spectrum(np.array([-0.0005, 0.0031, 0.0010])))

        self.assertTrue(
            is_within_visible_spectrum(np.array([0.4325, 0.3788, 0.1034])))

        self.assertFalse(
            is_within_visible_spectrum(np.array([0.0025, 0.0088, 0.0340])))

    def test_n_dimensional_is_within_visible_spectrum(self):
        """
        Tests :func:`colour.volume.spectrum.is_within_visible_spectrum`
        definition n-dimensional arrays support.
        """

        a = np.array([0.3205, 0.4131, 0.5100])
        b = np.array([True])
        np.testing.assert_almost_equal(is_within_visible_spectrum(a), b)

        a = np.tile(a, (6, 1))
        b = np.tile(b, 6)
        np.testing.assert_almost_equal(is_within_visible_spectrum(a), b)

        a = np.reshape(a, (2, 3, 3))
        b = np.reshape(b, (2, 3))
        np.testing.assert_almost_equal(is_within_visible_spectrum(a), b)

    @ignore_numpy_errors
    def test_nan_is_within_visible_spectrum(self):
        """
        Tests :func:`colour.volume.spectrum.is_within_visible_spectrum`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            is_within_visible_spectrum(case)


if __name__ == '__main__':
    unittest.main()
