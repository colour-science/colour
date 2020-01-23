# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.models.rgb.transfer_functions.common`
module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.models.rgb.transfer_functions import (CV_range, legal_to_full,
                                                  full_to_legal)
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Development'

__all__ = [
    'TestCV_range',
    'TestLegalToFull',
    'TestFullToLegal',
]


class TestCV_range(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.common.CV_range`
    definition unit tests methods.
    """

    def test_CV_range(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.common.CV_range`
        definition.
        """

        np.testing.assert_array_equal(
            CV_range(8, True, True), np.array([16, 235]))

        np.testing.assert_array_equal(
            CV_range(8, False, True), np.array([0, 255]))

        np.testing.assert_almost_equal(
            CV_range(8, True, False),
            np.array([0.06274510, 0.92156863]),
            decimal=7)

        np.testing.assert_array_equal(
            CV_range(8, False, False), np.array([0, 1]))

        np.testing.assert_array_equal(
            CV_range(10, True, True), np.array([64, 940]))

        np.testing.assert_array_equal(
            CV_range(10, False, True), np.array([0, 1023]))

        np.testing.assert_almost_equal(
            CV_range(10, True, False),
            np.array([0.06256109, 0.91886608]),
            decimal=7)

        np.testing.assert_array_equal(
            CV_range(10, False, False), np.array([0, 1]))


class TestLegalToFull(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.common.legal_to_full`
    definition unit tests methods.
    """

    def test_legal_to_full(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.common.legal_to_full`
        definition.
        """

        self.assertAlmostEqual(legal_to_full(64 / 1023), 0.0)

        self.assertAlmostEqual(legal_to_full(940 / 1023), 1.0)

        self.assertAlmostEqual(legal_to_full(64 / 1023, out_int=True), 0)

        self.assertAlmostEqual(legal_to_full(940 / 1023, out_int=True), 1023)

        self.assertAlmostEqual(legal_to_full(64, in_int=True), 0.0)

        self.assertAlmostEqual(legal_to_full(940, in_int=True), 1.0)

        self.assertAlmostEqual(legal_to_full(64, in_int=True, out_int=True), 0)

        self.assertAlmostEqual(
            legal_to_full(940, in_int=True, out_int=True), 1023)

    def test_n_dimensional_legal_to_full(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.common.legal_to_full`
        definition n-dimensional arrays support.
        """

        CV_l = 0.918866080156403
        CV_f = legal_to_full(CV_l, 10)

        CV_l = np.tile(CV_l, 6)
        CV_f = np.tile(CV_f, 6)
        np.testing.assert_almost_equal(
            legal_to_full(CV_l, 10), CV_f, decimal=7)

        CV_l = np.reshape(CV_l, (2, 3))
        CV_f = np.reshape(CV_f, (2, 3))
        np.testing.assert_almost_equal(
            legal_to_full(CV_l, 10), CV_f, decimal=7)

        CV_l = np.reshape(CV_l, (2, 3, 1))
        CV_f = np.reshape(CV_f, (2, 3, 1))
        np.testing.assert_almost_equal(
            legal_to_full(CV_l, 10), CV_f, decimal=7)

    @ignore_numpy_errors
    def test_nan_legal_to_full(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.common.legal_to_full`
        definition nan support.
        """

        legal_to_full(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]), 10)


class TestFullToLegal(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.common.full_to_legal`
    definition unit tests methods.
    """

    def test_full_to_legal(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.common.full_to_legal`
        definition.
        """

        self.assertAlmostEqual(full_to_legal(0.0), 0.062561094819159)

        self.assertAlmostEqual(full_to_legal(1.0), 0.918866080156403)

        self.assertAlmostEqual(full_to_legal(0.0, out_int=True), 64)

        self.assertAlmostEqual(full_to_legal(1.0, out_int=True), 940)

        self.assertAlmostEqual(
            full_to_legal(0, in_int=True), 0.062561094819159)

        self.assertAlmostEqual(
            full_to_legal(1023, in_int=True), 0.918866080156403)

        self.assertAlmostEqual(full_to_legal(0, in_int=True, out_int=True), 64)

        self.assertAlmostEqual(
            full_to_legal(1023, in_int=True, out_int=True), 940)

    def test_n_dimensional_full_to_legal(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.common.full_to_legal`
        definition n-dimensional arrays support.
        """

        CF_f = 1.0
        CV_l = full_to_legal(CF_f, 10)

        CF_f = np.tile(CF_f, 6)
        CV_l = np.tile(CV_l, 6)
        np.testing.assert_almost_equal(
            full_to_legal(CF_f, 10), CV_l, decimal=7)

        CF_f = np.reshape(CF_f, (2, 3))
        CV_l = np.reshape(CV_l, (2, 3))
        np.testing.assert_almost_equal(
            full_to_legal(CF_f, 10), CV_l, decimal=7)

        CF_f = np.reshape(CF_f, (2, 3, 1))
        CV_l = np.reshape(CV_l, (2, 3, 1))
        np.testing.assert_almost_equal(
            full_to_legal(CF_f, 10), CV_l, decimal=7)

    @ignore_numpy_errors
    def test_nan_full_to_legal(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.common.full_to_legal`
        definition nan support.
        """

        full_to_legal(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]), 10)


if __name__ == '__main__':
    unittest.main()
