#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.models.rgb.transfer_functions.common`
module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.models.rgb.transfer_functions import CV_range, CV_to_IRE, IRE_to_CV
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Development'

__all__ = ['TestCV_range', 'TestCV_to_IRE', 'TestIRE_to_CV']


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

        np.testing.assert_array_almost_equal(
            CV_range(8, True, False),
            np.array([0.06274510, 0.92156863]),
            decimal=7)

        np.testing.assert_array_equal(
            CV_range(8, False, False), np.array([0, 1]))

        np.testing.assert_array_equal(
            CV_range(10, True, True), np.array([64, 940]))

        np.testing.assert_array_equal(
            CV_range(10, False, True), np.array([0, 1023]))

        np.testing.assert_array_almost_equal(
            CV_range(10, True, False),
            np.array([0.06256109, 0.91886608]),
            decimal=7)

        np.testing.assert_array_equal(
            CV_range(10, False, False), np.array([0, 1]))


class TestCV_to_IRE(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.common.CV_to_IRE`
    definition unit tests methods.
    """

    def test_CV_to_IRE(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.common.CV_to_IRE`
        definition.
        """

        self.assertAlmostEqual(CV_to_IRE(64, 10, True), 0.0, places=7)

        self.assertAlmostEqual(
            CV_to_IRE(394, 10, True), 37.671232876712331, places=7)

        self.assertAlmostEqual(
            CV_to_IRE(394, 12, False), 9.6214896214896228, places=7)

        self.assertAlmostEqual(CV_to_IRE(940, 10, True), 100.0, places=7)

    def test_n_dimensional_CV_to_IRE(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.common.CV_to_IRE`
        definition n-dimensional arrays support.
        """

        CV = 394
        IRE = 37.671232876712331
        np.testing.assert_almost_equal(CV_to_IRE(CV, 10, True), IRE, decimal=7)

        CV = np.tile(CV, 6)
        IRE = np.tile(IRE, 6)
        np.testing.assert_almost_equal(CV_to_IRE(CV, 10, True), IRE, decimal=7)

        CV = np.reshape(CV, (2, 3))
        IRE = np.reshape(IRE, (2, 3))
        np.testing.assert_almost_equal(CV_to_IRE(CV, 10, True), IRE, decimal=7)

        CV = np.reshape(CV, (2, 3, 1))
        IRE = np.reshape(IRE, (2, 3, 1))
        np.testing.assert_almost_equal(CV_to_IRE(CV, 10, True), IRE, decimal=7)

    @ignore_numpy_errors
    def test_nan_CV_to_IRE(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.common.CV_to_IRE`
        definition nan support.
        """

        CV_to_IRE(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]), 10, True)


class TestIRE_to_CV(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.common.IRE_to_CV`
    definition unit tests methods.
    """

    def test_IRE_to_CV(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.common.IRE_to_CV`
        definition.
        """

        self.assertAlmostEqual(IRE_to_CV(0.0, 10, True), 64, places=7)

        self.assertAlmostEqual(
            IRE_to_CV(37.671232876712331, 10, True), 394, places=7)

        self.assertAlmostEqual(
            IRE_to_CV(9.6214896214896228, 12, False), 394, places=7)

        self.assertAlmostEqual(IRE_to_CV(100.0, 10, True), 940, places=7)

    def test_n_dimensional_IRE_to_CV(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.common.IRE_to_CV`
        definition n-dimensional arrays support.
        """

        IRE = 37.671232876712331
        CV = 394
        np.testing.assert_almost_equal(IRE_to_CV(IRE, 10, True), CV, decimal=7)

        IRE = np.tile(IRE, 6)
        CV = np.tile(CV, 6)
        np.testing.assert_almost_equal(IRE_to_CV(IRE, 10, True), CV, decimal=7)

        IRE = np.reshape(IRE, (2, 3))
        CV = np.reshape(CV, (2, 3))
        np.testing.assert_almost_equal(IRE_to_CV(IRE, 10, True), CV, decimal=7)

        IRE = np.reshape(IRE, (2, 3, 1))
        CV = np.reshape(CV, (2, 3, 1))
        np.testing.assert_almost_equal(IRE_to_CV(IRE, 10, True), CV, decimal=7)

    @ignore_numpy_errors
    def test_nan_IRE_to_CV(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.common.IRE_to_CV`
        definition nan support.
        """

        IRE_to_CV(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]), 10, True)


if __name__ == '__main__':
    unittest.main()
