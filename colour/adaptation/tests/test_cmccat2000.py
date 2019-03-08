# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.adaptation.cmccat2000.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from itertools import permutations

from colour.adaptation.cmccat2000 import (
    chromatic_adaptation_forward_CMCCAT2000,
    chromatic_adaptation_reverse_CMCCAT2000)

from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'TestChromaticAdaptationForwardCMCCAT2000',
    'TestChromaticAdaptationReverseCMCCAT2000'
]


class TestChromaticAdaptationForwardCMCCAT2000(unittest.TestCase):
    """
    Defines :func:`colour.adaptation.cmccat2000.\
chromatic_adaptation_forward_CMCCAT2000` definition unit tests methods.
    """

    def test_chromatic_adaptation_forward_CMCCAT2000(self):
        """
        Tests :func:`colour.adaptation.cmccat2000.\
chromatic_adaptation_forward_CMCCAT2000` definition.
        """

        np.testing.assert_almost_equal(
            chromatic_adaptation_forward_CMCCAT2000(
                np.array([22.48, 22.74, 8.54]),
                np.array([111.15, 100.00, 35.20]),
                np.array([94.81, 100.00, 107.30]), 200, 200),
            np.array([19.52698326, 23.06833960, 24.97175229]),
            decimal=7)

        np.testing.assert_almost_equal(
            chromatic_adaptation_forward_CMCCAT2000(
                np.array([0.14222010, 0.23042768, 0.10495772]) * 100,
                np.array([0.95045593, 1.00000000, 1.08905775]) * 100,
                np.array([1.09846607, 1.00000000, 0.35582280]) * 100, 100,
                100),
            np.array([17.90511171, 22.75299363, 3.79837384]),
            decimal=7)

        np.testing.assert_almost_equal(
            chromatic_adaptation_forward_CMCCAT2000(
                np.array([0.07818780, 0.06157201, 0.28099326]) * 100,
                np.array([0.95045593, 1.00000000, 1.08905775]) * 100,
                np.array([0.99144661, 1.00000000, 0.67315942]) * 100, 100,
                100),
            np.array([6.76564344, 5.86585763, 18.40577315]),
            decimal=7)

    def test_n_dimensional_chromatic_adaptation_forward_CMCCAT2000(self):
        """
        Tests :func:`colour.adaptation.cmccat2000.\
chromatic_adaptation_forward_CMCCAT2000` definition n-dimensional arrays
        support.
        """

        XYZ = np.array([22.48, 22.74, 8.54])
        XYZ_w = np.array([111.15, 100.00, 35.20])
        XYZ_wr = np.array([94.81, 100.00, 107.30])
        L_A1 = 200
        L_A2 = 200
        XYZ_c = np.array([19.52698326, 23.06833960, 24.97175229])
        np.testing.assert_almost_equal(
            chromatic_adaptation_forward_CMCCAT2000(XYZ, XYZ_w, XYZ_wr, L_A1,
                                                    L_A2),
            XYZ_c,
            decimal=7)

        XYZ = np.tile(XYZ, (6, 1))
        XYZ_c = np.tile(XYZ_c, (6, 1))
        np.testing.assert_almost_equal(
            chromatic_adaptation_forward_CMCCAT2000(XYZ, XYZ_w, XYZ_wr, L_A1,
                                                    L_A2),
            XYZ_c,
            decimal=7)

        XYZ_w = np.tile(XYZ_w, (6, 1))
        XYZ_wr = np.tile(XYZ_wr, (6, 1))
        L_A1 = np.tile(L_A1, 6)
        L_A2 = np.tile(L_A2, 6)
        np.testing.assert_almost_equal(
            chromatic_adaptation_forward_CMCCAT2000(XYZ, XYZ_w, XYZ_wr, L_A1,
                                                    L_A2),
            XYZ_c,
            decimal=7)

        XYZ = np.reshape(XYZ, (2, 3, 3))
        XYZ_w = np.reshape(XYZ_w, (2, 3, 3))
        XYZ_wr = np.reshape(XYZ_wr, (2, 3, 3))
        L_A1 = np.reshape(L_A1, (2, 3))
        L_A2 = np.reshape(L_A2, (2, 3))
        XYZ_c = np.reshape(XYZ_c, (2, 3, 3))
        np.testing.assert_almost_equal(
            chromatic_adaptation_forward_CMCCAT2000(XYZ, XYZ_w, XYZ_wr, L_A1,
                                                    L_A2),
            XYZ_c,
            decimal=7)

    def test_domain_range_scale_chromatic_adaptation_CMCCAT2000(self):
        """
        Tests :func:`colour.adaptation.cmccat2000.\
chromatic_adaptation_forward_CMCCAT2000` definition domain and range scale
        support.
        """

        XYZ = np.array([22.48, 22.74, 8.54])
        XYZ_w = np.array([111.15, 100.00, 35.20])
        XYZ_wr = np.array([94.81, 100.00, 107.30])
        L_A1 = 200
        L_A2 = 200
        XYZ_c = chromatic_adaptation_forward_CMCCAT2000(
            XYZ, XYZ_w, XYZ_wr, L_A1, L_A2)

        d_r = (('reference', 1), (1, 0.01), (100, 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    chromatic_adaptation_forward_CMCCAT2000(
                        XYZ * factor, XYZ_w * factor, XYZ_wr * factor, L_A1,
                        L_A2),
                    XYZ_c * factor,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_chromatic_adaptation_forward_CMCCAT2000(self):
        """
        Tests :func:`colour.adaptation.cmccat2000.\
chromatic_adaptation_forward_CMCCAT2000` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ = np.array(case)
            XYZ_w = np.array(case)
            XYZ_wr = np.array(case)
            L_A1 = case[0]
            L_A2 = case[0]
            chromatic_adaptation_forward_CMCCAT2000(XYZ, XYZ_w, XYZ_wr, L_A1,
                                                    L_A2)


class TestChromaticAdaptationReverseCMCCAT2000(unittest.TestCase):
    """
    Defines :func:`colour.adaptation.cmccat2000.\
chromatic_adaptation_reverse_CMCCAT2000` definition unit tests methods.
    """

    def test_chromatic_adaptation_reverse_CMCCAT2000(self):
        """
        Tests :func:`colour.adaptation.cmccat2000.\
chromatic_adaptation_reverse_CMCCAT2000` definition.
        """

        np.testing.assert_almost_equal(
            chromatic_adaptation_reverse_CMCCAT2000(
                np.array([19.52698326, 23.06833960, 24.97175229]),
                np.array([111.15, 100.00, 35.20]),
                np.array([94.81, 100.00, 107.30]), 200, 200),
            np.array([22.48, 22.74, 8.54]),
            decimal=7)

        np.testing.assert_almost_equal(
            chromatic_adaptation_reverse_CMCCAT2000(
                np.array([17.90511171, 22.75299363, 3.79837384]),
                np.array([0.95045593, 1.00000000, 1.08905775]) * 100,
                np.array([1.09846607, 1.00000000, 0.35582280]) * 100, 100,
                100),
            np.array([0.14222010, 0.23042768, 0.10495772]) * 100,
            decimal=7)

        np.testing.assert_almost_equal(
            chromatic_adaptation_reverse_CMCCAT2000(
                np.array([6.76564344, 5.86585763, 18.40577315]),
                np.array([0.95045593, 1.00000000, 1.08905775]) * 100,
                np.array([0.99144661, 1.00000000, 0.67315942]) * 100, 100,
                100),
            np.array([0.07818780, 0.06157201, 0.28099326]) * 100,
            decimal=7)

    def test_n_dimensional_chromatic_adaptation_reverse_CMCCAT2000(self):
        """
        Tests :func:`colour.adaptation.cmccat2000.\
chromatic_adaptation_reverse_CMCCAT2000` definition n-dimensional arrays
        support.
        """

        XYZ_c = np.array([19.52698326, 23.06833960, 24.97175229])
        XYZ_w = np.array([111.15, 100.00, 35.20])
        XYZ_wr = np.array([94.81, 100.00, 107.30])
        L_A1 = 200
        L_A2 = 200
        XYZ = np.array([22.48, 22.74, 8.54])
        np.testing.assert_almost_equal(
            chromatic_adaptation_reverse_CMCCAT2000(XYZ_c, XYZ_w, XYZ_wr, L_A1,
                                                    L_A2),
            XYZ,
            decimal=7)

        XYZ_c = np.tile(XYZ_c, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_almost_equal(
            chromatic_adaptation_reverse_CMCCAT2000(XYZ_c, XYZ_w, XYZ_wr, L_A1,
                                                    L_A2),
            XYZ,
            decimal=7)

        XYZ_w = np.tile(XYZ_w, (6, 1))
        XYZ_wr = np.tile(XYZ_wr, (6, 1))
        L_A1 = np.tile(L_A1, 6)
        L_A2 = np.tile(L_A2, 6)
        np.testing.assert_almost_equal(
            chromatic_adaptation_reverse_CMCCAT2000(XYZ_c, XYZ_w, XYZ_wr, L_A1,
                                                    L_A2),
            XYZ,
            decimal=7)

        XYZ_c = np.reshape(XYZ_c, (2, 3, 3))
        XYZ_w = np.reshape(XYZ_w, (2, 3, 3))
        XYZ_wr = np.reshape(XYZ_wr, (2, 3, 3))
        L_A1 = np.reshape(L_A1, (2, 3))
        L_A2 = np.reshape(L_A2, (2, 3))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_almost_equal(
            chromatic_adaptation_reverse_CMCCAT2000(XYZ_c, XYZ_w, XYZ_wr, L_A1,
                                                    L_A2),
            XYZ,
            decimal=7)

    def test_domain_range_scale_chromatic_adaptation_CMCCAT2000(self):
        """
        Tests :func:`colour.adaptation.cmccat2000.\
chromatic_adaptation_reverse_CMCCAT2000` definition domain and range scale
        support.
        """

        XYZ_c = np.array([19.52698326, 23.06833960, 24.97175229])
        XYZ_w = np.array([111.15, 100.00, 35.20])
        XYZ_wr = np.array([94.81, 100.00, 107.30])
        L_A1 = 200
        L_A2 = 200
        XYZ = chromatic_adaptation_reverse_CMCCAT2000(XYZ_c, XYZ_w, XYZ_wr,
                                                      L_A1, L_A2)

        d_r = (('reference', 1), (1, 0.01), (100, 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    chromatic_adaptation_reverse_CMCCAT2000(
                        XYZ_c * factor, XYZ_w * factor, XYZ_wr * factor, L_A1,
                        L_A2),
                    XYZ * factor,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_chromatic_adaptation_reverse_CMCCAT2000(self):
        """
        Tests :func:`colour.adaptation.cmccat2000.\
chromatic_adaptation_reverse_CMCCAT2000` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ_c = np.array(case)
            XYZ_w = np.array(case)
            XYZ_wr = np.array(case)
            L_A1 = case[0]
            L_A2 = case[0]
            chromatic_adaptation_reverse_CMCCAT2000(XYZ_c, XYZ_w, XYZ_wr, L_A1,
                                                    L_A2)


if __name__ == '__main__':
    unittest.main()
