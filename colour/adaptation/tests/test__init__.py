# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.adaptation` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from six.moves import zip

from colour.adaptation import chromatic_adaptation
from colour.utilities import domain_range_scale

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestChromaticAdaptation']


class TestChromaticAdaptation(unittest.TestCase):
    """
    Defines :func:`colour.adaptation.chromatic_adaptation` definition unit
    tests methods.
    """

    def test_chromatic_adaptation(self):
        """
        Tests :func:`colour.adaptation.chromatic_adaptation` definition.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        XYZ_w = np.array([0.95045593, 1.00000000, 1.08905775])
        XYZ_wr = np.array([0.96429568, 1.00000000, 0.82510460])
        np.testing.assert_almost_equal(
            chromatic_adaptation(XYZ, XYZ_w, XYZ_wr),
            np.array([0.21638819, 0.12570000, 0.03847494]),
            decimal=7)

        Y_o = 0.2
        E_o = 1000
        np.testing.assert_almost_equal(
            chromatic_adaptation(
                XYZ,
                XYZ_w,
                XYZ_wr,
                method='CIE 1994',
                Y_o=Y_o,
                E_o1=E_o,
                E_o2=E_o),
            np.array([0.21347453, 0.12252986, 0.03347887]),
            decimal=7)

        L_A = 200
        np.testing.assert_almost_equal(
            chromatic_adaptation(
                XYZ, XYZ_w, XYZ_wr, method='CMCCAT2000', L_A1=L_A, L_A2=L_A),
            np.array([0.21498829, 0.12474711, 0.03910138]),
            decimal=7)

        Y_n = 200
        np.testing.assert_almost_equal(
            chromatic_adaptation(
                XYZ, XYZ_w, XYZ_wr, method='Fairchild 1990', Y_n=Y_n),
            np.array([0.21394049, 0.12262315, 0.03891917]),
            decimal=7)

    def test_domain_range_scale_chromatic_adaptation(self):
        """
        Tests :func:`colour.adaptation.chromatic_adaptation` definition domain
        and range scale support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        XYZ_w = np.array([0.95045593, 1.00000000, 1.08905775])
        XYZ_wr = np.array([0.96429568, 1.00000000, 0.82510460])
        Y_o = 0.2
        E_o = 1000
        L_A = 200
        Y_n = 200

        m = ('Von Kries', 'CIE 1994', 'CMCCAT2000', 'Fairchild 1990')
        v = [
            chromatic_adaptation(
                XYZ,
                XYZ_w,
                XYZ_wr,
                method=method,
                Y_o=Y_o,
                E_o1=E_o,
                E_o2=E_o,
                L_A1=L_A,
                L_A2=L_A,
                Y_n=Y_n) for method in m
        ]

        d_r = (('reference', 1), (1, 1), (100, 100))
        for method, value in zip(m, v):
            for scale, factor in d_r:
                with domain_range_scale(scale):
                    np.testing.assert_almost_equal(
                        chromatic_adaptation(
                            XYZ * factor,
                            XYZ_w * factor,
                            XYZ_wr * factor,
                            method=method,
                            Y_o=Y_o * factor,
                            E_o1=E_o,
                            E_o2=E_o,
                            L_A1=L_A,
                            L_A2=L_A,
                            Y_n=Y_n),
                        value * factor,
                        decimal=7)


if __name__ == '__main__':
    unittest.main()
