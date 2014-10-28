#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.colorimetry.whiteness` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

from colour.colorimetry import (
    whiteness_Berger1959,
    whiteness_Taube1960,
    whiteness_Stensby1968,
    whiteness_ASTM313,
    whiteness_Ganz1979,
    whiteness_CIE2004)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestWhitenessBerger1959',
           'TestWhitenessTaube1960',
           'TestWhitenessStensby1968',
           'TestWhitenessASTM313',
           'TestWhitenessGanz1979',
           'TestWhitenessCIE2004']


class TestWhitenessBerger1959(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.whiteness.whiteness_Berger1959`
    definition unit tests methods.
    """

    def test_whiteness_Berger1959(self):
        """
        Tests :func:`colour.colorimetry.whiteness.whiteness_Berger1959`
        definition.
        """

        self.assertAlmostEqual(
            whiteness_Berger1959(
                np.array([95., 100., 105.]),
                np.array([94.80966767, 100., 107.30513595])),
            30.36380178871724,
            places=7)
        self.assertAlmostEqual(
            whiteness_Berger1959(
                np.array([105., 100., 95.]),
                np.array([94.80966767, 100., 107.30513595])),
            5.5304692806739411,
            places=7)
        self.assertAlmostEqual(
            whiteness_Berger1959(
                np.array([100., 100., 100.]),
                np.array([100, 100., 100.])),
            33.300000000000011,
            places=7)


class TestWhitenessTaube1960(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.whiteness.whiteness_Taube1960`
    definition unit tests methods.
    """

    def test_whiteness_Taube1960(self):
        """
        Tests :func:`colour.colorimetry.whiteness.whiteness_Taube1960`
        definition.
        """

        self.assertAlmostEqual(
            whiteness_Taube1960(
                np.array([95., 100., 105.]),
                np.array([94.80966767, 100., 107.30513595])),
            91.407173833416152,
            places=7)
        self.assertAlmostEqual(
            whiteness_Taube1960(
                np.array([105., 100., 95.]),
                np.array([94.80966767, 100., 107.30513595])),
            54.130300134995593,
            places=7)
        self.assertAlmostEqual(
            whiteness_Taube1960(
                np.array([100., 100., 100.]),
                np.array([100, 100., 100.])),
            100.0,
            places=7)


class TestWhitenessStensby1968(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.whiteness.whiteness_Stensby1968`
    definition unit tests methods.
    """

    def test_whiteness_Stensby1968(self):
        """
        Tests :func:`colour.colorimetry.whiteness.whiteness_Stensby1968`
        definition.
        """

        self.assertAlmostEqual(
            whiteness_Stensby1968(
                np.array([100., -2.46875131, -16.72486654])),
            142.76834569000002,
            places=7)
        self.assertAlmostEqual(
            whiteness_Stensby1968(
                np.array([100., 14.40943727, -9.61394885])),
            172.07015836000002,
            places=7)
        self.assertAlmostEqual(
            whiteness_Stensby1968(
                np.array([1., 1., 1.])),
            1.0,
            places=7)


class TestWhitenessASTM313(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.whiteness.whiteness_ASTM313`
    definition unit tests methods.
    """

    def test_whiteness_ASTM313(self):
        """
        Tests :func:`colour.colorimetry.whiteness.whiteness_ASTM313`
        definition.
        """

        self.assertAlmostEqual(
            whiteness_ASTM313(
                np.array([95., 100., 105.])),
            55.740000000000009,
            places=7)
        self.assertAlmostEqual(
            whiteness_ASTM313(
                np.array([105., 100., 95.])),
            21.860000000000014,
            places=7)
        self.assertAlmostEqual(
            whiteness_ASTM313(
                np.array([100., 100., 100.])),
            38.800000000000011,
            places=7)


class TestWhitenessGanz1979(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.whiteness.whiteness_Ganz1979`
    definition unit tests methods.
    """

    def test_whiteness_Ganz1979(self):
        """
        Tests :func:`colour.colorimetry.whiteness.whiteness_Ganz1979`
        definition.
        """

        np.testing.assert_almost_equal(
            whiteness_Ganz1979((0.3139, 0.3311), 100),
            np.array([99.3317652, 1.7610829]),
            decimal=7)
        np.testing.assert_almost_equal(
            whiteness_Ganz1979((0.3500, 0.3334), 100),
            np.array([23.385254, -32.6618256]),
            decimal=7)
        np.testing.assert_almost_equal(
            whiteness_Ganz1979((0.3334, 0.3334), 100),
            np.array([54.3993992, -16.0415238]),
            decimal=7)


class TestWhitenessCIE2004(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.whiteness.whiteness_CIE2004`
    definition unit tests methods.
    """

    def test_whiteness_CIE2004(self):
        """
        Tests :func:`colour.colorimetry.whiteness.whiteness_CIE2004`
        definition.
        """

        np.testing.assert_almost_equal(
            whiteness_CIE2004((0.3139, 0.3311), 100, (0.3139, 0.3311)),
            np.array([100., 0.]),
            decimal=7)
        np.testing.assert_almost_equal(
            whiteness_CIE2004((0.3500, 0.3334), 100, (0.3139, 0.3311)),
            np.array([67.21, -34.605]),
            decimal=7)
        np.testing.assert_almost_equal(
            whiteness_CIE2004((0.3334, 0.3334), 100, (0.3139, 0.3311)),
            np.array([80.49, -18.005]),
            decimal=7)


if __name__ == '__main__':
    unittest.main()
