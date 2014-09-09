# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.adaptation.cmccat2000.
"""

from __future__ import division, unicode_literals

import numpy as np
import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

from colour.adaptation import CMCCAT2000_forward, CMCCAT2000_reverse

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestCMCCAT2000Forward',
           'TestCMCCAT2000Reverse']


class TestCMCCAT2000Forward(unittest.TestCase):
    """
    Defines :func:`colour.adaptation.cmccat2000.CMCCAT2000_forward` definition
    unit tests methods.
    """

    def test_CMCCAT2000_forward(self):
        """
        Tests :func:`colour.adaptation.cmccat2000.CMCCAT2000_forward`
        definition.
        """

        np.testing.assert_almost_equal(
            CMCCAT2000_forward(
                np.array([96.907232, 100, 112.179215]),
                np.array([111.15, 100.00, 35.20]),
                np.array([94.81, 100.00, 107.30])),
            np.array([107.01491291, 107.00756767, 313.56612459]),
            decimal=7)

        np.testing.assert_almost_equal(
            CMCCAT2000_forward(
                np.array([192.001986, 100, -12.41347]),
                np.array([109.923822, 100, 35.445412]),
                np.array([101.31677, 100, 211.217686])),
            np.array([134.6417251, 82.44555833, -53.67730762]),
            decimal=7)

        np.testing.assert_almost_equal(
            CMCCAT2000_forward(
                np.array([101.31677, 100, 211.217686]),
                np.array([192001986, 100, -12.41347]),
                np.array([101.31677, 100, 211.217686])),
            np.array([8.14737727, 8.0090015, 17.16407632]),
            decimal=7)


class TestCMCCAT2000Reverse(unittest.TestCase):
    """
    Defines :func:`colour.adaptation.cmccat2000.CMCCAT2000_reverse` definition
    unit tests methods.
    """

    def test_CMCCAT2000_reverse(self):
        """
        Tests :func:`colour.adaptation.cmccat2000.CMCCAT2000_reverse`
        definition.
        """

        np.testing.assert_almost_equal(
            CMCCAT2000_reverse(
                np.array([107.01491291, 107.00756767, 313.56612459]),
                np.array([111.15, 100.00, 35.20]),
                np.array([94.81, 100.00, 107.30])),
            np.array([96.907232, 100, 112.179215]),
            decimal=7)

        np.testing.assert_almost_equal(
            CMCCAT2000_reverse(
                np.array([134.6417251, 82.44555833, -53.67730762]),
                np.array([109.923822, 100, 35.445412]),
                np.array([101.31677, 100, 211.217686])),
            np.array([192.001986, 100, -12.41347]),
            decimal=7)

        np.testing.assert_almost_equal(
            CMCCAT2000_reverse(
                np.array([8.14737727, 8.0090015, 17.16407632]),
                np.array([192001986, 100, -12.41347]),
                np.array([101.31677, 100, 211.217686])),
            np.array([101.31677, 100, 211.217686]),
            decimal=7)


if __name__ == '__main__':
    unittest.main()
