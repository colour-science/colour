#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.algebra.geometry` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.algebra import normalise_vector

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestNormaliseVector']


class TestNormaliseVector(unittest.TestCase):
    """
    Defines :func:`colour.algebra.geometry.normalise_vector` definition unit
    tests methods.
    """

    def test_normalise_vector(self):
        """
        Tests :func:`colour.algebra.geometry.normalise_vector` definition.
        """

        np.testing.assert_almost_equal(
            normalise_vector(np.array([0.07049534, 0.10080000, 0.09558313])),
            np.array([0.4525411, 0.6470803, 0.6135908]),
            decimal=7)

        np.testing.assert_almost_equal(
            normalise_vector(np.array([0.47097710, 0.34950000, 0.11301649])),
            np.array([0.7885376, 0.5851535, 0.1892189]),
            decimal=7)

        np.testing.assert_almost_equal(
            normalise_vector(np.array([0.25506814, 0.19150000, 0.08849752])),
            np.array([0.7705887, 0.5785424, 0.2673607]),
            decimal=7)


if __name__ == '__main__':
    unittest.main()
