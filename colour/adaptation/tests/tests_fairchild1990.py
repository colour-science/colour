# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.adaptation.fairchild1990` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

from colour.adaptation import chromatic_adaptation_Fairchild1990

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestChromaticAdaptationFairchild1990']


class TestChromaticAdaptationFairchild1990(unittest.TestCase):
    """
    Defines
    :func:`colour.adaptation.fairchild1990.chromatic_adaptation_Fairchild1990`
    definition unit tests methods.
    """

    def test_chromatic_adaptation_Fairchild1990(self):
        """
        Tests
        :func:`colour.adaptation.fairchild1990.chromatic_adaptation_Fairchild1990`  # noqa
        definition.
        """

        np.testing.assert_almost_equal(
            chromatic_adaptation_Fairchild1990(
                np.array([0.07049534, 0.1008, 0.09558313]) * 100,
                np.array([1.09846607, 1., 0.3558228]) * 100,
                np.array([0.95042855, 1., 1.08890037]) * 100,
                200),
            np.array([8.35782287, 10.21428897, 29.25065668]),
            decimal=7)

        np.testing.assert_almost_equal(
            chromatic_adaptation_Fairchild1990(
                np.array([0.4709771, 0.3495, 0.11301649]) * 100,
                np.array([0.99092745, 1., 0.85313273]) * 100,
                np.array([1.01679082, 1., 0.67610122]) * 100,
                200),
            np.array([49.00577034, 35.03909328, 8.95647114]),
            decimal=7)

        np.testing.assert_almost_equal(
            chromatic_adaptation_Fairchild1990(
                np.array([0.25506814, 0.1915, 0.08849752]) * 100,
                np.array([0.98070597, 1., 1.18224949]) * 100,
                np.array([0.92833635, 1., 1.0366472]) * 100,
                200),
            np.array([24.79473034, 19.13024207, 7.75984317]),
            decimal=7)


if __name__ == '__main__':
    unittest.main()
