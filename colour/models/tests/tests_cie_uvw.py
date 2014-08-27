#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.models.cie_uvw` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

from colour.models import XYZ_to_UVW

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestXYZ_to_UVW']


class TestXYZ_to_UVW(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_uvw.XYZ_to_UVW` definition unit tests
    methods.
    """

    def test_XYZ_to_UVW(self):
        """
        Tests :func:`colour.models.cie_uvw.XYZ_to_UVW` definition.
        """

        np.testing.assert_almost_equal(
            XYZ_to_UVW(np.array([0.96907232, 1, 1.12179215])),
            np.array([-0.90199113, -1.56588889, 8.]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_UVW(np.array([1.92001986, 1, - 0.1241347])),
            np.array([26.5159289, 3.8694711, 8.]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_UVW(np.array([1.0131677, 1, 2.11217686])),
            np.array([-2.89423113, -5.92004891, 8.]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_UVW(np.array([1.0131677, 1, 2.11217686]),
                       (0.44757, 0.40745)),
            np.array([-7.76195429, -8.43122502, 8.]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_UVW(np.array([1.0131677, 1, 2.11217686]),
                       (1 / 3, 1 / 3)),
            np.array([-3.03641679, -4.92226526, 8.]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_UVW(np.array([1.0131677, 1, 2.11217686]),
                       (0.31271, 0.32902)),
            np.array([-1.7159427, -4.55119033, 8.]),
            decimal=7)


if __name__ == '__main__':
    unittest.main()
