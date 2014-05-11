#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**tests_matrix.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines units tests for :mod:`color.matrix` module.

**Others:**

"""

from __future__ import unicode_literals

import sys
import numpy

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

import color.matrix

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["COLOR_MATRIX_1", "COLOR_MATRIX_2", "TestIsIdentity", "TestInterpolateMatrices"]

COLOR_MATRIX_1 = numpy.matrix([0.5309, -0.0229, -0.0336,
                               -0.6241, 1.3265, 0.3337,
                               -0.0817, 0.1215, 0.6664]).reshape((3, 3))
COLOR_MATRIX_2 = numpy.matrix([0.4716, 0.0603, -0.083,
                               -0.7798, 1.5474, 0.248,
                               -0.1496, 0.1937, 0.6651]).reshape((3, 3))

class TestIsIdentity(unittest.TestCase):
    """
    Defines :func:`color.matrix.is_identity` definition units tests methods.
    """

    def test_is_identity(self):
        """
        Tests :func:`color.matrix.is_identity` definition.
        """

        self.assertTrue(color.matrix.is_identity(numpy.matrix([1, 0, 0, 0, 1, 0, 0, 0, 1]).reshape(3, 3)))
        self.assertFalse(color.matrix.is_identity(numpy.matrix([1, 2, 0, 0, 1, 0, 0, 0, 1]).reshape(3, 3)))
        self.assertTrue(color.matrix.is_identity(numpy.matrix([1, 0, 0, 1]).reshape(2, 2), n=2))
        self.assertFalse(color.matrix.is_identity(numpy.matrix([1, 2, 0, 1]).reshape(2, 2), n=2))

class TestLinearInterpolateMatrices(unittest.TestCase):
    """
    Defines :func:`color.matrix.linear_interpolate_matrices` definition units tests methods.
    """

    def test_linear_interpolate_matrices(self):
        """
        Tests :func:`color.matrix.linear_interpolate_matrices` definition.
        """

        numpy.testing.assert_almost_equal(color.matrix.linear_interpolate_matrices(2850,
                                                                                 7500,
                                                                                 COLOR_MATRIX_1,
                                                                                 COLOR_MATRIX_2,
                                                                                 6500),
                                          numpy.matrix([0.48435269, 0.04240753, -0.07237634,
                                                        -0.74631613, 1.49989462, 0.26643011,
                                                        -0.13499785, 0.17817312, 0.66537957]).reshape((3, 3)),
                                          decimal=7)
        numpy.testing.assert_almost_equal(color.matrix.linear_interpolate_matrices(2850,
                                                                                 7500,
                                                                                 COLOR_MATRIX_1,
                                                                                 COLOR_MATRIX_2,
                                                                                 1000),
                                          numpy.matrix([0.55449247, -0.05600108, -0.01394624,
                                                        -0.56215484, 1.23861505, 0.3677957,
                                                        -0.05468602, 0.09277527, 0.6669172]).reshape((3, 3)),
                                          decimal=7)
        numpy.testing.assert_almost_equal(color.matrix.linear_interpolate_matrices(2850,
                                                                                 7500,
                                                                                 COLOR_MATRIX_1,
                                                                                 COLOR_MATRIX_2,
                                                                                 50000),
                                          numpy.matrix([-0.07038925, 0.82073011, -0.53450538,
                                                        -2.20286452, 3.56637849, -0.53527957,
                                                        -0.7701914, 0.85359247, 0.65321828]).reshape((3, 3)),
                                          decimal=7)

        numpy.testing.assert_almost_equal(color.matrix.linear_interpolate_matrices(2850,
                                                                                 2850,
                                                                                 COLOR_MATRIX_1,
                                                                                 COLOR_MATRIX_2,
                                                                                 50000),
                                          COLOR_MATRIX_1,
                                          decimal=7)

if __name__ == "__main__":
    unittest.main()
