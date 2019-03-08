# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.volume.mesh` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from itertools import permutations

from colour.volume import is_within_mesh_volume
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestIsWithinMeshVolume']


class TestIsWithinMeshVolume(unittest.TestCase):
    """
    Defines :func:`colour.volume.mesh.is_within_mesh_volume` definition unit
    tests methods.
    """

    def setUp(self):
        """
        Initialises common tests attributes.
        """

        self._mesh = np.array([
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0],
            [0.0, 1.0, 0.0],
        ])

    def test_is_within_mesh_volume(self):
        """
        Tests :func:`colour.volume.mesh.is_within_mesh_volume` definition.
        """

        self.assertTrue(
            is_within_mesh_volume(
                np.array([0.0005, 0.0031, 0.0010]), self._mesh))

        self.assertFalse(
            is_within_mesh_volume(
                np.array([0.3205, 0.4131, 0.5100]), self._mesh))

        self.assertTrue(
            is_within_mesh_volume(
                np.array([0.0025, 0.0088, 0.0340]), self._mesh))

        self.assertFalse(
            is_within_mesh_volume(
                np.array([0.4325, 0.3788, 0.1034]), self._mesh))

    def test_n_dimensional_is_within_mesh_volume(self):
        """
        Tests :func:`colour.volume.mesh.is_within_mesh_volume` definition
        n-dimensional arrays support.
        """

        a = np.array([0.0005, 0.0031, 0.0010])
        b = np.array([True])
        np.testing.assert_almost_equal(is_within_mesh_volume(a, self._mesh), b)

        a = np.tile(a, (6, 1))
        b = np.tile(b, 6)
        np.testing.assert_almost_equal(is_within_mesh_volume(a, self._mesh), b)

        a = np.reshape(a, (2, 3, 3))
        b = np.reshape(b, (2, 3))
        np.testing.assert_almost_equal(is_within_mesh_volume(a, self._mesh), b)

    @ignore_numpy_errors
    def test_nan_is_within_mesh_volume(self):
        """
        Tests :func:`colour.volume.mesh.is_within_mesh_volume` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            is_within_mesh_volume(case, self._mesh)


if __name__ == '__main__':
    unittest.main()
