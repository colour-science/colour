# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.volume.mesh` module."""

import unittest
from itertools import product

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.utilities import ignore_numpy_errors
from colour.volume import is_within_mesh_volume

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestIsWithinMeshVolume",
]


class TestIsWithinMeshVolume(unittest.TestCase):
    """
    Define :func:`colour.volume.mesh.is_within_mesh_volume` definition unit
    tests methods.
    """

    def setUp(self):
        """Initialise the common tests attributes."""

        self._mesh = np.array(
            [
                [-1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0],
                [1.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0],
                [0.0, 1.0, 0.0],
            ]
        )

    def test_is_within_mesh_volume(self):
        """Test :func:`colour.volume.mesh.is_within_mesh_volume` definition."""

        self.assertTrue(
            is_within_mesh_volume(
                np.array([0.0005, 0.0031, 0.0010]), self._mesh
            )
        )

        self.assertFalse(
            is_within_mesh_volume(
                np.array([0.3205, 0.4131, 0.5100]), self._mesh
            )
        )

        self.assertTrue(
            is_within_mesh_volume(
                np.array([0.0025, 0.0088, 0.0340]), self._mesh
            )
        )

        self.assertFalse(
            is_within_mesh_volume(
                np.array([0.4325, 0.3788, 0.1034]), self._mesh
            )
        )

    def test_n_dimensional_is_within_mesh_volume(self):
        """
        Test :func:`colour.volume.mesh.is_within_mesh_volume` definition
        n-dimensional arrays support.
        """

        a = np.array([0.0005, 0.0031, 0.0010])
        b = is_within_mesh_volume(a, self._mesh)

        a = np.tile(a, (6, 1))
        b = np.tile(b, 6)
        np.testing.assert_allclose(
            is_within_mesh_volume(a, self._mesh),
            b,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = np.reshape(a, (2, 3, 3))
        b = np.reshape(b, (2, 3))
        np.testing.assert_allclose(
            is_within_mesh_volume(a, self._mesh),
            b,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_is_within_mesh_volume(self):
        """
        Test :func:`colour.volume.mesh.is_within_mesh_volume` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        is_within_mesh_volume(cases, self._mesh)


if __name__ == "__main__":
    unittest.main()
