# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.volume.macadam_limits` module."""

import unittest
from itertools import product

import numpy as np

from colour.utilities import ignore_numpy_errors
from colour.volume import is_within_macadam_limits

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestIsWithinMacadamLimits",
]


class TestIsWithinMacadamLimits(unittest.TestCase):
    """
    Define :func:`colour.volume.macadam_limits.is_within_macadam_limits`
    definition unit tests methods.
    """

    def test_is_within_macadam_limits(self):
        """
        Test :func:`colour.volume.macadam_limits.is_within_macadam_limits`
        definition.
        """

        self.assertTrue(
            is_within_macadam_limits(np.array([0.3205, 0.4131, 0.5100]), "A")
        )

        self.assertFalse(
            is_within_macadam_limits(np.array([0.0005, 0.0031, 0.0010]), "A")
        )

        self.assertTrue(
            is_within_macadam_limits(np.array([0.4325, 0.3788, 0.1034]), "C")
        )

        self.assertFalse(
            is_within_macadam_limits(np.array([0.0025, 0.0088, 0.0340]), "C")
        )

    def test_n_dimensional_is_within_macadam_limits(self):
        """
        Test :func:`colour.volume.macadam_limits.is_within_macadam_limits`
        definition n-dimensional arrays support.
        """

        a = np.array([0.3205, 0.4131, 0.5100])
        b = is_within_macadam_limits(a, "A")

        a = np.tile(a, (6, 1))
        b = np.tile(b, 6)
        np.testing.assert_allclose(is_within_macadam_limits(a, "A"), b)

        a = np.reshape(a, (2, 3, 3))
        b = np.reshape(b, (2, 3))
        np.testing.assert_allclose(is_within_macadam_limits(a, "A"), b)

    @ignore_numpy_errors
    def test_nan_is_within_macadam_limits(self):
        """
        Test :func:`colour.volume.macadam_limits.is_within_macadam_limits`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        is_within_macadam_limits(cases, "A")


if __name__ == "__main__":
    unittest.main()
