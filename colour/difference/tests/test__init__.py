# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.difference` module."""

import unittest

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.difference import delta_E
from colour.utilities import domain_range_scale

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestDelta_E",
]


class TestDelta_E(unittest.TestCase):
    """Define :func:`colour.difference.delta_E` definition unit tests methods."""

    def test_domain_range_scale_delta_E(self):
        """
        Test :func:`colour.difference.delta_E` definition domain and range
        scale support.
        """

        Lab_1 = np.array([100.00000000, 21.57210357, 272.22819350])
        Lab_2 = np.array([100.00000000, 426.67945353, 72.39590835])

        m = ("CIE 1976", "CIE 1994", "CIE 2000", "CMC", "DIN99")
        v = [delta_E(Lab_1, Lab_2, method) for method in m]

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for method, value in zip(m, v):
            for scale, factor in d_r:
                with domain_range_scale(scale):
                    np.testing.assert_allclose(
                        delta_E(Lab_1 * factor, Lab_2 * factor, method),
                        value,
                        atol=TOLERANCE_ABSOLUTE_TESTS,
                    )


if __name__ == "__main__":
    unittest.main()
