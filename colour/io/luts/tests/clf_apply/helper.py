# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.io.clf` module."""
import unittest

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

import numpy as np

from colour.io.luts.clf import from_f16_to_uint16, from_uint16_to_f16



class TestHelpers:
    """
    Define :func:`colour.io.luts.clf.from_uint16_to_f16` and
    :func:`colour.io.luts.clf.from_f16_to_uint16` unit tests methods.
    """

    def test_uint16_to_f16(self):
        """
        Test :func:`colour.io.luts.clf.from_uint16_to_f16` method.
        """
        value = np.array([0, 15360])
        output = from_uint16_to_f16(value)
        expected = np.array([0, 1.0])
        np.testing.assert_almost_equal(output, expected)

    def test_f16_to_uint16(self):
        """
        Test :func:`colour.io.luts.clf.from_f16_to_uint16` method.
        """
        value = np.array([0, 1.0])
        output = from_f16_to_uint16(value)
        expected = np.array([0, 15360])
        np.testing.assert_almost_equal(output, expected)

    def test_conversion_reversible(self):
        """
        Test :func:`colour.io.luts.clf.from_uint16_to_f16` method with
        :func:`colour.io.luts.clf.from_f16_to_uint16`.

        """
        for i in range(2**16):
            value = np.array([i])
            float_value = from_uint16_to_f16(value)
            int_value = from_f16_to_uint16(float_value)
            np.testing.assert_almost_equal(value, int_value)


if __name__ == "__main__":
    unittest.main()
