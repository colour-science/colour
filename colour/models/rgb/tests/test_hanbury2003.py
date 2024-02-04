"""Defines unit tests for :mod:`colour.models.rgb.hanbury2003` module."""

import unittest
from itertools import product

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb import IHLS_to_RGB, RGB_to_IHLS
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestRGB_to_IHLS",
    "TestIHLS_to_RGB",
]


class TestRGB_to_IHLS(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.hanbury2003.RGB_to_IHLS` definition unit
    tests methods.
    """

    def test_RGB_to_IHLS(self):
        """Test :func:`colour.models.rgb.hanbury2003.RGB_to_IHLS` definition."""

        np.testing.assert_allclose(
            RGB_to_IHLS(np.array([0.45620519, 0.03081071, 0.04091952])),
            np.array([6.26236117, 0.12197943, 0.42539448]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            RGB_to_IHLS(np.array([0.00000000, 0.00000000, 0.00000000])),
            np.array([0.00000000, 0.00000000, 0.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            RGB_to_IHLS(np.array([1.00000000, 1.00000000, 1.00000000])),
            np.array([0.00000000, 1.00000000, 0.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_RGB_to_IHLS(self):
        """
        Test :func:`colour.models.rgb.hanbury2003.RGB_to_IHLS` definition
        n-dimensional arrays support.
        """

        RGB = np.array([0.45620519, 0.03081071, 0.04091952])
        HYS = RGB_to_IHLS(RGB)

        RGB = np.tile(RGB, (6, 1))
        HYS = np.tile(HYS, (6, 1))
        np.testing.assert_allclose(RGB_to_IHLS(RGB), HYS, atol=TOLERANCE_ABSOLUTE_TESTS)

        RGB = np.reshape(RGB, (2, 3, 3))
        HYS = np.reshape(HYS, (2, 3, 3))
        np.testing.assert_allclose(RGB_to_IHLS(RGB), HYS, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_RGB_to_IHLS(self):
        """
        Test :func:`colour.models.rgb.hanbury2003.RGB_to_IHLS` definition
        domain and range scale support.
        """

        RGB = np.array([0.45620519, 0.03081071, 0.04091952])
        HYS = RGB_to_IHLS(RGB)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    RGB_to_IHLS(RGB * factor),
                    HYS * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_RGB_to_IHLS(self):
        """
        Test :func:`colour.models.rgb.hanbury2003.RGB_to_IHLS` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        RGB_to_IHLS(cases)


class TestIHLS_to_RGB(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.hanbury2003.RGB_to_IHLS` definition unit
    tests methods.
    """

    def test_IHLS_to_RGB(self):
        """Test :func:`colour.models.rgb.hanbury2003.IHLS_to_RGB` definition."""

        np.testing.assert_allclose(
            IHLS_to_RGB(np.array([6.26236117, 0.12197943, 0.42539448])),
            np.array([0.45620519, 0.03081071, 0.04091952]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            IHLS_to_RGB(np.array([0.00000000, 0.00000000, 0.00000000])),
            np.array([0.00000000, 0.00000000, 0.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            IHLS_to_RGB(np.array([0.00000000, 1.00000000, 0.00000000])),
            np.array([1.00000000, 1.00000000, 1.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_IHLS_to_RGB(self):
        """
        Test :func:`colour.models.rgb.hanbury2003.IHLS_to_RGB` definition
        n-dimensional arrays support.
        """

        HYS = np.array([6.26236117, 0.12197943, 0.42539448])
        RGB = IHLS_to_RGB(HYS)

        HYS = np.tile(HYS, (6, 1))
        RGB = np.tile(RGB, (6, 1))
        np.testing.assert_allclose(IHLS_to_RGB(HYS), RGB, atol=TOLERANCE_ABSOLUTE_TESTS)

        HYS = np.reshape(HYS, (2, 3, 3))
        RGB = np.reshape(RGB, (2, 3, 3))
        np.testing.assert_allclose(IHLS_to_RGB(HYS), RGB, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_IHLS_to_RGB(self):
        """
        Test :func:`colour.models.rgb.hanbury2003.IHLS_to_RGB` definition
        domain and range scale support.
        """

        HYS = np.array([6.26236117, 0.12197943, 0.42539448])
        RGB = IHLS_to_RGB(HYS)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    IHLS_to_RGB(HYS * factor),
                    RGB * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_IHLS_to_RGB(self):
        """
        Test :func:`colour.models.rgb.hanbury2003.IHLS_to_RGB` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        IHLS_to_RGB(cases)


if __name__ == "__main__":
    unittest.main()
