# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.appearance.atd95` module."""

import unittest
from itertools import product

import numpy as np

from colour.appearance import XYZ_to_ATD95
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.utilities import (
    as_float_array,
    domain_range_scale,
    ignore_numpy_errors,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestXYZ_to_ATD95",
]


class TestXYZ_to_ATD95(unittest.TestCase):
    """
    Define :func:`colour.appearance.atd95.XYZ_to_ATD95` definition unit
    tests methods.
    """

    def test_XYZ_to_ATD95(self):
        """
        Test :func:`colour.appearance.atd95.XYZ_to_ATD95` definition.

        Notes
        -----
        -   The test values have been generated from data of the following file
            by *Fairchild (2013)*:
            http://rit-mcsl.org/fairchild//files/AppModEx.xls
        """

        XYZ = np.array([19.01, 20.00, 21.78])
        XYZ_0 = np.array([95.05, 100.00, 108.88])
        Y_02 = 318.31
        K_1 = 0
        K_2 = 50
        sigma = 300
        np.testing.assert_allclose(
            XYZ_to_ATD95(XYZ, XYZ_0, Y_02, K_1, K_2, sigma),
            np.array(
                [
                    1.91,
                    1.206,
                    0.1814,
                    0.1788,
                    0.0287,
                    0.0108,
                    0.0192,
                    0.0205,
                    0.0108,
                ]
            ),
            atol=0.01,
        )

        XYZ = np.array([57.06, 43.06, 31.96])
        Y_02 = 31.83
        np.testing.assert_allclose(
            XYZ_to_ATD95(XYZ, XYZ_0, Y_02, K_1, K_2, sigma),
            np.array(
                [
                    63.96,
                    1.371,
                    0.2142,
                    0.2031,
                    0.068,
                    0.0005,
                    0.0224,
                    0.0308,
                    0.0005,
                ]
            ),
            atol=0.01,
        )

        XYZ = np.array([3.53, 6.56, 2.14])
        XYZ_0 = np.array([109.85, 100.00, 35.58])
        Y_02 = 318.31
        np.testing.assert_allclose(
            XYZ_to_ATD95(XYZ, XYZ_0, Y_02, K_1, K_2, sigma),
            np.array(
                [
                    -0.31,
                    0.436,
                    0.1075,
                    0.1068,
                    -0.011,
                    0.0044,
                    0.0106,
                    -0.0014,
                    0.0044,
                ]
            ),
            atol=0.01,
        )

        XYZ = np.array([19.01, 20.00, 21.78])
        Y_02 = 31.83
        np.testing.assert_allclose(
            XYZ_to_ATD95(XYZ, XYZ_0, Y_02, K_1, K_2, sigma),
            np.array(
                [
                    0.79,
                    1.091,
                    0.1466,
                    0.146,
                    0.0007,
                    0.013,
                    0.0152,
                    0.0102,
                    0.013,
                ]
            ),
            atol=0.01,
        )

    def test_n_dimensional_XYZ_to_ATD95(self):
        """
        Test :func:`colour.appearance.atd95.XYZ_to_ATD95` definition
        n-dimensional support.
        """

        XYZ = np.array([19.01, 20.00, 21.78])
        XYZ_0 = np.array([95.05, 100.00, 108.88])
        Y_02 = 318.31
        K_1 = 0
        K_2 = 50
        sigma = 300
        specification = XYZ_to_ATD95(XYZ, XYZ_0, Y_02, K_1, K_2, sigma)

        XYZ = np.tile(XYZ, (6, 1))
        specification = np.tile(specification, (6, 1))
        np.testing.assert_allclose(
            XYZ_to_ATD95(XYZ, XYZ_0, Y_02, K_1, K_2, sigma),
            specification,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ_0 = np.tile(XYZ_0, (6, 1))
        np.testing.assert_allclose(
            XYZ_to_ATD95(XYZ, XYZ_0, Y_02, K_1, K_2, sigma),
            specification,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ = np.reshape(XYZ, (2, 3, 3))
        XYZ_0 = np.reshape(XYZ_0, (2, 3, 3))
        specification = np.reshape(specification, (2, 3, 9))
        np.testing.assert_allclose(
            XYZ_to_ATD95(XYZ, XYZ_0, Y_02, K_1, K_2, sigma),
            specification,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_domain_range_scale_XYZ_to_ATD95(self):
        """
        Test :func:`colour.appearance.atd95.XYZ_to_ATD95` definition domain
        and range scale support.
        """

        XYZ = np.array([19.01, 20.00, 21.78])
        XYZ_0 = np.array([95.05, 100.00, 108.88])
        Y_0 = 318.31
        k_1 = 0.0
        k_2 = 50.0
        specification = XYZ_to_ATD95(XYZ, XYZ_0, Y_0, k_1, k_2)

        d_r = (
            ("reference", 1, 1),
            ("1", 0.01, np.array([1 / 360, 1, 1, 1, 1, 1, 1, 1, 1])),
            ("100", 1, np.array([100 / 360, 1, 1, 1, 1, 1, 1, 1, 1])),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    XYZ_to_ATD95(
                        XYZ * factor_a, XYZ_0 * factor_a, Y_0, k_1, k_2
                    ),
                    as_float_array(specification) * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_ATD95(self):
        """
        Test :func:`colour.appearance.atd95.XYZ_to_ATD95` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        XYZ_to_ATD95(cases, cases, cases[..., 0], cases[..., 0], cases[..., 0])


if __name__ == "__main__":
    unittest.main()
