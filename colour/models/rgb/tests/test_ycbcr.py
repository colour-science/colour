# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.models.rgb.ycbcr` module."""

import numpy as np
import unittest
from itertools import product

from colour.models.rgb.ycbcr import (
    round_BT2100,
    ranges_YCbCr,
    matrix_YCbCr,
    offset_YCbCr,
    RGB_to_YCbCr,
    YCbCr_to_RGB,
    RGB_to_YcCbcCrc,
    YcCbcCrc_to_RGB,
    WEIGHTS_YCBCR,
)
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Development"

__all__ = [
    "TestRoundBT2100",
    "TestRangeYCbCr",
    "TestMatrixYCbCr",
    "TestOffsetYCbCr",
    "TestRGB_to_YCbCr",
    "TestYCbCr_to_RGB",
    "TestRGB_to_YcCbcCrc",
    "TestYcCbcCrc_to_RGB",
]


class TestRoundBT2100(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.ycbcr.round_BT2100` definition unit tests
    methods.
    """

    def test_round_BT2100(self):
        """Test :func:`colour.models.rgb.ycbcr.round_BT2100` definition."""

        np.testing.assert_array_equal(
            round_BT2100([-0.6, -0.5, -0.4, 0.4, 0.5, 0.6]),
            np.array([-1.0, -1.0, -0.0, 0.0, 1.0, 1.0]),
        )


class TestRangeYCbCr(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.ycbcr.ranges_YCbCr` definition unit tests
    methods.
    """

    def test_ranges_YCbCr(self):
        """Test :func:`colour.models.rgb.ycbcr.ranges_YCbCr` definition."""

        np.testing.assert_array_almost_equal(
            ranges_YCbCr(8, True, True),
            np.array([16.00000000, 235.00000000, 16.00000000, 240.00000000]),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            ranges_YCbCr(8, True, False),
            np.array([0.06274510, 0.92156863, 0.06274510, 0.94117647]),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            ranges_YCbCr(8, False, True),
            np.array([0.00000000, 255.00000000, 0.50000000, 255.50000000]),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            ranges_YCbCr(8, False, False),
            np.array([0.00000000, 1.00000000, -0.50000000, 0.50000000]),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            ranges_YCbCr(10, True, True),
            np.array([64.00000000, 940.00000000, 64.00000000, 960.00000000]),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            ranges_YCbCr(10, True, False),
            np.array([0.06256109, 0.91886608, 0.06256109, 0.93841642]),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            ranges_YCbCr(10, False, True),
            np.array([0.00000000, 1023.00000000, 0.50000000, 1023.50000000]),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            ranges_YCbCr(10, False, False),
            np.array([0.00000000, 1.00000000, -0.50000000, 0.50000000]),
            decimal=7,
        )


class TestMatrixYCbCr(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.ycbcr.matrix_YCbCr` definition unit tests
    methods.
    """

    def test_matrix_YCbCr(self):
        """Test :func:`colour.models.rgb.ycbcr.matrix_YCbCr` definition."""

        np.testing.assert_array_almost_equal(
            matrix_YCbCr(),
            np.array(
                [
                    [1.00000000, 0.00000000, 1.57480000],
                    [1.00000000, -0.18732427, -0.46812427],
                    [1.00000000, 1.85560000, 0.00000000],
                ]
            ),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            matrix_YCbCr(K=WEIGHTS_YCBCR["ITU-R BT.601"]),
            np.array(
                [
                    [1.00000000, 0.00000000, 1.40200000],
                    [1.00000000, -0.34413629, -0.71413629],
                    [1.00000000, 1.77200000, -0.00000000],
                ]
            ),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            matrix_YCbCr(is_legal=True),
            np.array(
                [
                    [1.16438356, 0.00000000, 1.79274107],
                    [1.16438356, -0.21324861, -0.53290933],
                    [1.16438356, 2.11240179, -0.00000000],
                ]
            ),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            matrix_YCbCr(bits=10),
            np.array(
                [
                    [1.00000000, 0.00000000, 1.57480000],
                    [1.00000000, -0.18732427, -0.46812427],
                    [1.00000000, 1.85560000, 0.00000000],
                ]
            ),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            matrix_YCbCr(bits=10, is_int=True),
            np.array(
                [
                    [0.00097752, 0.00000000, 0.00153939],
                    [0.00097752, -0.00018311, -0.00045760],
                    [0.00097752, 0.00181388, 0.00000000],
                ]
            ),
            decimal=7,
        )


class TestOffsetYCbCr(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.ycbcr.offset_YCbCr` definition unit tests
    methods.
    """

    def test_offset_YCbCr(self):
        """Test :func:`colour.models.rgb.ycbcr.offset_YCbCr` definition."""

        np.testing.assert_array_almost_equal(
            offset_YCbCr(),
            np.array([0.00000000, 0.00000000, 0.00000000]),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            offset_YCbCr(is_legal=True),
            np.array([0.06274510, 0.50196078, 0.50196078]),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            offset_YCbCr(bits=10),
            np.array([0.00000000, 0.00000000, 0.00000000]),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            offset_YCbCr(bits=10, is_int=True),
            np.array([0.00000000, 512.00000000, 512.00000000]),
            decimal=7,
        )


class TestRGB_to_YCbCr(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.ycbcr.RGB_to_YCbCr` definition unit tests
    methods.
    """

    def test_RGB_to_YCbCr(self):
        """Test :func:`colour.models.rgb.ycbcr.RGB_to_YCbCr` definition."""

        np.testing.assert_array_almost_equal(
            RGB_to_YCbCr(np.array([0.75, 0.75, 0.0])),
            np.array([0.66035745, 0.17254902, 0.53216593]),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            RGB_to_YCbCr(
                np.array([0.25, 0.5, 0.75]),
                K=WEIGHTS_YCBCR["ITU-R BT.601"],
                out_int=True,
                out_legal=True,
                out_bits=10,
            ),
            np.array([461, 662, 382]),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            RGB_to_YCbCr(
                np.array([0.0, 0.75, 0.75]),
                K=WEIGHTS_YCBCR["ITU-R BT.2020"],
                out_int=False,
                out_legal=False,
            ),
            np.array([0.55297500, 0.10472255, -0.37500000]),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            RGB_to_YCbCr(
                np.array([0.75, 0.0, 0.75]),
                K=WEIGHTS_YCBCR["ITU-R BT.709"],
                out_range=(16 / 255, 235 / 255, 15.5 / 255, 239.5 / 255),
            ),
            np.array([0.24618980, 0.75392897, 0.79920662]),
            decimal=7,
        )

    def test_n_dimensional_RGB_to_YCbCr(self):
        """
        Test :func:`colour.models.rgb.ycbcr.RGB_to_YCbCr` definition
        n-dimensional arrays support.
        """

        RGB = np.array([0.75, 0.5, 0.25])
        YCbCr = RGB_to_YCbCr(RGB)

        RGB = np.tile(RGB, 4)
        RGB = np.reshape(RGB, (4, 3))
        YCbCr = np.tile(YCbCr, 4)
        YCbCr = np.reshape(YCbCr, (4, 3))
        np.testing.assert_array_almost_equal(RGB_to_YCbCr(RGB), YCbCr)

        RGB = np.tile(RGB, 4)
        RGB = np.reshape(RGB, (4, 4, 3))
        YCbCr = np.tile(YCbCr, 4)
        YCbCr = np.reshape(YCbCr, (4, 4, 3))
        np.testing.assert_array_almost_equal(RGB_to_YCbCr(RGB), YCbCr)

        RGB = np.tile(RGB, 4)
        RGB = np.reshape(RGB, (4, 4, 4, 3))
        YCbCr = np.tile(YCbCr, 4)
        YCbCr = np.reshape(YCbCr, (4, 4, 4, 3))
        np.testing.assert_array_almost_equal(RGB_to_YCbCr(RGB), YCbCr)

    def test_domain_range_scale_RGB_to_YCbCr(self):
        """
        Test :func:`colour.models.rgb.prismatic.RGB_to_YCbCr` definition
        domain and range scale support.
        """

        RGB = np.array([0.75, 0.5, 0.25])
        YCbCr = RGB_to_YCbCr(RGB)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_array_almost_equal(
                    RGB_to_YCbCr(RGB * factor), YCbCr * factor, decimal=7
                )

    @ignore_numpy_errors
    def test_nan_RGB_to_YCbCr(self):
        """
        Test :func:`colour.models.rgb.ycbcr.RGB_to_YCbCr` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        RGB_to_YCbCr(cases)


class TestYCbCr_to_RGB(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.ycbcr.YCbCr_to_RGB` definition unit tests
    methods.
    """

    def test_YCbCr_to_RGB(self):
        """Test :func:`colour.models.rgb.ycbcr.YCbCr_to_RGB` definition."""

        np.testing.assert_array_almost_equal(
            YCbCr_to_RGB(np.array([0.66035745, 0.17254902, 0.53216593])),
            np.array([0.75, 0.75, 0.0]),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            YCbCr_to_RGB(
                np.array([471, 650, 390]),
                in_bits=10,
                in_legal=True,
                in_int=True,
            ),
            np.array([0.25018598, 0.49950072, 0.75040741]),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            YCbCr_to_RGB(
                np.array([150, 99, 175]),
                in_bits=8,
                in_legal=False,
                in_int=True,
                out_bits=8,
                out_legal=True,
                out_int=True,
            ),
            np.array([208, 131, 99]),
            decimal=7,
        )

    def test_n_dimensional_YCbCr_to_RGB(self):
        """
        Test :func:`colour.models.rgb.ycbcr.YCbCr_to_RGB` definition
        n-dimensional arrays support.
        """

        YCbCr = np.array([0.52230157, 0.36699593, 0.62183309])
        RGB = YCbCr_to_RGB(YCbCr)

        RGB = np.tile(RGB, 4)
        RGB = np.reshape(RGB, (4, 3))
        YCbCr = np.tile(YCbCr, 4)
        YCbCr = np.reshape(YCbCr, (4, 3))
        np.testing.assert_array_almost_equal(YCbCr_to_RGB(YCbCr), RGB)

        RGB = np.tile(RGB, 4)
        RGB = np.reshape(RGB, (4, 4, 3))
        YCbCr = np.tile(YCbCr, 4)
        YCbCr = np.reshape(YCbCr, (4, 4, 3))
        np.testing.assert_array_almost_equal(YCbCr_to_RGB(YCbCr), RGB)

        RGB = np.tile(RGB, 4)
        RGB = np.reshape(RGB, (4, 4, 4, 3))
        YCbCr = np.tile(YCbCr, 4)
        YCbCr = np.reshape(YCbCr, (4, 4, 4, 3))
        np.testing.assert_array_almost_equal(YCbCr_to_RGB(YCbCr), RGB)

    def test_domain_range_scale_YCbCr_to_RGB(self):
        """
        Test :func:`colour.models.rgb.prismatic.YCbCr_to_RGB` definition
        domain and range scale support.
        """

        YCbCr = np.array([0.52230157, 0.36699593, 0.62183309])
        RGB = YCbCr_to_RGB(YCbCr)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_array_almost_equal(
                    YCbCr_to_RGB(YCbCr * factor), RGB * factor, decimal=7
                )

    @ignore_numpy_errors
    def test_nan_YCbCr_to_RGB(self):
        """
        Test :func:`colour.models.rgb.ycbcr.YCbCr_to_RGB` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        YCbCr_to_RGB(cases)


class TestRGB_to_YcCbcCrc(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.ycbcr.RGB_to_YcCbcCrc` definition unit
    tests methods.
    """

    def test_RGB_to_YcCbcCrc(self):
        """Test :func:`colour.models.rgb.ycbcr.RGB_to_YcCbcCrc` definition."""

        np.testing.assert_array_almost_equal(
            RGB_to_YcCbcCrc(np.array([0.45620519, 0.03081071, 0.04091952])),
            np.array([0.37020379, 0.41137200, 0.77704674]),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            RGB_to_YcCbcCrc(
                np.array([0.18, 0.18, 0.18]),
                out_bits=10,
                out_legal=True,
                out_int=True,
                is_12_bits_system=False,
            ),
            np.array([422, 512, 512]),
            decimal=7,
        )

    def test_n_dimensional_RGB_to_YcCbcCrc(self):
        """
        Test :func:`colour.models.rgb.ycbcr.RGB_to_YcCbcCrc` definition
        n-dimensional arrays support.
        """

        RGB = np.array([0.45620519, 0.03081071, 0.04091952])
        YcCbcCrc = RGB_to_YcCbcCrc(RGB)

        RGB = np.tile(RGB, 4)
        RGB = np.reshape(RGB, (4, 3))
        YcCbcCrc = np.tile(YcCbcCrc, 4)
        YcCbcCrc = np.reshape(YcCbcCrc, (4, 3))
        np.testing.assert_array_almost_equal(
            RGB_to_YcCbcCrc(RGB), YcCbcCrc, decimal=7
        )

        RGB = np.tile(RGB, 4)
        RGB = np.reshape(RGB, (4, 4, 3))
        YcCbcCrc = np.tile(YcCbcCrc, 4)
        YcCbcCrc = np.reshape(YcCbcCrc, (4, 4, 3))
        np.testing.assert_array_almost_equal(
            RGB_to_YcCbcCrc(RGB), YcCbcCrc, decimal=7
        )

        RGB = np.tile(RGB, 4)
        RGB = np.reshape(RGB, (4, 4, 4, 3))
        YcCbcCrc = np.tile(YcCbcCrc, 4)
        YcCbcCrc = np.reshape(YcCbcCrc, (4, 4, 4, 3))
        np.testing.assert_array_almost_equal(
            RGB_to_YcCbcCrc(RGB), YcCbcCrc, decimal=7
        )

    def test_domain_range_scale_RGB_to_YcCbcCrc(self):
        """
        Test :func:`colour.models.rgb.prismatic.RGB_to_YcCbcCrc` definition
        domain and range scale support.
        """

        RGB = np.array([0.45620519, 0.03081071, 0.04091952])
        YcCbcCrc = RGB_to_YcCbcCrc(RGB)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_array_almost_equal(
                    RGB_to_YcCbcCrc(RGB * factor), YcCbcCrc * factor, decimal=7
                )

    @ignore_numpy_errors
    def test_nan_RGB_to_YcCbcCrc(self):
        """
        Test :func:`colour.models.rgb.ycbcr.RGB_to_YcCbcCrc` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        RGB_to_YcCbcCrc(cases)


class TestYcCbcCrc_to_RGB(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.ycbcr.YCbCr_to_RGB` definition unit tests
    methods.
    """

    def test_YcCbcCrc_to_RGB(self):
        """Test :func:`colour.models.rgb.ycbcr.YCbCr_to_RGB` definition."""

        np.testing.assert_array_almost_equal(
            YcCbcCrc_to_RGB(np.array([0.37020379, 0.41137200, 0.77704674])),
            np.array([0.45620519, 0.03081071, 0.04091952]),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            YcCbcCrc_to_RGB(
                np.array([1689, 2048, 2048]),
                in_bits=12,
                in_legal=True,
                in_int=True,
                is_12_bits_system=True,
            ),
            np.array([0.18009037, 0.18009037, 0.18009037]),
            decimal=7,
        )

    def test_n_dimensional_YcCbcCrc_to_RGB(self):
        """
        Test :func:`colour.models.rgb.ycbcr.YcCbcCrc_to_RGB` definition
        n-dimensional arrays support.
        """

        YcCbcCrc = np.array([0.37020379, 0.41137200, 0.77704674])
        RGB = YcCbcCrc_to_RGB(YcCbcCrc)

        RGB = np.tile(RGB, 4)
        RGB = np.reshape(RGB, (4, 3))
        YcCbcCrc = np.tile(YcCbcCrc, 4)
        YcCbcCrc = np.reshape(YcCbcCrc, (4, 3))
        np.testing.assert_array_almost_equal(
            YcCbcCrc_to_RGB(YcCbcCrc), RGB, decimal=7
        )

        RGB = np.tile(RGB, 4)
        RGB = np.reshape(RGB, (4, 4, 3))
        YcCbcCrc = np.tile(YcCbcCrc, 4)
        YcCbcCrc = np.reshape(YcCbcCrc, (4, 4, 3))
        np.testing.assert_array_almost_equal(
            YcCbcCrc_to_RGB(YcCbcCrc), RGB, decimal=7
        )

        RGB = np.tile(RGB, 4)
        RGB = np.reshape(RGB, (4, 4, 4, 3))
        YcCbcCrc = np.tile(YcCbcCrc, 4)
        YcCbcCrc = np.reshape(YcCbcCrc, (4, 4, 4, 3))
        np.testing.assert_array_almost_equal(
            YcCbcCrc_to_RGB(YcCbcCrc), RGB, decimal=7
        )

    def test_domain_range_scale_YcCbcCrc_to_RGB(self):
        """
        Test :func:`colour.models.rgb.prismatic.YcCbcCrc_to_RGB` definition
        domain and range scale support.
        """

        YcCbcCrc = np.array([0.69943807, 0.38814348, 0.61264549])
        RGB = YcCbcCrc_to_RGB(YcCbcCrc)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_array_almost_equal(
                    YcCbcCrc_to_RGB(YcCbcCrc * factor), RGB * factor, decimal=7
                )

    @ignore_numpy_errors
    def test_nan_YcCbcCrc_to_RGB(self):
        """
        Test :func:`colour.models.rgb.ycbcr.YcCbcCrc_to_RGB` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        YcCbcCrc_to_RGB(cases)


if __name__ == "__main__":
    unittest.main()
