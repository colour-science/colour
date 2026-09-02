"""Define the unit tests for the :mod:`colour.models.rgb.ycbcr` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.ycbcr import (
    SCALES_YCBCR,
    WEIGHTS_YCBCR,
    RGB_to_YCbCr,
    RGB_to_YcCbcCrc,
    YCbCr_to_RGB,
    YcCbcCrc_to_RGB,
    matrix_YCbCr,
    offset_YCbCr,
    ranges_YCbCr,
    round_BT2100,
)
from colour.utilities import (
    as_ndarray,
    domain_range_scale,
    ignore_numpy_errors,
    xp_as_array,
    xp_assert_close,
    xp_assert_equal,
    xp_reshape,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
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


class TestRoundBT2100:
    """
    Define :func:`colour.models.rgb.ycbcr.round_BT2100` definition unit tests
    methods.
    """

    def test_round_BT2100(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.rgb.ycbcr.round_BT2100` definition."""

        xp_assert_equal(
            round_BT2100(xp_as_array([-0.6, -0.5, -0.4, 0.4, 0.5, 0.6], xp=xp)),
            [-1.0, -1.0, -0.0, 0.0, 1.0, 1.0],
        )


class TestRangeYCbCr:
    """
    Define :func:`colour.models.rgb.ycbcr.ranges_YCbCr` definition unit tests
    methods.
    """

    def test_ranges_YCbCr(self) -> None:
        """Test :func:`colour.models.rgb.ycbcr.ranges_YCbCr` definition."""

        xp_assert_close(
            ranges_YCbCr(8, True, True),
            [16.00000000, 235.00000000, 16.00000000, 240.00000000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ranges_YCbCr(8, True, False),
            [0.06274510, 0.92156863, 0.06274510, 0.94117647],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ranges_YCbCr(8, False, True),
            [0.00000000, 255.00000000, 0.50000000, 255.50000000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ranges_YCbCr(8, False, False),
            [0.00000000, 1.00000000, -0.50000000, 0.50000000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ranges_YCbCr(10, True, True),
            [64.00000000, 940.00000000, 64.00000000, 960.00000000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ranges_YCbCr(10, True, False),
            [0.06256109, 0.91886608, 0.06256109, 0.93841642],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ranges_YCbCr(10, False, True),
            [0.00000000, 1023.00000000, 0.50000000, 1023.50000000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ranges_YCbCr(10, False, False),
            [0.00000000, 1.00000000, -0.50000000, 0.50000000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestMatrixYCbCr:
    """
    Define :func:`colour.models.rgb.ycbcr.matrix_YCbCr` definition unit tests
    methods.
    """

    def test_matrix_YCbCr(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.rgb.ycbcr.matrix_YCbCr` definition."""

        xp_assert_close(
            matrix_YCbCr(K=xp_as_array(WEIGHTS_YCBCR["ITU-R BT.709"], xp=xp)),
            [
                [1.00000000, 0.00000000, 1.57480000],
                [1.00000000, -0.18732427, -0.46812427],
                [1.00000000, 1.85560000, 0.00000000],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            matrix_YCbCr(K=xp_as_array(WEIGHTS_YCBCR["ITU-R BT.601"], xp=xp)),
            [
                [1.00000000, 0.00000000, 1.40200000],
                [1.00000000, -0.34413629, -0.71413629],
                [1.00000000, 1.77200000, -0.00000000],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            matrix_YCbCr(
                K=xp_as_array(WEIGHTS_YCBCR["ITU-R BT.709"], xp=xp), is_legal=True
            ),
            [
                [1.16438356, 0.00000000, 1.79274107],
                [1.16438356, -0.21324861, -0.53290933],
                [1.16438356, 2.11240179, -0.00000000],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            matrix_YCbCr(K=xp_as_array(WEIGHTS_YCBCR["ITU-R BT.709"], xp=xp), bits=10),
            [
                [1.00000000, 0.00000000, 1.57480000],
                [1.00000000, -0.18732427, -0.46812427],
                [1.00000000, 1.85560000, 0.00000000],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            matrix_YCbCr(
                K=xp_as_array(WEIGHTS_YCBCR["ITU-R BT.709"], xp=xp),
                bits=10,
                is_int=True,
            ),
            [
                [0.00097752, 0.00000000, 0.00153939],
                [0.00097752, -0.00018311, -0.00045760],
                [0.00097752, 0.00181388, 0.00000000],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            matrix_YCbCr(
                K=xp_as_array(WEIGHTS_YCBCR["ITU-R BT.709"], xp=xp),
                S=xp_as_array(SCALES_YCBCR["Y'UV"], xp=xp),
            ),
            [
                [1.00000000, 0.00000000, 1.28032520],
                [1.00000000, -0.21482141, -0.38058884],
                [1.00000000, 2.12798165, 0.00000000],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestOffsetYCbCr:
    """
    Define :func:`colour.models.rgb.ycbcr.offset_YCbCr` definition unit tests
    methods.
    """

    def test_offset_YCbCr(self) -> None:
        """Test :func:`colour.models.rgb.ycbcr.offset_YCbCr` definition."""

        xp_assert_close(
            offset_YCbCr(),
            [0.00000000, 0.00000000, 0.00000000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            offset_YCbCr(is_legal=True),
            [0.06274510, 0.50196078, 0.50196078],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            offset_YCbCr(bits=10),
            [0.00000000, 0.00000000, 0.00000000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            offset_YCbCr(bits=10, is_int=True),
            [0.00000000, 512.00000000, 512.00000000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestRGB_to_YCbCr:
    """
    Define :func:`colour.models.rgb.ycbcr.RGB_to_YCbCr` definition unit tests
    methods.
    """

    def test_RGB_to_YCbCr(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.rgb.ycbcr.RGB_to_YCbCr` definition."""

        xp_assert_close(
            RGB_to_YCbCr(xp_as_array([0.75, 0.75, 0.0], xp=xp)),
            [0.66035745, 0.17254902, 0.53216593],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            RGB_to_YCbCr(
                xp_as_array([0.25, 0.5, 0.75], xp=xp),
                K=WEIGHTS_YCBCR["ITU-R BT.601"],
                out_int=True,
                out_legal=True,
                out_bits=10,
            ),
            [461, 662, 382],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            RGB_to_YCbCr(
                xp_as_array([0.0, 0.75, 0.75], xp=xp),
                K=WEIGHTS_YCBCR["ITU-R BT.2020"],
                out_int=False,
                out_legal=False,
            ),
            [0.55297500, 0.10472255, -0.37500000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            RGB_to_YCbCr(
                xp_as_array([0.75, 0.0, 0.75], xp=xp),
                K=WEIGHTS_YCBCR["ITU-R BT.709"],
                out_range=(16 / 255, 235 / 255, 15.5 / 255, 239.5 / 255),
            ),
            [0.24618980, 0.75392897, 0.79920662],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            RGB_to_YCbCr(
                xp_as_array([0.75, 0.5, 0.25], xp=xp),
                S=SCALES_YCBCR["Y'UV"],
                out_legal=False,
                out_int=False,
            ),
            [0.53510000, -0.13397672, 0.16784798],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_RGB_to_YCbCr(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.ycbcr.RGB_to_YCbCr` definition
        n-dimensional arrays support.
        """

        RGB = xp_as_array([0.75, 0.5, 0.25], xp=xp)
        YCbCr = as_ndarray(RGB_to_YCbCr(RGB))

        RGB = xp.tile(xp_as_array(RGB, xp=xp), (4,))
        RGB = xp_reshape(xp_as_array(RGB, xp=xp), (4, 3), xp=xp)
        YCbCr = xp.tile(xp_as_array(YCbCr, xp=xp), (4,))
        YCbCr = xp_reshape(xp_as_array(YCbCr, xp=xp), (4, 3), xp=xp)
        xp_assert_close(
            RGB_to_YCbCr(RGB),
            YCbCr,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        RGB = xp.tile(xp_as_array(RGB, xp=xp), (4,))
        RGB = xp_reshape(xp_as_array(RGB, xp=xp), (4, 4, 3), xp=xp)
        YCbCr = xp.tile(xp_as_array(YCbCr, xp=xp), (4,))
        YCbCr = xp_reshape(xp_as_array(YCbCr, xp=xp), (4, 4, 3), xp=xp)
        xp_assert_close(
            RGB_to_YCbCr(RGB),
            YCbCr,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        RGB = xp.tile(xp_as_array(RGB, xp=xp), (4,))
        RGB = xp_reshape(xp_as_array(RGB, xp=xp), (4, 4, 4, 3), xp=xp)
        YCbCr = xp.tile(xp_as_array(YCbCr, xp=xp), (4,))
        YCbCr = xp_reshape(xp_as_array(YCbCr, xp=xp), (4, 4, 4, 3), xp=xp)
        xp_assert_close(
            RGB_to_YCbCr(RGB),
            YCbCr,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_RGB_to_YCbCr(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.prismatic.RGB_to_YCbCr` definition
        domain and range scale support.
        """

        RGB = xp_as_array([0.75, 0.5, 0.25], xp=xp)
        YCbCr = as_ndarray(RGB_to_YCbCr(RGB))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    RGB_to_YCbCr(RGB * factor),
                    YCbCr * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_RGB_to_YCbCr(self) -> None:
        """
        Test :func:`colour.models.rgb.ycbcr.RGB_to_YCbCr` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        RGB_to_YCbCr(cases)


class TestYCbCr_to_RGB:
    """
    Define :func:`colour.models.rgb.ycbcr.YCbCr_to_RGB` definition unit tests
    methods.
    """

    def test_YCbCr_to_RGB(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.rgb.ycbcr.YCbCr_to_RGB` definition."""

        xp_assert_close(
            YCbCr_to_RGB(xp_as_array([0.66035745, 0.17254902, 0.53216593], xp=xp)),
            [0.75, 0.75, 0.0],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            YCbCr_to_RGB(
                xp_as_array([471, 650, 390], xp=xp),
                in_bits=10,
                in_legal=True,
                in_int=True,
            ),
            [0.25018598, 0.49950072, 0.75040741],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            YCbCr_to_RGB(
                xp_as_array([150, 99, 175], xp=xp),
                in_bits=8,
                in_legal=False,
                in_int=True,
                out_bits=8,
                out_legal=True,
                out_int=True,
            ),
            [208, 131, 99],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            YCbCr_to_RGB(
                xp_as_array([0.53510000, -0.13397672, 0.16784798], xp=xp),
                S=SCALES_YCBCR["Y'UV"],
                in_legal=False,
                in_int=False,
            ),
            [0.75, 0.5, 0.25],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_roundtrip_YCbCr_YUV(self, xp: ModuleType) -> None:
        """Test *Y'UV* roundtrip with :func:`colour.models.rgb.ycbcr.RGB_to_YCbCr`
        and :func:`colour.models.rgb.ycbcr.YCbCr_to_RGB` definitions.
        """

        RGB = xp_as_array([0.75, 0.5, 0.25], xp=xp)
        YUV = RGB_to_YCbCr(RGB, S=SCALES_YCBCR["Y'UV"], out_legal=False, out_int=False)
        xp_assert_close(
            YCbCr_to_RGB(YUV, S=SCALES_YCBCR["Y'UV"], in_legal=False, in_int=False),
            [0.75, 0.5, 0.25],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_YCbCr_to_RGB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.ycbcr.YCbCr_to_RGB` definition
        n-dimensional arrays support.
        """

        YCbCr = xp_as_array([0.52230157, 0.36699593, 0.62183309], xp=xp)
        RGB = as_ndarray(YCbCr_to_RGB(YCbCr))

        RGB = xp.tile(xp_as_array(RGB, xp=xp), (4,))
        RGB = xp_reshape(xp_as_array(RGB, xp=xp), (4, 3), xp=xp)
        YCbCr = xp.tile(xp_as_array(YCbCr, xp=xp), (4,))
        YCbCr = xp_reshape(xp_as_array(YCbCr, xp=xp), (4, 3), xp=xp)
        xp_assert_close(
            YCbCr_to_RGB(YCbCr),
            RGB,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        RGB = xp.tile(xp_as_array(RGB, xp=xp), (4,))
        RGB = xp_reshape(xp_as_array(RGB, xp=xp), (4, 4, 3), xp=xp)
        YCbCr = xp.tile(xp_as_array(YCbCr, xp=xp), (4,))
        YCbCr = xp_reshape(xp_as_array(YCbCr, xp=xp), (4, 4, 3), xp=xp)
        xp_assert_close(
            YCbCr_to_RGB(YCbCr),
            RGB,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        RGB = xp.tile(xp_as_array(RGB, xp=xp), (4,))
        RGB = xp_reshape(xp_as_array(RGB, xp=xp), (4, 4, 4, 3), xp=xp)
        YCbCr = xp.tile(xp_as_array(YCbCr, xp=xp), (4,))
        YCbCr = xp_reshape(xp_as_array(YCbCr, xp=xp), (4, 4, 4, 3), xp=xp)
        xp_assert_close(
            YCbCr_to_RGB(YCbCr),
            RGB,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_YCbCr_to_RGB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.prismatic.YCbCr_to_RGB` definition
        domain and range scale support.
        """

        YCbCr = xp_as_array([0.52230157, 0.36699593, 0.62183309], xp=xp)
        RGB = as_ndarray(YCbCr_to_RGB(YCbCr))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    YCbCr_to_RGB(YCbCr * factor),
                    RGB * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_YCbCr_to_RGB(self) -> None:
        """
        Test :func:`colour.models.rgb.ycbcr.YCbCr_to_RGB` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        YCbCr_to_RGB(cases)


class TestRGB_to_YcCbcCrc:
    """
    Define :func:`colour.models.rgb.ycbcr.RGB_to_YcCbcCrc` definition unit
    tests methods.
    """

    def test_RGB_to_YcCbcCrc(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.rgb.ycbcr.RGB_to_YcCbcCrc` definition."""

        xp_assert_close(
            RGB_to_YcCbcCrc(xp_as_array([0.45620519, 0.03081071, 0.04091952], xp=xp)),
            [0.37020379, 0.41137200, 0.77704674],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            RGB_to_YcCbcCrc(
                xp_as_array([0.18, 0.18, 0.18], xp=xp),
                out_bits=10,
                out_legal=True,
                out_int=True,
                is_12_bits_system=False,
            ),
            [422, 512, 512],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_RGB_to_YcCbcCrc(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.ycbcr.RGB_to_YcCbcCrc` definition
        n-dimensional arrays support.
        """

        RGB = xp_as_array([0.45620519, 0.03081071, 0.04091952], xp=xp)
        YcCbcCrc = as_ndarray(RGB_to_YcCbcCrc(RGB))

        RGB = xp.tile(xp_as_array(RGB, xp=xp), (4,))
        RGB = xp_reshape(xp_as_array(RGB, xp=xp), (4, 3), xp=xp)
        YcCbcCrc = xp.tile(xp_as_array(YcCbcCrc, xp=xp), (4,))
        YcCbcCrc = xp_reshape(xp_as_array(YcCbcCrc, xp=xp), (4, 3), xp=xp)
        xp_assert_close(RGB_to_YcCbcCrc(RGB), YcCbcCrc, atol=TOLERANCE_ABSOLUTE_TESTS)

        RGB = xp.tile(xp_as_array(RGB, xp=xp), (4,))
        RGB = xp_reshape(xp_as_array(RGB, xp=xp), (4, 4, 3), xp=xp)
        YcCbcCrc = xp.tile(xp_as_array(YcCbcCrc, xp=xp), (4,))
        YcCbcCrc = xp_reshape(xp_as_array(YcCbcCrc, xp=xp), (4, 4, 3), xp=xp)
        xp_assert_close(RGB_to_YcCbcCrc(RGB), YcCbcCrc, atol=TOLERANCE_ABSOLUTE_TESTS)

        RGB = xp.tile(xp_as_array(RGB, xp=xp), (4,))
        RGB = xp_reshape(xp_as_array(RGB, xp=xp), (4, 4, 4, 3), xp=xp)
        YcCbcCrc = xp.tile(xp_as_array(YcCbcCrc, xp=xp), (4,))
        YcCbcCrc = xp_reshape(xp_as_array(YcCbcCrc, xp=xp), (4, 4, 4, 3), xp=xp)
        xp_assert_close(RGB_to_YcCbcCrc(RGB), YcCbcCrc, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_RGB_to_YcCbcCrc(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.prismatic.RGB_to_YcCbcCrc` definition
        domain and range scale support.
        """

        RGB = xp_as_array([0.45620519, 0.03081071, 0.04091952], xp=xp)
        YcCbcCrc = as_ndarray(RGB_to_YcCbcCrc(RGB))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    RGB_to_YcCbcCrc(RGB * factor),
                    YcCbcCrc * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_RGB_to_YcCbcCrc(self) -> None:
        """
        Test :func:`colour.models.rgb.ycbcr.RGB_to_YcCbcCrc` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        RGB_to_YcCbcCrc(cases)


class TestYcCbcCrc_to_RGB:
    """
    Define :func:`colour.models.rgb.ycbcr.YCbCr_to_RGB` definition unit tests
    methods.
    """

    def test_YcCbcCrc_to_RGB(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.rgb.ycbcr.YCbCr_to_RGB` definition."""

        xp_assert_close(
            YcCbcCrc_to_RGB(xp_as_array([0.37020379, 0.41137200, 0.77704674], xp=xp)),
            [0.45620519, 0.03081071, 0.04091952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            YcCbcCrc_to_RGB(
                xp_as_array([1689, 2048, 2048], xp=xp),
                in_bits=12,
                in_legal=True,
                in_int=True,
                is_12_bits_system=True,
            ),
            [0.18009037, 0.18009037, 0.18009037],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_YcCbcCrc_to_RGB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.ycbcr.YcCbcCrc_to_RGB` definition
        n-dimensional arrays support.
        """

        YcCbcCrc = xp_as_array([0.37020379, 0.41137200, 0.77704674], xp=xp)
        RGB = as_ndarray(YcCbcCrc_to_RGB(YcCbcCrc))

        RGB = xp.tile(xp_as_array(RGB, xp=xp), (4,))
        RGB = xp_reshape(xp_as_array(RGB, xp=xp), (4, 3), xp=xp)
        YcCbcCrc = xp.tile(xp_as_array(YcCbcCrc, xp=xp), (4,))
        YcCbcCrc = xp_reshape(xp_as_array(YcCbcCrc, xp=xp), (4, 3), xp=xp)
        xp_assert_close(YcCbcCrc_to_RGB(YcCbcCrc), RGB, atol=TOLERANCE_ABSOLUTE_TESTS)

        RGB = xp.tile(xp_as_array(RGB, xp=xp), (4,))
        RGB = xp_reshape(xp_as_array(RGB, xp=xp), (4, 4, 3), xp=xp)
        YcCbcCrc = xp.tile(xp_as_array(YcCbcCrc, xp=xp), (4,))
        YcCbcCrc = xp_reshape(xp_as_array(YcCbcCrc, xp=xp), (4, 4, 3), xp=xp)
        xp_assert_close(YcCbcCrc_to_RGB(YcCbcCrc), RGB, atol=TOLERANCE_ABSOLUTE_TESTS)

        RGB = xp.tile(xp_as_array(RGB, xp=xp), (4,))
        RGB = xp_reshape(xp_as_array(RGB, xp=xp), (4, 4, 4, 3), xp=xp)
        YcCbcCrc = xp.tile(xp_as_array(YcCbcCrc, xp=xp), (4,))
        YcCbcCrc = xp_reshape(xp_as_array(YcCbcCrc, xp=xp), (4, 4, 4, 3), xp=xp)
        xp_assert_close(YcCbcCrc_to_RGB(YcCbcCrc), RGB, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_YcCbcCrc_to_RGB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.prismatic.YcCbcCrc_to_RGB` definition
        domain and range scale support.
        """

        YcCbcCrc = xp_as_array([0.69943807, 0.38814348, 0.61264549], xp=xp)
        RGB = as_ndarray(YcCbcCrc_to_RGB(YcCbcCrc))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    YcCbcCrc_to_RGB(YcCbcCrc * factor),
                    RGB * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_YcCbcCrc_to_RGB(self) -> None:
        """
        Test :func:`colour.models.rgb.ycbcr.YcCbcCrc_to_RGB` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        YcCbcCrc_to_RGB(cases)
