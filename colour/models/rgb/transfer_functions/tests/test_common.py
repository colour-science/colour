"""
Define the unit tests for the
:mod:`colour.models.rgb.transfer_functions.common` module.
"""

from __future__ import annotations

import typing

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.transfer_functions import CV_range, full_to_legal, legal_to_full
from colour.utilities import (
    as_ndarray,
    ignore_numpy_errors,
    xp_as_array,
    xp_assert_close,
    xp_assert_equal,
    xp_reshape,
)

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Development"

__all__ = [
    "TestCV_range",
    "TestLegalToFull",
    "TestFullToLegal",
]


class TestCV_range:
    """
    Define :func:`colour.models.rgb.transfer_functions.common.CV_range`
    definition unit tests methods.
    """

    def test_CV_range(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.common.CV_range`
        definition.
        """

        xp_assert_equal(CV_range(8, True, True), [16, 235])

        xp_assert_equal(CV_range(8, False, True), [0, 255])

        xp_assert_close(
            CV_range(8, True, False),
            [0.06274510, 0.92156863],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_equal(CV_range(8, False, False), [0, 1])

        xp_assert_equal(CV_range(10, True, True), [64, 940])

        xp_assert_equal(CV_range(10, False, True), [0, 1023])

        xp_assert_close(
            CV_range(10, True, False),
            [0.06256109, 0.91886608],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_equal(CV_range(10, False, False), [0, 1])


class TestLegalToFull:
    """
    Define :func:`colour.models.rgb.transfer_functions.common.legal_to_full`
    definition unit tests methods.
    """

    def test_legal_to_full(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.common.legal_to_full`
        definition.
        """

        xp_assert_close(
            legal_to_full(xp_as_array(64 / 1023, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            legal_to_full(xp_as_array(940 / 1023, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            legal_to_full(xp_as_array(64 / 1023, xp=xp), out_int=True),
            0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            legal_to_full(xp_as_array(940 / 1023, xp=xp), out_int=True),
            1023,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            legal_to_full(xp_as_array(64, xp=xp), in_int=True),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            legal_to_full(xp_as_array(940, xp=xp), in_int=True),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            legal_to_full(xp_as_array(64, xp=xp), in_int=True, out_int=True),
            0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            legal_to_full(xp_as_array(940, xp=xp), in_int=True, out_int=True),
            1023,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_legal_to_full(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.common.legal_to_full`
        definition n-dimensional arrays support.
        """

        CV_l = 0.918866080156403
        CV_f = as_ndarray(legal_to_full(xp_as_array(CV_l, xp=xp), 10))

        CV_l = xp.tile(xp_as_array(CV_l, xp=xp), (6,))
        CV_f = xp.tile(xp_as_array(CV_f, xp=xp), (6,))
        xp_assert_close(legal_to_full(CV_l, 10), CV_f, atol=TOLERANCE_ABSOLUTE_TESTS)

        CV_l = xp_reshape(xp_as_array(CV_l, xp=xp), (2, 3), xp=xp)
        CV_f = xp_reshape(xp_as_array(CV_f, xp=xp), (2, 3), xp=xp)
        xp_assert_close(legal_to_full(CV_l, 10), CV_f, atol=TOLERANCE_ABSOLUTE_TESTS)

        CV_l = xp_reshape(xp_as_array(CV_l, xp=xp), (2, 3, 1), xp=xp)
        CV_f = xp_reshape(xp_as_array(CV_f, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(legal_to_full(CV_l, 10), CV_f, atol=TOLERANCE_ABSOLUTE_TESTS)

    @ignore_numpy_errors
    def test_nan_legal_to_full(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.common.legal_to_full`
        definition nan support.
        """

        legal_to_full(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]), 10)


class TestFullToLegal:
    """
    Define :func:`colour.models.rgb.transfer_functions.common.full_to_legal`
    definition unit tests methods.
    """

    def test_full_to_legal(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.common.full_to_legal`
        definition.
        """

        xp_assert_close(
            full_to_legal(xp_as_array(0.0, xp=xp)),
            0.062561094819159,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            full_to_legal(xp_as_array(1.0, xp=xp)),
            0.918866080156403,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            full_to_legal(xp_as_array(0.0, xp=xp), out_int=True),
            64,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            full_to_legal(xp_as_array(1.0, xp=xp), out_int=True),
            940,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            full_to_legal(xp_as_array(0, xp=xp), in_int=True),
            0.062561094819159,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            full_to_legal(xp_as_array(1023, xp=xp), in_int=True),
            0.918866080156403,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            full_to_legal(xp_as_array(0, xp=xp), in_int=True, out_int=True),
            64,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            full_to_legal(xp_as_array(1023, xp=xp), in_int=True, out_int=True),
            940,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_full_to_legal(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.common.full_to_legal`
        definition n-dimensional arrays support.
        """

        CF_f = 1.0
        CV_l = as_ndarray(full_to_legal(xp_as_array(CF_f, xp=xp), 10))

        CF_f = xp.tile(xp_as_array(CF_f, xp=xp), (6,))
        CV_l = xp.tile(xp_as_array(CV_l, xp=xp), (6,))
        xp_assert_close(full_to_legal(CF_f, 10), CV_l, atol=TOLERANCE_ABSOLUTE_TESTS)

        CF_f = xp_reshape(xp_as_array(CF_f, xp=xp), (2, 3), xp=xp)
        CV_l = xp_reshape(xp_as_array(CV_l, xp=xp), (2, 3), xp=xp)
        xp_assert_close(full_to_legal(CF_f, 10), CV_l, atol=TOLERANCE_ABSOLUTE_TESTS)

        CF_f = xp_reshape(xp_as_array(CF_f, xp=xp), (2, 3, 1), xp=xp)
        CV_l = xp_reshape(xp_as_array(CV_l, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(full_to_legal(CF_f, 10), CV_l, atol=TOLERANCE_ABSOLUTE_TESTS)

    @ignore_numpy_errors
    def test_nan_full_to_legal(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.common.full_to_legal`
        definition nan support.
        """

        full_to_legal(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]), 10)
