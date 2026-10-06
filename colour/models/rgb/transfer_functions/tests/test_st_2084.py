"""
Define the unit tests for the
:mod:`colour.models.rgb.transfer_functions.st_2084` module.
"""

from __future__ import annotations

import typing

import numpy as np
import pytest

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.transfer_functions import eotf_inverse_ST2084, eotf_ST2084
from colour.utilities import (
    as_ndarray,
    domain_range_scale,
    ignore_numpy_errors,
    xp_as_array,
    xp_assert_close,
    xp_reshape,
)

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestEotf_inverse_ST2084",
    "TestEotf_ST2084",
]


class TestEotf_inverse_ST2084:
    """
    Define :func:`colour.models.rgb.transfer_functions.st_2084.\
eotf_inverse_ST2084` definition unit tests methods.
    """

    def test_eotf_inverse_ST2084(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.st_2084.\
eotf_inverse_ST2084` definition.
        """

        xp_assert_close(
            eotf_inverse_ST2084(xp_as_array(0.0, xp=xp)),
            0.000000730955903,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_inverse_ST2084(xp_as_array(100, xp=xp)),
            0.508078421517399,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_inverse_ST2084(xp_as_array(400, xp=xp)),
            0.652578597563067,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_inverse_ST2084(xp_as_array(5000, xp=xp), 5000),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_eotf_inverse_ST2084(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.st_2084.\
eotf_inverse_ST2084` definition n-dimensional arrays support.
        """

        C = 100
        N = as_ndarray(eotf_inverse_ST2084(xp_as_array(C, xp=xp)))

        C = xp.tile(xp_as_array(C, xp=xp), (6,))
        N = xp.tile(xp_as_array(N, xp=xp), (6,))
        xp_assert_close(eotf_inverse_ST2084(C), N, atol=TOLERANCE_ABSOLUTE_TESTS)

        C = xp_reshape(xp_as_array(C, xp=xp), (2, 3), xp=xp)
        N = xp_reshape(xp_as_array(N, xp=xp), (2, 3), xp=xp)
        xp_assert_close(eotf_inverse_ST2084(C), N, atol=TOLERANCE_ABSOLUTE_TESTS)

        C = xp_reshape(xp_as_array(C, xp=xp), (2, 3, 1), xp=xp)
        N = xp_reshape(xp_as_array(N, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(eotf_inverse_ST2084(C), N, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_eotf_inverse_ST2084(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.st_2084.\
eotf_inverse_ST2084` definition domain and range scale support.
        """

        C = 100
        N = as_ndarray(eotf_inverse_ST2084(xp_as_array(C, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    eotf_inverse_ST2084(xp_as_array(C * factor, xp=xp)),
                    N * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_eotf_inverse_ST2084(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.st_2084.\
eotf_inverse_ST2084` definition nan support.
        """

        eotf_inverse_ST2084(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestEotf_ST2084:
    """
    Define :func:`colour.models.rgb.transfer_functions.st_2084.eotf_ST2084`
    definition unit tests methods.
    """

    @pytest.mark.mps_tolerance_absolute(1e-2)
    def test_eotf_ST2084(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.st_2084.\
eotf_ST2084` definition.
        """

        xp_assert_close(
            eotf_ST2084(xp_as_array(0.0, xp=xp)), 0.0, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        xp_assert_close(
            eotf_ST2084(xp_as_array(0.508078421517399, xp=xp)),
            100,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_ST2084(xp_as_array(0.652578597563067, xp=xp)),
            400,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_ST2084(xp_as_array(1.0, xp=xp), 5000),
            5000.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_eotf_ST2084(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.st_2084.\
eotf_ST2084` definition n-dimensional arrays support.
        """

        N = 0.508078421517399
        C = as_ndarray(eotf_ST2084(xp_as_array(N, xp=xp)))

        N = xp.tile(xp_as_array(N, xp=xp), (6,))
        C = xp.tile(xp_as_array(C, xp=xp), (6,))
        xp_assert_close(eotf_ST2084(N), C, atol=TOLERANCE_ABSOLUTE_TESTS)

        N = xp_reshape(xp_as_array(N, xp=xp), (2, 3), xp=xp)
        C = xp_reshape(xp_as_array(C, xp=xp), (2, 3), xp=xp)
        xp_assert_close(eotf_ST2084(N), C, atol=TOLERANCE_ABSOLUTE_TESTS)

        N = xp_reshape(xp_as_array(N, xp=xp), (2, 3, 1), xp=xp)
        C = xp_reshape(xp_as_array(C, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(eotf_ST2084(N), C, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_eotf_ST2084(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.st_2084.\
eotf_ST2084` definition domain and range scale support.
        """

        N = 0.508078421517399
        C = as_ndarray(eotf_ST2084(xp_as_array(N, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    eotf_ST2084(xp_as_array(N * factor, xp=xp)),
                    C * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_eotf_ST2084(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.st_2084.\
eotf_ST2084` definition nan support.
        """

        eotf_ST2084(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))
