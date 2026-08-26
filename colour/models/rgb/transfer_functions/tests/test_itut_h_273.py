"""
Define the unit tests for the
:mod:`colour.models.rgb.transfer_functions.itut_h_273` module.
"""

from __future__ import annotations

import typing

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.transfer_functions import (
    eotf_H273_ST428_1,
    eotf_inverse_H273_ST428_1,
    oetf_H273_IEC61966_2,
    oetf_H273_Log,
    oetf_H273_LogSqrt,
    oetf_inverse_H273_IEC61966_2,
    oetf_inverse_H273_Log,
    oetf_inverse_H273_LogSqrt,
)
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
    "TestOetf_H273_Log",
    "TestOetf_inverse_H273_Log",
    "TestOetf_H273_LogSqrt",
    "TestOetf_inverse_H273_LogSqrt",
    "TestOetf_H273_IEC61966_2",
    "TestOetf_inverse_H273_IEC61966_2",
    "TestEotf_inverse_H273_ST428_1",
    "TestEotf_H273_ST428_1",
]


class TestOetf_H273_Log:
    """
        Define :func:`colour.models.rgb.transfer_functions.itut_h_273.
    oetf_H273_Log` definition unit tests methods.
    """

    def test_oetf_H273_Log(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_H273_Log` definition.
        """

        xp_assert_close(
            oetf_H273_Log(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_H273_Log(xp_as_array(0.18, xp=xp)),
            0.627636252551653,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_H273_Log(xp_as_array(1.0, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_oetf_H273_Log(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_H273_Log` definition n-dimensional arrays support.
        """

        E = 0.18
        E_p = as_ndarray(oetf_H273_Log(xp_as_array(E, xp=xp)))

        E = xp.tile(xp_as_array(E, xp=xp), (6,))
        E_p = xp.tile(xp_as_array(E_p, xp=xp), (6,))
        xp_assert_close(oetf_H273_Log(E), E_p, atol=TOLERANCE_ABSOLUTE_TESTS)

        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3), xp=xp)
        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3), xp=xp)
        xp_assert_close(oetf_H273_Log(E), E_p, atol=TOLERANCE_ABSOLUTE_TESTS)

        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3, 1), xp=xp)
        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(oetf_H273_Log(E), E_p, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_oetf_H273_Log(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_H273_Log` definition domain and range scale support.
        """

        E = 0.18
        E_p = as_ndarray(oetf_H273_Log(xp_as_array(E, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    oetf_H273_Log(xp_as_array(E * factor, xp=xp)),
                    E_p * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_oetf_H273_Log(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_H273_Log` definition nan support.
        """

        oetf_H273_Log(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestOetf_inverse_H273_Log:
    """
    Define :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_inverse_H273_Log` definition unit tests methods.
    """

    def test_oetf_inverse_H273_Log(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_inverse_H273_Log` definition.
        """

        # NOTE: The function is unfortunately clamped and cannot roundtrip
        # properly.
        xp_assert_close(
            oetf_inverse_H273_Log(xp_as_array(0.0, xp=xp)),
            0.01,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_inverse_H273_Log(xp_as_array(0.627636252551653, xp=xp)),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_inverse_H273_Log(xp_as_array(1.0, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_oetf_inverse_H273_Log(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_inverse_H273_Log` definition n-dimensional arrays support.
        """

        E_p = 0.627636252551653
        E = as_ndarray(oetf_inverse_H273_Log(xp_as_array(E_p, xp=xp)))

        E_p = xp.tile(xp_as_array(E_p, xp=xp), (6,))
        E = xp.tile(xp_as_array(E, xp=xp), (6,))
        xp_assert_close(oetf_inverse_H273_Log(E_p), E, atol=TOLERANCE_ABSOLUTE_TESTS)

        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3), xp=xp)
        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3), xp=xp)
        xp_assert_close(oetf_inverse_H273_Log(E_p), E, atol=TOLERANCE_ABSOLUTE_TESTS)

        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3, 1), xp=xp)
        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(oetf_inverse_H273_Log(E_p), E, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_oetf_inverse_H273_Log(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_inverse_H273_Log` definition domain and range scale support.
        """

        E_p = 0.627636252551653
        E = as_ndarray(oetf_inverse_H273_Log(xp_as_array(E_p, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    oetf_inverse_H273_Log(xp_as_array(E_p * factor, xp=xp)),
                    E * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_oetf_inverse_H273_Log(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_inverse_H273_Log` definition nan support.
        """

        oetf_inverse_H273_Log(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestOetf_H273_LogSqrt:
    """
        Define :func:`colour.models.rgb.transfer_functions.itut_h_273.
    oetf_H273_LogSqrt` definition unit tests methods.
    """

    def test_oetf_H273_LogSqrt(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_H273_LogSqrt` definition.
        """

        xp_assert_close(
            oetf_H273_LogSqrt(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_H273_LogSqrt(xp_as_array(0.18, xp=xp)),
            0.702109002041322,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_H273_LogSqrt(xp_as_array(1.0, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_oetf_H273_LogSqrt(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_H273_LogSqrt` definition n-dimensional arrays support.
        """

        E = 0.18
        E_p = as_ndarray(oetf_H273_LogSqrt(xp_as_array(E, xp=xp)))

        E = xp.tile(xp_as_array(E, xp=xp), (6,))
        E_p = xp.tile(xp_as_array(E_p, xp=xp), (6,))
        xp_assert_close(oetf_H273_LogSqrt(E), E_p, atol=TOLERANCE_ABSOLUTE_TESTS)

        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3), xp=xp)
        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3), xp=xp)
        xp_assert_close(oetf_H273_LogSqrt(E), E_p, atol=TOLERANCE_ABSOLUTE_TESTS)

        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3, 1), xp=xp)
        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(oetf_H273_LogSqrt(E), E_p, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_oetf_H273_LogSqrt(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_H273_LogSqrt` definition domain and range scale support.
        """

        E = 0.18
        E_p = as_ndarray(oetf_H273_LogSqrt(xp_as_array(E, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    oetf_H273_LogSqrt(xp_as_array(E * factor, xp=xp)),
                    E_p * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_oetf_H273_LogSqrt(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_H273_LogSqrt` definition nan support.
        """

        oetf_H273_LogSqrt(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestOetf_inverse_H273_LogSqrt:
    """
    Define :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_inverse_H273_LogSqrt` definition unit tests methods.
    """

    def test_oetf_inverse_H273_LogSqrt(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_inverse_H273_LogSqrt` definition.
        """

        # NOTE: The function is unfortunately clamped and cannot roundtrip
        # properly.
        xp_assert_close(
            oetf_inverse_H273_LogSqrt(xp_as_array(0.0, xp=xp)),
            0.003162277660168,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_inverse_H273_LogSqrt(xp_as_array(0.702109002041322, xp=xp)),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_inverse_H273_LogSqrt(xp_as_array(1.0, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_oetf_inverse_H273_LogSqrt(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_inverse_H273_LogSqrt` definition n-dimensional arrays support.
        """

        E_p = 0.702109002041322
        E = as_ndarray(oetf_inverse_H273_LogSqrt(xp_as_array(E_p, xp=xp)))

        E_p = xp.tile(xp_as_array(E_p, xp=xp), (6,))
        E = xp.tile(xp_as_array(E, xp=xp), (6,))
        xp_assert_close(
            oetf_inverse_H273_LogSqrt(E_p), E, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3), xp=xp)
        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            oetf_inverse_H273_LogSqrt(E_p), E, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3, 1), xp=xp)
        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            oetf_inverse_H273_LogSqrt(E_p), E, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_oetf_inverse_H273_LogSqrt(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_inverse_H273_LogSqrt` definition domain and range scale support.
        """

        E_p = 0.702109002041322
        E = as_ndarray(oetf_inverse_H273_LogSqrt(xp_as_array(E_p, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    oetf_inverse_H273_LogSqrt(xp_as_array(E_p * factor, xp=xp)),
                    E * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_oetf_inverse_H273_LogSqrt(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_inverse_H273_LogSqrt` definition nan support.
        """

        oetf_inverse_H273_LogSqrt(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestOetf_H273_IEC61966_2:
    """
        Define :func:`colour.models.rgb.transfer_functions.itut_h_273.
    oetf_H273_IEC61966_2` definition unit tests methods.
    """

    def test_oetf_H273_IEC61966_2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_H273_IEC61966_2` definition.
        """

        xp_assert_close(
            oetf_H273_IEC61966_2(xp_as_array(-0.18, xp=xp)),
            -0.461356129500442,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_H273_IEC61966_2(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_H273_IEC61966_2(xp_as_array(0.18, xp=xp)),
            0.461356129500442,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_H273_IEC61966_2(xp_as_array(1.0, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_oetf_H273_IEC61966_2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_H273_IEC61966_2` definition n-dimensional arrays support.
        """

        E = 0.18
        E_p = as_ndarray(oetf_H273_IEC61966_2(xp_as_array(E, xp=xp)))

        E = xp.tile(xp_as_array(E, xp=xp), (6,))
        E_p = xp.tile(xp_as_array(E_p, xp=xp), (6,))
        xp_assert_close(oetf_H273_IEC61966_2(E), E_p, atol=TOLERANCE_ABSOLUTE_TESTS)

        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3), xp=xp)
        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3), xp=xp)
        xp_assert_close(oetf_H273_IEC61966_2(E), E_p, atol=TOLERANCE_ABSOLUTE_TESTS)

        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3, 1), xp=xp)
        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(oetf_H273_IEC61966_2(E), E_p, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_oetf_H273_IEC61966_2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_H273_IEC61966_2` definition domain and range scale support.
        """

        E = 0.18
        E_p = as_ndarray(oetf_H273_IEC61966_2(xp_as_array(E, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    oetf_H273_IEC61966_2(xp_as_array(E * factor, xp=xp)),
                    E_p * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_oetf_H273_IEC61966_2(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_H273_IEC61966_2` definition nan support.
        """

        oetf_H273_IEC61966_2(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestOetf_inverse_H273_IEC61966_2:
    """
    Define :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_inverse_H273_IEC61966_2` definition unit tests methods.
    """

    def test_oetf_inverse_H273_IEC61966_2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_inverse_H273_IEC61966_2` definition.
        """

        xp_assert_close(
            oetf_inverse_H273_IEC61966_2(xp_as_array(-0.461356129500442, xp=xp)),
            -0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_inverse_H273_IEC61966_2(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_inverse_H273_IEC61966_2(xp_as_array(0.461356129500442, xp=xp)),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_inverse_H273_IEC61966_2(xp_as_array(1.0, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_oetf_inverse_H273_IEC61966_2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_inverse_H273_IEC61966_2` definition n-dimensional arrays support.
        """

        E_p = 0.627636252551653
        E = as_ndarray(oetf_inverse_H273_IEC61966_2(xp_as_array(E_p, xp=xp)))

        E_p = xp.tile(xp_as_array(E_p, xp=xp), (6,))
        E = xp.tile(xp_as_array(E, xp=xp), (6,))
        xp_assert_close(
            oetf_inverse_H273_IEC61966_2(E_p),
            E,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3), xp=xp)
        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            oetf_inverse_H273_IEC61966_2(E_p),
            E,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3, 1), xp=xp)
        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            oetf_inverse_H273_IEC61966_2(E_p),
            E,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_oetf_inverse_H273_IEC61966_2(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_inverse_H273_IEC61966_2` definition domain and range scale support.
        """

        E_p = 0.627636252551653
        E = as_ndarray(oetf_inverse_H273_IEC61966_2(xp_as_array(E_p, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    oetf_inverse_H273_IEC61966_2(xp_as_array(E_p * factor, xp=xp)),
                    E * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_oetf_inverse_H273_IEC61966_2(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
oetf_inverse_H273_IEC61966_2` definition nan support.
        """

        oetf_inverse_H273_IEC61966_2(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestEotf_inverse_H273_ST428_1:
    """
        Define :func:`colour.models.rgb.transfer_functions.itut_h_273.
    eotf_inverse_H273_ST428_1` definition unit tests methods.
    """

    def test_eotf_inverse_H273_ST428_1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
eotf_inverse_H273_ST428_1` definition.
        """

        xp_assert_close(
            eotf_inverse_H273_ST428_1(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_inverse_H273_ST428_1(xp_as_array(0.18, xp=xp)),
            0.500048337717236,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_inverse_H273_ST428_1(xp_as_array(1.0, xp=xp)),
            0.967042675317934,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_eotf_inverse_H273_ST428_1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
eotf_inverse_H273_ST428_1` definition n-dimensional arrays support.
        """

        E = 0.18
        E_p = as_ndarray(eotf_inverse_H273_ST428_1(xp_as_array(E, xp=xp)))

        E = xp.tile(xp_as_array(E, xp=xp), (6,))
        E_p = xp.tile(xp_as_array(E_p, xp=xp), (6,))
        xp_assert_close(
            eotf_inverse_H273_ST428_1(E), E_p, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3), xp=xp)
        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            eotf_inverse_H273_ST428_1(E), E_p, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3, 1), xp=xp)
        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            eotf_inverse_H273_ST428_1(E), E_p, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_eotf_inverse_H273_ST428_1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
eotf_inverse_H273_ST428_1` definition domain and range scale support.
        """

        E = 0.18
        E_p = as_ndarray(eotf_inverse_H273_ST428_1(xp_as_array(E, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    eotf_inverse_H273_ST428_1(xp_as_array(E * factor, xp=xp)),
                    E_p * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_eotf_inverse_H273_ST428_1(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
eotf_inverse_H273_ST428_1` definition nan support.
        """

        eotf_inverse_H273_ST428_1(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestEotf_H273_ST428_1:
    """
    Define :func:`colour.models.rgb.transfer_functions.itut_h_273.\
eotf_H273_ST428_1` definition unit tests methods.
    """

    def test_eotf_H273_ST428_1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
eotf_H273_ST428_1` definition.
        """

        xp_assert_close(
            eotf_H273_ST428_1(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_H273_ST428_1(xp_as_array(0.500048337717236, xp=xp)),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_H273_ST428_1(xp_as_array(0.967042675317934, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_eotf_H273_ST428_1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
eotf_H273_ST428_1` definition n-dimensional arrays support.
        """

        E_p = 0.500048337717236
        E = as_ndarray(eotf_H273_ST428_1(xp_as_array(E_p, xp=xp)))

        E_p = xp.tile(xp_as_array(E_p, xp=xp), (6,))
        E = xp.tile(xp_as_array(E, xp=xp), (6,))
        xp_assert_close(eotf_H273_ST428_1(E_p), E, atol=TOLERANCE_ABSOLUTE_TESTS)

        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3), xp=xp)
        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3), xp=xp)
        xp_assert_close(eotf_H273_ST428_1(E_p), E, atol=TOLERANCE_ABSOLUTE_TESTS)

        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3, 1), xp=xp)
        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(eotf_H273_ST428_1(E_p), E, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_eotf_H273_ST428_1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
eotf_H273_ST428_1` definition domain and range scale support.
        """

        E_p = 0.500048337717236
        E = as_ndarray(eotf_H273_ST428_1(xp_as_array(E_p, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    eotf_H273_ST428_1(xp_as_array(E_p * factor, xp=xp)),
                    E * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_eotf_H273_ST428_1(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itut_h_273.\
eotf_H273_ST428_1` definition nan support.
        """

        eotf_H273_ST428_1(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))
