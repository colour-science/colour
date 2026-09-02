"""
Define the unit tests for the
:mod:`colour.models.rgb.transfer_functions.arib_std_b67` module.
"""

from __future__ import annotations

import typing

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.transfer_functions import (
    oetf_ARIBSTDB67,
    oetf_inverse_ARIBSTDB67,
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
    "TestOetf_ARIBSTDB67",
    "TestOetf_inverse_ARIBSTDB67",
]


class TestOetf_ARIBSTDB67:
    """
    Define :func:`colour.models.rgb.transfer_functions.arib_std_b67.\
oetf_ARIBSTDB67` definition unit tests methods.
    """

    def test_oetf_ARIBSTDB67(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.arib_std_b67.\
oetf_ARIBSTDB67` definition.
        """

        xp_assert_close(
            oetf_ARIBSTDB67(xp_as_array(-0.25, xp=xp)),
            -0.25,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_ARIBSTDB67(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_ARIBSTDB67(xp_as_array(0.18, xp=xp)),
            0.212132034355964,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_ARIBSTDB67(xp_as_array(1.0, xp=xp)),
            0.5,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_ARIBSTDB67(xp_as_array(64.0, xp=xp)),
            1.302858098046995,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_oetf_ARIBSTDB67(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.arib_std_b67.\
oetf_ARIBSTDB67` definition n-dimensional arrays support.
        """

        E = 0.18
        E_p = as_ndarray(oetf_ARIBSTDB67(xp_as_array(E, xp=xp)))

        E = xp.tile(xp_as_array(E, xp=xp), (6,))
        E_p = xp.tile(xp_as_array(E_p, xp=xp), (6,))
        xp_assert_close(oetf_ARIBSTDB67(E), E_p, atol=TOLERANCE_ABSOLUTE_TESTS)

        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3), xp=xp)
        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3), xp=xp)
        xp_assert_close(oetf_ARIBSTDB67(E), E_p, atol=TOLERANCE_ABSOLUTE_TESTS)

        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3, 1), xp=xp)
        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(oetf_ARIBSTDB67(E), E_p, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_oetf_ARIBSTDB67(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.arib_std_b67.\
oetf_ARIBSTDB67` definition domain and range scale support.
        """

        E = 0.18
        E_p = as_ndarray(oetf_ARIBSTDB67(xp_as_array(E, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    oetf_ARIBSTDB67(xp_as_array(E * factor, xp=xp)),
                    E_p * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_oetf_ARIBSTDB67(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.arib_std_b67.\
oetf_ARIBSTDB67` definition nan support.
        """

        oetf_ARIBSTDB67(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestOetf_inverse_ARIBSTDB67:
    """
    Define :func:`colour.models.rgb.transfer_functions.arib_std_b67.\
oetf_inverse_ARIBSTDB67` definition unit tests methods.
    """

    def test_oetf_inverse_ARIBSTDB67(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.arib_std_b67.\
oetf_inverse_ARIBSTDB67` definition.
        """

        xp_assert_close(
            oetf_inverse_ARIBSTDB67(xp_as_array(-0.25, xp=xp)),
            -0.25,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_inverse_ARIBSTDB67(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_inverse_ARIBSTDB67(xp_as_array(0.212132034355964, xp=xp)),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_inverse_ARIBSTDB67(xp_as_array(0.5, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_inverse_ARIBSTDB67(xp_as_array(1.302858098046995, xp=xp)),
            64.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_oetf_inverse_ARIBSTDB67(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.arib_std_b67.\
oetf_inverse_ARIBSTDB67` definition n-dimensional arrays support.
        """

        E_p = 0.212132034355964
        E = as_ndarray(oetf_inverse_ARIBSTDB67(xp_as_array(E_p, xp=xp)))

        E_p = xp.tile(xp_as_array(E_p, xp=xp), (6,))
        E = xp.tile(xp_as_array(E, xp=xp), (6,))
        xp_assert_close(oetf_inverse_ARIBSTDB67(E_p), E, atol=TOLERANCE_ABSOLUTE_TESTS)

        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3), xp=xp)
        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3), xp=xp)
        xp_assert_close(oetf_inverse_ARIBSTDB67(E_p), E, atol=TOLERANCE_ABSOLUTE_TESTS)

        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3, 1), xp=xp)
        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(oetf_inverse_ARIBSTDB67(E_p), E, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_oetf_inverse_ARIBSTDB67(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.arib_std_b67.\
oetf_inverse_ARIBSTDB67` definition domain and range scale support.
        """

        E_p = 0.212132034355964
        E = as_ndarray(oetf_inverse_ARIBSTDB67(xp_as_array(E_p, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    oetf_inverse_ARIBSTDB67(xp_as_array(E_p * factor, xp=xp)),
                    E * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_oetf_inverse_ARIBSTDB67(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.arib_std_b67.\
oetf_inverse_ARIBSTDB67` definition nan support.
        """

        oetf_inverse_ARIBSTDB67(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))
