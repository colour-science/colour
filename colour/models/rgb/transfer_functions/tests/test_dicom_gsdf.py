"""
Define the unit tests for the
:mod:`colour.models.rgb.transfer_functions.dicom_gsdf` module.
"""

from __future__ import annotations

import typing

import numpy as np
import pytest

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.transfer_functions import eotf_DICOMGSDF, eotf_inverse_DICOMGSDF
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
    "TestEotf_inverse_DICOMGSDF",
    "TestEotf_DICOMGSDF",
]


class TestEotf_inverse_DICOMGSDF:
    """
    Define :func:`colour.models.rgb.transfer_functions.dicom_gsdf.\
eotf_inverse_DICOMGSDF` definition unit tests methods.
    """

    def test_eotf_inverse_DICOMGSDF(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.dicom_gsdf.\
eotf_inverse_DICOMGSDF` definition.
        """

        xp_assert_close(
            eotf_inverse_DICOMGSDF(xp_as_array(0.05, xp=xp)),
            0.001007281350787,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_inverse_DICOMGSDF(xp_as_array(130.0662, xp=xp)),
            0.500486263438448,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_inverse_DICOMGSDF(xp_as_array(4000, xp=xp)),
            1.000160314715578,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_inverse_DICOMGSDF(xp_as_array(130.0662, xp=xp), out_int=True),
            512,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_eotf_inverse_DICOMGSDF(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.dicom_gsdf.\
eotf_inverse_DICOMGSDF` definition n-dimensional arrays support.
        """

        L = 130.0662
        J = as_ndarray(eotf_inverse_DICOMGSDF(xp_as_array(L, xp=xp)))

        L = xp.tile(xp_as_array(L, xp=xp), (6,))
        J = xp.tile(xp_as_array(J, xp=xp), (6,))
        xp_assert_close(eotf_inverse_DICOMGSDF(L), J, atol=TOLERANCE_ABSOLUTE_TESTS)

        L = xp_reshape(xp_as_array(L, xp=xp), (2, 3), xp=xp)
        J = xp_reshape(xp_as_array(J, xp=xp), (2, 3), xp=xp)
        xp_assert_close(eotf_inverse_DICOMGSDF(L), J, atol=TOLERANCE_ABSOLUTE_TESTS)

        L = xp_reshape(xp_as_array(L, xp=xp), (2, 3, 1), xp=xp)
        J = xp_reshape(xp_as_array(J, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(eotf_inverse_DICOMGSDF(L), J, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_eotf_inverse_DICOMGSDF(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.dicom_gsdf.\
eotf_inverse_DICOMGSDF` definition domain and range scale support.
        """

        L = 130.0662
        J = as_ndarray(eotf_inverse_DICOMGSDF(xp_as_array(L, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    eotf_inverse_DICOMGSDF(xp_as_array(L * factor, xp=xp)),
                    J * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_eotf_inverse_DICOMGSDF(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.dicom_gsdf.\
eotf_inverse_DICOMGSDF` definition nan support.
        """

        eotf_inverse_DICOMGSDF(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestEotf_DICOMGSDF:
    """
        Define :func:`colour.models.rgb.transfer_functions.dicom_gsdf.
    eotf_DICOMGSDF` definition unit tests methods.
    """

    @pytest.mark.mps_tolerance_absolute(2e-1)
    def test_eotf_DICOMGSDF(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.dicom_gsdf.\
eotf_DICOMGSDF` definition.
        """

        xp_assert_close(
            eotf_DICOMGSDF(xp_as_array(0.001007281350787, xp=xp)),
            0.050143440671692,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_DICOMGSDF(xp_as_array(0.500486263438448, xp=xp)),
            130.062864706476550,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_DICOMGSDF(xp_as_array(1.000160314715578, xp=xp)),
            3997.586161113322300,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_DICOMGSDF(xp_as_array(512, xp=xp), in_int=True),
            130.065284012159790,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_eotf_DICOMGSDF(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.dicom_gsdf.\
eotf_DICOMGSDF` definition n-dimensional arrays support.
        """

        J = 0.500486263438448
        L = as_ndarray(eotf_DICOMGSDF(xp_as_array(J, xp=xp)))

        J = xp.tile(xp_as_array(J, xp=xp), (6,))
        L = xp.tile(xp_as_array(L, xp=xp), (6,))
        xp_assert_close(eotf_DICOMGSDF(J), L, atol=TOLERANCE_ABSOLUTE_TESTS)

        J = xp_reshape(xp_as_array(J, xp=xp), (2, 3), xp=xp)
        L = xp_reshape(xp_as_array(L, xp=xp), (2, 3), xp=xp)
        xp_assert_close(eotf_DICOMGSDF(J), L, atol=TOLERANCE_ABSOLUTE_TESTS)

        J = xp_reshape(xp_as_array(J, xp=xp), (2, 3, 1), xp=xp)
        L = xp_reshape(xp_as_array(L, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(eotf_DICOMGSDF(J), L, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_eotf_DICOMGSDF(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.dicom_gsdf.\
eotf_DICOMGSDF` definition domain and range scale support.
        """

        J = 0.500486263438448
        L = as_ndarray(eotf_DICOMGSDF(xp_as_array(J, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    eotf_DICOMGSDF(xp_as_array(J * factor, xp=xp)),
                    L * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_eotf_DICOMGSDF(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.dicom_gsdf.\
eotf_DICOMGSDF` definition nan support.
        """

        eotf_DICOMGSDF(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))
