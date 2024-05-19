"""
Define the unit tests for the
:mod:`colour.models.rgb.transfer_functions.dicom_gsdf` module.
"""


import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.transfer_functions import (
    eotf_DICOMGSDF,
    eotf_inverse_DICOMGSDF,
)
from colour.utilities import domain_range_scale, ignore_numpy_errors

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

    def test_eotf_inverse_DICOMGSDF(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.dicom_gsdf.\
eotf_inverse_DICOMGSDF` definition.
        """

        np.testing.assert_allclose(
            eotf_inverse_DICOMGSDF(0.05),
            0.001007281350787,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            eotf_inverse_DICOMGSDF(130.0662),
            0.500486263438448,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            eotf_inverse_DICOMGSDF(4000),
            1.000160314715578,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            eotf_inverse_DICOMGSDF(130.0662, out_int=True),
            512,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_eotf_inverse_DICOMGSDF(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.dicom_gsdf.\
eotf_inverse_DICOMGSDF` definition n-dimensional arrays support.
        """

        L = 130.0662
        J = eotf_inverse_DICOMGSDF(L)

        L = np.tile(L, 6)
        J = np.tile(J, 6)
        np.testing.assert_allclose(
            eotf_inverse_DICOMGSDF(L), J, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        L = np.reshape(L, (2, 3))
        J = np.reshape(J, (2, 3))
        np.testing.assert_allclose(
            eotf_inverse_DICOMGSDF(L), J, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        L = np.reshape(L, (2, 3, 1))
        J = np.reshape(J, (2, 3, 1))
        np.testing.assert_allclose(
            eotf_inverse_DICOMGSDF(L), J, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_eotf_inverse_DICOMGSDF(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.dicom_gsdf.\
eotf_inverse_DICOMGSDF` definition domain and range scale support.
        """

        L = 130.0662
        J = eotf_inverse_DICOMGSDF(L)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    eotf_inverse_DICOMGSDF(L * factor),
                    J * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_eotf_inverse_DICOMGSDF(self):
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

    def test_eotf_DICOMGSDF(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.dicom_gsdf.\
eotf_DICOMGSDF` definition.
        """

        np.testing.assert_allclose(
            eotf_DICOMGSDF(0.001007281350787),
            0.050143440671692,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            eotf_DICOMGSDF(0.500486263438448),
            130.062864706476550,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            eotf_DICOMGSDF(1.000160314715578),
            3997.586161113322300,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            eotf_DICOMGSDF(512, in_int=True),
            130.065284012159790,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_eotf_DICOMGSDF(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.dicom_gsdf.\
eotf_DICOMGSDF` definition n-dimensional arrays support.
        """

        J = 0.500486263438448
        L = eotf_DICOMGSDF(J)

        J = np.tile(J, 6)
        L = np.tile(L, 6)
        np.testing.assert_allclose(eotf_DICOMGSDF(J), L, atol=TOLERANCE_ABSOLUTE_TESTS)

        J = np.reshape(J, (2, 3))
        L = np.reshape(L, (2, 3))
        np.testing.assert_allclose(eotf_DICOMGSDF(J), L, atol=TOLERANCE_ABSOLUTE_TESTS)

        J = np.reshape(J, (2, 3, 1))
        L = np.reshape(L, (2, 3, 1))
        np.testing.assert_allclose(eotf_DICOMGSDF(J), L, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_eotf_DICOMGSDF(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.dicom_gsdf.\
eotf_DICOMGSDF` definition domain and range scale support.
        """

        J = 0.500486263438448
        L = eotf_DICOMGSDF(J)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    eotf_DICOMGSDF(J * factor),
                    L * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_eotf_DICOMGSDF(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.dicom_gsdf.\
eotf_DICOMGSDF` definition nan support.
        """

        eotf_DICOMGSDF(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))
