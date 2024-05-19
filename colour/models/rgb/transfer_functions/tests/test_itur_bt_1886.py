"""
Define the unit tests for the
:mod:`colour.models.rgb.transfer_functions.itur_bt_1886` module.
"""


import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.transfer_functions import (
    eotf_BT1886,
    eotf_inverse_BT1886,
)
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestEotf_inverse_BT1886",
    "TestEotf_BT1886",
]


class TestEotf_inverse_BT1886:
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_1886.\
eotf_inverse_BT1886` definition unit tests methods.
    """

    def test_eotf_inverse_BT1886(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_1886.\
eotf_inverse_BT1886` definition.
        """

        np.testing.assert_allclose(
            eotf_inverse_BT1886(0.0), 0.0, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        np.testing.assert_allclose(
            eotf_inverse_BT1886(0.016317514686316),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            eotf_inverse_BT1886(1.0), 1.0, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_n_dimensional_eotf_inverse_BT1886(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_1886.\
eotf_inverse_BT1886` definition n-dimensional arrays support.
        """

        L = 0.016317514686316
        V = eotf_inverse_BT1886(L)

        L = np.tile(L, 6)
        V = np.tile(V, 6)
        np.testing.assert_allclose(
            eotf_inverse_BT1886(L), V, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        L = np.reshape(L, (2, 3))
        V = np.reshape(V, (2, 3))
        np.testing.assert_allclose(
            eotf_inverse_BT1886(L), V, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        L = np.reshape(L, (2, 3, 1))
        V = np.reshape(V, (2, 3, 1))
        np.testing.assert_allclose(
            eotf_inverse_BT1886(L), V, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_eotf_inverse_BT1886(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_1886.\
eotf_inverse_BT1886` definition domain and range scale support.
        """

        L = 0.18
        V = eotf_inverse_BT1886(L)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    eotf_inverse_BT1886(L * factor),
                    V * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_eotf_inverse_BT1886(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_1886.\
eotf_inverse_BT1886` definition nan support.
        """

        eotf_inverse_BT1886(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestEotf_BT1886:
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_1886.\
eotf_BT1886` definition unit tests methods.
    """

    def test_eotf_BT1886(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_1886.\
eotf_BT1886` definition.
        """

        np.testing.assert_allclose(eotf_BT1886(0.0), 0.0, atol=TOLERANCE_ABSOLUTE_TESTS)

        np.testing.assert_allclose(
            eotf_BT1886(0.18), 0.016317514686316, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        np.testing.assert_allclose(eotf_BT1886(1.0), 1.0, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_n_dimensional_eotf_BT1886(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_1886.\
eotf_BT1886` definition n-dimensional arrays support.
        """

        V = 0.18
        L = eotf_BT1886(V)

        V = np.tile(V, 6)
        L = np.tile(L, 6)
        np.testing.assert_allclose(eotf_BT1886(V), L, atol=TOLERANCE_ABSOLUTE_TESTS)

        V = np.reshape(V, (2, 3))
        L = np.reshape(L, (2, 3))
        np.testing.assert_allclose(eotf_BT1886(V), L, atol=TOLERANCE_ABSOLUTE_TESTS)

        V = np.reshape(V, (2, 3, 1))
        L = np.reshape(L, (2, 3, 1))
        np.testing.assert_allclose(eotf_BT1886(V), L, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_eotf_BT1886(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_1886.\
eotf_BT1886` definition domain and range scale support.
        """

        V = 0.016317514686316
        L = eotf_BT1886(V)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    eotf_BT1886(V * factor),
                    L * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_eotf_BT1886(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_1886.\
eotf_BT1886` definition nan support.
        """

        eotf_BT1886(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))
