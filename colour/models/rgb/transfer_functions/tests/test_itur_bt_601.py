"""
Define the unit tests for the
:mod:`colour.models.rgb.transfer_functions.itur_bt_601` module.
"""


import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.transfer_functions import oetf_BT601, oetf_inverse_BT601
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestOetf_BT601",
    "TestOetf_inverse_BT601",
]


class TestOetf_BT601:
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_601.oetf_BT601`
    definition unit tests methods.
    """

    def test_oetf_BT601(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_601.\
oetf_BT601` definition.
        """

        np.testing.assert_allclose(oetf_BT601(0.0), 0.0, atol=TOLERANCE_ABSOLUTE_TESTS)

        np.testing.assert_allclose(
            oetf_BT601(0.015), 0.067500000000000, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        np.testing.assert_allclose(
            oetf_BT601(0.18), 0.409007728864150, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        np.testing.assert_allclose(oetf_BT601(1.0), 1.0, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_n_dimensional_oetf_BT601(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_601.\
oetf_BT601` definition n-dimensional arrays support.
        """

        L = 0.18
        E = oetf_BT601(L)

        L = np.tile(L, 6)
        E = np.tile(E, 6)
        np.testing.assert_allclose(oetf_BT601(L), E, atol=TOLERANCE_ABSOLUTE_TESTS)

        L = np.reshape(L, (2, 3))
        E = np.reshape(E, (2, 3))
        np.testing.assert_allclose(oetf_BT601(L), E, atol=TOLERANCE_ABSOLUTE_TESTS)

        L = np.reshape(L, (2, 3, 1))
        E = np.reshape(E, (2, 3, 1))
        np.testing.assert_allclose(oetf_BT601(L), E, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_oetf_BT601(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_601.\
oetf_BT601` definition domain and range scale support.
        """

        L = 0.18
        E = oetf_BT601(L)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    oetf_BT601(L * factor),
                    E * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_oetf_BT601(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_601.\
oetf_BT601` definition nan support.
        """

        oetf_BT601(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestOetf_inverse_BT601:
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_601.\
oetf_inverse_BT601` definition unit tests methods.
    """

    def test_oetf_inverse_BT601(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_601.\
oetf_inverse_BT601` definition.
        """

        np.testing.assert_allclose(
            oetf_inverse_BT601(0.0), 0.0, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        np.testing.assert_allclose(
            oetf_inverse_BT601(0.067500000000000),
            0.015,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            oetf_inverse_BT601(0.409007728864150),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            oetf_inverse_BT601(1.0), 1.0, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_n_dimensional_oetf_inverse_BT601(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_601.\
oetf_inverse_BT601` definition n-dimensional arrays support.
        """

        E = 0.409007728864150
        L = oetf_inverse_BT601(E)

        E = np.tile(E, 6)
        L = np.tile(L, 6)
        np.testing.assert_allclose(
            oetf_inverse_BT601(E), L, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        E = np.reshape(E, (2, 3))
        L = np.reshape(L, (2, 3))
        np.testing.assert_allclose(
            oetf_inverse_BT601(E), L, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        E = np.reshape(E, (2, 3, 1))
        L = np.reshape(L, (2, 3, 1))
        np.testing.assert_allclose(
            oetf_inverse_BT601(E), L, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_oetf_inverse_BT601(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_601.\
oetf_inverse_BT601` definition domain and range scale support.
        """

        E = 0.409007728864150
        L = oetf_inverse_BT601(E)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    oetf_inverse_BT601(E * factor),
                    L * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_oetf_inverse_BT601(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_601.\
oetf_inverse_BT601` definition nan support.
        """

        oetf_inverse_BT601(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))
