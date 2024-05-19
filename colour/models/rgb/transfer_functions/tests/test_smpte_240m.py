"""
Define the unit tests for the
:mod:`colour.models.rgb.transfer_functions.smpte_240m` module.
"""


import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.transfer_functions import eotf_SMPTE240M, oetf_SMPTE240M
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestOetf_SMPTE240M",
    "TestEotf_SMPTE240M",
]


class TestOetf_SMPTE240M:
    """
    Define :func:`colour.models.rgb.transfer_functions.smpte_240m.\
oetf_SMPTE240M` definition unit tests methods.
    """

    def test_oetf_SMPTE240M(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.smpte_240m.\
oetf_SMPTE240M` definition.
        """

        np.testing.assert_allclose(
            oetf_SMPTE240M(0.0), 0.0, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        np.testing.assert_allclose(
            oetf_SMPTE240M(0.02),
            0.080000000000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            oetf_SMPTE240M(0.18),
            0.402285796753870,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            oetf_SMPTE240M(1.0), 1.0, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_n_dimensional_oetf_SMPTE240M(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.smpte_240m.\
oetf_SMPTE240M` definition n-dimensional arrays support.
        """

        L_c = 0.18
        V_c = oetf_SMPTE240M(L_c)

        L_c = np.tile(L_c, 6)
        V_c = np.tile(V_c, 6)
        np.testing.assert_allclose(
            oetf_SMPTE240M(L_c), V_c, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        L_c = np.reshape(L_c, (2, 3))
        V_c = np.reshape(V_c, (2, 3))
        np.testing.assert_allclose(
            oetf_SMPTE240M(L_c), V_c, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        L_c = np.reshape(L_c, (2, 3, 1))
        V_c = np.reshape(V_c, (2, 3, 1))
        np.testing.assert_allclose(
            oetf_SMPTE240M(L_c), V_c, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_oetf_SMPTE240M(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.smpte_240m.\
oetf_SMPTE240M` definition domain and range scale support.
        """

        L_c = 0.18
        V_c = oetf_SMPTE240M(L_c)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    oetf_SMPTE240M(L_c * factor),
                    V_c * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_oetf_SMPTE240M(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.smpte_240m.\
oetf_SMPTE240M` definition nan support.
        """

        oetf_SMPTE240M(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestEotf_SMPTE240M:
    """
    Define :func:`colour.models.rgb.transfer_functions.smpte_240m.\
eotf_SMPTE240M` definition unit tests methods.
    """

    def test_eotf_SMPTE240M(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.smpte_240m.\
eotf_SMPTE240M` definition.
        """

        np.testing.assert_allclose(
            eotf_SMPTE240M(0.0), 0.0, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        np.testing.assert_allclose(
            eotf_SMPTE240M(0.080000000000000),
            0.02,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            eotf_SMPTE240M(0.402285796753870),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            eotf_SMPTE240M(1.0), 1.0, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_n_dimensional_eotf_SMPTE240M(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.smpte_240m.\
eotf_SMPTE240M` definition n-dimensional arrays support.
        """

        V_r = 0.402285796753870
        L_r = eotf_SMPTE240M(V_r)

        V_r = np.tile(V_r, 6)
        L_r = np.tile(L_r, 6)
        np.testing.assert_allclose(
            eotf_SMPTE240M(V_r), L_r, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        V_r = np.reshape(V_r, (2, 3))
        L_r = np.reshape(L_r, (2, 3))
        np.testing.assert_allclose(
            eotf_SMPTE240M(V_r), L_r, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        V_r = np.reshape(V_r, (2, 3, 1))
        L_r = np.reshape(L_r, (2, 3, 1))
        np.testing.assert_allclose(
            eotf_SMPTE240M(V_r), L_r, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_eotf_SMPTE240M(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.smpte_240m.\
eotf_SMPTE240M` definition domain and range scale support.
        """

        V_r = 0.402285796753870
        L_r = eotf_SMPTE240M(V_r)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    eotf_SMPTE240M(V_r * factor),
                    L_r * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_eotf_SMPTE240M(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.smpte_240m.\
eotf_SMPTE240M` definition nan support.
        """

        eotf_SMPTE240M(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))
