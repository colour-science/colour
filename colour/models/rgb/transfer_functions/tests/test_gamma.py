"""
Define the unit tests for the
:mod:`colour.models.rgb.transfer_functions.gamma` module.
"""


import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.transfer_functions import gamma_function
from colour.utilities import ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestGammaFunction",
]


class TestGammaFunction:
    """
    Define :func:`colour.models.rgb.transfer_functions.gamma.gamma_function`
    definition unit tests methods.
    """

    def test_gamma_function(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.gamma.\
gamma_function` definition.
        """

        np.testing.assert_allclose(
            gamma_function(0.0, 2.2), 0.0, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        np.testing.assert_allclose(
            gamma_function(0.18, 2.2),
            0.022993204992707,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            gamma_function(0.022993204992707, 1.0 / 2.2),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            gamma_function(-0.18, 2.0),
            0.0323999999999998,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_array_equal(gamma_function(-0.18, 2.2), np.nan)

        np.testing.assert_allclose(
            gamma_function(-0.18, 2.2, "Mirror"),
            -0.022993204992707,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            gamma_function(-0.18, 2.2, "Preserve"),
            -0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            gamma_function(-0.18, 2.2, "Clamp"),
            0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_array_equal(gamma_function(-0.18, -2.2), np.nan)

        np.testing.assert_allclose(
            gamma_function(0.0, -2.2, "Mirror"),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            gamma_function(0.0, 2.2, "Preserve"),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            gamma_function(0.0, 2.2, "Clamp"), 0, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_n_dimensional_gamma_function(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.gamma.\
gamma_function` definition n-dimensional arrays support.
        """

        a = 0.18
        a_p = gamma_function(a, 2.2)

        a = np.tile(a, 6)
        a_p = np.tile(a_p, 6)
        np.testing.assert_allclose(
            gamma_function(a, 2.2), a_p, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        a = np.reshape(a, (2, 3))
        a_p = np.reshape(a_p, (2, 3))
        np.testing.assert_allclose(
            gamma_function(a, 2.2), a_p, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        a = np.reshape(a, (2, 3, 1))
        a_p = np.reshape(a_p, (2, 3, 1))
        np.testing.assert_allclose(
            gamma_function(a, 2.2), a_p, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        a = -0.18
        a_p = -0.022993204992707
        np.testing.assert_allclose(
            gamma_function(a, 2.2, "Mirror"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = np.tile(a, 6)
        a_p = np.tile(a_p, 6)
        np.testing.assert_allclose(
            gamma_function(a, 2.2, "Mirror"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = np.reshape(a, (2, 3))
        a_p = np.reshape(a_p, (2, 3))
        np.testing.assert_allclose(
            gamma_function(a, 2.2, "Mirror"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = np.reshape(a, (2, 3, 1))
        a_p = np.reshape(a_p, (2, 3, 1))
        np.testing.assert_allclose(
            gamma_function(a, 2.2, "Mirror"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = -0.18
        a_p = -0.18
        np.testing.assert_allclose(
            gamma_function(a, 2.2, "Preserve"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = np.tile(a, 6)
        a_p = np.tile(a_p, 6)
        np.testing.assert_allclose(
            gamma_function(a, 2.2, "Preserve"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = np.reshape(a, (2, 3))
        a_p = np.reshape(a_p, (2, 3))
        np.testing.assert_allclose(
            gamma_function(a, 2.2, "Preserve"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = np.reshape(a, (2, 3, 1))
        a_p = np.reshape(a_p, (2, 3, 1))
        np.testing.assert_allclose(
            gamma_function(a, 2.2, "Preserve"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = -0.18
        a_p = 0.0
        np.testing.assert_allclose(
            gamma_function(a, 2.2, "Clamp"), a_p, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        a = np.tile(a, 6)
        a_p = np.tile(a_p, 6)
        np.testing.assert_allclose(
            gamma_function(a, 2.2, "Clamp"), a_p, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        a = np.reshape(a, (2, 3))
        a_p = np.reshape(a_p, (2, 3))
        np.testing.assert_allclose(
            gamma_function(a, 2.2, "Clamp"), a_p, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        a = np.reshape(a, (2, 3, 1))
        a_p = np.reshape(a_p, (2, 3, 1))
        np.testing.assert_allclose(
            gamma_function(a, 2.2, "Clamp"), a_p, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    @ignore_numpy_errors
    def test_nan_gamma_function(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.gamma.\
gamma_function` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        gamma_function(cases, cases)
