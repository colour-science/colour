"""
Define the unit tests for the
:mod:`colour.models.rgb.transfer_functions.linear` module.
"""

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.transfer_functions import linear_function
from colour.utilities import ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestLinearFunction",
]


class TestLinearFunction:
    """
    Define :func:`colour.models.rgb.transfer_functions.linear.\
linear_function` definition unit tests methods.
    """

    def test_linear_function(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.linear.\
linear_function` definition.
        """

        assert linear_function(0.0) == 0.0

        assert linear_function(0.18) == 0.18

        assert linear_function(1.0) == 1.0

    def test_n_dimensional_linear_function(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.linear.\
linear_function` definition n-dimensional arrays support.
        """

        a = 0.18
        a_p = linear_function(a)

        a = np.tile(a, 6)
        a_p = np.tile(a_p, 6)
        np.testing.assert_allclose(
            linear_function(a), a_p, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        a = np.reshape(a, (2, 3))
        a_p = np.reshape(a_p, (2, 3))
        np.testing.assert_allclose(
            linear_function(a), a_p, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        a = np.reshape(a, (2, 3, 1))
        a_p = np.reshape(a_p, (2, 3, 1))
        np.testing.assert_allclose(
            linear_function(a), a_p, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    @ignore_numpy_errors
    def test_nan_linear_function(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.linear.\
linear_function` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        linear_function(cases)
