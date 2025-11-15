"""Define the unit tests for the :mod:`colour.contrast` module."""

from __future__ import annotations

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.contrast import contrast_sensitivity_function

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestContrastSensitivityFunction",
]


class TestContrastSensitivityFunction:
    """
    Define :func:`colour.contrast.contrast_sensitivity_function` definition
    unit tests methods.
    """

    def test_contrast_sensitivity_function(self) -> None:
        """Test :func:`colour.contrast.contrast_sensitivity_function` definition."""

        # Test default method (Barten 1999)
        np.testing.assert_allclose(
            contrast_sensitivity_function(
                u=4,
                sigma=0.01,
                E=65,
                X_0=60,
                X_max=12,
                Y_0=60,
                Y_max=12,
                p=1.2 * 10**6,
            ),
            352.761342126727020,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # Test explicit Barten 1999 method with different parameters
        np.testing.assert_allclose(
            contrast_sensitivity_function(
                "Barten 1999",
                u=8,
                sigma=0.01,
                E=65,
                X_0=60,
                X_max=12,
                Y_0=60,
                Y_max=12,
                p=1.2 * 10**6,
            ),
            177.706338840717340,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # Test with another set of parameters
        np.testing.assert_allclose(
            contrast_sensitivity_function(
                u=20,
                sigma=0.01,
                E=65,
                X_0=60,
                X_max=12,
                Y_0=60,
                Y_max=12,
                p=1.2 * 10**6,
            ),
            37.455090830648620,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
