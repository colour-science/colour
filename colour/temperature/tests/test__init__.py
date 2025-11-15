"""Define the unit tests for the :mod:`colour.temperature` module."""

from __future__ import annotations

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.temperature import CCT_to_xy, xy_to_CCT

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestXy_to_CCT",
    "TestCCT_to_xy",
]


class TestXy_to_CCT:
    """
    Define :func:`colour.temperature.xy_to_CCT` definition unit tests methods.
    """

    def test_xy_to_CCT(self) -> None:
        """Test :func:`colour.temperature.xy_to_CCT` definition."""

        xy = np.array([0.31270, 0.32900])

        # Test default method (CIE Illuminant D Series)
        np.testing.assert_allclose(
            xy_to_CCT(xy),
            6508.1175148,
            atol=0.01,
        )

        # Test Hernandez 1999 method
        np.testing.assert_allclose(
            xy_to_CCT(xy, "Hernandez 1999"),
            6500.7420431,
            atol=0.01,
        )

        # Test McCamy 1992 method
        np.testing.assert_allclose(
            xy_to_CCT(xy, "McCamy 1992"),
            6505.08059131,
            atol=0.01,
        )


class TestCCT_to_xy:
    """
    Define :func:`colour.temperature.CCT_to_xy` definition unit tests methods.
    """

    def test_CCT_to_xy(self) -> None:
        """Test :func:`colour.temperature.CCT_to_xy` definition."""

        # Test default method (CIE Illuminant D Series)
        np.testing.assert_allclose(
            CCT_to_xy(6500),
            np.array([0.31277888, 0.3291835]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # Test explicit CIE Illuminant D Series method
        np.testing.assert_allclose(
            CCT_to_xy(6500, method="CIE Illuminant D Series"),
            np.array([0.31277888, 0.3291835]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # Test Hernandez 1999 method
        np.testing.assert_allclose(
            CCT_to_xy(6500, "Hernandez 1999"),
            np.array([0.31191663, 0.33419]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
