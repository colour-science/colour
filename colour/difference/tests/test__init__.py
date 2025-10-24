"""Define the unit tests for the :mod:`colour.difference` module."""

from __future__ import annotations

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.difference import delta_E, metamerism_index
from colour.utilities import domain_range_scale

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestDelta_E",
    "TestMetamerismIndex",
]


class TestDelta_E:
    """Define :func:`colour.difference.delta_E` definition unit tests methods."""

    def test_domain_range_scale_delta_E(self) -> None:
        """
        Test :func:`colour.difference.delta_E` definition domain and range
        scale support.
        """

        Lab_1 = np.array([48.99183622, -0.10561667, 400.65619925])
        Lab_2 = np.array([50.65907324, -0.11671910, 402.82235718])

        m = ("CIE 1976", "CIE 1994", "CIE 2000", "CMC", "DIN99")
        v = [delta_E(Lab_1, Lab_2, method) for method in m]

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for method, value in zip(m, v, strict=True):
            for scale, factor in d_r:
                with domain_range_scale(scale):
                    np.testing.assert_allclose(
                        delta_E(Lab_1 * factor, Lab_2 * factor, method),
                        value,
                        atol=TOLERANCE_ABSOLUTE_TESTS,
                    )


class TestMetamerismIndex:
    """
    Define :func:`colour.difference.metamerism_index` definition
    unit tests methods.
    """

    def test_domain_range_scale_metamerism_index(self) -> None:
        """
        Test :func:`colour.difference.metamerism_index` definition domain and
        range scale support.
        """

        Lab_1 = np.array([48.99183622, -0.10561667, 400.65619925])
        Lab_2 = np.array([50.65907324, -0.11671910, 402.82235718])
        offset = np.array([2, 0, 0])

        methods = ("CIE 1976", "CIE 1994", "CIE 2000", "CMC", "DIN99")

        # Compute baseline (reference scale) values for both modes
        v_dE = [
            metamerism_index(Lab_1, Lab_2, Lab_1, Lab_2 + offset, method, use_dE=True)
            for method in methods
        ]
        v_dLCH = [
            metamerism_index(Lab_1, Lab_2, Lab_1, Lab_2 + offset, method, use_dE=False)
            for method in methods
        ]

        # Domain-range scaling factors: ("scale name", factor)
        d_r = (("reference", 1), ("1", 0.01), ("100", 1))

        for method, value_dE, value_dLCH in zip(methods, v_dE, v_dLCH, strict=True):
            for scale, factor in d_r:
                with domain_range_scale(scale):
                    # ΔE-based mode
                    np.testing.assert_allclose(
                        metamerism_index(
                            Lab_1 * factor,
                            Lab_2 * factor,
                            Lab_1 * factor,
                            (Lab_2 + offset) * factor,
                            method,
                            use_dE=True,
                        ),
                        value_dE,
                        atol=TOLERANCE_ABSOLUTE_TESTS,
                    )

                    # Componentwise ΔLCH-based mode
                    np.testing.assert_allclose(
                        metamerism_index(
                            Lab_1 * factor,
                            Lab_2 * factor,
                            Lab_1 * factor,
                            (Lab_2 + offset) * factor,
                            method,
                            use_dE=False,
                        ),
                        value_dLCH,
                        atol=TOLERANCE_ABSOLUTE_TESTS,
                    )
