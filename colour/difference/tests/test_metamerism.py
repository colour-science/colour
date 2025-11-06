"""Define the unit tests for the :mod:`colour.difference.metamerism` module."""

from __future__ import annotations

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.difference.metamerism import (
    metamerism_index_from_Lab,
    metamerism_index_from_XYZ,
)
from colour.utilities import domain_range_scale

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestMetamerism_Index_From_Lab",
    "TestMetamerism_Index_From_XYZ",
]


class TestMetamerism_Index_From_Lab:
    """
    Define :func:`colour.difference.metamerism.metamerism_index_from_Lab`
    definition unit tests methods.
    """

    def test_domain_range_scale_metamerism_index_from_lab(self) -> None:
        """
        Test :func:`colour.difference.metamerism.metamerism_index_from_Lab`
        definition domain and range scale support.
        """

        Lab_1 = np.array([48.99183622, -0.10561667, 400.65619925])
        offset = np.array([0, 0, 2])

        c = ("additive", "multiplicative")
        m = ("CIE 1976", "CIE 1994", "CIE 2000", "CMC", "DIN99")
        it = [
            (
                correction,
                method,
                metamerism_index_from_Lab(
                    Lab_1 + offset,
                    Lab_1,
                    Lab_1,
                    Lab_1,
                    correction=correction,
                    method=method,
                ),
            )
            for correction in c
            for method in m
        ]

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for correction, method, value in it:
            for scale, factor in d_r:
                with domain_range_scale(scale):
                    np.testing.assert_allclose(
                        metamerism_index_from_Lab(
                            (Lab_1 + offset) * factor,
                            Lab_1 * factor,
                            Lab_1 * factor,
                            Lab_1 * factor,
                            correction=correction,
                            method=method,
                        ),
                        value,
                        atol=TOLERANCE_ABSOLUTE_TESTS,
                    )


class TestMetamerism_Index_From_XYZ:
    """
    Define :func:`colour.difference.metamerism.metamerism_index_from_XYZ`
    definition unit tests methods.
    """

    def test_domain_range_scale_metamerism_index_from_XYZ(self) -> None:
        """
        Test :func:`colour.difference.metamerism.metamerism_index_from_XYZ`
        definition domain and range scale support.
        """

        XYZ_1 = np.array([0.20654008, 0.12197225, 0.05136952])
        offset = np.array([0, 0, 0.01])

        c = ("additive", "multiplicative")
        m = ("CIE 1976", "CIE 1994", "CIE 2000", "CMC", "DIN99")
        it = [
            (
                correction,
                method,
                metamerism_index_from_XYZ(
                    XYZ_1 + offset,
                    XYZ_1,
                    XYZ_1,
                    XYZ_1,
                    correction=correction,
                    method=method,
                ),
            )
            for correction in c
            for method in m
        ]

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for correction, method, value in it:
            for scale, factor in d_r:
                with domain_range_scale(scale):
                    np.testing.assert_allclose(
                        metamerism_index_from_XYZ(
                            (XYZ_1 + offset) * factor,
                            XYZ_1 * factor,
                            XYZ_1 * factor,
                            XYZ_1 * factor,
                            correction=correction,
                            method=method,
                        ),
                        value,
                        atol=TOLERANCE_ABSOLUTE_TESTS,
                    )
