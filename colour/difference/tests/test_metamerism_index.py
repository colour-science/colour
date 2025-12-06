"""Define the unit tests for the :mod:`colour.difference.metamerism_index` module."""

from __future__ import annotations

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.difference.metamerism_index import (
    Lab_to_metamerism_index,
    XYZ_to_metamerism_index,
)
from colour.utilities import domain_range_scale

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestLab_to_Metamerism_Index",
    "TestXYZ_to_Metamerism_Index",
]


class TestLab_to_Metamerism_Index:
    """
    Define :func:`colour.difference.metamerism_index.Lab_to_metamerism_index`
    definition unit tests methods.
    """

    def test_domain_range_scale_Lab_to_metamerism_index(self) -> None:
        """
        Test :func:`colour.difference.metamerism_index.Lab_to_metamerism_index`
        definition domain and range scale support.
        """

        Lab_1 = np.array([48.99183622, -0.10561667, 400.65619925])
        offset = np.array([0, 0, 2])

        c = ("Additive", "Multiplicative")
        m = ("CIE 1976", "CIE 1994", "CIE 2000", "CMC", "DIN99")
        it = [
            (
                correction,
                method,
                Lab_to_metamerism_index(
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
                        Lab_to_metamerism_index(
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


class TestXYZ_to_Metamerism_Index:
    """
    Define :func:`colour.difference.metamerism_index.XYZ_to_metamerism_index`
    definition unit tests methods.
    """

    def test_domain_range_scale_XYZ_to_metamerism_index(self) -> None:
        """
        Test :func:`colour.difference.metamerism_index.XYZ_to_metamerism_index`
        definition domain and range scale support.
        """

        XYZ_1 = np.array([0.20654008, 0.12197225, 0.05136952])
        offset = np.array([0, 0, 0.01])

        c = ("Additive", "Multiplicative")
        m = ("CIE 1976", "CIE 1994", "CIE 2000", "CMC", "DIN99")
        it = [
            (
                correction,
                method,
                XYZ_to_metamerism_index(
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
                        XYZ_to_metamerism_index(
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
