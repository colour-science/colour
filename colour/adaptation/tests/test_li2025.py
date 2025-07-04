"""Define the unit tests for the :mod:`colour.adaptation.li2025` module."""

from __future__ import annotations

from itertools import product

import numpy as np

from colour.adaptation import chromatic_adaptation_Li2025
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestChromaticAdaptationLi2025",
]


class TestChromaticAdaptationLi2025:
    """
    Define :func:`colour.adaptation.li2025.chromatic_adaptation_Li2025`
    definition unit tests methods.
    """

    def test_chromatic_adaptation_Li2025(self) -> None:
        """
        Test :func:`colour.adaptation.li2025.chromatic_adaptation_Li2025`
        definition.
        """

        np.testing.assert_allclose(
            chromatic_adaptation_Li2025(
                XYZ_s=np.array([48.900, 43.620, 6.250]),
                XYZ_ws=np.array([109.850, 100, 35.585]),
                XYZ_wd=np.array([95.047, 100, 108.883]),
                L_A=318.31,
                F_surround=1.0,
            ),
            np.array([40.00725815, 43.70148954, 21.32902932]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            chromatic_adaptation_Li2025(
                XYZ_s=np.array([52.034, 58.824, 23.703]),
                XYZ_ws=np.array([92.288, 100, 38.775]),
                XYZ_wd=np.array([105.432, 100, 137.392]),
                L_A=318.31,
                F_surround=1.0,
            ),
            np.array([59.99869086, 58.81067197, 83.41018242]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            chromatic_adaptation_Li2025(
                XYZ_s=np.array([48.900, 43.620, 6.250]),
                XYZ_ws=np.array([109.850, 100, 35.585]),
                XYZ_wd=np.array([95.047, 100, 108.883]),
                L_A=20.0,
                F_surround=1.0,
            ),
            np.array([41.22388901, 43.69034082, 19.26604215]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            chromatic_adaptation_Li2025(
                XYZ_s=np.array([48.900, 43.620, 6.250]),
                XYZ_ws=np.array([109.850, 100, 35.585]),
                XYZ_wd=np.array([95.047, 100, 108.883]),
                L_A=318.31,
                F_surround=1.0,
                discount_illuminant=True,
            ),
            np.array([39.95779686, 43.70194278, 21.41289865]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_chromatic_adaptation_Li2025(self) -> None:
        """
        Test :func:`colour.adaptation.li2025.chromatic_adaptation_Li2025`
        definition n-dimensional arrays support.
        """

        XYZ_s = np.array([48.900, 43.620, 6.250])
        XYZ_ws = np.array([109.850, 100, 35.585])
        XYZ_wd = np.array([95.047, 100, 108.883])
        L_A = 318.31
        F_surround = 1.0
        XYZ_d = chromatic_adaptation_Li2025(XYZ_s, XYZ_ws, XYZ_wd, L_A, F_surround)

        XYZ_s = np.tile(XYZ_s, (6, 1))
        XYZ_d = np.tile(XYZ_d, (6, 1))
        np.testing.assert_allclose(
            chromatic_adaptation_Li2025(XYZ_s, XYZ_ws, XYZ_wd, L_A, F_surround),
            XYZ_d,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ_ws = np.tile(XYZ_ws, (6, 1))
        XYZ_wd = np.tile(XYZ_wd, (6, 1))
        L_A = np.tile(L_A, 6)
        F_surround = np.tile(F_surround, 6)
        np.testing.assert_allclose(
            chromatic_adaptation_Li2025(XYZ_s, XYZ_ws, XYZ_wd, L_A, F_surround),
            XYZ_d,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ_s = np.reshape(XYZ_s, (2, 3, 3))
        XYZ_ws = np.reshape(XYZ_ws, (2, 3, 3))
        XYZ_wd = np.reshape(XYZ_wd, (2, 3, 3))
        L_A = np.reshape(L_A, (2, 3))
        F_surround = np.reshape(F_surround, (2, 3))
        XYZ_d = np.reshape(XYZ_d, (2, 3, 3))
        np.testing.assert_allclose(
            chromatic_adaptation_Li2025(XYZ_s, XYZ_ws, XYZ_wd, L_A, F_surround),
            XYZ_d,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_chromatic_adaptation_Li2025(self) -> None:
        """
        Test :func:`colour.adaptation.li2025.chromatic_adaptation_Li2025`
        definition domain and range scale support.
        """

        XYZ_s = np.array([48.900, 43.620, 6.250])
        XYZ_ws = np.array([109.850, 100, 35.585])
        XYZ_wd = np.array([95.047, 100, 108.883])
        L_A = 318.31
        F_surround = 1.0
        XYZ_d = chromatic_adaptation_Li2025(XYZ_s, XYZ_ws, XYZ_wd, L_A, F_surround)

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    chromatic_adaptation_Li2025(
                        XYZ_s * factor,
                        XYZ_ws * factor,
                        XYZ_wd * factor,
                        L_A,
                        F_surround,
                    ),
                    XYZ_d * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_chromatic_adaptation_Li2025(self) -> None:
        """
        Test :func:`colour.adaptation.li2025.chromatic_adaptation_Li2025`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        chromatic_adaptation_Li2025(cases, cases, cases, cases[0, 0], cases[0, 0])
