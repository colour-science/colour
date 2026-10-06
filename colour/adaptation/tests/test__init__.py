"""Define the unit tests for the :mod:`colour.adaptation` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from colour.adaptation import chromatic_adaptation
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.utilities import (
    as_ndarray,
    domain_range_scale,
    xp_as_array,
    xp_assert_close,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestChromaticAdaptation",
]


class TestChromaticAdaptation:
    """
    Define :func:`colour.adaptation.chromatic_adaptation` definition unit
    tests methods.
    """

    def test_chromatic_adaptation(self, xp: ModuleType) -> None:
        """Test :func:`colour.adaptation.chromatic_adaptation` definition."""

        XYZ = xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)
        XYZ_w = xp_as_array([0.95045593, 1.00000000, 1.08905775], xp=xp)
        XYZ_wr = xp_as_array([0.96429568, 1.00000000, 0.82510460], xp=xp)
        xp_assert_close(
            chromatic_adaptation(XYZ, XYZ_w, XYZ_wr),
            [0.21638819, 0.12570000, 0.03847494],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        Y_o = 0.2
        E_o = 1000
        xp_assert_close(
            chromatic_adaptation(
                XYZ,
                XYZ_w,
                XYZ_wr,
                method="CIE 1994",
                Y_o=Y_o,
                E_o1=E_o,
                E_o2=E_o,
            ),
            [0.21347453, 0.12252986, 0.03347887],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        L_A = 200
        xp_assert_close(
            chromatic_adaptation(
                XYZ, XYZ_w, XYZ_wr, method="CMCCAT2000", L_A1=L_A, L_A2=L_A
            ),
            [0.21498829, 0.12474711, 0.03910138],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        Y_n = 200
        xp_assert_close(
            chromatic_adaptation(XYZ, XYZ_w, XYZ_wr, method="Fairchild 1990", Y_n=Y_n),
            [0.21394049, 0.12262315, 0.03891917],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            chromatic_adaptation(
                XYZ, XYZ_w, XYZ_wr, method="Li 2025", L_A=100, F_surround=1
            ),
            [0.21166965, 0.12234633, 0.03888754],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            chromatic_adaptation(XYZ, XYZ_w, XYZ_wr, method="Zhai 2018", L_A=100),
            [0.21638819, 0.1257, 0.03847494],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ_wo = xp_as_array([1.0, 1.0, 1.0], xp=xp)
        xp_assert_close(
            chromatic_adaptation(
                XYZ,
                XYZ_w,
                XYZ_wr,
                method="Zhai 2018",
                L_A=100,
                XYZ_wo=XYZ_wo,
            ),
            [0.21638819, 0.1257, 0.03847494],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            chromatic_adaptation(XYZ, XYZ_w, XYZ_wr, method="vK20"),
            [0.21468842, 0.12456164, 0.04662558],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_chromatic_adaptation(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.adaptation.chromatic_adaptation` definition domain
        and range scale support.
        """

        XYZ = xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)
        XYZ_w = xp_as_array([0.95045593, 1.00000000, 1.08905775], xp=xp)
        XYZ_wr = xp_as_array([0.96429568, 1.00000000, 0.82510460], xp=xp)
        Y_o = 0.2
        E_o = 1000
        L_A = 200
        Y_n = 200

        m = ("Von Kries", "CIE 1994", "CMCCAT2000", "Fairchild 1990")
        v = [
            as_ndarray(
                chromatic_adaptation(
                    XYZ,
                    XYZ_w,
                    XYZ_wr,
                    method=method,
                    Y_o=Y_o,
                    E_o1=E_o,
                    E_o2=E_o,
                    L_A1=L_A,
                    L_A2=L_A,
                    Y_n=Y_n,
                )
            )
            for method in m
        ]

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for method, value in zip(m, v, strict=True):
            for scale, factor in d_r:
                with domain_range_scale(scale):
                    xp_assert_close(
                        chromatic_adaptation(
                            XYZ * factor,
                            XYZ_w * factor,
                            XYZ_wr * factor,
                            method=method,
                            Y_o=Y_o * factor,
                            E_o1=E_o,
                            E_o2=E_o,
                            L_A1=L_A,
                            L_A2=L_A,
                            Y_n=Y_n,
                        ),
                        value * factor,
                        atol=TOLERANCE_ABSOLUTE_TESTS,
                    )
