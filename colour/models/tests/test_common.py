"""Define the unit tests for the :mod:`colour.models.common` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models import Iab_to_XYZ, Jab_to_JCh, JCh_to_Jab, XYZ_to_Iab
from colour.utilities import (
    as_ndarray,
    domain_range_scale,
    ignore_numpy_errors,
    xp_as_array,
    xp_assert_close,
    xp_reshape,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestJab_to_JCh",
    "TestJCh_to_Jab",
    "TestXYZ_to_Iab",
    "TestIab_to_XYZ",
]


class TestJab_to_JCh:
    """
    Define :func:`colour.models.common.Jab_to_JCh` definition unit tests
    methods.
    """

    def test_Jab_to_JCh(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.common.Jab_to_JCh` definition."""

        xp_assert_close(
            Jab_to_JCh(xp_as_array([41.52787529, 52.63858304, 26.92317922], xp=xp)),
            [41.52787529, 59.12425901, 27.08848784],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            Jab_to_JCh(xp_as_array([55.11636304, -41.08791787, 30.91825778], xp=xp)),
            [55.11636304, 51.42135412, 143.03889556],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            Jab_to_JCh(xp_as_array([29.80565520, 20.01830466, -48.34913874], xp=xp)),
            [29.80565520, 52.32945383, 292.49133666],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_Jab_to_JCh(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.common.Jab_to_JCh` definition n-dimensional
        arrays support.
        """

        Lab = xp_as_array([41.52787529, 52.63858304, 26.92317922], xp=xp)
        LCHab = as_ndarray(Jab_to_JCh(Lab))

        Lab = xp.tile(xp_as_array(Lab, xp=xp), (6, 1))
        LCHab = xp.tile(xp_as_array(LCHab, xp=xp), (6, 1))
        xp_assert_close(Jab_to_JCh(Lab), LCHab, atol=TOLERANCE_ABSOLUTE_TESTS)

        Lab = xp_reshape(xp_as_array(Lab, xp=xp), (2, 3, 3), xp=xp)
        LCHab = xp_reshape(xp_as_array(LCHab, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(Jab_to_JCh(Lab), LCHab, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_Jab_to_JCh(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.common.Jab_to_JCh` definition domain and
        range scale support.
        """

        Lab = xp_as_array([41.52787529, 52.63858304, 26.92317922], xp=xp)
        LCHab = as_ndarray(Jab_to_JCh(Lab))

        d_r = (
            ("reference", 1, 1),
            ("1", 0.01, np.array([0.01, 0.01, 1 / 360])),
            ("100", 1, np.array([1, 1, 1 / 3.6])),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    Jab_to_JCh(Lab * xp_as_array(factor_a, xp=xp)),
                    LCHab * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_Jab_to_JCh(self) -> None:
        """Test :func:`colour.models.common.Jab_to_JCh` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        Jab_to_JCh(cases)


class TestJCh_to_Jab:
    """
    Define :func:`colour.models.common.JCh_to_Jab` definition unit tests
    methods.
    """

    def test_JCh_to_Jab(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.common.JCh_to_Jab` definition."""

        xp_assert_close(
            JCh_to_Jab(xp_as_array([41.52787529, 59.12425901, 27.08848784], xp=xp)),
            [41.52787529, 52.63858304, 26.92317922],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            JCh_to_Jab(xp_as_array([55.11636304, 51.42135412, 143.03889556], xp=xp)),
            [55.11636304, -41.08791787, 30.91825778],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            JCh_to_Jab(xp_as_array([29.80565520, 52.32945383, 292.49133666], xp=xp)),
            [29.80565520, 20.01830466, -48.34913874],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_JCh_to_Jab(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.common.JCh_to_Jab` definition n-dimensional
        arrays support.
        """

        LCHab = xp_as_array([41.52787529, 59.12425901, 27.08848784], xp=xp)
        Lab = as_ndarray(JCh_to_Jab(LCHab))

        LCHab = xp.tile(xp_as_array(LCHab, xp=xp), (6, 1))
        Lab = xp.tile(xp_as_array(Lab, xp=xp), (6, 1))
        xp_assert_close(JCh_to_Jab(LCHab), Lab, atol=TOLERANCE_ABSOLUTE_TESTS)

        LCHab = xp_reshape(xp_as_array(LCHab, xp=xp), (2, 3, 3), xp=xp)
        Lab = xp_reshape(xp_as_array(Lab, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(JCh_to_Jab(LCHab), Lab, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_JCh_to_Jab(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.common.JCh_to_Jab` definition domain and
        range scale support.
        """

        LCHab = xp_as_array([41.52787529, 59.12425901, 27.08848784], xp=xp)
        Lab = as_ndarray(JCh_to_Jab(LCHab))

        d_r = (
            ("reference", 1, 1),
            ("1", np.array([0.01, 0.01, 1 / 360]), 0.01),
            ("100", np.array([1, 1, 1 / 3.6]), 1),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    JCh_to_Jab(LCHab * xp_as_array(factor_a, xp=xp)),
                    Lab * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_JCh_to_Jab(self) -> None:
        """Test :func:`colour.models.common.JCh_to_Jab` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        JCh_to_Jab(cases)


class TestXYZ_to_Iab:
    """Define :func:`colour.models.common.XYZ_to_Iab` definition unit tests methods."""

    def setup_method(self) -> None:
        """Initialise the common tests attributes."""

        self.LMS_to_LMS_p = lambda x: x**0.43
        self.M_XYZ_to_LMS = np.array(
            [
                [0.4002, 0.7075, -0.0807],
                [-0.2280, 1.1500, 0.0612],
                [0.0000, 0.0000, 0.9184],
            ]
        )
        self.M_LMS_p_to_Iab = np.array(
            [
                [0.4000, 0.4000, 0.2000],
                [4.4550, -4.8510, 0.3960],
                [0.8056, 0.3572, -1.1628],
            ]
        )

    def test_XYZ_to_Iab(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.common.XYZ_to_Iab` definition."""

        xp_assert_close(
            XYZ_to_Iab(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp),
                self.LMS_to_LMS_p,
                self.M_XYZ_to_LMS,
                self.M_LMS_p_to_Iab,
            ),
            [0.38426191, 0.38487306, 0.18886838],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_Iab(
                xp_as_array([0.14222010, 0.23042768, 0.10495772], xp=xp),
                self.LMS_to_LMS_p,
                self.M_XYZ_to_LMS,
                self.M_LMS_p_to_Iab,
            ),
            [0.49437481, -0.19251742, 0.18080304],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_Iab(
                xp_as_array([0.07818780, 0.06157201, 0.28099326], xp=xp),
                self.LMS_to_LMS_p,
                self.M_XYZ_to_LMS,
                self.M_LMS_p_to_Iab,
            ),
            [0.35167774, -0.07525627, -0.30921279],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_XYZ_to_Iab(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.common.XYZ_to_Iab` definition n-dimensional
        support.
        """

        XYZ = xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)
        Iab = as_ndarray(
            XYZ_to_Iab(XYZ, self.LMS_to_LMS_p, self.M_XYZ_to_LMS, self.M_LMS_p_to_Iab)
        )

        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        Iab = xp.tile(xp_as_array(Iab, xp=xp), (6, 1))
        xp_assert_close(
            XYZ_to_Iab(XYZ, self.LMS_to_LMS_p, self.M_XYZ_to_LMS, self.M_LMS_p_to_Iab),
            Iab,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        Iab = xp_reshape(xp_as_array(Iab, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(
            XYZ_to_Iab(XYZ, self.LMS_to_LMS_p, self.M_XYZ_to_LMS, self.M_LMS_p_to_Iab),
            Iab,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_XYZ_to_Iab(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.common.XYZ_to_Iab` definition domain and
        range scale support.
        """

        XYZ = xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)
        Iab = as_ndarray(
            XYZ_to_Iab(XYZ, self.LMS_to_LMS_p, self.M_XYZ_to_LMS, self.M_LMS_p_to_Iab)
        )

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    XYZ_to_Iab(
                        XYZ * factor,
                        self.LMS_to_LMS_p,
                        self.M_XYZ_to_LMS,
                        self.M_LMS_p_to_Iab,
                    ),
                    Iab * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_Iab(self) -> None:
        """Test :func:`colour.models.common.XYZ_to_Iab` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        XYZ_to_Iab(cases, self.LMS_to_LMS_p, self.M_XYZ_to_LMS, self.M_LMS_p_to_Iab)


class TestIab_to_XYZ:
    """
    Define :func:`colour.models.common.Iab_to_XYZ` definition unit tests
    methods.
    """

    def setup_method(self) -> None:
        """Initialise the common tests attributes."""

        self.LMS_p_to_LMS = lambda x: x ** (1 / 0.43)
        self.M_Iab_to_LMS_p = np.linalg.inv(
            np.array(
                [
                    [0.4000, 0.4000, 0.2000],
                    [4.4550, -4.8510, 0.3960],
                    [0.8056, 0.3572, -1.1628],
                ]
            )
        )
        self.M_LMS_to_XYZ = np.linalg.inv(
            np.array(
                [
                    [0.4002, 0.7075, -0.0807],
                    [-0.2280, 1.1500, 0.0612],
                    [0.0000, 0.0000, 0.9184],
                ]
            )
        )

    def test_Iab_to_XYZ(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.common.Iab_to_XYZ` definition."""

        xp_assert_close(
            Iab_to_XYZ(
                xp_as_array([0.38426191, 0.38487306, 0.18886838], xp=xp),
                self.LMS_p_to_LMS,
                self.M_Iab_to_LMS_p,
                self.M_LMS_to_XYZ,
            ),
            [0.20654008, 0.12197225, 0.05136952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            Iab_to_XYZ(
                xp_as_array([0.49437481, -0.19251742, 0.18080304], xp=xp),
                self.LMS_p_to_LMS,
                self.M_Iab_to_LMS_p,
                self.M_LMS_to_XYZ,
            ),
            [0.14222010, 0.23042768, 0.10495772],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            Iab_to_XYZ(
                xp_as_array([0.35167774, -0.07525627, -0.30921279], xp=xp),
                self.LMS_p_to_LMS,
                self.M_Iab_to_LMS_p,
                self.M_LMS_to_XYZ,
            ),
            [0.07818780, 0.06157201, 0.28099326],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_Iab_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.common.Iab_to_XYZ` definition n-dimensional
        support.
        """

        Iab = xp_as_array([0.38426191, 0.38487306, 0.18886838], xp=xp)
        XYZ = as_ndarray(
            Iab_to_XYZ(Iab, self.LMS_p_to_LMS, self.M_Iab_to_LMS_p, self.M_LMS_to_XYZ)
        )

        Iab = xp.tile(xp_as_array(Iab, xp=xp), (6, 1))
        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        xp_assert_close(
            Iab_to_XYZ(Iab, self.LMS_p_to_LMS, self.M_Iab_to_LMS_p, self.M_LMS_to_XYZ),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        Iab = xp_reshape(xp_as_array(Iab, xp=xp), (2, 3, 3), xp=xp)
        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(
            Iab_to_XYZ(Iab, self.LMS_p_to_LMS, self.M_Iab_to_LMS_p, self.M_LMS_to_XYZ),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_Iab_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.common.Iab_to_XYZ` definition domain and
        range scale support.
        """

        Iab = xp_as_array([0.38426191, 0.38487306, 0.18886838], xp=xp)
        XYZ = as_ndarray(
            Iab_to_XYZ(Iab, self.LMS_p_to_LMS, self.M_Iab_to_LMS_p, self.M_LMS_to_XYZ)
        )

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    Iab_to_XYZ(
                        Iab * factor,
                        self.LMS_p_to_LMS,
                        self.M_Iab_to_LMS_p,
                        self.M_LMS_to_XYZ,
                    ),
                    XYZ * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_Iab_to_XYZ(self) -> None:
        """Test :func:`colour.models.common.Iab_to_XYZ` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        Iab_to_XYZ(cases, self.LMS_p_to_LMS, self.M_Iab_to_LMS_p, self.M_LMS_to_XYZ)
