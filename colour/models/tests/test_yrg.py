"""Define the unit tests for the :mod:`colour.models.yrg` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models import LMS_to_Yrg, XYZ_to_Yrg, Yrg_to_LMS, Yrg_to_XYZ
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
    "TestLMS_to_Yrg",
    "TestYrg_to_LMS",
    "TestXYZ_to_Yrg",
    "TestYrg_to_XYZ",
]


class TestLMS_to_Yrg:
    """
    Define :func:`colour.models.yrg.TestLMS_to_Yrg` definition unit tests
    methods.
    """

    def test_LMS_to_Yrg(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.yrg.LMS_to_Yrg` definition."""

        xp_assert_close(
            LMS_to_Yrg(xp_as_array([0.15639195, 0.06741689, 0.03281398], xp=xp)),
            [0.13137801, 0.49037644, 0.37777391],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            LMS_to_Yrg(xp_as_array([0.23145723, 0.22601133, 0.05033211], xp=xp)),
            [0.23840767, 0.20110504, 0.69668437],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            LMS_to_Yrg(xp_as_array([1.07423297, 0.91295620, 0.61375713], xp=xp)),
            [1.05911888, 0.22010094, 0.53660290],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_LMS_to_Yrg(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.yrg.LMS_to_Yrg` definition n-dimensional
        support.
        """

        LMS = xp_as_array([0.15639195, 0.06741689, 0.03281398], xp=xp)
        Yrg = as_ndarray(LMS_to_Yrg(LMS))

        LMS = xp.tile(xp_as_array(LMS, xp=xp), (6, 1))
        Yrg = xp.tile(xp_as_array(Yrg, xp=xp), (6, 1))
        xp_assert_close(LMS_to_Yrg(LMS), Yrg, atol=TOLERANCE_ABSOLUTE_TESTS)

        LMS = xp_reshape(xp_as_array(LMS, xp=xp), (2, 3, 3), xp=xp)
        Yrg = xp_reshape(xp_as_array(Yrg, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(LMS_to_Yrg(LMS), Yrg, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_LMS_to_Yrg(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.yrg.LMS_to_Yrg` definition domain and range
        scale support.
        """

        LMS = xp_as_array([0.15639195, 0.06741689, 0.03281398], xp=xp)
        Yrg = as_ndarray(LMS_to_Yrg(LMS))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    LMS_to_Yrg(LMS * factor),
                    Yrg * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_LMS_to_Yrg(self) -> None:
        """Test :func:`colour.models.yrg.LMS_to_Yrg` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        LMS_to_Yrg(cases)


class TestYrg_to_LMS:
    """
    Define :func:`colour.models.yrg.Yrg_to_LMS` definition unit tests methods.
    """

    def test_Yrg_to_LMS(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.yrg.Yrg_to_LMS` definition."""

        xp_assert_close(
            Yrg_to_LMS(xp_as_array([0.13137801, 0.49037644, 0.37777391], xp=xp)),
            [0.15639195, 0.06741689, 0.03281398],
            atol=TOLERANCE_ABSOLUTE_TESTS * 1000,
        )

        xp_assert_close(
            Yrg_to_LMS(xp_as_array([0.23840767, 0.20110504, 0.69668437], xp=xp)),
            [0.23145723, 0.22601133, 0.05033211],
            atol=TOLERANCE_ABSOLUTE_TESTS * 1000,
        )

        xp_assert_close(
            Yrg_to_LMS(xp_as_array([1.05911888, 0.22010094, 0.53660290], xp=xp)),
            [1.07423297, 0.91295620, 0.61375713],
            atol=TOLERANCE_ABSOLUTE_TESTS * 1000,
        )

    def test_n_dimensional_Yrg_to_LMS(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.yrg.Yrg_to_LMS` definition n-dimensional
        support.
        """

        Yrg = xp_as_array([0.00535048, 0.00924302, 0.00526007], xp=xp)
        LMS = as_ndarray(Yrg_to_LMS(Yrg))

        Yrg = xp.tile(xp_as_array(Yrg, xp=xp), (6, 1))
        LMS = xp.tile(xp_as_array(LMS, xp=xp), (6, 1))
        xp_assert_close(Yrg_to_LMS(Yrg), LMS, atol=TOLERANCE_ABSOLUTE_TESTS)

        Yrg = xp_reshape(xp_as_array(Yrg, xp=xp), (2, 3, 3), xp=xp)
        LMS = xp_reshape(xp_as_array(LMS, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(Yrg_to_LMS(Yrg), LMS, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_Yrg_to_LMS(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.yrg.Yrg_to_LMS` definition domain and range
        scale support.
        """

        Yrg = xp_as_array([0.00535048, 0.00924302, 0.00526007], xp=xp)
        LMS = as_ndarray(Yrg_to_LMS(Yrg))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    Yrg_to_LMS(Yrg * factor),
                    LMS * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_Yrg_to_LMS(self) -> None:
        """Test :func:`colour.models.yrg.Yrg_to_LMS` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        Yrg_to_LMS(cases)


class TestXYZ_to_Yrg:
    """
    Define :func:`colour.models.yrg.TestXYZ_to_Yrg` definition unit tests
    methods.
    """

    def test_XYZ_to_Yrg(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.yrg.XYZ_to_Yrg` definition."""

        xp_assert_close(
            XYZ_to_Yrg(xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)),
            [0.13137801, 0.49037645, 0.37777388],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_Yrg(xp_as_array([0.14222010, 0.23042768, 0.10495772], xp=xp)),
            [0.23840767, 0.20110503, 0.69668437],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_Yrg(xp_as_array([0.96907232, 1.00000000, 1.12179215], xp=xp)),
            [1.05911888, 0.22010094, 0.53660290],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_XYZ_to_Yrg(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.yrg.XYZ_to_Yrg` definition n-dimensional
        support.
        """

        XYZ = xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)
        Yrg = as_ndarray(XYZ_to_Yrg(XYZ))

        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        Yrg = xp.tile(xp_as_array(Yrg, xp=xp), (6, 1))
        xp_assert_close(XYZ_to_Yrg(XYZ), Yrg, atol=TOLERANCE_ABSOLUTE_TESTS)

        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        Yrg = xp_reshape(xp_as_array(Yrg, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(XYZ_to_Yrg(XYZ), Yrg, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_XYZ_to_Yrg(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.yrg.XYZ_to_Yrg` definition domain and range
        scale support.
        """

        XYZ = xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)
        Yrg = as_ndarray(XYZ_to_Yrg(XYZ))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    XYZ_to_Yrg(XYZ * factor),
                    Yrg * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_Yrg(self) -> None:
        """Test :func:`colour.models.yrg.XYZ_to_Yrg` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        XYZ_to_Yrg(cases)


class TestYrg_to_XYZ:
    """
    Define :func:`colour.models.yrg.Yrg_to_XYZ` definition unit tests methods.
    """

    def test_Yrg_to_XYZ(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.yrg.Yrg_to_XYZ` definition."""

        xp_assert_close(
            Yrg_to_XYZ(xp_as_array([0.13137801, 0.49037645, 0.37777388], xp=xp)),
            [0.20654008, 0.12197225, 0.05136952],
            atol=TOLERANCE_ABSOLUTE_TESTS * 1000,
        )

        xp_assert_close(
            Yrg_to_XYZ(xp_as_array([0.23840767, 0.20110503, 0.69668437], xp=xp)),
            [0.14222010, 0.23042768, 0.10495772],
            atol=TOLERANCE_ABSOLUTE_TESTS * 2000,
        )

        xp_assert_close(
            Yrg_to_XYZ(xp_as_array([1.05911888, 0.22010094, 0.53660290], xp=xp)),
            [0.96907232, 1.00000000, 1.12179215],
            atol=TOLERANCE_ABSOLUTE_TESTS * 2000,
        )

    def test_n_dimensional_Yrg_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.yrg.Yrg_to_XYZ` definition n-dimensional
        support.
        """

        Yrg = xp_as_array([0.13137801, 0.49037645, 0.37777388], xp=xp)
        XYZ = as_ndarray(Yrg_to_XYZ(Yrg))

        Yrg = xp.tile(xp_as_array(Yrg, xp=xp), (6, 1))
        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        xp_assert_close(Yrg_to_XYZ(Yrg), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS)

        Yrg = xp_reshape(xp_as_array(Yrg, xp=xp), (2, 3, 3), xp=xp)
        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(Yrg_to_XYZ(Yrg), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_Yrg_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.yrg.Yrg_to_XYZ` definition domain and range
        scale support.
        """

        Yrg = xp_as_array([0.13137801, 0.49037645, 0.37777388], xp=xp)
        XYZ = as_ndarray(Yrg_to_XYZ(Yrg))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    Yrg_to_XYZ(Yrg * factor),
                    XYZ * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS * 1000,
                )

    @ignore_numpy_errors
    def test_nan_Yrg_to_XYZ(self) -> None:
        """Test :func:`colour.models.yrg.Yrg_to_XYZ` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        Yrg_to_XYZ(cases)
