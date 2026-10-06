"""Define the unit tests for the :mod:`colour.models.prolab` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np
import pytest

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models import ProLab_to_XYZ, XYZ_to_ProLab
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
    "TestXYZ_to_ProLab",
    "TestProLab_to_XYZ",
]


class TestXYZ_to_ProLab:
    """
    Define :func:`colour.models.ProLab.TestXYZ_to_ProLab` definition unit
    tests methods.
    """

    def test_XYZ_to_ProLab(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.ProLab.XYZ_to_ProLab` definition."""

        xp_assert_close(
            XYZ_to_ProLab(xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)),
            [48.7948929, 35.31503175, 13.30044932],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_ProLab(xp_as_array([0.14222010, 0.23042768, 0.10495772], xp=xp)),
            [64.45929636, -21.67007419, 13.25749056],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_ProLab(xp_as_array([0.96907232, 1.00000000, 0.12179215], xp=xp)),
            [100.0, 5.47367608, 37.26313098],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_XYZ_to_ProLab(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.prolab.XYZ_to_ProLab` definition
        n-dimensional support.
        """

        XYZ = xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)
        ProLab = XYZ_to_ProLab(XYZ)

        XYZ = xp.tile(XYZ, (6, 1))
        ProLab = xp.tile(ProLab, (6, 1))
        xp_assert_close(XYZ_to_ProLab(XYZ), ProLab, atol=TOLERANCE_ABSOLUTE_TESTS)

        XYZ = xp_reshape(XYZ, (2, 3, 3), xp=xp)
        ProLab = xp_reshape(ProLab, (2, 3, 3), xp=xp)
        xp_assert_close(XYZ_to_ProLab(XYZ), ProLab, atol=TOLERANCE_ABSOLUTE_TESTS)

    @pytest.mark.mps_tolerance_absolute(5e-3)
    def test_domain_range_scale_XYZ_to_ProLab(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.prolab.XYZ_to_ProLab` definition domain and
        range scale support.
        """

        XYZ = xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)
        ProLab = XYZ_to_ProLab(XYZ)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    XYZ_to_ProLab(XYZ * factor),
                    ProLab * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_ProLab(self) -> None:
        """
        Test :func:`colour.models.ProLab.XYZ_to_ProLab` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        XYZ_to_ProLab(cases)


class TestProLab_to_XYZ:
    """
    Define :func:`colour.models.ProLab.ProLab_to_XYZ` definition unit tests
    methods.
    """

    def test_ProLab_to_XYZ(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.ProLab.ProLab_to_XYZ` definition."""

        xp_assert_close(
            ProLab_to_XYZ(xp_as_array([48.7948929, 35.31503175, 13.30044932], xp=xp)),
            [0.20654008, 0.12197225, 0.05136952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ProLab_to_XYZ(xp_as_array([64.45929636, -21.67007419, 13.25749056], xp=xp)),
            [0.14222010, 0.23042768, 0.10495772],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ProLab_to_XYZ(xp_as_array([100.0, 5.47367608, 37.26313098], xp=xp)),
            [0.96907232, 1.00000000, 0.12179215],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_XYZ_to_ProLab(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.prolab.XYZ_to_ProLab` definition
        n-dimensional support.
        """

        ProLab = xp_as_array([48.7948929, 35.31503175, 13.30044932], xp=xp)
        XYZ = ProLab_to_XYZ(ProLab)

        ProLab = xp.tile(ProLab, (6, 1))
        XYZ = xp.tile(XYZ, (6, 1))
        xp_assert_close(ProLab_to_XYZ(ProLab), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS)

        ProLab = xp_reshape(ProLab, (2, 3, 3), xp=xp)
        XYZ = xp_reshape(XYZ, (2, 3, 3), xp=xp)
        xp_assert_close(ProLab_to_XYZ(ProLab), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_XYZ_to_ProLab(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.prolab.XYZ_to_ProLab` definition domain and
        range scale support.
        """

        ProLab = xp_as_array([48.7948929, 35.31503175, 13.30044932], xp=xp)
        XYZ = as_ndarray(XYZ_to_ProLab(ProLab))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    XYZ_to_ProLab(ProLab * factor),
                    XYZ * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_ProLab_to_XYZ(self) -> None:
        """
        Test :func:`colour.models.ProLab.ProLab_to_XYZ` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        ProLab_to_XYZ(cases)
