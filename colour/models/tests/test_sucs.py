"""Define the unit tests for the :mod:`colour.models.sucs` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models import (
    XYZ_to_sUCS,
    sUCS_chroma,
    sUCS_hue_angle,
    sUCS_Iab_to_sUCS_ICh,
    sUCS_ICh_to_sUCS_Iab,
    sUCS_to_XYZ,
)
from colour.utilities import (
    as_ndarray,
    domain_range_scale,
    ignore_numpy_errors,
    xp_as_array,
    xp_assert_close,
    xp_reshape,
)

__author__ = "UltraMo114(Molin Li), Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestXYZ_to_sUCS",
    "TestsUCS_to_XYZ",
    "TestsUCSChroma",
    "TestsUCSHueAngle",
    "TestsUCS_Iab_to_sUCS_ICh",
    "TestsUCS_ICh_to_sUCS_Iab",
]


class TestXYZ_to_sUCS:
    """Define :func:`colour.models.sucs.XYZ_to_sUCS` definition unit tests methods."""

    def test_XYZ_to_sUCS(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.sucs.XYZ_to_sUCS` definition."""

        xp_assert_close(
            XYZ_to_sUCS(xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)),
            [42.62923653, 36.97646831, 14.12301358],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_sUCS(xp_as_array([0.14222010, 0.23042768, 0.10495772], xp=xp)),
            [51.93649255, -18.89245582, 15.76112395],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_sUCS(xp_as_array([0.07818780, 0.06157201, 0.28099326], xp=xp)),
            [29.79456846, -6.83806757, -25.33884097],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_XYZ_to_sUCS(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.sucs.XYZ_to_sUCS` definition n-dimensional
        support.
        """

        XYZ = xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)
        Iab = as_ndarray(XYZ_to_sUCS(XYZ))

        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        Iab = xp.tile(xp_as_array(Iab, xp=xp), (6, 1))
        xp_assert_close(XYZ_to_sUCS(XYZ), Iab, atol=TOLERANCE_ABSOLUTE_TESTS)

        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        Iab = xp_reshape(xp_as_array(Iab, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(XYZ_to_sUCS(XYZ), Iab, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_XYZ_to_sUCS(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.sucs.XYZ_to_sUCS` definition domain and
        range scale support.
        """

        XYZ = xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)
        Iab = as_ndarray(XYZ_to_sUCS(XYZ))

        d_r = (("reference", 1, 1), ("1", 1, 0.01), ("100", 100, 1))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    XYZ_to_sUCS(XYZ * xp_as_array(factor_a, xp=xp)),
                    Iab * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_sUCS(self) -> None:
        """Test :func:`colour.models.sucs.XYZ_to_sUCS` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        XYZ_to_sUCS(cases)


class TestsUCS_to_XYZ:
    """
    Define :func:`colour.models.sucs.sUCS_to_XYZ` definition unit tests
    methods.
    """

    def test_sUCS_to_XYZ(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.sucs.sUCS_to_XYZ` definition."""

        xp_assert_close(
            sUCS_to_XYZ(xp_as_array([42.62923653, 36.97646831, 14.12301358], xp=xp)),
            [0.20654008, 0.12197225, 0.05136952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            sUCS_to_XYZ(xp_as_array([51.93649255, -18.89245582, 15.76112395], xp=xp)),
            [0.14222010, 0.23042768, 0.10495772],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            sUCS_to_XYZ(xp_as_array([29.79456846, -6.83806757, -25.33884097], xp=xp)),
            [0.07818780, 0.06157201, 0.28099326],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_sUCS_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.sucs.sUCS_to_XYZ` definition n-dimensional
        support.
        """

        Iab = xp_as_array([42.62923653, 36.97646831, 14.12301358], xp=xp)
        XYZ = as_ndarray(sUCS_to_XYZ(Iab))

        Iab = xp.tile(xp_as_array(Iab, xp=xp), (6, 1))
        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        xp_assert_close(sUCS_to_XYZ(Iab), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS)

        Iab = xp_reshape(xp_as_array(Iab, xp=xp), (2, 3, 3), xp=xp)
        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(sUCS_to_XYZ(Iab), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_sUCS_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.sucs.sUCS_to_XYZ` definition domain and
        range scale support.
        """

        Iab = xp_as_array([42.62923653, 36.97646831, 14.12301358], xp=xp)
        XYZ = as_ndarray(sUCS_to_XYZ(Iab))

        d_r = (("reference", 1, 1), ("1", 0.01, 1), ("100", 1, 100))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    sUCS_to_XYZ(Iab * xp_as_array(factor_a, xp=xp)),
                    XYZ * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_sUCS_to_XYZ(self) -> None:
        """Test :func:`colour.models.sucs.sUCS_to_XYZ` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        sUCS_to_XYZ(cases)


class TestsUCSChroma:
    """
    Define :func:`colour.models.sucs.sUCS_chroma` definition unit tests
    methods.
    """

    def test_sUCS_hue_angle(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.sucs.sUCS_chroma` definition."""

        xp_assert_close(
            sUCS_chroma(xp_as_array([42.62923653, 36.97646831, 14.12301358], xp=xp)),
            40.420511061137226,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            sUCS_chroma(xp_as_array([51.93649255, -18.89245582, 15.76112395], xp=xp)),
            29.437831501432590,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            sUCS_chroma(xp_as_array([29.79456846, -6.83806757, -25.33884097], xp=xp)),
            30.800979756091614,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_sUCS_hue_angle(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.sucs.sUCS_chroma` definition n-dimensional
        support.
        """

        Iab = xp_as_array([42.62923653, 36.97646831, 14.12301358], xp=xp)
        C = sUCS_chroma(Iab)

        Iab = xp.tile(xp_as_array(Iab, xp=xp), (6, 1))
        C = xp.tile(xp_as_array(C, xp=xp), (6,))
        xp_assert_close(sUCS_chroma(Iab), C, atol=TOLERANCE_ABSOLUTE_TESTS)

        Iab = xp_reshape(xp_as_array(Iab, xp=xp), (2, 3, 3), xp=xp)
        C = xp_reshape(xp_as_array(C, xp=xp), (2, 3), xp=xp)
        xp_assert_close(sUCS_chroma(Iab), C, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_sUCS_chroma(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.sucs.sUCS_chroma` definition domain and
        range scale support.
        """

        Iab = xp_as_array([42.62923653, 36.97646831, 14.12301358], xp=xp)
        C = as_ndarray(sUCS_chroma(Iab))

        d_r = (("reference", 1, 1), ("1", 0.01, 0.01), ("100", 1, 1))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    sUCS_chroma(Iab * xp_as_array(factor_a, xp=xp)),
                    C * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_sUCS_hue_angle(self) -> None:
        """Test :func:`colour.models.sucs.sUCS_chroma` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        sUCS_chroma(cases)


class TestsUCSHueAngle:
    """
    Define :func:`colour.models.sucs.sUCS_hue_angle` definition unit tests
    methods.
    """

    def test_sUCS_hue_angle(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.sucs.sUCS_hue_angle` definition."""

        xp_assert_close(
            sUCS_hue_angle(xp_as_array([42.62923653, 36.97646831, 14.12301358], xp=xp)),
            20.904156072136217,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            sUCS_hue_angle(
                xp_as_array([51.93649255, -18.89245582, 15.76112395], xp=xp)
            ),
            140.163281067124470,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            sUCS_hue_angle(
                xp_as_array([29.79456846, -6.83806757, -25.33884097], xp=xp)
            ),
            254.897631851863850,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_sUCS_hue_angle(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.sucs.sUCS_hue_angle` definition n-dimensional
        support.
        """

        Iab = xp_as_array([42.62923653, 36.97646831, 14.12301358], xp=xp)
        hue = as_ndarray(sUCS_hue_angle(Iab))

        Iab = xp.tile(xp_as_array(Iab, xp=xp), (6, 1))
        hue = xp.tile(xp_as_array(hue, xp=xp), (6,))
        xp_assert_close(sUCS_hue_angle(Iab), hue, atol=TOLERANCE_ABSOLUTE_TESTS)

        Iab = xp_reshape(xp_as_array(Iab, xp=xp), (2, 3, 3), xp=xp)
        hue = xp_reshape(xp_as_array(hue, xp=xp), (2, 3), xp=xp)
        xp_assert_close(sUCS_hue_angle(Iab), hue, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_sUCS_hue_angle(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.sucs.sUCS_hue_angle` definition domain and
        range scale support.
        """

        Iab = xp_as_array([42.62923653, 36.97646831, 14.12301358], xp=xp)
        hue = as_ndarray(sUCS_hue_angle(Iab))

        d_r = (("reference", 1, 1), ("1", 1, 1 / 360), ("100", 100, 1 / 3.6))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    sUCS_hue_angle(Iab * xp_as_array(factor_a, xp=xp)),
                    hue * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_sUCS_hue_angle(self) -> None:
        """Test :func:`colour.models.sucs.sUCS_hue_angle` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        sUCS_hue_angle(cases)


class TestsUCS_Iab_to_sUCS_ICh:
    """
    Define :func:`colour.models.sucs.sUCS_Iab_to_sUCS_ICh` definition unit tests
    methods.
    """

    def test_sUCS_Iab_to_sUCS_ICh(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.sucs.sUCS_Iab_to_sUCS_ICh` definition."""

        xp_assert_close(
            sUCS_Iab_to_sUCS_ICh(
                xp_as_array([42.62923653, 36.97646831, 14.12301358], xp=xp)
            ),
            [42.62923653, 40.42051106, 20.90415607],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            sUCS_Iab_to_sUCS_ICh(
                xp_as_array([51.93649255, -18.89245582, 15.76112395], xp=xp)
            ),
            [51.93649255, 29.43783150, 140.16328107],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            sUCS_Iab_to_sUCS_ICh(
                xp_as_array([29.79456846, -6.83806757, -25.33884097], xp=xp)
            ),
            [29.79456846, 30.80097976, 254.89763185],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_sUCS_Iab_to_sUCS_ICh(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.sucs.sUCS_Iab_to_sUCS_ICh` definition
        n-dimensional support.
        """

        Iab = xp_as_array([42.62923653, 36.97646831, 14.12301358], xp=xp)
        ICh = as_ndarray(sUCS_Iab_to_sUCS_ICh(Iab))

        Iab = xp.tile(xp_as_array(Iab, xp=xp), (6, 1))
        ICh = xp.tile(xp_as_array(ICh, xp=xp), (6, 1))
        xp_assert_close(sUCS_Iab_to_sUCS_ICh(Iab), ICh, atol=TOLERANCE_ABSOLUTE_TESTS)

        Iab = xp_reshape(xp_as_array(Iab, xp=xp), (2, 3, 3), xp=xp)
        ICh = xp_reshape(xp_as_array(ICh, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(sUCS_Iab_to_sUCS_ICh(Iab), ICh, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_sUCS_Iab_to_sUCS_ICh(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.sucs.sUCS_Iab_to_sUCS_ICh` definition domain
        and range scale support.
        """

        Iab = xp_as_array([42.62923653, 36.97646831, 14.12301358], xp=xp)
        ICh = as_ndarray(sUCS_Iab_to_sUCS_ICh(Iab))

        d_r = (
            ("reference", 1, 1),
            ("1", 0.01, np.array([0.01, 0.01, 1 / 360])),
            ("100", 1, np.array([1, 1, 100 / 360])),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    sUCS_Iab_to_sUCS_ICh(Iab * xp_as_array(factor_a, xp=xp)),
                    ICh * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_sUCS_Iab_to_sUCS_ICh(self) -> None:
        """
        Test :func:`colour.models.sucs.sUCS_Iab_to_sUCS_ICh` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        sUCS_Iab_to_sUCS_ICh(cases)


class TestsUCS_ICh_to_sUCS_Iab:
    """
    Define :func:`colour.models.sucs.sUCS_ICh_to_sUCS_Iab` definition unit tests
    methods.
    """

    def test_sUCS_ICh_to_sUCS_Iab(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.sucs.sUCS_ICh_to_sUCS_Iab` definition."""

        xp_assert_close(
            sUCS_ICh_to_sUCS_Iab(
                xp_as_array([42.62923653, 40.42051106, 20.90415607], xp=xp)
            ),
            [42.62923653, 36.97646831, 14.12301358],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            sUCS_ICh_to_sUCS_Iab(
                xp_as_array([51.93649255, 29.43783150, 140.16328107], xp=xp)
            ),
            [51.93649255, -18.89245582, 15.76112395],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            sUCS_ICh_to_sUCS_Iab(
                xp_as_array([29.79456846, 30.80097976, 254.89763185], xp=xp)
            ),
            [29.79456846, -6.83806757, -25.33884097],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_sUCS_ICh_to_sUCS_Iab(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.sucs.sUCS_ICh_to_sUCS_Iab` definition
        n-dimensional support.
        """

        ICh = xp_as_array([42.62923653, 40.42051106, 20.90415607], xp=xp)
        Iab = as_ndarray(sUCS_ICh_to_sUCS_Iab(ICh))

        ICh = xp.tile(xp_as_array(ICh, xp=xp), (6, 1))
        Iab = xp.tile(xp_as_array(Iab, xp=xp), (6, 1))
        xp_assert_close(sUCS_ICh_to_sUCS_Iab(ICh), Iab, atol=TOLERANCE_ABSOLUTE_TESTS)

        ICh = xp_reshape(xp_as_array(ICh, xp=xp), (2, 3, 3), xp=xp)
        Iab = xp_reshape(xp_as_array(Iab, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(sUCS_ICh_to_sUCS_Iab(ICh), Iab, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_sUCS_ICh_to_sUCS_Iab(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.sucs.sUCS_ICh_to_sUCS_Iab` definition domain
        and range scale support.
        """

        ICh = xp_as_array([42.62923653, 40.42051106, 20.90415607], xp=xp)
        Iab = as_ndarray(sUCS_ICh_to_sUCS_Iab(ICh))

        d_r = (
            ("reference", 1, 1),
            ("1", np.array([0.01, 0.01, 1 / 360]), 0.01),
            ("100", np.array([1, 1, 100 / 360]), 1),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    sUCS_ICh_to_sUCS_Iab(ICh * xp_as_array(factor_a, xp=xp)),
                    Iab * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_sUCS_ICh_to_sUCS_Iab(self) -> None:
        """
        Test :func:`colour.models.sucs.sUCS_ICh_to_sUCS_Iab` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        sUCS_ICh_to_sUCS_Iab(cases)
