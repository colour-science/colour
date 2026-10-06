"""Define the unit tests for the :mod:`colour.models.cam02_ucs` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np

from colour.appearance import (
    VIEWING_CONDITIONS_CIECAM02,
    CAM_KWARGS_CIECAM02_sRGB,
    XYZ_to_CIECAM02,
)
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models import (
    CAM02LCD_to_JMh_CIECAM02,
    CAM02LCD_to_XYZ,
    CAM02SCD_to_JMh_CIECAM02,
    CAM02SCD_to_XYZ,
    CAM02UCS_to_JMh_CIECAM02,
    CAM02UCS_to_XYZ,
    JMh_CIECAM02_to_CAM02LCD,
    JMh_CIECAM02_to_CAM02SCD,
    JMh_CIECAM02_to_CAM02UCS,
    XYZ_to_CAM02LCD,
    XYZ_to_CAM02SCD,
    XYZ_to_CAM02UCS,
)
from colour.models.cam02_ucs import (
    COEFFICIENTS_UCS_LUO2006,
    JMh_CIECAM02_to_UCS_Luo2006,
    UCS_Luo2006_to_JMh_CIECAM02,
    UCS_Luo2006_to_XYZ,
    XYZ_to_UCS_Luo2006,
)
from colour.utilities import (
    as_ndarray,
    attest,
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
    "TestJMh_CIECAM02_to_UCS_Luo2006",
    "TestUCS_Luo2006_to_JMh_CIECAM02",
    "TestXYZ_to_UCS_Luo2006",
    "TestUCS_Luo2006_to_XYZ",
]


class TestJMh_CIECAM02_to_UCS_Luo2006:
    """
    Define :func:`colour.models.cam02_ucs.JMh_CIECAM02_to_UCS_Luo2006`
    definition unit tests methods.
    """

    def setup_method(self) -> None:
        """Initialise the common tests attributes."""

        XYZ = np.array([19.01, 20.00, 21.78])
        XYZ_w = np.array([95.05, 100.00, 108.88])
        L_A = 318.31
        Y_b = 20.0
        surround = VIEWING_CONDITIONS_CIECAM02["Average"]
        specification = XYZ_to_CIECAM02(XYZ, XYZ_w, L_A, Y_b, surround)

        self._JMh = np.array([specification.J, specification.M, specification.h])

    def test_JMh_CIECAM02_to_UCS_Luo2006(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.cam02_ucs.JMh_CIECAM02_to_UCS_Luo2006`
        definition.
        """

        JMh = xp_as_array(self._JMh, xp=xp)

        xp_assert_close(
            JMh_CIECAM02_to_UCS_Luo2006(JMh, COEFFICIENTS_UCS_LUO2006["CAM02-LCD"]),
            [54.90433134, -0.08450395, -0.06854831],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            JMh_CIECAM02_to_UCS_Luo2006(JMh, COEFFICIENTS_UCS_LUO2006["CAM02-LCD"]),
            as_ndarray(JMh_CIECAM02_to_CAM02LCD(JMh)),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            JMh_CIECAM02_to_UCS_Luo2006(JMh, COEFFICIENTS_UCS_LUO2006["CAM02-SCD"]),
            [54.90433134, -0.08436178, -0.06843298],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            JMh_CIECAM02_to_UCS_Luo2006(JMh, COEFFICIENTS_UCS_LUO2006["CAM02-SCD"]),
            as_ndarray(JMh_CIECAM02_to_CAM02SCD(JMh)),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            JMh_CIECAM02_to_UCS_Luo2006(JMh, COEFFICIENTS_UCS_LUO2006["CAM02-UCS"]),
            [54.90433134, -0.08442362, -0.06848314],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            JMh_CIECAM02_to_UCS_Luo2006(JMh, COEFFICIENTS_UCS_LUO2006["CAM02-UCS"]),
            as_ndarray(JMh_CIECAM02_to_CAM02UCS(JMh)),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_JMh_CIECAM02_to_UCS_Luo2006(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.cam02_ucs.JMh_CIECAM02_to_UCS_Luo2006`
        definition n-dimensional support.
        """

        JMh = xp_as_array(self._JMh, xp=xp)
        Jpapbp = as_ndarray(
            JMh_CIECAM02_to_UCS_Luo2006(JMh, COEFFICIENTS_UCS_LUO2006["CAM02-LCD"])
        )

        JMh = xp.tile(xp_as_array(JMh, xp=xp), (6, 1))
        Jpapbp = xp.tile(xp_as_array(Jpapbp, xp=xp), (6, 1))
        xp_assert_close(
            JMh_CIECAM02_to_UCS_Luo2006(JMh, COEFFICIENTS_UCS_LUO2006["CAM02-LCD"]),
            Jpapbp,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        JMh = xp_reshape(xp_as_array(JMh, xp=xp), (2, 3, 3), xp=xp)
        Jpapbp = xp_reshape(xp_as_array(Jpapbp, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(
            JMh_CIECAM02_to_UCS_Luo2006(JMh, COEFFICIENTS_UCS_LUO2006["CAM02-LCD"]),
            Jpapbp,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_JMh_CIECAM02_to_UCS_Luo2006(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.models.cam02_ucs.JMh_CIECAM02_to_UCS_Luo2006`
        definition domain and range scale support.
        """

        JMh = xp_as_array(self._JMh, xp=xp)
        Jpapbp = as_ndarray(
            JMh_CIECAM02_to_UCS_Luo2006(JMh, COEFFICIENTS_UCS_LUO2006["CAM02-LCD"])
        )

        d_r = (
            ("reference", 1, 1),
            ("1", xp_as_array([0.01, 0.01, 1 / 360], xp=xp), 0.01),
            ("100", xp_as_array([1, 1, 1 / 3.6], xp=xp), 1),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    JMh_CIECAM02_to_UCS_Luo2006(
                        JMh * factor_a, COEFFICIENTS_UCS_LUO2006["CAM02-LCD"]
                    ),
                    Jpapbp * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_JMh_CIECAM02_to_UCS_Luo2006(self) -> None:
        """
        Test :func:`colour.models.cam02_ucs.JMh_CIECAM02_to_UCS_Luo2006`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        JMh_CIECAM02_to_UCS_Luo2006(cases, COEFFICIENTS_UCS_LUO2006["CAM02-LCD"])


class TestUCS_Luo2006_to_JMh_CIECAM02:
    """
    Define :func:`colour.models.cam02_ucs.UCS_Luo2006_to_JMh_CIECAM02`
    definition unit tests methods.
    """

    def test_UCS_Luo2006_to_JMh_CIECAM02(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.cam02_ucs.UCS_Luo2006_to_JMh_CIECAM02`
        definition.
        """

        xp_assert_close(
            UCS_Luo2006_to_JMh_CIECAM02(
                xp_as_array([54.90433134, -0.08442362, -0.06848314], xp=xp),
                COEFFICIENTS_UCS_LUO2006["CAM02-LCD"],
            ),
            [41.73109113, 0.10873867, 219.04843202],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            UCS_Luo2006_to_JMh_CIECAM02(
                xp_as_array([54.90433134, -0.08442362, -0.06848314], xp=xp),
                COEFFICIENTS_UCS_LUO2006["CAM02-LCD"],
            ),
            CAM02LCD_to_JMh_CIECAM02([54.90433134, -0.08442362, -0.06848314]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            UCS_Luo2006_to_JMh_CIECAM02(
                xp_as_array([54.90433134, -0.08442362, -0.06848314], xp=xp),
                COEFFICIENTS_UCS_LUO2006["CAM02-SCD"],
            ),
            [41.73109113, 0.10892212, 219.04843202],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            UCS_Luo2006_to_JMh_CIECAM02(
                xp_as_array([54.90433134, -0.08442362, -0.06848314], xp=xp),
                COEFFICIENTS_UCS_LUO2006["CAM02-SCD"],
            ),
            CAM02SCD_to_JMh_CIECAM02([54.90433134, -0.08442362, -0.06848314]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            UCS_Luo2006_to_JMh_CIECAM02(
                xp_as_array([54.90433134, -0.08442362, -0.06848314], xp=xp),
                COEFFICIENTS_UCS_LUO2006["CAM02-UCS"],
            ),
            [41.73109113, 0.10884218, 219.04843202],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            UCS_Luo2006_to_JMh_CIECAM02(
                xp_as_array([54.90433134, -0.08442362, -0.06848314], xp=xp),
                COEFFICIENTS_UCS_LUO2006["CAM02-UCS"],
            ),
            CAM02UCS_to_JMh_CIECAM02([54.90433134, -0.08442362, -0.06848314]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_UCS_Luo2006_to_JMh_CIECAM02(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.cam02_ucs.UCS_Luo2006_to_JMh_CIECAM02`
        definition n-dimensional support.
        """

        Jpapbp = xp_as_array([54.90433134, -0.08442362, -0.06848314], xp=xp)
        JMh = as_ndarray(
            UCS_Luo2006_to_JMh_CIECAM02(Jpapbp, COEFFICIENTS_UCS_LUO2006["CAM02-LCD"])
        )

        Jpapbp = xp.tile(xp_as_array(Jpapbp, xp=xp), (6, 1))
        JMh = xp.tile(xp_as_array(JMh, xp=xp), (6, 1))
        xp_assert_close(
            UCS_Luo2006_to_JMh_CIECAM02(Jpapbp, COEFFICIENTS_UCS_LUO2006["CAM02-LCD"]),
            JMh,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        Jpapbp = xp_reshape(xp_as_array(Jpapbp, xp=xp), (2, 3, 3), xp=xp)
        JMh = xp_reshape(xp_as_array(JMh, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(
            UCS_Luo2006_to_JMh_CIECAM02(Jpapbp, COEFFICIENTS_UCS_LUO2006["CAM02-LCD"]),
            JMh,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_UCS_Luo2006_to_JMh_CIECAM02(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.models.cam02_ucs.UCS_Luo2006_to_JMh_CIECAM02`
        definition domain and range scale support.
        """

        Jpapbp = xp_as_array([54.90433134, -0.08442362, -0.06848314], xp=xp)
        JMh = as_ndarray(
            UCS_Luo2006_to_JMh_CIECAM02(Jpapbp, COEFFICIENTS_UCS_LUO2006["CAM02-LCD"])
        )

        d_r = (
            ("reference", 1, 1),
            ("1", 0.01, np.array([0.01, 0.01, 1 / 360])),
            ("100", 1, np.array([1, 1, 1 / 3.6])),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    UCS_Luo2006_to_JMh_CIECAM02(
                        Jpapbp * factor_a,
                        COEFFICIENTS_UCS_LUO2006["CAM02-LCD"],
                    ),
                    JMh * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_UCS_Luo2006_to_JMh_CIECAM02(self) -> None:
        """
        Test :func:`colour.models.cam02_ucs.UCS_Luo2006_to_JMh_CIECAM02`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        UCS_Luo2006_to_JMh_CIECAM02(cases, COEFFICIENTS_UCS_LUO2006["CAM02-LCD"])


class TestXYZ_to_UCS_Luo2006:
    """
    Define :func:`colour.models.cam02_ucs.XYZ_to_UCS_Luo2006` definition
    unit tests methods.
    """

    def test_XYZ_to_UCS_Luo2006(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.cam02_ucs.XYZ_to_UCS_Luo2006` definition."""

        xp_assert_close(
            XYZ_to_UCS_Luo2006(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp),
                COEFFICIENTS_UCS_LUO2006["CAM02-LCD"],
            ),
            [46.61386154, 39.35760236, 15.96730435],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_UCS_Luo2006(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp),
                COEFFICIENTS_UCS_LUO2006["CAM02-LCD"],
            ),
            XYZ_to_CAM02LCD([0.20654008, 0.12197225, 0.05136952]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_UCS_Luo2006(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp),
                COEFFICIENTS_UCS_LUO2006["CAM02-SCD"],
            ),
            [46.61386154, 25.62879882, 10.39755489],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_UCS_Luo2006(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp),
                COEFFICIENTS_UCS_LUO2006["CAM02-SCD"],
            ),
            XYZ_to_CAM02SCD([0.20654008, 0.12197225, 0.05136952]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_UCS_Luo2006(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp),
                COEFFICIENTS_UCS_LUO2006["CAM02-UCS"],
            ),
            [46.61386154, 29.88310013, 12.12351683],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_UCS_Luo2006(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp),
                COEFFICIENTS_UCS_LUO2006["CAM02-UCS"],
            ),
            XYZ_to_CAM02UCS([0.20654008, 0.12197225, 0.05136952]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_XYZ_to_UCS_Luo2006(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.cam02_ucs.XYZ_to_UCS_Luo2006` definition
        n-dimensional support.
        """

        XYZ = xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)
        Jpapbp = as_ndarray(
            XYZ_to_UCS_Luo2006(XYZ, COEFFICIENTS_UCS_LUO2006["CAM02-LCD"])
        )

        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        Jpapbp = xp.tile(xp_as_array(Jpapbp, xp=xp), (6, 1))
        xp_assert_close(
            XYZ_to_UCS_Luo2006(XYZ, COEFFICIENTS_UCS_LUO2006["CAM02-LCD"]),
            Jpapbp,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        Jpapbp = xp_reshape(xp_as_array(Jpapbp, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(
            XYZ_to_UCS_Luo2006(XYZ, COEFFICIENTS_UCS_LUO2006["CAM02-LCD"]),
            Jpapbp,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_XYZ_to_UCS_Luo2006(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.cam02_ucs.XYZ_to_UCS_Luo2006` definition
        domain and range scale support.
        """

        XYZ = xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)
        XYZ_w = xp_as_array(CAM_KWARGS_CIECAM02_sRGB["XYZ_w"] / 100, xp=xp)
        Jpapbp = as_ndarray(
            XYZ_to_UCS_Luo2006(XYZ, COEFFICIENTS_UCS_LUO2006["CAM02-LCD"])
        )

        d_r = (("reference", 1, 1), ("1", 1, 0.01), ("100", 100, 1))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    XYZ_to_UCS_Luo2006(
                        XYZ * xp_as_array(factor_a, xp=xp),
                        COEFFICIENTS_UCS_LUO2006["CAM02-LCD"],
                        XYZ_w=XYZ_w * xp_as_array(factor_a, xp=xp),
                    ),
                    Jpapbp * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_UCS_Luo2006(self) -> None:
        """
        Test :func:`colour.models.cam02_ucs.XYZ_to_UCS_Luo2006` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        XYZ_to_UCS_Luo2006(cases, COEFFICIENTS_UCS_LUO2006["CAM02-LCD"])


class TestUCS_Luo2006_to_XYZ:
    """
    Define :func:`colour.models.cam02_ucs.UCS_Luo2006_to_XYZ` definition
    unit tests methods.
    """

    def test_UCS_Luo2006_to_XYZ(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.cam02_ucs.UCS_Luo2006_to_XYZ` definition."""

        xp_assert_close(
            UCS_Luo2006_to_XYZ(
                xp_as_array([46.61386154, 39.35760236, 15.96730435], xp=xp),
                COEFFICIENTS_UCS_LUO2006["CAM02-LCD"],
            ),
            [0.20654008, 0.12197225, 0.05136952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            UCS_Luo2006_to_XYZ(
                xp_as_array([46.61386154, 39.35760236, 15.96730435], xp=xp),
                COEFFICIENTS_UCS_LUO2006["CAM02-LCD"],
            ),
            CAM02LCD_to_XYZ([46.61386154, 39.35760236, 15.96730435]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            UCS_Luo2006_to_XYZ(
                xp_as_array([46.61386154, 39.35760236, 15.96730435], xp=xp),
                COEFFICIENTS_UCS_LUO2006["CAM02-SCD"],
            ),
            [0.28264475, 0.11036927, 0.00824593],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            UCS_Luo2006_to_XYZ(
                xp_as_array([46.61386154, 39.35760236, 15.96730435], xp=xp),
                COEFFICIENTS_UCS_LUO2006["CAM02-SCD"],
            ),
            CAM02SCD_to_XYZ([46.61386154, 39.35760236, 15.96730435]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            UCS_Luo2006_to_XYZ(
                xp_as_array([46.61386154, 39.35760236, 15.96730435], xp=xp),
                COEFFICIENTS_UCS_LUO2006["CAM02-UCS"],
            ),
            [0.24229809, 0.11573005, 0.02517649],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            UCS_Luo2006_to_XYZ(
                xp_as_array([46.61386154, 39.35760236, 15.96730435], xp=xp),
                COEFFICIENTS_UCS_LUO2006["CAM02-UCS"],
            ),
            CAM02UCS_to_XYZ([46.61386154, 39.35760236, 15.96730435]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_UCS_Luo2006_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.cam02_ucs.UCS_Luo2006_to_XYZ` definition
        n-dimensional support.
        """

        Jpapbp = xp_as_array([46.61386154, 39.35760236, 15.96730435], xp=xp)
        XYZ = as_ndarray(
            UCS_Luo2006_to_XYZ(Jpapbp, COEFFICIENTS_UCS_LUO2006["CAM02-LCD"])
        )

        Jpapbp = xp.tile(xp_as_array(Jpapbp, xp=xp), (6, 1))
        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        xp_assert_close(
            UCS_Luo2006_to_XYZ(Jpapbp, COEFFICIENTS_UCS_LUO2006["CAM02-LCD"]),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        Jpapbp = xp_reshape(xp_as_array(Jpapbp, xp=xp), (2, 3, 3), xp=xp)
        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(
            UCS_Luo2006_to_XYZ(Jpapbp, COEFFICIENTS_UCS_LUO2006["CAM02-LCD"]),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_UCS_Luo2006_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.cam02_ucs.UCS_Luo2006_to_XYZ` definition
        domain and range scale support.
        """

        Jpapbp = xp_as_array([46.61386154, 39.35760236, 15.96730435], xp=xp)
        XYZ = as_ndarray(
            UCS_Luo2006_to_XYZ(Jpapbp, COEFFICIENTS_UCS_LUO2006["CAM02-LCD"])
        )
        XYZ_w = xp_as_array(CAM_KWARGS_CIECAM02_sRGB["XYZ_w"] / 100, xp=xp)

        d_r = (("reference", 1, 1, 1), ("1", 0.01, 1, 1), ("100", 1, 100, 100))
        for scale, factor_a, factor_b, factor_c in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    UCS_Luo2006_to_XYZ(
                        Jpapbp * xp_as_array(factor_a, xp=xp),
                        COEFFICIENTS_UCS_LUO2006["CAM02-LCD"],
                        XYZ_w=XYZ_w * factor_c,
                    ),
                    XYZ * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_UCS_Luo2006_to_XYZ(self) -> None:
        """
        Test :func:`colour.models.cam02_ucs.UCS_Luo2006_to_XYZ` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        for case in cases:
            try:
                UCS_Luo2006_to_XYZ(case, COEFFICIENTS_UCS_LUO2006["CAM02-LCD"])
            except ValueError as error:
                attest("CAM_Specification_CIECAM02" in str(error))
