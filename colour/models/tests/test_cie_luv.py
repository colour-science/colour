"""Define the unit tests for the :mod:`colour.models.cie_luv` module."""

from itertools import product

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models import (
    CIE1976UCS_to_XYZ,
    Luv_to_uv,
    Luv_to_XYZ,
    Luv_uv_to_xy,
    XYZ_to_CIE1976UCS,
    XYZ_to_Luv,
    uv_to_Luv,
    xy_to_Luv_uv,
)
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestXYZ_to_Luv",
    "TestLuv_to_XYZ",
    "TestLuv_to_uv",
    "Testuv_to_Luv",
    "TestLuv_uv_to_xy",
    "TestXy_to_Luv_uv",
    "TestXYZ_to_CIE1976UCS",
    "TestCIE1976UCS_to_XYZ",
]


class TestXYZ_to_Luv:
    """
    Define :func:`colour.models.cie_luv.XYZ_to_Luv` definition unit tests
    methods.
    """

    def test_XYZ_to_Luv(self):
        """Test :func:`colour.models.cie_luv.XYZ_to_Luv` definition."""

        np.testing.assert_allclose(
            XYZ_to_Luv(np.array([0.20654008, 0.12197225, 0.05136952])),
            np.array([41.52787529, 96.83626054, 17.75210149]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            XYZ_to_Luv(np.array([0.14222010, 0.23042768, 0.10495772])),
            np.array([55.11636304, -37.59308176, 44.13768458]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            XYZ_to_Luv(np.array([0.07818780, 0.06157201, 0.28099326])),
            np.array([29.80565520, -10.96316802, -65.06751860]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            XYZ_to_Luv(
                np.array([0.20654008, 0.12197225, 0.05136952]),
                np.array([0.44757, 0.40745]),
            ),
            np.array([41.52787529, 65.45180940, -12.46626977]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            XYZ_to_Luv(
                np.array([0.20654008, 0.12197225, 0.05136952]),
                np.array([0.34570, 0.35850]),
            ),
            np.array([41.52787529, 90.70925962, 7.08455273]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            XYZ_to_Luv(
                np.array([0.20654008, 0.12197225, 0.05136952]),
                np.array([0.34570, 0.35850, 1.00000]),
            ),
            np.array([41.52787529, 90.70925962, 7.08455273]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_XYZ_to_Luv(self):
        """
        Test :func:`colour.models.cie_luv.XYZ_to_Luv` definition n-dimensional
        support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        illuminant = np.array([0.31270, 0.32900])
        Luv = XYZ_to_Luv(XYZ, illuminant)

        XYZ = np.tile(XYZ, (6, 1))
        Luv = np.tile(Luv, (6, 1))
        np.testing.assert_allclose(
            XYZ_to_Luv(XYZ, illuminant), Luv, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        illuminant = np.tile(illuminant, (6, 1))
        np.testing.assert_allclose(
            XYZ_to_Luv(XYZ, illuminant), Luv, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        XYZ = np.reshape(XYZ, (2, 3, 3))
        illuminant = np.reshape(illuminant, (2, 3, 2))
        Luv = np.reshape(Luv, (2, 3, 3))
        np.testing.assert_allclose(
            XYZ_to_Luv(XYZ, illuminant), Luv, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_XYZ_to_Luv(self):
        """
        Test :func:`colour.models.cie_luv.XYZ_to_Luv` definition
        domain and range scale support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        illuminant = np.array([0.31270, 0.32900])
        Luv = XYZ_to_Luv(XYZ, illuminant)

        d_r = (("reference", 1, 1), ("1", 1, 0.01), ("100", 100, 1))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    XYZ_to_Luv(XYZ * factor_a, illuminant),
                    Luv * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_Luv(self):
        """Test :func:`colour.models.cie_luv.XYZ_to_Luv` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        XYZ_to_Luv(cases, cases[..., 0:2])


class TestLuv_to_XYZ:
    """
    Define :func:`colour.models.cie_luv.Luv_to_XYZ` definition unit tests
    methods.
    """

    def test_Luv_to_XYZ(self):
        """Test :func:`colour.models.cie_luv.Luv_to_XYZ` definition."""

        np.testing.assert_allclose(
            Luv_to_XYZ(np.array([41.52787529, 96.83626054, 17.75210149])),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            Luv_to_XYZ(np.array([55.11636304, -37.59308176, 44.13768458])),
            np.array([0.14222010, 0.23042768, 0.10495772]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            Luv_to_XYZ(np.array([29.80565520, -10.96316802, -65.06751860])),
            np.array([0.07818780, 0.06157201, 0.28099326]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            Luv_to_XYZ(
                np.array([41.52787529, 65.45180940, -12.46626977]),
                np.array([0.44757, 0.40745]),
            ),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            Luv_to_XYZ(
                np.array([41.52787529, 90.70925962, 7.08455273]),
                np.array([0.34570, 0.35850]),
            ),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            Luv_to_XYZ(
                np.array([41.52787529, 90.70925962, 7.08455273]),
                np.array([0.34570, 0.35850, 1.00000]),
            ),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_Luv_to_XYZ(self):
        """
        Test :func:`colour.models.cie_luv.Luv_to_XYZ` definition n-dimensional
        support.
        """

        Luv = np.array([41.52787529, 96.83626054, 17.75210149])
        illuminant = np.array([0.31270, 0.32900])
        XYZ = Luv_to_XYZ(Luv, illuminant)

        Luv = np.tile(Luv, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_allclose(
            Luv_to_XYZ(Luv, illuminant), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        illuminant = np.tile(illuminant, (6, 1))
        np.testing.assert_allclose(
            Luv_to_XYZ(Luv, illuminant), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        Luv = np.reshape(Luv, (2, 3, 3))
        illuminant = np.reshape(illuminant, (2, 3, 2))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_allclose(
            Luv_to_XYZ(Luv, illuminant), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_Luv_to_XYZ(self):
        """
        Test :func:`colour.models.cie_luv.Luv_to_XYZ` definition
        domain and range scale support.
        """

        Luv = np.array([41.52787529, 96.83626054, 17.75210149])
        illuminant = np.array([0.31270, 0.32900])
        XYZ = Luv_to_XYZ(Luv, illuminant)

        d_r = (("reference", 1, 1), ("1", 0.01, 1), ("100", 1, 100))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    Luv_to_XYZ(Luv * factor_a, illuminant),
                    XYZ * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_Luv_to_XYZ(self):
        """Test :func:`colour.models.cie_luv.Luv_to_XYZ` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        Luv_to_XYZ(cases, cases[..., 0:2])


class TestLuv_to_uv:
    """
    Define :func:`colour.models.cie_luv.Luv_to_uv` definition unit tests
    methods.
    """

    def test_Luv_to_uv(self):
        """Test :func:`colour.models.cie_luv.Luv_to_uv` definition."""

        np.testing.assert_allclose(
            Luv_to_uv(np.array([41.52787529, 96.83626054, 17.75210149])),
            np.array([0.37720213, 0.50120264]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            Luv_to_uv(np.array([55.11636304, -37.59308176, 44.13768458])),
            np.array([0.14536327, 0.52992069]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            Luv_to_uv(np.array([29.80565520, -10.96316802, -65.06751860])),
            np.array([0.16953603, 0.30039234]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            Luv_to_uv(
                np.array([41.52787529, 65.45180940, -12.46626977]),
                np.array([0.44757, 0.40745]),
            ),
            np.array([0.37720213, 0.50120264]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            Luv_to_uv(
                np.array([41.52787529, 90.70925962, 7.08455273]),
                np.array([0.34570, 0.35850]),
            ),
            np.array([0.37720213, 0.50120264]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            Luv_to_uv(
                np.array([41.52787529, 90.70925962, 7.08455273]),
                np.array([0.34570, 0.35850, 1.00000]),
            ),
            np.array([0.37720213, 0.50120264]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_Luv_to_uv(self):
        """
        Test :func:`colour.models.cie_luv.Luv_to_uv` definition n-dimensional
        support.
        """

        Luv = np.array([41.52787529, 96.83626054, 17.75210149])
        illuminant = np.array([0.31270, 0.32900])
        uv = Luv_to_uv(Luv, illuminant)

        Luv = np.tile(Luv, (6, 1))
        uv = np.tile(uv, (6, 1))
        np.testing.assert_allclose(
            Luv_to_uv(Luv, illuminant), uv, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        illuminant = np.tile(illuminant, (6, 1))
        np.testing.assert_allclose(
            Luv_to_uv(Luv, illuminant), uv, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        Luv = np.reshape(Luv, (2, 3, 3))
        illuminant = np.reshape(illuminant, (2, 3, 2))
        uv = np.reshape(uv, (2, 3, 2))
        np.testing.assert_allclose(
            Luv_to_uv(Luv, illuminant), uv, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_Luv_to_uv(self):
        """
        Test :func:`colour.models.cie_luv.Luv_to_uv` definition
        domain and range scale support.
        """

        Luv = np.array([41.52787529, 96.83626054, 17.75210149])
        illuminant = np.array([0.31270, 0.32900])
        uv = Luv_to_uv(Luv, illuminant)

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    Luv_to_uv(Luv * factor, illuminant),
                    uv,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_Luv_to_uv(self):
        """Test :func:`colour.models.cie_luv.Luv_to_uv` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        Luv_to_uv(cases, cases[..., 0:2])


class Testuv_to_Luv:
    """
    Define :func:`colour.models.cie_luv.uv_to_Luv` definition unit tests
    methods.
    """

    def test_uv_to_Luv(self):
        """Test :func:`colour.models.cie_luv.uv_to_Luv` definition."""

        np.testing.assert_allclose(
            uv_to_Luv(np.array([0.37720213, 0.50120264])),
            np.array([100.00000000, 233.18376036, 42.74743858]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            uv_to_Luv(np.array([0.14536327, 0.52992069])),
            np.array([100.00000000, -68.20675764, 80.08090358]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            uv_to_Luv(np.array([0.16953603, 0.30039234])),
            np.array([100.00000000, -36.78216964, -218.3059514]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            uv_to_Luv(
                np.array([0.37720213, 0.50120264]),
                np.array([0.44757, 0.40745]),
            ),
            np.array([100.00000000, 157.60933976, -30.01903705]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            uv_to_Luv(
                np.array([0.37720213, 0.50120264]),
                np.array([0.34570, 0.35850]),
            ),
            np.array([100.00000000, 218.42981284, 17.05975609]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            uv_to_Luv(
                np.array([0.37720213, 0.50120264]),
                np.array([0.34570, 0.35850, 1.00000]),
            ),
            np.array([100.00000000, 218.42981284, 17.05975609]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            uv_to_Luv(np.array([0.37720213, 0.50120264]), L=41.5278752),
            np.array([41.52787529, 96.83626054, 17.75210149]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_uv_to_Luv(self):
        """
        Test :func:`colour.models.cie_luv.uv_to_Luv` definition n-dimensional
        support.
        """

        uv = np.array([0.37720213, 0.50120264])
        illuminant = np.array([0.31270, 0.32900])
        Luv = uv_to_Luv(uv, illuminant)

        uv = np.tile(uv, (6, 1))
        Luv = np.tile(Luv, (6, 1))
        np.testing.assert_allclose(
            uv_to_Luv(uv, illuminant), Luv, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        illuminant = np.tile(illuminant, (6, 1))
        np.testing.assert_allclose(
            uv_to_Luv(uv, illuminant), Luv, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        uv = np.reshape(uv, (2, 3, 2))
        illuminant = np.reshape(illuminant, (2, 3, 2))
        Luv = np.reshape(Luv, (2, 3, 3))
        np.testing.assert_allclose(
            uv_to_Luv(uv, illuminant), Luv, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_uv_to_Luv(self):
        """
        Test :func:`colour.models.cie_luv.uv_to_Luv` definition
        domain and range scale support.
        """

        uv = np.array([0.37720213, 0.50120264])
        illuminant = np.array([0.31270, 0.32900])
        L = 100
        Luv = uv_to_Luv(uv, illuminant, L)

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    uv_to_Luv(uv, illuminant, L * factor),
                    Luv * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_uv_to_Luv(self):
        """Test :func:`colour.models.cie_luv.uv_to_Luv` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        uv_to_Luv(cases, cases[..., 0:2])


class TestLuv_uv_to_xy:
    """
    Define :func:`colour.models.cie_luv.Luv_uv_to_xy` definition unit tests
    methods.
    """

    def test_Luv_uv_to_xy(self):
        """Test :func:`colour.models.cie_luv.Luv_uv_to_xy` definition."""

        np.testing.assert_allclose(
            Luv_uv_to_xy(np.array([0.37720213, 0.50120264])),
            np.array([0.54369558, 0.32107944]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            Luv_uv_to_xy(np.array([0.14536327, 0.52992069])),
            np.array([0.29777734, 0.48246445]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            Luv_uv_to_xy(np.array([0.16953603, 0.30039234])),
            np.array([0.18582824, 0.14633764]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_Luv_uv_to_xy(self):
        """
        Test :func:`colour.models.cie_luv.Luv_uv_to_xy` definition
        n-dimensional arrays support.
        """

        uv = np.array([0.37720213, 0.50120264])
        xy = Luv_uv_to_xy(uv)

        uv = np.tile(uv, (6, 1))
        xy = np.tile(xy, (6, 1))
        np.testing.assert_allclose(Luv_uv_to_xy(uv), xy, atol=TOLERANCE_ABSOLUTE_TESTS)

        uv = np.reshape(uv, (2, 3, 2))
        xy = np.reshape(xy, (2, 3, 2))
        np.testing.assert_allclose(Luv_uv_to_xy(uv), xy, atol=TOLERANCE_ABSOLUTE_TESTS)

    @ignore_numpy_errors
    def test_nan_Luv_uv_to_xy(self):
        """
        Test :func:`colour.models.cie_luv.Luv_uv_to_xy` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        Luv_uv_to_xy(cases)


class TestXy_to_Luv_uv:
    """
    Define :func:`colour.models.cie_luv.xy_to_Luv_uv` definition unit tests
    methods.
    """

    def test_xy_to_Luv_uv(self):
        """Test :func:`colour.models.cie_luv.xy_to_Luv_uv` definition."""

        np.testing.assert_allclose(
            xy_to_Luv_uv(np.array([0.54369558, 0.32107944])),
            np.array([0.37720213, 0.50120264]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            xy_to_Luv_uv(np.array([0.29777734, 0.48246445])),
            np.array([0.14536327, 0.52992069]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            xy_to_Luv_uv(np.array([0.18582824, 0.14633764])),
            np.array([0.16953603, 0.30039234]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_xy_to_Luv_uv(self):
        """
        Test :func:`colour.models.cie_luv.xy_to_Luv_uv` definition
        n-dimensional arrays support.
        """

        xy = np.array([0.54369558, 0.32107944])
        uv = xy_to_Luv_uv(xy)

        xy = np.tile(xy, (6, 1))
        uv = np.tile(uv, (6, 1))
        np.testing.assert_allclose(xy_to_Luv_uv(xy), uv, atol=TOLERANCE_ABSOLUTE_TESTS)

        xy = np.reshape(xy, (2, 3, 2))
        uv = np.reshape(uv, (2, 3, 2))
        np.testing.assert_allclose(xy_to_Luv_uv(xy), uv, atol=TOLERANCE_ABSOLUTE_TESTS)

    @ignore_numpy_errors
    def test_nan_xy_to_Luv_uv(self):
        """
        Test :func:`colour.models.cie_luv.xy_to_Luv_uv` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        xy_to_Luv_uv(cases)


class TestXYZ_to_CIE1976UCS:
    """
    Define :func:`colour.models.cie_luv.XYZ_to_CIE1976UCS` definition unit tests
    methods.
    """

    def test_XYZ_to_CIE1976UCS(self):
        """Test :func:`colour.models.cie_luv.XYZ_to_CIE1976UCS` definition."""

        np.testing.assert_allclose(
            XYZ_to_CIE1976UCS(np.array([0.20654008, 0.12197225, 0.05136952])),
            np.array([0.37720213, 0.50120264, 41.52787529]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            XYZ_to_CIE1976UCS(np.array([0.14222010, 0.23042768, 0.10495772])),
            np.array([0.14536327, 0.52992069, 55.11636304]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            XYZ_to_CIE1976UCS(np.array([0.07818780, 0.06157201, 0.28099326])),
            np.array([0.16953603, 0.30039234, 29.80565520]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            XYZ_to_CIE1976UCS(
                np.array([0.20654008, 0.12197225, 0.05136952]),
                np.array([0.44757, 0.40745]),
            ),
            np.array([0.37720213, 0.50120264, 41.52787529]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            XYZ_to_CIE1976UCS(
                np.array([0.20654008, 0.12197225, 0.05136952]),
                np.array([0.34570, 0.35850]),
            ),
            np.array([0.37720213, 0.50120264, 41.52787529]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            XYZ_to_CIE1976UCS(
                np.array([0.20654008, 0.12197225, 0.05136952]),
                np.array([0.34570, 0.35850, 1.00000]),
            ),
            np.array([0.37720213, 0.50120264, 41.52787529]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_XYZ_to_CIE1976UCS(self):
        """
        Test :func:`colour.models.cie_luv.XYZ_to_CIE1976UCS` definition n-dimensional
        support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        illuminant = np.array([0.31270, 0.32900])
        Luv = XYZ_to_CIE1976UCS(XYZ, illuminant)

        XYZ = np.tile(XYZ, (6, 1))
        Luv = np.tile(Luv, (6, 1))
        np.testing.assert_allclose(
            XYZ_to_CIE1976UCS(XYZ, illuminant), Luv, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        illuminant = np.tile(illuminant, (6, 1))
        np.testing.assert_allclose(
            XYZ_to_CIE1976UCS(XYZ, illuminant), Luv, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        XYZ = np.reshape(XYZ, (2, 3, 3))
        illuminant = np.reshape(illuminant, (2, 3, 2))
        Luv = np.reshape(Luv, (2, 3, 3))
        np.testing.assert_allclose(
            XYZ_to_CIE1976UCS(XYZ, illuminant), Luv, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_XYZ_to_CIE1976UCS(self):
        """
        Test :func:`colour.models.cie_luv.XYZ_to_CIE1976UCS` definition
        domain and range scale support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        illuminant = np.array([0.31270, 0.32900])
        uvL = XYZ_to_CIE1976UCS(XYZ, illuminant)

        d_r = (("reference", 1, 1), ("1", 1, np.array([1, 1, 0.01])), ("100", 100, 1))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    XYZ_to_CIE1976UCS(XYZ * factor_a, illuminant),
                    uvL * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_CIE1976UCS(self):
        """
        Test :func:`colour.models.cie_luv.XYZ_to_CIE1976UCS` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        XYZ_to_CIE1976UCS(cases, cases[..., 0:2])


class TestCIE1976UCS_to_XYZ:
    """
    Define :func:`colour.models.cie_luv.CIE1976UCS_to_XYZ` definition unit tests
    methods.
    """

    def test_CIE1976UCS_to_XYZ(self):
        """Test :func:`colour.models.cie_luv.CIE1976UCS_to_XYZ` definition."""

        np.testing.assert_allclose(
            CIE1976UCS_to_XYZ(np.array([0.37720213, 0.50120264, 41.52787529])),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            CIE1976UCS_to_XYZ(np.array([0.14536327, 0.52992069, 55.11636304])),
            np.array([0.14222010, 0.23042768, 0.10495772]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            CIE1976UCS_to_XYZ(np.array([0.16953603, 0.30039234, 29.80565520])),
            np.array([0.07818780, 0.06157201, 0.28099326]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            CIE1976UCS_to_XYZ(
                np.array([0.37720213, 0.50120264, 41.52787529]),
                np.array([0.44757, 0.40745]),
            ),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            CIE1976UCS_to_XYZ(
                np.array([0.37720213, 0.50120264, 41.52787529]),
                np.array([0.34570, 0.35850]),
            ),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            CIE1976UCS_to_XYZ(
                np.array([0.37720213, 0.50120264, 41.52787529]),
                np.array([0.34570, 0.35850, 1.00000]),
            ),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_CIE1976UCS_to_XYZ(self):
        """
        Test :func:`colour.models.cie_luv.CIE1976UCS_to_XYZ` definition n-dimensional
        support.
        """

        Luv = np.array([0.37720213, 0.50120264, 41.52787529])
        illuminant = np.array([0.31270, 0.32900])
        XYZ = CIE1976UCS_to_XYZ(Luv, illuminant)

        Luv = np.tile(Luv, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_allclose(
            CIE1976UCS_to_XYZ(Luv, illuminant), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        illuminant = np.tile(illuminant, (6, 1))
        np.testing.assert_allclose(
            CIE1976UCS_to_XYZ(Luv, illuminant), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        Luv = np.reshape(Luv, (2, 3, 3))
        illuminant = np.reshape(illuminant, (2, 3, 2))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_allclose(
            CIE1976UCS_to_XYZ(Luv, illuminant), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_CIE1976UCS_to_XYZ(self):
        """
        Test :func:`colour.models.cie_luv.CIE1976UCS_to_XYZ` definition
        domain and range scale support.
        """

        uvL = np.array([0.37720213, 0.50120264, 41.52787529])
        illuminant = np.array([0.31270, 0.32900])
        XYZ = CIE1976UCS_to_XYZ(uvL, illuminant)

        d_r = (("reference", 1, 1), ("1", np.array([1, 1, 0.01]), 1), ("100", 1, 100))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    CIE1976UCS_to_XYZ(uvL * factor_a, illuminant),
                    XYZ * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_CIE1976UCS_to_XYZ(self):
        """
        Test :func:`colour.models.cie_luv.CIE1976UCS_to_XYZ` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        CIE1976UCS_to_XYZ(cases, cases[..., 0:2])
