# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.models.cie_ucs` module."""

from itertools import product

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models import (
    CIE1960UCS_to_XYZ,
    UCS_to_uv,
    UCS_to_XYZ,
    UCS_uv_to_xy,
    XYZ_to_CIE1960UCS,
    XYZ_to_UCS,
    uv_to_UCS,
    xy_to_UCS_uv,
)
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestXYZ_to_UCS",
    "TestUCS_to_XYZ",
    "TestUCS_to_uv",
    "Testuv_to_UCS",
    "TestUCS_uv_to_xy",
    "TestXy_to_UCS_uv",
    "TestXYZ_to_CIE1960UCS",
    "TestCIE1960UCS_to_XYZ",
]


class TestXYZ_to_UCS:
    """
    Define :func:`colour.models.cie_ucs.XYZ_to_UCS` definition unit tests
    methods.
    """

    def test_XYZ_to_UCS(self):
        """Test :func:`colour.models.cie_ucs.XYZ_to_UCS` definition."""

        np.testing.assert_allclose(
            XYZ_to_UCS(np.array([0.20654008, 0.12197225, 0.05136952])),
            np.array([0.13769339, 0.12197225, 0.10537310]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            XYZ_to_UCS(np.array([0.14222010, 0.23042768, 0.10495772])),
            np.array([0.09481340, 0.23042768, 0.32701033]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            XYZ_to_UCS(np.array([0.07818780, 0.06157201, 0.28099326])),
            np.array([0.05212520, 0.06157201, 0.19376075]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_XYZ_to_UCS(self):
        """
        Test :func:`colour.models.cie_ucs.XYZ_to_UCS` definition n-dimensional
        support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        UCS = XYZ_to_UCS(XYZ)

        UCS = np.tile(UCS, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_allclose(XYZ_to_UCS(XYZ), UCS, atol=TOLERANCE_ABSOLUTE_TESTS)

        UCS = np.reshape(UCS, (2, 3, 3))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_allclose(XYZ_to_UCS(XYZ), UCS, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_XYZ_to_UCS(self):
        """
        Test :func:`colour.models.cie_ucs.XYZ_to_UCS` definition domain and
        range scale support.
        """

        XYZ = np.array([0.0704953400, 0.1008000000, 0.0955831300])
        UCS = XYZ_to_UCS(XYZ)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    XYZ_to_UCS(XYZ * factor),
                    UCS * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_UCS(self):
        """Test :func:`colour.models.cie_ucs.XYZ_to_UCS` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        XYZ_to_UCS(cases)


class TestUCS_to_XYZ:
    """
    Define :func:`colour.models.cie_ucs.UCS_to_XYZ` definition unit tests
    methods.
    """

    def test_UCS_to_XYZ(self):
        """Test :func:`colour.models.cie_ucs.UCS_to_XYZ` definition."""

        np.testing.assert_allclose(
            UCS_to_XYZ(np.array([0.13769339, 0.12197225, 0.10537310])),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            UCS_to_XYZ(np.array([0.09481340, 0.23042768, 0.32701033])),
            np.array([0.14222010, 0.23042768, 0.10495772]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            UCS_to_XYZ(np.array([0.05212520, 0.06157201, 0.19376075])),
            np.array([0.07818780, 0.06157201, 0.28099326]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_UCS_to_XYZ(self):
        """
        Test :func:`colour.models.cie_ucs.UCS_to_XYZ` definition n-dimensional
        support.
        """

        UCS = np.array([0.13769339, 0.12197225, 0.10537310])
        XYZ = UCS_to_XYZ(UCS)

        UCS = np.tile(UCS, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_allclose(UCS_to_XYZ(UCS), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS)

        UCS = np.reshape(UCS, (2, 3, 3))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_allclose(UCS_to_XYZ(UCS), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_UCS_to_XYZ(self):
        """
        Test :func:`colour.models.cie_ucs.UCS_to_XYZ` definition domain and
        range scale support.
        """

        UCS = np.array([0.0469968933, 0.1008000000, 0.1637438950])
        XYZ = UCS_to_XYZ(UCS)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    UCS_to_XYZ(UCS * factor),
                    XYZ * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_UCS_to_XYZ(self):
        """Test :func:`colour.models.cie_ucs.UCS_to_XYZ` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        UCS_to_XYZ(cases)


class TestUCS_to_uv:
    """
    Define :func:`colour.models.cie_ucs.UCS_to_uv` definition unit tests
    methods.
    """

    def test_UCS_to_uv(self):
        """Test :func:`colour.models.cie_ucs.UCS_to_uv` definition."""

        np.testing.assert_allclose(
            UCS_to_uv(np.array([0.13769339, 0.12197225, 0.10537310])),
            np.array([0.37720213, 0.33413508]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            UCS_to_uv(np.array([0.09481340, 0.23042768, 0.32701033])),
            np.array([0.14536327, 0.35328046]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            UCS_to_uv(np.array([0.05212520, 0.06157201, 0.19376075])),
            np.array([0.16953602, 0.20026156]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_UCS_to_uv(self):
        """
        Test :func:`colour.models.cie_ucs.UCS_to_uv` definition n-dimensional
        support.
        """

        UCS = np.array([0.13769339, 0.12197225, 0.10537310])
        uv = UCS_to_uv(UCS)

        UCS = np.tile(UCS, (6, 1))
        uv = np.tile(uv, (6, 1))
        np.testing.assert_allclose(UCS_to_uv(UCS), uv, atol=TOLERANCE_ABSOLUTE_TESTS)

        UCS = np.reshape(UCS, (2, 3, 3))
        uv = np.reshape(uv, (2, 3, 2))
        np.testing.assert_allclose(UCS_to_uv(UCS), uv, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_UCS_to_uv(self):
        """
        Test :func:`colour.models.cie_ucs.UCS_to_uv` definition domain and
        range scale support.
        """

        UCS = np.array([0.0469968933, 0.1008000000, 0.1637438950])
        uv = UCS_to_uv(UCS)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    UCS_to_uv(UCS * factor), uv, atol=TOLERANCE_ABSOLUTE_TESTS
                )

    @ignore_numpy_errors
    def test_nan_UCS_to_uv(self):
        """Test :func:`colour.models.cie_ucs.UCS_to_uv` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        UCS_to_uv(cases)


class Testuv_to_UCS:
    """
    Define :func:`colour.models.cie_ucs.uv_to_UCS` definition unit tests
    methods.
    """

    def test_uv_to_UCS(self):
        """Test :func:`colour.models.cie_ucs.uv_to_UCS` definition."""

        np.testing.assert_allclose(
            uv_to_UCS(np.array([0.37720213, 0.33413508])),
            np.array([1.12889114, 1.00000000, 0.86391046]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            uv_to_UCS(np.array([0.14536327, 0.35328046])),
            np.array([0.41146705, 1.00000000, 1.41914520]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            uv_to_UCS(np.array([0.16953602, 0.20026156])),
            np.array([0.84657295, 1.00000000, 3.14689659]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            uv_to_UCS(np.array([0.37720213, 0.33413508]), V=0.18),
            np.array([0.20320040, 0.18000000, 0.15550388]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_uv_to_UCS(self):
        """
        Test :func:`colour.models.cie_ucs.uv_to_UCS` definition n-dimensional
        support.
        """

        uv = np.array([0.37720213, 0.33413508])
        UCS = uv_to_UCS(uv)

        uv = np.tile(uv, (6, 1))
        UCS = np.tile(UCS, (6, 1))
        np.testing.assert_allclose(uv_to_UCS(uv), UCS, atol=TOLERANCE_ABSOLUTE_TESTS)

        uv = np.reshape(uv, (2, 3, 2))
        UCS = np.reshape(UCS, (2, 3, 3))
        np.testing.assert_allclose(uv_to_UCS(uv), UCS, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_uv_to_UCS(self):
        """
        Test :func:`colour.models.cie_ucs.uv_to_UCS` definition domain and
        range scale support.
        """

        uv = np.array([0.37720213, 0.33413508])
        V = 1
        UCS = uv_to_UCS(uv, V)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    uv_to_UCS(uv, V * factor),
                    UCS * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_uv_to_UCS(self):
        """Test :func:`colour.models.cie_ucs.uv_to_UCS` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        uv_to_UCS(cases)


class TestUCS_uv_to_xy:
    """
    Define :func:`colour.models.cie_ucs.UCS_uv_to_xy` definition unit tests
    methods.
    """

    def test_UCS_uv_to_xy(self):
        """Test :func:`colour.models.cie_ucs.UCS_uv_to_xy` definition."""

        np.testing.assert_allclose(
            UCS_uv_to_xy(np.array([0.37720213, 0.33413508])),
            np.array([0.54369555, 0.32107941]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            UCS_uv_to_xy(np.array([0.14536327, 0.35328046])),
            np.array([0.29777734, 0.48246445]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            UCS_uv_to_xy(np.array([0.16953602, 0.20026156])),
            np.array([0.18582823, 0.14633764]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_UCS_uv_to_xy(self):
        """
        Test :func:`colour.models.cie_ucs.UCS_uv_to_xy` definition
        n-dimensional arrays support.
        """

        uv = np.array([0.37720213, 0.33413508])
        xy = UCS_uv_to_xy(uv)

        uv = np.tile(uv, (6, 1))
        xy = np.tile(xy, (6, 1))
        np.testing.assert_allclose(UCS_uv_to_xy(uv), xy, atol=TOLERANCE_ABSOLUTE_TESTS)

        uv = np.reshape(uv, (2, 3, 2))
        xy = np.reshape(xy, (2, 3, 2))
        np.testing.assert_allclose(UCS_uv_to_xy(uv), xy, atol=TOLERANCE_ABSOLUTE_TESTS)

    @ignore_numpy_errors
    def test_nan_UCS_uv_to_xy(self):
        """
        Test :func:`colour.models.cie_ucs.UCS_uv_to_xy` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        UCS_uv_to_xy(cases)


class TestXy_to_UCS_uv:
    """
    Define :func:`colour.models.cie_ucs.xy_to_UCS_uv` definition unit tests
    methods.
    """

    def test_xy_to_UCS_uv(self):
        """Test :func:`colour.models.cie_ucs.xy_to_UCS_uv` definition."""

        np.testing.assert_allclose(
            xy_to_UCS_uv(np.array([0.54369555, 0.32107941])),
            np.array([0.37720213, 0.33413508]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            xy_to_UCS_uv(np.array([0.29777734, 0.48246445])),
            np.array([0.14536327, 0.35328046]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            xy_to_UCS_uv(np.array([0.18582823, 0.14633764])),
            np.array([0.16953602, 0.20026156]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_xy_to_UCS_uv(self):
        """
        Test :func:`colour.models.cie_ucs.xy_to_UCS_uv` definition
        n-dimensional arrays support.
        """

        xy = np.array([0.54369555, 0.32107941])
        uv = xy_to_UCS_uv(xy)

        xy = np.tile(xy, (6, 1))
        uv = np.tile(uv, (6, 1))
        np.testing.assert_allclose(xy_to_UCS_uv(xy), uv, atol=TOLERANCE_ABSOLUTE_TESTS)

        xy = np.reshape(xy, (2, 3, 2))
        uv = np.reshape(uv, (2, 3, 2))
        np.testing.assert_allclose(xy_to_UCS_uv(xy), uv, atol=TOLERANCE_ABSOLUTE_TESTS)

    @ignore_numpy_errors
    def test_nan_xy_to_UCS_uv(self):
        """
        Test :func:`colour.models.cie_ucs.xy_to_UCS_uv` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        xy_to_UCS_uv(cases)


class TestXYZ_to_CIE1960UCS:
    """
    Define :func:`colour.models.cie_ucs.XYZ_to_CIE1960UCS` definition unit tests
    methods.
    """

    def test_XYZ_to_CIE1960UCS(self):
        """Test :func:`colour.models.cie_ucs.XYZ_to_CIE1960UCS` definition."""

        np.testing.assert_allclose(
            XYZ_to_CIE1960UCS(np.array([0.20654008, 0.12197225, 0.05136952])),
            np.array([0.37720213, 0.33413509, 0.12197225]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            XYZ_to_CIE1960UCS(np.array([0.14222010, 0.23042768, 0.10495772])),
            np.array([0.14536327, 0.35328046, 0.23042768]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            XYZ_to_CIE1960UCS(np.array([0.07818780, 0.06157201, 0.28099326])),
            np.array([0.16953603, 0.20026156, 0.06157201]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_XYZ_to_CIE1960UCS(self):
        """
        Test :func:`colour.models.cie_ucs.XYZ_to_CIE1960UCS` definition n-dimensional
        support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        uvV = XYZ_to_CIE1960UCS(XYZ)

        uvV = np.tile(uvV, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_allclose(
            XYZ_to_CIE1960UCS(XYZ), uvV, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        uvV = np.reshape(uvV, (2, 3, 3))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_allclose(
            XYZ_to_CIE1960UCS(XYZ), uvV, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_XYZ_to_CIE1960UCS(self):
        """
        Test :func:`colour.models.cie_ucs.XYZ_to_CIE1960UCS` definition domain and
        range scale support.
        """

        XYZ = np.array([0.0704953400, 0.1008000000, 0.0955831300])
        uvV = XYZ_to_CIE1960UCS(XYZ)

        d_r = (("reference", 1, 1), ("1", 1, 1), ("100", 100, np.array([1, 1, 100])))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    XYZ_to_CIE1960UCS(XYZ * factor_a),
                    uvV * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_CIE1960UCS(self):
        """
        Test :func:`colour.models.cie_ucs.XYZ_to_CIE1960UCS` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        XYZ_to_CIE1960UCS(cases)


class TestCIE1960UCS_to_XYZ:
    """
    Define :func:`colour.models.cie_ucs.CIE1960UCS_to_XYZ` definition unit tests
    methods.
    """

    def test_CIE1960UCS_to_XYZ(self):
        """Test :func:`colour.models.cie_ucs.CIE1960UCS_to_XYZ` definition."""

        np.testing.assert_allclose(
            CIE1960UCS_to_XYZ(np.array([0.37720213, 0.33413509, 0.12197225])),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            CIE1960UCS_to_XYZ(np.array([0.14536327, 0.35328046, 0.23042768])),
            np.array([0.14222010, 0.23042768, 0.10495772]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            CIE1960UCS_to_XYZ(np.array([0.16953603, 0.20026156, 0.06157201])),
            np.array([0.07818780, 0.06157201, 0.28099326]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_CIE1960UCS_to_XYZ(self):
        """
        Test :func:`colour.models.cie_ucs.CIE1960UCS_to_XYZ` definition n-dimensional
        support.
        """

        uvV = np.array([0.37720213, 0.33413509, 0.12197225])
        XYZ = CIE1960UCS_to_XYZ(uvV)

        uvV = np.tile(uvV, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_allclose(
            CIE1960UCS_to_XYZ(uvV), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        uvV = np.reshape(uvV, (2, 3, 3))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_allclose(
            CIE1960UCS_to_XYZ(uvV), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_CIE1960UCS_to_XYZ(self):
        """
        Test :func:`colour.models.cie_ucs.CIE1960UCS_to_XYZ` definition domain and
        range scale support.
        """

        uvV = np.array([0.0469968933, 0.1008000000, 0.1637438950])
        XYZ = CIE1960UCS_to_XYZ(uvV)

        d_r = (("reference", 1, 1), ("1", 1, 1), ("100", np.array([1, 1, 100]), 100))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    CIE1960UCS_to_XYZ(uvV * factor_a),
                    XYZ * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_CIE1960UCS_to_XYZ(self):
        """
        Test :func:`colour.models.cie_ucs.CIE1960UCS_to_XYZ` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        CIE1960UCS_to_XYZ(cases)
