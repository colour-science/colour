# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.models.ipt` module."""

import unittest
from itertools import product

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models import IPT_hue_angle, IPT_to_XYZ, XYZ_to_IPT
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestXYZ_to_IPT",
    "TestIPT_to_XYZ",
    "TestIPTHueAngle",
]


class TestXYZ_to_IPT(unittest.TestCase):
    """Define :func:`colour.models.ipt.XYZ_to_IPT` definition unit tests methods."""

    def test_XYZ_to_IPT(self):
        """Test :func:`colour.models.ipt.XYZ_to_IPT` definition."""

        np.testing.assert_allclose(
            XYZ_to_IPT(np.array([0.20654008, 0.12197225, 0.05136952])),
            np.array([0.38426191, 0.38487306, 0.18886838]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            XYZ_to_IPT(np.array([0.14222010, 0.23042768, 0.10495772])),
            np.array([0.49437481, -0.19251742, 0.18080304]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            XYZ_to_IPT(np.array([0.07818780, 0.06157201, 0.28099326])),
            np.array([0.35167774, -0.07525627, -0.30921279]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_XYZ_to_IPT(self):
        """
        Test :func:`colour.models.ipt.XYZ_to_IPT` definition n-dimensional
        support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        IPT = XYZ_to_IPT(XYZ)

        XYZ = np.tile(XYZ, (6, 1))
        IPT = np.tile(IPT, (6, 1))
        np.testing.assert_allclose(
            XYZ_to_IPT(XYZ), IPT, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        XYZ = np.reshape(XYZ, (2, 3, 3))
        IPT = np.reshape(IPT, (2, 3, 3))
        np.testing.assert_allclose(
            XYZ_to_IPT(XYZ), IPT, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_XYZ_to_IPT(self):
        """
        Test :func:`colour.models.ipt.XYZ_to_IPT` definition domain and
        range scale support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        IPT = XYZ_to_IPT(XYZ)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    XYZ_to_IPT(XYZ * factor),
                    IPT * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_IPT(self):
        """Test :func:`colour.models.ipt.XYZ_to_IPT` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        XYZ_to_IPT(cases)


class TestIPT_to_XYZ(unittest.TestCase):
    """
    Define :func:`colour.models.ipt.IPT_to_XYZ` definition unit tests
    methods.
    """

    def test_IPT_to_XYZ(self):
        """Test :func:`colour.models.ipt.IPT_to_XYZ` definition."""

        np.testing.assert_allclose(
            IPT_to_XYZ(np.array([0.38426191, 0.38487306, 0.18886838])),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            IPT_to_XYZ(np.array([0.49437481, -0.19251742, 0.18080304])),
            np.array([0.14222010, 0.23042768, 0.10495772]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            IPT_to_XYZ(np.array([0.35167774, -0.07525627, -0.30921279])),
            np.array([0.07818780, 0.06157201, 0.28099326]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_IPT_to_XYZ(self):
        """
        Test :func:`colour.models.ipt.IPT_to_XYZ` definition n-dimensional
        support.
        """

        IPT = np.array([0.38426191, 0.38487306, 0.18886838])
        XYZ = IPT_to_XYZ(IPT)

        IPT = np.tile(IPT, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_allclose(
            IPT_to_XYZ(IPT), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        IPT = np.reshape(IPT, (2, 3, 3))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_allclose(
            IPT_to_XYZ(IPT), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_IPT_to_XYZ(self):
        """
        Test :func:`colour.models.ipt.IPT_to_XYZ` definition domain and
        range scale support.
        """

        IPT = np.array([0.38426191, 0.38487306, 0.18886838])
        XYZ = IPT_to_XYZ(IPT)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    IPT_to_XYZ(IPT * factor),
                    XYZ * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_IPT_to_XYZ(self):
        """Test :func:`colour.models.ipt.IPT_to_XYZ` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        IPT_to_XYZ(cases)


class TestIPTHueAngle(unittest.TestCase):
    """
    Define :func:`colour.models.ipt.IPT_hue_angle` definition unit tests
    methods.
    """

    def test_IPT_hue_angle(self):
        """Test :func:`colour.models.ipt.IPT_hue_angle` definition."""

        np.testing.assert_allclose(
            IPT_hue_angle(np.array([0.20654008, 0.12197225, 0.05136952])),
            22.838754548625527,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            IPT_hue_angle(np.array([0.14222010, 0.23042768, 0.10495772])),
            24.488834912466245,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            IPT_hue_angle(np.array([0.07818780, 0.06157201, 0.28099326])),
            77.640533743711813,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_IPT_hue_angle(self):
        """
        Test :func:`colour.models.ipt.IPT_hue_angle` definition n-dimensional
        support.
        """

        IPT = np.array([0.20654008, 0.12197225, 0.05136952])
        hue = IPT_hue_angle(IPT)

        IPT = np.tile(IPT, (6, 1))
        hue = np.tile(hue, 6)
        np.testing.assert_allclose(
            IPT_hue_angle(IPT), hue, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        IPT = np.reshape(IPT, (2, 3, 3))
        hue = np.reshape(hue, (2, 3))
        np.testing.assert_allclose(
            IPT_hue_angle(IPT), hue, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_IPT_hue_angle(self):
        """
        Test :func:`colour.models.ipt.IPT_hue_angle` definition domain and
        range scale support.
        """

        IPT = np.array([0.20654008, 0.12197225, 0.05136952])
        hue = IPT_hue_angle(IPT)

        d_r = (("reference", 1, 1), ("1", 1, 1 / 360), ("100", 100, 1 / 3.6))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    IPT_hue_angle(IPT * factor_a),
                    hue * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_IPT_hue_angle(self):
        """Test :func:`colour.models.ipt.IPT_hue_angle` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        IPT_hue_angle(cases)


if __name__ == "__main__":
    unittest.main()
