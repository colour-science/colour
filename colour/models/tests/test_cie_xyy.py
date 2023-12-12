# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.models.cie_xyy` module."""

import unittest
from itertools import product

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models import (
    XYZ_to_xy,
    XYZ_to_xyY,
    xy_to_xyY,
    xy_to_XYZ,
    xyY_to_xy,
    xyY_to_XYZ,
)
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestXYZ_to_xyY",
    "TestxyY_to_XYZ",
    "TestxyY_to_xy",
    "Testxy_to_xyY",
    "TestXYZ_to_xy",
    "Testxy_to_XYZ",
]


class TestXYZ_to_xyY(unittest.TestCase):
    """
    Define :func:`colour.models.cie_xyy.XYZ_to_xyY` definition unit tests
    methods.
    """

    def test_XYZ_to_xyY(self):
        """Test :func:`colour.models.cie_xyy.XYZ_to_xyY` definition."""

        np.testing.assert_allclose(
            XYZ_to_xyY(np.array([0.20654008, 0.12197225, 0.05136952])),
            np.array([0.54369557, 0.32107944, 0.12197225]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            XYZ_to_xyY(np.array([0.14222010, 0.23042768, 0.10495772])),
            np.array([0.29777735, 0.48246446, 0.23042768]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            XYZ_to_xyY(np.array([0.07818780, 0.06157201, 0.28099326])),
            np.array([0.18582823, 0.14633764, 0.06157201]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            XYZ_to_xyY(np.array([0.00000000, 0.00000000, 1.00000000])),
            np.array([0.00000000, 0.00000000, 0.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            XYZ_to_xyY(
                np.array(
                    [
                        [0.20654008, 0.12197225, 0.05136952],
                        [0.00000000, 0.00000000, 0.00000000],
                        [0.00000000, 1.00000000, 0.00000000],
                    ]
                )
            ),
            np.array(
                [
                    [0.54369557, 0.32107944, 0.12197225],
                    [0.00000000, 0.00000000, 0.00000000],
                    [0.00000000, 1.00000000, 1.00000000],
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_XYZ_to_xyY(self):
        """
        Test :func:`colour.models.cie_xyy.XYZ_to_xyY` definition n-dimensional
        support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        xyY = XYZ_to_xyY(XYZ)

        XYZ = np.tile(XYZ, (6, 1))
        xyY = np.tile(xyY, (6, 1))
        np.testing.assert_allclose(
            XYZ_to_xyY(XYZ), xyY, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        XYZ = np.reshape(XYZ, (2, 3, 3))
        xyY = np.reshape(xyY, (2, 3, 3))
        np.testing.assert_allclose(
            XYZ_to_xyY(XYZ), xyY, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_XYZ_to_xyY(self):
        """
        Test :func:`colour.models.cie_xyy.XYZ_to_xyY` definition domain and
        range scale support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        xyY = XYZ_to_xyY(XYZ)
        XYZ = np.reshape(np.tile(XYZ, (6, 1)), (2, 3, 3))
        xyY = np.reshape(np.tile(xyY, (6, 1)), (2, 3, 3))

        d_r = (
            ("reference", 1, 1),
            ("1", 1, 1),
            ("100", 100, np.array([1, 1, 100])),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    XYZ_to_xyY(XYZ * factor_a),
                    xyY * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_xyY(self):
        """Test :func:`colour.models.cie_xyy.XYZ_to_xyY` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        XYZ_to_xyY(cases)


class TestxyY_to_XYZ(unittest.TestCase):
    """
    Define :func:`colour.models.cie_xyy.xyY_to_XYZ` definition unit tests
    methods.
    """

    def test_xyY_to_XYZ(self):
        """Test :func:`colour.models.cie_xyy.xyY_to_XYZ` definition."""

        np.testing.assert_allclose(
            xyY_to_XYZ(np.array([0.54369557, 0.32107944, 0.12197225])),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            xyY_to_XYZ(np.array([0.29777735, 0.48246446, 0.23042768])),
            np.array([0.14222010, 0.23042768, 0.10495772]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            xyY_to_XYZ(np.array([0.18582823, 0.14633764, 0.06157201])),
            np.array([0.07818780, 0.06157201, 0.28099326]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            xyY_to_XYZ(np.array([0.34567, 0.3585, 0.00000000])),
            np.array([0.00000000, 0.00000000, 0.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            xyY_to_XYZ(
                np.array(
                    [
                        [0.54369557, 0.32107944, 0.12197225],
                        [0.31270000, 0.32900000, 0.00000000],
                        [0.00000000, 1.00000000, 1.00000000],
                    ]
                )
            ),
            np.array(
                [
                    [0.20654008, 0.12197225, 0.05136952],
                    [0.00000000, 0.00000000, 0.00000000],
                    [0.00000000, 1.00000000, 0.00000000],
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_xyY_to_XYZ(self):
        """
        Test :func:`colour.models.cie_xyy.xyY_to_XYZ` definition n-dimensional
        support.
        """

        xyY = np.array([0.54369557, 0.32107944, 0.12197225])
        XYZ = xyY_to_XYZ(xyY)

        xyY = np.tile(xyY, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_allclose(
            xyY_to_XYZ(xyY), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        xyY = np.reshape(xyY, (2, 3, 3))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_allclose(
            xyY_to_XYZ(xyY), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_xyY_to_XYZ(self):
        """
        Test :func:`colour.models.cie_xyy.xyY_to_XYZ` definition domain and
        range scale support.
        """

        xyY = np.array([0.54369557, 0.32107944, 0.12197225])
        XYZ = xyY_to_XYZ(xyY)
        xyY = np.reshape(np.tile(xyY, (6, 1)), (2, 3, 3))
        XYZ = np.reshape(np.tile(XYZ, (6, 1)), (2, 3, 3))

        d_r = (
            ("reference", 1, 1),
            ("1", 1, 1),
            ("100", np.array([1, 1, 100]), 100),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    xyY_to_XYZ(xyY * factor_a),
                    XYZ * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_xyY_to_XYZ(self):
        """Test :func:`colour.models.cie_xyy.xyY_to_XYZ` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        xyY_to_XYZ(cases)


class TestxyY_to_xy(unittest.TestCase):
    """
    Define :func:`colour.models.cie_xyy.xyY_to_xy` definition unit tests
    methods.
    """

    def test_xyY_to_xy(self):
        """Test :func:`colour.models.cie_xyy.xyY_to_xy` definition."""

        np.testing.assert_allclose(
            xyY_to_xy(np.array([0.54369557, 0.32107944, 0.12197225])),
            np.array([0.54369557, 0.32107944]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            xyY_to_xy(np.array([0.29777735, 0.48246446, 0.23042768])),
            np.array([0.29777735, 0.48246446]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            xyY_to_xy(np.array([0.18582823, 0.14633764, 0.06157201])),
            np.array([0.18582823, 0.14633764]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            xyY_to_xy(np.array([0.31270, 0.32900])),
            np.array([0.31270000, 0.32900000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_xyY_to_xy(self):
        """
        Test :func:`colour.models.cie_xyy.xyY_to_xy` definition n-dimensional
        support.
        """

        xyY = np.array([0.54369557, 0.32107944, 0.12197225])
        xy = xyY_to_xy(xyY)

        xyY = np.tile(xyY, (6, 1))
        xy = np.tile(xy, (6, 1))
        np.testing.assert_allclose(
            xyY_to_xy(xyY), xy, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        xyY = np.reshape(xyY, (2, 3, 3))
        xy = np.reshape(xy, (2, 3, 2))
        np.testing.assert_allclose(
            xyY_to_xy(xyY), xy, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_xyY_to_xy(self):
        """
        Test :func:`colour.models.cie_xyy.xyY_to_xy` definition domain and
        range scale support.
        """

        xyY = np.array([0.54369557, 0.32107944, 0.12197225])
        xy = xyY_to_xy(xyY)
        xyY = np.reshape(np.tile(xyY, (6, 1)), (2, 3, 3))
        xy = np.reshape(np.tile(xy, (6, 1)), (2, 3, 2))

        d_r = (
            ("reference", 1, 1),
            ("1", 1, 1),
            ("100", np.array([1, 1, 100]), 1),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    xyY_to_xy(xyY * factor_a),
                    xy * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_xyY_to_xy(self):
        """Test :func:`colour.models.cie_xyy.xyY_to_xy` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        xyY_to_xy(cases)


class Testxy_to_xyY(unittest.TestCase):
    """
    Define :func:`colour.models.cie_xyy.xy_to_xyY` definition unit tests
    methods.
    """

    def test_xy_to_xyY(self):
        """Test :func:`colour.models.cie_xyy.xy_to_xyY` definition."""

        np.testing.assert_allclose(
            xy_to_xyY(np.array([0.54369557, 0.32107944])),
            np.array([0.54369557, 0.32107944, 1.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            xy_to_xyY(np.array([0.29777735, 0.48246446])),
            np.array([0.29777735, 0.48246446, 1.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            xy_to_xyY(np.array([0.18582823, 0.14633764])),
            np.array([0.18582823, 0.14633764, 1.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            xy_to_xyY(np.array([0.31270000, 0.32900000, 1.00000000])),
            np.array([0.31270000, 0.32900000, 1.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            xy_to_xyY(np.array([0.31270000, 0.32900000]), 100),
            np.array([0.31270000, 0.32900000, 100.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_xy_to_xyY(self):
        """
        Test :func:`colour.models.cie_xyy.xy_to_xyY` definition n-dimensional
        support.
        """

        xy = np.array([0.54369557, 0.32107944])
        xyY = xy_to_xyY(xy)

        xy = np.tile(xy, (6, 1))
        xyY = np.tile(xyY, (6, 1))
        np.testing.assert_allclose(
            xy_to_xyY(xy), xyY, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        xy = np.reshape(xy, (2, 3, 2))
        xyY = np.reshape(xyY, (2, 3, 3))
        np.testing.assert_allclose(
            xy_to_xyY(xy), xyY, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_xy_to_xyY(self):
        """
        Test :func:`colour.models.cie_xyy.xy_to_xyY` definition domain and
        range scale support.
        """

        xy = np.array([0.54369557, 0.32107944, 0.12197225])
        xyY = xy_to_xyY(xy)
        xy = np.reshape(np.tile(xy, (6, 1)), (2, 3, 3))
        xyY = np.reshape(np.tile(xyY, (6, 1)), (2, 3, 3))

        d_r = (
            ("reference", 1, 1),
            (1, 1, 1),
            (
                100,
                np.array([1, 1, 100]),
                np.array([1, 1, 100]),
            ),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    xy_to_xyY(xy * factor_a),
                    xyY * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_xy_to_xyY(self):
        """Test :func:`colour.models.cie_xyy.xy_to_xyY` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        xy_to_xyY(cases)


class TestXYZ_to_xy(unittest.TestCase):
    """
    Define :func:`colour.models.cie_xyy.XYZ_to_xy` definition unit tests
    methods.
    """

    def test_XYZ_to_xy(self):
        """Test :func:`colour.models.cie_xyy.XYZ_to_xy` definition."""

        np.testing.assert_allclose(
            XYZ_to_xy(np.array([0.20654008, 0.12197225, 0.05136952])),
            np.array([0.54369557, 0.32107944]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            XYZ_to_xy(np.array([0.14222010, 0.23042768, 0.10495772])),
            np.array([0.29777735, 0.48246446]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            XYZ_to_xy(np.array([0.07818780, 0.06157201, 0.28099326])),
            np.array([0.18582823, 0.14633764]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            XYZ_to_xy(np.array([0.00000000, 0.00000000, 0.00000000])),
            np.array([0.00000000, 0.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_XYZ_to_xy(self):
        """
        Test :func:`colour.models.cie_xyy.XYZ_to_xy` definition n-dimensional
        support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        xy = XYZ_to_xy(XYZ)

        XYZ = np.tile(XYZ, (6, 1))
        xy = np.tile(xy, (6, 1))
        np.testing.assert_allclose(
            XYZ_to_xy(XYZ), xy, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        XYZ = np.reshape(XYZ, (2, 3, 3))
        xy = np.reshape(xy, (2, 3, 2))
        np.testing.assert_allclose(
            XYZ_to_xy(XYZ), xy, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_XYZ_to_xy(self):
        """
        Test :func:`colour.models.cie_xyy.XYZ_to_xy` definition domain and
        range scale support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        xy = XYZ_to_xy(XYZ)
        XYZ = np.reshape(np.tile(XYZ, (6, 1)), (2, 3, 3))
        xy = np.reshape(np.tile(xy, (6, 1)), (2, 3, 2))

        d_r = (("reference", 1), ("1", 1), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    XYZ_to_xy(XYZ * factor), xy, atol=TOLERANCE_ABSOLUTE_TESTS
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_xy(self):
        """Test :func:`colour.models.cie_xyy.XYZ_to_xy` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        XYZ_to_xy(cases)


class Testxy_to_XYZ(unittest.TestCase):
    """
    Define :func:`colour.models.cie_xyy.xy_to_XYZ` definition unit tests
    methods.
    """

    def test_xy_to_XYZ(self):
        """Test :func:`colour.models.cie_xyy.xy_to_XYZ` definition."""

        np.testing.assert_allclose(
            xy_to_XYZ(np.array([0.54369557, 0.32107944])),
            np.array([1.69333661, 1.00000000, 0.42115742]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            xy_to_XYZ(np.array([0.29777735, 0.48246446])),
            np.array([0.61720059, 1.00000000, 0.45549094]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            xy_to_XYZ(np.array([0.18582823, 0.14633764])),
            np.array([1.26985942, 1.00000000, 4.56365245]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            xy_to_XYZ(np.array([0.31270000, 0.32900000])),
            np.array([0.95045593, 1.00000000, 1.08905775]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_xy_to_XYZ(self):
        """
        Test :func:`colour.models.cie_xyy.xy_to_XYZ` definition n-dimensional
        support.
        """

        xy = np.array([0.54369557, 0.32107944])
        XYZ = xy_to_XYZ(xy)

        xy = np.tile(xy, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_allclose(
            xy_to_XYZ(xy), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        xy = np.reshape(xy, (2, 3, 2))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_allclose(
            xy_to_XYZ(xy), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_xy_to_XYZ(self):
        """
        Test :func:`colour.models.cie_xyy.xy_to_XYZ` definition domain and
        range scale support.
        """

        xy = np.array([0.54369557, 0.32107944, 0.12197225])
        XYZ = xy_to_XYZ(xy)
        xy = np.reshape(np.tile(xy, (6, 1)), (2, 3, 3))
        XYZ = np.reshape(np.tile(XYZ, (6, 1)), (2, 3, 3))

        d_r = (
            ("reference", 1, 1),
            ("1", 1, 1),
            ("100", np.array([1, 1, 100]), 100),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    xy_to_XYZ(xy * factor_a),
                    XYZ * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_xy_to_XYZ(self):
        """Test :func:`colour.models.cie_xyy.xy_to_XYZ` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        xy_to_XYZ(cases)


if __name__ == "__main__":
    unittest.main()
