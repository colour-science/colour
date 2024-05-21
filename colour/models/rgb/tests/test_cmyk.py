"""Define the unit tests for the :mod:`colour.models.rgb.cmyk` module."""

from itertools import product

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.cmyk import (
    CMY_to_CMYK,
    CMY_to_RGB,
    CMYK_to_CMY,
    RGB_to_CMY,
)
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestRGB_to_CMY",
    "TestCMY_to_RGB",
    "TestCMY_to_CMYK",
    "TestCMYK_to_CMY",
]


class TestRGB_to_CMY:
    """
    Define :func:`colour.models.rgb.cmyk.RGB_to_CMY` definition unit tests
    methods.
    """

    def test_RGB_to_CMY(self):
        """Test :func:`colour.models.rgb.cmyk.RGB_to_CMY` definition."""

        np.testing.assert_allclose(
            RGB_to_CMY(np.array([0.45620519, 0.03081071, 0.04091952])),
            np.array([0.54379481, 0.96918929, 0.95908048]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            RGB_to_CMY(np.array([0.00000000, 0.00000000, 0.00000000])),
            np.array([1.00000000, 1.00000000, 1.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            RGB_to_CMY(np.array([1.00000000, 1.00000000, 1.00000000])),
            np.array([0.00000000, 0.00000000, 0.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_RGB_to_CMY(self):
        """
        Test :func:`colour.models.rgb.cmyk.RGB_to_CMY` definition
        n-dimensional arrays support.
        """

        RGB = np.array([0.45620519, 0.03081071, 0.04091952])
        CMY = RGB_to_CMY(RGB)

        RGB = np.tile(RGB, (6, 1))
        CMY = np.tile(CMY, (6, 1))
        np.testing.assert_allclose(RGB_to_CMY(RGB), CMY, atol=TOLERANCE_ABSOLUTE_TESTS)

        RGB = np.reshape(RGB, (2, 3, 3))
        CMY = np.reshape(CMY, (2, 3, 3))
        np.testing.assert_allclose(RGB_to_CMY(RGB), CMY, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_RGB_to_CMY(self):
        """
        Test :func:`colour.models.rgb.cmyk.RGB_to_CMY` definition domain and
        range scale support.
        """

        RGB = np.array([0.45620519, 0.03081071, 0.04091952])
        CMY = RGB_to_CMY(RGB)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    RGB_to_CMY(RGB * factor),
                    CMY * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_RGB_to_CMY(self):
        """
        Test :func:`colour.models.rgb.cmyk.RGB_to_CMY` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        RGB_to_CMY(cases)


class TestCMY_to_RGB:
    """
    Define :func:`colour.models.rgb.cmyk.CMY_to_RGB` definition unit tests
    methods.
    """

    def test_CMY_to_RGB(self):
        """Test :func:`colour.models.rgb.cmyk.CMY_to_RGB` definition."""

        np.testing.assert_allclose(
            CMY_to_RGB(np.array([0.54379481, 0.96918929, 0.95908048])),
            np.array([0.45620519, 0.03081071, 0.04091952]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            CMY_to_RGB(np.array([1.00000000, 1.00000000, 1.00000000])),
            np.array([0.00000000, 0.00000000, 0.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            CMY_to_RGB(np.array([0.00000000, 0.00000000, 0.00000000])),
            np.array([1.00000000, 1.00000000, 1.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_CMY_to_RGB(self):
        """
        Test :func:`colour.models.rgb.cmyk.CMY_to_RGB` definition
        n-dimensional arrays support.
        """

        CMY = np.array([0.54379481, 0.96918929, 0.95908048])
        RGB = CMY_to_RGB(CMY)

        CMY = np.tile(CMY, (6, 1))
        RGB = np.tile(RGB, (6, 1))
        np.testing.assert_allclose(CMY_to_RGB(CMY), RGB, atol=TOLERANCE_ABSOLUTE_TESTS)

        CMY = np.reshape(CMY, (2, 3, 3))
        RGB = np.reshape(RGB, (2, 3, 3))
        np.testing.assert_allclose(CMY_to_RGB(CMY), RGB, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_CMY_to_RGB(self):
        """
        Test :func:`colour.models.rgb.cmyk.CMY_to_RGB` definition domain and
        range scale support.
        """

        CMY = np.array([0.54379481, 0.96918929, 0.95908048])
        RGB = CMY_to_RGB(CMY)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    CMY_to_RGB(CMY * factor),
                    RGB * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_CMY_to_RGB(self):
        """Test :func:`colour.models.rgb.cmyk.CMY_to_RGB` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        CMY_to_RGB(cases)


class TestCMY_to_CMYK:
    """
    Define :func:`colour.models.rgb.cmyk.CMY_to_CMYK` definition unit tests
    methods.
    """

    def test_CMY_to_CMYK(self):
        """Test :func:`colour.models.rgb.cmyk.CMY_to_CMYK` definition."""

        np.testing.assert_allclose(
            CMY_to_CMYK(np.array([0.54379481, 0.96918929, 0.95908048])),
            np.array([0.00000000, 0.93246304, 0.91030457, 0.54379481]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            CMY_to_CMYK(np.array([0.15000000, 1.00000000, 1.00000000])),
            np.array([0.00000000, 1.00000000, 1.00000000, 0.15000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            CMY_to_CMYK(np.array([0.15000000, 0.00000000, 0.00000000])),
            np.array([0.15000000, 0.00000000, 0.00000000, 0.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_CMY_to_CMYK(self):
        """
        Test :func:`colour.models.rgb.cmyk.CMY_to_CMYK` definition
        n-dimensional arrays support.
        """

        CMY = np.array([0.54379481, 0.96918929, 0.95908048])
        CMYK = CMY_to_CMYK(CMY)

        CMY = np.tile(CMY, (6, 1))
        CMYK = np.tile(CMYK, (6, 1))
        np.testing.assert_allclose(
            CMY_to_CMYK(CMY), CMYK, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        CMY = np.reshape(CMY, (2, 3, 3))
        CMYK = np.reshape(CMYK, (2, 3, 4))
        np.testing.assert_allclose(
            CMY_to_CMYK(CMY), CMYK, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_CMY_to_CMYK(self):
        """
        Test :func:`colour.models.rgb.cmyk.CMY_to_CMYK` definition domain and
        range scale support.
        """

        CMY = np.array([0.54379481, 0.96918929, 0.95908048])
        CMYK = CMY_to_CMYK(CMY)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    CMY_to_CMYK(CMY * factor),
                    CMYK * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_CMY_to_CMYK(self):
        """
        Test :func:`colour.models.rgb.cmyk.CMY_to_CMYK` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        CMY_to_CMYK(cases)


class TestCMYK_to_CMY:
    """
    Define :func:`colour.models.rgb.cmyk.CMYK_to_CMY` definition unit tests
    methods.
    """

    def test_CMYK_to_CMY(self):
        """Test :func:`colour.models.rgb.cmyk.CMYK_to_CMY` definition."""

        np.testing.assert_allclose(
            CMYK_to_CMY(np.array([0.00000000, 0.93246304, 0.91030457, 0.54379481])),
            np.array([0.54379481, 0.96918929, 0.95908048]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            CMYK_to_CMY(np.array([0.00000000, 1.00000000, 1.00000000, 0.15000000])),
            np.array([0.15000000, 1.00000000, 1.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            CMYK_to_CMY(np.array([0.15000000, 0.00000000, 0.00000000, 0.00000000])),
            np.array([0.15000000, 0.00000000, 0.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_CMYK_to_CMY(self):
        """
        Test :func:`colour.models.rgb.cmyk.CMYK_to_CMY` definition
        n-dimensional arrays support.
        """

        CMYK = np.array([0.00000000, 0.93246304, 0.91030457, 0.54379481])
        CMY = CMYK_to_CMY(CMYK)

        CMYK = np.tile(CMYK, (6, 1))
        CMY = np.tile(CMY, (6, 1))
        np.testing.assert_allclose(
            CMYK_to_CMY(CMYK), CMY, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        CMYK = np.reshape(CMYK, (2, 3, 4))
        CMY = np.reshape(CMY, (2, 3, 3))
        np.testing.assert_allclose(
            CMYK_to_CMY(CMYK), CMY, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_CMYK_to_CMY(self):
        """
        Test :func:`colour.models.rgb.cmyk.CMYK_to_CMY` definition domain and
        range scale support.
        """

        CMYK = np.array([0.00000000, 0.93246304, 0.91030457, 0.54379481])
        CMY = CMYK_to_CMY(CMYK)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    CMYK_to_CMY(CMYK * factor),
                    CMY * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_CMYK_to_CMY(self):
        """
        Test :func:`colour.models.rgb.cmyk.CMYK_to_CMY` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=4))))
        CMYK_to_CMY(cases)
