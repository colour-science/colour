"""Define the unit tests for the :mod:`colour.models.yrg` module."""

from itertools import product

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models import (
    LMS_to_Yrg,
    XYZ_to_Yrg,
    Yrg_to_LMS,
    Yrg_to_XYZ,
)
from colour.utilities import domain_range_scale, ignore_numpy_errors

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

    def test_LMS_to_Yrg(self):
        """Test :func:`colour.models.yrg.LMS_to_Yrg` definition."""

        np.testing.assert_allclose(
            LMS_to_Yrg(np.array([0.15639195, 0.06741689, 0.03281398])),
            np.array([0.13137801, 0.49037644, 0.37777391]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            LMS_to_Yrg(np.array([0.23145723, 0.22601133, 0.05033211])),
            np.array([0.23840767, 0.20110504, 0.69668437]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            LMS_to_Yrg(np.array([1.07423297, 0.91295620, 0.61375713])),
            np.array([1.05911888, 0.22010094, 0.53660290]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_LMS_to_Yrg(self):
        """
        Test :func:`colour.models.yrg.LMS_to_Yrg` definition n-dimensional
        support.
        """

        LMS = np.array([0.15639195, 0.06741689, 0.03281398])
        Yrg = LMS_to_Yrg(LMS)

        LMS = np.tile(LMS, (6, 1))
        Yrg = np.tile(Yrg, (6, 1))
        np.testing.assert_allclose(LMS_to_Yrg(LMS), Yrg, atol=TOLERANCE_ABSOLUTE_TESTS)

        LMS = np.reshape(LMS, (2, 3, 3))
        Yrg = np.reshape(Yrg, (2, 3, 3))
        np.testing.assert_allclose(LMS_to_Yrg(LMS), Yrg, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_LMS_to_Yrg(self):
        """
        Test :func:`colour.models.yrg.LMS_to_Yrg` definition domain and range
        scale support.
        """

        LMS = np.array([0.15639195, 0.06741689, 0.03281398])
        Yrg = LMS_to_Yrg(LMS)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    LMS_to_Yrg(LMS * factor),
                    Yrg * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_LMS_to_Yrg(self):
        """Test :func:`colour.models.yrg.LMS_to_Yrg` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        LMS_to_Yrg(cases)


class TestYrg_to_LMS:
    """
    Define :func:`colour.models.yrg.Yrg_to_LMS` definition unit tests methods.
    """

    def test_Yrg_to_LMS(self):
        """Test :func:`colour.models.yrg.Yrg_to_LMS` definition."""

        np.testing.assert_allclose(
            Yrg_to_LMS(np.array([0.13137801, 0.49037644, 0.37777391])),
            np.array([0.15639195, 0.06741689, 0.03281398]),
            atol=1e-4,
        )

        np.testing.assert_allclose(
            Yrg_to_LMS(np.array([0.23840767, 0.20110504, 0.69668437])),
            np.array([0.23145723, 0.22601133, 0.05033211]),
            atol=1e-4,
        )

        np.testing.assert_allclose(
            Yrg_to_LMS(np.array([1.05911888, 0.22010094, 0.53660290])),
            np.array([1.07423297, 0.91295620, 0.61375713]),
            atol=1e-4,
        )

    def test_n_dimensional_Yrg_to_LMS(self):
        """
        Test :func:`colour.models.yrg.Yrg_to_LMS` definition n-dimensional
        support.
        """

        Yrg = np.array([0.00535048, 0.00924302, 0.00526007])
        LMS = Yrg_to_LMS(Yrg)

        Yrg = np.tile(Yrg, (6, 1))
        LMS = np.tile(LMS, (6, 1))
        np.testing.assert_allclose(Yrg_to_LMS(Yrg), LMS, atol=TOLERANCE_ABSOLUTE_TESTS)

        Yrg = np.reshape(Yrg, (2, 3, 3))
        LMS = np.reshape(LMS, (2, 3, 3))
        np.testing.assert_allclose(Yrg_to_LMS(Yrg), LMS, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_Yrg_to_LMS(self):
        """
        Test :func:`colour.models.yrg.Yrg_to_LMS` definition domain and range
        scale support.
        """

        Yrg = np.array([0.00535048, 0.00924302, 0.00526007])
        LMS = Yrg_to_LMS(Yrg)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    Yrg_to_LMS(Yrg * factor),
                    LMS * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_Yrg_to_LMS(self):
        """Test :func:`colour.models.yrg.Yrg_to_LMS` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        Yrg_to_LMS(cases)


class TestXYZ_to_Yrg:
    """
    Define :func:`colour.models.yrg.TestXYZ_to_Yrg` definition unit tests
    methods.
    """

    def test_XYZ_to_Yrg(self):
        """Test :func:`colour.models.yrg.XYZ_to_Yrg` definition."""

        np.testing.assert_allclose(
            XYZ_to_Yrg(np.array([0.20654008, 0.12197225, 0.05136952])),
            np.array([0.13137801, 0.49037645, 0.37777388]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            XYZ_to_Yrg(np.array([0.14222010, 0.23042768, 0.10495772])),
            np.array([0.23840767, 0.20110503, 0.69668437]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            XYZ_to_Yrg(np.array([0.96907232, 1.00000000, 1.12179215])),
            np.array([1.05911888, 0.22010094, 0.53660290]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_XYZ_to_Yrg(self):
        """
        Test :func:`colour.models.yrg.XYZ_to_Yrg` definition n-dimensional
        support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        Yrg = XYZ_to_Yrg(XYZ)

        XYZ = np.tile(XYZ, (6, 1))
        Yrg = np.tile(Yrg, (6, 1))
        np.testing.assert_allclose(XYZ_to_Yrg(XYZ), Yrg, atol=TOLERANCE_ABSOLUTE_TESTS)

        XYZ = np.reshape(XYZ, (2, 3, 3))
        Yrg = np.reshape(Yrg, (2, 3, 3))
        np.testing.assert_allclose(XYZ_to_Yrg(XYZ), Yrg, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_XYZ_to_Yrg(self):
        """
        Test :func:`colour.models.yrg.XYZ_to_Yrg` definition domain and range
        scale support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        Yrg = XYZ_to_Yrg(XYZ)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    XYZ_to_Yrg(XYZ * factor),
                    Yrg * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_Yrg(self):
        """Test :func:`colour.models.yrg.XYZ_to_Yrg` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        XYZ_to_Yrg(cases)


class TestYrg_to_XYZ:
    """
    Define :func:`colour.models.yrg.Yrg_to_XYZ` definition unit tests methods.
    """

    def test_Yrg_to_XYZ(self):
        """Test :func:`colour.models.yrg.Yrg_to_XYZ` definition."""

        np.testing.assert_allclose(
            Yrg_to_XYZ(np.array([0.13137801, 0.49037645, 0.37777388])),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            atol=1e-4,
        )

        np.testing.assert_allclose(
            Yrg_to_XYZ(np.array([0.23840767, 0.20110503, 0.69668437])),
            np.array([0.14222010, 0.23042768, 0.10495772]),
            atol=2e-4,
        )

        np.testing.assert_allclose(
            Yrg_to_XYZ(np.array([1.05911888, 0.22010094, 0.53660290])),
            np.array([0.96907232, 1.00000000, 1.12179215]),
            atol=2e-4,
        )

    def test_n_dimensional_Yrg_to_XYZ(self):
        """
        Test :func:`colour.models.yrg.Yrg_to_XYZ` definition n-dimensional
        support.
        """

        Yrg = np.array([0.13137801, 0.49037645, 0.37777388])
        XYZ = Yrg_to_XYZ(Yrg)

        Yrg = np.tile(Yrg, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_allclose(Yrg_to_XYZ(Yrg), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS)

        Yrg = np.reshape(Yrg, (2, 3, 3))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_allclose(Yrg_to_XYZ(Yrg), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_Yrg_to_XYZ(self):
        """
        Test :func:`colour.models.yrg.Yrg_to_XYZ` definition domain and range
        scale support.
        """

        Yrg = np.array([0.13137801, 0.49037645, 0.37777388])
        XYZ = Yrg_to_XYZ(Yrg)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    Yrg_to_XYZ(Yrg * factor),
                    XYZ * factor,
                    atol=1e-4,
                )

    @ignore_numpy_errors
    def test_nan_Yrg_to_XYZ(self):
        """Test :func:`colour.models.yrg.Yrg_to_XYZ` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        Yrg_to_XYZ(cases)
