# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.colorimetry.dominant` module."""

import unittest
from itertools import product

import numpy as np

from colour.colorimetry import (
    CCS_ILLUMINANTS,
    MSDS_CMFS,
    colorimetric_purity,
    complementary_wavelength,
    dominant_wavelength,
    excitation_purity,
)
from colour.colorimetry.dominant import (
    closest_spectral_locus_wavelength,
)
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models import XYZ_to_xy
from colour.utilities import ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestClosestSpectralLocusWavelength",
    "TestDominantWavelength",
    "TestComplementaryWavelength",
    "TestExcitationPurity",
    "TestColorimetricPurity",
]


class TestClosestSpectralLocusWavelength(unittest.TestCase):
    """
    Define :func:`colour.colorimetry.dominant.\
closest_spectral_locus_wavelength` definition unit tests methods.
    """

    def setUp(self):
        """Initialise the common tests attributes."""

        self._xy_s = XYZ_to_xy(
            MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].values
        )

        self._xy_D65 = CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"][
            "D65"
        ]

    def test_closest_spectral_locus_wavelength(self):
        """
        Test :func:`colour.colorimetry.dominant.\
closest_spectral_locus_wavelength` definition.
        """

        xy = np.array([0.54369557, 0.32107944])
        xy_n = self._xy_D65
        i_wl, xy_wl = closest_spectral_locus_wavelength(xy, xy_n, self._xy_s)

        self.assertEqual(i_wl, np.array(256))
        np.testing.assert_allclose(
            xy_wl,
            np.array([0.68354746, 0.31628409]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xy = np.array([0.37605506, 0.24452225])
        i_wl, xy_wl = closest_spectral_locus_wavelength(xy, xy_n, self._xy_s)

        self.assertEqual(i_wl, np.array(248))
        np.testing.assert_allclose(
            xy_wl,
            np.array([0.45723147, 0.13628148]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_closest_spectral_locus_wavelength(self):
        """
        Test :func:`colour.colorimetry.dominant.\
closest_spectral_locus_wavelength` definition n-dimensional arrays support.
        """

        xy = np.array([0.54369557, 0.32107944])
        xy_n = self._xy_D65
        i_wl, xy_wl = closest_spectral_locus_wavelength(xy, xy_n, self._xy_s)
        i_wl_r, xy_wl_r = np.array(256), np.array([0.68354746, 0.31628409])
        np.testing.assert_allclose(i_wl, i_wl_r, atol=TOLERANCE_ABSOLUTE_TESTS)
        np.testing.assert_allclose(
            xy_wl, xy_wl_r, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        xy = np.tile(xy, (6, 1))
        xy_n = np.tile(xy_n, (6, 1))
        i_wl, xy_wl = closest_spectral_locus_wavelength(xy, xy_n, self._xy_s)
        i_wl_r = np.tile(i_wl_r, 6)
        xy_wl_r = np.tile(xy_wl_r, (6, 1))
        np.testing.assert_allclose(i_wl, i_wl_r, atol=TOLERANCE_ABSOLUTE_TESTS)
        np.testing.assert_allclose(
            xy_wl, xy_wl_r, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        xy = np.reshape(xy, (2, 3, 2))
        xy_n = np.reshape(xy_n, (2, 3, 2))
        i_wl, xy_wl = closest_spectral_locus_wavelength(xy, xy_n, self._xy_s)
        i_wl_r = np.reshape(i_wl_r, (2, 3))
        xy_wl_r = np.reshape(xy_wl_r, (2, 3, 2))
        np.testing.assert_allclose(i_wl, i_wl_r, atol=TOLERANCE_ABSOLUTE_TESTS)
        np.testing.assert_allclose(
            xy_wl, xy_wl_r, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    @ignore_numpy_errors
    def test_nan_closest_spectral_locus_wavelength(self):
        """
        Test :func:`colour.colorimetry.dominant.\
closest_spectral_locus_wavelength` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        for case in cases:
            closest_spectral_locus_wavelength(case, case, self._xy_s)


class TestDominantWavelength(unittest.TestCase):
    """
    Define :func:`colour.colorimetry.dominant.dominant_wavelength` definition
    unit tests methods.
    """

    def setUp(self):
        """Initialise the common tests attributes."""

        self._xy_D65 = CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"][
            "D65"
        ]

    def test_dominant_wavelength(self):
        """
        Test :func:`colour.colorimetry.dominant.dominant_wavelength`
        definition.
        """

        xy = np.array([0.54369557, 0.32107944])
        xy_n = self._xy_D65
        wl, xy_wl, xy_cwl = dominant_wavelength(xy, xy_n)

        self.assertEqual(wl, np.array(616.0))
        np.testing.assert_allclose(
            xy_wl,
            np.array([0.68354746, 0.31628409]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        np.testing.assert_allclose(
            xy_cwl,
            np.array([0.68354746, 0.31628409]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xy = np.array([0.37605506, 0.24452225])
        i_wl, xy_wl, xy_cwl = dominant_wavelength(xy, xy_n)

        self.assertEqual(i_wl, np.array(-509.0))
        np.testing.assert_allclose(
            xy_wl,
            np.array([0.45723147, 0.13628148]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        np.testing.assert_allclose(
            xy_cwl,
            np.array([0.01040962, 0.73207453]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_dominant_wavelength(self):
        """
        Test :func:`colour.colorimetry.dominant.dominant_wavelength`
        definition n-dimensional arrays support.
        """

        xy = np.array([0.54369557, 0.32107944])
        xy_n = self._xy_D65
        wl, xy_wl, xy_cwl = dominant_wavelength(xy, xy_n)
        wl_r, xy_wl_r, xy_cwl_r = (
            np.array(616.0),
            np.array([0.68354746, 0.31628409]),
            np.array([0.68354746, 0.31628409]),
        )
        np.testing.assert_allclose(wl, wl_r, atol=TOLERANCE_ABSOLUTE_TESTS)
        np.testing.assert_allclose(
            xy_wl, xy_wl_r, atol=TOLERANCE_ABSOLUTE_TESTS
        )
        np.testing.assert_allclose(
            xy_cwl, xy_cwl_r, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        xy = np.tile(xy, (6, 1))
        xy_n = np.tile(xy_n, (6, 1))
        wl, xy_wl, xy_cwl = dominant_wavelength(xy, xy_n)
        wl_r = np.tile(wl_r, 6)
        xy_wl_r = np.tile(xy_wl_r, (6, 1))
        xy_cwl_r = np.tile(xy_cwl_r, (6, 1))
        np.testing.assert_allclose(wl, wl_r, atol=TOLERANCE_ABSOLUTE_TESTS)
        np.testing.assert_allclose(
            xy_wl, xy_wl_r, atol=TOLERANCE_ABSOLUTE_TESTS
        )
        np.testing.assert_allclose(
            xy_cwl, xy_cwl_r, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        xy = np.reshape(xy, (2, 3, 2))
        xy_n = np.reshape(xy_n, (2, 3, 2))
        wl, xy_wl, xy_cwl = dominant_wavelength(xy, xy_n)
        wl_r = np.reshape(wl_r, (2, 3))
        xy_wl_r = np.reshape(xy_wl_r, (2, 3, 2))
        xy_cwl_r = np.reshape(xy_cwl_r, (2, 3, 2))
        np.testing.assert_allclose(wl, wl_r, atol=TOLERANCE_ABSOLUTE_TESTS)
        np.testing.assert_allclose(
            xy_wl, xy_wl_r, atol=TOLERANCE_ABSOLUTE_TESTS
        )
        np.testing.assert_allclose(
            xy_cwl, xy_cwl_r, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    @ignore_numpy_errors
    def test_nan_dominant_wavelength(self):
        """
        Test :func:`colour.colorimetry.dominant.dominant_wavelength`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        for case in cases:
            dominant_wavelength(case, case)


class TestComplementaryWavelength(unittest.TestCase):
    """
    Define :func:`colour.colorimetry.dominant.complementary_wavelength`
    definition unit tests methods.
    """

    def setUp(self):
        """Initialise the common tests attributes."""

        self._xy_D65 = CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"][
            "D65"
        ]

    def test_complementary_wavelength(self):
        """
        Test :func:`colour.colorimetry.dominant.complementary_wavelength`
        definition.
        """

        xy = np.array([0.54369557, 0.32107944])
        xy_n = self._xy_D65
        wl, xy_wl, xy_cwl = complementary_wavelength(xy, xy_n)

        self.assertEqual(wl, np.array(492.0))
        np.testing.assert_allclose(
            xy_wl,
            np.array([0.03647950, 0.33847127]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        np.testing.assert_allclose(
            xy_cwl,
            np.array([0.03647950, 0.33847127]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xy = np.array([0.37605506, 0.24452225])
        i_wl, xy_wl, xy_cwl = complementary_wavelength(xy, xy_n)

        self.assertEqual(i_wl, np.array(509.0))
        np.testing.assert_allclose(
            xy_wl,
            np.array([0.01040962, 0.73207453]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        np.testing.assert_allclose(
            xy_cwl,
            np.array([0.01040962, 0.73207453]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_complementary_wavelength(self):
        """
        Test :func:`colour.colorimetry.dominant.complementary_wavelength`
        definition n-dimensional arrays support.
        """

        xy = np.array([0.54369557, 0.32107944])
        xy_n = self._xy_D65
        wl, xy_wl, xy_cwl = complementary_wavelength(xy, xy_n)
        wl_r, xy_wl_r, xy_cwl_r = (
            np.array(492.0),
            np.array([0.03647950, 0.33847127]),
            np.array([0.03647950, 0.33847127]),
        )
        np.testing.assert_allclose(wl, wl_r, atol=TOLERANCE_ABSOLUTE_TESTS)
        np.testing.assert_allclose(
            xy_wl, xy_wl_r, atol=TOLERANCE_ABSOLUTE_TESTS
        )
        np.testing.assert_allclose(
            xy_cwl, xy_cwl_r, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        xy = np.tile(xy, (6, 1))
        xy_n = np.tile(xy_n, (6, 1))
        wl, xy_wl, xy_cwl = complementary_wavelength(xy, xy_n)
        wl_r = np.tile(wl_r, 6)
        xy_wl_r = np.tile(xy_wl_r, (6, 1))
        xy_cwl_r = np.tile(xy_cwl_r, (6, 1))
        np.testing.assert_allclose(wl, wl_r, atol=TOLERANCE_ABSOLUTE_TESTS)
        np.testing.assert_allclose(
            xy_wl, xy_wl_r, atol=TOLERANCE_ABSOLUTE_TESTS
        )
        np.testing.assert_allclose(
            xy_cwl, xy_cwl_r, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        xy = np.reshape(xy, (2, 3, 2))
        xy_n = np.reshape(xy_n, (2, 3, 2))
        wl, xy_wl, xy_cwl = complementary_wavelength(xy, xy_n)
        wl_r = np.reshape(wl_r, (2, 3))
        xy_wl_r = np.reshape(xy_wl_r, (2, 3, 2))
        xy_cwl_r = np.reshape(xy_cwl_r, (2, 3, 2))
        np.testing.assert_allclose(wl, wl_r, atol=TOLERANCE_ABSOLUTE_TESTS)
        np.testing.assert_allclose(
            xy_wl, xy_wl_r, atol=TOLERANCE_ABSOLUTE_TESTS
        )
        np.testing.assert_allclose(
            xy_cwl, xy_cwl_r, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    @ignore_numpy_errors
    def test_nan_complementary_wavelength(self):
        """
        Test :func:`colour.colorimetry.dominant.complementary_wavelength`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        for case in cases:
            complementary_wavelength(case, case)


class TestExcitationPurity(unittest.TestCase):
    """
    Define :func:`colour.colorimetry.dominant.excitation_purity` definition
    unit tests methods.
    """

    def setUp(self):
        """Initialise the common tests attributes."""

        self._xy_D65 = CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"][
            "D65"
        ]

    def test_excitation_purity(self):
        """Test :func:`colour.colorimetry.dominant.excitation_purity` definition."""

        xy = np.array([0.54369557, 0.32107944])
        xy_n = self._xy_D65

        self.assertAlmostEqual(
            excitation_purity(xy, xy_n), 0.622885671878446, places=7
        )

        xy = np.array([0.37605506, 0.24452225])
        self.assertAlmostEqual(
            excitation_purity(xy, xy_n), 0.438347859215887, places=7
        )

    def test_n_dimensional_excitation_purity(self):
        """
        Test :func:`colour.colorimetry.dominant.excitation_purity` definition
        n-dimensional arrays support.
        """

        xy = np.array([0.54369557, 0.32107944])
        xy_n = self._xy_D65
        P_e = excitation_purity(xy, xy_n)

        xy = np.tile(xy, (6, 1))
        xy_n = np.tile(xy_n, (6, 1))
        P_e = np.tile(P_e, 6)
        np.testing.assert_allclose(
            excitation_purity(xy, xy_n), P_e, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        xy = np.reshape(xy, (2, 3, 2))
        xy_n = np.reshape(xy_n, (2, 3, 2))
        P_e = np.reshape(P_e, (2, 3))
        np.testing.assert_allclose(
            excitation_purity(xy, xy_n), P_e, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    @ignore_numpy_errors
    def test_nan_excitation_purity(self):
        """
        Test :func:`colour.colorimetry.dominant.excitation_purity` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        for case in cases:
            excitation_purity(case, case)


class TestColorimetricPurity(unittest.TestCase):
    """
    Define :func:`colour.colorimetry.dominant.colorimetric_purity` definition
    unit tests methods.
    """

    def setUp(self):
        """Initialise the common tests attributes."""

        self._xy_D65 = CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"][
            "D65"
        ]

    def test_colorimetric_purity(self):
        """
        Test :func:`colour.colorimetry.dominant.colorimetric_purity`
        definition.
        """

        xy = np.array([0.54369557, 0.32107944])
        xy_n = self._xy_D65

        self.assertAlmostEqual(
            colorimetric_purity(xy, xy_n), 0.613582813175483, places=7
        )

        xy = np.array([0.37605506, 0.24452225])
        self.assertAlmostEqual(
            colorimetric_purity(xy, xy_n), 0.244307811178847, places=7
        )

    def test_n_dimensional_colorimetric_purity(self):
        """
        Test :func:`colour.colorimetry.dominant.colorimetric_purity`
        definition n-dimensional arrays support.
        """

        xy = np.array([0.54369557, 0.32107944])
        xy_n = self._xy_D65
        P_e = colorimetric_purity(xy, xy_n)

        xy = np.tile(xy, (6, 1))
        xy_n = np.tile(xy_n, (6, 1))
        P_e = np.tile(P_e, 6)
        np.testing.assert_allclose(
            colorimetric_purity(xy, xy_n), P_e, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        xy = np.reshape(xy, (2, 3, 2))
        xy_n = np.reshape(xy_n, (2, 3, 2))
        P_e = np.reshape(P_e, (2, 3))
        np.testing.assert_allclose(
            colorimetric_purity(xy, xy_n), P_e, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    @ignore_numpy_errors
    def test_nan_colorimetric_purity(self):
        """
        Test :func:`colour.colorimetry.dominant.colorimetric_purity`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        for case in cases:
            colorimetric_purity(case, case)


if __name__ == "__main__":
    unittest.main()
