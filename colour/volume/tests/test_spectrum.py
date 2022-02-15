"""Defines the unit tests for the :mod:`colour.volume.spectrum` module."""

import numpy as np
import unittest
from itertools import permutations

from colour.colorimetry import (
    MSDS_CMFS,
    SPECTRAL_SHAPE_DEFAULT,
    SpectralShape,
    reshape_msds,
)
from colour.volume import (
    generate_pulse_waves,
    XYZ_outer_surface,
    is_within_visible_spectrum,
)
from colour.utilities import ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestGeneratePulseWaves",
    "TestXYZOuterSurface",
    "TestIsWithinVisibleSpectrum",
]


class TestGeneratePulseWaves(unittest.TestCase):
    """
    Define :func:`colour.volume.spectrum.generate_pulse_waves`
    definition unit tests methods.
    """

    def test_generate_pulse_waves(self):
        """
        Test :func:`colour.volume.spectrum.generate_pulse_waves`
        definition.
        """

        np.testing.assert_array_equal(
            generate_pulse_waves(5),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0, 1.0],
                    [1.0, 0.0, 0.0, 1.0, 1.0],
                    [1.0, 1.0, 0.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 0.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 0.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                ]
            ),
        )

        np.testing.assert_array_equal(
            generate_pulse_waves(5, "Pulse Wave Width"),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0, 1.0],
                    [1.0, 0.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 1.0, 1.0],
                    [1.0, 1.0, 0.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                ]
            ),
        )

    np.testing.assert_equal(
        np.sort(generate_pulse_waves(5), axis=0),
        np.sort(generate_pulse_waves(5, "Pulse Wave Width"), axis=0),
    )

    np.testing.assert_array_equal(
        generate_pulse_waves(5, "Pulse Wave Width", True),
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        ),
    )


class TestXYZOuterSurface(unittest.TestCase):
    """
    Define :func:`colour.volume.spectrum.XYZ_outer_surface`
    definition unit tests methods.
    """

    def test_XYZ_outer_surface(self):
        """
        Test :func:`colour.volume.spectrum.XYZ_outer_surface`
        definition.
        """

        shape = SpectralShape(
            SPECTRAL_SHAPE_DEFAULT.start, SPECTRAL_SHAPE_DEFAULT.end, 84
        )
        cmfs = MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]

        # pylint: disable=E1102
        np.testing.assert_array_almost_equal(
            XYZ_outer_surface(reshape_msds(cmfs, shape)),
            np.array(
                [
                    [0.00000000e00, 0.00000000e00, 0.00000000e00],
                    [9.63613812e-05, 2.90567768e-06, 4.49612264e-04],
                    [2.59105294e-01, 2.10312980e-02, 1.32074689e00],
                    [1.05610219e-01, 6.20382435e-01, 3.54235713e-02],
                    [7.26479803e-01, 3.54608696e-01, 2.10051491e-04],
                    [1.09718745e-02, 3.96354538e-03, 0.00000000e00],
                    [3.07925724e-05, 1.11197622e-05, 0.00000000e00],
                    [2.59201656e-01, 2.10342037e-02, 1.32119651e00],
                    [3.64715514e-01, 6.41413733e-01, 1.35617047e00],
                    [8.32090022e-01, 9.74991131e-01, 3.56336228e-02],
                    [7.37451677e-01, 3.58572241e-01, 2.10051491e-04],
                    [1.10026671e-02, 3.97466514e-03, 0.00000000e00],
                    [1.27153954e-04, 1.40254398e-05, 4.49612264e-04],
                    [3.64811875e-01, 6.41416639e-01, 1.35662008e00],
                    [1.09119532e00, 9.96022429e-01, 1.35638052e00],
                    [8.43061896e-01, 9.78954677e-01, 3.56336228e-02],
                    [7.37482470e-01, 3.58583361e-01, 2.10051491e-04],
                    [1.10990285e-02, 3.97757082e-03, 4.49612264e-04],
                    [2.59232448e-01, 2.10453234e-02, 1.32119651e00],
                    [1.09129168e00, 9.96025335e-01, 1.35683013e00],
                    [1.10216719e00, 9.99985975e-01, 1.35638052e00],
                    [8.43092689e-01, 9.78965796e-01, 3.56336228e-02],
                    [7.37578831e-01, 3.58586267e-01, 6.59663755e-04],
                    [2.70204323e-01, 2.50088688e-02, 1.32119651e00],
                    [3.64842668e-01, 6.41427759e-01, 1.35662008e00],
                    [1.10226355e00, 9.99988880e-01, 1.35683013e00],
                    [1.10219798e00, 9.99997094e-01, 1.35638052e00],
                    [8.43189050e-01, 9.78968702e-01, 3.60832350e-02],
                    [9.96684125e-01, 3.79617565e-01, 1.32140656e00],
                    [3.75814542e-01, 6.45391304e-01, 1.35662008e00],
                    [1.09132247e00, 9.96036455e-01, 1.35683013e00],
                    [1.10229434e00, 1.00000000e00, 1.35683013e00],
                ]
            ),
            decimal=7,
        )


class TestIsWithinVisibleSpectrum(unittest.TestCase):
    """
    Define :func:`colour.volume.spectrum.is_within_visible_spectrum`
    definition unit tests methods.
    """

    def test_is_within_visible_spectrum(self):
        """
        Test :func:`colour.volume.spectrum.is_within_visible_spectrum`
        definition.
        """

        self.assertTrue(
            is_within_visible_spectrum(np.array([0.3205, 0.4131, 0.5100]))
        )

        self.assertFalse(
            is_within_visible_spectrum(np.array([-0.0005, 0.0031, 0.0010]))
        )

        self.assertTrue(
            is_within_visible_spectrum(np.array([0.4325, 0.3788, 0.1034]))
        )

        self.assertFalse(
            is_within_visible_spectrum(np.array([0.0025, 0.0088, 0.0340]))
        )

    def test_n_dimensional_is_within_visible_spectrum(self):
        """
        Test :func:`colour.volume.spectrum.is_within_visible_spectrum`
        definition n-dimensional arrays support.
        """

        a = np.array([0.3205, 0.4131, 0.5100])
        b = is_within_visible_spectrum(a)

        a = np.tile(a, (6, 1))
        b = np.tile(b, 6)
        np.testing.assert_almost_equal(is_within_visible_spectrum(a), b)

        a = np.reshape(a, (2, 3, 3))
        b = np.reshape(b, (2, 3))
        np.testing.assert_almost_equal(is_within_visible_spectrum(a), b)

    @ignore_numpy_errors
    def test_nan_is_within_visible_spectrum(self):
        """
        Test :func:`colour.volume.spectrum.is_within_visible_spectrum`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            is_within_visible_spectrum(case)


if __name__ == "__main__":
    unittest.main()
