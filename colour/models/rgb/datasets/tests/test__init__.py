"""Defines the unit tests for the :mod:`colour.models.rgb.datasets` module."""

import numpy as np
import pickle
import unittest
from copy import deepcopy

from colour.models import (
    RGB_COLOURSPACES,
    normalised_primary_matrix,
)
from colour.utilities import as_int, ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestRGB_COLOURSPACES",
]


class TestRGB_COLOURSPACES(unittest.TestCase):
    """
    Define :attr:`colour.models.rgb.datasets.RGB_COLOURSPACES`
    attribute unit tests methods.
    """

    def test_transformation_matrices(self):
        """
        Test the transformations matrices from the
        :attr:`colour.models.rgb.datasets.RGB_COLOURSPACES` attribute
        colourspace models.
        """

        tolerances = {
            "Adobe RGB (1998)": 1e-5,
            "ALEXA Wide Gamut": 1e-6,
            "DJI D-Gamut": 1e-4,
            "ERIMM RGB": 1e-3,
            "ProPhoto RGB": 1e-3,
            "REDWideGamutRGB": 1e-6,
            "RIMM RGB": 1e-3,
            "ROMM RGB": 1e-3,
            "sRGB": 1e-4,
            "V-Gamut": 1e-6,
        }
        XYZ_r = np.array([0.5, 0.5, 0.5]).reshape([3, 1])
        for colourspace in RGB_COLOURSPACES.values():
            M = normalised_primary_matrix(
                colourspace.primaries, colourspace.whitepoint
            )

            tolerance = tolerances.get(colourspace.name, 1e-7)
            np.testing.assert_allclose(
                colourspace.matrix_RGB_to_XYZ,
                M,
                rtol=tolerance,
                atol=tolerance,
                verbose=False,
            )

            RGB = np.dot(colourspace.matrix_XYZ_to_RGB, XYZ_r)
            XYZ = np.dot(colourspace.matrix_RGB_to_XYZ, RGB)
            np.testing.assert_allclose(
                XYZ_r, XYZ, rtol=tolerance, atol=tolerance, verbose=False
            )

            # Derived transformation matrices.
            colourspace = deepcopy(colourspace)
            colourspace.use_derived_transformation_matrices(True)
            RGB = np.dot(colourspace.matrix_XYZ_to_RGB, XYZ_r)
            XYZ = np.dot(colourspace.matrix_RGB_to_XYZ, RGB)
            np.testing.assert_almost_equal(XYZ_r, XYZ, decimal=7)

    def test_cctf(self):
        """
        Test colour component transfer functions from the
        :attr:`colour.models.rgb.datasets.RGB_COLOURSPACES` attribute
        colourspace models.
        """

        ignored_colourspaces = ("ACESproxy",)

        decimals = {"DJI D-Gamut": 1, "F-Gamut": 4, "N-Gamut": 3}

        samples = np.hstack(
            [np.linspace(0, 1, as_int(1e5)), np.linspace(0, 65504, 65504 * 10)]
        )

        for colourspace in RGB_COLOURSPACES.values():
            if colourspace.name in ignored_colourspaces:
                continue

            cctf_encoding_s = colourspace.cctf_encoding(samples)
            cctf_decoding_s = colourspace.cctf_decoding(cctf_encoding_s)

            np.testing.assert_almost_equal(
                samples,
                cctf_decoding_s,
                decimal=decimals.get(colourspace.name, 7),
            )

    def test_n_dimensional_cctf(self):
        """
        Test colour component transfer functions from the
        :attr:`colour.models.rgb.datasets.RGB_COLOURSPACES` attribute
        colourspace models n-dimensional arrays support.
        """

        decimals = {"DJI D-Gamut": 6, "F-Gamut": 4}

        for colourspace in RGB_COLOURSPACES.values():
            value_cctf_encoding = 0.5
            value_cctf_decoding = colourspace.cctf_decoding(
                colourspace.cctf_encoding(value_cctf_encoding)
            )
            np.testing.assert_almost_equal(
                value_cctf_encoding,
                value_cctf_decoding,
                decimal=decimals.get(colourspace.name, 7),
            )

            value_cctf_encoding = np.tile(value_cctf_encoding, 6)
            value_cctf_decoding = np.tile(value_cctf_decoding, 6)
            np.testing.assert_almost_equal(
                value_cctf_encoding,
                value_cctf_decoding,
                decimal=decimals.get(colourspace.name, 7),
            )

            value_cctf_encoding = np.reshape(value_cctf_encoding, (3, 2))
            value_cctf_decoding = np.reshape(value_cctf_decoding, (3, 2))
            np.testing.assert_almost_equal(
                value_cctf_encoding,
                value_cctf_decoding,
                decimal=decimals.get(colourspace.name, 7),
            )

            value_cctf_encoding = np.reshape(value_cctf_encoding, (3, 2, 1))
            value_cctf_decoding = np.reshape(value_cctf_decoding, (3, 2, 1))
            np.testing.assert_almost_equal(
                value_cctf_encoding,
                value_cctf_decoding,
                decimal=decimals.get(colourspace.name, 7),
            )

    @ignore_numpy_errors
    def test_nan_cctf(self):
        """
        Test colour component transfer functions from the
        :attr:`colour.models.rgb.datasets.RGB_COLOURSPACES` attribute
        colourspace models nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        for colourspace in RGB_COLOURSPACES.values():
            for case in cases:
                colourspace.cctf_encoding(case)
                colourspace.cctf_decoding(case)

    def test_pickle(self):
        """Test the "pickle-ability" of the *RGB* colourspaces."""

        for colourspace in RGB_COLOURSPACES.values():
            pickle.dumps(colourspace)


if __name__ == "__main__":
    unittest.main()
