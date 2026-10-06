"""Define the unit tests for the :mod:`colour.models.rgb.datasets` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

import pickle
from copy import deepcopy

import numpy as np
import pytest

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models import RGB_COLOURSPACES, normalised_primary_matrix
from colour.utilities import (
    as_ndarray,
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
    "TestRGB_COLOURSPACES",
]


class TestRGB_COLOURSPACES:
    """
    Define :attr:`colour.models.rgb.datasets.RGB_COLOURSPACES`
    attribute unit tests methods.
    """

    @pytest.mark.mps_xfail("MPS float32; test uses hard-coded tolerance literals")
    def test_transformation_matrices(self, xp: ModuleType) -> None:
        """
        Test the transformations matrices from the
        :attr:`colour.models.rgb.datasets.RGB_COLOURSPACES` attribute
        colourspace models.
        """

        tolerances = {
            "Adobe RGB (1998)": 1e-5,
            "ARRI Wide Gamut 3": 1e-6,
            "DJI D-Gamut": 5e-4,
            "ERIMM RGB": 1e-3,
            "FilmLight E-Gamut 2": 1e-6,
            "Gamma 2.2 Encoded AdobeRGB": 1e-5,
            "Linear AdobeRGB": 1e-5,
            "ProPhoto RGB": 1e-3,
            "REDWideGamutRGB": 1e-6,
            "RIMM RGB": 1e-3,
            "ROMM RGB": 1e-3,
            "sRGB": 1e-4,
            "V-Gamut": 1e-6,
        }
        XYZ_r = xp_reshape(xp_as_array([0.5, 0.5, 0.5], xp=xp), (3, 1), xp=xp)
        for colourspace in RGB_COLOURSPACES.values():
            M = as_ndarray(
                normalised_primary_matrix(colourspace.primaries, colourspace.whitepoint)
            )

            tolerance = tolerances.get(colourspace.name, 1e-7)
            xp_assert_close(
                colourspace.matrix_RGB_to_XYZ,
                M,
                atol=tolerance,
            )

            RGB = np.dot(colourspace.matrix_XYZ_to_RGB, as_ndarray(XYZ_r))
            XYZ = np.dot(colourspace.matrix_RGB_to_XYZ, RGB)
            xp_assert_close(XYZ_r, XYZ, atol=tolerance)

            # Derived transformation matrices.
            colourspace = deepcopy(colourspace)  # noqa: PLW2901
            colourspace.use_derived_transformation_matrices(True)
            RGB = np.dot(colourspace.matrix_XYZ_to_RGB, as_ndarray(XYZ_r))
            XYZ = np.dot(colourspace.matrix_RGB_to_XYZ, RGB)
            xp_assert_close(XYZ_r, XYZ, atol=tolerance)

    @pytest.mark.mps_xfail("MPS float32; test uses hard-coded tolerance literals")
    def test_cctf(self, xp: ModuleType) -> None:
        """
        Test colour component transfer functions from the
        :attr:`colour.models.rgb.datasets.RGB_COLOURSPACES` attribute
        colourspace models.
        """

        ignored_colourspaces = ("ACESproxy",)

        tolerance = {"DJI D-Gamut": 0.1, "F-Gamut": 1e-4, "N-Gamut": 1e-3}

        samples = xp_as_array(
            np.hstack([np.linspace(0, 1, int(1e5)), np.linspace(0, 65504, 65504 * 10)]),
            xp=xp,
        )

        for colourspace in RGB_COLOURSPACES.values():
            if colourspace.name in ignored_colourspaces:
                continue

            cctf_encoding_s = colourspace.cctf_encoding(samples)
            cctf_decoding_s = colourspace.cctf_decoding(cctf_encoding_s)

            xp_assert_close(
                samples,
                as_ndarray(cctf_decoding_s),
                atol=tolerance.get(colourspace.name, TOLERANCE_ABSOLUTE_TESTS),
            )

    @pytest.mark.mps_xfail("MPS float32; test uses hard-coded tolerance literals")
    def test_n_dimensional_cctf(self, xp: ModuleType) -> None:
        """
        Test colour component transfer functions from the
        :attr:`colour.models.rgb.datasets.RGB_COLOURSPACES` attribute
        colourspace models n-dimensional arrays support.
        """

        tolerance = {"DJI D-Gamut": 1e-6, "F-Gamut": 1e-4}

        for colourspace in RGB_COLOURSPACES.values():
            value_cctf_encoding = 0.5
            value_cctf_decoding = as_ndarray(
                colourspace.cctf_decoding(
                    colourspace.cctf_encoding(value_cctf_encoding)
                )
            )
            xp_assert_close(
                value_cctf_encoding,
                value_cctf_decoding,
                atol=tolerance.get(colourspace.name, 1e-7),
            )

            value_cctf_encoding = xp.tile(xp_as_array(value_cctf_encoding, xp=xp), (6,))
            value_cctf_decoding = xp.tile(xp_as_array(value_cctf_decoding, xp=xp), (6,))
            xp_assert_close(
                value_cctf_encoding,
                value_cctf_decoding,
                atol=tolerance.get(colourspace.name, 1e-7),
            )

            value_cctf_encoding = xp_reshape(
                xp_as_array(value_cctf_encoding, xp=xp), (3, 2), xp=xp
            )
            value_cctf_decoding = xp_reshape(
                xp_as_array(value_cctf_decoding, xp=xp), (3, 2), xp=xp
            )
            xp_assert_close(
                value_cctf_encoding,
                value_cctf_decoding,
                atol=tolerance.get(colourspace.name, 1e-7),
            )

            value_cctf_encoding = xp_reshape(
                xp_as_array(value_cctf_encoding, xp=xp), (3, 2, 1), xp=xp
            )
            value_cctf_decoding = xp_reshape(
                xp_as_array(value_cctf_decoding, xp=xp), (3, 2, 1), xp=xp
            )
            xp_assert_close(
                value_cctf_encoding,
                value_cctf_decoding,
                atol=tolerance.get(colourspace.name, 1e-7),
            )

    @ignore_numpy_errors
    def test_nan_cctf(self) -> None:
        """
        Test colour component transfer functions from the
        :attr:`colour.models.rgb.datasets.RGB_COLOURSPACES` attribute
        colourspace models nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        for colourspace in RGB_COLOURSPACES.values():
            colourspace.cctf_encoding(cases)
            colourspace.cctf_decoding(cases)

    def test_pickle(self) -> None:
        """Test the "pickle-ability" of the *RGB* colourspaces."""

        for colourspace in RGB_COLOURSPACES.values():
            pickle.dumps(colourspace)
