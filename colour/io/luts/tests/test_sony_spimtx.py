# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.io.luts.sony_spimtx` module."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.io import read_LUT_SonySPImtx, write_LUT_SonySPImtx

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "ROOT_LUTS",
    "TestReadLUTSonySPImtx",
    "TestWriteLUTSonySPImtx",
]

ROOT_LUTS: str = os.path.join(os.path.dirname(__file__), "resources", "sony_spimtx")


class TestReadLUTSonySPImtx(unittest.TestCase):
    """
    Define :func:`colour.io.luts.sony_spimtx.read_LUT_SonySPImtx` definition
    unit tests methods.
    """

    def test_read_LUT_SonySPImtx(self):
        """
        Test :func:`colour.io.luts.sony_spimtx.read_LUT_SonySPImtx`
        definition.
        """

        LUT_1 = read_LUT_SonySPImtx(os.path.join(ROOT_LUTS, "dt.spimtx"))

        np.testing.assert_allclose(
            LUT_1.matrix,
            np.array(
                [
                    [0.864274, 0.000000, 0.000000, 0.000000],
                    [0.000000, 0.864274, 0.000000, 0.000000],
                    [0.000000, 0.000000, 0.864274, 0.000000],
                    [0.000000, 0.000000, 0.000000, 1.000000],
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        np.testing.assert_allclose(
            LUT_1.offset,
            np.array([0.000000, 0.000000, 0.000000, 0.000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        self.assertEqual(LUT_1.name, "dt")

        LUT_2 = read_LUT_SonySPImtx(os.path.join(ROOT_LUTS, "p3_to_xyz16.spimtx"))
        np.testing.assert_allclose(
            LUT_2.matrix,
            np.array(
                [
                    [0.44488, 0.27717, 0.17237, 0.00000],
                    [0.20936, 0.72170, 0.06895, 0.00000],
                    [0.00000, 0.04707, 0.90780, 0.00000],
                    [0.00000, 0.00000, 0.00000, 1.00000],
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        np.testing.assert_allclose(
            LUT_2.offset,
            np.array([0.000000, 0.000000, 0.000000, 0.000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        self.assertEqual(LUT_2.name, "p3 to xyz16")

        LUT_3 = read_LUT_SonySPImtx(os.path.join(ROOT_LUTS, "Matrix_Offset.spimtx"))
        np.testing.assert_allclose(
            LUT_3.matrix,
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        np.testing.assert_allclose(
            LUT_3.offset,
            np.array([0.0, 0.0, 1.0, 0.0]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        self.assertEqual(LUT_3.name, "Matrix Offset")


class TestWriteLUTSonySPImtx(unittest.TestCase):
    """
    Define :func:`colour.io.luts.sony_spimtx.write_LUT_SonySPImtx` definition
    unit tests methods.
    """

    def setUp(self):
        """Initialise the common tests attributes."""

        self._temporary_directory = tempfile.mkdtemp()

    def tearDown(self):
        """After tests actions."""

        shutil.rmtree(self._temporary_directory)

    def test_write_LUT_SonySPImtx(self):
        """
        Test :func:`colour.io.luts.sony_spimtx.write_LUT_SonySPImtx`
        definition.
        """

        LUT_1_r = read_LUT_SonySPImtx(os.path.join(ROOT_LUTS, "dt.spimtx"))
        write_LUT_SonySPImtx(
            LUT_1_r, os.path.join(self._temporary_directory, "dt.spimtx")
        )
        LUT_1_t = read_LUT_SonySPImtx(
            os.path.join(self._temporary_directory, "dt.spimtx")
        )
        self.assertEqual(LUT_1_r, LUT_1_t)

        LUT_2_r = read_LUT_SonySPImtx(os.path.join(ROOT_LUTS, "p3_to_xyz16.spimtx"))
        write_LUT_SonySPImtx(
            LUT_2_r,
            os.path.join(self._temporary_directory, "p3_to_xyz16.spimtx"),
        )
        LUT_2_t = read_LUT_SonySPImtx(
            os.path.join(self._temporary_directory, "p3_to_xyz16.spimtx")
        )
        self.assertEqual(LUT_2_r, LUT_2_t)
        self.assertListEqual(LUT_2_r.comments, LUT_2_t.comments)


if __name__ == "__main__":
    unittest.main()
