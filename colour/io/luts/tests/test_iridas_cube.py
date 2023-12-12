# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.io.luts.iridas_cube` module."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.io import (
    LUT1D,
    LUTSequence,
    read_LUT_IridasCube,
    write_LUT_IridasCube,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "ROOT_LUTS",
    "TestReadLUTIridasCube",
    "TestWriteLUTIridasCube",
]

ROOT_LUTS: str = os.path.join(
    os.path.dirname(__file__), "resources", "iridas_cube"
)


class TestReadLUTIridasCube(unittest.TestCase):
    """
    Define :func:`colour.io.luts.iridas_cube.read_LUT_IridasCube` definition
    unit tests methods.
    """

    def test_read_LUT_IridasCube(self):
        """
        Test :func:`colour.io.luts.iridas_cube.read_LUT_IridasCube`
        definition.
        """

        LUT_1 = read_LUT_IridasCube(
            os.path.join(ROOT_LUTS, "ACES_Proxy_10_to_ACES.cube")
        )

        np.testing.assert_allclose(
            LUT_1.table,
            np.array(
                [
                    [4.88300000e-04, 4.88300000e-04, 4.88300000e-04],
                    [7.71400000e-04, 7.71400000e-04, 7.71400000e-04],
                    [1.21900000e-03, 1.21900000e-03, 1.21900000e-03],
                    [1.92600000e-03, 1.92600000e-03, 1.92600000e-03],
                    [3.04400000e-03, 3.04400000e-03, 3.04400000e-03],
                    [4.80900000e-03, 4.80900000e-03, 4.80900000e-03],
                    [7.59900000e-03, 7.59900000e-03, 7.59900000e-03],
                    [1.20100000e-02, 1.20100000e-02, 1.20100000e-02],
                    [1.89700000e-02, 1.89700000e-02, 1.89700000e-02],
                    [2.99800000e-02, 2.99800000e-02, 2.99800000e-02],
                    [4.73700000e-02, 4.73700000e-02, 4.73700000e-02],
                    [7.48400000e-02, 7.48400000e-02, 7.48400000e-02],
                    [1.18300000e-01, 1.18300000e-01, 1.18300000e-01],
                    [1.86900000e-01, 1.86900000e-01, 1.86900000e-01],
                    [2.95200000e-01, 2.95200000e-01, 2.95200000e-01],
                    [4.66500000e-01, 4.66500000e-01, 4.66500000e-01],
                    [7.37100000e-01, 7.37100000e-01, 7.37100000e-01],
                    [1.16500000e00, 1.16500000e00, 1.16500000e00],
                    [1.84000000e00, 1.84000000e00, 1.84000000e00],
                    [2.90800000e00, 2.90800000e00, 2.90800000e00],
                    [4.59500000e00, 4.59500000e00, 4.59500000e00],
                    [7.26000000e00, 7.26000000e00, 7.26000000e00],
                    [1.14700000e01, 1.14700000e01, 1.14700000e01],
                    [1.81300000e01, 1.81300000e01, 1.81300000e01],
                    [2.86400000e01, 2.86400000e01, 2.86400000e01],
                    [4.52500000e01, 4.52500000e01, 4.52500000e01],
                    [7.15100000e01, 7.15100000e01, 7.15100000e01],
                    [1.13000000e02, 1.13000000e02, 1.13000000e02],
                    [1.78500000e02, 1.78500000e02, 1.78500000e02],
                    [2.82100000e02, 2.82100000e02, 2.82100000e02],
                    [4.45700000e02, 4.45700000e02, 4.45700000e02],
                    [7.04300000e02, 7.04300000e02, 7.04300000e02],
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        self.assertEqual(LUT_1.name, "ACES Proxy 10 to ACES")
        self.assertEqual(LUT_1.dimensions, 2)
        np.testing.assert_array_equal(
            LUT_1.domain, np.array([[0, 0, 0], [1, 1, 1]])
        )
        self.assertEqual(LUT_1.size, 32)
        self.assertListEqual(LUT_1.comments, [])

        LUT_2 = read_LUT_IridasCube(os.path.join(ROOT_LUTS, "Demo.cube"))
        self.assertListEqual(LUT_2.comments, ["Comments can go anywhere"])
        np.testing.assert_array_equal(
            LUT_2.domain, np.array([[0, 0, 0], [1, 2, 3]])
        )

        LUT_3 = read_LUT_IridasCube(
            os.path.join(ROOT_LUTS, "Three_Dimensional_Table.cube")
        )
        self.assertEqual(LUT_3.dimensions, 3)
        self.assertEqual(LUT_3.size, 2)


class TestWriteLUTIridasCube(unittest.TestCase):
    """
    Define :func:`colour.io.luts.iridas_cube.write_LUT_IridasCube` definition
    unit tests methods.
    """

    def setUp(self):
        """Initialise the common tests attributes."""

        self._temporary_directory = tempfile.mkdtemp()

    def tearDown(self):
        """After tests actions."""

        shutil.rmtree(self._temporary_directory)

    def test_write_LUT_IridasCube(self):
        """
        Test :func:`colour.io.luts.iridas_cube.write_LUT_IridasCube`
        definition.
        """

        LUT_1_r = read_LUT_IridasCube(
            os.path.join(ROOT_LUTS, "ACES_Proxy_10_to_ACES.cube")
        )
        write_LUT_IridasCube(
            LUT_1_r,
            os.path.join(
                self._temporary_directory, "ACES_Proxy_10_to_ACES.cube"
            ),
        )
        LUT_1_t = read_LUT_IridasCube(
            os.path.join(
                self._temporary_directory, "ACES_Proxy_10_to_ACES.cube"
            )
        )
        self.assertEqual(LUT_1_r, LUT_1_t)

        write_LUT_IridasCube(
            LUTSequence(LUT_1_r),
            os.path.join(
                self._temporary_directory, "ACES_Proxy_10_to_ACES.cube"
            ),
        )
        self.assertEqual(LUT_1_r, LUT_1_t)

        LUT_2_r = read_LUT_IridasCube(os.path.join(ROOT_LUTS, "Demo.cube"))
        write_LUT_IridasCube(
            LUT_2_r, os.path.join(self._temporary_directory, "Demo.cube")
        )
        LUT_2_t = read_LUT_IridasCube(
            os.path.join(self._temporary_directory, "Demo.cube")
        )
        self.assertEqual(LUT_2_r, LUT_2_t)
        self.assertListEqual(LUT_2_r.comments, LUT_2_t.comments)

        LUT_3_r = read_LUT_IridasCube(
            os.path.join(ROOT_LUTS, "Three_Dimensional_Table.cube")
        )
        write_LUT_IridasCube(
            LUT_3_r,
            os.path.join(
                self._temporary_directory, "Three_Dimensional_Table.cube"
            ),
        )
        LUT_3_t = read_LUT_IridasCube(
            os.path.join(
                self._temporary_directory, "Three_Dimensional_Table.cube"
            )
        )
        self.assertEqual(LUT_3_r, LUT_3_t)

        LUT_4_r = read_LUT_IridasCube(
            os.path.join(ROOT_LUTS, "ACES_Proxy_10_to_ACES.cube")
        )
        write_LUT_IridasCube(
            LUT_4_r.convert(LUT1D, force_conversion=True),
            os.path.join(
                self._temporary_directory, "ACES_Proxy_10_to_ACES.cube"
            ),
        )
        LUT_4_t = read_LUT_IridasCube(
            os.path.join(
                self._temporary_directory, "ACES_Proxy_10_to_ACES.cube"
            )
        )
        self.assertEqual(LUT_4_r, LUT_4_t)


if __name__ == "__main__":
    unittest.main()
