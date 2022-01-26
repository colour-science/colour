# -*- coding: utf-8 -*-
"""
Defines the unit tests for the :mod:`colour.io.luts.iridas_cube` module.
"""

from __future__ import annotations

import numpy as np
import os
import unittest
import shutil
import tempfile

from colour.io import (
    LUT1D,
    LUTSequence,
    read_LUT_IridasCube,
    write_LUT_IridasCube,
)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'LUTS_DIRECTORY',
    'TestReadLUTIridasCube',
    'TestWriteLUTIridasCube',
]

LUTS_DIRECTORY: str = os.path.join(
    os.path.dirname(__file__), 'resources', 'iridas_cube')


class TestReadLUTIridasCube(unittest.TestCase):
    """
    Defines :func:`colour.io.luts.iridas_cube.read_LUT_IridasCube` definition
    unit tests methods.
    """

    def test_read_LUT_IridasCube(self):
        """
        Tests :func:`colour.io.luts.iridas_cube.read_LUT_IridasCube`
        definition.
        """

        LUT_1 = read_LUT_IridasCube(
            os.path.join(LUTS_DIRECTORY, 'ACES_Proxy_10_to_ACES.cube'))

        np.testing.assert_almost_equal(
            LUT_1.table,
            np.array([
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
                [1.16500000e+00, 1.16500000e+00, 1.16500000e+00],
                [1.84000000e+00, 1.84000000e+00, 1.84000000e+00],
                [2.90800000e+00, 2.90800000e+00, 2.90800000e+00],
                [4.59500000e+00, 4.59500000e+00, 4.59500000e+00],
                [7.26000000e+00, 7.26000000e+00, 7.26000000e+00],
                [1.14700000e+01, 1.14700000e+01, 1.14700000e+01],
                [1.81300000e+01, 1.81300000e+01, 1.81300000e+01],
                [2.86400000e+01, 2.86400000e+01, 2.86400000e+01],
                [4.52500000e+01, 4.52500000e+01, 4.52500000e+01],
                [7.15100000e+01, 7.15100000e+01, 7.15100000e+01],
                [1.13000000e+02, 1.13000000e+02, 1.13000000e+02],
                [1.78500000e+02, 1.78500000e+02, 1.78500000e+02],
                [2.82100000e+02, 2.82100000e+02, 2.82100000e+02],
                [4.45700000e+02, 4.45700000e+02, 4.45700000e+02],
                [7.04300000e+02, 7.04300000e+02, 7.04300000e+02],
            ]))
        self.assertEqual(LUT_1.name, 'ACES Proxy 10 to ACES')
        self.assertEqual(LUT_1.dimensions, 2)
        np.testing.assert_array_equal(LUT_1.domain,
                                      np.array([[0, 0, 0], [1, 1, 1]]))
        self.assertEqual(LUT_1.size, 32)
        self.assertListEqual(LUT_1.comments, [])

        LUT_2 = read_LUT_IridasCube(os.path.join(LUTS_DIRECTORY, 'Demo.cube'))
        self.assertListEqual(LUT_2.comments, ['Comments can go anywhere'])
        np.testing.assert_array_equal(LUT_2.domain,
                                      np.array([[0, 0, 0], [1, 2, 3]]))

        LUT_3 = read_LUT_IridasCube(
            os.path.join(LUTS_DIRECTORY, 'Three_Dimensional_Table.cube'))
        self.assertEqual(LUT_3.dimensions, 3)
        self.assertEqual(LUT_3.size, 2)


class TestWriteLUTIridasCube(unittest.TestCase):
    """
    Defines :func:`colour.io.luts.iridas_cube.write_LUT_IridasCube` definition
    unit tests methods.
    """

    def setUp(self):
        """
        Initialises common tests attributes.
        """

        self._temporary_directory = tempfile.mkdtemp()

    def tearDown(self):
        """
        After tests actions.
        """

        shutil.rmtree(self._temporary_directory)

    def test_write_LUT_IridasCube(self):
        """
        Tests :func:`colour.io.luts.iridas_cube.write_LUT_IridasCube`
        definition.
        """

        LUT_1_r = read_LUT_IridasCube(
            os.path.join(LUTS_DIRECTORY, 'ACES_Proxy_10_to_ACES.cube'))
        write_LUT_IridasCube(
            LUT_1_r,
            os.path.join(self._temporary_directory,
                         'ACES_Proxy_10_to_ACES.cube'))
        LUT_1_t = read_LUT_IridasCube(
            os.path.join(self._temporary_directory,
                         'ACES_Proxy_10_to_ACES.cube'))
        self.assertEqual(LUT_1_r, LUT_1_t)

        write_LUT_IridasCube(
            LUTSequence(LUT_1_r),
            os.path.join(self._temporary_directory,
                         'ACES_Proxy_10_to_ACES.cube'))
        self.assertEqual(LUT_1_r, LUT_1_t)

        LUT_2_r = read_LUT_IridasCube(
            os.path.join(LUTS_DIRECTORY, 'Demo.cube'))
        write_LUT_IridasCube(
            LUT_2_r, os.path.join(self._temporary_directory, 'Demo.cube'))
        LUT_2_t = read_LUT_IridasCube(
            os.path.join(self._temporary_directory, 'Demo.cube'))
        self.assertEqual(LUT_2_r, LUT_2_t)
        self.assertListEqual(LUT_2_r.comments, LUT_2_t.comments)

        LUT_3_r = read_LUT_IridasCube(
            os.path.join(LUTS_DIRECTORY, 'Three_Dimensional_Table.cube'))
        write_LUT_IridasCube(
            LUT_3_r,
            os.path.join(self._temporary_directory,
                         'Three_Dimensional_Table.cube'))
        LUT_3_t = read_LUT_IridasCube(
            os.path.join(self._temporary_directory,
                         'Three_Dimensional_Table.cube'))
        self.assertEqual(LUT_3_r, LUT_3_t)

        LUT_4_r = read_LUT_IridasCube(
            os.path.join(LUTS_DIRECTORY, 'ACES_Proxy_10_to_ACES.cube'))
        write_LUT_IridasCube(
            LUT_4_r.as_LUT(LUT1D, force_conversion=True),
            os.path.join(self._temporary_directory,
                         'ACES_Proxy_10_to_ACES.cube'))
        LUT_4_t = read_LUT_IridasCube(
            os.path.join(self._temporary_directory,
                         'ACES_Proxy_10_to_ACES.cube'))
        self.assertEqual(LUT_4_r, LUT_4_t)


if __name__ == '__main__':
    unittest.main()
