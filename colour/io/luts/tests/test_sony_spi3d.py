# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.io.luts.sony_spi3d` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import os
import shutil
import tempfile
import unittest

from colour.constants import DEFAULT_INT_DTYPE
from colour.io import (LUT3D, LUTSequence, read_LUT_SonySPI3D,
                       write_LUT_SonySPI3D)
from colour.io.luts.common import parse_array
from colour.utilities import as_int_array

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['LUTS_DIRECTORY', 'TestReadLUTSonySPI3D', 'TestWriteLUTSonySPI3D']

LUTS_DIRECTORY = os.path.join(
    os.path.dirname(__file__), 'resources', 'sony_spi3d')


class TestReadLUTSonySPI3D(unittest.TestCase):
    """
    Defines :func:`colour.io.luts.sony_spi3d.read_LUT_SonySPI3D` definition
    unit tests methods.
    """

    def test_read_LUT_SonySPI3D(self):
        """
        Tests :func:`colour.io.luts.sony_spi3d.read_LUT_SonySPI3D` definition.
        """

        LUT_1 = read_LUT_SonySPI3D(
            os.path.join(LUTS_DIRECTORY, 'Colour_Correct.spi3d'))

        np.testing.assert_almost_equal(
            LUT_1.table,
            np.array([
                [
                    [
                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                        [0.00000000e+00, 0.00000000e+00, 4.16653000e-01],
                        [0.00000000e+00, 0.00000000e+00, 8.33306000e-01],
                        [1.00000000e-06, 1.00000000e-06, 1.24995900e+00],
                    ],
                    [
                        [-2.62310000e-02, 3.77102000e-01, -2.62310000e-02],
                        [1.96860000e-02, 2.44702000e-01, 2.44702000e-01],
                        [1.43270000e-02, 3.30993000e-01, 6.47660000e-01],
                        [9.02200000e-03, 3.72791000e-01, 1.10033100e+00],
                    ],
                    [
                        [-5.24630000e-02, 7.54204000e-01, -5.24630000e-02],
                        [0.00000000e+00, 6.16667000e-01, 3.08333000e-01],
                        [3.93720000e-02, 4.89403000e-01, 4.89403000e-01],
                        [3.57730000e-02, 5.78763000e-01, 8.50258000e-01],
                    ],
                    [
                        [-7.86940000e-02, 1.13130600e+00, -7.86940000e-02],
                        [-3.59270000e-02, 1.02190800e+00, 3.16685000e-01],
                        [3.09040000e-02, 8.31171000e-01, 5.64415000e-01],
                        [5.90590000e-02, 7.34105000e-01, 7.34105000e-01],
                    ],
                ],
                [
                    [
                        [3.98947000e-01, -1.77060000e-02, -1.77060000e-02],
                        [3.33333000e-01, 0.00000000e+00, 3.33333000e-01],
                        [3.90623000e-01, 0.00000000e+00, 7.81246000e-01],
                        [4.04320000e-01, 0.00000000e+00, 1.21296000e+00],
                    ],
                    [
                        [2.94597000e-01, 2.94597000e-01, 6.95820000e-02],
                        [4.16655000e-01, 4.16655000e-01, 4.16655000e-01],
                        [4.16655000e-01, 4.16655000e-01, 8.33308000e-01],
                        [4.16656000e-01, 4.16656000e-01, 1.24996100e+00],
                    ],
                    [
                        [3.49416000e-01, 6.57749000e-01, 4.10830000e-02],
                        [3.40435000e-01, 7.43769000e-01, 3.40435000e-01],
                        [2.69700000e-01, 4.94715000e-01, 4.94715000e-01],
                        [3.47660000e-01, 6.64327000e-01, 9.80993000e-01],
                    ],
                    [
                        [3.44991000e-01, 1.05021300e+00, -7.62100000e-03],
                        [3.14204000e-01, 1.12087100e+00, 3.14204000e-01],
                        [3.08333000e-01, 9.25000000e-01, 6.16667000e-01],
                        [2.89386000e-01, 7.39417000e-01, 7.39417000e-01],
                    ],
                ],
                [
                    [
                        [7.97894000e-01, -3.54120000e-02, -3.54120000e-02],
                        [7.52767000e-01, -2.84790000e-02, 3.62144000e-01],
                        [6.66667000e-01, 0.00000000e+00, 6.66667000e-01],
                        [7.46911000e-01, 0.00000000e+00, 1.12036600e+00],
                    ],
                    [
                        [6.33333000e-01, 3.16667000e-01, 0.00000000e+00],
                        [7.32278000e-01, 3.15626000e-01, 3.15626000e-01],
                        [6.66667000e-01, 3.33333000e-01, 6.66667000e-01],
                        [7.81246000e-01, 3.90623000e-01, 1.17186900e+00],
                    ],
                    [
                        [5.89195000e-01, 5.89195000e-01, 1.39164000e-01],
                        [5.94601000e-01, 5.94601000e-01, 3.69586000e-01],
                        [8.33311000e-01, 8.33311000e-01, 8.33311000e-01],
                        [8.33311000e-01, 8.33311000e-01, 1.24996300e+00],
                    ],
                    [
                        [6.63432000e-01, 9.30188000e-01, 1.29920000e-01],
                        [6.82749000e-01, 9.91082000e-01, 3.74416000e-01],
                        [7.07102000e-01, 1.11043500e+00, 7.07102000e-01],
                        [5.19714000e-01, 7.44729000e-01, 7.44729000e-01],
                    ],
                ],
                [
                    [
                        [1.19684100e+00, -5.31170000e-02, -5.31170000e-02],
                        [1.16258800e+00, -5.03720000e-02, 3.53948000e-01],
                        [1.08900300e+00, -3.13630000e-02, 7.15547000e-01],
                        [1.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                    ],
                    [
                        [1.03843900e+00, 3.10899000e-01, -5.28700000e-02],
                        [1.13122500e+00, 2.97920000e-01, 2.97920000e-01],
                        [1.08610100e+00, 3.04855000e-01, 6.95478000e-01],
                        [1.00000000e+00, 3.33333000e-01, 1.00000000e+00],
                    ],
                    [
                        [8.91318000e-01, 6.19823000e-01, 7.68330000e-02],
                        [9.50000000e-01, 6.33333000e-01, 3.16667000e-01],
                        [1.06561000e+00, 6.48957000e-01, 6.48957000e-01],
                        [1.00000000e+00, 6.66667000e-01, 1.00000000e+00],
                    ],
                    [
                        [8.83792000e-01, 8.83792000e-01, 2.08746000e-01],
                        [8.89199000e-01, 8.89199000e-01, 4.39168000e-01],
                        [8.94606000e-01, 8.94606000e-01, 6.69590000e-01],
                        [1.24996600e+00, 1.24996600e+00, 1.24996600e+00],
                    ],
                ],
            ]))
        self.assertEqual(LUT_1.name, 'Colour Correct')
        self.assertEqual(LUT_1.dimensions, 3)
        np.testing.assert_array_equal(LUT_1.domain,
                                      np.array([[0, 0, 0], [1, 1, 1]]))
        self.assertEqual(LUT_1.size, 4)
        self.assertListEqual(LUT_1.comments,
                             ['Adapted from a LUT generated by Foundry::LUT.'])


class TestWriteLUTSonySPI3D(unittest.TestCase):
    """
    Defines :func:`colour.io.luts.sony_spi3d.write_LUT_SonySPI3D` definition
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

    def test_write_LUT_SonySPI3D(self):
        """
        Tests :func:`colour.io.luts.sony_spi3d.write_LUT_SonySPI3D` definition.
        """

        LUT_r = read_LUT_SonySPI3D(
            os.path.join(LUTS_DIRECTORY, 'Colour_Correct.spi3d'))

        write_LUT_SonySPI3D(
            LUT_r,
            os.path.join(self._temporary_directory, 'Colour_Correct.spi3d'))
        LUT_t = read_LUT_SonySPI3D(
            os.path.join(self._temporary_directory, 'Colour_Correct.spi3d'))
        self.assertEqual(LUT_r, LUT_t)

        write_LUT_SonySPI3D(
            LUTSequence(LUT_r),
            os.path.join(self._temporary_directory, 'Colour_Correct.spi3d'))
        self.assertEqual(LUT_r, LUT_t)

        # Test for proper indexes sequentiality.
        path = os.path.join(self._temporary_directory, 'Size_10_Indexes.spi3d')
        write_LUT_SonySPI3D(LUT3D(size=10), path)
        indexes = []

        with open(path) as spi3d_file:
            lines = filter(None,
                           (line.strip() for line in spi3d_file.readlines()))
            for line in lines:
                if line.startswith('#'):
                    continue

                tokens = line.split()
                if len(tokens) == 6:
                    indexes.append(parse_array(tokens[:3], DEFAULT_INT_DTYPE))

        np.testing.assert_array_equal(
            as_int_array(indexes)[:200, ...],
            np.array([
                [0, 0, 0],
                [0, 0, 1],
                [0, 0, 2],
                [0, 0, 3],
                [0, 0, 4],
                [0, 0, 5],
                [0, 0, 6],
                [0, 0, 7],
                [0, 0, 8],
                [0, 0, 9],
                [0, 1, 0],
                [0, 1, 1],
                [0, 1, 2],
                [0, 1, 3],
                [0, 1, 4],
                [0, 1, 5],
                [0, 1, 6],
                [0, 1, 7],
                [0, 1, 8],
                [0, 1, 9],
                [0, 2, 0],
                [0, 2, 1],
                [0, 2, 2],
                [0, 2, 3],
                [0, 2, 4],
                [0, 2, 5],
                [0, 2, 6],
                [0, 2, 7],
                [0, 2, 8],
                [0, 2, 9],
                [0, 3, 0],
                [0, 3, 1],
                [0, 3, 2],
                [0, 3, 3],
                [0, 3, 4],
                [0, 3, 5],
                [0, 3, 6],
                [0, 3, 7],
                [0, 3, 8],
                [0, 3, 9],
                [0, 4, 0],
                [0, 4, 1],
                [0, 4, 2],
                [0, 4, 3],
                [0, 4, 4],
                [0, 4, 5],
                [0, 4, 6],
                [0, 4, 7],
                [0, 4, 8],
                [0, 4, 9],
                [0, 5, 0],
                [0, 5, 1],
                [0, 5, 2],
                [0, 5, 3],
                [0, 5, 4],
                [0, 5, 5],
                [0, 5, 6],
                [0, 5, 7],
                [0, 5, 8],
                [0, 5, 9],
                [0, 6, 0],
                [0, 6, 1],
                [0, 6, 2],
                [0, 6, 3],
                [0, 6, 4],
                [0, 6, 5],
                [0, 6, 6],
                [0, 6, 7],
                [0, 6, 8],
                [0, 6, 9],
                [0, 7, 0],
                [0, 7, 1],
                [0, 7, 2],
                [0, 7, 3],
                [0, 7, 4],
                [0, 7, 5],
                [0, 7, 6],
                [0, 7, 7],
                [0, 7, 8],
                [0, 7, 9],
                [0, 8, 0],
                [0, 8, 1],
                [0, 8, 2],
                [0, 8, 3],
                [0, 8, 4],
                [0, 8, 5],
                [0, 8, 6],
                [0, 8, 7],
                [0, 8, 8],
                [0, 8, 9],
                [0, 9, 0],
                [0, 9, 1],
                [0, 9, 2],
                [0, 9, 3],
                [0, 9, 4],
                [0, 9, 5],
                [0, 9, 6],
                [0, 9, 7],
                [0, 9, 8],
                [0, 9, 9],
                [1, 0, 0],
                [1, 0, 1],
                [1, 0, 2],
                [1, 0, 3],
                [1, 0, 4],
                [1, 0, 5],
                [1, 0, 6],
                [1, 0, 7],
                [1, 0, 8],
                [1, 0, 9],
                [1, 1, 0],
                [1, 1, 1],
                [1, 1, 2],
                [1, 1, 3],
                [1, 1, 4],
                [1, 1, 5],
                [1, 1, 6],
                [1, 1, 7],
                [1, 1, 8],
                [1, 1, 9],
                [1, 2, 0],
                [1, 2, 1],
                [1, 2, 2],
                [1, 2, 3],
                [1, 2, 4],
                [1, 2, 5],
                [1, 2, 6],
                [1, 2, 7],
                [1, 2, 8],
                [1, 2, 9],
                [1, 3, 0],
                [1, 3, 1],
                [1, 3, 2],
                [1, 3, 3],
                [1, 3, 4],
                [1, 3, 5],
                [1, 3, 6],
                [1, 3, 7],
                [1, 3, 8],
                [1, 3, 9],
                [1, 4, 0],
                [1, 4, 1],
                [1, 4, 2],
                [1, 4, 3],
                [1, 4, 4],
                [1, 4, 5],
                [1, 4, 6],
                [1, 4, 7],
                [1, 4, 8],
                [1, 4, 9],
                [1, 5, 0],
                [1, 5, 1],
                [1, 5, 2],
                [1, 5, 3],
                [1, 5, 4],
                [1, 5, 5],
                [1, 5, 6],
                [1, 5, 7],
                [1, 5, 8],
                [1, 5, 9],
                [1, 6, 0],
                [1, 6, 1],
                [1, 6, 2],
                [1, 6, 3],
                [1, 6, 4],
                [1, 6, 5],
                [1, 6, 6],
                [1, 6, 7],
                [1, 6, 8],
                [1, 6, 9],
                [1, 7, 0],
                [1, 7, 1],
                [1, 7, 2],
                [1, 7, 3],
                [1, 7, 4],
                [1, 7, 5],
                [1, 7, 6],
                [1, 7, 7],
                [1, 7, 8],
                [1, 7, 9],
                [1, 8, 0],
                [1, 8, 1],
                [1, 8, 2],
                [1, 8, 3],
                [1, 8, 4],
                [1, 8, 5],
                [1, 8, 6],
                [1, 8, 7],
                [1, 8, 8],
                [1, 8, 9],
                [1, 9, 0],
                [1, 9, 1],
                [1, 9, 2],
                [1, 9, 3],
                [1, 9, 4],
                [1, 9, 5],
                [1, 9, 6],
                [1, 9, 7],
                [1, 9, 8],
                [1, 9, 9],
            ]))


if __name__ == '__main__':
    unittest.main()
