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

from colour.io import LUTSequence, read_LUT_SonySPI3D, write_LUT_SonySPI3D

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
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

        LUT = read_LUT_SonySPI3D(
            os.path.join(LUTS_DIRECTORY, 'ColourCorrect.spi3d'))

        np.testing.assert_almost_equal(
            LUT.table,
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
        self.assertEqual(LUT.name, 'ColourCorrect')
        self.assertEqual(LUT.dimensions, 3)
        np.testing.assert_array_equal(LUT.domain,
                                      np.array([[0, 0, 0], [1, 1, 1]]))
        self.assertEqual(LUT.size, 4)
        self.assertListEqual(LUT.comments,
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
            os.path.join(LUTS_DIRECTORY, 'ColourCorrect.spi3d'))

        write_LUT_SonySPI3D(
            LUT_r,
            os.path.join(self._temporary_directory, 'ColourCorrect.spi3d'))

        LUT_t = read_LUT_SonySPI3D(
            os.path.join(self._temporary_directory, 'ColourCorrect.spi3d'))

        self.assertEqual(LUT_r, LUT_t)

        write_LUT_SonySPI3D(
            LUTSequence(LUT_r),
            os.path.join(self._temporary_directory, 'ColourCorrect.spi3d'))

        self.assertEqual(LUT_r, LUT_t)


if __name__ == '__main__':
    unittest.main()
