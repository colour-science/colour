# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.io.luts.cinespace_csp` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import os
import unittest
import shutil
import tempfile

from colour.io import LUTSequence, LUT3x1D
from colour.io import read_LUT_Cinespace, write_LUT_Cinespace
from colour.utilities import tstack

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['LUTS_DIRECTORY', 'TestReadLUTCinespace', 'TestWriteLUTCinespace']

LUTS_DIRECTORY = os.path.join(
    os.path.dirname(__file__), 'resources', 'cinespace')


class TestReadLUTCinespace(unittest.TestCase):
    """
    Defines :func:`colour.io.luts.cinespace_csp.read_LUT_Cinespace` definition
    unit tests methods.
    """

    def test_read_LUT_Cinespace(self):
        """
        Tests :func:`colour.io.luts.cinespace_csp.read_LUT_Cinespace`
        definition.
        """

        LUT_1 = read_LUT_Cinespace(
            os.path.join(LUTS_DIRECTORY, 'ACES_Proxy_10_to_ACES.csp'))

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

        LUT_2 = read_LUT_Cinespace(os.path.join(LUTS_DIRECTORY, 'Demo.csp'))
        self.assertListEqual(LUT_2.comments,
                             ['Comments are ignored by most parsers'])
        np.testing.assert_array_equal(LUT_2.domain,
                                      np.array([[0, 0, 0], [1, 2, 3]]))

        LUT_3 = read_LUT_Cinespace(
            os.path.join(LUTS_DIRECTORY, 'ThreeDimensionalTable.csp'))
        self.assertEqual(LUT_3.dimensions, 3)
        self.assertEqual(LUT_3.size, 2)

        LUT_4 = read_LUT_Cinespace(
            os.path.join(LUTS_DIRECTORY, 'NonUniform.csp'))
        self.assertEqual(LUT_4[0].is_domain_explicit(), True)
        self.assertEqual(LUT_4[1].table.shape, (2, 3, 4, 3))


class TestWriteLUTCinespace(unittest.TestCase):
    """
    Defines :func:`colour.io.luts.cinespace_csp.write_LUT_Cinespace` definition
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

    def test_write_LUT_Cinespace(self):
        """
        Tests :func:`colour.io.luts.cinespace_csp.write_LUT_Cinespace`
        definition.
        """

        LUT_1_r = read_LUT_Cinespace(
            os.path.join(LUTS_DIRECTORY, 'ACES_Proxy_10_to_ACES.csp'))

        write_LUT_Cinespace(
            LUT_1_r,
            os.path.join(self._temporary_directory,
                         'ACES_Proxy_10_to_ACES.csp'))

        LUT_1_t = read_LUT_Cinespace(
            os.path.join(self._temporary_directory,
                         'ACES_Proxy_10_to_ACES.csp'))

        self.assertEqual(LUT_1_r, LUT_1_t)

        self.assertEqual(LUT_1_r, LUT_1_t)

        LUT_2_r = read_LUT_Cinespace(os.path.join(LUTS_DIRECTORY, 'Demo.csp'))

        write_LUT_Cinespace(
            LUT_2_r, os.path.join(self._temporary_directory, 'Demo.csp'))

        LUT_2_t = read_LUT_Cinespace(
            os.path.join(self._temporary_directory, 'Demo.csp'))

        self.assertEqual(LUT_2_r, LUT_2_t)
        self.assertListEqual(LUT_2_r.comments, LUT_2_t.comments)

        LUT_3_r = read_LUT_Cinespace(
            os.path.join(LUTS_DIRECTORY, 'ThreeDimensionalTable.csp'))

        write_LUT_Cinespace(
            LUT_3_r,
            os.path.join(self._temporary_directory,
                         'ThreeDimensionalTable.csp'))

        LUT_3_t = read_LUT_Cinespace(
            os.path.join(self._temporary_directory,
                         'ThreeDimensionalTable.csp'))

        self.assertEqual(LUT_3_r, LUT_3_t)

        write_LUT_Cinespace(
            LUTSequence(LUT_1_r, LUT_3_r),
            os.path.join(self._temporary_directory, 'test_sequence.csp'))

        r = np.array([0.0, 0.1, 0.2, 0.4, 0.8, 1.2])
        g = np.array([-0.1, 0.5, 1.0, np.nan, np.nan, np.nan])
        b = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, np.nan])

        domain = tstack((r, g, b))

        LUT_4_t = LUT3x1D(domain=domain, table=domain * 2)

        write_LUT_Cinespace(
            LUT_4_t,
            os.path.join(self._temporary_directory, 'ragged_domain.csp'))

        LUT_4_r = read_LUT_Cinespace(
            os.path.join(self._temporary_directory, 'ragged_domain.csp'))

        np.testing.assert_almost_equal(LUT_4_t.domain, LUT_4_r.domain)

        np.testing.assert_almost_equal(LUT_4_t.table, LUT_4_r.table, decimal=6)


if __name__ == '__main__':
    unittest.main()
