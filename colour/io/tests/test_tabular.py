# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.io.tabular` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import os
import shutil
import unittest
import tempfile
from six import PY2, text_type

from colour.colorimetry import SpectralDistribution
from colour.io import (read_spectral_data_from_csv_file,
                       read_sds_from_csv_file, write_sds_to_csv_file)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'RESOURCES_DIRECTORY', 'COLOURCHECKER_N_OHTA_1',
    'TestReadSpectralDataFromCsvFile', 'TestReadSdsFromCsvFile',
    'TestWriteSdsToCsvFile'
]

RESOURCES_DIRECTORY = os.path.join(os.path.dirname(__file__), 'resources')

COLOURCHECKER_N_OHTA_1 = {
    380.0: 0.048,
    385.0: 0.051,
    390.0: 0.055,
    395.0: 0.060,
    400.0: 0.065,
    405.0: 0.068,
    410.0: 0.068,
    415.0: 0.067,
    420.0: 0.064,
    425.0: 0.062,
    430.0: 0.059,
    435.0: 0.057,
    440.0: 0.055,
    445.0: 0.054,
    450.0: 0.053,
    455.0: 0.053,
    460.0: 0.052,
    465.0: 0.052,
    470.0: 0.052,
    475.0: 0.053,
    480.0: 0.054,
    485.0: 0.055,
    490.0: 0.057,
    495.0: 0.059,
    500.0: 0.061,
    505.0: 0.062,
    510.0: 0.065,
    515.0: 0.067,
    520.0: 0.070,
    525.0: 0.072,
    530.0: 0.074,
    535.0: 0.075,
    540.0: 0.076,
    545.0: 0.078,
    550.0: 0.079,
    555.0: 0.082,
    560.0: 0.087,
    565.0: 0.092,
    570.0: 0.100,
    575.0: 0.107,
    580.0: 0.115,
    585.0: 0.122,
    590.0: 0.129,
    595.0: 0.134,
    600.0: 0.138,
    605.0: 0.142,
    610.0: 0.146,
    615.0: 0.150,
    620.0: 0.154,
    625.0: 0.158,
    630.0: 0.163,
    635.0: 0.167,
    640.0: 0.173,
    645.0: 0.180,
    650.0: 0.188,
    655.0: 0.196,
    660.0: 0.204,
    665.0: 0.213,
    670.0: 0.222,
    675.0: 0.231,
    680.0: 0.242,
    685.0: 0.251,
    690.0: 0.261,
    695.0: 0.271,
    700.0: 0.282,
    705.0: 0.294,
    710.0: 0.305,
    715.0: 0.318,
    720.0: 0.334,
    725.0: 0.354,
    730.0: 0.372,
    735.0: 0.392,
    740.0: 0.409,
    745.0: 0.420,
    750.0: 0.436,
    755.0: 0.450,
    760.0: 0.462,
    765.0: 0.465,
    770.0: 0.448,
    775.0: 0.432,
    780.0: 0.421
}


class TestReadSpectralDataFromCsvFile(unittest.TestCase):
    """
    Defines :func:`colour.io.tabular.read_spectral_data_from_csv_file`
    definition unit tests methods.
    """

    def test_read_spectral_data_from_csv_file(self):
        """
        Tests :func:`colour.io.tabular.read_spectral_data_from_csv_file`
        definition.
        """

        colour_checker_n_ohta = os.path.join(RESOURCES_DIRECTORY,
                                             'colorchecker_n_ohta.csv')
        data = read_spectral_data_from_csv_file(colour_checker_n_ohta)
        self.assertListEqual(
            sorted(data), sorted([text_type(x) for x in range(1, 25)]))
        self.assertDictEqual(data['1'], COLOURCHECKER_N_OHTA_1)

        linss2_10e_5 = os.path.join(RESOURCES_DIRECTORY, 'linss2_10e_5.csv')
        data = read_spectral_data_from_csv_file(
            linss2_10e_5, fields=['wavelength', 'l_bar', 'm_bar', 's_bar'])
        self.assertListEqual(sorted(data), ['l_bar', 'm_bar', 's_bar'])
        self.assertEqual(data['s_bar'][760], 0)
        data = read_spectral_data_from_csv_file(
            linss2_10e_5,
            fields=['wavelength', 'l_bar', 'm_bar', 's_bar'],
            default=-1)
        self.assertEqual(data['s_bar'][760], -1)


class TestReadSdsFromCsvFile(unittest.TestCase):
    """
    Defines :func:`colour.io.tabular.read_sds_from_csv_file` definition units
    tests methods.
    """

    def test_read_sds_from_csv_file(self):
        """
        Tests :func:`colour.io.tabular.read_sds_from_csv_file` definition.
        """

        colour_checker_n_ohta = os.path.join(RESOURCES_DIRECTORY,
                                             'colorchecker_n_ohta.csv')
        sds = read_sds_from_csv_file(colour_checker_n_ohta)
        for sd in sds.values():
            self.assertIsInstance(sd, SpectralDistribution)

        self.assertEqual(
            sds['1'], SpectralDistribution(COLOURCHECKER_N_OHTA_1, name='1'))


class TestWriteSdsToCsvFile(unittest.TestCase):
    """
    Defines :func:`colour.io.tabular.write_sds_to_csv_file` definition units
    tests methods.
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

    def test_write_sds_to_csv_file(self):
        """
        Tests :func:`colour.io.tabular.write_sds_to_csv_file` definition.
        """

        colour_checker_n_ohta = os.path.join(RESOURCES_DIRECTORY,
                                             'colorchecker_n_ohta.csv')
        sds = read_sds_from_csv_file(colour_checker_n_ohta)
        colour_checker_n_ohta_test = os.path.join(self._temporary_directory,
                                                  'colorchecker_n_ohta.csv')
        write_sds_to_csv_file(sds, colour_checker_n_ohta_test)
        sds_test = read_sds_from_csv_file(colour_checker_n_ohta_test)
        for key, value in sds.items():
            if PY2:
                # Running into precision issues with Python 2.x, applying
                # conservative rounding.
                value.wavelengths = np.around(value.wavelengths, decimals=7)
                value.values = np.around(value.values, decimals=7)
                sds_test[key].wavelengths = np.around(
                    sds_test[key].wavelengths, decimals=7)
                sds_test[key].values = np.around(
                    sds_test[key].values, decimals=7)

            self.assertEqual(value, sds_test[key])

        write_sds_to_csv_file(sds, colour_checker_n_ohta_test, fields=['1'])
        sds_test = read_sds_from_csv_file(colour_checker_n_ohta_test)
        self.assertEqual(len(sds_test), 1)


if __name__ == '__main__':
    unittest.main()
