"""Defines the unit tests for the :mod:`colour.io.tabular` module."""

from __future__ import annotations

import os
import shutil
import unittest
import tempfile

from colour.colorimetry import SpectralDistribution, SpectralShape
from colour.hints import Dict
from colour.io import (
    read_spectral_data_from_csv_file,
    read_sds_from_csv_file,
    write_sds_to_csv_file,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "RESOURCES_DIRECTORY",
    "COLOURCHECKER_N_OHTA_1",
    "TestReadSpectralDataFromCsvFile",
    "TestReadSdsFromCsvFile",
    "TestWriteSdsToCsvFile",
]

RESOURCES_DIRECTORY: str = os.path.join(os.path.dirname(__file__), "resources")

COLOURCHECKER_N_OHTA_1: Dict = {
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
    780.0: 0.421,
}


class TestReadSpectralDataFromCsvFile(unittest.TestCase):
    """
    Define :func:`colour.io.tabular.read_spectral_data_from_csv_file`
    definition unit tests methods.
    """

    def test_read_spectral_data_from_csv_file(self):
        """
        Test :func:`colour.io.tabular.read_spectral_data_from_csv_file`
        definition.
        """

        colour_checker_n_ohta = os.path.join(
            RESOURCES_DIRECTORY, "colorchecker_n_ohta.csv"
        )
        data = read_spectral_data_from_csv_file(colour_checker_n_ohta)
        self.assertListEqual(
            list(data.keys()), ["wavelength"] + [str(x) for x in range(1, 25)]
        )
        self.assertDictEqual(
            dict(zip(data["wavelength"], data["1"])), COLOURCHECKER_N_OHTA_1
        )

        colour_checker_n_ohta_transposed = os.path.join(
            RESOURCES_DIRECTORY, "colorchecker_n_ohta_transposed.csv"
        )
        data = read_spectral_data_from_csv_file(
            colour_checker_n_ohta_transposed, transpose=True, delimiter="\t"
        )
        self.assertListEqual(
            list(data.keys()), ["wavelength"] + [str(x) for x in range(1, 25)]
        )
        self.assertDictEqual(
            dict(zip(data["wavelength"], data["1"])), COLOURCHECKER_N_OHTA_1
        )

        linss2_10e_5 = os.path.join(RESOURCES_DIRECTORY, "linss2_10e_5.csv")
        data = read_spectral_data_from_csv_file(
            linss2_10e_5,
            names=["wavelength", "l_bar", "m_bar", "s_bar"],
            filling_values=0,
        )
        self.assertListEqual(
            list(data.keys()), ["wavelength", "l_bar", "m_bar", "s_bar"]
        )
        self.assertEqual(data["s_bar"][77], 0)
        data = read_spectral_data_from_csv_file(
            linss2_10e_5,
            names=["wavelength", "l_bar", "m_bar", "s_bar"],
            filling_values=-1,
        )
        self.assertEqual(data["s_bar"][77], -1)


class TestReadSdsFromCsvFile(unittest.TestCase):
    """
    Define :func:`colour.io.tabular.read_sds_from_csv_file` definition unit
    tests methods.
    """

    def test_read_sds_from_csv_file(self):
        """Test :func:`colour.io.tabular.read_sds_from_csv_file` definition."""

        colour_checker_n_ohta = os.path.join(
            RESOURCES_DIRECTORY, "colorchecker_n_ohta.csv"
        )
        sds = read_sds_from_csv_file(colour_checker_n_ohta)
        for sd in sds.values():
            self.assertIsInstance(sd, SpectralDistribution)

        self.assertEqual(
            sds["1"], SpectralDistribution(COLOURCHECKER_N_OHTA_1, name="1")
        )


class TestWriteSdsToCsvFile(unittest.TestCase):
    """
    Define :func:`colour.io.tabular.write_sds_to_csv_file` definition unit
    tests methods.
    """

    def setUp(self):
        """Initialise the common tests attributes."""

        self._temporary_directory = tempfile.mkdtemp()

    def tearDown(self):
        """After tests actions."""

        shutil.rmtree(self._temporary_directory)

    def test_write_sds_to_csv_file(self):
        """Test :func:`colour.io.tabular.write_sds_to_csv_file` definition."""

        colour_checker_n_ohta = os.path.join(
            RESOURCES_DIRECTORY, "colorchecker_n_ohta.csv"
        )
        sds = read_sds_from_csv_file(colour_checker_n_ohta)
        colour_checker_n_ohta_test = os.path.join(
            self._temporary_directory, "colorchecker_n_ohta.csv"
        )
        write_sds_to_csv_file(sds, colour_checker_n_ohta_test)
        sds_test = read_sds_from_csv_file(colour_checker_n_ohta_test)
        for key, value in sds.items():
            self.assertEqual(value, sds_test[key])

    def test_raise_exception_write_sds_to_csv_file(self):
        """
        Test :func:`colour.io.tabular.write_sds_to_csv_file` definition
        raised exception.
        """

        colour_checker_n_ohta = os.path.join(
            RESOURCES_DIRECTORY, "colorchecker_n_ohta.csv"
        )
        sds = read_sds_from_csv_file(colour_checker_n_ohta)
        key = list(sds.keys())[0]
        sds[key] = sds[key].align(SpectralShape(400, 700, 10))

        self.assertRaises(ValueError, write_sds_to_csv_file, sds, "")


if __name__ == "__main__":
    unittest.main()
