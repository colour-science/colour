# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.io.tm2714` module."""

from __future__ import annotations

import numpy as np
import os
import re
import shutil
import unittest
import tempfile
import textwrap
from copy import deepcopy

from colour.colorimetry import SpectralDistribution
from colour.hints import Optional, List, Tuple, Union, cast
from colour.io.tm2714 import Header_IESTM2714, SpectralDistribution_IESTM2714
from colour.utilities import optional

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "ROOT_RESOURCES",
    "FLUORESCENT_FILE_HEADER",
    "FLUORESCENT_FILE_SPECTRAL_DESCRIPTION",
    "FLUORESCENT_FILE_SPECTRAL_DATA",
    "TestIES_TM2714_Header",
    "TestIES_TM2714_Sd",
]

ROOT_RESOURCES: str = os.path.join(os.path.dirname(__file__), "resources")

FLUORESCENT_FILE_HEADER: dict = {
    "Manufacturer": "Unknown",
    "CatalogNumber": "N/A",
    "Description": "Rare earth fluorescent lamp",
    "DocumentCreator": "byHeart Consultants",
    "Laboratory": "N/A",
    "UniqueIdentifier": "C3567553-C75B-4354-961E-35CEB9FEB42C",
    "ReportNumber": "N/A",
    "ReportDate": "N/A",
    "DocumentCreationDate": "2014-06-23",
    "Comments": "Ambient temperature 25 degrees C.",
}

FLUORESCENT_FILE_SPECTRAL_DESCRIPTION: dict = {
    "SpectralQuantity": "relative",
    "BandwidthFWHM": 2.0,
    "BandwidthCorrected": True,
}

FLUORESCENT_FILE_SPECTRAL_DATA: dict = {
    400.0: 0.034,
    403.1: 0.037,
    405.5: 0.069,
    407.5: 0.037,
    420.6: 0.042,
    431.0: 0.049,
    433.7: 0.060,
    437.0: 0.357,
    438.9: 0.060,
    460.0: 0.068,
    477.0: 0.075,
    481.0: 0.085,
    488.2: 0.204,
    492.6: 0.166,
    501.7: 0.095,
    507.6: 0.078,
    517.6: 0.071,
    529.9: 0.076,
    535.4: 0.099,
    539.9: 0.423,
    543.2: 0.802,
    544.4: 0.713,
    547.2: 0.999,
    548.7: 0.573,
    550.2: 0.340,
    553.8: 0.208,
    557.3: 0.139,
    563.7: 0.129,
    574.8: 0.131,
    578.0: 0.198,
    579.2: 0.190,
    580.4: 0.205,
    584.8: 0.244,
    585.9: 0.236,
    587.5: 0.256,
    590.3: 0.180,
    593.5: 0.218,
    595.5: 0.159,
    597.0: 0.147,
    599.4: 0.170,
    602.2: 0.134,
    604.6: 0.121,
    607.4: 0.140,
    609.4: 0.229,
    610.2: 0.465,
    612.0: 0.952,
    614.6: 0.477,
    616.9: 0.208,
    618.5: 0.135,
    622.1: 0.150,
    625.6: 0.155,
    628.4: 0.134,
    631.2: 0.168,
    633.2: 0.087,
    635.6: 0.068,
    642.7: 0.058,
    648.7: 0.058,
    650.7: 0.074,
    652.6: 0.063,
    656.2: 0.053,
    657.0: 0.056,
    660.6: 0.049,
    662.6: 0.059,
    664.2: 0.048,
    686.0: 0.041,
    687.6: 0.048,
    689.2: 0.039,
    692.4: 0.038,
    693.5: 0.044,
    695.5: 0.034,
    702.3: 0.036,
    706.7: 0.042,
    707.1: 0.061,
    710.2: 0.061,
    711.0: 0.041,
    712.2: 0.052,
    714.2: 0.033,
    748.4: 0.034,
    757.9: 0.031,
    760.7: 0.039,
    763.9: 0.029,
    808.8: 0.029,
    810.7: 0.039,
    812.7: 0.030,
    850.1: 0.030,
}


class TestIES_TM2714_Header(unittest.TestCase):
    """
    Define :class:`colour.io.tm2714.Header_IESTM2714` class unit tests
    methods.
    """

    def setUp(self):
        """Initialise the common tests attributes."""

        self._header = Header_IESTM2714(
            manufacturer="a",
            catalog_number="b",
            description="c",
            document_creator="d",
            unique_identifier="e",
            measurement_equipment="f",
            laboratory="g",
            report_number="h",
            report_date="i",
            document_creation_date="j",
            comments="k",
        )

    def test_required_attributes(self):
        """Test the presence of required attributes."""

        required_attributes = (
            "mapping",
            "manufacturer",
            "catalog_number",
            "description",
            "document_creator",
            "unique_identifier",
            "measurement_equipment",
            "laboratory",
            "report_number",
            "report_date",
            "document_creation_date",
            "comments",
        )

        for attribute in required_attributes:
            self.assertIn(attribute, dir(Header_IESTM2714))

    def test_required_methods(self):
        """Test the presence of required methods."""

        required_methods = (
            "__init__",
            "__str__",
            "__repr__",
            "__hash__",
            "__eq__",
            "__ne__",
        )

        for method in required_methods:
            self.assertIn(method, dir(Header_IESTM2714))

    def test__str__(self):
        """Test :meth:`colour.io.tm2714.Header_IESTM2714.__str__` method."""

        self.assertEqual(
            str(self._header),
            textwrap.dedent(
                """
                Manufacturer           : a
                Catalog Number         : b
                Description            : c
                Document Creator       : d
                Unique Identifier      : e
                Measurement Equipment  : f
                Laboratory             : g
                Report Number          : h
                Report Date            : i
                Document Creation Date : j
                Comments               : k
                """
            ).strip(),
        )

    def test__repr__(self):
        """Test :meth:`colour.io.tm2714.Header_IESTM2714.__repr__` method."""

        self.assertEqual(
            repr(self._header),
            textwrap.dedent(
                """
                Header_IESTM2714('a',
                                 'b',
                                 'c',
                                 'd',
                                 'e',
                                 'f',
                                 'g',
                                 'h',
                                 'i',
                                 'j',
                                 'k')
                """
            ).strip(),
        )

    def test__eq__(self):
        """Test :meth:`colour.io.tm2714.Header_IESTM2714.__eq__` method."""

        header = deepcopy(self._header)
        self.assertEqual(self._header, header)

        self.assertNotEqual(self._header, None)

    def test__ne__(self):
        """Test :meth:`colour.io.tm2714.Header_IESTM2714.__ne__` method."""

        header = deepcopy(self._header)

        header.manufacturer = "aa"
        self.assertNotEqual(self._header, header)

        header.manufacturer = "a"
        self.assertEqual(self._header, header)

    def test__hash__(self):
        """Test :meth:`colour.io.tm2714.Header_IESTM2714.__hash__` method."""

        self.assertIsInstance(hash(self._header), int)


class TestIES_TM2714_Sd(unittest.TestCase):
    """
    Define :class:`colour.io.tm2714.SpectralDistribution_IESTM2714` class unit
    tests methods.
    """

    def setUp(self):
        """Initialise the common tests attributes."""

        self._temporary_directory = tempfile.mkdtemp()

        self._sd = SpectralDistribution_IESTM2714(
            os.path.join(ROOT_RESOURCES, "Fluorescent.spdx")
        ).read()

    def tearDown(self):
        """After tests actions."""

        shutil.rmtree(self._temporary_directory)

    def test_required_attributes(self):
        """Test the presence of required attributes."""

        required_attributes = (
            "mapping",
            "path",
            "header",
            "spectral_quantity",
            "reflection_geometry",
            "transmission_geometry",
            "bandwidth_FWHM",
            "bandwidth_corrected",
        )

        for attribute in required_attributes:
            self.assertIn(attribute, dir(SpectralDistribution_IESTM2714))

    def test_required_methods(self):
        """Test the presence of required methods."""

        required_methods = ("__init__", "__str__", "__repr__", "read", "write")

        for method in required_methods:
            self.assertIn(method, dir(SpectralDistribution_IESTM2714))

    def test__str__(self):
        """
        Test :meth:`colour.io.tm2714.SpectralDistribution_IESTM2714.__str__`
        method.
        """

        self.assertEqual(
            re.sub(
                "Path                  :.*",
                "Path                  :",
                str(self._sd),
            ),
            textwrap.dedent(
                """
                IES TM-27-14 Spectral Distribution
                ==================================

                Path                  :
                Spectral Quantity     : relative
                Reflection Geometry   : other
                Transmission Geometry : other
                Bandwidth (FWHM)      : 2.0
                Bandwidth Corrected   : True

                Header
                ------

                Manufacturer           : Unknown
                Catalog Number         : N/A
                Description            : Rare earth fluorescent lamp
                Document Creator       : byHeart Consultants
                Unique Identifier      : C3567553-C75B-4354-961E-35CEB9FEB42C
                Measurement Equipment  : None
                Laboratory             : N/A
                Report Number          : N/A
                Report Date            : N/A
                Document Creation Date : 2014-06-23
                Comments               : Ambient temperature 25 degrees C.

                Spectral Data
                -------------

                [[  4.00000000e+02   3.40000000e-02]
                 [  4.03100000e+02   3.70000000e-02]
                 [  4.05500000e+02   6.90000000e-02]
                 [  4.07500000e+02   3.70000000e-02]
                 [  4.20600000e+02   4.20000000e-02]
                 [  4.31000000e+02   4.90000000e-02]
                 [  4.33700000e+02   6.00000000e-02]
                 [  4.37000000e+02   3.57000000e-01]
                 [  4.38900000e+02   6.00000000e-02]
                 [  4.60000000e+02   6.80000000e-02]
                 [  4.77000000e+02   7.50000000e-02]
                 [  4.81000000e+02   8.50000000e-02]
                 [  4.88200000e+02   2.04000000e-01]
                 [  4.92600000e+02   1.66000000e-01]
                 [  5.01700000e+02   9.50000000e-02]
                 [  5.07600000e+02   7.80000000e-02]
                 [  5.17600000e+02   7.10000000e-02]
                 [  5.29900000e+02   7.60000000e-02]
                 [  5.35400000e+02   9.90000000e-02]
                 [  5.39900000e+02   4.23000000e-01]
                 [  5.43200000e+02   8.02000000e-01]
                 [  5.44400000e+02   7.13000000e-01]
                 [  5.47200000e+02   9.99000000e-01]
                 [  5.48700000e+02   5.73000000e-01]
                 [  5.50200000e+02   3.40000000e-01]
                 [  5.53800000e+02   2.08000000e-01]
                 [  5.57300000e+02   1.39000000e-01]
                 [  5.63700000e+02   1.29000000e-01]
                 [  5.74800000e+02   1.31000000e-01]
                 [  5.78000000e+02   1.98000000e-01]
                 [  5.79200000e+02   1.90000000e-01]
                 [  5.80400000e+02   2.05000000e-01]
                 [  5.84800000e+02   2.44000000e-01]
                 [  5.85900000e+02   2.36000000e-01]
                 [  5.87500000e+02   2.56000000e-01]
                 [  5.90300000e+02   1.80000000e-01]
                 [  5.93500000e+02   2.18000000e-01]
                 [  5.95500000e+02   1.59000000e-01]
                 [  5.97000000e+02   1.47000000e-01]
                 [  5.99400000e+02   1.70000000e-01]
                 [  6.02200000e+02   1.34000000e-01]
                 [  6.04600000e+02   1.21000000e-01]
                 [  6.07400000e+02   1.40000000e-01]
                 [  6.09400000e+02   2.29000000e-01]
                 [  6.10200000e+02   4.65000000e-01]
                 [  6.12000000e+02   9.52000000e-01]
                 [  6.14600000e+02   4.77000000e-01]
                 [  6.16900000e+02   2.08000000e-01]
                 [  6.18500000e+02   1.35000000e-01]
                 [  6.22100000e+02   1.50000000e-01]
                 [  6.25600000e+02   1.55000000e-01]
                 [  6.28400000e+02   1.34000000e-01]
                 [  6.31200000e+02   1.68000000e-01]
                 [  6.33200000e+02   8.70000000e-02]
                 [  6.35600000e+02   6.80000000e-02]
                 [  6.42700000e+02   5.80000000e-02]
                 [  6.48700000e+02   5.80000000e-02]
                 [  6.50700000e+02   7.40000000e-02]
                 [  6.52600000e+02   6.30000000e-02]
                 [  6.56200000e+02   5.30000000e-02]
                 [  6.57000000e+02   5.60000000e-02]
                 [  6.60600000e+02   4.90000000e-02]
                 [  6.62600000e+02   5.90000000e-02]
                 [  6.64200000e+02   4.80000000e-02]
                 [  6.86000000e+02   4.10000000e-02]
                 [  6.87600000e+02   4.80000000e-02]
                 [  6.89200000e+02   3.90000000e-02]
                 [  6.92400000e+02   3.80000000e-02]
                 [  6.93500000e+02   4.40000000e-02]
                 [  6.95500000e+02   3.40000000e-02]
                 [  7.02300000e+02   3.60000000e-02]
                 [  7.06700000e+02   4.20000000e-02]
                 [  7.07100000e+02   6.10000000e-02]
                 [  7.10200000e+02   6.10000000e-02]
                 [  7.11000000e+02   4.10000000e-02]
                 [  7.12200000e+02   5.20000000e-02]
                 [  7.14200000e+02   3.30000000e-02]
                 [  7.48400000e+02   3.40000000e-02]
                 [  7.57900000e+02   3.10000000e-02]
                 [  7.60700000e+02   3.90000000e-02]
                 [  7.63900000e+02   2.90000000e-02]
                 [  8.08800000e+02   2.90000000e-02]
                 [  8.10700000e+02   3.90000000e-02]
                 [  8.12700000e+02   3.00000000e-02]
                 [  8.50100000e+02   3.00000000e-02]]
                """
            ).strip(),
        )

    def test__repr__(self):
        """
        Test :meth:`colour.io.tm2714.SpectralDistribution_IESTM2714.__repr__`
        method.
        """

        self.assertEqual(
            re.sub(
                "SpectralDistribution_IESTM2714.*",
                "SpectralDistribution_IESTM2714(...,",
                repr(self._sd),
            ),
            textwrap.dedent(  # noqa
                """
SpectralDistribution_IESTM2714(...,
                               Header_IESTM2714('Unknown',
                                                'N/A',
                                                'Rare earth fluorescent lamp',
                                                'byHeart Consultants',
                                                'C3567553-C75B-4354-961E-35CEB9FEB42C',
                                                None,
                                                'N/A',
                                                'N/A',
                                                'N/A',
                                                '2014-06-23',
                                                'Ambient temperature 25 degrees C.'),
                               'relative',
                               'other',
                               'other',
                               2.0,
                               True,
                               [[  4.00000000e+02,   3.40000000e-02],
                                [  4.03100000e+02,   3.70000000e-02],
                                [  4.05500000e+02,   6.90000000e-02],
                                [  4.07500000e+02,   3.70000000e-02],
                                [  4.20600000e+02,   4.20000000e-02],
                                [  4.31000000e+02,   4.90000000e-02],
                                [  4.33700000e+02,   6.00000000e-02],
                                [  4.37000000e+02,   3.57000000e-01],
                                [  4.38900000e+02,   6.00000000e-02],
                                [  4.60000000e+02,   6.80000000e-02],
                                [  4.77000000e+02,   7.50000000e-02],
                                [  4.81000000e+02,   8.50000000e-02],
                                [  4.88200000e+02,   2.04000000e-01],
                                [  4.92600000e+02,   1.66000000e-01],
                                [  5.01700000e+02,   9.50000000e-02],
                                [  5.07600000e+02,   7.80000000e-02],
                                [  5.17600000e+02,   7.10000000e-02],
                                [  5.29900000e+02,   7.60000000e-02],
                                [  5.35400000e+02,   9.90000000e-02],
                                [  5.39900000e+02,   4.23000000e-01],
                                [  5.43200000e+02,   8.02000000e-01],
                                [  5.44400000e+02,   7.13000000e-01],
                                [  5.47200000e+02,   9.99000000e-01],
                                [  5.48700000e+02,   5.73000000e-01],
                                [  5.50200000e+02,   3.40000000e-01],
                                [  5.53800000e+02,   2.08000000e-01],
                                [  5.57300000e+02,   1.39000000e-01],
                                [  5.63700000e+02,   1.29000000e-01],
                                [  5.74800000e+02,   1.31000000e-01],
                                [  5.78000000e+02,   1.98000000e-01],
                                [  5.79200000e+02,   1.90000000e-01],
                                [  5.80400000e+02,   2.05000000e-01],
                                [  5.84800000e+02,   2.44000000e-01],
                                [  5.85900000e+02,   2.36000000e-01],
                                [  5.87500000e+02,   2.56000000e-01],
                                [  5.90300000e+02,   1.80000000e-01],
                                [  5.93500000e+02,   2.18000000e-01],
                                [  5.95500000e+02,   1.59000000e-01],
                                [  5.97000000e+02,   1.47000000e-01],
                                [  5.99400000e+02,   1.70000000e-01],
                                [  6.02200000e+02,   1.34000000e-01],
                                [  6.04600000e+02,   1.21000000e-01],
                                [  6.07400000e+02,   1.40000000e-01],
                                [  6.09400000e+02,   2.29000000e-01],
                                [  6.10200000e+02,   4.65000000e-01],
                                [  6.12000000e+02,   9.52000000e-01],
                                [  6.14600000e+02,   4.77000000e-01],
                                [  6.16900000e+02,   2.08000000e-01],
                                [  6.18500000e+02,   1.35000000e-01],
                                [  6.22100000e+02,   1.50000000e-01],
                                [  6.25600000e+02,   1.55000000e-01],
                                [  6.28400000e+02,   1.34000000e-01],
                                [  6.31200000e+02,   1.68000000e-01],
                                [  6.33200000e+02,   8.70000000e-02],
                                [  6.35600000e+02,   6.80000000e-02],
                                [  6.42700000e+02,   5.80000000e-02],
                                [  6.48700000e+02,   5.80000000e-02],
                                [  6.50700000e+02,   7.40000000e-02],
                                [  6.52600000e+02,   6.30000000e-02],
                                [  6.56200000e+02,   5.30000000e-02],
                                [  6.57000000e+02,   5.60000000e-02],
                                [  6.60600000e+02,   4.90000000e-02],
                                [  6.62600000e+02,   5.90000000e-02],
                                [  6.64200000e+02,   4.80000000e-02],
                                [  6.86000000e+02,   4.10000000e-02],
                                [  6.87600000e+02,   4.80000000e-02],
                                [  6.89200000e+02,   3.90000000e-02],
                                [  6.92400000e+02,   3.80000000e-02],
                                [  6.93500000e+02,   4.40000000e-02],
                                [  6.95500000e+02,   3.40000000e-02],
                                [  7.02300000e+02,   3.60000000e-02],
                                [  7.06700000e+02,   4.20000000e-02],
                                [  7.07100000e+02,   6.10000000e-02],
                                [  7.10200000e+02,   6.10000000e-02],
                                [  7.11000000e+02,   4.10000000e-02],
                                [  7.12200000e+02,   5.20000000e-02],
                                [  7.14200000e+02,   3.30000000e-02],
                                [  7.48400000e+02,   3.40000000e-02],
                                [  7.57900000e+02,   3.10000000e-02],
                                [  7.60700000e+02,   3.90000000e-02],
                                [  7.63900000e+02,   2.90000000e-02],
                                [  8.08800000e+02,   2.90000000e-02],
                                [  8.10700000e+02,   3.90000000e-02],
                                [  8.12700000e+02,   3.00000000e-02],
                                [  8.50100000e+02,   3.00000000e-02]],
                               CubicSplineInterpolator,
                               {},
                               Extrapolator,
                               {'method': 'Constant', 'left': None, 'right': None})
                """
            ).strip(),
        )

    def test_read(self, sd: Optional[SpectralDistribution] = None):
        """
        Test :meth:`colour.io.tm2714.SpectralDistribution_IESTM2714.read`
        method.

        Parameters
        ----------
        sd
            Optional *IES TM-27-14* spectral distribution for read tests.
        """

        sd = cast(
            SpectralDistribution_IESTM2714,
            optional(
                sd,
                SpectralDistribution_IESTM2714(
                    os.path.join(ROOT_RESOURCES, "Fluorescent.spdx")
                ).read(),
            ),
        )

        sd_r = SpectralDistribution(FLUORESCENT_FILE_SPECTRAL_DATA)

        np.testing.assert_array_equal(sd_r.domain, sd.domain)
        np.testing.assert_array_almost_equal(sd_r.values, sd.values, decimal=7)

        test_read: List[
            Tuple[
                dict, Union[Header_IESTM2714, SpectralDistribution_IESTM2714]
            ]
        ] = [
            (FLUORESCENT_FILE_HEADER, sd.header),
            (FLUORESCENT_FILE_SPECTRAL_DESCRIPTION, sd),
        ]
        for test, read in test_read:
            for key, value in test.items():
                for specification in read.mapping.elements:
                    if key == specification.element:
                        self.assertEqual(
                            getattr(read, specification.attribute), value
                        )

    def test_raise_exception_read(self):
        """
        Test :func:`colour.io.tm2714.SpectralDistribution_IESTM2714.read`
        method raised exception.
        """

        sd = SpectralDistribution_IESTM2714()
        self.assertRaises(ValueError, sd.read)

        sd = SpectralDistribution_IESTM2714(
            os.path.join(ROOT_RESOURCES, "Invalid.spdx")
        )
        self.assertRaises(ValueError, sd.read)

    def test_write(self):
        """
        Test :meth:`colour.io.tm2714.SpectralDistribution_IESTM2714.write`
        method.
        """

        sd_r = self._sd

        sd_r.path = os.path.join(self._temporary_directory, "Fluorescent.spdx")
        self.assertTrue(sd_r.write())
        sd_t = SpectralDistribution_IESTM2714(sd_r.path).read()

        self.test_read(sd_t)
        self.assertEqual(sd_r, sd_t)

        for attribute in (
            "manufacturer",
            "catalog_number",
            "description",
            "document_creator",
            "unique_identifier",
            "measurement_equipment",
            "laboratory",
            "report_number",
            "report_date",
            "document_creation_date",
            "comments",
        ):
            self.assertEqual(
                getattr(sd_r.header, attribute),
                getattr(sd_t.header, attribute),
            )

        for attribute in (
            "spectral_quantity",
            "reflection_geometry",
            "transmission_geometry",
            "bandwidth_FWHM",
            "bandwidth_corrected",
        ):
            self.assertEqual(
                getattr(sd_r, attribute), getattr(sd_t, attribute)
            )

    def test_raise_exception_write(self):
        """
        Test :func:`colour.io.tm2714.SpectralDistribution_IESTM2714.write`
        method raised exception.
        """

        sd = SpectralDistribution_IESTM2714()
        self.assertRaises(ValueError, sd.write)


if __name__ == "__main__":
    unittest.main()
