"""Define the unit tests for the :mod:`colour.io.tm2714` module."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import textwrap
import typing
from copy import deepcopy

import pytest

from colour.colorimetry import SpectralDistribution
from colour.constants import TOLERANCE_ABSOLUTE_TESTS

if typing.TYPE_CHECKING:
    from colour.hints import List, Tuple

from colour.hints import cast
from colour.io.tm2714 import Header_IESTM2714, SpectralDistribution_IESTM2714
from colour.utilities import (
    optional,
    xp_assert_close,
    xp_assert_equal,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
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


class TestIES_TM2714_Header:
    """
    Define :class:`colour.io.tm2714.Header_IESTM2714` class unit tests
    methods.
    """

    def setup_method(self) -> None:
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

    def test_required_attributes(self) -> None:
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
            assert attribute in dir(Header_IESTM2714)

    def test_required_methods(self) -> None:
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
            assert method in dir(Header_IESTM2714)

    def test__str__(self) -> None:
        """Test :meth:`colour.io.tm2714.Header_IESTM2714.__str__` method."""

        assert str(self._header) == (
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
            ).strip()
        )

    def test__repr__(self) -> None:
        """Test :meth:`colour.io.tm2714.Header_IESTM2714.__repr__` method."""

        assert repr(self._header) == (
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
            ).strip()
        )

    def test__eq__(self) -> None:
        """Test :meth:`colour.io.tm2714.Header_IESTM2714.__eq__` method."""

        header = deepcopy(self._header)

        assert self._header == header

        assert self._header != ()

    def test__ne__(self) -> None:
        """Test :meth:`colour.io.tm2714.Header_IESTM2714.__ne__` method."""

        header = deepcopy(self._header)

        header.manufacturer = "aa"
        assert self._header != header

        header.manufacturer = "a"
        assert self._header == header

    def test__hash__(self) -> None:
        """Test :meth:`colour.io.tm2714.Header_IESTM2714.__hash__` method."""

        assert isinstance(hash(self._header), int)


class TestIES_TM2714_Sd:
    """
    Define :class:`colour.io.tm2714.SpectralDistribution_IESTM2714` class unit
    tests methods.
    """

    def setup_method(self) -> None:
        """Initialise the common tests attributes."""

        self._temporary_directory = tempfile.mkdtemp()

        self._sd = SpectralDistribution_IESTM2714(
            os.path.join(ROOT_RESOURCES, "Fluorescent.spdx")
        ).read()

    def teardown_method(self) -> None:
        """After tests actions."""

        shutil.rmtree(self._temporary_directory)

    def test_required_attributes(self) -> None:
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
            assert attribute in dir(SpectralDistribution_IESTM2714)

    def test_required_methods(self) -> None:
        """Test the presence of required methods."""

        required_methods = ("__init__", "__str__", "__repr__", "read", "write")

        for method in required_methods:
            assert method in dir(SpectralDistribution_IESTM2714)

    def test__str__(self) -> None:
        """
        Test :meth:`colour.io.tm2714.SpectralDistribution_IESTM2714.__str__`
        method.
        """

        assert re.sub(
            "Path                  :.*",
            "Path                  :",
            str(self._sd),
        ) == (
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

                [[4.000e+02 3.400e-02]
                 [4.031e+02 3.700e-02]
                 [4.055e+02 6.900e-02]
                 [4.075e+02 3.700e-02]
                 [4.206e+02 4.200e-02]
                 [4.310e+02 4.900e-02]
                 [4.337e+02 6.000e-02]
                 [4.370e+02 3.570e-01]
                 [4.389e+02 6.000e-02]
                 [4.600e+02 6.800e-02]
                 [4.770e+02 7.500e-02]
                 [4.810e+02 8.500e-02]
                 [4.882e+02 2.040e-01]
                 [4.926e+02 1.660e-01]
                 [5.017e+02 9.500e-02]
                 [5.076e+02 7.800e-02]
                 [5.176e+02 7.100e-02]
                 [5.299e+02 7.600e-02]
                 [5.354e+02 9.900e-02]
                 [5.399e+02 4.230e-01]
                 [5.432e+02 8.020e-01]
                 [5.444e+02 7.130e-01]
                 [5.472e+02 9.990e-01]
                 [5.487e+02 5.730e-01]
                 [5.502e+02 3.400e-01]
                 [5.538e+02 2.080e-01]
                 [5.573e+02 1.390e-01]
                 [5.637e+02 1.290e-01]
                 [5.748e+02 1.310e-01]
                 [5.780e+02 1.980e-01]
                 [5.792e+02 1.900e-01]
                 [5.804e+02 2.050e-01]
                 [5.848e+02 2.440e-01]
                 [5.859e+02 2.360e-01]
                 [5.875e+02 2.560e-01]
                 [5.903e+02 1.800e-01]
                 [5.935e+02 2.180e-01]
                 [5.955e+02 1.590e-01]
                 [5.970e+02 1.470e-01]
                 [5.994e+02 1.700e-01]
                 [6.022e+02 1.340e-01]
                 [6.046e+02 1.210e-01]
                 [6.074e+02 1.400e-01]
                 [6.094e+02 2.290e-01]
                 [6.102e+02 4.650e-01]
                 [6.120e+02 9.520e-01]
                 [6.146e+02 4.770e-01]
                 [6.169e+02 2.080e-01]
                 [6.185e+02 1.350e-01]
                 [6.221e+02 1.500e-01]
                 [6.256e+02 1.550e-01]
                 [6.284e+02 1.340e-01]
                 [6.312e+02 1.680e-01]
                 [6.332e+02 8.700e-02]
                 [6.356e+02 6.800e-02]
                 [6.427e+02 5.800e-02]
                 [6.487e+02 5.800e-02]
                 [6.507e+02 7.400e-02]
                 [6.526e+02 6.300e-02]
                 [6.562e+02 5.300e-02]
                 [6.570e+02 5.600e-02]
                 [6.606e+02 4.900e-02]
                 [6.626e+02 5.900e-02]
                 [6.642e+02 4.800e-02]
                 [6.860e+02 4.100e-02]
                 [6.876e+02 4.800e-02]
                 [6.892e+02 3.900e-02]
                 [6.924e+02 3.800e-02]
                 [6.935e+02 4.400e-02]
                 [6.955e+02 3.400e-02]
                 [7.023e+02 3.600e-02]
                 [7.067e+02 4.200e-02]
                 [7.071e+02 6.100e-02]
                 [7.102e+02 6.100e-02]
                 [7.110e+02 4.100e-02]
                 [7.122e+02 5.200e-02]
                 [7.142e+02 3.300e-02]
                 [7.484e+02 3.400e-02]
                 [7.579e+02 3.100e-02]
                 [7.607e+02 3.900e-02]
                 [7.639e+02 2.900e-02]
                 [8.088e+02 2.900e-02]
                 [8.107e+02 3.900e-02]
                 [8.127e+02 3.000e-02]
                 [8.501e+02 3.000e-02]]
                """
            ).strip()
        )

    def test__repr__(self) -> None:
        """
        Test :meth:`colour.io.tm2714.SpectralDistribution_IESTM2714.__repr__`
        method.
        """

        assert re.sub(
            "SpectralDistribution_IESTM2714.*",
            "SpectralDistribution_IESTM2714(...,",
            repr(self._sd),
        ) == (
            textwrap.dedent(
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
                               np.float64(2.0),
                               True,
                               [[4.000e+02, 3.400e-02],
                                [4.031e+02, 3.700e-02],
                                [4.055e+02, 6.900e-02],
                                [4.075e+02, 3.700e-02],
                                [4.206e+02, 4.200e-02],
                                [4.310e+02, 4.900e-02],
                                [4.337e+02, 6.000e-02],
                                [4.370e+02, 3.570e-01],
                                [4.389e+02, 6.000e-02],
                                [4.600e+02, 6.800e-02],
                                [4.770e+02, 7.500e-02],
                                [4.810e+02, 8.500e-02],
                                [4.882e+02, 2.040e-01],
                                [4.926e+02, 1.660e-01],
                                [5.017e+02, 9.500e-02],
                                [5.076e+02, 7.800e-02],
                                [5.176e+02, 7.100e-02],
                                [5.299e+02, 7.600e-02],
                                [5.354e+02, 9.900e-02],
                                [5.399e+02, 4.230e-01],
                                [5.432e+02, 8.020e-01],
                                [5.444e+02, 7.130e-01],
                                [5.472e+02, 9.990e-01],
                                [5.487e+02, 5.730e-01],
                                [5.502e+02, 3.400e-01],
                                [5.538e+02, 2.080e-01],
                                [5.573e+02, 1.390e-01],
                                [5.637e+02, 1.290e-01],
                                [5.748e+02, 1.310e-01],
                                [5.780e+02, 1.980e-01],
                                [5.792e+02, 1.900e-01],
                                [5.804e+02, 2.050e-01],
                                [5.848e+02, 2.440e-01],
                                [5.859e+02, 2.360e-01],
                                [5.875e+02, 2.560e-01],
                                [5.903e+02, 1.800e-01],
                                [5.935e+02, 2.180e-01],
                                [5.955e+02, 1.590e-01],
                                [5.970e+02, 1.470e-01],
                                [5.994e+02, 1.700e-01],
                                [6.022e+02, 1.340e-01],
                                [6.046e+02, 1.210e-01],
                                [6.074e+02, 1.400e-01],
                                [6.094e+02, 2.290e-01],
                                [6.102e+02, 4.650e-01],
                                [6.120e+02, 9.520e-01],
                                [6.146e+02, 4.770e-01],
                                [6.169e+02, 2.080e-01],
                                [6.185e+02, 1.350e-01],
                                [6.221e+02, 1.500e-01],
                                [6.256e+02, 1.550e-01],
                                [6.284e+02, 1.340e-01],
                                [6.312e+02, 1.680e-01],
                                [6.332e+02, 8.700e-02],
                                [6.356e+02, 6.800e-02],
                                [6.427e+02, 5.800e-02],
                                [6.487e+02, 5.800e-02],
                                [6.507e+02, 7.400e-02],
                                [6.526e+02, 6.300e-02],
                                [6.562e+02, 5.300e-02],
                                [6.570e+02, 5.600e-02],
                                [6.606e+02, 4.900e-02],
                                [6.626e+02, 5.900e-02],
                                [6.642e+02, 4.800e-02],
                                [6.860e+02, 4.100e-02],
                                [6.876e+02, 4.800e-02],
                                [6.892e+02, 3.900e-02],
                                [6.924e+02, 3.800e-02],
                                [6.935e+02, 4.400e-02],
                                [6.955e+02, 3.400e-02],
                                [7.023e+02, 3.600e-02],
                                [7.067e+02, 4.200e-02],
                                [7.071e+02, 6.100e-02],
                                [7.102e+02, 6.100e-02],
                                [7.110e+02, 4.100e-02],
                                [7.122e+02, 5.200e-02],
                                [7.142e+02, 3.300e-02],
                                [7.484e+02, 3.400e-02],
                                [7.579e+02, 3.100e-02],
                                [7.607e+02, 3.900e-02],
                                [7.639e+02, 2.900e-02],
                                [8.088e+02, 2.900e-02],
                                [8.107e+02, 3.900e-02],
                                [8.127e+02, 3.000e-02],
                                [8.501e+02, 3.000e-02]],
                               CubicSplineInterpolator,
                               {},
                               Extrapolator,
                               {'method': 'Constant', 'left': None, 'right': None})
                """
            ).strip()
        )

    def test_read(self, sd: SpectralDistribution | None = None) -> None:
        """
        Test :meth:`colour.io.tm2714.SpectralDistribution_IESTM2714.read`
        method.

        Parameters
        ----------
        sd
            Optional *IES TM-27-14* spectral distribution for read tests.
        """

        sd = cast(
            "SpectralDistribution_IESTM2714",
            optional(
                sd,
                SpectralDistribution_IESTM2714(
                    os.path.join(ROOT_RESOURCES, "Fluorescent.spdx")
                ).read(),
            ),
        )

        sd_r = SpectralDistribution(FLUORESCENT_FILE_SPECTRAL_DATA)

        xp_assert_equal(sd_r.domain, sd.domain)
        xp_assert_close(sd_r.values, sd.values, atol=TOLERANCE_ABSOLUTE_TESTS)

        test_read: List[
            Tuple[dict, Header_IESTM2714 | SpectralDistribution_IESTM2714]
        ] = [
            (FLUORESCENT_FILE_HEADER, sd.header),
            (FLUORESCENT_FILE_SPECTRAL_DESCRIPTION, sd),
        ]
        for test, read in test_read:
            for key, value in test.items():
                for specification in read.mapping.elements:
                    if key == specification.element:
                        assert getattr(read, specification.attribute) == value

    def test_raise_exception_read(self) -> None:
        """
        Test :func:`colour.io.tm2714.SpectralDistribution_IESTM2714.read`
        method raised exception.
        """

        sd = SpectralDistribution_IESTM2714()
        with pytest.raises(ValueError):
            sd.read()

        with pytest.raises(ValueError):
            sd = SpectralDistribution_IESTM2714(
                os.path.join(ROOT_RESOURCES, "Invalid.spdx")
            )

    def test_write(self) -> None:
        """
        Test :meth:`colour.io.tm2714.SpectralDistribution_IESTM2714.write`
        method.
        """

        sd_r = self._sd

        sd_r.path = os.path.join(self._temporary_directory, "Fluorescent.spdx")
        assert sd_r.write()
        sd_t = SpectralDistribution_IESTM2714(sd_r.path).read()

        self.test_read(sd_t)
        assert sd_r == sd_t

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
            assert getattr(sd_r.header, attribute) == getattr(sd_t.header, attribute)

        for attribute in (
            "spectral_quantity",
            "reflection_geometry",
            "transmission_geometry",
            "bandwidth_FWHM",
            "bandwidth_corrected",
        ):
            assert getattr(sd_r, attribute) == getattr(sd_t, attribute)

    def test_raise_exception_write(self) -> None:
        """
        Test :func:`colour.io.tm2714.SpectralDistribution_IESTM2714.write`
        method raised exception.
        """

        sd = SpectralDistribution_IESTM2714()
        with pytest.raises(ValueError):
            sd.write()
