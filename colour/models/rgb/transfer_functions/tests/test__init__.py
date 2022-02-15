"""
Defines the unit tests for the
:mod:`colour.models.rgb.transfer_functions.common` module.
"""

import numpy as np
import unittest

from colour.models.rgb.transfer_functions import (
    CCTF_DECODINGS,
    CCTF_ENCODINGS,
    EOTFS,
    EOTF_INVERSES,
    LOG_DECODINGS,
    LOG_ENCODINGS,
    OETFS,
    OETF_INVERSES,
    OOTFS,
    OOTF_INVERSES,
    cctf_encoding,
    cctf_decoding,
)
from colour.utilities import ColourUsageWarning, as_int

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Development"

__all__ = [
    "TestCctfEncoding",
    "TestCctfDecoding",
    "TestTransferFunctions",
]


class TestCctfEncoding(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.cctf_encoding`
    definition unit tests methods.
    """

    def test_raise_exception_cctf_encoding(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACESproxy` definition raised exception.
        """

        self.assertWarns(
            ColourUsageWarning,
            cctf_encoding,
            0.18,
            function="ITU-R BT.2100 HLG",
        )
        self.assertWarns(
            ColourUsageWarning,
            cctf_encoding,
            0.18,
            function="ITU-R BT.2100 PQ",
        )


class TestCctfDecoding(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.cctf_decoding`
    definition unit tests methods.
    """

    def test_raise_exception_cctf_decoding(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACESproxy` definition raised exception.
        """

        self.assertWarns(
            ColourUsageWarning,
            cctf_decoding,
            0.18,
            function="ITU-R BT.2100 HLG",
        )
        self.assertWarns(
            ColourUsageWarning,
            cctf_decoding,
            0.18,
            function="ITU-R BT.2100 PQ",
        )


class TestTransferFunctions(unittest.TestCase):
    """Define the transfer functions unit tests methods."""

    def test_transfer_functions(self):
        """Test the transfer functions reciprocity."""

        ignored_transfer_functions = (
            "ACESproxy",
            "DICOM GSDF",
            "Filmic Pro 6",
        )

        decimals = {"D-Log": 1, "F-Log": 4, "N-Log": 3}

        reciprocal_mappings = [
            (LOG_ENCODINGS, LOG_DECODINGS),
            (OETFS, OETF_INVERSES),
            (EOTFS, EOTF_INVERSES),
            (CCTF_ENCODINGS, CCTF_DECODINGS),
            (OOTFS, OOTF_INVERSES),
        ]

        samples = np.hstack(
            [np.linspace(0, 1, as_int(1e5)), np.linspace(0, 65504, 65504 * 10)]
        )

        for encoding_mapping, _decoding_mapping in reciprocal_mappings:
            for name in encoding_mapping:
                if name in ignored_transfer_functions:
                    continue

                encoded_s = CCTF_ENCODINGS[name](samples)
                decoded_s = CCTF_DECODINGS[name](encoded_s)

                np.testing.assert_almost_equal(
                    samples, decoded_s, decimal=decimals.get(name, 7)
                )


if __name__ == "__main__":
    unittest.main()
