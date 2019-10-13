# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.models.rgb.transfer_functions.common`
module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.models.rgb.transfer_functions import (
    CCTFS_DECODING, CCTFS_ENCODING, EOTFS, EOTFS_INVERSE, LOGS_DECODING,
    LOGS_ENCODING, OETFS, OETFS_INVERSE, OOTFS, OOTFS_INVERSE, cctf_encoding,
    cctf_decoding)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Development'

__all__ = ['TestCctfEncoding', 'TestCctfDecoding', 'TestTransferFunctions']


class TestCctfEncoding(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.cctf_encoding`
    definition unit tests methods.
    """

    def test_raise_exception_cctf_encoding(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACESproxy` definition raised exception.
        """

        # TODO: Use "assertWarns" when dropping Python 2.7.
        cctf_encoding(0.18, 'ITU-R BT.2100 HLG')
        cctf_encoding(0.18, 'ITU-R BT.2100 PQ')


class TestCctfDecoding(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.cctf_decoding`
    definition unit tests methods.
    """

    def test_raise_exception_cctf_decoding(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACESproxy` definition raised exception.
        """

        # TODO: Use "assertWarns" when dropping Python 2.7.
        cctf_decoding(0.18, 'ITU-R BT.2100 HLG')
        cctf_decoding(0.18, 'ITU-R BT.2100 PQ')


class TestTransferFunctions(unittest.TestCase):
    """
    Defines transfer functions unit tests methods.
    """

    def test_transfer_functions(self):
        """
        Tests transfer functions reciprocity.
        """

        ignored_transfer_functions = ('ACESproxy', 'DICOM GSDF',
                                      'Filmic Pro 6')

        decimals = {'D-Log': 1, 'F-Log': 4}

        reciprocal_mappings = [
            (LOGS_ENCODING, LOGS_DECODING),
            (OETFS, OETFS_INVERSE),
            (EOTFS, EOTFS_INVERSE),
            (CCTFS_ENCODING, CCTFS_DECODING),
            (OOTFS, OOTFS_INVERSE),
        ]

        samples = np.hstack(
            [np.linspace(0, 1, 1e5),
             np.linspace(0, 65504, 65504 * 10)])

        for encoding_mapping, _decoding_mapping in reciprocal_mappings:
            for name in encoding_mapping:
                if name in ignored_transfer_functions:
                    continue

                encoded_s = CCTFS_ENCODING[name](samples)
                decoded_s = CCTFS_DECODING[name](encoded_s)

                np.testing.assert_almost_equal(
                    samples, decoded_s, decimal=decimals.get(name, 7))


if __name__ == '__main__':
    unittest.main()
