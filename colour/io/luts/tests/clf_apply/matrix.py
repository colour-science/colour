# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.io.clf` module."""
import os
import unittest

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

import numpy as np
import pytest
from colour_clf_io.values import BitDepth

from colour.io.luts.clf import from_f16_to_uint16, from_uint16_to_f16
from colour.io.luts.tests.test_clf_common import (
    assert_ocio_consistency,
    assert_ocio_consistency_for_file,
)

def rgb_iter(step=0.2):
    for r in np.arange(0.0, 1.0, step):
        for g in np.arange(0.0, 1.0, step):
            for b in np.arange(0.0, 1.0, step):
                yield r, g, b


class TestMatrix:

    def test_ocio_consistency_simple(self):
        """
        Test that the execution is consistent with `ociochecklut`.
        """

        example = """
        <Matrix id="lut-28" name="AP0 to AP1" inBitDepth="16f" outBitDepth="16f" >
            <Description>3x3 color space conversion from AP0 to AP1</Description>
            <Array dim="3 3">
                 1.45143931614567     -0.236510746893740    -0.214928569251925
                -0.0765537733960204    1.17622969983357     -0.0996759264375522
                 0.00831614842569772  -0.00603244979102103   0.997716301365324
            </Array>
        </Matrix>   
        """
        for rgb in rgb_iter():
            value_rgb = np.array(rgb)
            assert_ocio_consistency(value_rgb, example)

if __name__ == "__main__":
    unittest.main()
