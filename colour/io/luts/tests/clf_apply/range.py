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


class TestRange:

    def test_ocio_consistency_simple(self):
        """
        Test that the execution is consistent with `ociochecklut`.
        """

        example = """
        <Range inBitDepth="10i" outBitDepth="10i">
            <Description>10-bit full range to SMPTE range</Description>
            <minInValue>0</minInValue>
            <maxInValue>1023</maxInValue>
            <minOutValue>64</minOutValue>
            <maxOutValue>940</maxOutValue>
        </Range>
        """
        for rgb in rgb_iter():
            value_rgb = np.array(rgb)
            assert_ocio_consistency(value_rgb, example, f"Input value was {rgb}")

if __name__ == "__main__":
    unittest.main()
