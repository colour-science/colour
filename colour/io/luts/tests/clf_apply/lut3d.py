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


class TestLUT3D:

    EXAMPLE_SIMPLE = """
    <LUT3D 
        id="lut-24" 
        name="green look" 
        interpolation="trilinear" 
        inBitDepth="12i" 
        outBitDepth="16f"
    >
        <Description>3D LUT</Description>
        <Array dim="2 2 2 3">
            0.0 0.0 0.0
            0.0 0.0 1.0
            0.0 1.0 0.0
            0.0 1.0 1.0
            1.0 0.0 0.0
            1.0 0.0 1.0
            1.0 1.0 0.0
            1.0 1.0 1.0
        </Array>
    </LUT3D>
    """

    def test_ocio_consistency_simple(self):
        """
        Test that the execution of a simple 1D LUT is consistent with `ociochecklut`.
        """
        value_rgb = np.array([1.0, 0.5, 0.0])
        assert_ocio_consistency(
            value_rgb, self.EXAMPLE_SIMPLE
        )

    def test_ocio_consistency_tetrahedral_interolation(self):
        example =  """
            <LUT3D 
                id="lut-24" 
                name="green look" 
                interpolation="tetrahedral" 
                inBitDepth="32f" 
                outBitDepth="32f"
            >
                <Description>3D LUT</Description>
                <Array dim="2 2 2 3">
                    0.0 0.0 0.0
                    0.0 0.0 1.0
                    0.0 1.0 0.0
                    0.0 1.0 1.0
                    1.0 0.0 0.0
                    1.0 0.0 1.0
                    1.0 1.0 0.0
                    1.0 1.0 1.0
                </Array>
            </LUT3D>
            """
        for rgb in rgb_iter():
            value_rgb = np.array(rgb)
            assert_ocio_consistency(value_rgb, example)


if __name__ == "__main__":
    unittest.main()
