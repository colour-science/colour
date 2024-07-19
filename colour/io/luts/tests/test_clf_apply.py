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


class TestHelpers:
    """
    Define :func:`colour.io.luts.clf.from_uint16_to_f16` and
    :func:`colour.io.luts.clf.from_f16_to_uint16` unit tests methods.
    """

    def test_uint16_to_f16(self):
        """
        Test :func:`colour.io.luts.clf.from_uint16_to_f16` method.
        """
        value = np.array([0, 15360])
        output = from_uint16_to_f16(value)
        expected = np.array([0, 1.0])
        np.testing.assert_almost_equal(output, expected)

    def test_f16_to_uint16(self):
        """
        Test :func:`colour.io.luts.clf.from_f16_to_uint16` method.
        """
        value = np.array([0, 1.0])
        output = from_f16_to_uint16(value)
        expected = np.array([0, 15360])
        np.testing.assert_almost_equal(output, expected)

    def test_conversion_reversible(self):
        """
        Test :func:`colour.io.luts.clf.from_uint16_to_f16` method with
        :func:`colour.io.luts.clf.from_f16_to_uint16`.

        """
        for i in range(2**16):
            value = np.array([i])
            float_value = from_uint16_to_f16(value)
            int_value = from_f16_to_uint16(float_value)
            np.testing.assert_almost_equal(value, int_value)


@pytest.mark.parametrize("bit_depth", BitDepth.all())
class TestLUT1DWithBitDepth:
    """
    Define test for applying 1D LUTSs from a CLF file. Tests are parametrised with
    `bit_depth` and the CLF will be executed with each bit depht value.
    """

    EXAMPLE_SIMPLE = """
        <LUT1D
            id="lut-23"
            name="4 Value Lut"
            inBitDepth="{bit_depth}"
            outBitDepth="{bit_depth}"
        >
            <Description>
                1D LUT - Turn 4 grey levels into 4 inverted codes
            </Description>
            <Array dim="4 1">
                3
                2
                1
                0
            </Array>
        </LUT1D>
    """

    def test_ocio_consistency_simple(self, bit_depth):
        """
        Test that the execution of a simple 1D LUT is consistent with `ociochecklut`.
        """
        value_rgb = np.array([1.0, 0.5, 0.0])
        assert_ocio_consistency(
            value_rgb, self.EXAMPLE_SIMPLE.format(bit_depth=bit_depth)
        )

    EXAMPLE_MULTI_TABLE = """
        <LUT1D
            id="lut-23"
            name="4 Value Lut"
            inBitDepth="{bit_depth}"
            outBitDepth="{bit_depth}"
        >
            <Array dim="4 1">
                3
                2
                1
                0
            </Array>
        </LUT1D>
        <LUT1D
            id="lut-23"
            name="4 Value Lut"
            inBitDepth="{bit_depth}"
            outBitDepth="{bit_depth}"
        >
            <Array dim="4 1">
                0
                1
                2
                3
            </Array>
        </LUT1D>
    """

    def test_ocio_consistency_multi_table(self, bit_depth):
        """
        Test that the execution of multiple 1D LUTs is consistent with `ociochecklut`.
        """
        value_rgb = np.array([1.0, 0.5, 0.0])
        assert_ocio_consistency(
            value_rgb, self.EXAMPLE_MULTI_TABLE.format(bit_depth=bit_depth)
        )

    EXAMPLE_RAW_HALFS = """
        <LUT1D rawHalfs="true" inBitDepth="{bit_depth}" outBitDepth="{bit_depth}">
            <Array dim="2 1">
                15360
                0
            </Array>
        </LUT1D>
    """

    def test_ocio_consistency_raw_halfs(self, bit_depth):
        """
        Test that the execution of a 1D LUT with the `raw_halfs` attribute is
        consistent with `ociochecklut`.
        """
        value_rgb = np.array([1.0, 0.5, 0.0])
        assert_ocio_consistency(
            value_rgb, self.EXAMPLE_RAW_HALFS.format(bit_depth=bit_depth)
        )


class TestLUT1D:
    """
    Define test for applying 1D LUTSs from a CLF file.
    """

    def test_ocio_consistency_half_domain(self):
        """
        Test that the execution of a 1D LUT with the `half_domain` attribute is
        consistent with `ociochecklut`.
        """
        value_rgb = np.array([1.0, 0.5, 0.0])
        path = os.path.abspath("./resources/clf/lut1_with_half_domain_sample.xml")
        assert_ocio_consistency_for_file(value_rgb, path)


if __name__ == "__main__":
    unittest.main()
