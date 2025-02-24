# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.io.clf` module."""

import unittest

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

import numpy as np

from colour.io.luts.tests.test_clf_common import (
    assert_ocio_consistency,
    rgb_sample_iter,
)


class TestRange:
    """
    Define test for applying Range nodes from a CLF file.
    """

    def test_ocio_consistency_simple(self) -> None:
        """
        Test that the execution is consistent with the OCIO reference.
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
        for rgb in rgb_sample_iter():
            value_rgb = np.array(rgb)
            assert_ocio_consistency(value_rgb, example, f"Input value was {rgb}")

    def test_ocio_consistency_no_clamp(self) -> None:
        """
        Test that the execution is consistent with the OCIO reference.
        """

        example = """
        <Range inBitDepth="10i" outBitDepth="10i" style="noClamp">
            <minInValue>0</minInValue>
            <maxInValue>1023</maxInValue>
            <minOutValue>64</minOutValue>
            <maxOutValue>940</maxOutValue>
        </Range>
        """
        for rgb in rgb_sample_iter():
            value_rgb = np.array(rgb)
            assert_ocio_consistency(value_rgb, example, f"Input value was {rgb}")

    def test_ocio_consistency_only_max_values(self) -> None:
        """
        Test that the execution is consistent with the OCIO reference.
        """

        example = """
        <Range inBitDepth="8i" outBitDepth="12i">
            <maxInValue>940</maxInValue>
            <maxOutValue>15095.29411764706</maxOutValue>
        </Range>
        """
        for rgb in rgb_sample_iter():
            value_rgb = np.array(rgb)
            assert_ocio_consistency(value_rgb, example, f"Input value was {rgb}")

    def test_ocio_consistency_only_min_values(self) -> None:
        """
        Test that the execution is consistent with the OCIO reference.
        """

        example = """
        <Range inBitDepth="12i" outBitDepth="10i">
            <minInValue>64</minInValue>
            <minOutValue>15.988278388278388</minOutValue>
        </Range>
        """
        for rgb in rgb_sample_iter():
            value_rgb = np.array(rgb)
            assert_ocio_consistency(value_rgb, example, f"Input value was {rgb}")


if __name__ == "__main__":
    unittest.main()
