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


def assert_snippet_consistency(snippet):
    """
    Evaluate the snippet with multiple values anc check that they are the same as the
    `ociochecklut` tools output.
    """
    for rgb in rgb_sample_iter():
        value_rgb = np.array(rgb)
        assert_ocio_consistency(
            value_rgb, snippet, f"Failed to assert consistency for {rgb}"
        )


class TestLog:
    """
    Define test for applying Log nodes from a CLF file.
    """

    def test_ocio_consistency_log_10(self):
        """
        Test that the execution is consistent with `ociochecklut`.
        """

        example = """
        <Log inBitDepth="16f" outBitDepth="16f" style="log10">
        </Log>
        """
        assert_snippet_consistency(example)

    def test_ocio_consistency_anti_log_10(self):
        """
        Test that the execution is consistent with `ociochecklut`.
        """

        example = """
        <Log inBitDepth="16f" outBitDepth="16f" style="antiLog10">
        </Log>
        """
        assert_snippet_consistency(example)

    def test_ocio_consistency_log_2(self):
        """
        Test that the execution is consistent with `ociochecklut`.
        """

        example = """
        <Log inBitDepth="16f" outBitDepth="16f" style="log2">
        </Log>
        """
        assert_snippet_consistency(example)

    def test_ocio_consistency_anti_log_2(self):
        """
        Test that the execution is consistent with `ociochecklut`.
        """

        example = """
        <Log inBitDepth="16f" outBitDepth="16f" style="antiLog2">
        </Log>
        """
        assert_snippet_consistency(example)

    def test_ocio_consistency_lin_to_log(self):
        """
        Test that the execution is consistent with `ociochecklut`.
        """

        example = """
        <Log inBitDepth="32f" outBitDepth="32f" style="linToLog">
            <LogParams base="10" logSideSlope="0.256663" logSideOffset="0.584555"
                linSideSlope="0.9892" linSideOffset="0.0108"
            />
        </Log>
        """
        assert_snippet_consistency(example)

    def test_ocio_consistency_log_to_lin(self):
        """
        Test that the execution is consistent with `ociochecklut`.
        """

        example = """
        <Log inBitDepth="32f" outBitDepth="32f" style="logToLin">
            <LogParams base="10" logSideSlope="0.256663" logSideOffset="0.584555"
                linSideSlope="0.9892" linSideOffset="0.0108"
            />
        </Log>
        """
        assert_snippet_consistency(example)

    def test_ocio_consistency_camera_lin_to_log(self):
        """
        Test that the execution is consistent with `ociochecklut`.
        """

        example = """
        <Log inBitDepth="32f" outBitDepth="32f" style="cameraLinToLog">
            <Description>Linear to DJI D-Log</Description>
            <LogParams base="10" logSideSlope="0.256663" logSideOffset="0.584555"
                linSideSlope="0.9892" linSideOffset="0.0108" linSideBreak="0.0078"
                linearSlope="6.025"/>
        </Log>
        """
        assert_snippet_consistency(example)

    def test_ocio_consistency_camera_log_to_lin(self):
        """
        Test that the execution is consistent with `ociochecklut`.
        """

        example = """
        <Log inBitDepth="32f" outBitDepth="32f" style="cameraLinToLog">
            <Description>Linear to DJI D-Log</Description>
            <LogParams base="10" logSideSlope="0.256663" logSideOffset="0.584555"
                linSideSlope="0.9892" linSideOffset="0.0108" linSideBreak="0.0078"
                linearSlope="6.025"/>
        </Log>
        """
        assert_snippet_consistency(example)


if __name__ == "__main__":
    unittest.main()
