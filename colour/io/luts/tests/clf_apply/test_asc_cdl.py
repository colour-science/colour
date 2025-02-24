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


def assert_snippet_consistency(snippet: str) -> None:
    """
    Evaluate the snippet with multiple values anc check that they are the same as the
    `ociochecklut` tools output.
    """
    for rgb in rgb_sample_iter():
        value_rgb = np.array(rgb)
        assert_ocio_consistency(
            value_rgb, snippet, f"Failed to assert consistency for {rgb}"
        )


class TestASC_CDL:
    """
    Define test for applying Exponent nodes from a CLF file.
    """

    def test_ocio_consistency_fwd(self) -> None:
        """
        Test that the execution is consistent with the OCIO reference.
        """

        example = """
        <ASC_CDL id="cc01234" inBitDepth="16f" outBitDepth="16f" style="Fwd">
            <SOPNode>
                <Slope>1.000000 1.000000 0.900000</Slope>
                <Offset>-0.030000 -0.020000 0.000000</Offset>
                <Power>1.2500000 1.000000 1.000000</Power>
            </SOPNode>
            <SatNode>
                <Saturation>1.700000</Saturation>
            </SatNode>
        </ASC_CDL>
        """
        assert_snippet_consistency(example)

    def test_ocio_consistency_rev(self) -> None:
        """
        Test that the execution is consistent with the OCIO reference.
        """

        example = """
        <ASC_CDL id="cc01234" inBitDepth="16f" outBitDepth="16f" style="Rev">
            <SOPNode>
                <Slope>1.000000 1.000000 0.900000</Slope>
                <Offset>-0.030000 -0.020000 0.000000</Offset>
                <Power>1.2500000 1.000000 1.000000</Power>
            </SOPNode>
            <SatNode>
                <Saturation>1.700000</Saturation>
            </SatNode>
        </ASC_CDL>
        """
        assert_snippet_consistency(example)

    def test_ocio_consistency_fwd_no_clamp(self) -> None:
        """
        Test that the execution is consistent with the OCIO reference.
        """

        example = """
        <ASC_CDL id="cc01234" inBitDepth="16f" outBitDepth="16f" style="FwdNoClamp">
            <SOPNode>
                <Slope>1.000000 1.000000 0.900000</Slope>
                <Offset>-0.030000 -0.020000 0.000000</Offset>
                <Power>1.2500000 1.000000 1.000000</Power>
            </SOPNode>
            <SatNode>
                <Saturation>1.700000</Saturation>
            </SatNode>
        </ASC_CDL>
        """
        assert_snippet_consistency(example)

    def test_ocio_consistency_rev_no_clamp(self) -> None:
        """
        Test that the execution is consistent with the OCIO reference.
        """

        example = """
        <ASC_CDL id="cc01234" inBitDepth="16f" outBitDepth="16f" style="RevNoClamp">
            <SOPNode>
                <Slope>1.000000 1.000000 0.900000</Slope>
                <Offset>-0.030000 -0.020000 0.000000</Offset>
                <Power>1.2500000 1.000000 1.000000</Power>
            </SOPNode>
            <SatNode>
                <Saturation>1.700000</Saturation>
            </SatNode>
        </ASC_CDL>
        """
        assert_snippet_consistency(example)

    def test_ocio_consistency_default_args(self) -> None:
        """
        Test that the execution is consistent with the OCIO reference.
        """

        example = """
        <ASC_CDL id="cc01234" inBitDepth="16f" outBitDepth="16f" style="Fwd">
        </ASC_CDL>
        """
        assert_snippet_consistency(example)


if __name__ == "__main__":
    unittest.main()
