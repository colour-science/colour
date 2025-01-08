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


# TODO assert correct ranges for exponents and offsets


class TestExponent:
    """
    Define test for applying Exponent nodes from a CLF file.
    """

    def test_ocio_consistency_fwd(self) -> None:
        """
        Test that the execution is consistent with `ociochecklut`.
        """

        example = """
        <Exponent inBitDepth="32f" outBitDepth="32f" style="basicFwd">
            <ExponentParams exponent="2.2"/>
        </Exponent>
        """
        assert_snippet_consistency(example)

    def test_ocio_consistency_rev(self) -> None:
        """
        Test that the execution is consistent with `ociochecklut`.
        """

        example = """
        <Exponent inBitDepth="32f" outBitDepth="32f" style="basicRev">
            <ExponentParams exponent="2.2"/>
        </Exponent>
        """
        assert_snippet_consistency(example)

    def test_ocio_consistency_mirror_fwd(self) -> None:
        """
        Test that the execution is consistent with `ociochecklut`.
        """

        example = """
        <Exponent inBitDepth="32f" outBitDepth="32f" style="basicMirrorFwd">
            <ExponentParams exponent="2.2"/>
        </Exponent>
        """
        assert_snippet_consistency(example)

    def test_ocio_consistency_basic_mirror_rev(self) -> None:
        """
        Test that the execution is consistent with `ociochecklut`.
        """

        example = """
        <Exponent inBitDepth="32f" outBitDepth="32f" style="basicMirrorRev">
            <ExponentParams exponent="2.2"/>
        </Exponent>
        """
        assert_snippet_consistency(example)

    def test_ocio_consistency_basic_pass_thru_fwd(self) -> None:
        """
        Test that the execution is consistent with `ociochecklut`.
        """

        example = """
        <Exponent inBitDepth="32f" outBitDepth="32f" style="basicPassThruFwd">
            <ExponentParams exponent="2.2"/>
        </Exponent>
        """
        assert_snippet_consistency(example)

    def test_ocio_consistency_basic_pass_thru_rev(self) -> None:
        """
        Test that the execution is consistent with `ociochecklut`.
        """

        example = """
        <Exponent inBitDepth="32f" outBitDepth="32f" style="basicPassThruRev">
            <ExponentParams exponent="2.2"/>
        </Exponent>
        """
        assert_snippet_consistency(example)

    def test_ocio_consistency_mon_curve_fwd(self) -> None:
        """
        Test that the execution is consistent with `ociochecklut`.
        """

        example = """
        <Exponent inBitDepth="32f" outBitDepth="32f" style="monCurveFwd">
            <ExponentParams exponent="3.0" offset="0.16" />
        </Exponent>
        """
        assert_snippet_consistency(example)

    def test_ocio_consistency_mon_curve_rev(self) -> None:
        """
        Test that the execution is consistent with `ociochecklut`.
        """

        example = """
        <Exponent inBitDepth="32f" outBitDepth="32f" style="monCurveRev">
            <ExponentParams exponent="3.0" offset="0.16" />
        </Exponent>
        """
        assert_snippet_consistency(example)

    def test_ocio_consistency_mon_curve_rev_2(self) -> None:
        """
        Test that the execution is consistent with `ociochecklut`.
        """

        example = """
        <Exponent inBitDepth="32f" outBitDepth="32f" style="monCurveRev">
            <ExponentParams exponent="2.2222222222222222" offset="0.099" />
        </Exponent>
        """
        assert_snippet_consistency(example)

    def test_ocio_consistency_mon_curve_mirror_fwd(self) -> None:
        """
        Test that the execution is consistent with `ociochecklut`.
        """

        example = """
        <Exponent inBitDepth="32f" outBitDepth="32f" style="monCurveMirrorFwd">
            <ExponentParams exponent="3.0" offset="0.16" />
        </Exponent>
        """
        assert_snippet_consistency(example)

    def test_ocio_consistency_mon_curve_mirror_rev(self) -> None:
        """
        Test that the execution is consistent with `ociochecklut`.
        """

        example = """
        <Exponent inBitDepth="32f" outBitDepth="32f" style="monCurveMirrorRev">
            <ExponentParams exponent="3.0" offset="0.16" />
        </Exponent>
        """
        assert_snippet_consistency(example)

    def test_ocio_consistency_single_channel_application(self) -> None:
        """
        Test that the execution is consistent with `ociochecklut`.
        """

        example = """
        <Exponent inBitDepth="32f" outBitDepth="32f" style="monCurveMirrorRev">
            <ExponentParams exponent="3.0" offset="0.16" channel="G" />
        </Exponent>
        """
        assert_snippet_consistency(example)

    def test_ocio_consistency_multi_channel_application(self) -> None:
        """
        Test that the execution is consistent with `ociochecklut`.
        """

        example = """
        <Exponent inBitDepth="32f" outBitDepth="32f" style="monCurveMirrorRev">
            <ExponentParams exponent="3.0" offset="0.16" channel="G" />
            <ExponentParams exponent="4.0" offset="0.21" channel="B" />
        </Exponent>
        """
        assert_snippet_consistency(example)


if __name__ == "__main__":
    unittest.main()
