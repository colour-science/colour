"""Defines the unit tests for the :mod:`colour.models.rgb.common` module."""

import numpy as np
import unittest

from colour.models import XYZ_to_sRGB, sRGB_to_XYZ

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestXYZ_to_sRGB",
    "TestsRGB_to_XYZ",
]


class TestXYZ_to_sRGB(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.common.XYZ_to_sRGB` definition unit tests
    methods.
    """

    def test_XYZ_to_sRGB(self):
        """Test :func:`colour.models.rgb.common.XYZ_to_sRGB` definition."""

        np.testing.assert_almost_equal(
            XYZ_to_sRGB(np.array([0.20654008, 0.12197225, 0.05136952])),
            np.array([0.70573936, 0.19248266, 0.22354169]),
            decimal=7,
        )

        np.testing.assert_almost_equal(
            XYZ_to_sRGB(np.array([0.14222010, 0.23042768, 0.10495772])),
            np.array([0.25847003, 0.58276102, 0.29718877]),
            decimal=7,
        )

        np.testing.assert_almost_equal(
            XYZ_to_sRGB(
                np.array([0.07818780, 0.06157201, 0.28099326]),
                np.array([0.34570, 0.35850]),
            ),
            np.array([0.09838967, 0.25404426, 0.65130925]),
            decimal=7,
        )

        np.testing.assert_almost_equal(
            XYZ_to_sRGB(
                np.array([0.00000000, 0.00000000, 0.00000000]),
                np.array([0.44757, 0.40745]),
            ),
            np.array([0.00000000, 0.00000000, 0.00000000]),
            decimal=7,
        )

        np.testing.assert_almost_equal(
            XYZ_to_sRGB(
                np.array([0.20654008, 0.12197225, 0.05136952]),
                np.array([0.44757, 0.40745]),
                chromatic_adaptation_transform="Bradford",
            ),
            np.array([0.60873814, 0.23259548, 0.43714892]),
            decimal=7,
        )

        np.testing.assert_almost_equal(
            XYZ_to_sRGB(
                np.array([0.20654008, 0.12197225, 0.05136952]),
                apply_cctf_encoding=False,
            ),
            np.array([0.45620520, 0.03081070, 0.04091953]),
            decimal=7,
        )


class TestsRGB_to_XYZ(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.common.sRGB_to_XYZ` definition unit tests
    methods.
    """

    def test_sRGB_to_XYZ(self):
        """Test :func:`colour.models.rgb.common.sRGB_to_XYZ` definition."""

        np.testing.assert_almost_equal(
            sRGB_to_XYZ(np.array([0.70573936, 0.19248266, 0.22354169])),
            np.array([0.20654290, 0.12197943, 0.05137140]),
            decimal=7,
        )

        np.testing.assert_almost_equal(
            sRGB_to_XYZ(np.array([0.25847003, 0.58276102, 0.29718877])),
            np.array([0.14222582, 0.23043727, 0.10496290]),
            decimal=7,
        )

        np.testing.assert_almost_equal(
            sRGB_to_XYZ(
                np.array([0.09838967, 0.25404426, 0.65130925]),
                np.array([0.34570, 0.35850]),
            ),
            np.array([0.07819162, 0.06157356, 0.28099475]),
            decimal=7,
        )

        np.testing.assert_almost_equal(
            sRGB_to_XYZ(
                np.array([0.00000000, 0.00000000, 0.00000000]),
                np.array([0.44757, 0.40745]),
            ),
            np.array([0.00000000, 0.00000000, 0.00000000]),
            decimal=7,
        )

        np.testing.assert_almost_equal(
            sRGB_to_XYZ(
                np.array([0.60873814, 0.23259548, 0.43714892]),
                np.array([0.44757, 0.40745]),
                chromatic_adaptation_transform="Bradford",
            ),
            np.array([0.20654449, 0.12197792, 0.05137030]),
            decimal=7,
        )

        np.testing.assert_almost_equal(
            sRGB_to_XYZ(
                np.array([0.45620520, 0.03081070, 0.04091953]),
                apply_cctf_decoding=False,
            ),
            np.array([0.20654291, 0.12197943, 0.05137141]),
            decimal=7,
        )


if __name__ == "__main__":
    unittest.main()
