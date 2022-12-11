# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.io.xrite` module."""

from __future__ import annotations

import os
import unittest

from colour.colorimetry import SpectralDistribution
from colour.io import read_sds_from_xrite_file

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "ROOT_RESOURCES",
    "COLOURCHECKER_XRITE_1",
    "TestReadSdsFromXRiteFile",
]

ROOT_RESOURCES: str = os.path.join(os.path.dirname(__file__), "resources")

COLOURCHECKER_XRITE_1: dict = {
    380.0: 0.0069,
    390.0: 0.0069,
    400.0: 0.0068,
    410.0: 0.0068,
    420.0: 0.0073,
    430.0: 0.0075,
    440.0: 0.0065,
    450.0: 0.0074,
    460.0: 0.0073,
    470.0: 0.0073,
    480.0: 0.0074,
    490.0: 0.0074,
    500.0: 0.0075,
    510.0: 0.0075,
    520.0: 0.0072,
    530.0: 0.0072,
    540.0: 0.0072,
    550.0: 0.0072,
    560.0: 0.0072,
    570.0: 0.0071,
    580.0: 0.0071,
    590.0: 0.0071,
    600.0: 0.0071,
    610.0: 0.0072,
    620.0: 0.0071,
    630.0: 0.0071,
    640.0: 0.0071,
    650.0: 0.0070,
    660.0: 0.0074,
    670.0: 0.0068,
    680.0: 0.0067,
    690.0: 0.0067,
    700.0: 0.0066,
    710.0: 0.0066,
    720.0: 0.0066,
    730.0: 0.0065,
}


class TestReadSdsFromXRiteFile(unittest.TestCase):
    """
    Define :func:`colour.io.xrite.read_sds_from_xrite_file` definition unit
    tests methods.
    """

    def test_read_sds_from_xrite_file(self):
        """Test :func:`colour.io.xrite.read_sds_from_xrite_file` definition."""

        colour_checker_xrite = os.path.join(
            ROOT_RESOURCES, "X-Rite_Digital_Colour_Checker.txt"
        )
        sds = read_sds_from_xrite_file(colour_checker_xrite)
        for sd in sds.values():
            self.assertIsInstance(sd, SpectralDistribution)

        self.assertEqual(
            sds["X1"], SpectralDistribution(COLOURCHECKER_XRITE_1, name="X1")
        )


if __name__ == "__main__":
    unittest.main()
