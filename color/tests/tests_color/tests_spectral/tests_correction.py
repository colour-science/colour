#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**tests_correction.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines units tests for :mod:`color.spectral.correction` module.

**Others:**

"""

from __future__ import unicode_literals

import numpy
import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

import color.spectral.correction
from color.spectral.spd import SpectralPowerDistribution

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["SPD_DATA",
           "BANDPASS_CORRECTED_STEARNS_SPD_DATA",
           "TestBandpassCorrectionStearns"]

SPD_DATA = numpy.array([9.3700,
                        12.3200,
                        12.4600,
                        9.5100,
                        5.9200,
                        4.3300,
                        4.2900,
                        3.8800,
                        4.5100,
                        10.9200,
                        27.5000,
                        49.6700,
                        69.5900,
                        81.7300,
                        88.1900,
                        86.0500])

BANDPASS_CORRECTED_STEARNS_SPD_DATA = numpy.array([8.83893259,
                                                   12.87341058,
                                                   12.94225552,
                                                   9.57625607,
                                                   5.56121014,
                                                   4.09411985,
                                                   4.39463228,
                                                   3.71965425,
                                                   3.55967631,
                                                   9.24920718,
                                                   26.69337799,
                                                   50.12780392,
                                                   70.89460494,
                                                   82.5245482,
                                                   89.68223615,
                                                   85.62243747])


class TestBandpassCorrectionStearns(unittest.TestCase):
    """
    Defines :func:`color.spectral.correction.bandpass_correction_stearns` definition units tests methods.
    """

    def test_bandpass_correction_stearns(self):
        """
        Tests :func:`color.spectral.correction.bandpass_correction_stearns` definition.
        """

        spd = SpectralPowerDistribution("Spd", dict(zip(range(len(SPD_DATA)), SPD_DATA)))
        self.assertTrue(spd is color.spectral.correction.bandpass_correction_stearns(spd, in_place=True))
        self.assertFalse(spd is color.spectral.correction.bandpass_correction_stearns(spd, in_place=False))
        numpy.testing.assert_almost_equal(BANDPASS_CORRECTED_STEARNS_SPD_DATA, color.spectral.correction.bandpass_correction_stearns(spd).values)


if __name__ == "__main__":
    unittest.main()
