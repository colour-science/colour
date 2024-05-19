# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.colorimetry.illuminants` module."""

from __future__ import annotations

import numpy as np

from colour.colorimetry import (
    SDS_ILLUMINANTS,
    SpectralShape,
    daylight_locus_function,
    sd_CIE_illuminant_D_series,
    sd_CIE_standard_illuminant_A,
)
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.hints import NDArrayFloat
from colour.temperature import CCT_to_xy_CIE_D
from colour.utilities import ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "DATA_A",
    "TestSdCIEStandardIlluminantA",
    "TestSdCIEIlluminantDSeries",
    "TestDaylightLocusFunction",
]

DATA_A: NDArrayFloat = np.array(
    [
        6.14461778,
        6.94719899,
        7.82134941,
        8.76980228,
        9.79509961,
        10.89957616,
        12.08534536,
        13.35428726,
        14.70803845,
        16.14798414,
        17.67525215,
        19.29070890,
        20.99495729,
        22.78833636,
        24.67092269,
        26.64253337,
        28.70273044,
        30.85082676,
        33.08589297,
        35.40676571,
        37.81205669,
        40.30016269,
        42.86927625,
        45.51739693,
        48.24234315,
        51.04176432,
        53.91315329,
        56.85385894,
        59.86109896,
        62.93197247,
        66.06347275,
        69.25249966,
        72.49587199,
        75.79033948,
        79.13259455,
        82.51928367,
        85.94701837,
        89.41238582,
        92.91195891,
        96.44230599,
        100.00000000,
        103.58162718,
        107.18379528,
        110.80314124,
        114.43633837,
        118.08010305,
        121.73120094,
        125.38645263,
        129.04273891,
        132.69700551,
        136.34626740,
        139.98761262,
        143.61820577,
        147.23529096,
        150.83619449,
        154.41832708,
        157.97918574,
        161.51635535,
        165.02750987,
        168.51041325,
        171.96292009,
        175.38297597,
        178.76861756,
        182.11797252,
        185.42925911,
        188.70078570,
        191.93094995,
        195.11823798,
        198.26122323,
        201.35856531,
        204.40900861,
        207.41138086,
        210.36459156,
        213.26763031,
        216.11956508,
        218.91954039,
        221.66677545,
        224.36056217,
        227.00026327,
        229.58531022,
        232.11520118,
        234.58949900,
        237.00782911,
        239.36987744,
        241.67538835,
        243.92416253,
        246.11605493,
        248.25097273,
        250.32887325,
        252.34976196,
        254.31369047,
        256.22075454,
        258.07109218,
        259.86488167,
        261.60233977,
    ]
)


class TestSdCIEStandardIlluminantA:
    """
    Define :func:`colour.colorimetry.illuminants.\
sd_CIE_standard_illuminant_A` definition unit tests methods.
    """

    def test_sd_CIE_standard_illuminant_A(self):
        """
        Test :func:`colour.colorimetry.illuminants.\
sd_CIE_standard_illuminant_A` definition.
        """

        np.testing.assert_allclose(
            sd_CIE_standard_illuminant_A(SpectralShape(360, 830, 5)).values,
            DATA_A,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestSdCIEIlluminantDSeries:
    """
    Define :func:`colour.colorimetry.illuminants.sd_CIE_illuminant_D_series`
    definition unit tests methods.
    """

    def test_sd_CIE_illuminant_D_series(self):
        """
        Test :func:`colour.colorimetry.illuminants.\
sd_CIE_illuminant_D_series` definition.
        """

        for name, CCT, tolerance in (
            ("D50", 5000, 0.001),
            ("D55", 5500, 0.001),
            ("D65", 6500, 0.001),
            ("D75", 7500, 0.001),
        ):
            CCT *= 1.4388 / 1.4380  # noqa: PLW2901
            xy = CCT_to_xy_CIE_D(CCT)
            sd_r = SDS_ILLUMINANTS[name]
            sd_t = sd_CIE_illuminant_D_series(xy)

            np.testing.assert_allclose(
                sd_r.values,
                sd_t[sd_r.wavelengths],
                atol=tolerance,
            )


class TestDaylightLocusFunction:
    """
    Define :func:`colour.colorimetry.illuminants.daylight_locus_function`
    definition unit tests methods.
    """

    def test_daylight_locus_function(self):
        """
        Test :func:`colour.colorimetry.illuminants.daylight_locus_function`
        definition.
        """

        np.testing.assert_allclose(
            daylight_locus_function(0.31270),
            0.329105129999999,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            daylight_locus_function(0.34570),
            0.358633529999999,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            daylight_locus_function(0.44758),
            0.408571030799999,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_daylight_locus_function(self):
        """
        Test :func:`colour.colorimetry.illuminants.daylight_locus_function`
        definition n-dimensional support.
        """

        x_D = np.array([0.31270])
        y_D = daylight_locus_function(x_D)

        x_D = np.tile(x_D, (6, 1))
        y_D = np.tile(y_D, (6, 1))
        np.testing.assert_allclose(
            daylight_locus_function(x_D), y_D, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        x_D = np.reshape(x_D, (2, 3, 1))
        y_D = np.reshape(y_D, (2, 3, 1))
        np.testing.assert_allclose(
            daylight_locus_function(x_D), y_D, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    @ignore_numpy_errors
    def test_nan_daylight_locus_function(self):
        """
        Test :func:`colour.colorimetry.illuminants.daylight_locus_function`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        daylight_locus_function(cases)
