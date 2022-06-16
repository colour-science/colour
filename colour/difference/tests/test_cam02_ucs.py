# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.difference.cam02_ucs` module."""

import numpy as np
import unittest
from itertools import product

from colour.difference import (
    delta_E_CAM02LCD,
    delta_E_CAM02SCD,
    delta_E_CAM02UCS,
)
from colour.difference.cam02_ucs import delta_E_Luo2006
from colour.models.cam02_ucs import COEFFICIENTS_UCS_LUO2006
from colour.utilities import ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestDelta_E_Luo2006",
]


class TestDelta_E_Luo2006(unittest.TestCase):
    """
    Define :func:`colour.difference.cam02_ucs.delta_E_Luo2006` definition unit
    tests methods.
    """

    def test_delta_E_Luo2006(self):
        """Test :func:`colour.difference.cam02_ucs.delta_E_Luo2006` definition."""

        self.assertAlmostEqual(
            delta_E_Luo2006(
                np.array([54.90433134, -0.08450395, -0.06854831]),
                np.array([54.80352754, -3.96940084, -13.57591013]),
                COEFFICIENTS_UCS_LUO2006["CAM02-LCD"],
            ),
            14.055546437777583,
            places=7,
        )

        self.assertAlmostEqual(
            delta_E_Luo2006(
                np.array([54.90433134, -0.08450395, -0.06854831]),
                np.array([54.80352754, -3.96940084, -13.57591013]),
                COEFFICIENTS_UCS_LUO2006["CAM02-LCD"],
            ),
            delta_E_CAM02LCD(
                np.array([54.90433134, -0.08450395, -0.06854831]),
                np.array([54.80352754, -3.96940084, -13.57591013]),
            ),
            places=7,
        )

        self.assertAlmostEqual(
            delta_E_Luo2006(
                np.array([54.90433134, -0.08450395, -0.06854831]),
                np.array([54.80352754, -3.96940084, -13.57591013]),
                COEFFICIENTS_UCS_LUO2006["CAM02-SCD"],
            ),
            delta_E_CAM02SCD(
                np.array([54.90433134, -0.08450395, -0.06854831]),
                np.array([54.80352754, -3.96940084, -13.57591013]),
            ),
            places=7,
        )

        self.assertAlmostEqual(
            delta_E_Luo2006(
                np.array([54.90433134, -0.08450395, -0.06854831]),
                np.array([54.80352754, -3.96940084, -13.57591013]),
                COEFFICIENTS_UCS_LUO2006["CAM02-UCS"],
            ),
            delta_E_CAM02UCS(
                np.array([54.90433134, -0.08450395, -0.06854831]),
                np.array([54.80352754, -3.96940084, -13.57591013]),
            ),
            places=7,
        )

    def test_n_dimensional_delta_E_Luo2006(self):
        """
        Test :func:`colour.difference.cam02_ucs.delta_E_Luo2006` definition
        n-dimensional arrays support.
        """

        Jpapbp_1 = np.array([54.90433134, -0.08450395, -0.06854831])
        Jpapbp_2 = np.array([54.80352754, -3.96940084, -13.57591013])
        delta_E_p = delta_E_Luo2006(
            Jpapbp_1, Jpapbp_2, COEFFICIENTS_UCS_LUO2006["CAM02-LCD"]
        )

        Jpapbp_1 = np.tile(Jpapbp_1, (6, 1))
        Jpapbp_2 = np.tile(Jpapbp_2, (6, 1))
        delta_E_p = np.tile(delta_E_p, 6)
        np.testing.assert_almost_equal(
            delta_E_Luo2006(
                Jpapbp_1, Jpapbp_2, COEFFICIENTS_UCS_LUO2006["CAM02-LCD"]
            ),
            delta_E_p,
            decimal=7,
        )

        Jpapbp_1 = np.reshape(Jpapbp_1, (2, 3, 3))
        Jpapbp_2 = np.reshape(Jpapbp_2, (2, 3, 3))
        delta_E_p = np.reshape(delta_E_p, (2, 3))
        np.testing.assert_almost_equal(
            delta_E_Luo2006(
                Jpapbp_1, Jpapbp_2, COEFFICIENTS_UCS_LUO2006["CAM02-LCD"]
            ),
            delta_E_p,
            decimal=7,
        )

    @ignore_numpy_errors
    def test_nan_delta_E_Luo2006(self):
        """
        Test :func:`colour.difference.cam02_ucs.delta_E_Luo2006`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        delta_E_Luo2006(cases, cases, COEFFICIENTS_UCS_LUO2006["CAM02-LCD"])


if __name__ == "__main__":
    unittest.main()
