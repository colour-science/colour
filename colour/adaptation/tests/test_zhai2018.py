# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.adaptation.zhai2018` module."""

import unittest
from itertools import product

import numpy as np

from colour.adaptation import chromatic_adaptation_Zhai2018
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestChromaticAdaptationZhai2018",
]


class TestChromaticAdaptationZhai2018(unittest.TestCase):
    """
    Define :func:`colour.adaptation.zhai2018.chromatic_adaptation_Zhai2018`
    definition unit tests methods.
    """

    def test_chromatic_adaptation_Zhai2018(self):
        """
        Test :func:`colour.adaptation.zhai2018.chromatic_adaptation_Zhai2018`
        definition.
        """

        np.testing.assert_allclose(
            chromatic_adaptation_Zhai2018(
                XYZ_b=np.array([48.900, 43.620, 6.250]),
                XYZ_wb=np.array([109.850, 100, 35.585]),
                XYZ_wd=np.array([95.047, 100, 108.883]),
                D_b=0.9407,
                D_d=0.9800,
                XYZ_wo=np.array([100, 100, 100]),
            ),
            np.array([39.18561644, 42.15461798, 19.23672036]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            chromatic_adaptation_Zhai2018(
                XYZ_b=np.array([48.900, 43.620, 6.250]),
                XYZ_wb=np.array([109.850, 100, 35.585]),
                XYZ_wd=np.array([95.047, 100, 108.883]),
                D_b=0.9407,
                D_d=0.9800,
                XYZ_wo=np.array([100, 100, 100]),
                transform="CAT16",
            ),
            np.array([40.37398343, 43.69426311, 20.51733764]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            chromatic_adaptation_Zhai2018(
                XYZ_b=np.array([52.034, 58.824, 23.703]),
                XYZ_wb=np.array([92.288, 100, 38.775]),
                XYZ_wd=np.array([105.432, 100, 137.392]),
                D_b=0.6709,
                D_d=0.5331,
                XYZ_wo=np.array([97.079, 100, 141.798]),
            ),
            np.array([57.03242915, 58.93434364, 64.76261333]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            chromatic_adaptation_Zhai2018(
                XYZ_b=np.array([52.034, 58.824, 23.703]),
                XYZ_wb=np.array([92.288, 100, 38.775]),
                XYZ_wd=np.array([105.432, 100, 137.392]),
                D_b=0.6709,
                D_d=0.5331,
                XYZ_wo=np.array([97.079, 100, 141.798]),
                transform="CAT16",
            ),
            np.array([56.77130011, 58.81317888, 64.66922808]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            chromatic_adaptation_Zhai2018(
                XYZ_b=np.array([48.900, 43.620, 6.250]),
                XYZ_wb=np.array([109.850, 100, 35.585]),
                XYZ_wd=np.array([95.047, 100, 108.883]),
            ),
            np.array([38.72444735, 42.09232891, 20.05297620]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_chromatic_adaptation_Zhai2018(self):
        """
        Test :func:`colour.adaptation.zhai2018.chromatic_adaptation_Zhai2018`
        definition n-dimensional arrays support.
        """

        XYZ_b = np.array([48.900, 43.620, 6.250])
        XYZ_wb = np.array([109.850, 100, 35.585])
        XYZ_wd = np.array([95.047, 100, 108.883])
        D_b = 0.9407
        D_d = 0.9800
        XYZ_d = chromatic_adaptation_Zhai2018(XYZ_b, XYZ_wb, XYZ_wd, D_b, D_d)

        XYZ_b = np.tile(XYZ_b, (6, 1))
        XYZ_d = np.tile(XYZ_d, (6, 1))
        np.testing.assert_allclose(
            chromatic_adaptation_Zhai2018(XYZ_b, XYZ_wb, XYZ_wd, D_b, D_d),
            XYZ_d,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ_wb = np.tile(XYZ_wb, (6, 1))
        XYZ_wd = np.tile(XYZ_wd, (6, 1))
        D_b = np.tile(D_b, (6, 1))
        D_d = np.tile(D_d, (6, 1))
        np.testing.assert_allclose(
            chromatic_adaptation_Zhai2018(XYZ_b, XYZ_wb, XYZ_wd, D_b, D_d),
            XYZ_d,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ_b = np.reshape(XYZ_b, (2, 3, 3))
        XYZ_wb = np.reshape(XYZ_wb, (2, 3, 3))
        XYZ_wd = np.reshape(XYZ_wd, (2, 3, 3))
        D_b = np.reshape(D_b, (2, 3, 1))
        D_d = np.reshape(D_d, (2, 3, 1))
        XYZ_d = np.reshape(XYZ_d, (2, 3, 3))
        np.testing.assert_allclose(
            chromatic_adaptation_Zhai2018(XYZ_b, XYZ_wb, XYZ_wd, D_b, D_d),
            XYZ_d,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_chromatic_adaptation_Zhai2018(self):
        """
        Test :func:`colour.adaptation.zhai2018.chromatic_adaptation_Zhai2018`
        definition domain and range scale support.
        """

        XYZ_b = np.array([48.900, 43.620, 6.250])
        XYZ_wb = np.array([109.850, 100, 35.585])
        XYZ_wd = np.array([95.047, 100, 108.883])
        XYZ_d = chromatic_adaptation_Zhai2018(XYZ_b, XYZ_wb, XYZ_wd)

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    chromatic_adaptation_Zhai2018(
                        XYZ_b * factor, XYZ_wb * factor, XYZ_wd * factor
                    ),
                    XYZ_d * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_chromatic_adaptation_Zhai2018(self):
        """
        Test :func:`colour.adaptation.zhai2018.chromatic_adaptation_Zhai2018`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        chromatic_adaptation_Zhai2018(
            cases, cases, cases, cases[0, 0], cases[0, 0], cases
        )


if __name__ == "__main__":
    unittest.main()
