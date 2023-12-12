# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.adaptation.cie1994` module."""

import unittest
from itertools import product

import numpy as np

from colour.adaptation import chromatic_adaptation_CIE1994
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestChromaticAdaptationCIE1994",
]


class TestChromaticAdaptationCIE1994(unittest.TestCase):
    """
    Define :func:`colour.adaptation.cie1994.chromatic_adaptation_CIE1994`
    definition unit tests methods.
    """

    def test_chromatic_adaptation_CIE1994(self):
        """
        Test :func:`colour.adaptation.cie1994.chromatic_adaptation_CIE1994`
        definition.
        """

        np.testing.assert_allclose(
            chromatic_adaptation_CIE1994(
                XYZ_1=np.array([28.00, 21.26, 5.27]),
                xy_o1=np.array([0.44760, 0.40740]),
                xy_o2=np.array([0.31270, 0.32900]),
                Y_o=20,
                E_o1=1000,
                E_o2=1000,
            ),
            np.array([24.03379521, 21.15621214, 17.64301199]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            chromatic_adaptation_CIE1994(
                XYZ_1=np.array([21.77, 19.18, 16.73]),
                xy_o1=np.array([0.31270, 0.32900]),
                xy_o2=np.array([0.31270, 0.32900]),
                Y_o=50,
                E_o1=100,
                E_o2=1000,
            ),
            np.array([21.12891746, 19.42980532, 19.49577765]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            chromatic_adaptation_CIE1994(
                XYZ_1=np.array([0.07818780, 0.06157201, 0.28099326]) * 100,
                xy_o1=np.array([0.31270, 0.32900]),
                xy_o2=np.array([0.37208, 0.37529]),
                Y_o=20,
                E_o1=100,
                E_o2=1000,
            ),
            np.array([9.14287406, 9.35843355, 15.95753504]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_chromatic_adaptation_CIE1994(self):
        """
        Test :func:`colour.adaptation.cie1994.chromatic_adaptation_CIE1994`
        definition n-dimensional arrays support.
        """

        XYZ_1 = np.array([28.00, 21.26, 5.27])
        xy_o1 = np.array([0.44760, 0.40740])
        xy_o2 = np.array([0.31270, 0.32900])
        Y_o = 20
        E_o1 = 1000
        E_o2 = 1000
        XYZ_2 = chromatic_adaptation_CIE1994(
            XYZ_1, xy_o1, xy_o2, Y_o, E_o1, E_o2
        )

        XYZ_1 = np.tile(XYZ_1, (6, 1))
        XYZ_2 = np.tile(XYZ_2, (6, 1))
        np.testing.assert_allclose(
            chromatic_adaptation_CIE1994(XYZ_1, xy_o1, xy_o2, Y_o, E_o1, E_o2),
            XYZ_2,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xy_o1 = np.tile(xy_o1, (6, 1))
        xy_o2 = np.tile(xy_o2, (6, 1))
        Y_o = np.tile(Y_o, 6)
        E_o1 = np.tile(E_o1, 6)
        E_o2 = np.tile(E_o2, 6)
        np.testing.assert_allclose(
            chromatic_adaptation_CIE1994(XYZ_1, xy_o1, xy_o2, Y_o, E_o1, E_o2),
            XYZ_2,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ_1 = np.reshape(XYZ_1, (2, 3, 3))
        xy_o1 = np.reshape(xy_o1, (2, 3, 2))
        xy_o2 = np.reshape(xy_o2, (2, 3, 2))
        Y_o = np.reshape(Y_o, (2, 3))
        E_o1 = np.reshape(E_o1, (2, 3))
        E_o2 = np.reshape(E_o2, (2, 3))
        XYZ_2 = np.reshape(XYZ_2, (2, 3, 3))
        np.testing.assert_allclose(
            chromatic_adaptation_CIE1994(XYZ_1, xy_o1, xy_o2, Y_o, E_o1, E_o2),
            XYZ_2,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_chromatic_adaptation_CIE1994(self):
        """
        Test :func:`colour.adaptation.cie1994.chromatic_adaptation_CIE1994`
        definition domain and range scale support.
        """

        XYZ_1 = np.array([28.00, 21.26, 5.27])
        xy_o1 = np.array([0.44760, 0.40740])
        xy_o2 = np.array([0.31270, 0.32900])
        Y_o = 20
        E_o1 = 1000
        E_o2 = 1000
        XYZ_2 = chromatic_adaptation_CIE1994(
            XYZ_1, xy_o1, xy_o2, Y_o, E_o1, E_o2
        )

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    chromatic_adaptation_CIE1994(
                        XYZ_1 * factor, xy_o1, xy_o2, Y_o * factor, E_o1, E_o2
                    ),
                    XYZ_2 * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_chromatic_adaptation_CIE1994(self):
        """
        Test :func:`colour.adaptation.cie1994.chromatic_adaptation_CIE1994`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        chromatic_adaptation_CIE1994(
            cases,
            cases[..., 0:2],
            cases[..., 0:2],
            cases[..., 0],
            cases[..., 0],
            cases[..., 0],
        )


if __name__ == "__main__":
    unittest.main()
