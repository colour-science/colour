# !/usr/bin/env python
"""
Define the unit tests for the :mod:`colour.appearance.hellwig2022` module.

References
----------
-   :cite:`Fairchild2022` : Fairchild, M. D., & Hellwig, L. (2022). Private
    Discussion with Mansencal, T.
"""

import numpy as np
import unittest
from itertools import product

from colour.appearance import (
    VIEWING_CONDITIONS_HELLWIG2022,
    InductionFactors_Hellwig2022,
    CAM_Specification_Hellwig2022,
    XYZ_to_Hellwig2022,
    Hellwig2022_to_XYZ,
)
from colour.utilities import (
    as_float_array,
    domain_range_scale,
    ignore_numpy_errors,
    tsplit,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestXYZ_to_Hellwig2022",
    "TestHellwig2022_to_XYZ",
]


class TestXYZ_to_Hellwig2022(unittest.TestCase):
    """
    Define :func:`colour.appearance.hellwig2022.XYZ_to_Hellwig2022` definition
    unit tests methods.
    """

    def test_XYZ_to_Hellwig2022(self):
        """
        Test :func:`colour.appearance.hellwig2022.XYZ_to_Hellwig2022`
        definition.
        """

        XYZ = np.array([19.01, 20.00, 21.78])
        XYZ_w = np.array([95.05, 100.00, 108.88])
        L_A = 318.31
        Y_b = 20
        surround = VIEWING_CONDITIONS_HELLWIG2022["Average"]
        np.testing.assert_allclose(
            XYZ_to_Hellwig2022(XYZ, XYZ_w, L_A, Y_b, surround),
            np.array(
                [
                    41.731,
                    0.026,
                    217.068,
                    0.061,
                    55.852,
                    0.034,
                    275.59498615,
                    np.nan,
                    41.88027828,
                    56.05183586,
                ]
            ),
            rtol=0.01,
            atol=0.01,
        )

        XYZ = np.array([57.06, 43.06, 31.96])
        L_A = 31.83
        np.testing.assert_allclose(
            XYZ_to_Hellwig2022(XYZ, XYZ_w, L_A, Y_b, surround),
            np.array(
                [
                    65.428,
                    31.330,
                    17.487,
                    47.200,
                    64.077,
                    30.245,
                    398.03047943,
                    np.nan,
                    70.50187436,
                    69.04574688,
                ]
            ),
            rtol=0.01,
            atol=0.01,
        )

        XYZ = np.array([3.53, 6.56, 2.14])
        XYZ_w = np.array([109.85, 100, 35.58])
        L_A = 318.31
        np.testing.assert_allclose(
            XYZ_to_Hellwig2022(XYZ, XYZ_w, L_A, Y_b, surround),
            np.array(
                [
                    21.361,
                    30.603,
                    178.867,
                    141.223,
                    28.590,
                    40.376,
                    223.01823806,
                    np.nan,
                    29.35191711,
                    39.28664523,
                ]
            ),
            rtol=0.01,
            atol=0.01,
        )

        XYZ = np.array([19.01, 20.00, 21.78])
        XYZ_w = np.array([109.85, 100.00, 35.58])
        L_A = 31.38
        np.testing.assert_allclose(
            XYZ_to_Hellwig2022(XYZ, XYZ_w, L_A, Y_b, surround),
            np.array(
                [
                    41.064050542871215,
                    31.939561618552826,
                    259.034056616436715,
                    76.668720573462167,
                    40.196783565499423,
                    30.818359671352116,
                    311.329371306428470,
                    np.nan,
                    49.676917719967385,
                    48.627748198047854,
                ]
            ),
            rtol=0.01,
            atol=0.01,
        )

    def test_n_dimensional_XYZ_to_Hellwig2022(self):
        """
        Test :func:`colour.appearance.hellwig2022.XYZ_to_Hellwig2022` definition
        n-dimensional support.
        """

        XYZ = np.array([19.01, 20.00, 21.78])
        XYZ_w = np.array([95.05, 100.00, 108.88])
        L_A = 318.31
        Y_b = 20
        surround = VIEWING_CONDITIONS_HELLWIG2022["Average"]
        specification = XYZ_to_Hellwig2022(XYZ, XYZ_w, L_A, Y_b, surround)

        XYZ = np.tile(XYZ, (6, 1))
        specification = np.tile(specification, (6, 1))
        np.testing.assert_array_almost_equal(
            XYZ_to_Hellwig2022(XYZ, XYZ_w, L_A, Y_b, surround),
            specification,
            decimal=7,
        )

        XYZ_w = np.tile(XYZ_w, (6, 1))
        np.testing.assert_array_almost_equal(
            XYZ_to_Hellwig2022(XYZ, XYZ_w, L_A, Y_b, surround),
            specification,
            decimal=7,
        )

        XYZ = np.reshape(XYZ, (2, 3, 3))
        XYZ_w = np.reshape(XYZ_w, (2, 3, 3))
        specification = np.reshape(specification, (2, 3, 10))
        np.testing.assert_array_almost_equal(
            XYZ_to_Hellwig2022(XYZ, XYZ_w, L_A, Y_b, surround),
            specification,
            decimal=7,
        )

    @ignore_numpy_errors
    def test_domain_range_scale_XYZ_to_Hellwig2022(self):
        """
        Test :func:`colour.appearance.hellwig2022.XYZ_to_Hellwig2022`
        definition domain and range scale support.
        """

        XYZ = np.array([19.01, 20.00, 21.78])
        XYZ_w = np.array([95.05, 100.00, 108.88])
        L_A = 318.31
        Y_b = 20
        surround = VIEWING_CONDITIONS_HELLWIG2022["Average"]
        specification = XYZ_to_Hellwig2022(XYZ, XYZ_w, L_A, Y_b, surround)

        d_r = (
            ("reference", 1, 1),
            (
                "1",
                0.01,
                np.array(
                    [
                        1 / 100,
                        1 / 100,
                        1 / 360,
                        1 / 100,
                        1 / 100,
                        1 / 100,
                        1 / 400,
                        np.nan,
                        1 / 100,
                        1 / 100,
                    ]
                ),
            ),
            (
                "100",
                1,
                np.array([1, 1, 100 / 360, 1, 1, 1, 100 / 400, np.nan, 1, 1]),
            ),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_array_almost_equal(
                    XYZ_to_Hellwig2022(
                        XYZ * factor_a, XYZ_w * factor_a, L_A, Y_b, surround
                    ),
                    as_float_array(specification) * factor_b,
                    decimal=7,
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_Hellwig2022(self):
        """
        Test :func:`colour.appearance.hellwig2022.XYZ_to_Hellwig2022
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        surround = InductionFactors_Hellwig2022(
            cases[0, 0], cases[0, 0], cases[0, 0]
        )
        XYZ_to_Hellwig2022(
            cases, cases, cases[..., 0], cases[..., 0], surround
        )


class TestHellwig2022_to_XYZ(unittest.TestCase):
    """
    Define :func:`colour.appearance.hellwig2022.Hellwig2022_to_XYZ` definition
    unit tests methods.
    """

    def test_Hellwig2022_to_XYZ(self):
        """
        Test :func:`colour.appearance.hellwig2022.Hellwig2022_to_XYZ`
        definition.
        """

        specification = CAM_Specification_Hellwig2022(
            41.731207905126638, 0.025763615829912909, 217.06795976739301
        )
        XYZ_w = np.array([95.05, 100.00, 108.88])
        L_A = 318.31
        Y_b = 20
        surround = VIEWING_CONDITIONS_HELLWIG2022["Average"]
        np.testing.assert_array_almost_equal(
            Hellwig2022_to_XYZ(specification, XYZ_w, L_A, Y_b, surround),
            np.array([19.01, 20.00, 21.78]),
            decimal=7,
        )

        specification = CAM_Specification_Hellwig2022(
            65.428280687118473, 31.330032520870901, 17.486592427576902
        )
        L_A = 31.83
        np.testing.assert_array_almost_equal(
            Hellwig2022_to_XYZ(specification, XYZ_w, L_A, Y_b, surround),
            np.array([57.06, 43.06, 31.96]),
            decimal=7,
        )

        specification = CAM_Specification_Hellwig2022(
            21.360528925833027, 30.603219780800902, 178.8672426588991
        )
        XYZ_w = np.array([109.85, 100, 35.58])
        L_A = 318.31
        np.testing.assert_array_almost_equal(
            Hellwig2022_to_XYZ(specification, XYZ_w, L_A, Y_b, surround),
            np.array([3.53, 6.56, 2.14]),
            decimal=7,
        )

        specification = CAM_Specification_Hellwig2022(
            41.064050542871215, 31.939561618552826, 259.03405661643671
        )
        L_A = 31.38
        np.testing.assert_array_almost_equal(
            Hellwig2022_to_XYZ(specification, XYZ_w, L_A, Y_b, surround),
            np.array([19.01, 20.00, 21.78]),
            decimal=7,
        )

        specification = CAM_Specification_Hellwig2022(
            J_HK=41.880278283880095, C=0.025763615829913, h=217.067959767393010
        )
        XYZ_w = np.array([95.05, 100.00, 108.88])
        L_A = 318.31
        Y_b = 20
        surround = VIEWING_CONDITIONS_HELLWIG2022["Average"]
        np.testing.assert_array_almost_equal(
            Hellwig2022_to_XYZ(specification, XYZ_w, L_A, Y_b, surround),
            np.array([19.01, 20.00, 21.78]),
            decimal=7,
        )

    def test_n_dimensional_Hellwig2022_to_XYZ(self):
        """
        Test :func:`colour.appearance.hellwig2022.Hellwig2022_to_XYZ`
        definition n-dimensional support.
        """

        XYZ = np.array([19.01, 20.00, 21.78])
        XYZ_w = np.array([95.05, 100.00, 108.88])
        L_A = 318.31
        Y_b = 20
        surround = VIEWING_CONDITIONS_HELLWIG2022["Average"]
        specification = XYZ_to_Hellwig2022(XYZ, XYZ_w, L_A, Y_b, surround)
        XYZ = Hellwig2022_to_XYZ(specification, XYZ_w, L_A, Y_b, surround)

        specification = CAM_Specification_Hellwig2022(
            *np.transpose(np.tile(tsplit(specification), (6, 1))).tolist()
        )
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_array_almost_equal(
            Hellwig2022_to_XYZ(specification, XYZ_w, L_A, Y_b, surround),
            XYZ,
            decimal=7,
        )

        XYZ_w = np.tile(XYZ_w, (6, 1))
        np.testing.assert_array_almost_equal(
            Hellwig2022_to_XYZ(specification, XYZ_w, L_A, Y_b, surround),
            XYZ,
            decimal=7,
        )

        specification = CAM_Specification_Hellwig2022(
            *tsplit(np.reshape(specification, (2, 3, 10))).tolist()
        )
        XYZ_w = np.reshape(XYZ_w, (2, 3, 3))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_array_almost_equal(
            Hellwig2022_to_XYZ(specification, XYZ_w, L_A, Y_b, surround),
            XYZ,
            decimal=7,
        )

    @ignore_numpy_errors
    def test_domain_range_scale_Hellwig2022_to_XYZ(self):
        """
        Test :func:`colour.appearance.hellwig2022.Hellwig2022_to_XYZ`
        definition domain and range scale support.
        """

        XYZ = np.array([19.01, 20.00, 21.78])
        XYZ_w = np.array([95.05, 100.00, 108.88])
        L_A = 318.31
        Y_b = 20
        surround = VIEWING_CONDITIONS_HELLWIG2022["Average"]
        specification = XYZ_to_Hellwig2022(XYZ, XYZ_w, L_A, Y_b, surround)
        XYZ = Hellwig2022_to_XYZ(specification, XYZ_w, L_A, Y_b, surround)

        d_r = (
            ("reference", 1, 1),
            (
                "1",
                np.array(
                    [
                        1 / 100,
                        1 / 100,
                        1 / 360,
                        1 / 100,
                        1 / 100,
                        1 / 100,
                        1 / 400,
                        np.nan,
                        1 / 100,
                        1 / 100,
                    ]
                ),
                0.01,
            ),
            (
                "100",
                np.array([1, 1, 100 / 360, 1, 1, 1, 100 / 400, np.nan, 1, 1]),
                1,
            ),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_array_almost_equal(
                    Hellwig2022_to_XYZ(
                        specification * factor_a,
                        XYZ_w * factor_b,
                        L_A,
                        Y_b,
                        surround,
                    ),
                    XYZ * factor_b,
                    decimal=7,
                )

    @ignore_numpy_errors
    def test_raise_exception_Hellwig2022_to_XYZ(self):
        """
        Test :func:`colour.appearance.hellwig2022.Hellwig2022_to_XYZ`
        definition raised exception.
        """
        self.assertRaises(
            ValueError,
            Hellwig2022_to_XYZ,
            CAM_Specification_Hellwig2022(
                J_HK=None, C=0.025763615829912909, h=217.06795976739301
            ),
            np.array([95.05, 100.00, 108.88]),
            318.31,
            20.0,
            VIEWING_CONDITIONS_HELLWIG2022["Average"],
        )

        self.assertRaises(
            ValueError,
            Hellwig2022_to_XYZ,
            CAM_Specification_Hellwig2022(
                41.731207905126638, None, 217.06795976739301
            ),
            np.array([95.05, 100.00, 108.88]),
            318.31,
            20.0,
            VIEWING_CONDITIONS_HELLWIG2022["Average"],
        )

    @ignore_numpy_errors
    def test_nan_Hellwig2022_to_XYZ(self):
        """
        Test :func:`colour.appearance.hellwig2022.Hellwig2022_to_XYZ`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        surround = InductionFactors_Hellwig2022(
            cases[0, 0], cases[0, 0], cases[0, 0]
        )
        Hellwig2022_to_XYZ(
            CAM_Specification_Hellwig2022(
                cases[..., 0], cases[..., 0], cases[..., 0], M=50
            ),
            cases,
            cases[..., 0],
            cases[..., 0],
            surround,
        )


if __name__ == "__main__":
    unittest.main()
