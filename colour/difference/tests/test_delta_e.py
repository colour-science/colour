"""
Define the unit tests for the :mod:`colour.difference.delta_e` module.

References
----------
-   :cite:`Sharma2005b` : Sharma, G., Wu, W., & Dalal, E. N. (2005). The
    CIEDE2000 color-difference formula: Implementation notes, supplementary
    test data, and mathematical observations. Color Research & Application,
    30(1), 21-30. doi:10.1002/col.20070
"""

from itertools import product

import numpy as np

from colour.algebra import euclidean_distance
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.difference import (
    delta_E_CIE1976,
    delta_E_CIE1994,
    delta_E_CIE2000,
    delta_E_CMC,
    delta_E_ITP,
)
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestDelta_E_CIE1976",
    "TestDelta_E_CIE1994",
    "TestDelta_E_CIE2000",
    "TestDelta_E_CMC",
    "TestDelta_E_ITP",
]


class TestDelta_E_CIE1976:
    """
    Define :func:`colour.difference.delta_e.delta_E_CIE1976` definition unit
    tests methods.

    Notes
    -----
    -   :func:`colour.difference.delta_e.delta_E_CIE1976` definition is a
        wrapper around :func:`colour.algebra.geometry.euclidean_distance`
        definition, thus unit tests are not entirely implemented.
    """

    def test_delta_E_CIE1976(self):
        """Test :func:`colour.difference.delta_e.delta_E_CIE1976` definition."""

        Lab_1 = np.array([100.00000000, 21.57210357, 272.22819350])
        Lab_2 = np.array([100.00000000, 426.67945353, 72.39590835])
        Lab_1 = np.reshape(np.tile(Lab_1, (6, 1)), (2, 3, 3))
        Lab_2 = np.reshape(np.tile(Lab_2, (6, 1)), (2, 3, 3))

        np.testing.assert_allclose(
            delta_E_CIE1976(Lab_1, Lab_2),
            euclidean_distance(Lab_1, Lab_2),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_delta_E_CIE1976(self):
        """
        Test :func:`colour.difference.delta_e.delta_E_CIE1976` definition
        n-dimensional arrays support.
        """

    def test_domain_range_scale_delta_E_CIE1976(self):
        """
        Test :func:`colour.difference.delta_e.delta_E_CIE1976` definition
        domain and range scale support.
        """

        Lab_1 = np.array([100.00000000, 21.57210357, 272.22819350])
        Lab_2 = np.array([100.00000000, 426.67945353, 72.39590835])

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    delta_E_CIE1976(Lab_1 * factor, Lab_2 * factor),
                    euclidean_distance(Lab_1, Lab_2),
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_delta_E_CIE1976(self):
        """
        Test :func:`colour.difference.delta_e.delta_E_CIE1976` definition nan
        support.
        """


class TestDelta_E_CIE1994:
    """
    Define :func:`colour.difference.delta_e.delta_E_CIE1994` definition unit
    tests methods.
    """

    def test_delta_E_CIE1994(self):
        """Test :func:`colour.difference.delta_e.delta_E_CIE1994` definition."""

        np.testing.assert_allclose(
            delta_E_CIE1994(
                np.array([100.00000000, 21.57210357, 272.22819350]),
                np.array([100.00000000, 426.67945353, 72.39590835]),
            ),
            83.779225500887094,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            delta_E_CIE1994(
                np.array([100.00000000, 21.57210357, 272.22819350]),
                np.array([100.00000000, 74.05216981, 276.45318193]),
            ),
            10.053931954553839,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            delta_E_CIE1994(
                np.array([100.00000000, 21.57210357, 272.22819350]),
                np.array([100.00000000, 8.32281957, -73.58297716]),
            ),
            57.535453706667425,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            delta_E_CIE1994(
                np.array([100.00000000, 21.57210357, 272.22819350]),
                np.array([100.00000000, 426.67945353, 72.39590835]),
                textiles=True,
            ),
            88.335553057506502,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            delta_E_CIE1994(
                np.array([100.00000000, 21.57210357, 272.22819350]),
                np.array([100.00000000, 74.05216981, 276.45318193]),
                textiles=True,
            ),
            10.612657890048272,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            delta_E_CIE1994(
                np.array([100.00000000, 21.57210357, 272.22819350]),
                np.array([100.00000000, 8.32281957, -73.58297716]),
                textiles=True,
            ),
            60.368687261063329,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_delta_E_CIE1994(self):
        """
        Test :func:`colour.difference.delta_e.delta_E_CIE1994` definition
        n-dimensional arrays support.
        """

        Lab_1 = np.array([100.00000000, 21.57210357, 272.22819350])
        Lab_2 = np.array([100.00000000, 426.67945353, 72.39590835])
        delta_E = delta_E_CIE1994(Lab_1, Lab_2)

        Lab_1 = np.tile(Lab_1, (6, 1))
        Lab_2 = np.tile(Lab_2, (6, 1))
        delta_E = np.tile(delta_E, 6)
        np.testing.assert_allclose(
            delta_E_CIE1994(Lab_1, Lab_2),
            delta_E,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        Lab_1 = np.reshape(Lab_1, (2, 3, 3))
        Lab_2 = np.reshape(Lab_2, (2, 3, 3))
        delta_E = np.reshape(delta_E, (2, 3))
        np.testing.assert_allclose(
            delta_E_CIE1994(Lab_1, Lab_2),
            delta_E,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_delta_E_CIE1994(self):
        """
        Test :func:`colour.difference.delta_e.delta_E_CIE1994` definition
        domain and range scale support.
        """

        Lab_1 = np.array([100.00000000, 21.57210357, 272.22819350])
        Lab_2 = np.array([100.00000000, 426.67945353, 72.39590835])
        delta_E = delta_E_CIE1994(Lab_1, Lab_2)

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    delta_E_CIE1994(Lab_1 * factor, Lab_2 * factor),
                    delta_E,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_delta_E_CIE1994(self):
        """
        Test :func:`colour.difference.delta_e.delta_E_CIE1994` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        delta_E_CIE1994(cases, cases)


class TestDelta_E_CIE2000:
    """
    Define :func:`colour.difference.delta_e.delta_E_CIE2000` definition unit
    tests methods.
    """

    def test_delta_E_CIE2000(self):
        """Test :func:`colour.difference.delta_e.delta_E_CIE2000` definition."""

        np.testing.assert_allclose(
            delta_E_CIE2000(
                np.array([100.00000000, 21.57210357, 272.22819350]),
                np.array([100.00000000, 426.67945353, 72.39590835]),
            ),
            94.03564903,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            delta_E_CIE2000(
                np.array([100.00000000, 21.57210357, 272.22819350]),
                np.array([100.00000000, 74.05216981, 276.45318193]),
            ),
            14.87906419,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            delta_E_CIE2000(
                np.array([100.00000000, 21.57210357, 272.22819350]),
                np.array([100.00000000, 8.32281957, -73.58297716]),
            ),
            68.23111251,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            delta_E_CIE2000(
                np.array([100.00000000, 21.57210357, 272.22819350]),
                np.array([50.00000000, 426.67945353, 72.39590835]),
                textiles=True,
            ),
            95.79205352,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            delta_E_CIE2000(
                np.array([100.00000000, 21.57210357, 272.22819350]),
                np.array([50.00000000, 74.05216981, 276.45318193]),
                textiles=True,
            ),
            23.55420943,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            delta_E_CIE2000(
                np.array([100.00000000, 21.57210357, 272.22819350]),
                np.array([50.00000000, 8.32281957, -73.58297716]),
                textiles=True,
            ),
            70.63213819,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_delta_E_CIE2000(self):
        """
        Test :func:`colour.difference.delta_e.delta_E_CIE2000` definition
        n-dimensional arrays support.
        """

        Lab_1 = np.array([100.00000000, 21.57210357, 272.22819350])
        Lab_2 = np.array([100.00000000, 426.67945353, 72.39590835])
        delta_E = delta_E_CIE2000(Lab_1, Lab_2)

        Lab_1 = np.tile(Lab_1, (6, 1))
        Lab_2 = np.tile(Lab_2, (6, 1))
        delta_E = np.tile(delta_E, 6)
        np.testing.assert_allclose(
            delta_E_CIE2000(Lab_1, Lab_2),
            delta_E,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        Lab_1 = np.reshape(Lab_1, (2, 3, 3))
        Lab_2 = np.reshape(Lab_2, (2, 3, 3))
        delta_E = np.reshape(delta_E, (2, 3))
        np.testing.assert_allclose(
            delta_E_CIE2000(Lab_1, Lab_2),
            delta_E,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_delta_E_CIE2000(self):
        """
        Test :func:`colour.difference.delta_e.delta_E_CIE2000` definition
        domain and range scale support.
        """

        Lab_1 = np.array([100.00000000, 21.57210357, 272.22819350])
        Lab_2 = np.array([100.00000000, 426.67945353, 72.39590835])
        delta_E = delta_E_CIE2000(Lab_1, Lab_2)

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    delta_E_CIE2000(Lab_1 * factor, Lab_2 * factor),
                    delta_E,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_delta_E_CIE2000(self):
        """
        Test :func:`colour.difference.delta_e.delta_E_CIE2000` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        delta_E_CIE2000(cases, cases)

    def test_delta_E_CIE2000_Sharma2004(self):
        """
        Test :func:`colour.difference.delta_e.delta_E_CIE2000` definition
        using Sharma (2004) dataset.

        References
        ----------
        :cite:`Sharma2005b`
        """

        # NOTE: The 14th test case is excluded as "Numpy" 1.24.0 introduced
        # numerical differences between "Linux" and "macOS / Windows" with the
        # "np.arctan2" definition :
        #
        #             |               Ubuntu |      macOS / Windows |
        # C_1_ab      |    2.490000200803205 |    2.490000200803205 |
        # C_2_ab      |    2.490000200803205 |    2.490000200803205 |
        # C_bar_ab    |    2.490000200803205 |    2.490000200803205 |
        # C_bar_ab_7  |  593.465770158617033 |  593.465770158617033 |
        # G           |    0.499844088629080 |    0.499844088629080 |
        # a_p_1       |   -0.001499844088629 |   -0.001499844088629 |
        # a_p_2       |    0.001499844088629 |    0.001499844088629 |
        # C_p_1       |    2.490000451713271 |    2.490000451713271 |
        # C_p_2       |    2.490000451713271 |    2.490000451713271 |
        # h_p_1       |   90.034511938077543 |   90.034511938077557 | <--
        # h_p_2       |  270.034511938077571 |  270.034511938077571 |
        # delta_L_p   |    0.000000000000000 |    0.000000000000000 |
        # delta_C_p   |    0.000000000000000 |    0.000000000000000 |
        # h_p_2_s_1   |  180.000000000000028 |  180.000000000000000 | <--
        # C_p_1_m_2   |    6.200102249532291 |    6.200102249532291 |
        # delta_h_p   | -179.999999999999972 |  180.000000000000000 | <--
        # delta_H_p   |   -4.980000903426540 |    4.980000903426541 | <--
        # L_bar_p     |   50.000000000000000 |   50.000000000000000 |
        # C_bar_p     |    2.490000451713271 |    2.490000451713271 |
        # a_h_p_1_s_2 |  180.000000000000028 |  180.000000000000000 | <--
        # h_p_1_a_2   |  360.069023876155143 |  360.069023876155143 |
        # h_bar_p     |    0.034511938077571 |  180.034511938077571 |
        # T           |    1.319683185432364 |    0.977862082189372 | <--
        # delta_theta |    0.000000000000000 |    0.000016235458767 | <--
        # C_bar_p_7   |  593.466188771459770 |  593.466188771459770 |
        # R_C         |    0.000623645703630 |    0.000623645703630 |
        # L_bar_p_2   |    0.000000000000000 |    0.000000000000000 |
        # S_L         |    1.000000000000000 |    1.000000000000000 |
        # S_C         |    1.112050020327097 |    1.112050020327097 |
        # S_H         |    1.049290175917675 |    1.036523155395472 | <--
        # R_T         |   -0.000000000000000 |   -0.000000000353435 | <--
        # d_E         |    4.746066453039259 |    4.804524508211768 | <--

        Lab_1 = np.array(
            [
                [50.0000, 2.6772, -79.7751],
                [50.0000, 3.1571, -77.2803],
                [50.0000, 2.8361, -74.0200],
                [50.0000, -1.3802, -84.2814],
                [50.0000, -1.1848, -84.8006],
                [50.0000, -0.9009, -85.5211],
                [50.0000, 0.0000, 0.0000],
                [50.0000, -1.0000, 2.0000],
                [50.0000, 2.4900, -0.0010],
                [50.0000, 2.4900, -0.0010],
                [50.0000, 2.4900, -0.0010],
                [50.0000, 2.4900, -0.0010],
                [50.0000, -0.0010, 2.4900],
                # [50.0000, -0.0010, 2.4900],
                [50.0000, -0.0010, 2.4900],
                [50.0000, 2.5000, 0.0000],
                [50.0000, 2.5000, 0.0000],
                [50.0000, 2.5000, 0.0000],
                [50.0000, 2.5000, 0.0000],
                [50.0000, 2.5000, 0.0000],
                [50.0000, 2.5000, 0.0000],
                [50.0000, 2.5000, 0.0000],
                [50.0000, 2.5000, 0.0000],
                [50.0000, 2.5000, 0.0000],
                [60.2574, -34.0099, 36.2677],
                [63.0109, -31.0961, -5.8663],
                [61.2901, 3.7196, -5.3901],
                [35.0831, -44.1164, 3.7933],
                [22.7233, 20.0904, -46.6940],
                [36.4612, 47.8580, 18.3852],
                [90.8027, -2.0831, 1.4410],
                [90.9257, -0.5406, -0.9208],
                [6.7747, -0.2908, -2.4247],
                [2.0776, 0.0795, -1.1350],
            ]
        )

        Lab_2 = np.array(
            [
                [50.0000, 0.0000, -82.7485],
                [50.0000, 0.0000, -82.7485],
                [50.0000, 0.0000, -82.7485],
                [50.0000, 0.0000, -82.7485],
                [50.0000, 0.0000, -82.7485],
                [50.0000, 0.0000, -82.7485],
                [50.0000, -1.0000, 2.0000],
                [50.0000, 0.0000, 0.0000],
                [50.0000, -2.4900, 0.0009],
                [50.0000, -2.4900, 0.0010],
                [50.0000, -2.4900, 0.0011],
                [50.0000, -2.4900, 0.0012],
                [50.0000, 0.0009, -2.4900],
                # [50.0000, 0.0010, -2.4900],
                [50.0000, 0.0011, -2.4900],
                [50.0000, 0.0000, -2.5000],
                [73.0000, 25.0000, -18.0000],
                [61.0000, -5.0000, 29.0000],
                [56.0000, -27.0000, -3.0000],
                [58.0000, 24.0000, 15.0000],
                [50.0000, 3.1736, 0.5854],
                [50.0000, 3.2972, 0.0000],
                [50.0000, 1.8634, 0.5757],
                [50.0000, 3.2592, 0.3350],
                [60.4626, -34.1751, 39.4387],
                [62.8187, -29.7946, -4.0864],
                [61.4292, 2.2480, -4.9620],
                [35.0232, -40.0716, 1.5901],
                [23.0331, 14.9730, -42.5619],
                [36.2715, 50.5065, 21.2231],
                [91.1528, -1.6435, 0.0447],
                [88.6381, -0.8985, -0.7239],
                [5.8714, -0.0985, -2.2286],
                [0.9033, -0.0636, -0.5514],
            ]
        )

        d_E = np.array(
            [
                2.0425,
                2.8615,
                3.4412,
                1.0000,
                1.0000,
                1.0000,
                2.3669,
                2.3669,
                7.1792,
                7.1792,
                7.2195,
                7.2195,
                4.8045,
                # 4.8045,
                4.7461,
                4.3065,
                27.1492,
                22.8977,
                31.9030,
                19.4535,
                1.0000,
                1.0000,
                1.0000,
                1.0000,
                1.2644,
                1.2630,
                1.8731,
                1.8645,
                2.0373,
                1.4146,
                1.4441,
                1.5381,
                0.6377,
                0.9082,
            ]
        )

        np.testing.assert_allclose(delta_E_CIE2000(Lab_1, Lab_2), d_E, atol=1e-4)


class TestDelta_E_CMC:
    """
    Define :func:`colour.difference.delta_e.delta_E_CMC` definition unit tests
    methods.
    """

    def test_delta_E_CMC(self):
        """Test :func:`colour.difference.delta_e.delta_E_CMC` definition."""

        np.testing.assert_allclose(
            delta_E_CMC(
                np.array([100.00000000, 21.57210357, 272.22819350]),
                np.array([100.00000000, 426.67945353, 72.39590835]),
            ),
            172.70477129,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            delta_E_CMC(
                np.array([100.00000000, 21.57210357, 272.22819350]),
                np.array([100.00000000, 74.05216981, 276.45318193]),
            ),
            20.59732717,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            delta_E_CMC(
                np.array([100.00000000, 21.57210357, 272.22819350]),
                np.array([100.00000000, 8.32281957, -73.58297716]),
            ),
            121.71841479,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            delta_E_CMC(
                np.array([100.00000000, 21.57210357, 272.22819350]),
                np.array([100.00000000, 426.67945353, 72.39590835]),
                l=1,
            ),
            172.70477129,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            delta_E_CMC(
                np.array([100.00000000, 21.57210357, 272.22819350]),
                np.array([100.00000000, 74.05216981, 276.45318193]),
                l=1,
            ),
            20.59732717,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            delta_E_CMC(
                np.array([100.00000000, 21.57210357, 272.22819350]),
                np.array([100.00000000, 8.32281957, -73.58297716]),
                l=1,
            ),
            121.71841479,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_delta_E_CMC(self):
        """
        Test :func:`colour.difference.delta_e.delta_E_CMC` definition
        n-dimensional arrays support.
        """

        Lab_1 = np.array([100.00000000, 21.57210357, 272.22819350])
        Lab_2 = np.array([100.00000000, 426.67945353, 72.39590835])
        delta_E = delta_E_CMC(Lab_1, Lab_2)

        Lab_1 = np.tile(Lab_1, (6, 1))
        Lab_2 = np.tile(Lab_2, (6, 1))
        delta_E = np.tile(delta_E, 6)
        np.testing.assert_allclose(
            delta_E_CMC(Lab_1, Lab_2), delta_E, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        Lab_1 = np.reshape(Lab_1, (2, 3, 3))
        Lab_2 = np.reshape(Lab_2, (2, 3, 3))
        delta_E = np.reshape(delta_E, (2, 3))
        np.testing.assert_allclose(
            delta_E_CMC(Lab_1, Lab_2), delta_E, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_delta_E_CMC(self):
        """
        Test :func:`colour.difference.delta_e.delta_E_CMC` definition
        domain and range scale support.
        """

        Lab_1 = np.array([100.00000000, 21.57210357, 272.22819350])
        Lab_2 = np.array([100.00000000, 426.67945353, 72.39590835])
        delta_E = delta_E_CMC(Lab_1, Lab_2)

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    delta_E_CMC(Lab_1 * factor, Lab_2 * factor),
                    delta_E,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_delta_E_CMC(self):
        """
        Test :func:`colour.difference.delta_e.delta_E_CMC` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        delta_E_CMC(cases, cases)


class TestDelta_E_ITP:
    """
    Define :func:`colour.difference.delta_e.delta_E_ITP` definition unit tests
    methods.
    """

    def test_delta_E_ITP(self):
        """Test :func:`colour.difference.delta_e.delta_E_ITP` definition."""

        np.testing.assert_allclose(
            delta_E_ITP(
                # RGB: (110, 82, 69), Dark Skin
                np.array([0.4885468072, -0.04739350675, 0.07475401302]),
                np.array([0.4899203231, -0.04567508203, 0.07361341775]),
            ),
            1.426572247,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            delta_E_ITP(
                # RGB: (110, 82, 69), 100% White
                np.array([0.7538438727, 0, -6.25e-16]),
                np.array([0.7538912244, 0.001930922514, -0.0003599955951]),
            ),
            0.7426668055,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            delta_E_ITP(
                # RGB: (0, 0, 0), 100% Black
                np.array([0.1596179061, 0, -1.21e-16]),
                np.array([0.1603575152, 0.02881444889, -0.009908665843]),
            ),
            12.60096264,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            delta_E_ITP(
                # RGB: (255, 0, 0), 100% Red
                np.array([0.5965650331, -0.2083210482, 0.3699729716]),
                np.array([0.596263079, -0.1629742033, 0.3617767026]),
            ),
            17.36012552,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            delta_E_ITP(
                # RGB: (0, 255, 0), 100% Green
                np.array([0.7055787513, -0.4063731514, -0.07278767382]),
                np.array([0.7046946082, -0.3771037586, -0.07141626753]),
            ),
            10.60227327,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            delta_E_ITP(
                # RGB: (255, 0, 0), 100% Blue
                np.array([0.5180652611, 0.2932420978, -0.1873112695]),
                np.array([0.5167090868, 0.298191609, -0.1824609953]),
            ),
            4.040270489,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            delta_E_ITP(
                # RGB: (0, 255, 255), 100% Cyan
                np.array([0.7223275939, -0.01290632441, -0.1139004748]),
                np.array([0.7215329274, -0.007863821961, -0.1106683944]),
            ),
            3.00633812,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            delta_E_ITP(
                # RGB: (255, 0, 255), 100% Magenta
                np.array([0.6401125212, 0.280225698, 0.1665590804]),
                np.array([0.640473651, 0.2819981563, 0.1654050172]),
            ),
            1.07944277,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            delta_E_ITP(
                # RGB: (255, 255, 0), 100% Yellow
                np.array([0.7413041405, -0.3638807621, 0.04959414794]),
                np.array([0.7412815181, -0.3299076141, 0.04545287368]),
            ),
            12.5885645,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_delta_E_ITP(self):
        """
        Test :func:`colour.difference.delta_e.delta_E_ITP` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        delta_E_ITP(cases, cases)
