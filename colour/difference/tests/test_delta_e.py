# -*- coding: utf-8 -*-
"""
Defines the unit tests for the :mod:`colour.difference.delta_e` module.

References
----------
-   :cite:`Sharma2005b` : Sharma, G., Wu, W., & Dalal, E. N. (2005). The
    CIEDE2000 color-difference formula: Implementation notes, supplementary
    test data, and mathematical observations. Color Research & Application,
    30(1), 21-30. doi:10.1002/col.20070
"""

import numpy as np
import unittest
from itertools import permutations

from colour.difference import (
    delta_E_CIE1976,
    delta_E_CIE1994,
    delta_E_CIE2000,
    delta_E_CMC,
)

from colour.algebra import euclidean_distance
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestDelta_E_CIE1976',
    'TestDelta_E_CIE1994',
    'TestDelta_E_CIE2000',
    'TestDelta_E_CMC',
]


class TestDelta_E_CIE1976(unittest.TestCase):
    """
    Defines :func:`colour.difference.delta_e.delta_E_CIE1976` definition unit
    tests methods.

    Notes
    -----
    -   :func:`colour.difference.delta_e.delta_E_CIE1976` definition is a
        wrapper around :func:`colour.algebra.geometry.euclidean_distance`
        definition, thus unit tests are not entirely implemented.
    """

    def test_delta_E_CIE1976(self):
        """
        Tests :func:`colour.difference.delta_e.delta_E_CIE1976` definition.
        """

        Lab_1 = np.array([100.00000000, 21.57210357, 272.22819350])
        Lab_2 = np.array([100.00000000, 426.67945353, 72.39590835])
        Lab_1 = np.tile(Lab_1, (6, 1)).reshape([2, 3, 3])
        Lab_2 = np.tile(Lab_2, (6, 1)).reshape([2, 3, 3])

        np.testing.assert_almost_equal(
            delta_E_CIE1976(Lab_1, Lab_2),
            euclidean_distance(Lab_1, Lab_2),
            decimal=7)

    def test_n_dimensional_delta_E_CIE1976(self):
        """
        Tests :func:`colour.difference.delta_e.delta_E_CIE1976` definition
        n-dimensional arrays support.
        """

        pass

    def test_domain_range_scale_delta_E_CIE1976(self):
        """
        Tests :func:`colour.difference.delta_e.delta_E_CIE1976` definition
        domain and range scale support.
        """

        Lab_1 = np.array([100.00000000, 21.57210357, 272.22819350])
        Lab_2 = np.array([100.00000000, 426.67945353, 72.39590835])

        d_r = (('reference', 1), ('1', 0.01), ('100', 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    delta_E_CIE1976(Lab_1 * factor, Lab_2 * factor),
                    euclidean_distance(Lab_1, Lab_2),
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_delta_E_CIE1976(self):
        """
        Tests :func:`colour.difference.delta_e.delta_E_CIE1976` definition nan
        support.
        """

        pass


class TestDelta_E_CIE1994(unittest.TestCase):
    """
    Defines :func:`colour.difference.delta_e.delta_E_CIE1994` definition unit
    tests methods.
    """

    def test_delta_E_CIE1994(self):
        """
        Tests :func:`colour.difference.delta_e.delta_E_CIE1994` definition.
        """

        self.assertAlmostEqual(
            delta_E_CIE1994(
                np.array([100.00000000, 21.57210357, 272.22819350]),
                np.array([100.00000000, 426.67945353, 72.39590835])),
            83.779225500887094,
            places=7)

        self.assertAlmostEqual(
            delta_E_CIE1994(
                np.array([100.00000000, 21.57210357, 272.22819350]),
                np.array([100.00000000, 74.05216981, 276.45318193])),
            10.053931954553839,
            places=7)

        self.assertAlmostEqual(
            delta_E_CIE1994(
                np.array([100.00000000, 21.57210357, 272.22819350]),
                np.array([100.00000000, 8.32281957, -73.58297716])),
            57.535453706667425,
            places=7)

        self.assertAlmostEqual(
            delta_E_CIE1994(
                np.array([100.00000000, 21.57210357, 272.22819350]),
                np.array([100.00000000, 426.67945353, 72.39590835]),
                textiles=True),
            88.335553057506502,
            places=7)

        self.assertAlmostEqual(
            delta_E_CIE1994(
                np.array([100.00000000, 21.57210357, 272.22819350]),
                np.array([100.00000000, 74.05216981, 276.45318193]),
                textiles=True),
            10.612657890048272,
            places=7)

        self.assertAlmostEqual(
            delta_E_CIE1994(
                np.array([100.00000000, 21.57210357, 272.22819350]),
                np.array([100.00000000, 8.32281957, -73.58297716]),
                textiles=True),
            60.368687261063329,
            places=7)

    def test_n_dimensional_delta_E_CIE1994(self):
        """
        Tests :func:`colour.difference.delta_e.delta_E_CIE1994` definition
        n-dimensional arrays support.
        """

        Lab_1 = np.array([100.00000000, 21.57210357, 272.22819350])
        Lab_2 = np.array([100.00000000, 426.67945353, 72.39590835])
        delta_E = delta_E_CIE1994(Lab_1, Lab_2)

        Lab_1 = np.tile(Lab_1, (6, 1))
        Lab_2 = np.tile(Lab_2, (6, 1))
        delta_E = np.tile(delta_E, 6)
        np.testing.assert_almost_equal(
            delta_E_CIE1994(Lab_1, Lab_2), delta_E, decimal=7)

        Lab_1 = np.reshape(Lab_1, (2, 3, 3))
        Lab_2 = np.reshape(Lab_2, (2, 3, 3))
        delta_E = np.reshape(delta_E, (2, 3))
        np.testing.assert_almost_equal(
            delta_E_CIE1994(Lab_1, Lab_2), delta_E, decimal=7)

    def test_domain_range_scale_delta_E_CIE1994(self):
        """
        Tests :func:`colour.difference.delta_e.delta_E_CIE1994` definition
        domain and range scale support.
        """

        Lab_1 = np.array([100.00000000, 21.57210357, 272.22819350])
        Lab_2 = np.array([100.00000000, 426.67945353, 72.39590835])
        delta_E = delta_E_CIE1994(Lab_1, Lab_2)

        d_r = (('reference', 1), ('1', 0.01), ('100', 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    delta_E_CIE1994(Lab_1 * factor, Lab_2 * factor),
                    delta_E,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_delta_E_CIE1994(self):
        """
        Tests :func:`colour.difference.delta_e.delta_E_CIE1994` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            Lab_1 = np.array(case)
            Lab_2 = np.array(case)
            delta_E_CIE1994(Lab_1, Lab_2)


class TestDelta_E_CIE2000(unittest.TestCase):
    """
    Defines :func:`colour.difference.delta_e.delta_E_CIE2000` definition unit
    tests methods.
    """

    def test_delta_E_CIE2000(self):
        """
        Tests :func:`colour.difference.delta_e.delta_E_CIE2000` definition.
        """

        self.assertAlmostEqual(
            delta_E_CIE2000(
                np.array([100.00000000, 21.57210357, 272.22819350]),
                np.array([100.00000000, 426.67945353, 72.39590835])),
            94.03564903,
            places=7)

        self.assertAlmostEqual(
            delta_E_CIE2000(
                np.array([100.00000000, 21.57210357, 272.22819350]),
                np.array([100.00000000, 74.05216981, 276.45318193])),
            14.87906419,
            places=7)

        self.assertAlmostEqual(
            delta_E_CIE2000(
                np.array([100.00000000, 21.57210357, 272.22819350]),
                np.array([100.00000000, 8.32281957, -73.58297716])),
            68.23094879,
            places=7)

        self.assertAlmostEqual(
            delta_E_CIE2000(
                np.array([100.00000000, 21.57210357, 272.22819350]),
                np.array([50.00000000, 426.67945353, 72.39590835]),
                textiles=True),
            95.79205352,
            places=7)

        self.assertAlmostEqual(
            delta_E_CIE2000(
                np.array([100.00000000, 21.57210357, 272.22819350]),
                np.array([50.00000000, 74.05216981, 276.45318193]),
                textiles=True),
            23.55420943,
            places=7)

        self.assertAlmostEqual(
            delta_E_CIE2000(
                np.array([100.00000000, 21.57210357, 272.22819350]),
                np.array([50.00000000, 8.32281957, -73.58297716]),
                textiles=True),
            70.63198003,
            places=7)

    def test_n_dimensional_delta_E_CIE2000(self):
        """
        Tests :func:`colour.difference.delta_e.delta_E_CIE2000` definition
        n-dimensional arrays support.
        """

        Lab_1 = np.array([100.00000000, 21.57210357, 272.22819350])
        Lab_2 = np.array([100.00000000, 426.67945353, 72.39590835])
        delta_E = delta_E_CIE2000(Lab_1, Lab_2)

        Lab_1 = np.tile(Lab_1, (6, 1))
        Lab_2 = np.tile(Lab_2, (6, 1))
        delta_E = np.tile(delta_E, 6)
        np.testing.assert_almost_equal(
            delta_E_CIE2000(Lab_1, Lab_2), delta_E, decimal=7)

        Lab_1 = np.reshape(Lab_1, (2, 3, 3))
        Lab_2 = np.reshape(Lab_2, (2, 3, 3))
        delta_E = np.reshape(delta_E, (2, 3))
        np.testing.assert_almost_equal(
            delta_E_CIE2000(Lab_1, Lab_2), delta_E, decimal=7)

    def test_domain_range_scale_delta_E_CIE2000(self):
        """
        Tests :func:`colour.difference.delta_e.delta_E_CIE2000` definition
        domain and range scale support.
        """

        Lab_1 = np.array([100.00000000, 21.57210357, 272.22819350])
        Lab_2 = np.array([100.00000000, 426.67945353, 72.39590835])
        delta_E = delta_E_CIE2000(Lab_1, Lab_2)

        d_r = (('reference', 1), ('1', 0.01), ('100', 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    delta_E_CIE2000(Lab_1 * factor, Lab_2 * factor),
                    delta_E,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_delta_E_CIE2000(self):
        """
        Tests :func:`colour.difference.delta_e.delta_E_CIE2000` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            Lab_1 = np.array(case)
            Lab_2 = np.array(case)
            delta_E_CIE2000(Lab_1, Lab_2)

    def test_delta_E_CIE2000_Sharma2004(self):
        """
        Tests :func:`colour.difference.delta_e.delta_E_CIE2000` definition
        using Sharma (2004) dataset.

        References
        ----------
        :cite:`Sharma2005b`
        """

        Lab_1 = np.array([
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
            [50.0000, -0.0010, 2.4900],
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
        ])

        Lab_2 = np.array([
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
            [50.0000, 0.0010, -2.4900],
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
        ])

        d_E = np.array([
            2.0425, 2.8615, 3.4412, 1.0000, 1.0000, 1.0000, 2.3669, 2.3669,
            7.1792, 7.1792, 7.2195, 7.2195, 4.8045, 4.8045, 4.7461, 4.3065,
            27.1492, 22.8977, 31.9030, 19.4535, 1.0000, 1.0000, 1.0000, 1.0000,
            1.2644, 1.2630, 1.8731, 1.8645, 2.0373, 1.4146, 1.4441, 1.5381,
            0.6377, 0.9082
        ])

        np.testing.assert_almost_equal(
            delta_E_CIE2000(Lab_1, Lab_2), d_E, decimal=4)


class TestDelta_E_CMC(unittest.TestCase):
    """
    Defines :func:`colour.difference.delta_e.delta_E_CMC` definition unit
    tests methods.
    """

    def test_delta_E_CMC(self):
        """
        Tests :func:`colour.difference.delta_e.delta_E_CMC` definition.
        """

        self.assertAlmostEqual(
            delta_E_CMC(
                np.array([100.00000000, 21.57210357, 272.22819350]),
                np.array([100.00000000, 426.67945353, 72.39590835])),
            172.70477129,
            places=7)

        self.assertAlmostEqual(
            delta_E_CMC(
                np.array([100.00000000, 21.57210357, 272.22819350]),
                np.array([100.00000000, 74.05216981, 276.45318193])),
            20.59732717,
            places=7)

        self.assertAlmostEqual(
            delta_E_CMC(
                np.array([100.00000000, 21.57210357, 272.22819350]),
                np.array([100.00000000, 8.32281957, -73.58297716])),
            121.71841479,
            places=7)

        self.assertAlmostEqual(
            delta_E_CMC(
                np.array([100.00000000, 21.57210357, 272.22819350]),
                np.array([100.00000000, 426.67945353, 72.39590835]),
                l=1),  # noqa
            172.70477129,
            places=7)

        self.assertAlmostEqual(
            delta_E_CMC(
                np.array([100.00000000, 21.57210357, 272.22819350]),
                np.array([100.00000000, 74.05216981, 276.45318193]),
                l=1),  # noqa
            20.59732717,
            places=7)

        self.assertAlmostEqual(
            delta_E_CMC(
                np.array([100.00000000, 21.57210357, 272.22819350]),
                np.array([100.00000000, 8.32281957, -73.58297716]),
                l=1),  # noqa
            121.71841479,
            places=7)

    def test_n_dimensional_delta_E_CMC(self):
        """
        Tests :func:`colour.difference.delta_e.delta_E_CMC` definition
        n-dimensional arrays support.
        """

        Lab_1 = np.array([100.00000000, 21.57210357, 272.22819350])
        Lab_2 = np.array([100.00000000, 426.67945353, 72.39590835])
        delta_E = delta_E_CMC(Lab_1, Lab_2)

        Lab_1 = np.tile(Lab_1, (6, 1))
        Lab_2 = np.tile(Lab_2, (6, 1))
        delta_E = np.tile(delta_E, 6)
        np.testing.assert_almost_equal(
            delta_E_CMC(Lab_1, Lab_2), delta_E, decimal=7)

        Lab_1 = np.reshape(Lab_1, (2, 3, 3))
        Lab_2 = np.reshape(Lab_2, (2, 3, 3))
        delta_E = np.reshape(delta_E, (2, 3))
        np.testing.assert_almost_equal(
            delta_E_CMC(Lab_1, Lab_2), delta_E, decimal=7)

    def test_domain_range_scale_delta_E_CMC(self):
        """
        Tests :func:`colour.difference.delta_e.delta_E_CMC` definition
        domain and range scale support.
        """

        Lab_1 = np.array([100.00000000, 21.57210357, 272.22819350])
        Lab_2 = np.array([100.00000000, 426.67945353, 72.39590835])
        delta_E = delta_E_CMC(Lab_1, Lab_2)

        d_r = (('reference', 1), ('1', 0.01), ('100', 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    delta_E_CMC(Lab_1 * factor, Lab_2 * factor),
                    delta_E,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_delta_E_CMC(self):
        """
        Tests :func:`colour.difference.delta_e.delta_E_CMC` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            Lab_1 = np.array(case)
            Lab_2 = np.array(case)
            delta_E_CMC(Lab_1, Lab_2)


if __name__ == '__main__':
    unittest.main()
