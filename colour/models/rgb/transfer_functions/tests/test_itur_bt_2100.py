# -*- coding: utf-8 -*-
"""
Defines the unit tests for the
:mod:`colour.models.rgb.transfer_functions.itur_bt_2100` module.
"""

import numpy as np
import unittest

from colour.models.rgb.transfer_functions import (
    oetf_PQ_BT2100,
    oetf_inverse_PQ_BT2100,
    eotf_PQ_BT2100,
    eotf_inverse_PQ_BT2100,
    ootf_PQ_BT2100,
    ootf_inverse_PQ_BT2100,
    oetf_HLG_BT2100,
    oetf_inverse_HLG_BT2100,
)
from colour.models.rgb.transfer_functions.itur_bt_2100 import (
    eotf_HLG_BT2100_1,
    eotf_HLG_BT2100_2,
    eotf_inverse_HLG_BT2100_1,
    eotf_inverse_HLG_BT2100_2,
    ootf_HLG_BT2100_1,
    ootf_HLG_BT2100_2,
    ootf_inverse_HLG_BT2100_1,
    ootf_inverse_HLG_BT2100_2,
)
from colour.models.rgb.transfer_functions.itur_bt_2100 import (
    gamma_function_HLG_BT2100, )
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestOetf_PQ_BT2100',
    'TestOetf_inverse_PQ_BT2100',
    'TestEotf_PQ_BT2100',
    'TestEotf_inverse_PQ_BT2100',
    'TestOotf_PQ_BT2100',
    'TestOotf_inverse_PQ_BT2100',
    'TestGamma_function_HLG_BT2100',
    'TestOetf_HLG_BT2100',
    'TestOetf_inverse_HLG_BT2100',
    'TestEotf_HLG_BT2100_1',
    'TestEotf_HLG_BT2100_2',
    'TestEotf_inverse_HLG_BT2100_1',
    'TestEotf_inverse_HLG_BT2100_2',
    'TestOotf_HLG_BT2100_1',
    'TestOotf_HLG_BT2100_2',
    'TestOotf_inverse_HLG_BT2100_1',
    'TestOotf_inverse_HLG_BT2100_2',
]


class TestOetf_PQ_BT2100(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_PQ_BT2100` definition unit tests methods.
    """

    def test_oetf_PQ_BT2100(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_PQ_BT2100` definition.
        """

        self.assertAlmostEqual(
            oetf_PQ_BT2100(0.0), 0.000000730955903, places=7)

        self.assertAlmostEqual(
            oetf_PQ_BT2100(0.1), 0.724769816665726, places=7)

        self.assertAlmostEqual(
            oetf_PQ_BT2100(1.0), 0.999999934308041, places=7)

    def test_n_dimensional_oetf_PQ_BT2100(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_PQ_BT2100` definition n-dimensional arrays support.
        """

        E = 0.1
        E_p = oetf_PQ_BT2100(E)

        E = np.tile(E, 6)
        E_p = np.tile(E_p, 6)
        np.testing.assert_almost_equal(oetf_PQ_BT2100(E), E_p, decimal=7)

        E = np.reshape(E, (2, 3))
        E_p = np.reshape(E_p, (2, 3))
        np.testing.assert_almost_equal(oetf_PQ_BT2100(E), E_p, decimal=7)

        E = np.reshape(E, (2, 3, 1))
        E_p = np.reshape(E_p, (2, 3, 1))
        np.testing.assert_almost_equal(oetf_PQ_BT2100(E), E_p, decimal=7)

    def test_domain_range_scale_oetf_PQ_BT2100(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_PQ_BT2100` definition domain and range scale support.
        """

        E = 0.1
        E_p = oetf_PQ_BT2100(E)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    oetf_PQ_BT2100(E * factor), E_p * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_oetf_PQ_BT2100(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_PQ_BT2100` definition nan support.
        """

        oetf_PQ_BT2100(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestOetf_inverse_PQ_BT2100(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_inverse_PQ_BT2100` definition unit tests methods.
    """

    def test_oetf_inverse_PQ_BT2100(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_inverse_PQ_BT2100` definition.
        """

        self.assertAlmostEqual(
            oetf_inverse_PQ_BT2100(0.000000730955903), 0.0, places=7)

        self.assertAlmostEqual(
            oetf_inverse_PQ_BT2100(0.724769816665726), 0.1, places=7)

        self.assertAlmostEqual(
            oetf_inverse_PQ_BT2100(0.999999934308041), 1.0, places=7)

    def test_n_dimensional_oetf_inverse_PQ_BT2100(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_inverse_PQ_BT2100` definition n-dimensional arrays support.
        """

        E_p = 0.724769816665726
        E = oetf_inverse_PQ_BT2100(E_p)

        E_p = np.tile(E_p, 6)
        E = np.tile(E, 6)
        np.testing.assert_almost_equal(
            oetf_inverse_PQ_BT2100(E_p), E, decimal=7)

        E_p = np.reshape(E_p, (2, 3))
        E = np.reshape(E, (2, 3))
        np.testing.assert_almost_equal(
            oetf_inverse_PQ_BT2100(E_p), E, decimal=7)

        E_p = np.reshape(E_p, (2, 3, 1))
        E = np.reshape(E, (2, 3, 1))
        np.testing.assert_almost_equal(
            oetf_inverse_PQ_BT2100(E_p), E, decimal=7)

    def test_domain_range_scale_oetf_inverse_PQ_BT2100(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_inverse_PQ_BT2100` definition domain and range scale support.
        """

        E_p = 0.724769816665726
        E = oetf_inverse_PQ_BT2100(E_p)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    oetf_inverse_PQ_BT2100(E_p * factor),
                    E * factor,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_oetf_inverse_PQ_BT2100(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_inverse_PQ_BT2100` definition nan support.
        """

        oetf_inverse_PQ_BT2100(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestEotf_PQ_BT2100(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_PQ_BT2100` definition unit tests methods.
    """

    def test_eotf_PQ_BT2100(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_PQ_BT2100` definition.
        """

        self.assertAlmostEqual(eotf_PQ_BT2100(0.0), 0.0, places=7)

        self.assertAlmostEqual(
            eotf_PQ_BT2100(0.724769816665726), 779.98836083408537, places=7)

        self.assertAlmostEqual(eotf_PQ_BT2100(1.0), 10000.0, places=7)

    def test_n_dimensional_eotf_PQ_BT2100(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_PQ_BT2100` definition n-dimensional arrays support.
        """

        E_p = 0.724769816665726
        F_D = eotf_PQ_BT2100(E_p)

        E_p = np.tile(E_p, 6)
        F_D = np.tile(F_D, 6)
        np.testing.assert_almost_equal(eotf_PQ_BT2100(E_p), F_D, decimal=7)

        E_p = np.reshape(E_p, (2, 3))
        F_D = np.reshape(F_D, (2, 3))
        np.testing.assert_almost_equal(eotf_PQ_BT2100(E_p), F_D, decimal=7)

        E_p = np.reshape(E_p, (2, 3, 1))
        F_D = np.reshape(F_D, (2, 3, 1))
        np.testing.assert_almost_equal(eotf_PQ_BT2100(E_p), F_D, decimal=7)

    def test_domain_range_scale_eotf_PQ_BT2100(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_PQ_BT2100` definition domain and range scale support.
        """

        E_p = 0.724769816665726
        F_D = eotf_PQ_BT2100(E_p)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    eotf_PQ_BT2100(E_p * factor), F_D * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_eotf_PQ_BT2100(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_PQ_BT2100` definition nan support.
        """

        eotf_PQ_BT2100(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestEotf_inverse_PQ_BT2100(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_PQ_BT2100` definition unit tests methods.
    """

    def test_eotf_inverse_PQ_BT2100(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_PQ_BT2100` definition.
        """

        self.assertAlmostEqual(
            eotf_inverse_PQ_BT2100(0.0), 0.000000730955903, places=7)

        self.assertAlmostEqual(
            eotf_inverse_PQ_BT2100(779.98836083408537),
            0.724769816665726,
            places=7)

        self.assertAlmostEqual(eotf_inverse_PQ_BT2100(10000.0), 1.0, places=7)

    def test_n_dimensional_eotf_inverse_PQ_BT2100(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_PQ_BT2100` definition n-dimensional arrays support.
        """

        F_D = 779.98836083408537
        E_p = eotf_inverse_PQ_BT2100(F_D)

        F_D = np.tile(F_D, 6)
        E_p = np.tile(E_p, 6)
        np.testing.assert_almost_equal(
            eotf_inverse_PQ_BT2100(F_D), E_p, decimal=7)

        F_D = np.reshape(F_D, (2, 3))
        E_p = np.reshape(E_p, (2, 3))
        np.testing.assert_almost_equal(
            eotf_inverse_PQ_BT2100(F_D), E_p, decimal=7)

        F_D = np.reshape(F_D, (2, 3, 1))
        E_p = np.reshape(E_p, (2, 3, 1))
        np.testing.assert_almost_equal(
            eotf_inverse_PQ_BT2100(F_D), E_p, decimal=7)

    def test_domain_range_scale_eotf_inverse_PQ_BT2100(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_PQ_BT2100` definition domain and range scale support.
        """

        F_D = 779.98836083408537
        E_p = eotf_inverse_PQ_BT2100(F_D)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    eotf_inverse_PQ_BT2100(F_D * factor),
                    E_p * factor,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_eotf_inverse_PQ_BT2100(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_PQ_BT2100` definition nan support.
        """

        eotf_inverse_PQ_BT2100(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestOotf_PQ_BT2100(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_PQ_BT2100` definition unit tests methods.
    """

    def test_ootf_PQ_BT2100(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_PQ_BT2100` definition.
        """

        self.assertAlmostEqual(ootf_PQ_BT2100(0.0), 0.0, places=7)

        self.assertAlmostEqual(
            ootf_PQ_BT2100(0.1), 779.98836083411584, places=7)

        self.assertAlmostEqual(
            ootf_PQ_BT2100(1.0), 9999.993723673924300, places=7)

    def test_n_dimensional_ootf_PQ_BT2100(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_PQ_BT2100` definition n-dimensional arrays support.
        """

        E = 0.1
        F_D = ootf_PQ_BT2100(E)

        E = np.tile(E, 6)
        F_D = np.tile(F_D, 6)
        np.testing.assert_almost_equal(ootf_PQ_BT2100(E), F_D, decimal=7)

        E = np.reshape(E, (2, 3))
        F_D = np.reshape(F_D, (2, 3))
        np.testing.assert_almost_equal(ootf_PQ_BT2100(E), F_D, decimal=7)

        E = np.reshape(E, (2, 3, 1))
        F_D = np.reshape(F_D, (2, 3, 1))
        np.testing.assert_almost_equal(ootf_PQ_BT2100(E), F_D, decimal=7)

    def test_domain_range_scale_ootf_PQ_BT2100(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_PQ_BT2100` definition domain and range scale support.
        """

        E = 0.1
        F_D = ootf_PQ_BT2100(E)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    ootf_PQ_BT2100(E * factor), F_D * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_ootf_PQ_BT2100(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_PQ_BT2100` definition nan support.
        """

        ootf_PQ_BT2100(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestOotf_inverse_PQ_BT2100(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_PQ_BT2100` definition unit tests methods.
    """

    def test_ootf_inverse_PQ_BT2100(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_PQ_BT2100` definition.
        """

        self.assertAlmostEqual(ootf_inverse_PQ_BT2100(0.0), 0.0, places=7)

        self.assertAlmostEqual(
            ootf_inverse_PQ_BT2100(779.98836083411584), 0.1, places=7)

        self.assertAlmostEqual(
            ootf_inverse_PQ_BT2100(9999.993723673924300), 1.0, places=7)

    def test_n_dimensional_ootf_inverse_PQ_BT2100(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_PQ_BT2100` definition n-dimensional arrays support.
        """

        F_D = 779.98836083411584
        E = ootf_inverse_PQ_BT2100(F_D)

        F_D = np.tile(F_D, 6)
        E = np.tile(E, 6)
        np.testing.assert_almost_equal(
            ootf_inverse_PQ_BT2100(F_D), E, decimal=7)

        F_D = np.reshape(F_D, (2, 3))
        E = np.reshape(E, (2, 3))
        np.testing.assert_almost_equal(
            ootf_inverse_PQ_BT2100(F_D), E, decimal=7)

        F_D = np.reshape(F_D, (2, 3, 1))
        E = np.reshape(E, (2, 3, 1))
        np.testing.assert_almost_equal(
            ootf_inverse_PQ_BT2100(F_D), E, decimal=7)

    def test_domain_range_scale_ootf_inverse_PQ_BT2100(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_PQ_BT2100` definition domain and range scale support.
        """

        F_D = 779.98836083411584
        E = ootf_inverse_PQ_BT2100(F_D)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    ootf_inverse_PQ_BT2100(F_D * factor),
                    E * factor,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_ootf_inverse_PQ_BT2100(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_PQ_BT2100` definition nan support.
        """

        ootf_inverse_PQ_BT2100(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestGamma_function_HLG_BT2100(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
gamma_function_HLG_BT2100` definition unit tests methods.
    """

    def test_gamma_function_HLG_BT2100(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
gamma_function_HLG_BT2100` definition.
        """

        self.assertAlmostEqual(
            gamma_function_HLG_BT2100(1000.0), 1.2, places=7)

        self.assertAlmostEqual(
            gamma_function_HLG_BT2100(2000.0), 1.326432598178872, places=7)

        self.assertAlmostEqual(
            gamma_function_HLG_BT2100(4000.0), 1.452865196357744, places=7)

        self.assertAlmostEqual(
            gamma_function_HLG_BT2100(10000.0), 1.619999999999999, places=7)


class TestOetf_HLG_BT2100(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_HLG_BT2100` definition unit tests methods.
    """

    def test_oetf_HLG_BT2100(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_HLG_BT2100` definition.
        """

        self.assertAlmostEqual(oetf_HLG_BT2100(0.0), 0.0, places=7)

        self.assertAlmostEqual(
            oetf_HLG_BT2100(0.18 / 12), 0.212132034355964, places=7)

        self.assertAlmostEqual(
            oetf_HLG_BT2100(1.0), 0.999999995536569, places=7)

    def test_n_dimensional_oetf_HLG_BT2100(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_HLG_BT2100` definition n-dimensional arrays support.
        """

        E = 0.18 / 12
        E_p = oetf_HLG_BT2100(E)

        E = np.tile(E, 6)
        E_p = np.tile(E_p, 6)
        np.testing.assert_almost_equal(oetf_HLG_BT2100(E), E_p, decimal=7)

        E = np.reshape(E, (2, 3))
        E_p = np.reshape(E_p, (2, 3))
        np.testing.assert_almost_equal(oetf_HLG_BT2100(E), E_p, decimal=7)

        E = np.reshape(E, (2, 3, 1))
        E_p = np.reshape(E_p, (2, 3, 1))
        np.testing.assert_almost_equal(oetf_HLG_BT2100(E), E_p, decimal=7)

    def test_domain_range_scale_oetf_HLG_BT2100(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_HLG_BT2100` definition domain and range scale support.
        """

        E = 0.18 / 12
        E_p = oetf_HLG_BT2100(E)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    oetf_HLG_BT2100(E * factor), E_p * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_oetf_HLG_BT2100(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_HLG_BT2100` definition nan support.
        """

        oetf_HLG_BT2100(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestOetf_inverse_HLG_BT2100(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_inverse_HLG_BT2100` definition unit tests methods.
    """

    def test_oetf_inverse_HLG_BT2100(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_inverse_HLG_BT2100` definition.
        """

        self.assertAlmostEqual(oetf_inverse_HLG_BT2100(0.0), 0.0, places=7)

        self.assertAlmostEqual(
            oetf_inverse_HLG_BT2100(0.212132034355964), 0.18 / 12, places=7)

        self.assertAlmostEqual(
            oetf_inverse_HLG_BT2100(0.999999995536569), 1.0, places=7)

    def test_n_dimensional_oetf_inverse_HLG_BT2100(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_inverse_HLG_BT2100` definition n-dimensional arrays support.
        """

        E_p = 0.212132034355964
        E = oetf_inverse_HLG_BT2100(E_p)

        E_p = np.tile(E_p, 6)
        E = np.tile(E, 6)
        np.testing.assert_almost_equal(
            oetf_inverse_HLG_BT2100(E_p), E, decimal=7)

        E_p = np.reshape(E_p, (2, 3))
        E = np.reshape(E, (2, 3))
        np.testing.assert_almost_equal(
            oetf_inverse_HLG_BT2100(E_p), E, decimal=7)

        E_p = np.reshape(E_p, (2, 3, 1))
        E = np.reshape(E, (2, 3, 1))
        np.testing.assert_almost_equal(
            oetf_inverse_HLG_BT2100(E_p), E, decimal=7)

    def test_domain_range_scale_oetf_inverse_HLG_BT2100(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_inverse_HLG_BT2100` definition domain and range scale support.
        """

        E_p = 0.212132034355964
        E = oetf_inverse_HLG_BT2100(E_p)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    oetf_inverse_HLG_BT2100(E_p * factor),
                    E * factor,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_oetf_inverse_HLG_BT2100(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_inverse_HLG_BT2100` definition nan support.
        """

        oetf_inverse_HLG_BT2100(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestEotf_HLG_BT2100_1(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_HLG_BT2100_1` definition unit tests methods.
    """

    def test_eotf_HLG_BT2100_1(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_HLG_BT2100_1` definition.
        """

        self.assertAlmostEqual(eotf_HLG_BT2100_1(0.0), 0.0, places=7)

        self.assertAlmostEqual(
            eotf_HLG_BT2100_1(0.212132034355964), 6.476039825649814, places=7)

        self.assertAlmostEqual(
            eotf_HLG_BT2100_1(1.0), 1000.000032321769100, places=7)

        self.assertAlmostEqual(
            eotf_HLG_BT2100_1(0.212132034355964, 0.001, 10000, 1.4),
            27.96039175299561,
            places=7)

    def test_n_dimensional_eotf_HLG_BT2100_1(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_HLG_BT2100_1` definition n-dimensional arrays support.
        """

        E_p = 0.212132034355964
        F_D = eotf_HLG_BT2100_1(E_p)

        E_p = np.tile(E_p, 6)
        F_D = np.tile(F_D, 6)
        np.testing.assert_almost_equal(eotf_HLG_BT2100_1(E_p), F_D, decimal=7)

        E_p = np.reshape(E_p, (2, 3))
        F_D = np.reshape(F_D, (2, 3))
        np.testing.assert_almost_equal(eotf_HLG_BT2100_1(E_p), F_D, decimal=7)

        E_p = np.reshape(E_p, (2, 3, 1))
        F_D = np.reshape(F_D, (2, 3, 1))
        np.testing.assert_almost_equal(eotf_HLG_BT2100_1(E_p), F_D, decimal=7)

        E_p = np.reshape(E_p, (6, 1))
        F_D = np.reshape(F_D, (6, 1))
        np.testing.assert_almost_equal(eotf_HLG_BT2100_1(E_p), F_D, decimal=7)

        E_p = np.array([0.25, 0.50, 0.75])
        F_D = np.array([12.49759413, 49.99037650, 158.94693786])
        np.testing.assert_almost_equal(eotf_HLG_BT2100_1(E_p), F_D, decimal=7)

        E_p = np.tile(E_p, (6, 1))
        F_D = np.tile(F_D, (6, 1))
        np.testing.assert_almost_equal(eotf_HLG_BT2100_1(E_p), F_D, decimal=7)

        E_p = np.reshape(E_p, (2, 3, 3))
        F_D = np.reshape(F_D, (2, 3, 3))
        np.testing.assert_almost_equal(eotf_HLG_BT2100_1(E_p), F_D, decimal=7)

    def test_domain_range_scale_eotf_HLG_BT2100_1(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_HLG_BT2100_1` definition domain and range scale support.
        """

        E_p = 0.212132034355964
        F_D = eotf_HLG_BT2100_1(E_p)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    eotf_HLG_BT2100_1(E_p * factor), F_D * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_eotf_HLG_BT2100_1(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_HLG_BT2100_1` definition nan support.
        """

        eotf_HLG_BT2100_1(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestEotf_HLG_BT2100_2(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_HLG_BT2100_2` definition unit tests methods.
    """

    def test_eotf_HLG_BT2100_2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_HLG_BT2100_2` definition.
        """

        self.assertAlmostEqual(eotf_HLG_BT2100_2(0.0), 0.0, places=7)

        self.assertAlmostEqual(
            eotf_HLG_BT2100_2(0.212132034355964), 6.476039825649814, places=7)

        self.assertAlmostEqual(
            eotf_HLG_BT2100_2(1.0), 1000.000032321769100, places=7)

        self.assertAlmostEqual(
            eotf_HLG_BT2100_2(0.212132034355964, 0.001, 10000, 1.4),
            29.581261576946076,
            places=7)

    def test_n_dimensional_eotf_HLG_BT2100_2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_HLG_BT2100_2` definition n-dimensional arrays support.
        """

        E_p = 0.212132034355964
        F_D = eotf_HLG_BT2100_2(E_p)

        E_p = np.tile(E_p, 6)
        F_D = np.tile(F_D, 6)
        np.testing.assert_almost_equal(eotf_HLG_BT2100_2(E_p), F_D, decimal=7)

        E_p = np.reshape(E_p, (2, 3))
        F_D = np.reshape(F_D, (2, 3))
        np.testing.assert_almost_equal(eotf_HLG_BT2100_2(E_p), F_D, decimal=7)

        E_p = np.reshape(E_p, (2, 3, 1))
        F_D = np.reshape(F_D, (2, 3, 1))
        np.testing.assert_almost_equal(eotf_HLG_BT2100_2(E_p), F_D, decimal=7)

        E_p = np.reshape(E_p, (6, 1))
        F_D = np.reshape(F_D, (6, 1))
        np.testing.assert_almost_equal(eotf_HLG_BT2100_2(E_p), F_D, decimal=7)

        E_p = np.array([0.25, 0.50, 0.75])
        F_D = np.array([12.49759413, 49.99037650, 158.94693786])
        np.testing.assert_almost_equal(eotf_HLG_BT2100_2(E_p), F_D, decimal=7)

        E_p = np.tile(E_p, (6, 1))
        F_D = np.tile(F_D, (6, 1))
        np.testing.assert_almost_equal(eotf_HLG_BT2100_2(E_p), F_D, decimal=7)

        E_p = np.reshape(E_p, (2, 3, 3))
        F_D = np.reshape(F_D, (2, 3, 3))
        np.testing.assert_almost_equal(eotf_HLG_BT2100_2(E_p), F_D, decimal=7)

    def test_domain_range_scale_eotf_HLG_BT2100_2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_HLG_BT2100_2` definition domain and range scale support.
        """

        E_p = 0.212132034355964
        F_D = eotf_HLG_BT2100_2(E_p)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    eotf_HLG_BT2100_2(E_p * factor), F_D * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_eotf_HLG_BT2100_2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_HLG_BT2100_2` definition nan support.
        """

        eotf_HLG_BT2100_2(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestEotf_inverse_HLG_BT2100_1(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_HLG_BT2100_1` definition unit tests methods.
    """

    def test_eotf_inverse_HLG_BT2100_1(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_HLG_BT2100_1` definition.
        """

        self.assertAlmostEqual(eotf_inverse_HLG_BT2100_1(0.0), 0.0, places=7)

        self.assertAlmostEqual(
            eotf_inverse_HLG_BT2100_1(6.476039825649814),
            0.212132034355964,
            places=7)

        self.assertAlmostEqual(
            eotf_inverse_HLG_BT2100_1(1000.000032321769100), 1.0, places=7)

        self.assertAlmostEqual(
            eotf_inverse_HLG_BT2100_1(27.96039175299561, 0.001, 10000, 1.4),
            0.212132034355964,
            places=7)

    def test_n_dimensional_eotf_inverse_HLG_BT2100_1(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_HLG_BT2100_1` definition n-dimensional arrays support.
        """

        F_D = 6.476039825649814
        E_p = eotf_inverse_HLG_BT2100_1(F_D)

        F_D = np.tile(F_D, 6)
        E_p = np.tile(E_p, 6)
        np.testing.assert_almost_equal(
            eotf_inverse_HLG_BT2100_1(F_D), E_p, decimal=7)

        F_D = np.reshape(F_D, (2, 3))
        E_p = np.reshape(E_p, (2, 3))
        np.testing.assert_almost_equal(
            eotf_inverse_HLG_BT2100_1(F_D), E_p, decimal=7)

        F_D = np.reshape(F_D, (2, 3, 1))
        E_p = np.reshape(E_p, (2, 3, 1))
        np.testing.assert_almost_equal(
            eotf_inverse_HLG_BT2100_1(F_D), E_p, decimal=7)

        F_D = np.reshape(F_D, (6, 1))
        E_p = np.reshape(E_p, (6, 1))
        np.testing.assert_almost_equal(
            eotf_inverse_HLG_BT2100_1(F_D), E_p, decimal=7)

        F_D = np.array([12.49759413, 49.99037650, 158.94693786])
        E_p = np.array([0.25, 0.50, 0.75])
        np.testing.assert_almost_equal(
            eotf_inverse_HLG_BT2100_1(F_D), E_p, decimal=7)

        F_D = np.tile(F_D, (6, 1))
        E_p = np.tile(E_p, (6, 1))
        np.testing.assert_almost_equal(
            eotf_inverse_HLG_BT2100_1(F_D), E_p, decimal=7)

        F_D = np.reshape(F_D, (2, 3, 3))
        E_p = np.reshape(E_p, (2, 3, 3))
        np.testing.assert_almost_equal(
            eotf_inverse_HLG_BT2100_1(F_D), E_p, decimal=7)

    def test_domain_range_scale_eotf_inverse_HLG_BT2100_1(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_HLG_BT2100_1` definition domain and range scale support.
        """

        F_D = 6.476039825649814
        E_p = eotf_inverse_HLG_BT2100_1(F_D)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    eotf_inverse_HLG_BT2100_1(F_D * factor),
                    E_p * factor,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_eotf_inverse_HLG_BT2100_1(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_HLG_BT2100_1` definition nan support.
        """

        eotf_inverse_HLG_BT2100_1(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestEotf_inverse_HLG_BT2100_2(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_HLG_BT2100_2` definition unit tests methods.
    """

    def test_eotf_inverse_HLG_BT2100_2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_HLG_BT2100_2` definition.
        """

        self.assertAlmostEqual(eotf_inverse_HLG_BT2100_2(0.0), 0.0, places=7)

        self.assertAlmostEqual(
            eotf_inverse_HLG_BT2100_2(6.476039825649814),
            0.212132034355964,
            places=7)

        self.assertAlmostEqual(
            eotf_inverse_HLG_BT2100_2(1000.000032321769100), 1.0, places=7)

        self.assertAlmostEqual(
            eotf_inverse_HLG_BT2100_2(29.581261576946076, 0.001, 10000, 1.4),
            0.212132034355964,
            places=7)

    def test_n_dimensional_eotf_inverse_HLG_BT2100_2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_HLG_BT2100_2` definition n-dimensional arrays support.
        """

        F_D = 6.476039825649814
        E_p = eotf_inverse_HLG_BT2100_2(F_D)

        F_D = np.tile(F_D, 6)
        E_p = np.tile(E_p, 6)
        np.testing.assert_almost_equal(
            eotf_inverse_HLG_BT2100_2(F_D), E_p, decimal=7)

        F_D = np.reshape(F_D, (2, 3))
        E_p = np.reshape(E_p, (2, 3))
        np.testing.assert_almost_equal(
            eotf_inverse_HLG_BT2100_2(F_D), E_p, decimal=7)

        F_D = np.reshape(F_D, (2, 3, 1))
        E_p = np.reshape(E_p, (2, 3, 1))
        np.testing.assert_almost_equal(
            eotf_inverse_HLG_BT2100_2(F_D), E_p, decimal=7)

        F_D = np.reshape(F_D, (6, 1))
        E_p = np.reshape(E_p, (6, 1))
        np.testing.assert_almost_equal(
            eotf_inverse_HLG_BT2100_2(F_D), E_p, decimal=7)

        F_D = np.array([12.49759413, 49.99037650, 158.94693786])
        E_p = np.array([0.25, 0.50, 0.75])
        np.testing.assert_almost_equal(
            eotf_inverse_HLG_BT2100_2(F_D), E_p, decimal=7)

        F_D = np.tile(F_D, (6, 1))
        E_p = np.tile(E_p, (6, 1))
        np.testing.assert_almost_equal(
            eotf_inverse_HLG_BT2100_2(F_D), E_p, decimal=7)

        F_D = np.reshape(F_D, (2, 3, 3))
        E_p = np.reshape(E_p, (2, 3, 3))
        np.testing.assert_almost_equal(
            eotf_inverse_HLG_BT2100_2(F_D), E_p, decimal=7)

    def test_domain_range_scale_eotf_inverse_HLG_BT2100_2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_HLG_BT2100_2` definition domain and range scale support.
        """

        F_D = 6.476039825649814
        E_p = eotf_inverse_HLG_BT2100_2(F_D)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    eotf_inverse_HLG_BT2100_2(F_D * factor),
                    E_p * factor,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_eotf_inverse_HLG_BT2100_2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_HLG_BT2100_2` definition nan support.
        """

        eotf_inverse_HLG_BT2100_2(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestOotf_HLG_BT2100_1(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_HLG_BT2100_1` definition unit tests methods.
    """

    def test_ootf_HLG_BT2100_1(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_HLG_BT2100_1` definition.
        """

        self.assertAlmostEqual(ootf_HLG_BT2100_1(0.0), 0.0, places=7)

        self.assertAlmostEqual(
            ootf_HLG_BT2100_1(0.1), 63.095734448019336, places=7)

        self.assertAlmostEqual(ootf_HLG_BT2100_1(1.0), 1000.0, places=7)

        self.assertAlmostEqual(
            ootf_HLG_BT2100_1(0.1, 0.001, 10000, 1.4),
            398.108130742780300,
            places=7)

        a = np.array(
            [[45.884942278760597, 0.000000000000000, -45.884942278760597], [
                -63.095734448019336, -63.095734448019336, -63.095734448019336
            ], [63.095734448019336, 63.095734448019336, 63.095734448019336],
             [51.320396090100672, -51.320396090100672, 51.320396090100672]], )
        np.testing.assert_almost_equal(
            ootf_HLG_BT2100_1(
                np.array([[0.1, 0.0, -0.1], [-0.1, -0.1, -0.1],
                          [0.1, 0.1, 0.1], [0.1, -0.1, 0.1]])),
            a,
            decimal=7)

    def test_n_dimensional_ootf_HLG_BT2100_1(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_HLG_BT2100_1` definition n-dimensional arrays support.
        """

        E = 0.1
        F_D = ootf_HLG_BT2100_1(E)

        E = np.tile(E, 6)
        F_D = np.tile(F_D, 6)
        np.testing.assert_almost_equal(ootf_HLG_BT2100_1(E), F_D, decimal=7)

        E = np.reshape(E, (2, 3))
        F_D = np.reshape(F_D, (2, 3))
        np.testing.assert_almost_equal(ootf_HLG_BT2100_1(E), F_D, decimal=7)

        E = np.reshape(E, (2, 3, 1))
        F_D = np.reshape(F_D, (2, 3, 1))
        np.testing.assert_almost_equal(ootf_HLG_BT2100_1(E), F_D, decimal=7)

        E = np.reshape(E, (6, 1))
        F_D = np.reshape(F_D, (6, 1))
        np.testing.assert_almost_equal(ootf_HLG_BT2100_1(E), F_D, decimal=7)

        E = np.array([0.25, 0.50, 0.75])
        F_D = np.array([213.01897444, 426.03794887, 639.05692331])
        np.testing.assert_almost_equal(ootf_HLG_BT2100_1(E), F_D, decimal=7)

        E = np.tile(E, (6, 1))
        F_D = np.tile(F_D, (6, 1))
        np.testing.assert_almost_equal(ootf_HLG_BT2100_1(E), F_D, decimal=7)

        E = np.reshape(E, (2, 3, 3))
        F_D = np.reshape(F_D, (2, 3, 3))
        np.testing.assert_almost_equal(ootf_HLG_BT2100_1(E), F_D, decimal=7)

    def test_domain_range_scale_ootf_HLG_BT2100_1(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_HLG_BT2100_1` definition domain and range scale support.
        """

        E = 0.1
        F_D = ootf_HLG_BT2100_1(E)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    ootf_HLG_BT2100_1(E * factor), F_D * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_ootf_HLG_BT2100_1(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_HLG_BT2100_1` definition nan support.
        """

        ootf_HLG_BT2100_1(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestOotf_HLG_BT2100_2(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_HLG_BT2100_2` definition unit tests methods.
    """

    def test_ootf_HLG_BT2100_2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_HLG_BT2100_2` definition.
        """

        self.assertAlmostEqual(ootf_HLG_BT2100_2(0.0), 0.0, places=7)

        self.assertAlmostEqual(
            ootf_HLG_BT2100_2(0.1), 63.095734448019336, places=7)

        self.assertAlmostEqual(ootf_HLG_BT2100_2(1.0), 1000.0, places=7)

        self.assertAlmostEqual(
            ootf_HLG_BT2100_2(0.1, 10000, 1.4), 398.107170553497380, places=7)

        a = np.array([
            [45.884942278760597, 0.000000000000000, -45.884942278760597],
            [-63.095734448019336, -63.095734448019336, -63.095734448019336],
            [63.095734448019336, 63.095734448019336, 63.095734448019336],
            [51.320396090100672, -51.320396090100672, 51.320396090100672],
        ], )
        np.testing.assert_almost_equal(
            ootf_HLG_BT2100_2(
                np.array([[0.1, 0.0, -0.1], [-0.1, -0.1, -0.1],
                          [0.1, 0.1, 0.1], [0.1, -0.1, 0.1]])),
            a,
            decimal=7)

    def test_n_dimensional_ootf_HLG_BT2100_2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_HLG_BT2100_2` definition n-dimensional arrays support.
        """

        E = 0.1
        F_D = ootf_HLG_BT2100_2(E)

        E = np.tile(E, 6)
        F_D = np.tile(F_D, 6)
        np.testing.assert_almost_equal(ootf_HLG_BT2100_2(E), F_D, decimal=7)

        E = np.reshape(E, (2, 3))
        F_D = np.reshape(F_D, (2, 3))
        np.testing.assert_almost_equal(ootf_HLG_BT2100_2(E), F_D, decimal=7)

        E = np.reshape(E, (2, 3, 1))
        F_D = np.reshape(F_D, (2, 3, 1))
        np.testing.assert_almost_equal(ootf_HLG_BT2100_2(E), F_D, decimal=7)

        E = np.reshape(E, (6, 1))
        F_D = np.reshape(F_D, (6, 1))
        np.testing.assert_almost_equal(ootf_HLG_BT2100_2(E), F_D, decimal=7)

        E = np.array([0.25, 0.50, 0.75])
        F_D = np.array([213.01897444, 426.03794887, 639.05692331])
        np.testing.assert_almost_equal(ootf_HLG_BT2100_2(E), F_D, decimal=7)

        E = np.tile(E, (6, 1))
        F_D = np.tile(F_D, (6, 1))
        np.testing.assert_almost_equal(ootf_HLG_BT2100_2(E), F_D, decimal=7)

        E = np.reshape(E, (2, 3, 3))
        F_D = np.reshape(F_D, (2, 3, 3))
        np.testing.assert_almost_equal(ootf_HLG_BT2100_2(E), F_D, decimal=7)

    def test_domain_range_scale_ootf_HLG_BT2100_2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_HLG_BT2100_2` definition domain and range scale support.
        """

        E = 0.1
        F_D = ootf_HLG_BT2100_1(E)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    ootf_HLG_BT2100_1(E * factor), F_D * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_ootf_HLG_BT2100_1(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_HLG_BT2100_1` definition nan support.
        """

        ootf_HLG_BT2100_1(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestOotf_inverse_HLG_BT2100_1(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_HLG_BT2100_1` definition unit tests methods.
    """

    def test_ootf_inverse_HLG_BT2100_1(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_HLG_BT2100_1` definition.
        """

        self.assertAlmostEqual(ootf_inverse_HLG_BT2100_1(0.0), 0.0, places=7)

        self.assertAlmostEqual(
            ootf_inverse_HLG_BT2100_1(63.095734448019336), 0.1, places=7)

        self.assertAlmostEqual(
            ootf_inverse_HLG_BT2100_1(1000.0), 1.0, places=7)

        self.assertAlmostEqual(
            ootf_inverse_HLG_BT2100_1(398.108130742780300, 0.001, 10000, 1.4),
            0.1,
            places=7)

        a = np.array(
            [[45.884942278760597, 0.000000000000000, -45.884942278760597], [
                -63.095734448019336, -63.095734448019336, -63.095734448019336
            ], [63.095734448019336, 63.095734448019336, 63.095734448019336],
             [51.320396090100672, -51.320396090100672, 51.320396090100672]])
        np.testing.assert_almost_equal(
            ootf_inverse_HLG_BT2100_1(a),
            np.array([[0.1, 0.0, -0.1], [-0.1, -0.1, -0.1], [0.1, 0.1, 0.1],
                      [0.1, -0.1, 0.1]]),
            decimal=7)

    def test_n_dimensional_ootf_inverse_HLG_BT2100_1(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_HLG_BT2100_1` definition n-dimensional arrays support.
        """

        F_D = 63.095734448019336
        E = ootf_inverse_HLG_BT2100_1(F_D)

        F_D = np.tile(F_D, 6)
        E = np.tile(E, 6)
        np.testing.assert_almost_equal(
            ootf_inverse_HLG_BT2100_1(F_D), E, decimal=7)

        F_D = np.reshape(F_D, (2, 3))
        E = np.reshape(E, (2, 3))
        np.testing.assert_almost_equal(
            ootf_inverse_HLG_BT2100_1(F_D), E, decimal=7)

        F_D = np.reshape(F_D, (2, 3, 1))
        E = np.reshape(E, (2, 3, 1))
        np.testing.assert_almost_equal(
            ootf_inverse_HLG_BT2100_1(F_D), E, decimal=7)

        F_D = np.reshape(F_D, (6, 1))
        E = np.reshape(E, (6, 1))
        np.testing.assert_almost_equal(
            ootf_inverse_HLG_BT2100_1(F_D), E, decimal=7)

        F_D = np.array([213.01897444, 426.03794887, 639.05692331])
        E = np.array([0.25, 0.50, 0.75])
        np.testing.assert_almost_equal(
            ootf_inverse_HLG_BT2100_1(F_D), E, decimal=7)

        F_D = np.tile(F_D, (6, 1))
        E = np.tile(E, (6, 1))
        np.testing.assert_almost_equal(
            ootf_inverse_HLG_BT2100_1(F_D), E, decimal=7)

        F_D = np.reshape(F_D, (2, 3, 3))
        E = np.reshape(E, (2, 3, 3))
        np.testing.assert_almost_equal(
            ootf_inverse_HLG_BT2100_1(F_D), E, decimal=7)

    def test_domain_range_scale_ootf_inverse_HLG_BT2100_1(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_HLG_BT2100_1` definition domain and range scale support.
        """

        F_D = 63.095734448019336
        E = ootf_inverse_HLG_BT2100_1(F_D)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    ootf_inverse_HLG_BT2100_1(F_D * factor),
                    E * factor,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_ootf_inverse_HLG_BT2100_1(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_HLG_BT2100_1` definition nan support.
        """

        ootf_inverse_HLG_BT2100_1(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestOotf_inverse_HLG_BT2100_2(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_HLG_BT2100_2` definition unit tests methods.
    """

    def test_ootf_inverse_HLG_BT2100_2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_HLG_BT2100_2` definition.
        """

        self.assertAlmostEqual(ootf_inverse_HLG_BT2100_2(0.0), 0.0, places=7)

        self.assertAlmostEqual(
            ootf_inverse_HLG_BT2100_2(63.095734448019336), 0.1, places=7)

        self.assertAlmostEqual(
            ootf_inverse_HLG_BT2100_2(1000.0), 1.0, places=7)

        self.assertAlmostEqual(
            ootf_inverse_HLG_BT2100_2(398.107170553497380, 10000, 1.4),
            0.1,
            places=7)

        a = np.array(
            [[45.884942278760597, 0.000000000000000, -45.884942278760597], [
                -63.095734448019336, -63.095734448019336, -63.095734448019336
            ], [63.095734448019336, 63.095734448019336, 63.095734448019336],
             [51.320396090100672, -51.320396090100672, 51.320396090100672]])
        np.testing.assert_almost_equal(
            ootf_inverse_HLG_BT2100_2(a),
            np.array([[0.1, 0.0, -0.1], [-0.1, -0.1, -0.1], [0.1, 0.1, 0.1],
                      [0.1, -0.1, 0.1]]),
            decimal=7)

    def test_n_dimensional_ootf_inverse_HLG_BT2100_2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_HLG_BT2100_2` definition n-dimensional arrays support.
        """

        F_D = 63.095734448019336
        E = ootf_inverse_HLG_BT2100_2(F_D)

        F_D = np.tile(F_D, 6)
        E = np.tile(E, 6)
        np.testing.assert_almost_equal(
            ootf_inverse_HLG_BT2100_2(F_D), E, decimal=7)

        F_D = np.reshape(F_D, (2, 3))
        E = np.reshape(E, (2, 3))
        np.testing.assert_almost_equal(
            ootf_inverse_HLG_BT2100_2(F_D), E, decimal=7)

        F_D = np.reshape(F_D, (2, 3, 1))
        E = np.reshape(E, (2, 3, 1))
        np.testing.assert_almost_equal(
            ootf_inverse_HLG_BT2100_2(F_D), E, decimal=7)

        F_D = np.reshape(F_D, (6, 1))
        E = np.reshape(E, (6, 1))
        np.testing.assert_almost_equal(
            ootf_inverse_HLG_BT2100_2(F_D), E, decimal=7)

        F_D = np.array([213.01897444, 426.03794887, 639.05692331])
        E = np.array([0.25, 0.50, 0.75])
        np.testing.assert_almost_equal(
            ootf_inverse_HLG_BT2100_2(F_D), E, decimal=7)

        F_D = np.tile(F_D, (6, 1))
        E = np.tile(E, (6, 1))
        np.testing.assert_almost_equal(
            ootf_inverse_HLG_BT2100_2(F_D), E, decimal=7)

        F_D = np.reshape(F_D, (2, 3, 3))
        E = np.reshape(E, (2, 3, 3))
        np.testing.assert_almost_equal(
            ootf_inverse_HLG_BT2100_2(F_D), E, decimal=7)

    def test_domain_range_scale_ootf_inverse_HLG_BT2100_2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_HLG_BT2100_2` definition domain and range scale support.
        """

        F_D = 63.095734448019336
        E = ootf_inverse_HLG_BT2100_2(F_D)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    ootf_inverse_HLG_BT2100_2(F_D * factor),
                    E * factor,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_ootf_inverse_HLG_BT2100_2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_HLG_BT2100_2` definition nan support.
        """

        ootf_inverse_HLG_BT2100_2(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


if __name__ == '__main__':
    unittest.main()
