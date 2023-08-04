"""
Define the unit tests for the
:mod:`colour.models.rgb.transfer_functions.itur_bt_2100` module.
"""

import numpy as np
import unittest

from colour.models.rgb.transfer_functions import (
    oetf_BT2100_PQ,
    oetf_inverse_BT2100_PQ,
    eotf_BT2100_PQ,
    eotf_inverse_BT2100_PQ,
    ootf_BT2100_PQ,
    ootf_inverse_BT2100_PQ,
    oetf_BT2100_HLG,
    oetf_inverse_BT2100_HLG,
)
from colour.models.rgb.transfer_functions.itur_bt_2100 import (
    eotf_BT2100_HLG_1,
    eotf_BT2100_HLG_2,
    eotf_inverse_BT2100_HLG_1,
    eotf_inverse_BT2100_HLG_2,
    ootf_BT2100_HLG_1,
    ootf_BT2100_HLG_2,
    ootf_inverse_BT2100_HLG_1,
    ootf_inverse_BT2100_HLG_2,
)
from colour.models.rgb.transfer_functions.itur_bt_2100 import (
    gamma_function_BT2100_HLG,
)
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestOetf_BT2100_PQ",
    "TestOetf_inverse_BT2100_PQ",
    "TestEotf_BT2100_PQ",
    "TestEotf_inverse_BT2100_PQ",
    "TestOotf_BT2100_PQ",
    "TestOotf_inverse_BT2100_PQ",
    "TestGamma_function_BT2100_HLG",
    "TestOetf_BT2100_HLG",
    "TestOetf_inverse_BT2100_HLG",
    "TestEotf_BT2100_HLG_1",
    "TestEotf_BT2100_HLG_2",
    "TestEotf_inverse_BT2100_HLG_1",
    "TestEotf_inverse_BT2100_HLG_2",
    "TestOotf_BT2100_HLG_1",
    "TestOotf_BT2100_HLG_2",
    "TestOotf_inverse_BT2100_HLG_1",
    "TestOotf_inverse_BT2100_HLG_2",
]


class TestOetf_BT2100_PQ(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_BT2100_PQ` definition unit tests methods.
    """

    def test_oetf_BT2100_PQ(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_BT2100_PQ` definition.
        """

        self.assertAlmostEqual(
            oetf_BT2100_PQ(0.0), 0.000000730955903, places=7
        )

        self.assertAlmostEqual(
            oetf_BT2100_PQ(0.1), 0.724769816665726, places=7
        )

        self.assertAlmostEqual(
            oetf_BT2100_PQ(1.0), 0.999999934308041, places=7
        )

    def test_n_dimensional_oetf_BT2100_PQ(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_BT2100_PQ` definition n-dimensional arrays support.
        """

        E = 0.1
        E_p = oetf_BT2100_PQ(E)

        E = np.tile(E, 6)
        E_p = np.tile(E_p, 6)
        np.testing.assert_array_almost_equal(oetf_BT2100_PQ(E), E_p, decimal=7)

        E = np.reshape(E, (2, 3))
        E_p = np.reshape(E_p, (2, 3))
        np.testing.assert_array_almost_equal(oetf_BT2100_PQ(E), E_p, decimal=7)

        E = np.reshape(E, (2, 3, 1))
        E_p = np.reshape(E_p, (2, 3, 1))
        np.testing.assert_array_almost_equal(oetf_BT2100_PQ(E), E_p, decimal=7)

    def test_domain_range_scale_oetf_BT2100_PQ(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_BT2100_PQ` definition domain and range scale support.
        """

        E = 0.1
        E_p = oetf_BT2100_PQ(E)

        d_r = (("reference", 1), ("1", 1), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_array_almost_equal(
                    oetf_BT2100_PQ(E * factor), E_p * factor, decimal=7
                )

    @ignore_numpy_errors
    def test_nan_oetf_BT2100_PQ(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_BT2100_PQ` definition nan support.
        """

        oetf_BT2100_PQ(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestOetf_inverse_BT2100_PQ(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_inverse_BT2100_PQ` definition unit tests methods.
    """

    def test_oetf_inverse_BT2100_PQ(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_inverse_BT2100_PQ` definition.
        """

        self.assertAlmostEqual(
            oetf_inverse_BT2100_PQ(0.000000730955903), 0.0, places=7
        )

        self.assertAlmostEqual(
            oetf_inverse_BT2100_PQ(0.724769816665726), 0.1, places=7
        )

        self.assertAlmostEqual(
            oetf_inverse_BT2100_PQ(0.999999934308041), 1.0, places=7
        )

    def test_n_dimensional_oetf_inverse_BT2100_PQ(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_inverse_BT2100_PQ` definition n-dimensional arrays support.
        """

        E_p = 0.724769816665726
        E = oetf_inverse_BT2100_PQ(E_p)

        E_p = np.tile(E_p, 6)
        E = np.tile(E, 6)
        np.testing.assert_array_almost_equal(
            oetf_inverse_BT2100_PQ(E_p), E, decimal=7
        )

        E_p = np.reshape(E_p, (2, 3))
        E = np.reshape(E, (2, 3))
        np.testing.assert_array_almost_equal(
            oetf_inverse_BT2100_PQ(E_p), E, decimal=7
        )

        E_p = np.reshape(E_p, (2, 3, 1))
        E = np.reshape(E, (2, 3, 1))
        np.testing.assert_array_almost_equal(
            oetf_inverse_BT2100_PQ(E_p), E, decimal=7
        )

    def test_domain_range_scale_oetf_inverse_BT2100_PQ(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_inverse_BT2100_PQ` definition domain and range scale support.
        """

        E_p = 0.724769816665726
        E = oetf_inverse_BT2100_PQ(E_p)

        d_r = (("reference", 1), ("1", 1), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_array_almost_equal(
                    oetf_inverse_BT2100_PQ(E_p * factor), E * factor, decimal=7
                )

    @ignore_numpy_errors
    def test_nan_oetf_inverse_BT2100_PQ(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_inverse_BT2100_PQ` definition nan support.
        """

        oetf_inverse_BT2100_PQ(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestEotf_BT2100_PQ(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_BT2100_PQ` definition unit tests methods.
    """

    def test_eotf_BT2100_PQ(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_BT2100_PQ` definition.
        """

        self.assertAlmostEqual(eotf_BT2100_PQ(0.0), 0.0, places=7)

        self.assertAlmostEqual(
            eotf_BT2100_PQ(0.724769816665726), 779.98836083408537, places=7
        )

        self.assertAlmostEqual(eotf_BT2100_PQ(1.0), 10000.0, places=7)

    def test_n_dimensional_eotf_BT2100_PQ(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_BT2100_PQ` definition n-dimensional arrays support.
        """

        E_p = 0.724769816665726
        F_D = eotf_BT2100_PQ(E_p)

        E_p = np.tile(E_p, 6)
        F_D = np.tile(F_D, 6)
        np.testing.assert_array_almost_equal(
            eotf_BT2100_PQ(E_p), F_D, decimal=7
        )

        E_p = np.reshape(E_p, (2, 3))
        F_D = np.reshape(F_D, (2, 3))
        np.testing.assert_array_almost_equal(
            eotf_BT2100_PQ(E_p), F_D, decimal=7
        )

        E_p = np.reshape(E_p, (2, 3, 1))
        F_D = np.reshape(F_D, (2, 3, 1))
        np.testing.assert_array_almost_equal(
            eotf_BT2100_PQ(E_p), F_D, decimal=7
        )

    def test_domain_range_scale_eotf_BT2100_PQ(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_BT2100_PQ` definition domain and range scale support.
        """

        E_p = 0.724769816665726
        F_D = eotf_BT2100_PQ(E_p)

        d_r = (("reference", 1), ("1", 1), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_array_almost_equal(
                    eotf_BT2100_PQ(E_p * factor), F_D * factor, decimal=7
                )

    @ignore_numpy_errors
    def test_nan_eotf_BT2100_PQ(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_BT2100_PQ` definition nan support.
        """

        eotf_BT2100_PQ(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestEotf_inverse_BT2100_PQ(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_BT2100_PQ` definition unit tests methods.
    """

    def test_eotf_inverse_BT2100_PQ(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_BT2100_PQ` definition.
        """

        self.assertAlmostEqual(
            eotf_inverse_BT2100_PQ(0.0), 0.000000730955903, places=7
        )

        self.assertAlmostEqual(
            eotf_inverse_BT2100_PQ(779.98836083408537),
            0.724769816665726,
            places=7,
        )

        self.assertAlmostEqual(eotf_inverse_BT2100_PQ(10000.0), 1.0, places=7)

    def test_n_dimensional_eotf_inverse_BT2100_PQ(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_BT2100_PQ` definition n-dimensional arrays support.
        """

        F_D = 779.98836083408537
        E_p = eotf_inverse_BT2100_PQ(F_D)

        F_D = np.tile(F_D, 6)
        E_p = np.tile(E_p, 6)
        np.testing.assert_array_almost_equal(
            eotf_inverse_BT2100_PQ(F_D), E_p, decimal=7
        )

        F_D = np.reshape(F_D, (2, 3))
        E_p = np.reshape(E_p, (2, 3))
        np.testing.assert_array_almost_equal(
            eotf_inverse_BT2100_PQ(F_D), E_p, decimal=7
        )

        F_D = np.reshape(F_D, (2, 3, 1))
        E_p = np.reshape(E_p, (2, 3, 1))
        np.testing.assert_array_almost_equal(
            eotf_inverse_BT2100_PQ(F_D), E_p, decimal=7
        )

    def test_domain_range_scale_eotf_inverse_BT2100_PQ(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_BT2100_PQ` definition domain and range scale support.
        """

        F_D = 779.98836083408537
        E_p = eotf_inverse_BT2100_PQ(F_D)

        d_r = (("reference", 1), ("1", 1), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_array_almost_equal(
                    eotf_inverse_BT2100_PQ(F_D * factor),
                    E_p * factor,
                    decimal=7,
                )

    @ignore_numpy_errors
    def test_nan_eotf_inverse_BT2100_PQ(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_BT2100_PQ` definition nan support.
        """

        eotf_inverse_BT2100_PQ(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestOotf_BT2100_PQ(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_BT2100_PQ` definition unit tests methods.
    """

    def test_ootf_BT2100_PQ(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_BT2100_PQ` definition.
        """

        self.assertAlmostEqual(ootf_BT2100_PQ(0.0), 0.0, places=7)

        self.assertAlmostEqual(
            ootf_BT2100_PQ(0.1), 779.98836083411584, places=7
        )

        self.assertAlmostEqual(
            ootf_BT2100_PQ(1.0), 9999.993723673924300, places=7
        )

    def test_n_dimensional_ootf_BT2100_PQ(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_BT2100_PQ` definition n-dimensional arrays support.
        """

        E = 0.1
        F_D = ootf_BT2100_PQ(E)

        E = np.tile(E, 6)
        F_D = np.tile(F_D, 6)
        np.testing.assert_array_almost_equal(ootf_BT2100_PQ(E), F_D, decimal=7)

        E = np.reshape(E, (2, 3))
        F_D = np.reshape(F_D, (2, 3))
        np.testing.assert_array_almost_equal(ootf_BT2100_PQ(E), F_D, decimal=7)

        E = np.reshape(E, (2, 3, 1))
        F_D = np.reshape(F_D, (2, 3, 1))
        np.testing.assert_array_almost_equal(ootf_BT2100_PQ(E), F_D, decimal=7)

    def test_domain_range_scale_ootf_BT2100_PQ(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_BT2100_PQ` definition domain and range scale support.
        """

        E = 0.1
        F_D = ootf_BT2100_PQ(E)

        d_r = (("reference", 1), ("1", 1), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_array_almost_equal(
                    ootf_BT2100_PQ(E * factor), F_D * factor, decimal=7
                )

    @ignore_numpy_errors
    def test_nan_ootf_BT2100_PQ(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_BT2100_PQ` definition nan support.
        """

        ootf_BT2100_PQ(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestOotf_inverse_BT2100_PQ(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_BT2100_PQ` definition unit tests methods.
    """

    def test_ootf_inverse_BT2100_PQ(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_BT2100_PQ` definition.
        """

        self.assertAlmostEqual(ootf_inverse_BT2100_PQ(0.0), 0.0, places=7)

        self.assertAlmostEqual(
            ootf_inverse_BT2100_PQ(779.98836083411584), 0.1, places=7
        )

        self.assertAlmostEqual(
            ootf_inverse_BT2100_PQ(9999.993723673924300), 1.0, places=7
        )

    def test_n_dimensional_ootf_inverse_BT2100_PQ(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_BT2100_PQ` definition n-dimensional arrays support.
        """

        F_D = 779.98836083411584
        E = ootf_inverse_BT2100_PQ(F_D)

        F_D = np.tile(F_D, 6)
        E = np.tile(E, 6)
        np.testing.assert_array_almost_equal(
            ootf_inverse_BT2100_PQ(F_D), E, decimal=7
        )

        F_D = np.reshape(F_D, (2, 3))
        E = np.reshape(E, (2, 3))
        np.testing.assert_array_almost_equal(
            ootf_inverse_BT2100_PQ(F_D), E, decimal=7
        )

        F_D = np.reshape(F_D, (2, 3, 1))
        E = np.reshape(E, (2, 3, 1))
        np.testing.assert_array_almost_equal(
            ootf_inverse_BT2100_PQ(F_D), E, decimal=7
        )

    def test_domain_range_scale_ootf_inverse_BT2100_PQ(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_BT2100_PQ` definition domain and range scale support.
        """

        F_D = 779.98836083411584
        E = ootf_inverse_BT2100_PQ(F_D)

        d_r = (("reference", 1), ("1", 1), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_array_almost_equal(
                    ootf_inverse_BT2100_PQ(F_D * factor), E * factor, decimal=7
                )

    @ignore_numpy_errors
    def test_nan_ootf_inverse_BT2100_PQ(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_BT2100_PQ` definition nan support.
        """

        ootf_inverse_BT2100_PQ(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestGamma_function_BT2100_HLG(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
gamma_function_BT2100_HLG` definition unit tests methods.
    """

    def test_gamma_function_BT2100_HLG(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
gamma_function_BT2100_HLG` definition.
        """

        self.assertAlmostEqual(
            gamma_function_BT2100_HLG(1000.0), 1.2, places=7
        )

        self.assertAlmostEqual(
            gamma_function_BT2100_HLG(2000.0), 1.326432598178872, places=7
        )

        self.assertAlmostEqual(
            gamma_function_BT2100_HLG(4000.0), 1.452865196357744, places=7
        )

        self.assertAlmostEqual(
            gamma_function_BT2100_HLG(10000.0), 1.619999999999999, places=7
        )


class TestOetf_BT2100_HLG(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_BT2100_HLG` definition unit tests methods.
    """

    def test_oetf_BT2100_HLG(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_BT2100_HLG` definition.
        """

        self.assertAlmostEqual(oetf_BT2100_HLG(0.0), 0.0, places=7)

        self.assertAlmostEqual(
            oetf_BT2100_HLG(0.18 / 12), 0.212132034355964, places=7
        )

        self.assertAlmostEqual(
            oetf_BT2100_HLG(1.0), 0.999999995536569, places=7
        )

    def test_n_dimensional_oetf_BT2100_HLG(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_BT2100_HLG` definition n-dimensional arrays support.
        """

        E = 0.18 / 12
        E_p = oetf_BT2100_HLG(E)

        E = np.tile(E, 6)
        E_p = np.tile(E_p, 6)
        np.testing.assert_array_almost_equal(
            oetf_BT2100_HLG(E), E_p, decimal=7
        )

        E = np.reshape(E, (2, 3))
        E_p = np.reshape(E_p, (2, 3))
        np.testing.assert_array_almost_equal(
            oetf_BT2100_HLG(E), E_p, decimal=7
        )

        E = np.reshape(E, (2, 3, 1))
        E_p = np.reshape(E_p, (2, 3, 1))
        np.testing.assert_array_almost_equal(
            oetf_BT2100_HLG(E), E_p, decimal=7
        )

    def test_domain_range_scale_oetf_BT2100_HLG(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_BT2100_HLG` definition domain and range scale support.
        """

        E = 0.18 / 12
        E_p = oetf_BT2100_HLG(E)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_array_almost_equal(
                    oetf_BT2100_HLG(E * factor), E_p * factor, decimal=7
                )

    @ignore_numpy_errors
    def test_nan_oetf_BT2100_HLG(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_BT2100_HLG` definition nan support.
        """

        oetf_BT2100_HLG(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestOetf_inverse_BT2100_HLG(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_inverse_BT2100_HLG` definition unit tests methods.
    """

    def test_oetf_inverse_BT2100_HLG(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_inverse_BT2100_HLG` definition.
        """

        self.assertAlmostEqual(oetf_inverse_BT2100_HLG(0.0), 0.0, places=7)

        self.assertAlmostEqual(
            oetf_inverse_BT2100_HLG(0.212132034355964), 0.18 / 12, places=7
        )

        self.assertAlmostEqual(
            oetf_inverse_BT2100_HLG(0.999999995536569), 1.0, places=7
        )

    def test_n_dimensional_oetf_inverse_BT2100_HLG(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_inverse_BT2100_HLG` definition n-dimensional arrays support.
        """

        E_p = 0.212132034355964
        E = oetf_inverse_BT2100_HLG(E_p)

        E_p = np.tile(E_p, 6)
        E = np.tile(E, 6)
        np.testing.assert_array_almost_equal(
            oetf_inverse_BT2100_HLG(E_p), E, decimal=7
        )

        E_p = np.reshape(E_p, (2, 3))
        E = np.reshape(E, (2, 3))
        np.testing.assert_array_almost_equal(
            oetf_inverse_BT2100_HLG(E_p), E, decimal=7
        )

        E_p = np.reshape(E_p, (2, 3, 1))
        E = np.reshape(E, (2, 3, 1))
        np.testing.assert_array_almost_equal(
            oetf_inverse_BT2100_HLG(E_p), E, decimal=7
        )

    def test_domain_range_scale_oetf_inverse_BT2100_HLG(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_inverse_BT2100_HLG` definition domain and range scale support.
        """

        E_p = 0.212132034355964
        E = oetf_inverse_BT2100_HLG(E_p)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_array_almost_equal(
                    oetf_inverse_BT2100_HLG(E_p * factor),
                    E * factor,
                    decimal=7,
                )

    @ignore_numpy_errors
    def test_nan_oetf_inverse_BT2100_HLG(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_inverse_BT2100_HLG` definition nan support.
        """

        oetf_inverse_BT2100_HLG(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestEotf_BT2100_HLG_1(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_BT2100_HLG_1` definition unit tests methods.
    """

    def test_eotf_BT2100_HLG_1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_BT2100_HLG_1` definition.
        """

        self.assertAlmostEqual(eotf_BT2100_HLG_1(0.0), 0.0, places=7)

        self.assertAlmostEqual(
            eotf_BT2100_HLG_1(0.212132034355964), 6.476039825649814, places=7
        )

        self.assertAlmostEqual(
            eotf_BT2100_HLG_1(1.0), 1000.000032321769100, places=7
        )

        self.assertAlmostEqual(
            eotf_BT2100_HLG_1(0.212132034355964, 0.001, 10000, 1.4),
            27.96039175299561,
            places=7,
        )

    def test_n_dimensional_eotf_BT2100_HLG_1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_BT2100_HLG_1` definition n-dimensional arrays support.
        """

        E_p = 0.212132034355964
        F_D = eotf_BT2100_HLG_1(E_p)

        E_p = np.tile(E_p, 6)
        F_D = np.tile(F_D, 6)
        np.testing.assert_array_almost_equal(
            eotf_BT2100_HLG_1(E_p), F_D, decimal=7
        )

        E_p = np.reshape(E_p, (2, 3))
        F_D = np.reshape(F_D, (2, 3))
        np.testing.assert_array_almost_equal(
            eotf_BT2100_HLG_1(E_p), F_D, decimal=7
        )

        E_p = np.reshape(E_p, (2, 3, 1))
        F_D = np.reshape(F_D, (2, 3, 1))
        np.testing.assert_array_almost_equal(
            eotf_BT2100_HLG_1(E_p), F_D, decimal=7
        )

        E_p = np.reshape(E_p, (6, 1))
        F_D = np.reshape(F_D, (6, 1))
        np.testing.assert_array_almost_equal(
            eotf_BT2100_HLG_1(E_p), F_D, decimal=7
        )

        E_p = np.array([0.25, 0.50, 0.75])
        F_D = np.array([12.49759413, 49.99037650, 158.94693786])
        np.testing.assert_array_almost_equal(
            eotf_BT2100_HLG_1(E_p), F_D, decimal=7
        )

        E_p = np.tile(E_p, (6, 1))
        F_D = np.tile(F_D, (6, 1))
        np.testing.assert_array_almost_equal(
            eotf_BT2100_HLG_1(E_p), F_D, decimal=7
        )

        E_p = np.reshape(E_p, (2, 3, 3))
        F_D = np.reshape(F_D, (2, 3, 3))
        np.testing.assert_array_almost_equal(
            eotf_BT2100_HLG_1(E_p), F_D, decimal=7
        )

    def test_domain_range_scale_eotf_BT2100_HLG_1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_BT2100_HLG_1` definition domain and range scale support.
        """

        E_p = 0.212132034355964
        F_D = eotf_BT2100_HLG_1(E_p)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_array_almost_equal(
                    eotf_BT2100_HLG_1(E_p * factor), F_D * factor, decimal=7
                )

    @ignore_numpy_errors
    def test_nan_eotf_BT2100_HLG_1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_BT2100_HLG_1` definition nan support.
        """

        eotf_BT2100_HLG_1(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestEotf_BT2100_HLG_2(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_BT2100_HLG_2` definition unit tests methods.
    """

    def test_eotf_BT2100_HLG_2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_BT2100_HLG_2` definition.
        """

        self.assertAlmostEqual(eotf_BT2100_HLG_2(0.0), 0.0, places=7)

        self.assertAlmostEqual(
            eotf_BT2100_HLG_2(0.212132034355964), 6.476039825649814, places=7
        )

        self.assertAlmostEqual(
            eotf_BT2100_HLG_2(1.0), 1000.000032321769100, places=7
        )

        self.assertAlmostEqual(
            eotf_BT2100_HLG_2(0.212132034355964, 0.001, 10000, 1.4),
            29.581261576946076,
            places=7,
        )

    def test_n_dimensional_eotf_BT2100_HLG_2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_BT2100_HLG_2` definition n-dimensional arrays support.
        """

        E_p = 0.212132034355964
        F_D = eotf_BT2100_HLG_2(E_p)

        E_p = np.tile(E_p, 6)
        F_D = np.tile(F_D, 6)
        np.testing.assert_array_almost_equal(
            eotf_BT2100_HLG_2(E_p), F_D, decimal=7
        )

        E_p = np.reshape(E_p, (2, 3))
        F_D = np.reshape(F_D, (2, 3))
        np.testing.assert_array_almost_equal(
            eotf_BT2100_HLG_2(E_p), F_D, decimal=7
        )

        E_p = np.reshape(E_p, (2, 3, 1))
        F_D = np.reshape(F_D, (2, 3, 1))
        np.testing.assert_array_almost_equal(
            eotf_BT2100_HLG_2(E_p), F_D, decimal=7
        )

        E_p = np.reshape(E_p, (6, 1))
        F_D = np.reshape(F_D, (6, 1))
        np.testing.assert_array_almost_equal(
            eotf_BT2100_HLG_2(E_p), F_D, decimal=7
        )

        E_p = np.array([0.25, 0.50, 0.75])
        F_D = np.array([12.49759413, 49.99037650, 158.94693786])
        np.testing.assert_array_almost_equal(
            eotf_BT2100_HLG_2(E_p), F_D, decimal=7
        )

        E_p = np.tile(E_p, (6, 1))
        F_D = np.tile(F_D, (6, 1))
        np.testing.assert_array_almost_equal(
            eotf_BT2100_HLG_2(E_p), F_D, decimal=7
        )

        E_p = np.reshape(E_p, (2, 3, 3))
        F_D = np.reshape(F_D, (2, 3, 3))
        np.testing.assert_array_almost_equal(
            eotf_BT2100_HLG_2(E_p), F_D, decimal=7
        )

    def test_domain_range_scale_eotf_BT2100_HLG_2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_BT2100_HLG_2` definition domain and range scale support.
        """

        E_p = 0.212132034355964
        F_D = eotf_BT2100_HLG_2(E_p)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_array_almost_equal(
                    eotf_BT2100_HLG_2(E_p * factor), F_D * factor, decimal=7
                )

    @ignore_numpy_errors
    def test_nan_eotf_BT2100_HLG_2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_BT2100_HLG_2` definition nan support.
        """

        eotf_BT2100_HLG_2(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestEotf_inverse_BT2100_HLG_1(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_BT2100_HLG_1` definition unit tests methods.
    """

    def test_eotf_inverse_BT2100_HLG_1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_BT2100_HLG_1` definition.
        """

        self.assertAlmostEqual(eotf_inverse_BT2100_HLG_1(0.0), 0.0, places=7)

        self.assertAlmostEqual(
            eotf_inverse_BT2100_HLG_1(6.476039825649814),
            0.212132034355964,
            places=7,
        )

        self.assertAlmostEqual(
            eotf_inverse_BT2100_HLG_1(1000.000032321769100), 1.0, places=7
        )

        self.assertAlmostEqual(
            eotf_inverse_BT2100_HLG_1(27.96039175299561, 0.001, 10000, 1.4),
            0.212132034355964,
            places=7,
        )

    def test_n_dimensional_eotf_inverse_BT2100_HLG_1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_BT2100_HLG_1` definition n-dimensional arrays support.
        """

        F_D = 6.476039825649814
        E_p = eotf_inverse_BT2100_HLG_1(F_D)

        F_D = np.tile(F_D, 6)
        E_p = np.tile(E_p, 6)
        np.testing.assert_array_almost_equal(
            eotf_inverse_BT2100_HLG_1(F_D), E_p, decimal=7
        )

        F_D = np.reshape(F_D, (2, 3))
        E_p = np.reshape(E_p, (2, 3))
        np.testing.assert_array_almost_equal(
            eotf_inverse_BT2100_HLG_1(F_D), E_p, decimal=7
        )

        F_D = np.reshape(F_D, (2, 3, 1))
        E_p = np.reshape(E_p, (2, 3, 1))
        np.testing.assert_array_almost_equal(
            eotf_inverse_BT2100_HLG_1(F_D), E_p, decimal=7
        )

        F_D = np.reshape(F_D, (6, 1))
        E_p = np.reshape(E_p, (6, 1))
        np.testing.assert_array_almost_equal(
            eotf_inverse_BT2100_HLG_1(F_D), E_p, decimal=7
        )

        F_D = np.array([12.49759413, 49.99037650, 158.94693786])
        E_p = np.array([0.25, 0.50, 0.75])
        np.testing.assert_array_almost_equal(
            eotf_inverse_BT2100_HLG_1(F_D), E_p, decimal=7
        )

        F_D = np.tile(F_D, (6, 1))
        E_p = np.tile(E_p, (6, 1))
        np.testing.assert_array_almost_equal(
            eotf_inverse_BT2100_HLG_1(F_D), E_p, decimal=7
        )

        F_D = np.reshape(F_D, (2, 3, 3))
        E_p = np.reshape(E_p, (2, 3, 3))
        np.testing.assert_array_almost_equal(
            eotf_inverse_BT2100_HLG_1(F_D), E_p, decimal=7
        )

    def test_domain_range_scale_eotf_inverse_BT2100_HLG_1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_BT2100_HLG_1` definition domain and range scale support.
        """

        F_D = 6.476039825649814
        E_p = eotf_inverse_BT2100_HLG_1(F_D)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_array_almost_equal(
                    eotf_inverse_BT2100_HLG_1(F_D * factor),
                    E_p * factor,
                    decimal=7,
                )

    @ignore_numpy_errors
    def test_nan_eotf_inverse_BT2100_HLG_1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_BT2100_HLG_1` definition nan support.
        """

        eotf_inverse_BT2100_HLG_1(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestEotf_inverse_BT2100_HLG_2(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_BT2100_HLG_2` definition unit tests methods.
    """

    def test_eotf_inverse_BT2100_HLG_2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_BT2100_HLG_2` definition.
        """

        self.assertAlmostEqual(eotf_inverse_BT2100_HLG_2(0.0), 0.0, places=7)

        self.assertAlmostEqual(
            eotf_inverse_BT2100_HLG_2(6.476039825649814),
            0.212132034355964,
            places=7,
        )

        self.assertAlmostEqual(
            eotf_inverse_BT2100_HLG_2(1000.000032321769100), 1.0, places=7
        )

        self.assertAlmostEqual(
            eotf_inverse_BT2100_HLG_2(29.581261576946076, 0.001, 10000, 1.4),
            0.212132034355964,
            places=7,
        )

    def test_n_dimensional_eotf_inverse_BT2100_HLG_2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_BT2100_HLG_2` definition n-dimensional arrays support.
        """

        F_D = 6.476039825649814
        E_p = eotf_inverse_BT2100_HLG_2(F_D)

        F_D = np.tile(F_D, 6)
        E_p = np.tile(E_p, 6)
        np.testing.assert_array_almost_equal(
            eotf_inverse_BT2100_HLG_2(F_D), E_p, decimal=7
        )

        F_D = np.reshape(F_D, (2, 3))
        E_p = np.reshape(E_p, (2, 3))
        np.testing.assert_array_almost_equal(
            eotf_inverse_BT2100_HLG_2(F_D), E_p, decimal=7
        )

        F_D = np.reshape(F_D, (2, 3, 1))
        E_p = np.reshape(E_p, (2, 3, 1))
        np.testing.assert_array_almost_equal(
            eotf_inverse_BT2100_HLG_2(F_D), E_p, decimal=7
        )

        F_D = np.reshape(F_D, (6, 1))
        E_p = np.reshape(E_p, (6, 1))
        np.testing.assert_array_almost_equal(
            eotf_inverse_BT2100_HLG_2(F_D), E_p, decimal=7
        )

        F_D = np.array([12.49759413, 49.99037650, 158.94693786])
        E_p = np.array([0.25, 0.50, 0.75])
        np.testing.assert_array_almost_equal(
            eotf_inverse_BT2100_HLG_2(F_D), E_p, decimal=7
        )

        F_D = np.tile(F_D, (6, 1))
        E_p = np.tile(E_p, (6, 1))
        np.testing.assert_array_almost_equal(
            eotf_inverse_BT2100_HLG_2(F_D), E_p, decimal=7
        )

        F_D = np.reshape(F_D, (2, 3, 3))
        E_p = np.reshape(E_p, (2, 3, 3))
        np.testing.assert_array_almost_equal(
            eotf_inverse_BT2100_HLG_2(F_D), E_p, decimal=7
        )

    def test_domain_range_scale_eotf_inverse_BT2100_HLG_2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_BT2100_HLG_2` definition domain and range scale support.
        """

        F_D = 6.476039825649814
        E_p = eotf_inverse_BT2100_HLG_2(F_D)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_array_almost_equal(
                    eotf_inverse_BT2100_HLG_2(F_D * factor),
                    E_p * factor,
                    decimal=7,
                )

    @ignore_numpy_errors
    def test_nan_eotf_inverse_BT2100_HLG_2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_BT2100_HLG_2` definition nan support.
        """

        eotf_inverse_BT2100_HLG_2(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestOotf_BT2100_HLG_1(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_BT2100_HLG_1` definition unit tests methods.
    """

    def test_ootf_BT2100_HLG_1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_BT2100_HLG_1` definition.
        """

        self.assertAlmostEqual(ootf_BT2100_HLG_1(0.0), 0.0, places=7)

        self.assertAlmostEqual(
            ootf_BT2100_HLG_1(0.1), 63.095734448019336, places=7
        )

        self.assertAlmostEqual(ootf_BT2100_HLG_1(1.0), 1000.0, places=7)

        self.assertAlmostEqual(
            ootf_BT2100_HLG_1(0.1, 0.001, 10000, 1.4),
            398.108130742780300,
            places=7,
        )

        a = np.array(
            [
                [45.884942278760597, 0.000000000000000, -45.884942278760597],
                [
                    -63.095734448019336,
                    -63.095734448019336,
                    -63.095734448019336,
                ],
                [63.095734448019336, 63.095734448019336, 63.095734448019336],
                [51.320396090100672, -51.320396090100672, 51.320396090100672],
            ],
        )
        np.testing.assert_array_almost_equal(
            ootf_BT2100_HLG_1(
                np.array(
                    [
                        [0.1, 0.0, -0.1],
                        [-0.1, -0.1, -0.1],
                        [0.1, 0.1, 0.1],
                        [0.1, -0.1, 0.1],
                    ]
                )
            ),
            a,
            decimal=7,
        )

    def test_n_dimensional_ootf_BT2100_HLG_1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_BT2100_HLG_1` definition n-dimensional arrays support.
        """

        E = 0.1
        F_D = ootf_BT2100_HLG_1(E)

        E = np.tile(E, 6)
        F_D = np.tile(F_D, 6)
        np.testing.assert_array_almost_equal(
            ootf_BT2100_HLG_1(E), F_D, decimal=7
        )

        E = np.reshape(E, (2, 3))
        F_D = np.reshape(F_D, (2, 3))
        np.testing.assert_array_almost_equal(
            ootf_BT2100_HLG_1(E), F_D, decimal=7
        )

        E = np.reshape(E, (2, 3, 1))
        F_D = np.reshape(F_D, (2, 3, 1))
        np.testing.assert_array_almost_equal(
            ootf_BT2100_HLG_1(E), F_D, decimal=7
        )

        E = np.reshape(E, (6, 1))
        F_D = np.reshape(F_D, (6, 1))
        np.testing.assert_array_almost_equal(
            ootf_BT2100_HLG_1(E), F_D, decimal=7
        )

        E = np.array([0.25, 0.50, 0.75])
        F_D = np.array([213.01897444, 426.03794887, 639.05692331])
        np.testing.assert_array_almost_equal(
            ootf_BT2100_HLG_1(E), F_D, decimal=7
        )

        E = np.tile(E, (6, 1))
        F_D = np.tile(F_D, (6, 1))
        np.testing.assert_array_almost_equal(
            ootf_BT2100_HLG_1(E), F_D, decimal=7
        )

        E = np.reshape(E, (2, 3, 3))
        F_D = np.reshape(F_D, (2, 3, 3))
        np.testing.assert_array_almost_equal(
            ootf_BT2100_HLG_1(E), F_D, decimal=7
        )

    def test_domain_range_scale_ootf_BT2100_HLG_1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_BT2100_HLG_1` definition domain and range scale support.
        """

        E = 0.1
        F_D = ootf_BT2100_HLG_1(E)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_array_almost_equal(
                    ootf_BT2100_HLG_1(E * factor), F_D * factor, decimal=7
                )

    @ignore_numpy_errors
    def test_nan_ootf_BT2100_HLG_1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_BT2100_HLG_1` definition nan support.
        """

        ootf_BT2100_HLG_1(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestOotf_BT2100_HLG_2(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_BT2100_HLG_2` definition unit tests methods.
    """

    def test_ootf_BT2100_HLG_2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_BT2100_HLG_2` definition.
        """

        self.assertAlmostEqual(ootf_BT2100_HLG_2(0.0), 0.0, places=7)

        self.assertAlmostEqual(
            ootf_BT2100_HLG_2(0.1), 63.095734448019336, places=7
        )

        self.assertAlmostEqual(ootf_BT2100_HLG_2(1.0), 1000.0, places=7)

        self.assertAlmostEqual(
            ootf_BT2100_HLG_2(0.1, 10000, 1.4), 398.107170553497380, places=7
        )

        a = np.array(
            [
                [45.884942278760597, 0.000000000000000, -45.884942278760597],
                [
                    -63.095734448019336,
                    -63.095734448019336,
                    -63.095734448019336,
                ],
                [63.095734448019336, 63.095734448019336, 63.095734448019336],
                [51.320396090100672, -51.320396090100672, 51.320396090100672],
            ],
        )
        np.testing.assert_array_almost_equal(
            ootf_BT2100_HLG_2(
                np.array(
                    [
                        [0.1, 0.0, -0.1],
                        [-0.1, -0.1, -0.1],
                        [0.1, 0.1, 0.1],
                        [0.1, -0.1, 0.1],
                    ]
                )
            ),
            a,
            decimal=7,
        )

    def test_n_dimensional_ootf_BT2100_HLG_2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_BT2100_HLG_2` definition n-dimensional arrays support.
        """

        E = 0.1
        F_D = ootf_BT2100_HLG_2(E)

        E = np.tile(E, 6)
        F_D = np.tile(F_D, 6)
        np.testing.assert_array_almost_equal(
            ootf_BT2100_HLG_2(E), F_D, decimal=7
        )

        E = np.reshape(E, (2, 3))
        F_D = np.reshape(F_D, (2, 3))
        np.testing.assert_array_almost_equal(
            ootf_BT2100_HLG_2(E), F_D, decimal=7
        )

        E = np.reshape(E, (2, 3, 1))
        F_D = np.reshape(F_D, (2, 3, 1))
        np.testing.assert_array_almost_equal(
            ootf_BT2100_HLG_2(E), F_D, decimal=7
        )

        E = np.reshape(E, (6, 1))
        F_D = np.reshape(F_D, (6, 1))
        np.testing.assert_array_almost_equal(
            ootf_BT2100_HLG_2(E), F_D, decimal=7
        )

        E = np.array([0.25, 0.50, 0.75])
        F_D = np.array([213.01897444, 426.03794887, 639.05692331])
        np.testing.assert_array_almost_equal(
            ootf_BT2100_HLG_2(E), F_D, decimal=7
        )

        E = np.tile(E, (6, 1))
        F_D = np.tile(F_D, (6, 1))
        np.testing.assert_array_almost_equal(
            ootf_BT2100_HLG_2(E), F_D, decimal=7
        )

        E = np.reshape(E, (2, 3, 3))
        F_D = np.reshape(F_D, (2, 3, 3))
        np.testing.assert_array_almost_equal(
            ootf_BT2100_HLG_2(E), F_D, decimal=7
        )

    def test_domain_range_scale_ootf_BT2100_HLG_2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_BT2100_HLG_2` definition domain and range scale support.
        """

        E = 0.1
        F_D = ootf_BT2100_HLG_1(E)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_array_almost_equal(
                    ootf_BT2100_HLG_1(E * factor), F_D * factor, decimal=7
                )

    @ignore_numpy_errors
    def test_nan_ootf_BT2100_HLG_1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_BT2100_HLG_1` definition nan support.
        """

        ootf_BT2100_HLG_1(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestOotf_inverse_BT2100_HLG_1(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_BT2100_HLG_1` definition unit tests methods.
    """

    def test_ootf_inverse_BT2100_HLG_1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_BT2100_HLG_1` definition.
        """

        self.assertAlmostEqual(ootf_inverse_BT2100_HLG_1(0.0), 0.0, places=7)

        self.assertAlmostEqual(
            ootf_inverse_BT2100_HLG_1(63.095734448019336), 0.1, places=7
        )

        self.assertAlmostEqual(
            ootf_inverse_BT2100_HLG_1(1000.0), 1.0, places=7
        )

        self.assertAlmostEqual(
            ootf_inverse_BT2100_HLG_1(398.108130742780300, 0.001, 10000, 1.4),
            0.1,
            places=7,
        )

        a = np.array(
            [
                [45.884942278760597, 0.000000000000000, -45.884942278760597],
                [
                    -63.095734448019336,
                    -63.095734448019336,
                    -63.095734448019336,
                ],
                [63.095734448019336, 63.095734448019336, 63.095734448019336],
                [51.320396090100672, -51.320396090100672, 51.320396090100672],
            ]
        )
        np.testing.assert_array_almost_equal(
            ootf_inverse_BT2100_HLG_1(a),
            np.array(
                [
                    [0.1, 0.0, -0.1],
                    [-0.1, -0.1, -0.1],
                    [0.1, 0.1, 0.1],
                    [0.1, -0.1, 0.1],
                ]
            ),
            decimal=7,
        )

    def test_n_dimensional_ootf_inverse_BT2100_HLG_1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_BT2100_HLG_1` definition n-dimensional arrays support.
        """

        F_D = 63.095734448019336
        E = ootf_inverse_BT2100_HLG_1(F_D)

        F_D = np.tile(F_D, 6)
        E = np.tile(E, 6)
        np.testing.assert_array_almost_equal(
            ootf_inverse_BT2100_HLG_1(F_D), E, decimal=7
        )

        F_D = np.reshape(F_D, (2, 3))
        E = np.reshape(E, (2, 3))
        np.testing.assert_array_almost_equal(
            ootf_inverse_BT2100_HLG_1(F_D), E, decimal=7
        )

        F_D = np.reshape(F_D, (2, 3, 1))
        E = np.reshape(E, (2, 3, 1))
        np.testing.assert_array_almost_equal(
            ootf_inverse_BT2100_HLG_1(F_D), E, decimal=7
        )

        F_D = np.reshape(F_D, (6, 1))
        E = np.reshape(E, (6, 1))
        np.testing.assert_array_almost_equal(
            ootf_inverse_BT2100_HLG_1(F_D), E, decimal=7
        )

        F_D = np.array([213.01897444, 426.03794887, 639.05692331])
        E = np.array([0.25, 0.50, 0.75])
        np.testing.assert_array_almost_equal(
            ootf_inverse_BT2100_HLG_1(F_D), E, decimal=7
        )

        F_D = np.tile(F_D, (6, 1))
        E = np.tile(E, (6, 1))
        np.testing.assert_array_almost_equal(
            ootf_inverse_BT2100_HLG_1(F_D), E, decimal=7
        )

        F_D = np.reshape(F_D, (2, 3, 3))
        E = np.reshape(E, (2, 3, 3))
        np.testing.assert_array_almost_equal(
            ootf_inverse_BT2100_HLG_1(F_D), E, decimal=7
        )

    def test_domain_range_scale_ootf_inverse_BT2100_HLG_1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_BT2100_HLG_1` definition domain and range scale support.
        """

        F_D = 63.095734448019336
        E = ootf_inverse_BT2100_HLG_1(F_D)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_array_almost_equal(
                    ootf_inverse_BT2100_HLG_1(F_D * factor),
                    E * factor,
                    decimal=7,
                )

    @ignore_numpy_errors
    def test_nan_ootf_inverse_BT2100_HLG_1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_BT2100_HLG_1` definition nan support.
        """

        ootf_inverse_BT2100_HLG_1(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestOotf_inverse_BT2100_HLG_2(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_BT2100_HLG_2` definition unit tests methods.
    """

    def test_ootf_inverse_BT2100_HLG_2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_BT2100_HLG_2` definition.
        """

        self.assertAlmostEqual(ootf_inverse_BT2100_HLG_2(0.0), 0.0, places=7)

        self.assertAlmostEqual(
            ootf_inverse_BT2100_HLG_2(63.095734448019336), 0.1, places=7
        )

        self.assertAlmostEqual(
            ootf_inverse_BT2100_HLG_2(1000.0), 1.0, places=7
        )

        self.assertAlmostEqual(
            ootf_inverse_BT2100_HLG_2(398.107170553497380, 10000, 1.4),
            0.1,
            places=7,
        )

        a = np.array(
            [
                [45.884942278760597, 0.000000000000000, -45.884942278760597],
                [
                    -63.095734448019336,
                    -63.095734448019336,
                    -63.095734448019336,
                ],
                [63.095734448019336, 63.095734448019336, 63.095734448019336],
                [51.320396090100672, -51.320396090100672, 51.320396090100672],
            ]
        )
        np.testing.assert_array_almost_equal(
            ootf_inverse_BT2100_HLG_2(a),
            np.array(
                [
                    [0.1, 0.0, -0.1],
                    [-0.1, -0.1, -0.1],
                    [0.1, 0.1, 0.1],
                    [0.1, -0.1, 0.1],
                ]
            ),
            decimal=7,
        )

    def test_n_dimensional_ootf_inverse_BT2100_HLG_2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_BT2100_HLG_2` definition n-dimensional arrays support.
        """

        F_D = 63.095734448019336
        E = ootf_inverse_BT2100_HLG_2(F_D)

        F_D = np.tile(F_D, 6)
        E = np.tile(E, 6)
        np.testing.assert_array_almost_equal(
            ootf_inverse_BT2100_HLG_2(F_D), E, decimal=7
        )

        F_D = np.reshape(F_D, (2, 3))
        E = np.reshape(E, (2, 3))
        np.testing.assert_array_almost_equal(
            ootf_inverse_BT2100_HLG_2(F_D), E, decimal=7
        )

        F_D = np.reshape(F_D, (2, 3, 1))
        E = np.reshape(E, (2, 3, 1))
        np.testing.assert_array_almost_equal(
            ootf_inverse_BT2100_HLG_2(F_D), E, decimal=7
        )

        F_D = np.reshape(F_D, (6, 1))
        E = np.reshape(E, (6, 1))
        np.testing.assert_array_almost_equal(
            ootf_inverse_BT2100_HLG_2(F_D), E, decimal=7
        )

        F_D = np.array([213.01897444, 426.03794887, 639.05692331])
        E = np.array([0.25, 0.50, 0.75])
        np.testing.assert_array_almost_equal(
            ootf_inverse_BT2100_HLG_2(F_D), E, decimal=7
        )

        F_D = np.tile(F_D, (6, 1))
        E = np.tile(E, (6, 1))
        np.testing.assert_array_almost_equal(
            ootf_inverse_BT2100_HLG_2(F_D), E, decimal=7
        )

        F_D = np.reshape(F_D, (2, 3, 3))
        E = np.reshape(E, (2, 3, 3))
        np.testing.assert_array_almost_equal(
            ootf_inverse_BT2100_HLG_2(F_D), E, decimal=7
        )

    def test_domain_range_scale_ootf_inverse_BT2100_HLG_2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_BT2100_HLG_2` definition domain and range scale support.
        """

        F_D = 63.095734448019336
        E = ootf_inverse_BT2100_HLG_2(F_D)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_array_almost_equal(
                    ootf_inverse_BT2100_HLG_2(F_D * factor),
                    E * factor,
                    decimal=7,
                )

    @ignore_numpy_errors
    def test_nan_ootf_inverse_BT2100_HLG_2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_BT2100_HLG_2` definition nan support.
        """

        ootf_inverse_BT2100_HLG_2(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


if __name__ == "__main__":
    unittest.main()
