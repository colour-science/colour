"""
Define the unit tests for the
:mod:`colour.models.rgb.transfer_functions.itur_bt_2020` module.
"""

import numpy as np
import unittest

from colour.models.rgb.transfer_functions import (
    oetf_BT2020,
    oetf_inverse_BT2020,
)
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestOetf_BT2020",
    "TestOetf_inverse_BT2020",
]


class TestOetf_BT2020(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2020.\
oetf_BT2020` definition unit tests methods.
    """

    def test_oetf_BT2020(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2020.\
oetf_BT2020` definition.
        """

        self.assertAlmostEqual(oetf_BT2020(0.0), 0.0, places=7)

        self.assertAlmostEqual(oetf_BT2020(0.18), 0.409007728864150, places=7)

        self.assertAlmostEqual(oetf_BT2020(1.0), 1.0, places=7)

    def test_n_dimensional_oetf_BT2020(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2020.\
oetf_BT2020` definition n-dimensional arrays support.
        """

        E = 0.18
        E_p = oetf_BT2020(E)

        E = np.tile(E, 6)
        E_p = np.tile(E_p, 6)
        np.testing.assert_almost_equal(oetf_BT2020(E), E_p, decimal=7)

        E = np.reshape(E, (2, 3))
        E_p = np.reshape(E_p, (2, 3))
        np.testing.assert_almost_equal(oetf_BT2020(E), E_p, decimal=7)

        E = np.reshape(E, (2, 3, 1))
        E_p = np.reshape(E_p, (2, 3, 1))
        np.testing.assert_almost_equal(oetf_BT2020(E), E_p, decimal=7)

    def test_domain_range_scale_oetf_BT2020(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2020.\
oetf_BT2020` definition domain and range scale support.
        """

        E = 0.18
        E_p = oetf_BT2020(E)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    oetf_BT2020(E * factor), E_p * factor, decimal=7
                )

    @ignore_numpy_errors
    def test_nan_oetf_BT2020(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2020.\
oetf_BT2020` definition nan support.
        """

        oetf_BT2020(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestOetf_inverse_BT2020(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2020.\
oetf_inverse_BT2020` definition unit tests methods.
    """

    def test_oetf_inverse_BT2020(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2020.\
oetf_inverse_BT2020` definition.
        """

        self.assertAlmostEqual(oetf_inverse_BT2020(0.0), 0.0, places=7)

        self.assertAlmostEqual(
            oetf_inverse_BT2020(0.409007728864150), 0.18, places=7
        )

        self.assertAlmostEqual(oetf_inverse_BT2020(1.0), 1.0, places=7)

    def test_n_dimensional_oetf_inverse_BT2020(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2020.\
oetf_inverse_BT2020` definition n-dimensional arrays support.
        """

        E_p = 0.409007728864150
        E = oetf_inverse_BT2020(E_p)

        E_p = np.tile(E_p, 6)
        E = np.tile(E, 6)
        np.testing.assert_almost_equal(oetf_inverse_BT2020(E_p), E, decimal=7)

        E_p = np.reshape(E_p, (2, 3))
        E = np.reshape(E, (2, 3))
        np.testing.assert_almost_equal(oetf_inverse_BT2020(E_p), E, decimal=7)

        E_p = np.reshape(E_p, (2, 3, 1))
        E = np.reshape(E, (2, 3, 1))
        np.testing.assert_almost_equal(oetf_inverse_BT2020(E_p), E, decimal=7)

    def test_domain_range_scale_oetf_inverse_BT2020(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2020.\
oetf_inverse_BT2020` definition domain and range scale support.
        """

        E_p = 0.409007728864150
        E = oetf_inverse_BT2020(E_p)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    oetf_inverse_BT2020(E_p * factor), E * factor, decimal=7
                )

    @ignore_numpy_errors
    def test_nan_oetf_inverse_BT2020(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2020.\
oetf_inverse_BT2020` definition nan support.
        """

        oetf_inverse_BT2020(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


if __name__ == "__main__":
    unittest.main()
