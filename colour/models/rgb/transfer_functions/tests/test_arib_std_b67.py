# -*- coding: utf-8 -*-
"""
Defines the unit tests for the
:mod:`colour.models.rgb.transfer_functions.arib_std_b67` module.
"""

import numpy as np
import unittest

from colour.models.rgb.transfer_functions import (
    oetf_ARIBSTDB67,
    oetf_inverse_ARIBSTDB67,
)
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestOetf_ARIBSTDB67',
    'TestOetf_inverse_ARIBSTDB67',
]


class TestOetf_ARIBSTDB67(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.arib_std_b67.\
oetf_ARIBSTDB67` definition unit tests methods.
    """

    def test_oetf_ARIBSTDB67(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.arib_std_b67.\
oetf_ARIBSTDB67` definition.
        """

        self.assertAlmostEqual(oetf_ARIBSTDB67(-0.25), -0.25, places=7)

        self.assertAlmostEqual(oetf_ARIBSTDB67(0.0), 0.0, places=7)

        self.assertAlmostEqual(
            oetf_ARIBSTDB67(0.18), 0.212132034355964, places=7)

        self.assertAlmostEqual(oetf_ARIBSTDB67(1.0), 0.5, places=7)

        self.assertAlmostEqual(
            oetf_ARIBSTDB67(64.0), 1.302858098046995, places=7)

    def test_n_dimensional_oetf_ARIBSTDB67(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.arib_std_b67.\
oetf_ARIBSTDB67` definition n-dimensional arrays support.
        """

        E = 0.18
        E_p = oetf_ARIBSTDB67(E)

        E = np.tile(E, 6)
        E_p = np.tile(E_p, 6)
        np.testing.assert_almost_equal(oetf_ARIBSTDB67(E), E_p, decimal=7)

        E = np.reshape(E, (2, 3))
        E_p = np.reshape(E_p, (2, 3))
        np.testing.assert_almost_equal(oetf_ARIBSTDB67(E), E_p, decimal=7)

        E = np.reshape(E, (2, 3, 1))
        E_p = np.reshape(E_p, (2, 3, 1))
        np.testing.assert_almost_equal(oetf_ARIBSTDB67(E), E_p, decimal=7)

    def test_domain_range_scale_oetf_ARIBSTDB67(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.arib_std_b67.\
oetf_ARIBSTDB67` definition domain and range scale support.
        """

        E = 0.18
        E_p = oetf_ARIBSTDB67(E)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    oetf_ARIBSTDB67(E * factor), E_p * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_oetf_ARIBSTDB67(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.arib_std_b67.\
oetf_ARIBSTDB67` definition nan support.
        """

        oetf_ARIBSTDB67(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestOetf_inverse_ARIBSTDB67(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.arib_std_b67.\
oetf_inverse_ARIBSTDB67` definition unit tests methods.
    """

    def test_oetf_inverse_ARIBSTDB67(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.arib_std_b67.\
oetf_inverse_ARIBSTDB67` definition.
        """

        self.assertAlmostEqual(oetf_inverse_ARIBSTDB67(-0.25), -0.25, places=7)

        self.assertAlmostEqual(oetf_inverse_ARIBSTDB67(0.0), 0.0, places=7)

        self.assertAlmostEqual(
            oetf_inverse_ARIBSTDB67(0.212132034355964), 0.18, places=7)

        self.assertAlmostEqual(oetf_inverse_ARIBSTDB67(0.5), 1.0, places=7)

        self.assertAlmostEqual(
            oetf_inverse_ARIBSTDB67(1.302858098046995), 64.0, places=7)

    def test_n_dimensional_oetf_inverse_ARIBSTDB67(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.arib_std_b67.\
oetf_inverse_ARIBSTDB67` definition n-dimensional arrays support.
        """

        E_p = 0.212132034355964
        E = oetf_inverse_ARIBSTDB67(E_p)

        E_p = np.tile(E_p, 6)
        E = np.tile(E, 6)
        np.testing.assert_almost_equal(
            oetf_inverse_ARIBSTDB67(E_p), E, decimal=7)

        E_p = np.reshape(E_p, (2, 3))
        E = np.reshape(E, (2, 3))
        np.testing.assert_almost_equal(
            oetf_inverse_ARIBSTDB67(E_p), E, decimal=7)

        E_p = np.reshape(E_p, (2, 3, 1))
        E = np.reshape(E, (2, 3, 1))
        np.testing.assert_almost_equal(
            oetf_inverse_ARIBSTDB67(E_p), E, decimal=7)

    def test_domain_range_scale_oetf_inverse_ARIBSTDB67(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.arib_std_b67.\
oetf_inverse_ARIBSTDB67` definition domain and range scale support.
        """

        E_p = 0.212132034355964
        E = oetf_inverse_ARIBSTDB67(E_p)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    oetf_inverse_ARIBSTDB67(E_p * factor),
                    E * factor,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_oetf_inverse_ARIBSTDB67(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.arib_std_b67.\
oetf_inverse_ARIBSTDB67` definition nan support.
        """

        oetf_inverse_ARIBSTDB67(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


if __name__ == '__main__':
    unittest.main()
