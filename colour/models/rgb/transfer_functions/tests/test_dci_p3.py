# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.models.rgb.transfer_functions.dci_p3`
module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.models.rgb.transfer_functions import (oetf_DCIP3, eotf_DCIP3)
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestOetf_DCIP3', 'TestEotf_DCIP3']


class TestOetf_DCIP3(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.dci_p3.\
oetf_DCIP3` definition unit tests methods.
    """

    def test_oetf_DCIP3(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.dci_p3.\
oetf_DCIP3` definition.
        """

        self.assertAlmostEqual(oetf_DCIP3(0.0), 0.0, places=7)

        self.assertAlmostEqual(oetf_DCIP3(0.18), 461.99220597, places=7)

        self.assertAlmostEqual(oetf_DCIP3(1.0), 893.44598341, places=7)

    def test_n_dimensional_oetf_DCIP3(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.dci_p3.\
oetf_DCIP3` definition n-dimensional arrays support.
        """

        XYZ = 0.18
        XYZ_p = 461.99220597
        np.testing.assert_almost_equal(oetf_DCIP3(XYZ), XYZ_p, decimal=7)

        XYZ = np.tile(XYZ, 6)
        XYZ_p = np.tile(XYZ_p, 6)
        np.testing.assert_almost_equal(oetf_DCIP3(XYZ), XYZ_p, decimal=7)

        XYZ = np.reshape(XYZ, (2, 3))
        XYZ_p = np.reshape(XYZ_p, (2, 3))
        np.testing.assert_almost_equal(oetf_DCIP3(XYZ), XYZ_p, decimal=7)

        XYZ = np.reshape(XYZ, (2, 3, 1))
        XYZ_p = np.reshape(XYZ_p, (2, 3, 1))
        np.testing.assert_almost_equal(oetf_DCIP3(XYZ), XYZ_p, decimal=7)

    def test_domain_range_scale_oetf_DCIP3(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.dci_p3.\
oetf_DCIP3` definition domain and range scale support.
        """

        XYZ = 0.18
        XYZ_p = oetf_DCIP3(XYZ)

        d_r = (
            ('reference', 1, 1),
            (1, 1, 1 / 4095),
            (100, 100, 100 / 4095),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    oetf_DCIP3(XYZ * factor_a), XYZ_p * factor_b, decimal=7)

    @ignore_numpy_errors
    def test_nan_oetf_DCIP3(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.dci_p3.\
oetf_DCIP3` definition nan support.
        """

        oetf_DCIP3(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestEotf_DCIP3(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.dci_p3.
eotf_DCIP3` definition unit tests methods.
    """

    def test_eotf_DCIP3(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.dci_p3.\
eotf_DCIP3` definition.
        """

        self.assertAlmostEqual(eotf_DCIP3(0.0), 0.0, places=7)

        self.assertAlmostEqual(eotf_DCIP3(461.99220597), 0.18, places=7)

        self.assertAlmostEqual(eotf_DCIP3(893.44598341), 1.0, places=7)

    def test_n_dimensional_eotf_DCIP3(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.dci_p3.\
eotf_DCIP3` definition n-dimensional arrays support.
        """

        XYZ_p = 461.99220597
        XYZ = 0.18
        np.testing.assert_almost_equal(eotf_DCIP3(XYZ_p), XYZ, decimal=7)

        XYZ_p = np.tile(XYZ_p, 6)
        XYZ = np.tile(XYZ, 6)
        np.testing.assert_almost_equal(eotf_DCIP3(XYZ_p), XYZ, decimal=7)

        XYZ_p = np.reshape(XYZ_p, (2, 3))
        XYZ = np.reshape(XYZ, (2, 3))
        np.testing.assert_almost_equal(eotf_DCIP3(XYZ_p), XYZ, decimal=7)

        XYZ_p = np.reshape(XYZ_p, (2, 3, 1))
        XYZ = np.reshape(XYZ, (2, 3, 1))
        np.testing.assert_almost_equal(eotf_DCIP3(XYZ_p), XYZ, decimal=7)

    def test_domain_range_scale_eotf_DCIP3(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.dci_p3.\
eotf_DCIP3` definition domain and range scale support.
        """

        XYZ_p = 426.0
        XYZ = eotf_DCIP3(XYZ_p)

        d_r = (
            ('reference', 1, 1),
            (1, 1 / 4095, 1),
            (100, 100 / 4095, 100),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    eotf_DCIP3(XYZ_p * factor_a), XYZ * factor_b, decimal=7)

    @ignore_numpy_errors
    def test_nan_eotf_DCIP3(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.dci_p3.\
eotf_DCIP3` definition nan support.
        """

        eotf_DCIP3(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


if __name__ == '__main__':
    unittest.main()
