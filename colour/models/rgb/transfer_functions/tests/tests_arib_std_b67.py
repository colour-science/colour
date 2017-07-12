#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.models.rgb.transfer_functions.arib_std_b67`
module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.models.rgb.transfer_functions import (oetf_ARIBSTDB67,
                                                  eotf_ARIBSTDB67)
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestOetf_ARIBSTDB67', 'TestEotf_ARIBSTDB67']


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
        E_p = 0.212132034355964
        np.testing.assert_almost_equal(oetf_ARIBSTDB67(E), E_p, decimal=7)

        E = np.tile(E, 6)
        E_p = np.tile(E_p, 6)
        np.testing.assert_almost_equal(oetf_ARIBSTDB67(E), E_p, decimal=7)

        E = np.reshape(E, (2, 3))
        E_p = np.reshape(E_p, (2, 3))
        np.testing.assert_almost_equal(oetf_ARIBSTDB67(E), E_p, decimal=7)

        E = np.reshape(E, (2, 3, 1))
        E_p = np.reshape(E_p, (2, 3, 1))
        np.testing.assert_almost_equal(oetf_ARIBSTDB67(E), E_p, decimal=7)

    @ignore_numpy_errors
    def test_nan_oetf_ARIBSTDB67(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.arib_std_b67.\
oetf_ARIBSTDB67` definition nan support.
        """

        oetf_ARIBSTDB67(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestEotf_ARIBSTDB67(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.arib_std_b67.\
eotf_ARIBSTDB67` definition unit tests methods.
    """

    def test_eotf_ARIBSTDB67(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.arib_std_b67.\
eotf_ARIBSTDB67` definition.
        """

        self.assertAlmostEqual(eotf_ARIBSTDB67(0.0), 0.0, places=7)

        self.assertAlmostEqual(
            eotf_ARIBSTDB67(0.212132034355964), 0.18, places=7)

        self.assertAlmostEqual(eotf_ARIBSTDB67(0.5), 1.0, places=7)

        self.assertAlmostEqual(
            eotf_ARIBSTDB67(1.302858098046995), 64.0, places=7)

    def test_n_dimensional_eotf_ARIBSTDB67(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.arib_std_b67.\
eotf_ARIBSTDB67` definition n-dimensional arrays support.
        """

        E_p = 0.212132034355964
        E = 0.18
        np.testing.assert_almost_equal(eotf_ARIBSTDB67(E_p), E, decimal=7)

        E_p = np.tile(E_p, 6)
        E = np.tile(E, 6)
        np.testing.assert_almost_equal(eotf_ARIBSTDB67(E_p), E, decimal=7)

        E_p = np.reshape(E_p, (2, 3))
        E = np.reshape(E, (2, 3))
        np.testing.assert_almost_equal(eotf_ARIBSTDB67(E_p), E, decimal=7)

        E_p = np.reshape(E_p, (2, 3, 1))
        E = np.reshape(E, (2, 3, 1))
        np.testing.assert_almost_equal(eotf_ARIBSTDB67(E_p), E, decimal=7)

    @ignore_numpy_errors
    def test_nan_eotf_ARIBSTDB67(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.arib_std_b67.\
eotf_ARIBSTDB67` definition nan support.
        """

        eotf_ARIBSTDB67(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


if __name__ == '__main__':
    unittest.main()
