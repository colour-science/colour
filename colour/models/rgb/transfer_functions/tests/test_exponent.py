#
"""
Defines unit tests for :mod:`colour.models.rgb.transfer_functions.exponent`
module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.models.rgb.transfer_functions import (
    exponent_function_basic, exponent_function_monitor_curve)
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['TestBasicExponentFunction', 'TestMonitorCurveExponentFunction']


class TestBasicExponentFunction(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.exponent.\
exponent_function_basic` definition unit tests methods.
    """

    def test_exponent_function_basic(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.exponent.\
exponent_function_basic` definition.
        """

        a = 2.0
        a_p = 4.0
        self.assertAlmostEqual(exponent_function_basic(a, 2.0), a_p, places=7)

        self.assertAlmostEqual(
            exponent_function_basic(a, 2.0, 'basicMirrorFwd'), a_p, places=7)

        self.assertAlmostEqual(
            exponent_function_basic(a, 2.0, 'basicPassThruFwd'), a_p, places=7)

        a = 4.0
        a_p = 2.0

        self.assertAlmostEqual(
            exponent_function_basic(a, 2.0, 'basicRev'), a_p, places=7)

        self.assertAlmostEqual(
            exponent_function_basic(a, 2.0, 'basicMirrorRev'), a_p, places=7)

        self.assertAlmostEqual(
            exponent_function_basic(a, 2.0, 'basicPassThruRev'), a_p, places=7)

        a = -2.0
        self.assertAlmostEqual(exponent_function_basic(a, 2.0), 0.0, places=7)

        self.assertAlmostEqual(
            exponent_function_basic(a, 2.0, 'basicMirrorFwd'), -4.0, places=7)

        self.assertAlmostEqual(
            exponent_function_basic(a, 2.0, 'basicPassThruFwd'),
            -2.0,
            places=7)

        a = -4.0
        self.assertAlmostEqual(
            exponent_function_basic(a, 2.0, 'basicRev'), 0.0, places=7)

        self.assertAlmostEqual(
            exponent_function_basic(a, 2.0, 'basicMirrorRev'), -2.0, places=7)

        self.assertAlmostEqual(
            exponent_function_basic(a, 2.0, 'basicPassThruRev'),
            -4.0,
            places=7)

    def test_n_dimensional_exponent_function_basic(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.exponent.\
exponent_function_basic` definition n-dimensional arrays support.
        """

        a = 0.18
        a_p = 0.022993204992706778
        np.testing.assert_almost_equal(
            exponent_function_basic(a, 2.2), a_p, decimal=7)
        np.testing.assert_almost_equal(
            exponent_function_basic(a, 2.2, 'basicMirrorFwd'), a_p, decimal=7)
        np.testing.assert_almost_equal(
            exponent_function_basic(a, 2.2, 'basicPassThruFwd'),
            a_p,
            decimal=7)

        a = np.tile(a, 6)
        a_p = np.tile(a_p, 6)
        np.testing.assert_almost_equal(
            exponent_function_basic(a, 2.2), a_p, decimal=7)
        np.testing.assert_almost_equal(
            exponent_function_basic(a, 2.2, 'basicMirrorFwd'), a_p, decimal=7)
        np.testing.assert_almost_equal(
            exponent_function_basic(a, 2.2, 'basicPassThruFwd'),
            a_p,
            decimal=7)

        a = np.reshape(a, (2, 3))
        a_p = np.reshape(a_p, (2, 3))
        np.testing.assert_almost_equal(
            exponent_function_basic(a, 2.2), a_p, decimal=7)
        np.testing.assert_almost_equal(
            exponent_function_basic(a, 2.2, 'basicMirrorFwd'), a_p, decimal=7)
        np.testing.assert_almost_equal(
            exponent_function_basic(a, 2.2, 'basicPassThruFwd'),
            a_p,
            decimal=7)

        a = np.reshape(a, (2, 3, 1))
        a_p = np.reshape(a_p, (2, 3, 1))
        np.testing.assert_almost_equal(
            exponent_function_basic(a, 2.2), a_p, decimal=7)
        np.testing.assert_almost_equal(
            exponent_function_basic(a, 2.2, 'basicMirrorFwd'), a_p, decimal=7)
        np.testing.assert_almost_equal(
            exponent_function_basic(a, 2.2, 'basicPassThruFwd'),
            a_p,
            decimal=7)

        a = 0.18
        a_p = 0.4586564468643811
        np.testing.assert_almost_equal(
            exponent_function_basic(a, 2.2, 'basicRev'), a_p, decimal=7)
        np.testing.assert_almost_equal(
            exponent_function_basic(a, 2.2, 'basicMirrorRev'), a_p, decimal=7)
        np.testing.assert_almost_equal(
            exponent_function_basic(a, 2.2, 'basicPassThruRev'),
            a_p,
            decimal=7)

        a = np.tile(a, 6)
        a_p = np.tile(a_p, 6)
        np.testing.assert_almost_equal(
            exponent_function_basic(a, 2.2, 'basicRev'), a_p, decimal=7)
        np.testing.assert_almost_equal(
            exponent_function_basic(a, 2.2, 'basicMirrorRev'), a_p, decimal=7)
        np.testing.assert_almost_equal(
            exponent_function_basic(a, 2.2, 'basicPassThruRev'),
            a_p,
            decimal=7)

        a = np.reshape(a, (2, 3))
        a_p = np.reshape(a_p, (2, 3))
        np.testing.assert_almost_equal(
            exponent_function_basic(a, 2.2, 'basicRev'), a_p, decimal=7)
        np.testing.assert_almost_equal(
            exponent_function_basic(a, 2.2, 'basicMirrorRev'), a_p, decimal=7)
        np.testing.assert_almost_equal(
            exponent_function_basic(a, 2.2, 'basicPassThruRev'),
            a_p,
            decimal=7)

        a = np.reshape(a, (2, 3, 1))
        a_p = np.reshape(a_p, (2, 3, 1))
        np.testing.assert_almost_equal(
            exponent_function_basic(a, 2.2, 'basicRev'), a_p, decimal=7)
        np.testing.assert_almost_equal(
            exponent_function_basic(a, 2.2, 'basicMirrorRev'), a_p, decimal=7)
        np.testing.assert_almost_equal(
            exponent_function_basic(a, 2.2, 'basicPassThruRev'),
            a_p,
            decimal=7)

    def test_raise_exception_exponent_function_basic(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.exponent.\
exponent_function_basic` definition raised exception.
        """

        self.assertRaises(ValueError, exponent_function_basic, 0.18, 1,
                          'Undefined')

    @ignore_numpy_errors
    def test_nan_exponent_function_basic(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.exponent.\
exponent_function_basic` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]

        for case in cases:
            exponent_function_basic(case, case)


class TestMonitorCurveExponentFunction(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.exponent.\
exponent_function_monitor_curve` definition unit tests methods.
    """

    def test_exponent_function_monitor_curve(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.exponent.\
exponent_function_monitor_curve` definition.
        """

        a = 2.0
        a_p = 1.7777777777777777
        self.assertAlmostEqual(
            exponent_function_monitor_curve(a, 2.0, 2.0), a_p, places=7)

        self.assertAlmostEqual(
            exponent_function_monitor_curve(a, 2.0, 2.0, 'monCurveMirrorFwd'),
            a_p,
            places=7)

        a = 1.7777777777777777
        a_p = 2.0
        self.assertAlmostEqual(
            exponent_function_monitor_curve(a, 2.0, 2.0, 'monCurveRev'),
            a_p,
            places=7)

        self.assertAlmostEqual(
            exponent_function_monitor_curve(a, 2.0, 2.0, 'monCurveMirrorRev'),
            a_p,
            places=7)

        a = -2.0
        a_p = -1.7777777777777777
        self.assertAlmostEqual(
            exponent_function_monitor_curve(a, 2.0, 2.0), a_p, places=7)

        self.assertAlmostEqual(
            exponent_function_monitor_curve(a, 2.0, 2.0, 'monCurveMirrorFwd'),
            a_p,
            places=7)

        a = -1.7777777777777777
        a_p = -2.0
        self.assertAlmostEqual(
            exponent_function_monitor_curve(a, 2.0, 2.0, 'monCurveRev'),
            a_p,
            places=7)

        self.assertAlmostEqual(
            exponent_function_monitor_curve(a, 2.0, 2.0, 'monCurveMirrorRev'),
            a_p,
            places=7)

    def test_n_dimensional_exponent_function_monitor_curve(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.exponent.\
exponent_function_monitor_curve` definition n-dimensional arrays support.
        """

        a = 0.18
        a_p = 0.1679399973777068
        np.testing.assert_almost_equal(
            exponent_function_monitor_curve(a, 2.2, 2.0), a_p, decimal=7)
        np.testing.assert_almost_equal(
            exponent_function_monitor_curve(a, 2.2, 2.0, 'monCurveMirrorFwd'),
            a_p,
            decimal=7)

        a = np.tile(a, 6)
        a_p = np.tile(a_p, 6)
        np.testing.assert_almost_equal(
            exponent_function_monitor_curve(a, 2.2, 2.0), a_p, decimal=7)
        np.testing.assert_almost_equal(
            exponent_function_monitor_curve(a, 2.2, 2.0, 'monCurveMirrorFwd'),
            a_p,
            decimal=7)

        a = np.reshape(a, (2, 3))
        a_p = np.reshape(a_p, (2, 3))
        np.testing.assert_almost_equal(
            exponent_function_monitor_curve(a, 2.2, 2.0), a_p, decimal=7)
        np.testing.assert_almost_equal(
            exponent_function_monitor_curve(a, 2.2, 2.0, 'monCurveMirrorFwd'),
            a_p,
            decimal=7)

        a = np.reshape(a, (2, 3, 1))
        a_p = np.reshape(a_p, (2, 3, 1))
        np.testing.assert_almost_equal(
            exponent_function_monitor_curve(a, 2.2, 2.0), a_p, decimal=7)
        np.testing.assert_almost_equal(
            exponent_function_monitor_curve(a, 2.2, 2.0, 'monCurveMirrorFwd'),
            a_p,
            decimal=7)

        a = -0.18
        a_p = -0.19292604802851415
        np.testing.assert_almost_equal(
            exponent_function_monitor_curve(a, 2.2, 2.0, 'monCurveRev'),
            a_p,
            decimal=7)
        np.testing.assert_almost_equal(
            exponent_function_monitor_curve(a, 2.2, 2.0, 'monCurveMirrorRev'),
            a_p,
            decimal=7)

        a = np.tile(a, 6)
        a_p = np.tile(a_p, 6)
        np.testing.assert_almost_equal(
            exponent_function_monitor_curve(a, 2.2, 2.0, 'monCurveRev'),
            a_p,
            decimal=7)
        np.testing.assert_almost_equal(
            exponent_function_monitor_curve(a, 2.2, 2.0, 'monCurveMirrorRev'),
            a_p,
            decimal=7)

        a = np.reshape(a, (2, 3))
        a_p = np.reshape(a_p, (2, 3))
        np.testing.assert_almost_equal(
            exponent_function_monitor_curve(a, 2.2, 2.0, 'monCurveRev'),
            a_p,
            decimal=7)
        np.testing.assert_almost_equal(
            exponent_function_monitor_curve(a, 2.2, 2.0, 'monCurveMirrorRev'),
            a_p,
            decimal=7)

        a = np.reshape(a, (2, 3, 1))
        a_p = np.reshape(a_p, (2, 3, 1))
        np.testing.assert_almost_equal(
            exponent_function_monitor_curve(a, 2.2, 2.0, 'monCurveRev'),
            a_p,
            decimal=7)
        np.testing.assert_almost_equal(
            exponent_function_monitor_curve(a, 2.2, 2.0, 'monCurveMirrorRev'),
            a_p,
            decimal=7)

    def test_raise_exception_exponent_function_monitor_curve(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.exponent.\
exponent_function_monitor_curve` definition raised exception.
        """

        self.assertRaises(ValueError, exponent_function_monitor_curve, 0.18, 1,
                          'Undefined')

    @ignore_numpy_errors
    def test_nan_exponent_function_monitor_curve(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.exponent.\
exponent_function_monitor_curve` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]

        for case in cases:
            exponent_function_monitor_curve(case, case, case)


if __name__ == '__main__':
    unittest.main()
