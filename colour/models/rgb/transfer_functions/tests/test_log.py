# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.models.rgb.transfer_functions.log` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.models.rgb.transfer_functions import (
    logarithmic_function_basic, logarithmic_function_quasilog,
    logarithmic_function_camera, log_encoding_Log2, log_decoding_Log2)
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestLogarithmFunction_Basic', 'TestLogarithmFunction_Quasilog',
    'TestLogarithmFunction_Camera', 'TestLogEncoding_Log2',
    'TestLogDecoding_Log2'
]


class TestLogarithmFunction_Basic(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.log.\
logarithmic_function_basic` definition unit tests methods.
    """

    def test_logarithmic_function_basic(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.log.\
logarithmic_function_basic` definition.
        """

        self.assertAlmostEqual(
            logarithmic_function_basic(0.18), -2.47393118833, places=7)

        self.assertAlmostEqual(
            logarithmic_function_basic(0.18, 10, 'log10'),
            -0.744727494897,
            places=7)

        self.assertAlmostEqual(
            logarithmic_function_basic(0.18, 2.2, 'logN'),
            -2.174877823830,
            places=7)

        self.assertAlmostEqual(
            logarithmic_function_basic(-0.744727494897, 10, 'antiLog10'),
            0.179999999999,
            places=7)

        self.assertAlmostEqual(
            logarithmic_function_basic(-2.47393118833, 2, 'antiLog2'),
            0.180000000000,
            places=7)

        self.assertAlmostEqual(
            logarithmic_function_basic(-2.174877823830, 2.2, 'antiLogN'),
            0.180000000000,
            places=7)


class TestLogarithmFunction_Quasilog(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.log.\
logarithmic_function_quasilog` definition unit tests methods.
    """

    def test_logarithmic_function_quasilog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.log.\
logarithmic_function_quasilog` definition.
        """

        self.assertAlmostEqual(
            logarithmic_function_quasilog(0.18), -2.47393118833, places=7)

        self.assertAlmostEqual(
            logarithmic_function_quasilog(0.18, 2.2, 'linToLog'),
            -2.17487782383,
            places=7)

        self.assertAlmostEqual(
            logarithmic_function_quasilog(0.18, 2.2, 'linToLog', 0.001),
            -0.002174877823,
            places=7)

        self.assertAlmostEqual(
            logarithmic_function_quasilog(0.18, 2.2, 'linToLog', 0.001, 0.12),
            -0.0048640068025,
            places=7)

        self.assertAlmostEqual(
            logarithmic_function_quasilog(0.18, 2.2, 'linToLog', 0.001, 0.12,
                                          0.001),
            -0.003864006802,
            places=7)

        self.assertAlmostEqual(
            logarithmic_function_quasilog(0.18, 2.2, 'linToLog', 0.001, 0.12,
                                          0.001, 0.12),
            -0.001479207115,
            places=7)

        self.assertAlmostEqual(
            logarithmic_function_quasilog(-2.47393118833, 2, 'logToLin'),
            0.18,
            places=7)

        self.assertAlmostEqual(
            logarithmic_function_quasilog(-2.17487782383, 2.2, 'logToLin'),
            0.18,
            places=7)

        self.assertAlmostEqual(
            logarithmic_function_quasilog(-0.002174877823, 2.2, 'logToLin',
                                          0.001),
            0.18,
            places=7)

        self.assertAlmostEqual(
            logarithmic_function_quasilog(-0.004864006802, 2.2, 'logToLin',
                                          0.001, 0.12),
            0.18,
            places=7)

        self.assertAlmostEqual(
            logarithmic_function_quasilog(-0.003864006802, 2.2, 'logToLin',
                                          0.001, 0.12, 0.001),
            0.18,
            places=7)

        self.assertAlmostEqual(
            logarithmic_function_quasilog(-0.001479207115, 2.2, 'logToLin',
                                          0.001, 0.12, 0.001, 0.12),
            0.18,
            places=7)


class TestLogarithmFunction_Camera(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.log.\
logarithmic_function_camera` definition unit tests methods.
    """

    def test_logarithmic_function_camera(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.log.\
logarithmic_function_camera` definition.
        """

        self.assertAlmostEqual(
            logarithmic_function_camera(0.18, 2, 'cameraLinToLog', 2.2),
            -0.187152831975,
            places=7)

        self.assertAlmostEqual(
            logarithmic_function_camera(0.18, 2.2, 'cameraLinToLog', 2.2),
            -0.164529452496,
            places=7)

        self.assertAlmostEqual(
            logarithmic_function_camera(0.18, 2.2, 'cameraLinToLog', 2.2,
                                        0.001),
            -0.000164529452,
            places=7)

        self.assertAlmostEqual(
            logarithmic_function_camera(0.18, 2.2, 'cameraLinToLog', 2.2,
                                        0.001, 0.001),
            -0.008925631353,
            places=7)

        self.assertAlmostEqual(
            logarithmic_function_camera(0.18, 2.2, 'cameraLinToLog', 2.2,
                                        0.001, 0.001, 0.12),
            0.111074368646,
            places=7)

        self.assertAlmostEqual(
            logarithmic_function_camera(0.18, 2.2, 'cameraLinToLog', 2.2,
                                        0.001, 0.001, 0.12, 0.12),
            0.11731294726,
            places=7)

        self.assertAlmostEqual(
            logarithmic_function_camera(-0.187152831975, 2, 'cameraLogToLin',
                                        2.2),
            0.180000000001,
            places=7)

        self.assertAlmostEqual(
            logarithmic_function_camera(-0.164529452496, 2.2, 'cameraLogToLin',
                                        2.2),
            0.180000000001,
            places=7)

        self.assertAlmostEqual(
            logarithmic_function_camera(-0.000164529452, 2.2, 'cameraLogToLin',
                                        2.2, 0.001),
            0.180000000001,
            places=7)

        self.assertAlmostEqual(
            logarithmic_function_camera(-0.008925631353, 2.2, 'cameraLogToLin',
                                        2.2, 0.001, 0.001),
            0.180000000001,
            places=7)

        self.assertAlmostEqual(
            logarithmic_function_camera(0.111074368646, 2.2, 'cameraLogToLin',
                                        2.2, 0.001, 0.001, 0.12),
            0.179999999649,
            places=7)

        self.assertAlmostEqual(
            logarithmic_function_camera(0.11731294726, 2.2, 'cameraLogToLin',
                                        2.2, 0.001, 0.001, 0.12, 0.12),
            0.17999999231,
            places=7)


class TestLogEncoding_Log2(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.log.\
log_encoding_Log2` definition unit tests methods.
    """

    def test_log_encoding_Log2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.log.\
log_encoding_Log2` definition.
        """

        self.assertAlmostEqual(log_encoding_Log2(0.0), -np.inf, places=7)

        self.assertAlmostEqual(log_encoding_Log2(0.18), 0.5, places=7)

        self.assertAlmostEqual(
            log_encoding_Log2(1.0), 0.690302399102493, places=7)

        self.assertAlmostEqual(
            log_encoding_Log2(0.18, 0.12), 0.544997115440089, places=7)

        self.assertAlmostEqual(
            log_encoding_Log2(0.18, 0.12, 2 ** -10),
            0.089857490719529,
            places=7)

        self.assertAlmostEqual(
            log_encoding_Log2(0.18, 0.12, 2 ** -10, 2 ** 10),
            0.000570299311674,
            places=7)

    def test_n_dimensional_log_encoding_Log2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.log.\
log_encoding_Log2` definition n-dimensional arrays support.
        """

        x = 0.18
        y = log_encoding_Log2(x)

        x = np.tile(x, 6)
        y = np.tile(y, 6)
        np.testing.assert_almost_equal(log_encoding_Log2(x), y, decimal=7)

        x = np.reshape(x, (2, 3))
        y = np.reshape(y, (2, 3))
        np.testing.assert_almost_equal(log_encoding_Log2(x), y, decimal=7)

        x = np.reshape(x, (2, 3, 1))
        y = np.reshape(y, (2, 3, 1))
        np.testing.assert_almost_equal(log_encoding_Log2(x), y, decimal=7)

    def test_domain_range_scale_log_encoding_Log2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.log.\
log_encoding_Log2` definition domain and range scale support.
        """

        x = 0.18
        y = log_encoding_Log2(x)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    log_encoding_Log2(x * factor), y * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_log_encoding_Log2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.log.\
log_encoding_Log2` definition nan support.
        """

        log_encoding_Log2(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_Log2(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.log.\
log_decoding_Log2` definition unit tests methods.
    """

    def test_log_decoding_Log2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.log.\
log_decoding_Log2` definition.
        """

        self.assertAlmostEqual(
            log_decoding_Log2(0.0), 0.001988737822087, places=7)

        self.assertAlmostEqual(log_decoding_Log2(0.5), 0.18, places=7)

        self.assertAlmostEqual(
            log_decoding_Log2(0.690302399102493), 1.0, places=7)

        self.assertAlmostEqual(
            log_decoding_Log2(0.544997115440089, 0.12), 0.18, places=7)

        self.assertAlmostEqual(
            log_decoding_Log2(0.089857490719529, 0.12, 2 ** -10),
            0.180000000000000,
            places=7)

        self.assertAlmostEqual(
            log_decoding_Log2(0.000570299311674, 0.12, 2 ** -10, 2 ** 10),
            0.180000000000000,
            places=7)

    def test_n_dimensional_log_decoding_Log2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.log.\
log_decoding_Log2` definition n-dimensional arrays support.
        """

        y = 0.5
        x = log_decoding_Log2(y)

        y = np.tile(y, 6)
        x = np.tile(x, 6)
        np.testing.assert_almost_equal(log_decoding_Log2(y), x, decimal=7)

        y = np.reshape(y, (2, 3))
        x = np.reshape(x, (2, 3))
        np.testing.assert_almost_equal(log_decoding_Log2(y), x, decimal=7)

        y = np.reshape(y, (2, 3, 1))
        x = np.reshape(x, (2, 3, 1))
        np.testing.assert_almost_equal(log_decoding_Log2(y), x, decimal=7)

    def test_domain_range_scale_log_decoding_Log2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.log.\
log_decoding_Log2` definition domain and range scale support.
        """

        y = 0.5
        x = log_decoding_Log2(y)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    log_decoding_Log2(y * factor), x * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_log_decoding_Log2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.log.\
log_decoding_Log2` definition nan support.
        """

        log_decoding_Log2(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


if __name__ == '__main__':
    unittest.main()
