# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.models.rgb.transfer_functions.\
filmlight_tlog` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.models.rgb.transfer_functions import (log_encoding_FilmLightTLog,
                                                  log_decoding_FilmLightTLog)
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestLogEncoding_FilmLightTLog', 'TestLogDecoding_FilmLightTLog']


class TestLogEncoding_FilmLightTLog(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.filmlight_tlog.\
log_encoding_FilmLightTLog` definition unit tests methods.
    """

    def test_log_encoding_FilmLightTLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.filmlight_tlog.\
log_encoding_FilmLightTLog` definition.
        """

        self.assertAlmostEqual(
            log_encoding_FilmLightTLog(0.0), 0.075, places=7)

        self.assertAlmostEqual(
            log_encoding_FilmLightTLog(0.18), 0.396567801298332, places=7)

        self.assertAlmostEqual(
            log_encoding_FilmLightTLog(1.0), 0.552537881005859, places=7)

    def test_n_dimensional_log_encoding_TLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.filmlight_tlog.\
log_encoding_FilmLightTLog` definition n-dimensional arrays support.
        """

        x = 0.18
        t = 0.396567801298332
        np.testing.assert_almost_equal(
            log_encoding_FilmLightTLog(x), t, decimal=7)

        x = np.tile(x, 6)
        t = np.tile(t, 6)
        np.testing.assert_almost_equal(
            log_encoding_FilmLightTLog(x), t, decimal=7)

        x = np.reshape(x, (2, 3))
        t = np.reshape(t, (2, 3))
        np.testing.assert_almost_equal(
            log_encoding_FilmLightTLog(x), t, decimal=7)

        x = np.reshape(x, (2, 3, 1))
        t = np.reshape(t, (2, 3, 1))
        np.testing.assert_almost_equal(
            log_encoding_FilmLightTLog(x), t, decimal=7)

    def test_domain_range_scale_log_encoding_TLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.filmlight_tlog.\
log_encoding_FilmLightTLog` definition domain and range scale support.
        """

        x = 0.18
        t = log_encoding_FilmLightTLog(x)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    log_encoding_FilmLightTLog(x * factor),
                    t * factor,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_log_encoding_TLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.filmlight_tlog.\
log_encoding_FilmLightTLog` definition nan support.
        """

        log_encoding_FilmLightTLog(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_FilmLightTLog(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.filmlight_tlog.\
log_decoding_FilmLightTLog` definition unit tests methods.
    """

    def test_log_decoding_FilmLightTLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.filmlight_tlog.\
log_decoding_FilmLightTLog` definition.
        """

        self.assertAlmostEqual(
            log_decoding_FilmLightTLog(0.075), 0.0, places=7)

        self.assertAlmostEqual(
            log_decoding_FilmLightTLog(0.396567801298332), 0.18, places=7)

        self.assertAlmostEqual(
            log_decoding_FilmLightTLog(0.552537881005859), 1.0, places=7)

    def test_n_dimensional_log_decoding_TLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.filmlight_tlog.\
log_decoding_FilmLightTLog` definition n-dimensional arrays support.
        """

        t = 0.396567801298332
        x = 0.18
        np.testing.assert_almost_equal(
            log_decoding_FilmLightTLog(t), x, decimal=7)

        t = np.tile(t, 6)
        x = np.tile(x, 6)
        np.testing.assert_almost_equal(
            log_decoding_FilmLightTLog(t), x, decimal=7)

        t = np.reshape(t, (2, 3))
        x = np.reshape(x, (2, 3))
        np.testing.assert_almost_equal(
            log_decoding_FilmLightTLog(t), x, decimal=7)

        t = np.reshape(t, (2, 3, 1))
        x = np.reshape(x, (2, 3, 1))
        np.testing.assert_almost_equal(
            log_decoding_FilmLightTLog(t), x, decimal=7)

    def test_domain_range_scale_log_decoding_TLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.filmlight_tlog.\
log_decoding_FilmLightTLog` definition domain and range scale support.
        """

        t = 0.396567801298332
        x = log_decoding_FilmLightTLog(t)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    log_decoding_FilmLightTLog(t * factor),
                    x * factor,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_log_decoding_TLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.filmlight_tlog.\
log_decoding_FilmLightTLog` definition nan support.
        """

        log_decoding_FilmLightTLog(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


if __name__ == '__main__':
    unittest.main()
