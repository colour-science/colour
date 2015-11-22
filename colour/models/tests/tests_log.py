#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.models.log` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.models import (
    linear_to_cineon,
    cineon_to_linear,
    linear_to_panalog,
    panalog_to_linear,
    linear_to_viper_log,
    viper_log_to_linear,
    linear_to_pivoted_log,
    pivoted_log_to_linear,
    linear_to_c_log,
    c_log_to_linear,
    linear_to_aces_cc,
    aces_cc_to_linear,
    linear_to_alexa_log_c,
    alexa_log_c_to_linear,
    linear_to_dci_p3_log,
    dci_p3_log_to_linear,
    linear_to_red_log_film,
    red_log_film_to_linear,
    linear_to_s_log,
    s_log_to_linear,
    linear_to_s_log2,
    s_log2_to_linear,
    linear_to_s_log3,
    s_log3_to_linear,
    linear_to_v_log,
    v_log_to_linear)
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestLinearToCineon',
           'TestCineonToLinear',
           'TestLinearToPanalog',
           'TestPanalogToLinear',
           'TestLinearToRedLogFilm',
           'TestRedLogFilmToLinear',
           'TestLinearToViperLog',
           'TestViperLogToLinear',
           'TestLinearToPivotedLog',
           'TestPivotedLogToLinear',
           'TestLinearToCLog',
           'TestCLogToLinear',
           'TestLinearToAcesCc',
           'TestAcesCcToLinear',
           'TestLinearToAlexaLogC',
           'TestAlexaLogCToLinear',
           'TestLinearToDciP3Log',
           'TestDciP3LogToLinear',
           'TestLinearToSLog',
           'TestSLogToLinear',
           'TestLinearToSLog2',
           'TestSLog2ToLinear',
           'TestLinearToSLog3',
           'TestSLog3ToLinear',
           'TestLinearToVLog',
           'TestVLogToLinear']


class TestLinearToCineon(unittest.TestCase):
    """
    Defines :func:`colour.models.log.linear_to_cineon` definition unit tests
    methods.
    """

    def test_linear_to_cineon(self):
        """
        Tests :func:`colour.models.log.linear_to_cineon` definition.
        """

        self.assertAlmostEqual(
            linear_to_cineon(0.00),
            0.092864125122189639,
            places=7)

        self.assertAlmostEqual(
            linear_to_cineon(0.18),
            0.45731961308541841,
            places=7)

        self.assertAlmostEqual(
            linear_to_cineon(1.00),
            0.66959921798631472,
            places=7)

    def test_n_dimensional_linear_to_cineon(self):
        """
        Tests :func:`colour.models.log.linear_to_cineon` definition
        n-dimensional arrays support.
        """

        linear = 0.18
        log = 0.45731961308541841
        np.testing.assert_almost_equal(
            linear_to_cineon(linear),
            log,
            decimal=7)

        linear = np.tile(linear, 6)
        log = np.tile(log, 6)
        np.testing.assert_almost_equal(
            linear_to_cineon(linear),
            log,
            decimal=7)

        linear = np.reshape(linear, (2, 3))
        log = np.reshape(log, (2, 3))
        np.testing.assert_almost_equal(
            linear_to_cineon(linear),
            log,
            decimal=7)

        linear = np.reshape(linear, (2, 3, 1))
        log = np.reshape(log, (2, 3, 1))
        np.testing.assert_almost_equal(
            linear_to_cineon(linear),
            log,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_linear_to_cineon(self):
        """
        Tests :func:`colour.models.log.linear_to_cineon` definition nan
        support.
        """

        linear_to_cineon(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestCineonToLinear(unittest.TestCase):
    """
    Defines :func:`colour.models.log.cineon_to_linear` definition unit tests
    methods.
    """

    def test_cineon_to_linear(self):
        """
        Tests :func:`colour.models.log.cineon_to_linear` definition.
        """

        self.assertAlmostEqual(
            cineon_to_linear(0.092864125122189639),
            0.,
            places=7)

        self.assertAlmostEqual(
            cineon_to_linear(0.45731961308541841),
            0.18,
            places=7)

        self.assertAlmostEqual(
            cineon_to_linear(0.66959921798631472),
            1.,
            places=7)

    def test_n_dimensional_cineon_to_linear(self):
        """
        Tests :func:`colour.models.log.cineon_to_linear` definition
        n-dimensional arrays support.
        """

        log = 0.45731961308541841
        linear = 0.18
        np.testing.assert_almost_equal(
            cineon_to_linear(log),
            linear,
            decimal=7)

        log = np.tile(log, 6)
        linear = np.tile(linear, 6)
        np.testing.assert_almost_equal(
            cineon_to_linear(log),
            linear,
            decimal=7)

        log = np.reshape(log, (2, 3))
        linear = np.reshape(linear, (2, 3))
        np.testing.assert_almost_equal(
            cineon_to_linear(log),
            linear,
            decimal=7)

        log = np.reshape(log, (2, 3, 1))
        linear = np.reshape(linear, (2, 3, 1))
        np.testing.assert_almost_equal(
            cineon_to_linear(log),
            linear,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_cineon_to_linear(self):
        """
        Tests :func:`colour.models.log.cineon_to_linear` definition nan
        support.
        """

        cineon_to_linear(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLinearToPanalog(unittest.TestCase):
    """
    Defines :func:`colour.models.log.linear_to_panalog` definition unit tests
    methods.
    """

    def test_linear_to_panalog(self):
        """
        Tests :func:`colour.models.log.linear_to_panalog` definition.
        """

        self.assertAlmostEqual(
            linear_to_panalog(0.00),
            0.062561094819159335,
            places=7)

        self.assertAlmostEqual(
            linear_to_panalog(0.18),
            0.37457679138229816,
            places=7)

        self.assertAlmostEqual(
            linear_to_panalog(1.00),
            0.66568914956011727,
            places=7)

    def test_n_dimensional_linear_to_panalog(self):
        """
        Tests :func:`colour.models.log.linear_to_panalog` definition
        n-dimensional arrays support.
        """

        linear = 0.18
        log = 0.37457679138229816
        np.testing.assert_almost_equal(
            linear_to_panalog(linear),
            log,
            decimal=7)

        linear = np.tile(linear, 6)
        log = np.tile(log, 6)
        np.testing.assert_almost_equal(
            linear_to_panalog(linear),
            log,
            decimal=7)

        linear = np.reshape(linear, (2, 3))
        log = np.reshape(log, (2, 3))
        np.testing.assert_almost_equal(
            linear_to_panalog(linear),
            log,
            decimal=7)

        linear = np.reshape(linear, (2, 3, 1))
        log = np.reshape(log, (2, 3, 1))
        np.testing.assert_almost_equal(
            linear_to_panalog(linear),
            log,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_linear_to_panalog(self):
        """
        Tests :func:`colour.models.log.linear_to_panalog` definition nan
        support.
        """

        linear_to_panalog(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestPanalogToLinear(unittest.TestCase):
    """
    Defines :func:`colour.models.log.panalog_to_linear` definition unit tests
    methods.
    """

    def test_panalog_to_linear(self):
        """
        Tests :func:`colour.models.log.panalog_to_linear` definition.
        """

        self.assertAlmostEqual(
            panalog_to_linear(0.062561094819159335),
            0.,
            places=7)

        self.assertAlmostEqual(
            panalog_to_linear(0.37457679138229816),
            0.18,
            places=7)

        self.assertAlmostEqual(
            panalog_to_linear(0.66568914956011727),
            1.,
            places=7)

    def test_n_dimensional_panalog_to_linear(self):
        """
        Tests :func:`colour.models.log.panalog_to_linear` definition
        n-dimensional arrays support.
        """

        log = 0.37457679138229816
        linear = 0.18
        np.testing.assert_almost_equal(
            panalog_to_linear(log),
            linear,
            decimal=7)

        log = np.tile(log, 6)
        linear = np.tile(linear, 6)
        np.testing.assert_almost_equal(
            panalog_to_linear(log),
            linear,
            decimal=7)

        log = np.reshape(log, (2, 3))
        linear = np.reshape(linear, (2, 3))
        np.testing.assert_almost_equal(
            panalog_to_linear(log),
            linear,
            decimal=7)

        log = np.reshape(log, (2, 3, 1))
        linear = np.reshape(linear, (2, 3, 1))
        np.testing.assert_almost_equal(
            panalog_to_linear(log),
            linear,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_panalog_to_linear(self):
        """
        Tests :func:`colour.models.log.panalog_to_linear` definition nan
        support.
        """

        panalog_to_linear(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLinearToViperLog(unittest.TestCase):
    """
    Defines :func:`colour.models.log.linear_to_viper_log` definition unit tests
    methods.
    """

    def test_linear_to_viper_log(self):
        """
        Tests :func:`colour.models.log.linear_to_viper_log` definition.
        """

        self.assertAlmostEqual(
            linear_to_viper_log(0.01),
            0.022482893450635387,
            places=7)

        self.assertAlmostEqual(
            linear_to_viper_log(0.18),
            0.63600806701041346,
            places=7)

        self.assertAlmostEqual(
            linear_to_viper_log(1.00),
            1.,
            places=7)

    def test_n_dimensional_linear_to_viper_log(self):
        """
        Tests :func:`colour.models.log.linear_to_viper_log` definition
        n-dimensional arrays support.
        """

        linear = 0.18
        log = 0.63600806701041346
        np.testing.assert_almost_equal(
            linear_to_viper_log(linear),
            log,
            decimal=7)

        linear = np.tile(linear, 6)
        log = np.tile(log, 6)
        np.testing.assert_almost_equal(
            linear_to_viper_log(linear),
            log,
            decimal=7)

        linear = np.reshape(linear, (2, 3))
        log = np.reshape(log, (2, 3))
        np.testing.assert_almost_equal(
            linear_to_viper_log(linear),
            log,
            decimal=7)

        linear = np.reshape(linear, (2, 3, 1))
        log = np.reshape(log, (2, 3, 1))
        np.testing.assert_almost_equal(
            linear_to_viper_log(linear),
            log,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_linear_to_viper_log(self):
        """
        Tests :func:`colour.models.log.linear_to_viper_log` definition nan
        support.
        """

        linear_to_viper_log(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestViperLogToLinear(unittest.TestCase):
    """
    Defines :func:`colour.models.log.viper_log_to_linear` definition unit tests
    methods.
    """

    def test_viper_log_to_linear(self):
        """
        Tests :func:`colour.models.log.viper_log_to_linear` definition.
        """

        self.assertAlmostEqual(
            viper_log_to_linear(0.022482893450635387),
            0.01,
            places=7)

        self.assertAlmostEqual(
            viper_log_to_linear(0.63600806701041346),
            0.18,
            places=7)

        self.assertAlmostEqual(
            viper_log_to_linear(1.00),
            1.,
            places=7)

    def test_n_dimensional_viper_log_to_linear(self):
        """
        Tests :func:`colour.models.log.viper_log_to_linear` definition
        n-dimensional arrays support.
        """

        log = 0.63600806701041346
        linear = 0.18
        np.testing.assert_almost_equal(
            viper_log_to_linear(log),
            linear,
            decimal=7)

        log = np.tile(log, 6)
        linear = np.tile(linear, 6)
        np.testing.assert_almost_equal(
            viper_log_to_linear(log),
            linear,
            decimal=7)

        log = np.reshape(log, (2, 3))
        linear = np.reshape(linear, (2, 3))
        np.testing.assert_almost_equal(
            viper_log_to_linear(log),
            linear,
            decimal=7)

        log = np.reshape(log, (2, 3, 1))
        linear = np.reshape(linear, (2, 3, 1))
        np.testing.assert_almost_equal(
            viper_log_to_linear(log),
            linear,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_viper_log_to_linear(self):
        """
        Tests :func:`colour.models.log.viper_log_to_linear` definition nan
        support.
        """

        viper_log_to_linear(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLinearToPivotedLog(unittest.TestCase):
    """
    Defines :func:`colour.models.log.linear_to_pivoted_log` definition unit
    tests methods.
    """

    def test_linear_to_pivoted_log(self):
        """
        Tests :func:`colour.models.log.linear_to_pivoted_log` definition.
        """

        self.assertAlmostEqual(
            linear_to_pivoted_log(0.01),
            0.066880008278600425,
            places=7)

        self.assertAlmostEqual(
            linear_to_pivoted_log(0.18),
            0.43499511241446726,
            places=7)

        self.assertAlmostEqual(
            linear_to_pivoted_log(1.00),
            0.6533902722082191,
            places=7)

    def test_n_dimensional_linear_to_pivoted_log(self):
        """
        Tests :func:`colour.models.log.linear_to_pivoted_log` definition
        n-dimensional arrays support.
        """

        linear = 0.18
        log = 0.43499511241446726
        np.testing.assert_almost_equal(
            linear_to_pivoted_log(linear),
            log,
            decimal=7)

        linear = np.tile(linear, 6)
        log = np.tile(log, 6)
        np.testing.assert_almost_equal(
            linear_to_pivoted_log(linear),
            log,
            decimal=7)

        linear = np.reshape(linear, (2, 3))
        log = np.reshape(log, (2, 3))
        np.testing.assert_almost_equal(
            linear_to_pivoted_log(linear),
            log,
            decimal=7)

        linear = np.reshape(linear, (2, 3, 1))
        log = np.reshape(log, (2, 3, 1))
        np.testing.assert_almost_equal(
            linear_to_pivoted_log(linear),
            log,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_linear_to_pivoted_log(self):
        """
        Tests :func:`colour.models.log.linear_to_pivoted_log` definition nan
        support.
        """

        linear_to_pivoted_log(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestPivotedLogToLinear(unittest.TestCase):
    """
    Defines :func:`colour.models.log.pivoted_log_to_linear` definition unit
    tests methods.
    """

    def test_pivoted_log_to_linear(self):
        """
        Tests :func:`colour.models.log.pivoted_log_to_linear` definition.
        """

        self.assertAlmostEqual(
            pivoted_log_to_linear(0.066880008278600425),
            0.01,
            places=7)

        self.assertAlmostEqual(
            pivoted_log_to_linear(0.43499511241446726),
            0.18,
            places=7)

        self.assertAlmostEqual(
            pivoted_log_to_linear(0.6533902722082191),
            1.,
            places=7)

    def test_n_dimensional_pivoted_log_to_linear(self):
        """
        Tests :func:`colour.models.log.pivoted_log_to_linear` definition
        n-dimensional arrays support.
        """

        log = 0.43499511241446726
        linear = 0.18
        np.testing.assert_almost_equal(
            pivoted_log_to_linear(log),
            linear,
            decimal=7)

        log = np.tile(log, 6)
        linear = np.tile(linear, 6)
        np.testing.assert_almost_equal(
            pivoted_log_to_linear(log),
            linear,
            decimal=7)

        log = np.reshape(log, (2, 3))
        linear = np.reshape(linear, (2, 3))
        np.testing.assert_almost_equal(
            pivoted_log_to_linear(log),
            linear,
            decimal=7)

        log = np.reshape(log, (2, 3, 1))
        linear = np.reshape(linear, (2, 3, 1))
        np.testing.assert_almost_equal(
            pivoted_log_to_linear(log),
            linear,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_pivoted_log_to_linear(self):
        """
        Tests :func:`colour.models.log.pivoted_log_to_linear` definition nan
        support.
        """

        pivoted_log_to_linear(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLinearToCLog(unittest.TestCase):
    """
    Defines :func:`colour.models.log.linear_to_c_log` definition unit tests
    methods.
    """

    def test_linear_to_c_log(self):
        """
        Tests :func:`colour.models.log.linear_to_c_log` definition.
        """

        self.assertAlmostEqual(
            linear_to_c_log(0.00),
            0.073059700000000005,
            places=7)

        self.assertAlmostEqual(
            linear_to_c_log(0.18),
            0.31201285555039493,
            places=7)

        self.assertAlmostEqual(
            linear_to_c_log(1.00),
            0.62740830453765284,
            places=7)

    def test_n_dimensional_linear_to_c_log(self):
        """
        Tests :func:`colour.models.log.linear_to_c_log` definition
        n-dimensional arrays support.
        """

        linear = 0.18
        log = 0.31201285555039493
        np.testing.assert_almost_equal(
            linear_to_c_log(linear),
            log,
            decimal=7)

        linear = np.tile(linear, 6)
        log = np.tile(log, 6)
        np.testing.assert_almost_equal(
            linear_to_c_log(linear),
            log,
            decimal=7)

        linear = np.reshape(linear, (2, 3))
        log = np.reshape(log, (2, 3))
        np.testing.assert_almost_equal(
            linear_to_c_log(linear),
            log,
            decimal=7)

        linear = np.reshape(linear, (2, 3, 1))
        log = np.reshape(log, (2, 3, 1))
        np.testing.assert_almost_equal(
            linear_to_c_log(linear),
            log,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_linear_to_c_log(self):
        """
        Tests :func:`colour.models.log.linear_to_c_log` definition nan
        support.
        """

        linear_to_c_log(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestCLogToLinear(unittest.TestCase):
    """
    Defines :func:`colour.models.log.c_log_to_linear` definition unit tests
    methods.
    """

    def test_c_log_to_linear(self):
        """
        Tests :func:`colour.models.log.c_log_to_linear` definition.
        """

        self.assertAlmostEqual(
            c_log_to_linear(0.073059700000000005),
            0.,
            places=7)

        self.assertAlmostEqual(
            c_log_to_linear(0.31201285555039493),
            0.18,
            places=7)

        self.assertAlmostEqual(
            c_log_to_linear(0.62740830453765284),
            1.,
            places=7)

    def test_n_dimensional_c_log_to_linear(self):
        """
        Tests :func:`colour.models.log.c_log_to_linear` definition
        n-dimensional arrays support.
        """

        log = 0.31201285555039493
        linear = 0.18
        np.testing.assert_almost_equal(
            c_log_to_linear(log),
            linear,
            decimal=7)

        log = np.tile(log, 6)
        linear = np.tile(linear, 6)
        np.testing.assert_almost_equal(
            c_log_to_linear(log),
            linear,
            decimal=7)

        log = np.reshape(log, (2, 3))
        linear = np.reshape(linear, (2, 3))
        np.testing.assert_almost_equal(
            c_log_to_linear(log),
            linear,
            decimal=7)

        log = np.reshape(log, (2, 3, 1))
        linear = np.reshape(linear, (2, 3, 1))
        np.testing.assert_almost_equal(
            c_log_to_linear(log),
            linear,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_c_log_to_linear(self):
        """
        Tests :func:`colour.models.log.c_log_to_linear` definition nan
        support.
        """

        c_log_to_linear(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLinearToAcesCc(unittest.TestCase):
    """
    Defines :func:`colour.models.log.linear_to_aces_cc` definition unit tests
    methods.
    """

    def test_linear_to_aces_cc(self):
        """
        Tests :func:`colour.models.log.linear_to_aces_cc` definition.
        """

        self.assertAlmostEqual(
            linear_to_aces_cc(0.00),
            -0.35844748858447484,
            places=7)

        self.assertAlmostEqual(
            linear_to_aces_cc(0.18),
            0.41358840249244228,
            places=7)

        self.assertAlmostEqual(
            linear_to_aces_cc(1.00),
            0.5547945205479452,
            places=7)

    def test_n_dimensional_linear_to_aces_cc(self):
        """
        Tests :func:`colour.models.log.linear_to_aces_cc` definition
        n-dimensional arrays support.
        """

        linear = 0.18
        log = 0.41358840249244228
        np.testing.assert_almost_equal(
            linear_to_aces_cc(linear),
            log,
            decimal=7)

        linear = np.tile(linear, 6)
        log = np.tile(log, 6)
        np.testing.assert_almost_equal(
            linear_to_aces_cc(linear),
            log,
            decimal=7)

        linear = np.reshape(linear, (2, 3))
        log = np.reshape(log, (2, 3))
        np.testing.assert_almost_equal(
            linear_to_aces_cc(linear),
            log,
            decimal=7)

        linear = np.reshape(linear, (2, 3, 1))
        log = np.reshape(log, (2, 3, 1))
        np.testing.assert_almost_equal(
            linear_to_aces_cc(linear),
            log,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_linear_to_aces_cc(self):
        """
        Tests :func:`colour.models.log.linear_to_aces_cc` definition nan
        support.
        """

        linear_to_aces_cc(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestAcesCcToLinear(unittest.TestCase):
    """
    Defines :func:`colour.models.log.aces_cc_to_linear` definition unit tests
    methods.
    """

    def test_aces_cc_to_linear(self):
        """
        Tests :func:`colour.models.log.aces_cc_to_linear` definition.
        """

        self.assertAlmostEqual(
            aces_cc_to_linear(0.00),
            0.0011857371917920374,
            places=7)

        self.assertAlmostEqual(
            aces_cc_to_linear(0.41358840249244228),
            0.18,
            places=7)

        self.assertAlmostEqual(
            aces_cc_to_linear(0.5547945205479452),
            1.,
            places=7)

    def test_n_dimensional_aces_cc_to_linear(self):
        """
        Tests :func:`colour.models.log.aces_cc_to_linear` definition
        n-dimensional arrays support.
        """

        log = 0.41358840249244228
        linear = 0.18
        np.testing.assert_almost_equal(
            aces_cc_to_linear(log),
            linear,
            decimal=7)

        log = np.tile(log, 6)
        linear = np.tile(linear, 6)
        np.testing.assert_almost_equal(
            aces_cc_to_linear(log),
            linear,
            decimal=7)

        log = np.reshape(log, (2, 3))
        linear = np.reshape(linear, (2, 3))
        np.testing.assert_almost_equal(
            aces_cc_to_linear(log),
            linear,
            decimal=7)

        log = np.reshape(log, (2, 3, 1))
        linear = np.reshape(linear, (2, 3, 1))
        np.testing.assert_almost_equal(
            aces_cc_to_linear(log),
            linear,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_aces_cc_to_linear(self):
        """
        Tests :func:`colour.models.log.aces_cc_to_linear` definition nan
        support.
        """

        aces_cc_to_linear(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLinearToAlexaLogC(unittest.TestCase):
    """
    Defines :func:`colour.models.log.linear_to_alexa_log_c` definition unit
    tests methods.
    """

    def test_linear_to_alexa_log_c(self):
        """
        Tests :func:`colour.models.log.linear_to_alexa_log_c` definition.
        """

        self.assertAlmostEqual(
            linear_to_alexa_log_c(0.00),
            0.092809,
            places=7)

        self.assertAlmostEqual(
            linear_to_alexa_log_c(0.18),
            0.39100683203408376,
            places=7)

        self.assertAlmostEqual(
            linear_to_alexa_log_c(1.00),
            0.57063155812041733,
            places=7)

    def test_n_dimensional_linear_to_alexa_log_c(self):
        """
        Tests :func:`colour.models.log.linear_to_alexa_log_c` definition
        n-dimensional arrays support.
        """

        linear = 0.18
        log = 0.39100683203408376
        np.testing.assert_almost_equal(
            linear_to_alexa_log_c(linear),
            log,
            decimal=7)

        linear = np.tile(linear, 6)
        log = np.tile(log, 6)
        np.testing.assert_almost_equal(
            linear_to_alexa_log_c(linear),
            log,
            decimal=7)

        linear = np.reshape(linear, (2, 3))
        log = np.reshape(log, (2, 3))
        np.testing.assert_almost_equal(
            linear_to_alexa_log_c(linear),
            log,
            decimal=7)

        linear = np.reshape(linear, (2, 3, 1))
        log = np.reshape(log, (2, 3, 1))
        np.testing.assert_almost_equal(
            linear_to_alexa_log_c(linear),
            log,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_linear_to_alexa_log_c(self):
        """
        Tests :func:`colour.models.log.linear_to_alexa_log_c` definition nan
        support.
        """

        linear_to_alexa_log_c(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestAlexaLogCToLinear(unittest.TestCase):
    """
    Defines :func:`colour.models.log.alexa_log_c_to_linear` definition unit
    tests methods.
    """

    def test_alexa_log_c_to_linear(self):
        """
        Tests :func:`colour.models.log.alexa_log_c_to_linear` definition.
        """

        self.assertAlmostEqual(
            alexa_log_c_to_linear(0.092809),
            0.,
            places=7)

        self.assertAlmostEqual(
            alexa_log_c_to_linear(0.39100683203408376),
            0.18,
            places=7)

        self.assertAlmostEqual(
            alexa_log_c_to_linear(0.57063155812041733),
            1.,
            places=7)

    def test_n_dimensional_alexa_log_c_to_linear(self):
        """
        Tests :func:`colour.models.log.alexa_log_c_to_linear` definition
        n-dimensional arrays support.
        """

        log = 0.39100683203408376
        linear = 0.18
        np.testing.assert_almost_equal(
            alexa_log_c_to_linear(log),
            linear,
            decimal=7)

        log = np.tile(log, 6)
        linear = np.tile(linear, 6)
        np.testing.assert_almost_equal(
            alexa_log_c_to_linear(log),
            linear,
            decimal=7)

        log = np.reshape(log, (2, 3))
        linear = np.reshape(linear, (2, 3))
        np.testing.assert_almost_equal(
            alexa_log_c_to_linear(log),
            linear,
            decimal=7)

        log = np.reshape(log, (2, 3, 1))
        linear = np.reshape(linear, (2, 3, 1))
        np.testing.assert_almost_equal(
            alexa_log_c_to_linear(log),
            linear,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_alexa_log_c_to_linear(self):
        """
        Tests :func:`colour.models.log.alexa_log_c_to_linear` definition nan
        support.
        """

        alexa_log_c_to_linear(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLinearToDciP3Log(unittest.TestCase):
    """
    Defines :func:`colour.models.log.linear_to_dci_p3_log` definition unit
    tests methods.
    """

    def test_linear_to_dci_p3_log(self):
        """
        Tests :func:`colour.models.log.linear_to_dci_p3_log` definition.
        """

        self.assertAlmostEqual(
            linear_to_dci_p3_log(0.00),
            0.0,
            places=7)

        self.assertAlmostEqual(
            linear_to_dci_p3_log(0.18),
            461.99220597484737,
            places=7)

        self.assertAlmostEqual(
            linear_to_dci_p3_log(1.00),
            893.4459834052784,
            places=7)

    def test_n_dimensional_linear_to_dci_p3_log(self):
        """
        Tests :func:`colour.models.log.linear_to_dci_p3_log` definition
        n-dimensional arrays support.
        """

        linear = 0.18
        log = 461.99220597484737
        np.testing.assert_almost_equal(
            linear_to_dci_p3_log(linear),
            log,
            decimal=7)

        linear = np.tile(linear, 6)
        log = np.tile(log, 6)
        np.testing.assert_almost_equal(
            linear_to_dci_p3_log(linear),
            log,
            decimal=7)

        linear = np.reshape(linear, (2, 3))
        log = np.reshape(log, (2, 3))
        np.testing.assert_almost_equal(
            linear_to_dci_p3_log(linear),
            log,
            decimal=7)

        linear = np.reshape(linear, (2, 3, 1))
        log = np.reshape(log, (2, 3, 1))
        np.testing.assert_almost_equal(
            linear_to_dci_p3_log(linear),
            log,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_linear_to_dci_p3_log(self):
        """
        Tests :func:`colour.models.log.linear_to_dci_p3_log` definition nan
        support.
        """

        linear_to_dci_p3_log(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestDciP3LogToLinear(unittest.TestCase):
    """
    Defines :func:`colour.models.log.dci_p3_log_to_linear` definition unit
    tests methods.
    """

    def test_dci_p3_log_to_linear(self):
        """
        Tests :func:`colour.models.log.dci_p3_log_to_linear` definition.
        """

        self.assertAlmostEqual(
            dci_p3_log_to_linear(0.0),
            0.,
            places=7)

        self.assertAlmostEqual(
            dci_p3_log_to_linear(461.99220597484737),
            0.18,
            places=7)

        self.assertAlmostEqual(
            dci_p3_log_to_linear(893.4459834052784),
            1.,
            places=7)

    def test_n_dimensional_dci_p3_log_to_linear(self):
        """
        Tests :func:`colour.models.log.dci_p3_log_to_linear` definition
        n-dimensional arrays support.
        """

        log = 461.99220597484737
        linear = 0.18
        np.testing.assert_almost_equal(
            dci_p3_log_to_linear(log),
            linear,
            decimal=7)

        log = np.tile(log, 6)
        linear = np.tile(linear, 6)
        np.testing.assert_almost_equal(
            dci_p3_log_to_linear(log),
            linear,
            decimal=7)

        log = np.reshape(log, (2, 3))
        linear = np.reshape(linear, (2, 3))
        np.testing.assert_almost_equal(
            dci_p3_log_to_linear(log),
            linear,
            decimal=7)

        log = np.reshape(log, (2, 3, 1))
        linear = np.reshape(linear, (2, 3, 1))
        np.testing.assert_almost_equal(
            dci_p3_log_to_linear(log),
            linear,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_dci_p3_log_to_linear(self):
        """
        Tests :func:`colour.models.log.dci_p3_log_to_linear` definition nan
        support.
        """

        dci_p3_log_to_linear(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLinearToRedLogFilm(unittest.TestCase):
    """
    Defines :func:`colour.models.log.linear_to_red_log_film` definition unit
    tests methods.
    """

    def test_linear_to_red_log_film(self):
        """
        Tests :func:`colour.models.log.linear_to_red_log_film` definition.
        """

        self.assertAlmostEqual(
            linear_to_red_log_film(0.00),
            0.,
            places=7)

        self.assertAlmostEqual(
            linear_to_red_log_film(0.18),
            0.63762184598817484,
            places=7)

        self.assertAlmostEqual(
            linear_to_red_log_film(1.00),
            1.,
            places=7)

    def test_n_dimensional_linear_to_red_log_film(self):
        """
        Tests :func:`colour.models.log.linear_to_red_log_film` definition
        n-dimensional arrays support.
        """

        linear = 0.18
        log = 0.63762184598817484
        np.testing.assert_almost_equal(
            linear_to_red_log_film(linear),
            log,
            decimal=7)

        linear = np.tile(linear, 6)
        log = np.tile(log, 6)
        np.testing.assert_almost_equal(
            linear_to_red_log_film(linear),
            log,
            decimal=7)

        linear = np.reshape(linear, (2, 3))
        log = np.reshape(log, (2, 3))
        np.testing.assert_almost_equal(
            linear_to_red_log_film(linear),
            log,
            decimal=7)

        linear = np.reshape(linear, (2, 3, 1))
        log = np.reshape(log, (2, 3, 1))
        np.testing.assert_almost_equal(
            linear_to_red_log_film(linear),
            log,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_linear_to_red_log_film(self):
        """
        Tests :func:`colour.models.log.linear_to_red_log_film` definition nan
        support.
        """

        linear_to_red_log_film(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestRedLogFilmToLinear(unittest.TestCase):
    """
    Defines :func:`colour.models.log.red_log_film_to_linear` definition unit
    tests methods.
    """

    def test_red_log_film_to_linear(self):
        """
        Tests :func:`colour.models.log.red_log_film_to_linear` definition.
        """

        self.assertAlmostEqual(
            red_log_film_to_linear(0.00),
            0.,
            places=7)

        self.assertAlmostEqual(
            red_log_film_to_linear(0.63762184598817484),
            0.18,
            places=7)

        self.assertAlmostEqual(
            red_log_film_to_linear(1.00),
            1.,
            places=7)

    def test_n_dimensional_red_log_film_to_linear(self):
        """
        Tests :func:`colour.models.log.red_log_film_to_linear` definition
        n-dimensional arrays support.
        """

        log = 0.63762184598817484
        linear = 0.18
        np.testing.assert_almost_equal(
            red_log_film_to_linear(log),
            linear,
            decimal=7)

        log = np.tile(log, 6)
        linear = np.tile(linear, 6)
        np.testing.assert_almost_equal(
            red_log_film_to_linear(log),
            linear,
            decimal=7)

        log = np.reshape(log, (2, 3))
        linear = np.reshape(linear, (2, 3))
        np.testing.assert_almost_equal(
            red_log_film_to_linear(log),
            linear,
            decimal=7)

        log = np.reshape(log, (2, 3, 1))
        linear = np.reshape(linear, (2, 3, 1))
        np.testing.assert_almost_equal(
            red_log_film_to_linear(log),
            linear,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_red_log_film_to_linear(self):
        """
        Tests :func:`colour.models.log.red_log_film_to_linear` definition nan
        support.
        """

        red_log_film_to_linear(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLinearToSLog(unittest.TestCase):
    """
    Defines :func:`colour.models.log.linear_to_s_log` definition unit tests
    methods.
    """

    def test_linear_to_s_log(self):
        """
        Tests :func:`colour.models.log.linear_to_s_log` definition.
        """

        self.assertAlmostEqual(
            linear_to_s_log(0.00),
            0.030001222851889303,
            places=7)

        self.assertAlmostEqual(
            linear_to_s_log(0.18),
            0.35998784642215442,
            places=7)

        self.assertAlmostEqual(
            linear_to_s_log(1.00),
            0.65352925122530825,
            places=7)

    def test_n_dimensional_linear_to_s_log(self):
        """
        Tests :func:`colour.models.log.linear_to_s_log` definition
        n-dimensional arrays support.
        """

        linear = 0.18
        log = 0.35998784642215442
        np.testing.assert_almost_equal(
            linear_to_s_log(linear),
            log,
            decimal=7)

        linear = np.tile(linear, 6)
        log = np.tile(log, 6)
        np.testing.assert_almost_equal(
            linear_to_s_log(linear),
            log,
            decimal=7)

        linear = np.reshape(linear, (2, 3))
        log = np.reshape(log, (2, 3))
        np.testing.assert_almost_equal(
            linear_to_s_log(linear),
            log,
            decimal=7)

        linear = np.reshape(linear, (2, 3, 1))
        log = np.reshape(log, (2, 3, 1))
        np.testing.assert_almost_equal(
            linear_to_s_log(linear),
            log,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_linear_to_s_log(self):
        """
        Tests :func:`colour.models.log.linear_to_s_log` definition nan
        support.
        """

        linear_to_s_log(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestSLogToLinear(unittest.TestCase):
    """
    Defines :func:`colour.models.log.s_log_to_linear` definition unit tests
    methods.
    """

    def test_s_log_to_linear(self):
        """
        Tests :func:`colour.models.log.s_log_to_linear` definition.
        """

        self.assertAlmostEqual(
            s_log_to_linear(0.030001222851889303),
            0.,
            places=7)

        self.assertAlmostEqual(
            s_log_to_linear(0.35998784642215442),
            0.18,
            places=7)

        self.assertAlmostEqual(
            s_log_to_linear(0.65352925122530825),
            1.,
            places=7)

    def test_n_dimensional_s_log_to_linear(self):
        """
        Tests :func:`colour.models.log.s_log_to_linear` definition
        n-dimensional arrays support.
        """

        log = 0.35998784642215442
        linear = 0.18
        np.testing.assert_almost_equal(
            s_log_to_linear(log),
            linear,
            decimal=7)

        log = np.tile(log, 6)
        linear = np.tile(linear, 6)
        np.testing.assert_almost_equal(
            s_log_to_linear(log),
            linear,
            decimal=7)

        log = np.reshape(log, (2, 3))
        linear = np.reshape(linear, (2, 3))
        np.testing.assert_almost_equal(
            s_log_to_linear(log),
            linear,
            decimal=7)

        log = np.reshape(log, (2, 3, 1))
        linear = np.reshape(linear, (2, 3, 1))
        np.testing.assert_almost_equal(
            s_log_to_linear(log),
            linear,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_s_log_to_linear(self):
        """
        Tests :func:`colour.models.log.s_log_to_linear` definition nan
        support.
        """

        s_log_to_linear(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLinearToSLog2(unittest.TestCase):
    """
    Defines :func:`colour.models.log.linear_to_s_log2` definition unit tests
    methods.
    """

    def test_linear_to_s_log2(self):
        """
        Tests :func:`colour.models.log.linear_to_s_log2` definition.
        """

        self.assertAlmostEqual(
            linear_to_s_log2(0.00),
            0.088251291513445795,
            places=7)

        self.assertAlmostEqual(
            linear_to_s_log2(0.18),
            0.38497081592867027,
            places=7)

        self.assertAlmostEqual(
            linear_to_s_log2(1.00),
            0.63855168462253165,
            places=7)

    def test_n_dimensional_linear_to_s_log2(self):
        """
        Tests :func:`colour.models.log.linear_to_s_log2` definition
        n-dimensional arrays support.
        """

        linear = 0.18
        log = 0.38497081592867027
        np.testing.assert_almost_equal(
            linear_to_s_log2(linear),
            log,
            decimal=7)

        linear = np.tile(linear, 6)
        log = np.tile(log, 6)
        np.testing.assert_almost_equal(
            linear_to_s_log2(linear),
            log,
            decimal=7)

        linear = np.reshape(linear, (2, 3))
        log = np.reshape(log, (2, 3))
        np.testing.assert_almost_equal(
            linear_to_s_log2(linear),
            log,
            decimal=7)

        linear = np.reshape(linear, (2, 3, 1))
        log = np.reshape(log, (2, 3, 1))
        np.testing.assert_almost_equal(
            linear_to_s_log2(linear),
            log,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_linear_to_s_log2(self):
        """
        Tests :func:`colour.models.log.linear_to_s_log2` definition nan
        support.
        """

        linear_to_s_log2(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestSLog2ToLinear(unittest.TestCase):
    """
    Defines :func:`colour.models.log.s_log2_to_linear` definition unit tests
    methods.
    """

    def test_s_log2_to_linear(self):
        """
        Tests :func:`colour.models.log.s_log2_to_linear` definition.
        """

        self.assertAlmostEqual(
            s_log2_to_linear(0.088251291513445795),
            0.,
            places=7)

        self.assertAlmostEqual(
            s_log2_to_linear(0.38497081592867027),
            0.18,
            places=7)

        self.assertAlmostEqual(
            s_log2_to_linear(0.63855168462253165),
            1.,
            places=7)

    def test_n_dimensional_s_log2_to_linear(self):
        """
        Tests :func:`colour.models.log.s_log2_to_linear` definition
        n-dimensional arrays support.
        """

        log = 0.38497081592867027
        linear = 0.18
        np.testing.assert_almost_equal(
            s_log2_to_linear(log),
            linear,
            decimal=7)

        log = np.tile(log, 6)
        linear = np.tile(linear, 6)
        np.testing.assert_almost_equal(
            s_log2_to_linear(log),
            linear,
            decimal=7)

        log = np.reshape(log, (2, 3))
        linear = np.reshape(linear, (2, 3))
        np.testing.assert_almost_equal(
            s_log2_to_linear(log),
            linear,
            decimal=7)

        log = np.reshape(log, (2, 3, 1))
        linear = np.reshape(linear, (2, 3, 1))
        np.testing.assert_almost_equal(
            s_log2_to_linear(log),
            linear,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_s_log2_to_linear(self):
        """
        Tests :func:`colour.models.log.s_log2_to_linear` definition nan
        support.
        """

        s_log2_to_linear(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLinearToSLog3(unittest.TestCase):
    """
    Defines :func:`colour.models.log.linear_to_s_log3` definition unit tests
    methods.
    """

    def test_linear_to_s_log3(self):
        """
        Tests :func:`colour.models.log.linear_to_s_log3` definition.
        """

        self.assertAlmostEqual(
            linear_to_s_log3(0.00),
            0.09286412512218964,
            places=7)

        self.assertAlmostEqual(
            linear_to_s_log3(0.18),
            0.41055718475073316,
            places=7)

        self.assertAlmostEqual(
            linear_to_s_log3(1.00),
            0.59602734369012345,
            places=7)

    def test_n_dimensional_linear_to_s_log3(self):
        """
        Tests :func:`colour.models.log.linear_to_s_log3` definition
        n-dimensional arrays support.
        """

        linear = 0.18
        log = 0.41055718475073316
        np.testing.assert_almost_equal(
            linear_to_s_log3(linear),
            log,
            decimal=7)

        linear = np.tile(linear, 6)
        log = np.tile(log, 6)
        np.testing.assert_almost_equal(
            linear_to_s_log3(linear),
            log,
            decimal=7)

        linear = np.reshape(linear, (2, 3))
        log = np.reshape(log, (2, 3))
        np.testing.assert_almost_equal(
            linear_to_s_log3(linear),
            log,
            decimal=7)

        linear = np.reshape(linear, (2, 3, 1))
        log = np.reshape(log, (2, 3, 1))
        np.testing.assert_almost_equal(
            linear_to_s_log3(linear),
            log,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_linear_to_s_log3(self):
        """
        Tests :func:`colour.models.log.linear_to_s_log3` definition nan
        support.
        """

        linear_to_s_log3(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestSLog3ToLinear(unittest.TestCase):
    """
    Defines :func:`colour.models.log.s_log3_to_linear` definition unit tests
    methods.
    """

    def test_s_log3_to_linear(self):
        """
        Tests :func:`colour.models.log.s_log3_to_linear` definition.
        """

        self.assertAlmostEqual(
            s_log3_to_linear(0.09286412512218964),
            0.,
            places=7)

        self.assertAlmostEqual(
            s_log3_to_linear(0.41055718475073316),
            0.18,
            places=7)

        self.assertAlmostEqual(
            s_log3_to_linear(0.59602734369012345),
            1.,
            places=7)

    def test_n_dimensional_s_log3_to_linear(self):
        """
        Tests :func:`colour.models.log.s_log3_to_linear` definition
        n-dimensional arrays support.
        """

        log = 0.41055718475073316
        linear = 0.18
        np.testing.assert_almost_equal(
            s_log3_to_linear(log),
            linear,
            decimal=7)

        log = np.tile(log, 6)
        linear = np.tile(linear, 6)
        np.testing.assert_almost_equal(
            s_log3_to_linear(log),
            linear,
            decimal=7)

        log = np.reshape(log, (2, 3))
        linear = np.reshape(linear, (2, 3))
        np.testing.assert_almost_equal(
            s_log3_to_linear(log),
            linear,
            decimal=7)

        log = np.reshape(log, (2, 3, 1))
        linear = np.reshape(linear, (2, 3, 1))
        np.testing.assert_almost_equal(
            s_log3_to_linear(log),
            linear,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_s_log3_to_linear(self):
        """
        Tests :func:`colour.models.log.s_log3_to_linear` definition nan
        support.
        """

        s_log3_to_linear(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLinearToVLog(unittest.TestCase):
    """
    Defines :func:`colour.models.log.linear_to_v_log` definition unit tests
    methods.
    """

    def test_linear_to_v_log(self):
        """
        Tests :func:`colour.models.log.linear_to_v_log` definition.
        """

        self.assertAlmostEqual(
            linear_to_v_log(0.00),
            0.125,
            places=7)

        self.assertAlmostEqual(
            linear_to_v_log(0.18),
            0.42331144876013616,
            places=7)

        self.assertAlmostEqual(
            linear_to_v_log(1.00),
            0.5991177001581459,
            places=7)

    def test_n_dimensional_linear_to_v_log(self):
        """
        Tests :func:`colour.models.log.linear_to_v_log` definition
        n-dimensional arrays support.
        """

        linear = 0.18
        log = 0.42331144876013616
        np.testing.assert_almost_equal(
            linear_to_v_log(linear),
            log,
            decimal=7)

        linear = np.tile(linear, 6)
        log = np.tile(log, 6)
        np.testing.assert_almost_equal(
            linear_to_v_log(linear),
            log,
            decimal=7)

        linear = np.reshape(linear, (2, 3))
        log = np.reshape(log, (2, 3))
        np.testing.assert_almost_equal(
            linear_to_v_log(linear),
            log,
            decimal=7)

        linear = np.reshape(linear, (2, 3, 1))
        log = np.reshape(log, (2, 3, 1))
        np.testing.assert_almost_equal(
            linear_to_v_log(linear),
            log,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_linear_to_v_log(self):
        """
        Tests :func:`colour.models.log.linear_to_v_log` definition nan
        support.
        """

        linear_to_v_log(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestVLogToLinear(unittest.TestCase):
    """
    Defines :func:`colour.models.log.v_log_to_linear` definition unit tests
    methods.
    """

    def test_v_log_to_linear(self):
        """
        Tests :func:`colour.models.log.v_log_to_linear` definition.
        """

        self.assertAlmostEqual(
            v_log_to_linear(0.125),
            0.,
            places=7)

        self.assertAlmostEqual(
            v_log_to_linear(0.42331144876013616),
            0.18,
            places=7)

        self.assertAlmostEqual(
            v_log_to_linear(0.5991177001581459),
            1.,
            places=7)

    def test_n_dimensional_v_log_to_linear(self):
        """
        Tests :func:`colour.models.log.v_log_to_linear` definition
        n-dimensional arrays support.
        """

        log = 0.42331144876013616
        linear = 0.18
        np.testing.assert_almost_equal(
            v_log_to_linear(log),
            linear,
            decimal=7)

        log = np.tile(log, 6)
        linear = np.tile(linear, 6)
        np.testing.assert_almost_equal(
            v_log_to_linear(log),
            linear,
            decimal=7)

        log = np.reshape(log, (2, 3))
        linear = np.reshape(linear, (2, 3))
        np.testing.assert_almost_equal(
            v_log_to_linear(log),
            linear,
            decimal=7)

        log = np.reshape(log, (2, 3, 1))
        linear = np.reshape(linear, (2, 3, 1))
        np.testing.assert_almost_equal(
            v_log_to_linear(log),
            linear,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_v_log_to_linear(self):
        """
        Tests :func:`colour.models.log.v_log_to_linear` definition nan
        support.
        """

        v_log_to_linear(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


if __name__ == '__main__':
    unittest.main()
