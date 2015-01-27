#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.models.log` module.
"""

from __future__ import division, unicode_literals

import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

from colour.models import (
    linear_to_cineon,
    cineon_to_linear,
    linear_to_panalog,
    panalog_to_linear,
    linear_to_red_log,
    red_log_to_linear,
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
    linear_to_s_log,
    s_log_to_linear,
    linear_to_s_log2,
    s_log2_to_linear,
    linear_to_s_log3,
    s_log3_to_linear)

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
           'TestLinearToRedLog',
           'TestRedLogToLinear',
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
           'TestSLog3ToLinear']


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
            linear_to_cineon(0),
            0.092864125122189639,
            places=7)

        self.assertAlmostEqual(
            linear_to_cineon(0.18),
            0.45731961308541841,
            places=7)

        self.assertAlmostEqual(
            linear_to_cineon(1),
            0.66959921798631472,
            places=7)


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
            0,
            places=7)

        self.assertAlmostEqual(
            cineon_to_linear(0.45731961308541841),
            0.18,
            places=7)

        self.assertAlmostEqual(
            cineon_to_linear(0.66959921798631472),
            1,
            places=7)


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
            linear_to_panalog(0),
            0.062561094819159335,
            places=7)

        self.assertAlmostEqual(
            linear_to_panalog(0.18),
            0.37457679138229816,
            places=7)

        self.assertAlmostEqual(
            linear_to_panalog(1),
            0.66568914956011727,
            places=7)


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
            0,
            places=7)

        self.assertAlmostEqual(
            panalog_to_linear(0.37457679138229816),
            0.18,
            places=7)

        self.assertAlmostEqual(
            panalog_to_linear(0.66568914956011727),
            1,
            places=7)


class TestLinearToRedLog(unittest.TestCase):
    """
    Defines :func:`colour.models.log.linear_to_red_log` definition unit tests
    methods.
    """

    def test_linear_to_red_log(self):
        """
        Tests :func:`colour.models.log.linear_to_red_log` definition.
        """

        self.assertAlmostEqual(
            linear_to_red_log(0),
            0,
            places=7)

        self.assertAlmostEqual(
            linear_to_red_log(0.18),
            0.63762184598817484,
            places=7)

        self.assertAlmostEqual(
            linear_to_red_log(1),
            1,
            places=7)


class TestRedLogToLinear(unittest.TestCase):
    """
    Defines :func:`colour.models.log.red_log_to_linear` definition unit tests
    methods.
    """

    def test_red_log_to_linear(self):
        """
        Tests :func:`colour.models.log.red_log_to_linear` definition.
        """

        self.assertAlmostEqual(
            red_log_to_linear(0),
            0,
            places=7)

        self.assertAlmostEqual(
            red_log_to_linear(0.63762184598817484),
            0.18,
            places=7)

        self.assertAlmostEqual(
            red_log_to_linear(1),
            1,
            places=7)


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
            linear_to_viper_log(1),
            1,
            places=7)


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
            viper_log_to_linear(1),
            1,
            places=7)


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
            linear_to_pivoted_log(1),
            0.6533902722082191,
            places=7)


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
            1,
            places=7)


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
            linear_to_c_log(0),
            0.073059700000000005,
            places=7)

        self.assertAlmostEqual(
            linear_to_c_log(0.18),
            0.31201285555039493,
            places=7)

        self.assertAlmostEqual(
            linear_to_c_log(1),
            0.62740830453765284,
            places=7)


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
            0,
            places=7)

        self.assertAlmostEqual(
            c_log_to_linear(0.31201285555039493),
            0.18,
            places=7)

        self.assertAlmostEqual(
            c_log_to_linear(0.62740830453765284),
            1,
            places=7)


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
            linear_to_aces_cc(0),
            -0.35844748858447484,
            places=7)

        self.assertAlmostEqual(
            linear_to_aces_cc(0.18),
            0.41358840249244228,
            places=7)

        self.assertAlmostEqual(
            linear_to_aces_cc(1),
            0.5547945205479452,
            places=7)


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
            aces_cc_to_linear(0),
            0.0011857371917920374,
            places=7)

        self.assertAlmostEqual(
            aces_cc_to_linear(0.41358840249244228),
            0.18,
            places=7)

        self.assertAlmostEqual(
            aces_cc_to_linear(0.5547945205479452),
            1,
            places=7)


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
            linear_to_alexa_log_c(0),
            0.092809,
            places=7)

        self.assertAlmostEqual(
            linear_to_alexa_log_c(0.18),
            0.39100683203408376,
            places=7)

        self.assertAlmostEqual(
            linear_to_alexa_log_c(1),
            0.57063155812041733,
            places=7)


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
            0,
            places=7)

        self.assertAlmostEqual(
            alexa_log_c_to_linear(0.39100683203408376),
            0.18,
            places=7)

        self.assertAlmostEqual(
            alexa_log_c_to_linear(0.57063155812041733),
            1,
            places=7)


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
            linear_to_dci_p3_log(0),
            0.0,
            places=7)

        self.assertAlmostEqual(
            linear_to_dci_p3_log(0.18),
            461.99220597484737,
            places=7)

        self.assertAlmostEqual(
            linear_to_dci_p3_log(1),
            893.4459834052784,
            places=7)


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
            0,
            places=7)

        self.assertAlmostEqual(
            dci_p3_log_to_linear(461.99220597484737),
            0.18,
            places=7)

        self.assertAlmostEqual(
            dci_p3_log_to_linear(893.4459834052784),
            1,
            places=7)


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
            linear_to_s_log(0),
            0.030001222851889303,
            places=7)

        self.assertAlmostEqual(
            linear_to_s_log(0.18),
            0.35998784642215442,
            places=7)

        self.assertAlmostEqual(
            linear_to_s_log(1),
            0.65352925122530825,
            places=7)


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
            0,
            places=7)

        self.assertAlmostEqual(
            s_log_to_linear(0.35998784642215442),
            0.18,
            places=7)

        self.assertAlmostEqual(
            s_log_to_linear(0.65352925122530825),
            1,
            places=7)


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
            linear_to_s_log2(0),
            0.088251291513445795,
            places=7)

        self.assertAlmostEqual(
            linear_to_s_log2(0.18),
            0.38497081592867027,
            places=7)

        self.assertAlmostEqual(
            linear_to_s_log2(1),
            0.63855168462253165,
            places=7)


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
            0,
            places=7)

        self.assertAlmostEqual(
            s_log2_to_linear(0.38497081592867027),
            0.18,
            places=7)

        self.assertAlmostEqual(
            s_log2_to_linear(0.63855168462253165),
            1,
            places=7)


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
            linear_to_s_log3(0),
            0.09286412512218964,
            places=7)

        self.assertAlmostEqual(
            linear_to_s_log3(0.18),
            0.41055718475073316,
            places=7)

        self.assertAlmostEqual(
            linear_to_s_log3(1),
            0.59602734369012345,
            places=7)


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
            0,
            places=7)

        self.assertAlmostEqual(
            s_log3_to_linear(0.41055718475073316),
            0.18,
            places=7)

        self.assertAlmostEqual(
            s_log3_to_linear(0.59602734369012345),
            1,
            places=7)
