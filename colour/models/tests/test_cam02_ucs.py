# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.models.cam02_ucs` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from itertools import permutations

from colour.appearance import (CIECAM02_VIEWING_CONDITIONS, XYZ_to_CIECAM02)
from colour.models.cam02_ucs import (COEFFICIENTS_UCS_LUO2006,
                                     JMh_CIECAM02_to_UCS_Luo2006,
                                     UCS_Luo2006_to_JMh_CIECAM02)
from colour.models import (JMh_CIECAM02_to_CAM02LCD, CAM02LCD_to_JMh_CIECAM02,
                           JMh_CIECAM02_to_CAM02SCD, CAM02SCD_to_JMh_CIECAM02,
                           JMh_CIECAM02_to_CAM02UCS, CAM02UCS_to_JMh_CIECAM02)
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'TestJMh_CIECAM02_to_UCS_Luo2006', 'TestUCS_Luo2006_to_JMh_CIECAM02'
]


class TestJMh_CIECAM02_to_UCS_Luo2006(unittest.TestCase):
    """
    Defines :func:`colour.models.cam02_ucs.TestJMh_CIECAM02_to_UCS_Luo2006`
    definition unit tests methods.
    """

    def setUp(self):
        """
        Initialises common tests attributes.
        """

        XYZ = np.array([19.01, 20.00, 21.78])
        XYZ_w = np.array([95.05, 100.00, 108.88])
        L_A = 318.31
        Y_b = 20.0
        surround = CIECAM02_VIEWING_CONDITIONS['Average']
        specification = XYZ_to_CIECAM02(XYZ, XYZ_w, L_A, Y_b, surround)

        self._JMh = np.array(
            [specification.J, specification.M, specification.h])

    def test_JMh_CIECAM02_to_UCS_Luo2006(self):
        """
        Tests :func:`colour.models.cam02_ucs.JMh_CIECAM02_to_UCS_Luo2006`
        definition.
        """

        np.testing.assert_almost_equal(
            JMh_CIECAM02_to_UCS_Luo2006(self._JMh,
                                        COEFFICIENTS_UCS_LUO2006['CAM02-LCD']),
            np.array([54.90433134, -0.08450395, -0.06854831]),
            decimal=7)

        np.testing.assert_almost_equal(
            JMh_CIECAM02_to_UCS_Luo2006(self._JMh,
                                        COEFFICIENTS_UCS_LUO2006['CAM02-LCD']),
            JMh_CIECAM02_to_CAM02LCD(self._JMh),
            decimal=7)

        np.testing.assert_almost_equal(
            JMh_CIECAM02_to_UCS_Luo2006(self._JMh,
                                        COEFFICIENTS_UCS_LUO2006['CAM02-SCD']),
            np.array([54.90433134, -0.08436178, -0.06843298]),
            decimal=7)

        np.testing.assert_almost_equal(
            JMh_CIECAM02_to_UCS_Luo2006(self._JMh,
                                        COEFFICIENTS_UCS_LUO2006['CAM02-SCD']),
            JMh_CIECAM02_to_CAM02SCD(self._JMh),
            decimal=7)

        np.testing.assert_almost_equal(
            JMh_CIECAM02_to_UCS_Luo2006(self._JMh,
                                        COEFFICIENTS_UCS_LUO2006['CAM02-UCS']),
            np.array([54.90433134, -0.08442362, -0.06848314]),
            decimal=7)

        np.testing.assert_almost_equal(
            JMh_CIECAM02_to_UCS_Luo2006(self._JMh,
                                        COEFFICIENTS_UCS_LUO2006['CAM02-UCS']),
            JMh_CIECAM02_to_CAM02UCS(self._JMh),
            decimal=7)

    def test_n_dimensional_JMh_CIECAM02_to_UCS_Luo2006(self):
        """
        Tests :func:`colour.models.cam02_ucs.JMh_CIECAM02_to_UCS_Luo2006`
        definition n-dimensions support.
        """

        JMh = self._JMh
        Jpapbp = np.array([54.90433134, -0.08450395, -0.06854831])
        np.testing.assert_almost_equal(
            JMh_CIECAM02_to_UCS_Luo2006(JMh,
                                        COEFFICIENTS_UCS_LUO2006['CAM02-LCD']),
            Jpapbp,
            decimal=7)

        JMh = np.tile(JMh, (6, 1))
        Jpapbp = np.tile(Jpapbp, (6, 1))
        np.testing.assert_almost_equal(
            JMh_CIECAM02_to_UCS_Luo2006(JMh,
                                        COEFFICIENTS_UCS_LUO2006['CAM02-LCD']),
            Jpapbp,
            decimal=7)

        JMh = np.reshape(JMh, (2, 3, 3))
        Jpapbp = np.reshape(Jpapbp, (2, 3, 3))
        np.testing.assert_almost_equal(
            JMh_CIECAM02_to_UCS_Luo2006(JMh,
                                        COEFFICIENTS_UCS_LUO2006['CAM02-LCD']),
            Jpapbp,
            decimal=7)

    def test_domain_range_scale_JMh_CIECAM02_to_UCS_Luo2006(self):
        """
        Tests :func:`colour.models.cam02_ucs.JMh_CIECAM02_to_UCS_Luo2006`
        definition domain and range scale support.
        """

        JMh = self._JMh
        Jpapbp = JMh_CIECAM02_to_UCS_Luo2006(
            JMh, COEFFICIENTS_UCS_LUO2006['CAM02-LCD'])

        d_r = (('reference', 1, 1), (1, np.array([0.01, 0.01, 1 / 360]), 0.01),
               (100, np.array([1, 1, 1 / 3.6]), 1))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    JMh_CIECAM02_to_UCS_Luo2006(
                        JMh * factor_a, COEFFICIENTS_UCS_LUO2006['CAM02-LCD']),
                    Jpapbp * factor_b,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_JMh_CIECAM02_to_UCS_Luo2006(self):
        """
        Tests :func:`colour.models.cam02_ucs.JMh_CIECAM02_to_UCS_Luo2006`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            JMh = np.array(case)
            JMh_CIECAM02_to_UCS_Luo2006(JMh,
                                        COEFFICIENTS_UCS_LUO2006['CAM02-LCD'])


class TestUCS_Luo2006_to_JMh_CIECAM02(unittest.TestCase):
    """
    Defines :func:`colour.models.cam02_ucs.TestUCS_Luo2006_to_JMh_CIECAM02`
    definition unit tests methods.
    """

    def test_UCS_Luo2006_to_JMh_CIECAM02(self):
        """
        Tests :func:`colour.models.cam02_ucs.UCS_Luo2006_to_JMh_CIECAM02`
        definition.
        """

        np.testing.assert_almost_equal(
            UCS_Luo2006_to_JMh_CIECAM02(
                np.array([54.90433134, -0.08442362, -0.06848314]),
                COEFFICIENTS_UCS_LUO2006['CAM02-LCD']),
            np.array([41.73109113, 0.10873867, 219.04843202]),
            decimal=7)

        np.testing.assert_almost_equal(
            UCS_Luo2006_to_JMh_CIECAM02(
                np.array([54.90433134, -0.08442362, -0.06848314]),
                COEFFICIENTS_UCS_LUO2006['CAM02-LCD']),
            CAM02LCD_to_JMh_CIECAM02(
                np.array([54.90433134, -0.08442362, -0.06848314])),
            decimal=7)

        np.testing.assert_almost_equal(
            UCS_Luo2006_to_JMh_CIECAM02(
                np.array([54.90433134, -0.08442362, -0.06848314]),
                COEFFICIENTS_UCS_LUO2006['CAM02-SCD']),
            np.array([41.73109113, 0.10892212, 219.04843202]),
            decimal=7)

        np.testing.assert_almost_equal(
            UCS_Luo2006_to_JMh_CIECAM02(
                np.array([54.90433134, -0.08442362, -0.06848314]),
                COEFFICIENTS_UCS_LUO2006['CAM02-SCD']),
            CAM02SCD_to_JMh_CIECAM02(
                np.array([54.90433134, -0.08442362, -0.06848314])),
            decimal=7)

        np.testing.assert_almost_equal(
            UCS_Luo2006_to_JMh_CIECAM02(
                np.array([54.90433134, -0.08442362, -0.06848314]),
                COEFFICIENTS_UCS_LUO2006['CAM02-UCS']),
            np.array([41.73109113, 0.10884218, 219.04843202]),
            decimal=7)

        np.testing.assert_almost_equal(
            UCS_Luo2006_to_JMh_CIECAM02(
                np.array([54.90433134, -0.08442362, -0.06848314]),
                COEFFICIENTS_UCS_LUO2006['CAM02-UCS']),
            CAM02UCS_to_JMh_CIECAM02(
                np.array([54.90433134, -0.08442362, -0.06848314])),
            decimal=7)

    def test_n_dimensional_UCS_Luo2006_to_JMh_CIECAM02(self):
        """
        Tests :func:`colour.models.cam02_ucs.UCS_Luo2006_to_JMh_CIECAM02`
        definition n-dimensions support.
        """

        Jpapbp = np.array([54.90433134, -0.08442362, -0.06848314])
        JMh = np.array([41.73109113, 0.10873867, 219.04843202])
        np.testing.assert_almost_equal(
            UCS_Luo2006_to_JMh_CIECAM02(Jpapbp,
                                        COEFFICIENTS_UCS_LUO2006['CAM02-LCD']),
            JMh,
            decimal=7)

        Jpapbp = np.tile(Jpapbp, (6, 1))
        JMh = np.tile(JMh, (6, 1))
        np.testing.assert_almost_equal(
            UCS_Luo2006_to_JMh_CIECAM02(Jpapbp,
                                        COEFFICIENTS_UCS_LUO2006['CAM02-LCD']),
            JMh,
            decimal=7)

        Jpapbp = np.reshape(Jpapbp, (2, 3, 3))
        JMh = np.reshape(JMh, (2, 3, 3))
        np.testing.assert_almost_equal(
            UCS_Luo2006_to_JMh_CIECAM02(Jpapbp,
                                        COEFFICIENTS_UCS_LUO2006['CAM02-LCD']),
            JMh,
            decimal=7)

    def test_domain_range_scale_UCS_Luo2006_to_JMh_CIECAM02(self):
        """
        Tests :func:`colour.models.cam02_ucs.UCS_Luo2006_to_JMh_CIECAM02`
        definition domain and range scale support.
        """

        Jpapbp = np.array([54.90433134, -0.08442362, -0.06848314])
        JMh = UCS_Luo2006_to_JMh_CIECAM02(
            Jpapbp, COEFFICIENTS_UCS_LUO2006['CAM02-LCD'])

        d_r = (('reference', 1, 1), (1, 0.01, np.array([0.01, 0.01, 1 / 360])),
               (100, 1, np.array([1, 1, 1 / 3.6])))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    UCS_Luo2006_to_JMh_CIECAM02(
                        Jpapbp * factor_a,
                        COEFFICIENTS_UCS_LUO2006['CAM02-LCD']),
                    JMh * factor_b,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_UCS_Luo2006_to_JMh_CIECAM02(self):
        """
        Tests :func:`colour.models.cam02_ucs.UCS_Luo2006_to_JMh_CIECAM02`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            Jpapbp = np.array(case)
            UCS_Luo2006_to_JMh_CIECAM02(Jpapbp,
                                        COEFFICIENTS_UCS_LUO2006['CAM02-LCD'])


if __name__ == '__main__':
    unittest.main()
