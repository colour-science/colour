# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines the unit tests for the :mod:`colour.appearance.hke` module.
"""

import numpy as np
import unittest

from colour.appearance.hke import (
    HelmholtzKohlrausch_effect_object_Nayatani1997,
    HelmholtzKohlrausch_effect_luminous_Nayatani1997,
    coefficient_K_Br_Nayatani1997, coefficient_q_Nayatani1997)

__author__ = 'Ilia Sibiryakov'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestHelmholtzKohlrauschEffectObjectNayatani1997',
    'TestHelmholtzKohlrauschEffectLuminousNayatani1997',
    'TestCoefficient_K_Br_Nayatani1997', 'TestCoefficient_q_Nayatani1997'
]


class TestHelmholtzKohlrauschEffectObjectNayatani1997(unittest.TestCase):
    """
    Defines :func:`colour.HelmholtzKohlrausch_effect_object_Nayatani1997`
    definition unit tests methods.
    """

    def test_HelmholtzKohlrausch_effect_object_Nayatani1997(self):
        """
        Tests :func:`colour.HelmholtzKohlrausch_effect_object_Nayatani1997`
        definition.
        """

        self.assertAlmostEqual(
            HelmholtzKohlrausch_effect_object_Nayatani1997(
                [0.40351010, 0.53933673], [0.19783001, 0.46831999],
                63.66,
                method='VCC'),
            1.344152435497761,
            places=7)

        self.assertAlmostEqual(
            HelmholtzKohlrausch_effect_object_Nayatani1997(
                [0.40351010, 0.53933673], [0.19783001, 0.46831999],
                63.66,
                method='VAC'),
            1.261777232837009,
            places=7)

    def test_n_dimensional_HelmholtzKohlrausch_effect_object_Nayatani1997(
            self):
        """
        Tests :func:`colour.HelmholtzKohlrausch_effect_object_Nayatani1997`
        definition n_dimensional arrays support.
        """

        uv_d65 = [0.19783001, 0.46831999]
        uv = [0.40351010, 0.53933673]
        L_a = 63.66

        result_vcc = HelmholtzKohlrausch_effect_object_Nayatani1997(
            uv, uv_d65, L_a, method='VCC')
        result_vac = HelmholtzKohlrausch_effect_object_Nayatani1997(
            uv, uv_d65, L_a, method='VAC')

        uv_d65 = np.tile(uv_d65, (6, 1))
        uv = np.tile(uv, (6, 1))
        result_vcc = np.tile(result_vcc, 6)
        result_vac = np.tile(result_vac, 6)

        np.testing.assert_almost_equal(
            HelmholtzKohlrausch_effect_object_Nayatani1997(
                uv, uv_d65, L_a, method='VCC'),
            result_vcc,
            decimal=7)

        np.testing.assert_almost_equal(
            HelmholtzKohlrausch_effect_object_Nayatani1997(
                uv, uv_d65, L_a, method='VAC'),
            result_vac,
            decimal=7)


class TestHelmholtzKohlrauschEffectLuminousNayatani1997(unittest.TestCase):
    """
    Defines :func:`colour.HelmholtzKohlrausch_effect_luminous_Nayatani1997`
    definition unit tests methods.
    """

    def test_HelmholtzKohlrausch_effect_luminous_Nayatani1997(self):
        """
        Tests :func:`colour.HelmholtzKohlrausch_effect_luminous_Nayatani1997`
        definition.
        """

        self.assertAlmostEqual(
            HelmholtzKohlrausch_effect_luminous_Nayatani1997(
                [0.40351010, 0.53933673], [0.19783001, 0.46831999],
                63.66,
                method='VCC'),
            2.014433723774654,
            places=7)

        self.assertAlmostEqual(
            HelmholtzKohlrausch_effect_luminous_Nayatani1997(
                [0.40351010, 0.53933673], [0.19783001, 0.46831999],
                63.66,
                method='VAC'),
            1.727991241148628,
            places=7)

    def test_n_dimensional_HelmholtzKohlrausch_effect_luminous_Nayatani1997(
            self):
        """
        Tests :func:`colour.HelmholtzKohlrausch_effect_luminous_Nayatani1997`
        definition n_dimensional arrays support.
        """

        uv_d65 = [0.19783001, 0.46831999]
        uv = [0.40351010, 0.53933673]
        L_a = 63.66

        result_vcc = HelmholtzKohlrausch_effect_luminous_Nayatani1997(
            uv, uv_d65, L_a, method='VCC')
        result_vac = HelmholtzKohlrausch_effect_luminous_Nayatani1997(
            uv, uv_d65, L_a, method='VAC')

        uv_d65 = np.tile(uv_d65, (6, 1))
        uv = np.tile(uv, (6, 1))
        result_vcc = np.tile(result_vcc, 6)
        result_vac = np.tile(result_vac, 6)

        np.testing.assert_almost_equal(
            HelmholtzKohlrausch_effect_luminous_Nayatani1997(
                uv, uv_d65, L_a, method='VCC'),
            result_vcc,
            decimal=7)

        np.testing.assert_almost_equal(
            HelmholtzKohlrausch_effect_luminous_Nayatani1997(
                uv, uv_d65, L_a, method='VAC'),
            result_vac,
            decimal=7)


class TestCoefficient_K_Br_Nayatani1997(unittest.TestCase):
    """
    Defines :func:`colour.appearance.coefficient_K_Br_Nayatani1997`
    definition unit tests methods.
    """

    def test_coefficient_K_Br_Nayatani1997(self):
        """
        Tests :func:`colour.appearance.coefficient_K_Br_Nayatani1997`
        definition.
        """

        self.assertAlmostEqual(
            coefficient_K_Br_Nayatani1997(10.00000000),
            0.71344817765758839,
            places=7)

        self.assertAlmostEqual(
            coefficient_K_Br_Nayatani1997(63.66000000),
            1.000128455584031,
            places=7)

        self.assertAlmostEqual(
            coefficient_K_Br_Nayatani1997(1000.00000000),
            1.401080840298197,
            places=7)

        self.assertAlmostEqual(
            coefficient_K_Br_Nayatani1997(10000.00000000),
            1.592511806930447,
            places=7)

    def test_n_dimensional_coefficient_K_Br_Nayatani1997(self):
        """
        Tests :func:`colour.appearance.coefficient_K_Br_Nayatani1997`
        definition n_dimensional arrays support.
        """

        L_a = 63.66
        K_Br = coefficient_K_Br_Nayatani1997(L_a)

        L_a = np.tile(L_a, 6)
        K_Br = np.tile(K_Br, 6)

        np.testing.assert_almost_equal(
            coefficient_K_Br_Nayatani1997(L_a), K_Br, decimal=7)


class TestCoefficient_q_Nayatani1997(unittest.TestCase):
    """
    Defines :func:`colour.appearance.coefficient_q_Nayatani1997`
    definition unit tests methods.
    """

    def test_coefficient_q_Nayatani1997(self):
        """
        Tests :func:`colour.appearance.coefficient_q_Nayatani1997`
        definition.
        """

        self.assertAlmostEqual(
            coefficient_q_Nayatani1997(0.00000000),
            -0.121200000000000,
            places=7)

        self.assertAlmostEqual(
            coefficient_q_Nayatani1997(0.78539816),
            0.125211117768464,
            places=7)

        self.assertAlmostEqual(
            coefficient_q_Nayatani1997(1.57079633),
            0.191679999416415,
            places=7)

        self.assertAlmostEqual(
            coefficient_q_Nayatani1997(2.35619449),
            0.028480866426611,
            places=7)

    def test_n_dimensional_coefficient_q_Nayatani1997(self):
        """
        Tests :func:`colour.appearance.coefficient_q_Nayatani1997`
        definition n_dimensional arrays support.
        """

        L_a = 63.66
        theta = 1.50000000
        q = coefficient_q_Nayatani1997(L_a)

        theta = np.tile(theta, 6)
        q = np.tile(q, 6)

        np.testing.assert_almost_equal(
            coefficient_q_Nayatani1997(theta), q, decimal=7)
