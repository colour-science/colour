# -*- coding: utf-8 -*-
"""
Defines the unit tests for the :mod:`colour.models.rgb.transfer_functions.\
blackmagic_design` module.
"""

import numpy as np
import unittest

from colour.models.rgb.transfer_functions import (
    oetf_BlackmagicFilmGeneration5,
    oetf_inverse_BlackmagicFilmGeneration5,
)
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestOetf_BlackmagicFilmGeneration5',
    'TestOetf_inverse_BlackmagicFilmGeneration5',
]


class TestOetf_BlackmagicFilmGeneration5(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.blackmagic_design.\
oetf_BlackmagicFilmGeneration5` definition unit tests methods.
    """

    def test_oetf_BlackmagicFilmGeneration5(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.\
blackmagic_design.oetf_BlackmagicFilmGeneration5` definition.
        """

        self.assertAlmostEqual(
            oetf_BlackmagicFilmGeneration5(0.0), 0.092465753424658, places=7)

        self.assertAlmostEqual(
            oetf_BlackmagicFilmGeneration5(0.18), 0.383561643835617, places=7)

        self.assertAlmostEqual(
            oetf_BlackmagicFilmGeneration5(1.0), 0.530489624957305, places=7)

        self.assertAlmostEqual(
            oetf_BlackmagicFilmGeneration5(100.0), 0.930339851899973, places=7)

        self.assertAlmostEqual(
            oetf_BlackmagicFilmGeneration5(222.86),
            0.999999631713769,
            places=7)

    def test_n_dimensional_oetf_BlackmagicFilmGeneration5(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.\
blackmagic_design.oetf_BlackmagicFilmGeneration5` definition n-dimensional
        arrays support.
        """

        L = 0.18
        V = oetf_BlackmagicFilmGeneration5(L)

        L = np.tile(L, 6)
        V = np.tile(V, 6)
        np.testing.assert_almost_equal(
            oetf_BlackmagicFilmGeneration5(L), V, decimal=7)

        L = np.reshape(L, (2, 3))
        V = np.reshape(V, (2, 3))
        np.testing.assert_almost_equal(
            oetf_BlackmagicFilmGeneration5(L), V, decimal=7)

        L = np.reshape(L, (2, 3, 1))
        V = np.reshape(V, (2, 3, 1))
        np.testing.assert_almost_equal(
            oetf_BlackmagicFilmGeneration5(L), V, decimal=7)

    def test_domain_range_scale_oetf_BlackmagicFilmGeneration5(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.\
blackmagic_design.oetf_BlackmagicFilmGeneration5` definition domain and range
        scale support.
        """

        L = 0.18
        V = oetf_BlackmagicFilmGeneration5(L)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    oetf_BlackmagicFilmGeneration5(L * factor),
                    V * factor,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_oetf_BlackmagicFilmGeneration5(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.\
blackmagic_design.oetf_BlackmagicFilmGeneration5` definition nan support.
        """

        oetf_BlackmagicFilmGeneration5(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestOetf_inverse_BlackmagicFilmGeneration5(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.\
blackmagic_design.oetf_inverse_BlackmagicFilmGeneration5` definition unit tests
    methods.
    """

    def test_oetf_inverse_BlackmagicFilmGeneration5(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.\
blackmagic_design.oetf_inverse_BlackmagicFilmGeneration5` definition.
        """

        self.assertAlmostEqual(
            oetf_inverse_BlackmagicFilmGeneration5(0.092465753424658),
            0.0,
            places=7)

        self.assertAlmostEqual(
            oetf_inverse_BlackmagicFilmGeneration5(0.383561643835617),
            0.18,
            places=7)

        self.assertAlmostEqual(
            oetf_inverse_BlackmagicFilmGeneration5(0.530489624957305),
            1.0,
            places=7)

        self.assertAlmostEqual(
            oetf_inverse_BlackmagicFilmGeneration5(0.930339851899973),
            100.0,
            places=7)

        self.assertAlmostEqual(
            oetf_inverse_BlackmagicFilmGeneration5(0.999999631713769),
            222.86,
            places=7)

    def test_n_dimensional_oetf_inverse_BlackmagicFilmGeneration5(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.\
blackmagic_design.oetf_inverse_BlackmagicFilmGeneration5` definition
        n-dimensional arrays support.
        """

        V = 0.383561643835617
        L = oetf_inverse_BlackmagicFilmGeneration5(V)

        V = np.tile(V, 6)
        L = np.tile(L, 6)
        np.testing.assert_almost_equal(
            oetf_inverse_BlackmagicFilmGeneration5(V), L, decimal=7)

        V = np.reshape(V, (2, 3))
        L = np.reshape(L, (2, 3))
        np.testing.assert_almost_equal(
            oetf_inverse_BlackmagicFilmGeneration5(V), L, decimal=7)

        V = np.reshape(V, (2, 3, 1))
        L = np.reshape(L, (2, 3, 1))
        np.testing.assert_almost_equal(
            oetf_inverse_BlackmagicFilmGeneration5(V), L, decimal=7)

    def test_domain_range_scale_oetf_inverse_BlackmagicFilmGeneration5(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.\
blackmagic_design.oetf_inverse_BlackmagicFilmGeneration5` definition domain and
        range scale support.
        """

        V = 0.383561643835617
        L = oetf_inverse_BlackmagicFilmGeneration5(V)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    oetf_inverse_BlackmagicFilmGeneration5(V * factor),
                    L * factor,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_oetf_inverse_BlackmagicFilmGeneration5(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.\
blackmagic_design.oetf_inverse_BlackmagicFilmGeneration5` definition nan
        support.
        """

        oetf_inverse_BlackmagicFilmGeneration5(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


if __name__ == '__main__':
    unittest.main()
