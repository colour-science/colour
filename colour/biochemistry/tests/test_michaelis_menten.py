# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.biochemistry.michaelis_menten` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from itertools import permutations

from colour.biochemistry import (reaction_rate_MichealisMenten,
                                 substrate_concentration_MichealisMenten)
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'TestReactionRateMichealisMenten',
    'TestSubstrateConcentrationMichealisMenten'
]


class TestReactionRateMichealisMenten(unittest.TestCase):
    """
    Defines :func:`colour.biochemistry.michaelis_menten.\
reaction_rate_MichealisMenten` definition unit tests methods.
    """

    def test_reaction_rate_MichealisMenten(self):
        """
        Tests :func:`colour.biochemistry.michaelis_menten.\
reaction_rate_MichealisMenten` definition.
        """

        self.assertAlmostEqual(
            reaction_rate_MichealisMenten(0.25, 0.5, 0.25),
            0.250000000000000,
            places=7)

        self.assertAlmostEqual(
            reaction_rate_MichealisMenten(0.5, 0.5, 0.25),
            0.333333333333333,
            places=7)

        self.assertAlmostEqual(
            reaction_rate_MichealisMenten(0.65, 0.75, 0.35),
            0.487500000000000,
            places=7)

    def test_n_dimensional_reaction_rate_MichealisMenten(self):
        """
        Tests :func:`colour.biochemistry.michaelis_menten.\
reaction_rate_MichealisMenten` definition n-dimensional arrays
        support.
        """

        v = 0.5
        V_max = 0.5
        K_m = 0.25
        S = 0.333333333333333
        np.testing.assert_almost_equal(
            reaction_rate_MichealisMenten(v, V_max, K_m), S, decimal=7)

        v = np.tile(v, (6, 1))
        S = np.tile(S, (6, 1))
        np.testing.assert_almost_equal(
            reaction_rate_MichealisMenten(v, V_max, K_m), S, decimal=7)

        V_max = np.tile(V_max, (6, 1))
        K_m = np.tile(K_m, (6, 1))
        np.testing.assert_almost_equal(
            reaction_rate_MichealisMenten(v, V_max, K_m), S, decimal=7)

        v = np.reshape(v, (2, 3, 1))
        V_max = np.reshape(V_max, (2, 3, 1))
        K_m = np.reshape(K_m, (2, 3, 1))
        S = np.reshape(S, (2, 3, 1))
        np.testing.assert_almost_equal(
            reaction_rate_MichealisMenten(v, V_max, K_m), S, decimal=7)

    @ignore_numpy_errors
    def test_nan_reaction_rate_MichealisMenten(self):
        """
        Tests :func:`colour.biochemistry.michaelis_menten.\
reaction_rate_MichealisMenten` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            v = np.array(case)
            V_max = np.array(case)
            K_m = np.array(case)
            reaction_rate_MichealisMenten(v, V_max, K_m)


class TestSubstrateConcentrationMichealisMenten(unittest.TestCase):
    """
    Defines :func:`colour.biochemistry.michaelis_menten.\
reaction_rate_MichealisMenten` definition unit tests methods.
    """

    def test_substrate_concentration_MichealisMenten(self):
        """
        Tests :func:`colour.biochemistry.michaelis_menten.\
substrate_concentration_MichealisMenten` definition.
        """

        self.assertAlmostEqual(
            substrate_concentration_MichealisMenten(0.25, 0.5, 0.25),
            0.250000000000000,
            places=7)

        self.assertAlmostEqual(
            substrate_concentration_MichealisMenten(1 / 3, 0.5, 0.25),
            0.500000000000000,
            places=7)

        self.assertAlmostEqual(
            substrate_concentration_MichealisMenten(0.4875, 0.75, 0.35),
            0.650000000000000,
            places=7)

    def test_n_dimensional_substrate_concentration_MichealisMenten(self):
        """
        Tests :func:`colour.biochemistry.michaelis_menten.\
substrate_concentration_MichealisMenten` definition n-dimensional arrays
        support.
        """

        S = 1 / 3
        V_max = 0.5
        K_m = 0.25
        v = 0.5
        np.testing.assert_almost_equal(
            substrate_concentration_MichealisMenten(S, V_max, K_m),
            v,
            decimal=7)

        S = np.tile(S, (6, 1))
        v = np.tile(v, (6, 1))
        np.testing.assert_almost_equal(
            substrate_concentration_MichealisMenten(S, V_max, K_m),
            v,
            decimal=7)

        V_max = np.tile(V_max, (6, 1))
        K_m = np.tile(K_m, (6, 1))
        np.testing.assert_almost_equal(
            substrate_concentration_MichealisMenten(S, V_max, K_m),
            v,
            decimal=7)

        S = np.reshape(S, (2, 3, 1))
        V_max = np.reshape(V_max, (2, 3, 1))
        K_m = np.reshape(K_m, (2, 3, 1))
        v = np.reshape(v, (2, 3, 1))
        np.testing.assert_almost_equal(
            substrate_concentration_MichealisMenten(S, V_max, K_m),
            v,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_substrate_concentration_MichealisMenten(self):
        """
        Tests :func:`colour.biochemistry.michaelis_menten.\
substrate_concentration_MichealisMenten` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            s = np.array(case)
            V_max = np.array(case)
            K_m = np.array(case)
            substrate_concentration_MichealisMenten(s, V_max, K_m)


if __name__ == '__main__':
    unittest.main()
