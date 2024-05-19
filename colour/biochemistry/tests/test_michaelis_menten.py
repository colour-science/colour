"""
Define the unit tests for the :mod:`colour.biochemistry.michaelis_menten`
module.
"""

from itertools import product

import numpy as np

from colour.biochemistry import (
    reaction_rate_MichaelisMenten_Abebe2017,
    reaction_rate_MichaelisMenten_Michaelis1913,
    substrate_concentration_MichaelisMenten_Abebe2017,
    substrate_concentration_MichaelisMenten_Michaelis1913,
)
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.utilities import ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestReactionRateMichaelisMentenMichaelis1913",
    "TestSubstrateConcentrationMichaelisMentenMichaelis1913",
    "TestReactionRateMichaelisMentenAbebe2017",
    "TestSubstrateConcentrationMichaelisMentenAbebe2017",
]


class TestReactionRateMichaelisMentenMichaelis1913:
    """
    Define :func:`colour.biochemistry.michaelis_menten.\
reaction_rate_MichaelisMenten_Michaelis1913` definition unit tests methods.
    """

    def test_reaction_rate_MichaelisMenten_Michaelis1913(self):
        """
        Test :func:`colour.biochemistry.michaelis_menten.\
reaction_rate_MichaelisMenten_Michaelis1913` definition.
        """

        np.testing.assert_allclose(
            reaction_rate_MichaelisMenten_Michaelis1913(0.25, 0.5, 0.25),
            0.250000000000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            reaction_rate_MichaelisMenten_Michaelis1913(0.5, 0.5, 0.25),
            0.333333333333333,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            reaction_rate_MichaelisMenten_Michaelis1913(0.65, 0.75, 0.35),
            0.487500000000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_reaction_rate_MichaelisMenten_Michaelis1913(self):
        """
        Test :func:`colour.biochemistry.michaelis_menten.\
reaction_rate_MichaelisMenten_Michaelis1913` definition n-dimensional arrays
        support.
        """

        v = 0.5
        V_max = 0.5
        K_m = 0.25
        S = reaction_rate_MichaelisMenten_Michaelis1913(v, V_max, K_m)

        v = np.tile(v, (6, 1))
        S = np.tile(S, (6, 1))
        np.testing.assert_allclose(
            reaction_rate_MichaelisMenten_Michaelis1913(v, V_max, K_m),
            S,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        V_max = np.tile(V_max, (6, 1))
        K_m = np.tile(K_m, (6, 1))
        np.testing.assert_allclose(
            reaction_rate_MichaelisMenten_Michaelis1913(v, V_max, K_m),
            S,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        v = np.reshape(v, (2, 3, 1))
        V_max = np.reshape(V_max, (2, 3, 1))
        K_m = np.reshape(K_m, (2, 3, 1))
        S = np.reshape(S, (2, 3, 1))
        np.testing.assert_allclose(
            reaction_rate_MichaelisMenten_Michaelis1913(v, V_max, K_m),
            S,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_reaction_rate_MichaelisMenten_Michaelis1913(self):
        """
        Test :func:`colour.biochemistry.michaelis_menten.\
reaction_rate_MichaelisMenten_Michaelis1913` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        reaction_rate_MichaelisMenten_Michaelis1913(cases, cases, cases)


class TestSubstrateConcentrationMichaelisMentenMichaelis1913:
    """
    Define :func:`colour.biochemistry.michaelis_menten.\
reaction_rate_MichaelisMenten_Michaelis1913` definition unit tests methods.
    """

    def test_substrate_concentration_MichaelisMenten_Michaelis1913(self):
        """
        Test :func:`colour.biochemistry.michaelis_menten.\
substrate_concentration_MichaelisMenten_Michaelis1913` definition.
        """

        np.testing.assert_allclose(
            substrate_concentration_MichaelisMenten_Michaelis1913(0.25, 0.5, 0.25),
            0.250000000000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            substrate_concentration_MichaelisMenten_Michaelis1913(1 / 3, 0.5, 0.25),
            0.500000000000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            substrate_concentration_MichaelisMenten_Michaelis1913(0.4875, 0.75, 0.35),
            0.650000000000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_substrate_concentration_MichaelisMenten_Michaelis1913(
        self,
    ):
        """
        Test :func:`colour.biochemistry.michaelis_menten.\
substrate_concentration_MichaelisMenten_Michaelis1913` definition n-dimensional
        arrays support.
        """

        S = 1 / 3
        V_max = 0.5
        K_m = 0.25
        v = substrate_concentration_MichaelisMenten_Michaelis1913(S, V_max, K_m)

        S = np.tile(S, (6, 1))
        v = np.tile(v, (6, 1))
        np.testing.assert_allclose(
            substrate_concentration_MichaelisMenten_Michaelis1913(S, V_max, K_m),
            v,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        V_max = np.tile(V_max, (6, 1))
        K_m = np.tile(K_m, (6, 1))
        np.testing.assert_allclose(
            substrate_concentration_MichaelisMenten_Michaelis1913(S, V_max, K_m),
            v,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        S = np.reshape(S, (2, 3, 1))
        V_max = np.reshape(V_max, (2, 3, 1))
        K_m = np.reshape(K_m, (2, 3, 1))
        v = np.reshape(v, (2, 3, 1))
        np.testing.assert_allclose(
            substrate_concentration_MichaelisMenten_Michaelis1913(S, V_max, K_m),
            v,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_substrate_concentration_MichaelisMenten_Michaelis1913(self):
        """
        Test :func:`colour.biochemistry.michaelis_menten.\
substrate_concentration_MichaelisMenten_Michaelis1913` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        substrate_concentration_MichaelisMenten_Michaelis1913(cases, cases, cases)


class TestReactionRateMichaelisMentenAbebe2017:
    """
    Define :func:`colour.biochemistry.michaelis_menten.\
reaction_rate_MichaelisMenten_Abebe2017` definition unit tests methods.
    """

    def test_reaction_rate_MichaelisMenten_Abebe2017(self):
        """
        Test :func:`colour.biochemistry.michaelis_menten.\
reaction_rate_MichaelisMenten_Abebe2017` definition.
        """

        np.testing.assert_allclose(
            reaction_rate_MichaelisMenten_Abebe2017(0.25, 0.5, 0.25, 0.25),
            0.400000000000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            reaction_rate_MichaelisMenten_Abebe2017(0.5, 0.5, 0.25, 0.25),
            0.666666666666666,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            reaction_rate_MichaelisMenten_Abebe2017(0.65, 0.75, 0.35, 0.25),
            0.951219512195122,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_reaction_rate_MichaelisMenten_Abebe2017(self):
        """
        Test :func:`colour.biochemistry.michaelis_menten.\
reaction_rate_MichaelisMenten_Abebe2017` definition n-dimensional arrays
        support.
        """

        v = 0.5
        V_max = 0.5
        K_m = 0.25
        b_m = 0.25
        S = reaction_rate_MichaelisMenten_Abebe2017(v, V_max, K_m, b_m)

        v = np.tile(v, (6, 1))
        S = np.tile(S, (6, 1))
        np.testing.assert_allclose(
            reaction_rate_MichaelisMenten_Abebe2017(v, V_max, K_m, b_m),
            S,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        V_max = np.tile(V_max, (6, 1))
        K_m = np.tile(K_m, (6, 1))
        b_m = np.tile(b_m, (6, 1))
        np.testing.assert_allclose(
            reaction_rate_MichaelisMenten_Abebe2017(v, V_max, K_m, b_m),
            S,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        v = np.reshape(v, (2, 3, 1))
        V_max = np.reshape(V_max, (2, 3, 1))
        K_m = np.reshape(K_m, (2, 3, 1))
        b_m = np.reshape(b_m, (2, 3, 1))
        S = np.reshape(S, (2, 3, 1))
        np.testing.assert_allclose(
            reaction_rate_MichaelisMenten_Abebe2017(v, V_max, K_m, b_m),
            S,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_reaction_rate_MichaelisMenten_Abebe2017(self):
        """
        Test :func:`colour.biochemistry.michaelis_menten.\
reaction_rate_MichaelisMenten_Abebe2017` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        reaction_rate_MichaelisMenten_Abebe2017(cases, cases, cases, cases)


class TestSubstrateConcentrationMichaelisMentenAbebe2017:
    """
    Define :func:`colour.biochemistry.michaelis_menten.\
reaction_rate_MichaelisMenten_Abebe2017` definition unit tests methods.
    """

    def test_substrate_concentration_MichaelisMenten_Abebe2017(self):
        """
        Test :func:`colour.biochemistry.michaelis_menten.\
substrate_concentration_MichaelisMenten_Abebe2017` definition.
        """

        np.testing.assert_allclose(
            substrate_concentration_MichaelisMenten_Abebe2017(
                0.400000000000000, 0.5, 0.25, 0.25
            ),
            0.250000000000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            substrate_concentration_MichaelisMenten_Abebe2017(
                0.666666666666666, 0.5, 0.25, 0.25
            ),
            0.500000000000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            substrate_concentration_MichaelisMenten_Abebe2017(
                0.951219512195122, 0.75, 0.35, 0.25
            ),
            0.650000000000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_substrate_concentration_MichaelisMenten_Abebe2017(
        self,
    ):
        """
        Test :func:`colour.biochemistry.michaelis_menten.\
substrate_concentration_MichaelisMenten_Abebe2017` definition n-dimensional
        arrays support.
        """

        S = 0.400000000000000
        V_max = 0.5
        K_m = 0.25
        b_m = 0.25
        v = substrate_concentration_MichaelisMenten_Abebe2017(S, V_max, K_m, b_m)

        S = np.tile(S, (6, 1))
        v = np.tile(v, (6, 1))
        np.testing.assert_allclose(
            substrate_concentration_MichaelisMenten_Abebe2017(S, V_max, K_m, b_m),
            v,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        V_max = np.tile(V_max, (6, 1))
        K_m = np.tile(K_m, (6, 1))
        b_m = np.tile(b_m, (6, 1))
        np.testing.assert_allclose(
            substrate_concentration_MichaelisMenten_Abebe2017(S, V_max, K_m, b_m),
            v,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        S = np.reshape(S, (2, 3, 1))
        V_max = np.reshape(V_max, (2, 3, 1))
        K_m = np.reshape(K_m, (2, 3, 1))
        b_m = np.reshape(b_m, (2, 3, 1))
        v = np.reshape(v, (2, 3, 1))
        np.testing.assert_allclose(
            substrate_concentration_MichaelisMenten_Abebe2017(S, V_max, K_m, b_m),
            v,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_substrate_concentration_MichaelisMenten_Abebe2017(self):
        """
        Test :func:`colour.biochemistry.michaelis_menten.\
substrate_concentration_MichaelisMenten_Abebe2017` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        substrate_concentration_MichaelisMenten_Abebe2017(cases, cases, cases, cases)
