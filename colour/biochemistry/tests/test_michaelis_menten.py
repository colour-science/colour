"""
Define the unit tests for the :mod:`colour.biochemistry.michaelis_menten`
module.
"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np

from colour.biochemistry import (
    reaction_rate_MichaelisMenten,
    reaction_rate_MichaelisMenten_Abebe2017,
    reaction_rate_MichaelisMenten_Michaelis1913,
    substrate_concentration_MichaelisMenten,
    substrate_concentration_MichaelisMenten_Abebe2017,
    substrate_concentration_MichaelisMenten_Michaelis1913,
)
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.utilities import (
    as_ndarray,
    ignore_numpy_errors,
    xp_as_array,
    xp_assert_close,
    xp_reshape,
)

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

    def test_reaction_rate_MichaelisMenten_Michaelis1913(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.biochemistry.michaelis_menten.\
reaction_rate_MichaelisMenten_Michaelis1913` definition.
        """

        xp_assert_close(
            reaction_rate_MichaelisMenten_Michaelis1913(
                xp_as_array([0.25], xp=xp),
                xp_as_array([0.5], xp=xp),
                xp_as_array([0.25], xp=xp),
            ),
            [0.250000000000000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            reaction_rate_MichaelisMenten_Michaelis1913(
                xp_as_array([0.5], xp=xp),
                xp_as_array([0.5], xp=xp),
                xp_as_array([0.25], xp=xp),
            ),
            [0.333333333333333],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            reaction_rate_MichaelisMenten_Michaelis1913(
                xp_as_array([0.65], xp=xp),
                xp_as_array([0.75], xp=xp),
                xp_as_array([0.35], xp=xp),
            ),
            [0.487500000000000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_reaction_rate_MichaelisMenten_Michaelis1913(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.biochemistry.michaelis_menten.\
reaction_rate_MichaelisMenten_Michaelis1913` definition n-dimensional arrays
        support.
        """

        v = 0.5
        V_max = 0.5
        K_m = 0.25
        S = as_ndarray(reaction_rate_MichaelisMenten_Michaelis1913(v, V_max, K_m))

        v = xp.tile(xp_as_array(v, xp=xp), (6, 1))
        S = xp.tile(xp_as_array(S, xp=xp), (6, 1))
        xp_assert_close(
            reaction_rate_MichaelisMenten_Michaelis1913(v, V_max, K_m),
            S,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        V_max = xp.tile(xp_as_array(V_max, xp=xp), (6, 1))
        K_m = xp.tile(xp_as_array(K_m, xp=xp), (6, 1))
        xp_assert_close(
            reaction_rate_MichaelisMenten_Michaelis1913(v, V_max, K_m),
            S,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        v = xp_reshape(xp_as_array(v, xp=xp), (2, 3, 1), xp=xp)
        V_max = xp_reshape(xp_as_array(V_max, xp=xp), (2, 3, 1), xp=xp)
        K_m = xp_reshape(xp_as_array(K_m, xp=xp), (2, 3, 1), xp=xp)
        S = xp_reshape(xp_as_array(S, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            reaction_rate_MichaelisMenten_Michaelis1913(v, V_max, K_m),
            S,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_reaction_rate_MichaelisMenten_Michaelis1913(self) -> None:
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

    def test_substrate_concentration_MichaelisMenten_Michaelis1913(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.biochemistry.michaelis_menten.\
substrate_concentration_MichaelisMenten_Michaelis1913` definition.
        """

        xp_assert_close(
            substrate_concentration_MichaelisMenten_Michaelis1913(
                xp_as_array([0.25], xp=xp),
                xp_as_array([0.5], xp=xp),
                xp_as_array([0.25], xp=xp),
            ),
            [0.250000000000000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            substrate_concentration_MichaelisMenten_Michaelis1913(
                xp_as_array([1 / 3], xp=xp),
                xp_as_array([0.5], xp=xp),
                xp_as_array([0.25], xp=xp),
            ),
            [0.500000000000000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            substrate_concentration_MichaelisMenten_Michaelis1913(
                xp_as_array([0.4875], xp=xp),
                xp_as_array([0.75], xp=xp),
                xp_as_array([0.35], xp=xp),
            ),
            [0.650000000000000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_substrate_concentration_MichaelisMenten_Michaelis1913(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.biochemistry.michaelis_menten.\
substrate_concentration_MichaelisMenten_Michaelis1913` definition n-dimensional
        arrays support.
        """

        S = 1 / 3
        V_max = 0.5
        K_m = 0.25
        v = as_ndarray(
            substrate_concentration_MichaelisMenten_Michaelis1913(S, V_max, K_m)
        )

        S = xp.tile(xp_as_array(S, xp=xp), (6, 1))
        v = xp.tile(xp_as_array(v, xp=xp), (6, 1))
        xp_assert_close(
            substrate_concentration_MichaelisMenten_Michaelis1913(S, V_max, K_m),
            v,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        V_max = xp.tile(xp_as_array(V_max, xp=xp), (6, 1))
        K_m = xp.tile(xp_as_array(K_m, xp=xp), (6, 1))
        xp_assert_close(
            substrate_concentration_MichaelisMenten_Michaelis1913(S, V_max, K_m),
            v,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        S = xp_reshape(xp_as_array(S, xp=xp), (2, 3, 1), xp=xp)
        V_max = xp_reshape(xp_as_array(V_max, xp=xp), (2, 3, 1), xp=xp)
        K_m = xp_reshape(xp_as_array(K_m, xp=xp), (2, 3, 1), xp=xp)
        v = xp_reshape(xp_as_array(v, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            substrate_concentration_MichaelisMenten_Michaelis1913(S, V_max, K_m),
            v,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_substrate_concentration_MichaelisMenten_Michaelis1913(self) -> None:
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

    def test_reaction_rate_MichaelisMenten_Abebe2017(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.biochemistry.michaelis_menten.\
reaction_rate_MichaelisMenten_Abebe2017` definition.
        """

        xp_assert_close(
            reaction_rate_MichaelisMenten_Abebe2017(
                xp_as_array([0.25], xp=xp),
                xp_as_array([0.5], xp=xp),
                xp_as_array([0.25], xp=xp),
                xp_as_array([0.25], xp=xp),
            ),
            [0.400000000000000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            reaction_rate_MichaelisMenten_Abebe2017(
                xp_as_array([0.5], xp=xp),
                xp_as_array([0.5], xp=xp),
                xp_as_array([0.25], xp=xp),
                xp_as_array([0.25], xp=xp),
            ),
            [0.666666666666666],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            reaction_rate_MichaelisMenten_Abebe2017(
                xp_as_array([0.65], xp=xp),
                xp_as_array([0.75], xp=xp),
                xp_as_array([0.35], xp=xp),
                xp_as_array([0.25], xp=xp),
            ),
            [0.951219512195122],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_reaction_rate_MichaelisMenten_Abebe2017(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.biochemistry.michaelis_menten.\
reaction_rate_MichaelisMenten_Abebe2017` definition n-dimensional arrays
        support.
        """

        v = 0.5
        V_max = 0.5
        K_m = 0.25
        b_m = 0.25
        S = as_ndarray(reaction_rate_MichaelisMenten_Abebe2017(v, V_max, K_m, b_m))

        v = xp.tile(xp_as_array(v, xp=xp), (6, 1))
        S = xp.tile(xp_as_array(S, xp=xp), (6, 1))
        xp_assert_close(
            reaction_rate_MichaelisMenten_Abebe2017(v, V_max, K_m, b_m),
            S,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        V_max = xp.tile(xp_as_array(V_max, xp=xp), (6, 1))
        K_m = xp.tile(xp_as_array(K_m, xp=xp), (6, 1))
        b_m = xp.tile(xp_as_array(b_m, xp=xp), (6, 1))
        xp_assert_close(
            reaction_rate_MichaelisMenten_Abebe2017(v, V_max, K_m, b_m),
            S,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        v = xp_reshape(xp_as_array(v, xp=xp), (2, 3, 1), xp=xp)
        V_max = xp_reshape(xp_as_array(V_max, xp=xp), (2, 3, 1), xp=xp)
        K_m = xp_reshape(xp_as_array(K_m, xp=xp), (2, 3, 1), xp=xp)
        b_m = xp_reshape(xp_as_array(b_m, xp=xp), (2, 3, 1), xp=xp)
        S = xp_reshape(xp_as_array(S, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            reaction_rate_MichaelisMenten_Abebe2017(v, V_max, K_m, b_m),
            S,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_reaction_rate_MichaelisMenten_Abebe2017(self) -> None:
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

    def test_substrate_concentration_MichaelisMenten_Abebe2017(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.biochemistry.michaelis_menten.\
substrate_concentration_MichaelisMenten_Abebe2017` definition.
        """

        xp_assert_close(
            substrate_concentration_MichaelisMenten_Abebe2017(
                xp_as_array([0.400000000000000], xp=xp),
                xp_as_array([0.5], xp=xp),
                xp_as_array([0.25], xp=xp),
                xp_as_array([0.25], xp=xp),
            ),
            [0.250000000000000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            substrate_concentration_MichaelisMenten_Abebe2017(
                xp_as_array([0.666666666666666], xp=xp),
                xp_as_array([0.5], xp=xp),
                xp_as_array([0.25], xp=xp),
                xp_as_array([0.25], xp=xp),
            ),
            [0.500000000000000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            substrate_concentration_MichaelisMenten_Abebe2017(
                xp_as_array([0.951219512195122], xp=xp),
                xp_as_array([0.75], xp=xp),
                xp_as_array([0.35], xp=xp),
                xp_as_array([0.25], xp=xp),
            ),
            [0.650000000000000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_substrate_concentration_MichaelisMenten_Abebe2017(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.biochemistry.michaelis_menten.\
substrate_concentration_MichaelisMenten_Abebe2017` definition n-dimensional
        arrays support.
        """

        S = 0.400000000000000
        V_max = 0.5
        K_m = 0.25
        b_m = 0.25
        v = as_ndarray(
            substrate_concentration_MichaelisMenten_Abebe2017(S, V_max, K_m, b_m)
        )

        S = xp.tile(xp_as_array(S, xp=xp), (6, 1))
        v = xp.tile(xp_as_array(v, xp=xp), (6, 1))
        xp_assert_close(
            substrate_concentration_MichaelisMenten_Abebe2017(S, V_max, K_m, b_m),
            v,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        V_max = xp.tile(xp_as_array(V_max, xp=xp), (6, 1))
        K_m = xp.tile(xp_as_array(K_m, xp=xp), (6, 1))
        b_m = xp.tile(xp_as_array(b_m, xp=xp), (6, 1))
        xp_assert_close(
            substrate_concentration_MichaelisMenten_Abebe2017(S, V_max, K_m, b_m),
            v,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        S = xp_reshape(xp_as_array(S, xp=xp), (2, 3, 1), xp=xp)
        V_max = xp_reshape(xp_as_array(V_max, xp=xp), (2, 3, 1), xp=xp)
        K_m = xp_reshape(xp_as_array(K_m, xp=xp), (2, 3, 1), xp=xp)
        b_m = xp_reshape(xp_as_array(b_m, xp=xp), (2, 3, 1), xp=xp)
        v = xp_reshape(xp_as_array(v, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            substrate_concentration_MichaelisMenten_Abebe2017(S, V_max, K_m, b_m),
            v,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_substrate_concentration_MichaelisMenten_Abebe2017(self) -> None:
        """
        Test :func:`colour.biochemistry.michaelis_menten.\
substrate_concentration_MichaelisMenten_Abebe2017` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        substrate_concentration_MichaelisMenten_Abebe2017(cases, cases, cases, cases)


class TestReactionRateMichaelisMenten:
    """
    Define :func:`colour.biochemistry.michaelis_menten.\
reaction_rate_MichaelisMenten` wrapper definition unit tests methods.
    """

    def test_reaction_rate_MichaelisMenten(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.biochemistry.michaelis_menten.\
reaction_rate_MichaelisMenten` wrapper definition.
        """

        xp_assert_close(
            reaction_rate_MichaelisMenten(
                xp_as_array([0.5], xp=xp),
                xp_as_array([2.5], xp=xp),
                xp_as_array([0.8], xp=xp),
            ),
            [0.961538461538461],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            reaction_rate_MichaelisMenten(
                xp_as_array([0.5], xp=xp),
                xp_as_array([2.5], xp=xp),
                xp_as_array([0.8], xp=xp),
                method="Abebe 2017",
                b_m=0.813,
            ),
            [1.036054742705597],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestSubstrateConcentrationMichaelisMenten:
    """
    Define :func:`colour.biochemistry.michaelis_menten.\
substrate_concentration_MichaelisMenten` wrapper definition unit tests methods.
    """

    def test_substrate_concentration_MichaelisMenten(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.biochemistry.michaelis_menten.\
substrate_concentration_MichaelisMenten` wrapper definition.
        """

        xp_assert_close(
            substrate_concentration_MichaelisMenten(
                xp_as_array([0.25], xp=xp),
                xp_as_array([0.5], xp=xp),
                xp_as_array([0.25], xp=xp),
            ),
            [0.250000000000000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            substrate_concentration_MichaelisMenten(
                xp_as_array([0.400000000000000], xp=xp),
                xp_as_array([0.5], xp=xp),
                xp_as_array([0.25], xp=xp),
                method="Abebe 2017",
                b_m=0.25,
            ),
            [0.250000000000000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
