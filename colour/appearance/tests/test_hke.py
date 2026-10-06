"""Define the unit tests for the :mod:`colour.appearance.hke` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np

from colour.appearance.hke import (
    HelmholtzKohlrausch_effect_luminous_Nayatani1997,
    HelmholtzKohlrausch_effect_object_Nayatani1997,
    coefficient_K_Br_Nayatani1997,
    coefficient_q_Nayatani1997,
)
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.utilities import (
    as_ndarray,
    ignore_numpy_errors,
    xp_as_array,
    xp_assert_close,
    xp_reshape,
)

__author__ = "Ilia Sibiryakov"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestHelmholtzKohlrauschEffectObjectNayatani1997",
    "TestHelmholtzKohlrauschEffectLuminousNayatani1997",
    "TestCoefficient_K_Br_Nayatani1997",
    "TestCoefficient_q_Nayatani1997",
]


class TestHelmholtzKohlrauschEffectObjectNayatani1997:
    """
    Define :func:`colour.colour.appearance.hke.\
HelmholtzKohlrausch_effect_object_Nayatani1997` definition unit tests methods.
    """

    def test_HelmholtzKohlrausch_effect_object_Nayatani1997(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.appearance.hke.\
HelmholtzKohlrausch_effect_object_Nayatani1997` definition.
        """

        xp_assert_close(
            HelmholtzKohlrausch_effect_object_Nayatani1997(
                xp_as_array([0.40351010, 0.53933673], xp=xp),
                xp_as_array([0.19783001, 0.46831999], xp=xp),
                63.66,
                method="VCC",
            ),
            1.344152435497761,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            HelmholtzKohlrausch_effect_object_Nayatani1997(
                xp_as_array([0.40351010, 0.53933673], xp=xp),
                xp_as_array([0.19783001, 0.46831999], xp=xp),
                63.66,
                method="VAC",
            ),
            1.261777232837009,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_HelmholtzKohlrausch_effect_object_Nayatani1997(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.appearance.hke.\
HelmholtzKohlrausch_effect_object_Nayatani1997` definition n_dimensional
        arrays support.
        """

        uv_d65 = xp_as_array([0.19783001, 0.46831999], xp=xp)
        uv = xp_as_array([0.40351010, 0.53933673], xp=xp)
        L_a = 63.66

        result_vcc = as_ndarray(
            HelmholtzKohlrausch_effect_object_Nayatani1997(
                uv, uv_d65, L_a, method="VCC"
            )
        )
        result_vac = as_ndarray(
            HelmholtzKohlrausch_effect_object_Nayatani1997(
                uv, uv_d65, L_a, method="VAC"
            )
        )

        uv_d65 = xp.tile(xp_as_array(uv_d65, xp=xp), (6, 1))
        uv = xp.tile(xp_as_array(uv, xp=xp), (6, 1))
        result_vcc = xp.tile(xp_as_array(result_vcc, xp=xp), (6,))
        result_vac = xp.tile(xp_as_array(result_vac, xp=xp), (6,))

        xp_assert_close(
            HelmholtzKohlrausch_effect_object_Nayatani1997(
                uv, uv_d65, L_a, method="VCC"
            ),
            result_vcc,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            HelmholtzKohlrausch_effect_object_Nayatani1997(
                uv, uv_d65, L_a, method="VAC"
            ),
            result_vac,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        uv_d65 = xp_reshape(xp_as_array(uv_d65, xp=xp), (2, 3, 2), xp=xp)
        uv = xp_reshape(xp_as_array(uv, xp=xp), (2, 3, 2), xp=xp)
        result_vcc = xp_reshape(xp_as_array(result_vcc, xp=xp), (2, 3), xp=xp)
        result_vac = xp_reshape(xp_as_array(result_vac, xp=xp), (2, 3), xp=xp)

        xp_assert_close(
            HelmholtzKohlrausch_effect_object_Nayatani1997(
                uv, uv_d65, L_a, method="VCC"
            ),
            result_vcc,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            HelmholtzKohlrausch_effect_object_Nayatani1997(
                uv, uv_d65, L_a, method="VAC"
            ),
            result_vac,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_HelmholtzKohlrausch_effect_object_Nayatani1997(self) -> None:
        """
        Test :func:`colour.appearance.hke.\
HelmholtzKohlrausch_effect_object_Nayatani1997` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        for case in cases:
            HelmholtzKohlrausch_effect_object_Nayatani1997(case, case, case[0])


class TestHelmholtzKohlrauschEffectLuminousNayatani1997:
    """
    Define :func:`colour.appearance.hke.\
HelmholtzKohlrausch_effect_luminous_Nayatani1997` definition unit tests
    methods.
    """

    def test_HelmholtzKohlrausch_effect_luminous_Nayatani1997(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.appearance.hke.\
HelmholtzKohlrausch_effect_luminous_Nayatani1997` definition.
        """

        xp_assert_close(
            HelmholtzKohlrausch_effect_luminous_Nayatani1997(
                xp_as_array([0.40351010, 0.53933673], xp=xp),
                xp_as_array([0.19783001, 0.46831999], xp=xp),
                63.66,
                method="VCC",
            ),
            2.014433723774654,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            HelmholtzKohlrausch_effect_luminous_Nayatani1997(
                xp_as_array([0.40351010, 0.53933673], xp=xp),
                xp_as_array([0.19783001, 0.46831999], xp=xp),
                63.66,
                method="VAC",
            ),
            1.727991241148628,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_HelmholtzKohlrausch_effect_luminous_Nayatani1997(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.appearance.hke.\
HelmholtzKohlrausch_effect_luminous_Nayatani1997` definition n_dimensional
        arrays support.
        """

        uv_d65 = xp_as_array([0.19783001, 0.46831999], xp=xp)
        uv = xp_as_array([0.40351010, 0.53933673], xp=xp)
        L_a = 63.66

        result_vcc = as_ndarray(
            HelmholtzKohlrausch_effect_luminous_Nayatani1997(
                uv, uv_d65, L_a, method="VCC"
            )
        )
        result_vac = as_ndarray(
            HelmholtzKohlrausch_effect_luminous_Nayatani1997(
                uv, uv_d65, L_a, method="VAC"
            )
        )

        uv_d65 = xp.tile(xp_as_array(uv_d65, xp=xp), (6, 1))
        uv = xp.tile(xp_as_array(uv, xp=xp), (6, 1))
        result_vcc = xp.tile(xp_as_array(result_vcc, xp=xp), (6,))
        result_vac = xp.tile(xp_as_array(result_vac, xp=xp), (6,))

        xp_assert_close(
            HelmholtzKohlrausch_effect_luminous_Nayatani1997(
                uv, uv_d65, L_a, method="VCC"
            ),
            result_vcc,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            HelmholtzKohlrausch_effect_luminous_Nayatani1997(
                uv, uv_d65, L_a, method="VAC"
            ),
            result_vac,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        uv_d65 = xp_reshape(xp_as_array(uv_d65, xp=xp), (2, 3, 2), xp=xp)
        uv = xp_reshape(xp_as_array(uv, xp=xp), (2, 3, 2), xp=xp)
        result_vcc = xp_reshape(xp_as_array(result_vcc, xp=xp), (2, 3), xp=xp)
        result_vac = xp_reshape(xp_as_array(result_vac, xp=xp), (2, 3), xp=xp)

        xp_assert_close(
            HelmholtzKohlrausch_effect_luminous_Nayatani1997(
                uv, uv_d65, L_a, method="VCC"
            ),
            result_vcc,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            HelmholtzKohlrausch_effect_luminous_Nayatani1997(
                uv, uv_d65, L_a, method="VAC"
            ),
            result_vac,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_HelmholtzKohlrausch_effect_luminous_Nayatani1997(self) -> None:
        """
        Test :func:`colour.appearance.hke.\
HelmholtzKohlrausch_effect_luminous_Nayatani1997` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        for case in cases:
            HelmholtzKohlrausch_effect_luminous_Nayatani1997(case, case, case[0])


class TestCoefficient_K_Br_Nayatani1997:
    """
    Define :func:`colour.appearance.hke.coefficient_K_Br_Nayatani1997`
    definition unit tests methods.
    """

    def test_coefficient_K_Br_Nayatani1997(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.appearance.hke.coefficient_K_Br_Nayatani1997`
        definition.
        """

        xp_assert_close(
            coefficient_K_Br_Nayatani1997(xp_as_array([10.0], xp=xp)),
            0.71344817765758839,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            coefficient_K_Br_Nayatani1997(xp_as_array([63.66], xp=xp)),
            1.000128455584031,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            coefficient_K_Br_Nayatani1997(xp_as_array([1000.0], xp=xp)),
            1.401080840298197,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            coefficient_K_Br_Nayatani1997(xp_as_array([10000.0], xp=xp)),
            1.592511806930447,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_coefficient_K_Br_Nayatani1997(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.appearance.hke.coefficient_K_Br_Nayatani1997`
        definition n_dimensional arrays support.
        """

        L_a = 63.66
        K_Br = coefficient_K_Br_Nayatani1997(L_a)

        L_a = xp.tile(xp_as_array(L_a, xp=xp), (6,))
        K_Br = xp.tile(xp_as_array(K_Br, xp=xp), (6,))
        xp_assert_close(
            coefficient_K_Br_Nayatani1997(L_a),
            K_Br,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        L_a = xp_reshape(xp_as_array(L_a, xp=xp), (2, 3), xp=xp)
        K_Br = xp_reshape(xp_as_array(K_Br, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            coefficient_K_Br_Nayatani1997(L_a),
            K_Br,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        L_a = xp_reshape(xp_as_array(L_a, xp=xp), (2, 3, 1), xp=xp)
        K_Br = xp_reshape(xp_as_array(K_Br, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            coefficient_K_Br_Nayatani1997(L_a),
            K_Br,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_coefficient_K_Br_Nayatani1997(self) -> None:
        """
        Test :func:`colour.appearance.hke.coefficient_K_Br_Nayatani1997`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        coefficient_K_Br_Nayatani1997(cases)


class TestCoefficient_q_Nayatani1997:
    """
    Define :func:`colour.appearance.hke.coefficient_q_Nayatani1997`
    definition unit tests methods.
    """

    def test_coefficient_q_Nayatani1997(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.appearance.hke.coefficient_q_Nayatani1997`
        definition.
        """

        xp_assert_close(
            coefficient_q_Nayatani1997(xp_as_array([0.0], xp=xp)),
            -0.121200000000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            coefficient_q_Nayatani1997(xp_as_array([0.78539816], xp=xp)),
            0.125211117768464,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            coefficient_q_Nayatani1997(xp_as_array([1.57079633], xp=xp)),
            0.191679999416415,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            coefficient_q_Nayatani1997(xp_as_array([2.35619449], xp=xp)),
            0.028480866426611,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_coefficient_q_Nayatani1997(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.appearance.hke.coefficient_q_Nayatani1997`
        definition n_dimensional arrays support.
        """

        L_a = 63.66
        q = coefficient_q_Nayatani1997(L_a)

        L_a = xp.tile(xp_as_array(L_a, xp=xp), (6,))
        q = xp.tile(xp_as_array(q, xp=xp), (6,))
        xp_assert_close(
            coefficient_q_Nayatani1997(L_a),
            q,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        L_a = xp_reshape(xp_as_array(L_a, xp=xp), (2, 3), xp=xp)
        q = xp_reshape(xp_as_array(q, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            coefficient_q_Nayatani1997(L_a),
            q,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        L_a = xp_reshape(xp_as_array(L_a, xp=xp), (2, 3, 1), xp=xp)
        q = xp_reshape(xp_as_array(q, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            coefficient_q_Nayatani1997(L_a),
            q,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_coefficient_q_Nayatani1997(self) -> None:
        """
        Test :func:`colour.appearance.hke.coefficient_q_Nayatani1997`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        coefficient_q_Nayatani1997(cases)
