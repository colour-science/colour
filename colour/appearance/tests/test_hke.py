# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.appearance.hke` module."""

from itertools import product

import numpy as np

from colour.appearance.hke import (
    HelmholtzKohlrausch_effect_luminous_Nayatani1997,
    HelmholtzKohlrausch_effect_object_Nayatani1997,
    coefficient_K_Br_Nayatani1997,
    coefficient_q_Nayatani1997,
)
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.utilities import ignore_numpy_errors

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

    def test_HelmholtzKohlrausch_effect_object_Nayatani1997(self):
        """
        Test :func:`colour.appearance.hke.\
HelmholtzKohlrausch_effect_object_Nayatani1997` definition.
        """

        np.testing.assert_allclose(
            HelmholtzKohlrausch_effect_object_Nayatani1997(
                np.array([0.40351010, 0.53933673]),
                np.array([0.19783001, 0.46831999]),
                63.66,
                method="VCC",
            ),
            1.344152435497761,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            HelmholtzKohlrausch_effect_object_Nayatani1997(
                np.array([0.40351010, 0.53933673]),
                np.array([0.19783001, 0.46831999]),
                63.66,
                method="VAC",
            ),
            1.261777232837009,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_HelmholtzKohlrausch_effect_object_Nayatani1997(
        self,
    ):
        """
        Test :func:`colour.appearance.hke.\
HelmholtzKohlrausch_effect_object_Nayatani1997` definition n_dimensional
        arrays support.
        """

        uv_d65 = np.array([0.19783001, 0.46831999])
        uv = np.array([0.40351010, 0.53933673])
        L_a = 63.66

        result_vcc = HelmholtzKohlrausch_effect_object_Nayatani1997(
            uv, uv_d65, L_a, method="VCC"
        )
        result_vac = HelmholtzKohlrausch_effect_object_Nayatani1997(
            uv, uv_d65, L_a, method="VAC"
        )

        uv_d65 = np.tile(uv_d65, (6, 1))
        uv = np.tile(uv, (6, 1))
        result_vcc = np.tile(result_vcc, 6)
        result_vac = np.tile(result_vac, 6)

        np.testing.assert_allclose(
            HelmholtzKohlrausch_effect_object_Nayatani1997(
                uv, uv_d65, L_a, method="VCC"
            ),
            result_vcc,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            HelmholtzKohlrausch_effect_object_Nayatani1997(
                uv, uv_d65, L_a, method="VAC"
            ),
            result_vac,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        uv_d65 = np.reshape(uv_d65, (2, 3, 2))
        uv = np.reshape(uv, (2, 3, 2))
        result_vcc = np.reshape(result_vcc, (2, 3))
        result_vac = np.reshape(result_vac, (2, 3))

        np.testing.assert_allclose(
            HelmholtzKohlrausch_effect_object_Nayatani1997(
                uv, uv_d65, L_a, method="VCC"
            ),
            result_vcc,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            HelmholtzKohlrausch_effect_object_Nayatani1997(
                uv, uv_d65, L_a, method="VAC"
            ),
            result_vac,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_HelmholtzKohlrausch_effect_object_Nayatani1997(self):
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

    def test_HelmholtzKohlrausch_effect_luminous_Nayatani1997(self):
        """
        Test :func:`colour.appearance.hke.\
HelmholtzKohlrausch_effect_luminous_Nayatani1997` definition.
        """

        np.testing.assert_allclose(
            HelmholtzKohlrausch_effect_luminous_Nayatani1997(
                np.array([0.40351010, 0.53933673]),
                np.array([0.19783001, 0.46831999]),
                63.66,
                method="VCC",
            ),
            2.014433723774654,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            HelmholtzKohlrausch_effect_luminous_Nayatani1997(
                np.array([0.40351010, 0.53933673]),
                np.array([0.19783001, 0.46831999]),
                63.66,
                method="VAC",
            ),
            1.727991241148628,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_HelmholtzKohlrausch_effect_luminous_Nayatani1997(
        self,
    ):
        """
        Test :func:`colour.appearance.hke.\
HelmholtzKohlrausch_effect_luminous_Nayatani1997` definition n_dimensional
        arrays support.
        """

        uv_d65 = np.array([0.19783001, 0.46831999])
        uv = np.array([0.40351010, 0.53933673])
        L_a = 63.66

        result_vcc = HelmholtzKohlrausch_effect_luminous_Nayatani1997(
            uv, uv_d65, L_a, method="VCC"
        )
        result_vac = HelmholtzKohlrausch_effect_luminous_Nayatani1997(
            uv, uv_d65, L_a, method="VAC"
        )

        uv_d65 = np.tile(uv_d65, (6, 1))
        uv = np.tile(uv, (6, 1))
        result_vcc = np.tile(result_vcc, 6)
        result_vac = np.tile(result_vac, 6)

        np.testing.assert_allclose(
            HelmholtzKohlrausch_effect_luminous_Nayatani1997(
                uv, uv_d65, L_a, method="VCC"
            ),
            result_vcc,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            HelmholtzKohlrausch_effect_luminous_Nayatani1997(
                uv, uv_d65, L_a, method="VAC"
            ),
            result_vac,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        uv_d65 = np.reshape(uv_d65, (2, 3, 2))
        uv = np.reshape(uv, (2, 3, 2))
        result_vcc = np.reshape(result_vcc, (2, 3))
        result_vac = np.reshape(result_vac, (2, 3))

        np.testing.assert_allclose(
            HelmholtzKohlrausch_effect_luminous_Nayatani1997(
                uv, uv_d65, L_a, method="VCC"
            ),
            result_vcc,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            HelmholtzKohlrausch_effect_luminous_Nayatani1997(
                uv, uv_d65, L_a, method="VAC"
            ),
            result_vac,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_HelmholtzKohlrausch_effect_luminous_Nayatani1997(self):
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

    def test_coefficient_K_Br_Nayatani1997(self):
        """
        Test :func:`colour.appearance.hke.coefficient_K_Br_Nayatani1997`
        definition.
        """

        np.testing.assert_allclose(
            coefficient_K_Br_Nayatani1997(10.00000000),
            0.71344817765758839,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            coefficient_K_Br_Nayatani1997(63.66000000),
            1.000128455584031,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            coefficient_K_Br_Nayatani1997(1000.00000000),
            1.401080840298197,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            coefficient_K_Br_Nayatani1997(10000.00000000),
            1.592511806930447,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_coefficient_K_Br_Nayatani1997(self):
        """
        Test :func:`colour.appearance.hke.coefficient_K_Br_Nayatani1997`
        definition n_dimensional arrays support.
        """

        L_a = 63.66
        K_Br = coefficient_K_Br_Nayatani1997(L_a)

        L_a = np.tile(L_a, 6)
        K_Br = np.tile(K_Br, 6)
        np.testing.assert_allclose(
            coefficient_K_Br_Nayatani1997(L_a),
            K_Br,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        L_a = np.reshape(L_a, (2, 3))
        K_Br = np.reshape(K_Br, (2, 3))
        np.testing.assert_allclose(
            coefficient_K_Br_Nayatani1997(L_a),
            K_Br,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        L_a = np.reshape(L_a, (2, 3, 1))
        K_Br = np.reshape(K_Br, (2, 3, 1))
        np.testing.assert_allclose(
            coefficient_K_Br_Nayatani1997(L_a),
            K_Br,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_coefficient_K_Br_Nayatani1997(self):
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

    def test_coefficient_q_Nayatani1997(self):
        """
        Test :func:`colour.appearance.hke.coefficient_q_Nayatani1997`
        definition.
        """

        np.testing.assert_allclose(
            coefficient_q_Nayatani1997(0.00000000),
            -0.121200000000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            coefficient_q_Nayatani1997(0.78539816),
            0.125211117768464,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            coefficient_q_Nayatani1997(1.57079633),
            0.191679999416415,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            coefficient_q_Nayatani1997(2.35619449),
            0.028480866426611,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_coefficient_q_Nayatani1997(self):
        """
        Test :func:`colour.appearance.hke.coefficient_q_Nayatani1997`
        definition n_dimensional arrays support.
        """

        L_a = 63.66
        q = coefficient_q_Nayatani1997(L_a)

        L_a = np.tile(L_a, 6)
        q = np.tile(q, 6)
        np.testing.assert_allclose(
            coefficient_q_Nayatani1997(L_a), q, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        L_a = np.reshape(L_a, (2, 3))
        q = np.reshape(q, (2, 3))
        np.testing.assert_allclose(
            coefficient_q_Nayatani1997(L_a), q, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        L_a = np.reshape(L_a, (2, 3, 1))
        q = np.reshape(q, (2, 3, 1))
        np.testing.assert_allclose(
            coefficient_q_Nayatani1997(L_a), q, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    @ignore_numpy_errors
    def test_nan_coefficient_q_Nayatani1997(self):
        """
        Test :func:`colour.appearance.hke.coefficient_q_Nayatani1997`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        coefficient_q_Nayatani1997(cases)
