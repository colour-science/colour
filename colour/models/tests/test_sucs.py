"""
Define the unit tests for the :mod:`colour.models.sucs` module.
"""

from __future__ import annotations

from itertools import product

import numpy as np

from colour.models import XYZ_to_sUCS, sUCS_to_XYZ
from colour.utilities import domain_range_scale, ignore_numpy_errors

TOLERANCE_ABSOLUTE_TESTS = 1e-7

__author__ = "Colour Developers, UltraMo114(Molin Li)"
__copyright__ = "Copyright 2024 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestXYZ_to_sUCS",
    "TestSUCS_to_XYZ",
]


class TestXYZ_to_sUCS:
    """
    Define :func:`colour.models.sucs.XYZ_to_sUCS` definition unit
    tests methods.
    """

    def test_XYZ_to_sUCS(self) -> None:
        """
        Test :func:`colour.models.sucs.XYZ_to_sUCS` definition.
        Input XYZ values are D65-adapted and in [0, 1] range.
        """
        # Example 1
        xyz_in1 = np.array([0.20654008, 0.12197225, 0.05136952])
        sucs_expected1 = np.array(
            [42.629236534849696, 37.759976239968240, 14.422271284176796]
        )
        np.testing.assert_allclose(
            XYZ_to_sUCS(xyz_in1),
            sucs_expected1,
            atol=TOLERANCE_ABSOLUTE_TESTS,
            err_msg="Test Case 1 for XYZ_to_sUCS failed.",
        )

        # Example 2: D65 White
        xyz_in2 = np.array([0.95047, 1.00000, 1.08883])  # D65 Y=1
        sucs_expected2 = np.array(
            [99.999257497377670, 0.027913411036824, -9.039967686374366e-04]
        )
        np.testing.assert_allclose(
            XYZ_to_sUCS(xyz_in2),
            sucs_expected2,
            atol=TOLERANCE_ABSOLUTE_TESTS,
            err_msg="Test Case 2 (D65 White) for XYZ_to_sUCS failed.",
        )

    def test_n_dimensional_XYZ_to_sUCS(self) -> None:
        """
        Test :func:`colour.models.sucs.XYZ_to_sUCS` definition
        n-dimensional support.
        """
        xyz_single = np.array([0.20654008, 0.12197225, 0.05136952])
        sucs_single = XYZ_to_sUCS(xyz_single)

        xyz_nd = np.tile(xyz_single, (6, 1))
        sucs_nd_expected = np.tile(sucs_single, (6, 1))
        np.testing.assert_allclose(
            XYZ_to_sUCS(xyz_nd), sucs_nd_expected, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        xyz_nd_reshaped = np.reshape(xyz_nd, (2, 3, 3))
        sucs_nd_expected_reshaped = np.reshape(sucs_nd_expected, (2, 3, 3))
        np.testing.assert_allclose(
            XYZ_to_sUCS(xyz_nd_reshaped),
            sucs_nd_expected_reshaped,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_XYZ_to_sUCS(self) -> None:
        """
        Test :func:`colour.models.sucs.XYZ_to_sUCS` definition domain and
        range scale support.
        This test checks `func(input * factor)` vs `output_ref * factor_output`.
        The sUCS model does not currently implement internal scaling based on
        `domain_range_scale` context; scaling factors here reflect this.
        """
        xyz_ref = np.array([0.20654008, 0.12197225, 0.05136952])  # Input in [0,1]
        sucs_ref = XYZ_to_sUCS(xyz_ref)  # Output I_S is ~[0,100]

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    XYZ_to_sUCS(xyz_ref * factor),
                    sucs_ref * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_sUCS(self) -> None:
        """
        Test :func:`colour.models.sucs.XYZ_to_sUCS` definition nan
        support.
        """
        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        xyz_cases = np.array(list(set(product(cases, repeat=3))))
        XYZ_to_sUCS(xyz_cases)  # Expects NaNs to propagate or handle gracefully


class TestSUCS_to_XYZ:
    """
    Define :func:`colour.models.sucs.sUCS_to_XYZ` definition unit tests
    methods.
    """

    def test_sUCS_to_XYZ(self) -> None:
        """
        Test :func:`colour.models.sucs.sUCS_to_XYZ` definition.
        This is effectively a round-trip test.
        """
        # Round-trip for Example 1
        sucs_in1 = np.array([35.65885236, 22.10004031, 9.01985036])
        xyz_expected1 = np.array(
            [0.113190936179253, 0.084693604651780, 0.056091522673190]
        )
        np.testing.assert_allclose(
            sUCS_to_XYZ(sucs_in1),
            xyz_expected1,
            atol=TOLERANCE_ABSOLUTE_TESTS,
            # Higher atol might be needed for some round trips
            err_msg="Test Case 1 for sUCS_to_XYZ (round-trip) failed.",
        )

        # Round-trip for Example 2 (D65 White)
        sucs_in2 = np.array(
            [99.999257497377670, 0.027913411036824, -9.039967686374366e-04]
        )
        xyz_expected2 = np.array([0.95047, 1.00000, 1.08883])
        np.testing.assert_allclose(
            sUCS_to_XYZ(sucs_in2),
            xyz_expected2,
            atol=TOLERANCE_ABSOLUTE_TESTS,
            err_msg="Test Case 2 (D65 White round-trip) for sUCS_to_XYZ failed.",
        )

    def test_n_dimensional_sUCS_to_XYZ(self) -> None:
        """
        Test :func:`colour.models.sucs.sUCS_to_XYZ` definition
        n-dimensional support.
        """
        sucs_single = np.array([35.65885236, 22.10004031, 9.01985036])
        xyz_single = sUCS_to_XYZ(sucs_single)

        sucs_nd = np.tile(sucs_single, (6, 1))
        xyz_nd_expected = np.tile(xyz_single, (6, 1))
        np.testing.assert_allclose(
            sUCS_to_XYZ(sucs_nd), xyz_nd_expected, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        sucs_nd_reshaped = np.reshape(sucs_nd, (2, 3, 3))
        xyz_nd_expected_reshaped = np.reshape(xyz_nd_expected, (2, 3, 3))
        np.testing.assert_allclose(
            sUCS_to_XYZ(sucs_nd_reshaped),
            xyz_nd_expected_reshaped,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_sUCS_to_XYZ(self) -> None:
        """
        Test :func:`colour.models.sucs.sUCS_to_XYZ` definition domain and
        range scale support.
        sUCS_to_XYZ, like XYZ_to_sUCS, doesn't internally adjust scale based on
        `domain_range_scale` context. This test checks `func(input * factor)`
        vs `output_ref * factor_output`.
        """
        sucs_ref = np.array([35.65885236, 22.10004031, 9.01985036])  # I_S ~0-100
        xyz_ref = sUCS_to_XYZ(sucs_ref)  # Output XYZ is ~[0,1]

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    sUCS_to_XYZ(sucs_ref * factor),
                    xyz_ref * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_sUCS_to_XYZ(self) -> None:
        """
        Test :func:`colour.models.sucs.sUCS_to_XYZ` definition nan
        support.
        """
        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        sucs_cases = np.array(list(set(product(cases, repeat=3))))
        sUCS_to_XYZ(sucs_cases)  # Expects NaNs to propagate or handle gracefully
