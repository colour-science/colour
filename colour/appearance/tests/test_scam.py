"""
Define the unit tests for the :mod:`colour.appearance.scam` module.
"""

from __future__ import annotations

from dataclasses import astuple  # For converting spec object to tuple
from itertools import product

import numpy as np
import pytest

from colour.appearance import (
    CAM_Specification_sCAM,
    VIEWING_CONDITIONS_sCAM,
    XYZ_to_sCAM,
    sCAM_to_XYZ,
)
from colour.utilities import domain_range_scale, ignore_numpy_errors

# This constant is typically defined in colour.constants
TOLERANCE_ABSOLUTE_TESTS = 1e-7  # Define a suitable tolerance

__author__ = "Colour Developers, UltraMo114(Molin Li)"
__copyright__ = "Copyright 2024 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"


class TestXYZ_to_sCAM:
    """
    Define :func:`colour.appearance.scam.XYZ_to_sCAM` definition unit
    tests methods.
    """

    def test_XYZ_to_sCAM(self) -> None:
        """
        Test :func:`colour.appearance.scam.XYZ_to_sCAM` definition.
        Expected array should contain [J, C, h, Q, M, H] in that order.
        """
        # Test Case 1:
        XYZ_input1 = np.array([19.01, 20.00, 21.78])
        XYZ_w1 = np.array([95.047, 100.00, 108.883])  # D65
        # Adapting luminance (e.g., 0.2 * L_W_D65 or typical screen L_A)
        L_A1 = 318.31
        Y_b1 = 20.0  # Background luminance factor
        surround1 = VIEWING_CONDITIONS_sCAM["Average"]
        # Expected [J, C, h, Q, M, H] for XYZ_input1 (Placeholder values)
        expected_output1 = np.array(
            [
                49.979698822617280,
                0.016657276820862,
                3.379328608949343e02,
                2.064281280216374e02,
                0.005896530843538,
                3.703962295562696e02,
            ]
        )

        computed_spec1 = XYZ_to_sCAM(XYZ_input1, XYZ_w1, L_A1, Y_b1, surround1)
        np.testing.assert_allclose(
            astuple(computed_spec1),
            expected_output1,
            atol=TOLERANCE_ABSOLUTE_TESTS,
            err_msg="Test Case 1 failed for XYZ_to_sCAM",
        )

        # Test Case 2:
        XYZ_input2 = np.array([57.06, 43.06, 31.96])
        XYZ_w2 = np.array([95.05, 100.00, 108.88])
        L_A2 = 60.0  # Different L_A
        Y_b2 = 18.0  # Different Y_b
        surround2 = VIEWING_CONDITIONS_sCAM["Dim"]
        # Expected [J, C, h, Q, M, H] for XYZ_input2 (Placeholder values)
        expected_output2 = np.array(
            [
                72.833650016596580,
                37.338497730766390,
                18.751345616609278,
                2.422256530300396e02,
                10.303798227523222,
                2.911665627688864,
            ]
        )

        computed_spec2 = XYZ_to_sCAM(XYZ_input2, XYZ_w2, L_A2, Y_b2, surround2)
        np.testing.assert_allclose(
            astuple(computed_spec2),
            expected_output2,
            atol=TOLERANCE_ABSOLUTE_TESTS,
            err_msg="Test Case 2 failed for XYZ_to_sCAM",
        )

    def test_n_dimensional_XYZ_to_sCAM(self) -> None:
        """
        Test :func:`colour.appearance.scam.XYZ_to_sCAM` definition
        n-dimensional support.
        """
        XYZ = np.array([19.01, 20.00, 21.78])
        XYZ_w = np.array([95.047, 100.00, 108.883])
        L_A = 318.31
        Y_b = 20.0
        surround = VIEWING_CONDITIONS_sCAM["Average"]

        spec_single_obj = XYZ_to_sCAM(XYZ, XYZ_w, L_A, Y_b, surround)
        spec_single_arr = np.array(astuple(spec_single_obj))

        XYZ_nd = np.tile(XYZ, (6, 1))
        spec_nd_expected = np.tile(spec_single_arr, (6, 1))
        computed_spec_nd_obj = XYZ_to_sCAM(XYZ_nd, XYZ_w, L_A, Y_b, surround)

        # Handle case where XYZ_to_sCAM might return a list of specs
        # or a spec object with array fields.
        if isinstance(computed_spec_nd_obj, list) and all(
            isinstance(s, CAM_Specification_sCAM) for s in computed_spec_nd_obj
        ):
            computed_values = np.array([astuple(s) for s in computed_spec_nd_obj])
        elif isinstance(computed_spec_nd_obj, CAM_Specification_sCAM):
            computed_values = np.vstack(
                astuple(computed_spec_nd_obj)
            ).T  # astuple gives tuple of arrays, stack and transpose
        else:
            pytest.fail(
                "Unexpected output type from XYZ_to_sCAM for n-dimensional input"
            )

        np.testing.assert_allclose(
            computed_values,
            spec_nd_expected,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_domain_range_scale_XYZ_to_sCAM(self) -> None:
        """
        Test :func:`colour.appearance.scam.XYZ_to_sCAM` definition
        domain and range scale support.
        NOTE: This test assumes XYZ_to_sCAM's output specification values
        will be scaled according to the active `domain_range_scale`.
        """
        XYZ = np.array([19.01, 20.00, 21.78])
        XYZ_w = np.array([95.05, 100.00, 108.88])
        L_A = 318.31
        Y_b = 20.0
        surround = VIEWING_CONDITIONS_sCAM["Average"]

        specification_ref_obj = XYZ_to_sCAM(XYZ, XYZ_w, L_A, Y_b, surround)
        specification_ref_arr = np.array(astuple(specification_ref_obj))

        # Define scaling factors for sCAM outputs [J, C, h, Q, M, H]
        # Factors assume output J,C,Q,M are 0-100, h 0-360, H 0-400 at ref scale.
        # For "1" scale, outputs are normalized.
        domain_range_definitions = (
            (
                "reference",
                1.0,
                np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            ),
            (
                "1",  # Input XYZ scaled by 0.01 (0-1 range)
                0.01,
                np.array(
                    [
                        1.0 / 100.0,  # J
                        1.0 / 100.0,  # C
                        1.0 / 360.0,  # h
                        1.0 / 100.0,  # Q
                        1.0 / 100.0,  # M
                        1.0 / 400.0,  # H
                    ]
                ),
            ),
        )

        for scale, factor_input_XYZ, factor_output_spec in domain_range_definitions:
            with domain_range_scale(scale):
                computed_spec_obj_scaled = XYZ_to_sCAM(
                    XYZ * factor_input_XYZ,
                    XYZ_w * factor_input_XYZ,
                    L_A,
                    Y_b,
                    surround,
                )
                computed_spec_arr_scaled = np.array(astuple(computed_spec_obj_scaled))
                expected_spec_scaled = specification_ref_arr * factor_output_spec

                np.testing.assert_allclose(
                    computed_spec_arr_scaled,
                    expected_spec_scaled,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                    err_msg=f"Domain_range_scale '{scale}' failed for XYZ_to_sCAM",
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_sCAM(self) -> None:
        """
        Test :func:`colour.appearance.scam.XYZ_to_sCAM` definition
        nan support.
        """
        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        test_XYZ = np.array(list(product(cases, repeat=3)))
        test_XYZ_w = np.array(list(product(cases, repeat=3)))
        L_A_nan_test = cases[0]  # a scalar
        Y_b_nan_test = cases[0]  # a scalar
        surround_nan_test = VIEWING_CONDITIONS_sCAM["Average"]

        XYZ_to_sCAM(
            test_XYZ,
            test_XYZ_w[
                0 : len(test_XYZ)
            ],  # Ensure XYZ_w matches test_XYZ length if different
            L_A_nan_test,
            Y_b_nan_test,
            surround_nan_test,
        )


class TestSCAM_to_XYZ:
    """
    Define :func:`colour.appearance.scam.sCAM_to_XYZ` definition unit
    tests methods.
    """

    def test_sCAM_to_XYZ(self) -> None:
        """
        Test :func:`colour.appearance.scam.sCAM_to_XYZ` definition.
        This is effectively a round-trip test using known pairs.
        """
        # Test Case 1: (Placeholder values)
        input_spec1_vals = CAM_Specification_sCAM(
            J=40.0,
            C=20.0,
            h=210.0,
            Q=150.0,
            M=30.0,
            H=250.0,
        )
        XYZ_w1 = np.array([95.05, 100.00, 108.88])
        L_A1 = 318.31
        Y_b1 = 20.0
        surround1 = VIEWING_CONDITIONS_sCAM["Average"]
        expected_XYZ1 = np.array(
            [9.407457878462013, 12.552327827760026, 19.761320558229954]
        )

        computed_XYZ1 = sCAM_to_XYZ(input_spec1_vals, XYZ_w1, L_A1, Y_b1, surround1)
        np.testing.assert_allclose(
            computed_XYZ1,
            expected_XYZ1,
            atol=TOLERANCE_ABSOLUTE_TESTS,
            # May need a slightly larger tolerance for round-trip
            err_msg="Test Case 1 failed for sCAM_to_XYZ (round-trip)",
        )

        # Test Case 2: Using M instead of C (Placeholder values)
        input_spec2_vals = CAM_Specification_sCAM(
            J=50.0,  # Corrected J to be float
            M=10.0,  # Corrected M to be float
            h=300.0,  # Corrected h to be float
            Q=2.064281280216374e02,
            H=3.703962295562696e02,  # C is None, M is provided
        )
        XYZ_w2 = np.array([95.047, 100.00, 108.883])
        L_A2 = 318.31
        Y_b2 = 20.0  # Corrected Y_b to be float
        surround2 = VIEWING_CONDITIONS_sCAM["Dim"]
        expected_XYZ2 = np.array(
            [24.220686694763213, 18.002320848318930, 46.269543558828445]
        )

        computed_XYZ2 = sCAM_to_XYZ(input_spec2_vals, XYZ_w2, L_A2, Y_b2, surround2)
        np.testing.assert_allclose(
            computed_XYZ2,
            expected_XYZ2,
            atol=TOLERANCE_ABSOLUTE_TESTS,
            err_msg="Test Case 2 (M input) failed for sCAM_to_XYZ",
        )

    def test_n_dimensional_sCAM_to_XYZ(self) -> None:
        """
        Test :func:`colour.appearance.scam.sCAM_to_XYZ` definition
        n-dimensional support.
        """
        spec_J = 40.0
        spec_C = 20.0
        spec_h = 210.0
        spec_Q = 150.0
        spec_M = 30.0
        spec_H = 250.0
        single_spec_obj = CAM_Specification_sCAM(
            J=spec_J, C=spec_C, h=spec_h, Q=spec_Q, M=spec_M, H=spec_H
        )

        XYZ_w = np.array([95.05, 100.00, 108.88])
        L_A = 318.31
        Y_b = 20.0
        surround = VIEWING_CONDITIONS_sCAM["Average"]

        expected_XYZ_single = sCAM_to_XYZ(single_spec_obj, XYZ_w, L_A, Y_b, surround)

        spec_values_tiled = np.tile(
            np.array([spec_J, spec_C, spec_h, spec_Q, spec_M, spec_H]), (6, 1)
        )

        nd_spec_obj = CAM_Specification_sCAM(
            J=spec_values_tiled[:, 0],
            C=spec_values_tiled[:, 1],
            h=spec_values_tiled[:, 2],
            Q=spec_values_tiled[:, 3],
            M=spec_values_tiled[:, 4],
            H=spec_values_tiled[:, 5],
        )

        expected_XYZ_nd = np.tile(expected_XYZ_single, (6, 1))
        computed_XYZ_nd = sCAM_to_XYZ(nd_spec_obj, XYZ_w, L_A, Y_b, surround)

        np.testing.assert_allclose(
            computed_XYZ_nd,
            expected_XYZ_nd,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_domain_range_scale_sCAM_to_XYZ(self) -> None:
        """
        Test :func:`colour.appearance.scam.sCAM_to_XYZ` definition
        domain and range scale support.
        """
        spec_ref_obj = CAM_Specification_sCAM(
            J=40.0, C=20.0, h=210.0, Q=150.0, M=30.0, H=250.0
        )
        XYZ_w_ref = np.array([95.05, 100.00, 108.88])
        L_A_ref = 318.31
        Y_b_ref = 20.0
        surround_ref = VIEWING_CONDITIONS_sCAM["Average"]
        XYZ_ref = sCAM_to_XYZ(spec_ref_obj, XYZ_w_ref, L_A_ref, Y_b_ref, surround_ref)

        domain_range_definitions = (
            ("reference", np.array([1.0] * 6), 1.0),
            (
                "1",
                np.array(
                    [
                        1.0 / 100.0,  # J
                        1.0 / 100.0,  # C
                        1.0 / 360.0,  # h
                        1.0 / 100.0,  # Q
                        1.0 / 100.0,  # M
                        1.0 / 400.0,  # H
                    ]
                ),
                0.01,
            ),
        )

        spec_ref_arr = np.array(astuple(spec_ref_obj))

        for scale, factor_input_spec, factor_output_XYZ in domain_range_definitions:
            with domain_range_scale(scale):
                scaled_spec_arr = spec_ref_arr * factor_input_spec
                scaled_spec_obj = CAM_Specification_sCAM(*scaled_spec_arr)
                XYZ_w_scaled = XYZ_w_ref * factor_output_XYZ

                computed_XYZ_scaled = sCAM_to_XYZ(
                    scaled_spec_obj, XYZ_w_scaled, L_A_ref, Y_b_ref, surround_ref
                )
                expected_XYZ_scaled = XYZ_ref * factor_output_XYZ

                np.testing.assert_allclose(
                    computed_XYZ_scaled,
                    expected_XYZ_scaled,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                    err_msg=f"Domain_range_scale '{scale}' failed for sCAM_to_XYZ",
                )

    @ignore_numpy_errors
    def test_raise_exception_sCAM_to_XYZ(self) -> None:
        """
        Test :func:`colour.appearance.scam.sCAM_to_XYZ` definition
        raised exception for missing critical inputs.
        """
        XYZ_w = np.array([95.05, 100.00, 108.88])
        L_A = 318.31
        Y_b = 20.0
        surround = VIEWING_CONDITIONS_sCAM["Average"]

        with pytest.raises(ValueError):
            sCAM_to_XYZ(
                CAM_Specification_sCAM(J=None, C=20.0, h=210.0),
                XYZ_w,
                L_A,
                Y_b,
                surround,
            )

        with pytest.raises(ValueError):
            sCAM_to_XYZ(
                CAM_Specification_sCAM(J=40.0, C=20.0, h=None),
                XYZ_w,
                L_A,
                Y_b,
                surround,
            )

        with pytest.raises(ValueError):
            sCAM_to_XYZ(
                CAM_Specification_sCAM(J=40.0, C=None, h=210.0, M=None),
                XYZ_w,
                L_A,
                Y_b,
                surround,
            )

    @ignore_numpy_errors
    def test_nan_sCAM_to_XYZ(self) -> None:
        """
        Test :func:`colour.appearance.scam.sCAM_to_XYZ` definition nan
        support.
        """
        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        scalar_case = cases[0]

        spec_with_nans = CAM_Specification_sCAM(
            J=scalar_case,
            C=scalar_case,
            h=scalar_case,
            Q=np.nan,
            M=scalar_case,
            H=np.nan,
        )

        XYZ_w_valid = np.array([95.05, 100.00, 108.88])
        L_A_valid = 300.0
        Y_b_valid = 20.0
        surround_valid = VIEWING_CONDITIONS_sCAM["Average"]

        sCAM_to_XYZ(spec_with_nans, XYZ_w_valid, L_A_valid, Y_b_valid, surround_valid)

        valid_spec = CAM_Specification_sCAM(
            J=40.0, C=20.0, h=210.0, Q=150.0, M=30.0, H=250.0
        )
        XYZ_w_nan = np.array([np.nan, scalar_case, scalar_case])
        L_A_nan = np.nan
        Y_b_nan = np.nan

        sCAM_to_XYZ(valid_spec, XYZ_w_nan, L_A_nan, Y_b_nan, surround_valid)
