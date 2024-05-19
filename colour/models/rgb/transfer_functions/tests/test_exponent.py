"""
Define the unit tests for the
:mod:`colour.models.rgb.transfer_functions.exponent` module.
"""


import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.transfer_functions import (
    exponent_function_basic,
    exponent_function_monitor_curve,
)
from colour.utilities import ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestExponentFunctionBasic",
    "TestExponentFunctionMonitorCurve",
]


class TestExponentFunctionBasic:
    """
    Define :func:`colour.models.rgb.transfer_functions.exponent.\
exponent_function_basic` definition unit tests methods.
    """

    def test_exponent_function_basic(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.exponent.\
exponent_function_basic` definition.
        """

        a = 0.18
        a_p = 0.0229932049927
        np.testing.assert_allclose(
            exponent_function_basic(a, 2.2), a_p, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        np.testing.assert_allclose(
            exponent_function_basic(a, 2.2, "basicMirrorFwd"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            exponent_function_basic(a, 2.2, "basicPassThruFwd"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = 0.0229932049927
        a_p = 0.18
        np.testing.assert_allclose(
            exponent_function_basic(a, 2.2, "basicRev"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            exponent_function_basic(a, 2.2, "basicMirrorRev"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            exponent_function_basic(a, 2.2, "basicPassThruRev"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = -0.18
        np.testing.assert_allclose(
            exponent_function_basic(a, 2.2), 0.0, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        np.testing.assert_allclose(
            exponent_function_basic(a, 2.2, "basicMirrorFwd"),
            -0.0229932049927,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            exponent_function_basic(a, 2.2, "basicPassThruFwd"),
            -0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = -0.0229932049927
        np.testing.assert_allclose(
            exponent_function_basic(a, 2.2, "basicRev"),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            exponent_function_basic(a, 2.2, "basicMirrorRev"),
            -0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            exponent_function_basic(a, 2.2, "basicPassThruRev"),
            -0.0229932049927,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_exponent_function_basic(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.exponent.\
exponent_function_basic` definition n-dimensional arrays support.
        """

        a = 0.18
        a_p = 0.0229932049927

        a = np.tile(a, 6)
        a_p = np.tile(a_p, 6)
        np.testing.assert_allclose(
            exponent_function_basic(a, 2.2), a_p, atol=TOLERANCE_ABSOLUTE_TESTS
        )
        np.testing.assert_allclose(
            exponent_function_basic(a, 2.2, "basicMirrorFwd"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        np.testing.assert_allclose(
            exponent_function_basic(a, 2.2, "basicPassThruFwd"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = np.reshape(a, (2, 3))
        a_p = np.reshape(a_p, (2, 3))
        np.testing.assert_allclose(
            exponent_function_basic(a, 2.2), a_p, atol=TOLERANCE_ABSOLUTE_TESTS
        )
        np.testing.assert_allclose(
            exponent_function_basic(a, 2.2, "basicMirrorFwd"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        np.testing.assert_allclose(
            exponent_function_basic(a, 2.2, "basicPassThruFwd"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = np.reshape(a, (2, 3, 1))
        a_p = np.reshape(a_p, (2, 3, 1))
        np.testing.assert_allclose(
            exponent_function_basic(a, 2.2), a_p, atol=TOLERANCE_ABSOLUTE_TESTS
        )
        np.testing.assert_allclose(
            exponent_function_basic(a, 2.2, "basicMirrorFwd"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        np.testing.assert_allclose(
            exponent_function_basic(a, 2.2, "basicPassThruFwd"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = 0.0229932049927
        a_p = 0.18

        a = np.tile(a, 6)
        a_p = np.tile(a_p, 6)
        np.testing.assert_allclose(
            exponent_function_basic(a, 2.2, "basicRev"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        np.testing.assert_allclose(
            exponent_function_basic(a, 2.2, "basicMirrorRev"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        np.testing.assert_allclose(
            exponent_function_basic(a, 2.2, "basicPassThruRev"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = np.reshape(a, (2, 3))
        a_p = np.reshape(a_p, (2, 3))
        np.testing.assert_allclose(
            exponent_function_basic(a, 2.2, "basicRev"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        np.testing.assert_allclose(
            exponent_function_basic(a, 2.2, "basicMirrorRev"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        np.testing.assert_allclose(
            exponent_function_basic(a, 2.2, "basicPassThruRev"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = np.reshape(a, (2, 3, 1))
        a_p = np.reshape(a_p, (2, 3, 1))
        np.testing.assert_allclose(
            exponent_function_basic(a, 2.2, "basicRev"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        np.testing.assert_allclose(
            exponent_function_basic(a, 2.2, "basicMirrorRev"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        np.testing.assert_allclose(
            exponent_function_basic(a, 2.2, "basicPassThruRev"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_exponent_function_basic(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.exponent.\
exponent_function_basic` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        for case in cases:
            exponent_function_basic(case, case)


class TestExponentFunctionMonitorCurve:
    """
    Define :func:`colour.models.rgb.transfer_functions.exponent.\
exponent_function_monitor_curve` definition unit tests methods.
    """

    def test_exponent_function_monitor_curve(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.exponent.\
exponent_function_monitor_curve` definition.
        """

        a = 0.18
        a_p = 0.0232240466001
        np.testing.assert_allclose(
            exponent_function_monitor_curve(a, 2.2, 0.001),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            exponent_function_monitor_curve(a, 2.2, 0.001, "monCurveMirrorFwd"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = 0.0232240466001
        a_p = 0.18
        np.testing.assert_allclose(
            exponent_function_monitor_curve(a, 2.2, 0.001, "monCurveRev"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            exponent_function_monitor_curve(a, 2.2, 0.001, "monCurveMirrorRev"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = -0.18
        np.testing.assert_allclose(
            exponent_function_monitor_curve(a, 2.2, 0.001),
            -0.000205413951,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            exponent_function_monitor_curve(a, 2.2, 0.001, "monCurveMirrorFwd"),
            -0.0232240466001,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = -0.000205413951
        np.testing.assert_allclose(
            exponent_function_monitor_curve(a, 2.2, 0.001, "monCurveRev"),
            -0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            exponent_function_monitor_curve(a, 2.2, 0.001, "monCurveMirrorRev"),
            -0.0201036111565,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_exponent_function_monitor_curve(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.exponent.\
exponent_function_monitor_curve` definition n-dimensional arrays support.
        """

        a = 0.18
        a_p = 0.0232240466001

        a = np.tile(a, 6)
        a_p = np.tile(a_p, 6)
        np.testing.assert_allclose(
            exponent_function_monitor_curve(a, 2.2, 0.001),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        np.testing.assert_allclose(
            exponent_function_monitor_curve(a, 2.2, 0.001, "monCurveMirrorFwd"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = np.reshape(a, (2, 3))
        a_p = np.reshape(a_p, (2, 3))
        np.testing.assert_allclose(
            exponent_function_monitor_curve(a, 2.2, 0.001),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        np.testing.assert_allclose(
            exponent_function_monitor_curve(a, 2.2, 0.001, "monCurveMirrorFwd"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = np.reshape(a, (2, 3, 1))
        a_p = np.reshape(a_p, (2, 3, 1))
        np.testing.assert_allclose(
            exponent_function_monitor_curve(a, 2.2, 0.001),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        np.testing.assert_allclose(
            exponent_function_monitor_curve(a, 2.2, 0.001, "monCurveMirrorFwd"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = 0.0232240466001
        a_p = 0.18

        a = np.tile(a, 6)
        a_p = np.tile(a_p, 6)
        np.testing.assert_allclose(
            exponent_function_monitor_curve(a, 2.2, 0.001, "monCurveRev"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        np.testing.assert_allclose(
            exponent_function_monitor_curve(a, 2.2, 0.001, "monCurveMirrorRev"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = np.reshape(a, (2, 3))
        a_p = np.reshape(a_p, (2, 3))
        np.testing.assert_allclose(
            exponent_function_monitor_curve(a, 2.2, 0.001, "monCurveRev"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        np.testing.assert_allclose(
            exponent_function_monitor_curve(a, 2.2, 0.001, "monCurveMirrorRev"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = np.reshape(a, (2, 3, 1))
        a_p = np.reshape(a_p, (2, 3, 1))
        np.testing.assert_allclose(
            exponent_function_monitor_curve(a, 2.2, 0.001, "monCurveRev"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        np.testing.assert_allclose(
            exponent_function_monitor_curve(a, 2.2, 0.001, "monCurveMirrorRev"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_exponent_function_monitor_curve(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.exponent.\
exponent_function_monitor_curve` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        for case in cases:
            exponent_function_monitor_curve(case, case, case)
