"""
Define the unit tests for the
:mod:`colour.models.rgb.transfer_functions.exponent` module.
"""

from __future__ import annotations

import typing

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.transfer_functions import (
    exponent_function_basic,
    exponent_function_monitor_curve,
)
from colour.utilities import (
    ignore_numpy_errors,
    xp_as_array,
    xp_assert_close,
    xp_reshape,
)

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

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

    def test_exponent_function_basic(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.exponent.\
exponent_function_basic` definition.
        """

        a = 0.18
        a_p = 0.0229932049927
        xp_assert_close(
            exponent_function_basic(xp_as_array(a, xp=xp), 2.2),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            exponent_function_basic(xp_as_array(a, xp=xp), 2.2, "basicMirrorFwd"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            exponent_function_basic(xp_as_array(a, xp=xp), 2.2, "basicPassThruFwd"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = 0.0229932049927
        a_p = 0.18
        xp_assert_close(
            exponent_function_basic(xp_as_array(a, xp=xp), 2.2, "basicRev"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            exponent_function_basic(xp_as_array(a, xp=xp), 2.2, "basicMirrorRev"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            exponent_function_basic(xp_as_array(a, xp=xp), 2.2, "basicPassThruRev"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = -0.18
        xp_assert_close(
            exponent_function_basic(xp_as_array(a, xp=xp), 2.2),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            exponent_function_basic(xp_as_array(a, xp=xp), 2.2, "basicMirrorFwd"),
            -0.0229932049927,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            exponent_function_basic(xp_as_array(a, xp=xp), 2.2, "basicPassThruFwd"),
            -0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = -0.0229932049927
        xp_assert_close(
            exponent_function_basic(xp_as_array(a, xp=xp), 2.2, "basicRev"),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            exponent_function_basic(xp_as_array(a, xp=xp), 2.2, "basicMirrorRev"),
            -0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            exponent_function_basic(xp_as_array(a, xp=xp), 2.2, "basicPassThruRev"),
            -0.0229932049927,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_exponent_function_basic(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.exponent.\
exponent_function_basic` definition n-dimensional arrays support.
        """

        a = 0.18
        a_p = 0.0229932049927

        a = xp.tile(xp_as_array(a, xp=xp), (6,))
        a_p = xp.tile(xp_as_array(a_p, xp=xp), (6,))
        xp_assert_close(
            exponent_function_basic(a, 2.2),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            exponent_function_basic(a, 2.2, "basicMirrorFwd"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            exponent_function_basic(a, 2.2, "basicPassThruFwd"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = xp_reshape(xp_as_array(a, xp=xp), (2, 3), xp=xp)
        a_p = xp_reshape(xp_as_array(a_p, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            exponent_function_basic(a, 2.2),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            exponent_function_basic(a, 2.2, "basicMirrorFwd"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            exponent_function_basic(a, 2.2, "basicPassThruFwd"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = xp_reshape(xp_as_array(a, xp=xp), (2, 3, 1), xp=xp)
        a_p = xp_reshape(xp_as_array(a_p, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            exponent_function_basic(a, 2.2),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            exponent_function_basic(a, 2.2, "basicMirrorFwd"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            exponent_function_basic(a, 2.2, "basicPassThruFwd"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = 0.0229932049927
        a_p = 0.18

        a = xp.tile(xp_as_array(a, xp=xp), (6,))
        a_p = xp.tile(xp_as_array(a_p, xp=xp), (6,))
        xp_assert_close(
            exponent_function_basic(a, 2.2, "basicRev"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            exponent_function_basic(a, 2.2, "basicMirrorRev"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            exponent_function_basic(a, 2.2, "basicPassThruRev"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = xp_reshape(xp_as_array(a, xp=xp), (2, 3), xp=xp)
        a_p = xp_reshape(xp_as_array(a_p, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            exponent_function_basic(a, 2.2, "basicRev"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            exponent_function_basic(a, 2.2, "basicMirrorRev"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            exponent_function_basic(a, 2.2, "basicPassThruRev"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = xp_reshape(xp_as_array(a, xp=xp), (2, 3, 1), xp=xp)
        a_p = xp_reshape(xp_as_array(a_p, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            exponent_function_basic(a, 2.2, "basicRev"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            exponent_function_basic(a, 2.2, "basicMirrorRev"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            exponent_function_basic(a, 2.2, "basicPassThruRev"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_exponent_function_basic(self) -> None:
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

    def test_exponent_function_monitor_curve(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.exponent.\
exponent_function_monitor_curve` definition.
        """

        a = 0.18
        a_p = 0.0232240466001
        xp_assert_close(
            exponent_function_monitor_curve(xp_as_array(a, xp=xp), 2.2, 0.001),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            exponent_function_monitor_curve(
                xp_as_array(a, xp=xp), 2.2, 0.001, "monCurveMirrorFwd"
            ),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = 0.0232240466001
        a_p = 0.18
        xp_assert_close(
            exponent_function_monitor_curve(
                xp_as_array(a, xp=xp), 2.2, 0.001, "monCurveRev"
            ),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            exponent_function_monitor_curve(
                xp_as_array(a, xp=xp), 2.2, 0.001, "monCurveMirrorRev"
            ),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = -0.18
        xp_assert_close(
            exponent_function_monitor_curve(xp_as_array(a, xp=xp), 2.2, 0.001),
            -0.000205413951,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            exponent_function_monitor_curve(
                xp_as_array(a, xp=xp), 2.2, 0.001, "monCurveMirrorFwd"
            ),
            -0.0232240466001,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = -0.000205413951
        xp_assert_close(
            exponent_function_monitor_curve(
                xp_as_array(a, xp=xp), 2.2, 0.001, "monCurveRev"
            ),
            -0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            exponent_function_monitor_curve(
                xp_as_array(a, xp=xp), 2.2, 0.001, "monCurveMirrorRev"
            ),
            -0.0201036111565,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_exponent_function_monitor_curve(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.exponent.\
exponent_function_monitor_curve` definition n-dimensional arrays support.
        """

        a = 0.18
        a_p = 0.0232240466001

        a = xp.tile(xp_as_array(a, xp=xp), (6,))
        a_p = xp.tile(xp_as_array(a_p, xp=xp), (6,))
        xp_assert_close(
            exponent_function_monitor_curve(a, 2.2, 0.001),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            exponent_function_monitor_curve(a, 2.2, 0.001, "monCurveMirrorFwd"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = xp_reshape(xp_as_array(a, xp=xp), (2, 3), xp=xp)
        a_p = xp_reshape(xp_as_array(a_p, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            exponent_function_monitor_curve(a, 2.2, 0.001),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            exponent_function_monitor_curve(a, 2.2, 0.001, "monCurveMirrorFwd"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = xp_reshape(xp_as_array(a, xp=xp), (2, 3, 1), xp=xp)
        a_p = xp_reshape(xp_as_array(a_p, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            exponent_function_monitor_curve(a, 2.2, 0.001),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            exponent_function_monitor_curve(a, 2.2, 0.001, "monCurveMirrorFwd"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = 0.0232240466001
        a_p = 0.18

        a = xp.tile(xp_as_array(a, xp=xp), (6,))
        a_p = xp.tile(xp_as_array(a_p, xp=xp), (6,))
        xp_assert_close(
            exponent_function_monitor_curve(a, 2.2, 0.001, "monCurveRev"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            exponent_function_monitor_curve(a, 2.2, 0.001, "monCurveMirrorRev"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = xp_reshape(xp_as_array(a, xp=xp), (2, 3), xp=xp)
        a_p = xp_reshape(xp_as_array(a_p, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            exponent_function_monitor_curve(a, 2.2, 0.001, "monCurveRev"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            exponent_function_monitor_curve(a, 2.2, 0.001, "monCurveMirrorRev"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = xp_reshape(xp_as_array(a, xp=xp), (2, 3, 1), xp=xp)
        a_p = xp_reshape(xp_as_array(a_p, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            exponent_function_monitor_curve(a, 2.2, 0.001, "monCurveRev"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            exponent_function_monitor_curve(a, 2.2, 0.001, "monCurveMirrorRev"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_exponent_function_monitor_curve(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.exponent.\
exponent_function_monitor_curve` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        for case in cases:
            exponent_function_monitor_curve(case, case, case)
