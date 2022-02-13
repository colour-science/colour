"""
Defines the unit tests for the
:mod:`colour.models.rgb.transfer_functions.exponent` module.
"""

import numpy as np
import unittest

from colour.models.rgb.transfer_functions import (
    exponent_function_basic,
    exponent_function_monitor_curve,
)
from colour.utilities import ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestExponentFunctionBasic",
    "TestExponentFunctionMonitorCurve",
]


class TestExponentFunctionBasic(unittest.TestCase):
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
        self.assertAlmostEqual(exponent_function_basic(a, 2.2), a_p, places=7)

        self.assertAlmostEqual(
            exponent_function_basic(a, 2.2, "basicMirrorFwd"), a_p, places=7
        )

        self.assertAlmostEqual(
            exponent_function_basic(a, 2.2, "basicPassThruFwd"), a_p, places=7
        )

        a = 0.0229932049927
        a_p = 0.18
        self.assertAlmostEqual(
            exponent_function_basic(a, 2.2, "basicRev"), a_p, places=7
        )

        self.assertAlmostEqual(
            exponent_function_basic(a, 2.2, "basicMirrorRev"), a_p, places=7
        )

        self.assertAlmostEqual(
            exponent_function_basic(a, 2.2, "basicPassThruRev"), a_p, places=7
        )

        a = -0.18
        self.assertAlmostEqual(exponent_function_basic(a, 2.2), 0.0, places=7)

        self.assertAlmostEqual(
            exponent_function_basic(a, 2.2, "basicMirrorFwd"),
            -0.0229932049927,
            places=7,
        )

        self.assertAlmostEqual(
            exponent_function_basic(a, 2.2, "basicPassThruFwd"),
            -0.18,
            places=7,
        )

        a = -0.0229932049927
        self.assertAlmostEqual(
            exponent_function_basic(a, 2.2, "basicRev"), 0.0, places=7
        )

        self.assertAlmostEqual(
            exponent_function_basic(a, 2.2, "basicMirrorRev"), -0.18, places=7
        )

        self.assertAlmostEqual(
            exponent_function_basic(a, 2.2, "basicPassThruRev"),
            -0.0229932049927,
            places=7,
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
        np.testing.assert_almost_equal(
            exponent_function_basic(a, 2.2), a_p, decimal=7
        )
        np.testing.assert_almost_equal(
            exponent_function_basic(a, 2.2, "basicMirrorFwd"), a_p, decimal=7
        )
        np.testing.assert_almost_equal(
            exponent_function_basic(a, 2.2, "basicPassThruFwd"), a_p, decimal=7
        )

        a = np.reshape(a, (2, 3))
        a_p = np.reshape(a_p, (2, 3))
        np.testing.assert_almost_equal(
            exponent_function_basic(a, 2.2), a_p, decimal=7
        )
        np.testing.assert_almost_equal(
            exponent_function_basic(a, 2.2, "basicMirrorFwd"), a_p, decimal=7
        )
        np.testing.assert_almost_equal(
            exponent_function_basic(a, 2.2, "basicPassThruFwd"), a_p, decimal=7
        )

        a = np.reshape(a, (2, 3, 1))
        a_p = np.reshape(a_p, (2, 3, 1))
        np.testing.assert_almost_equal(
            exponent_function_basic(a, 2.2), a_p, decimal=7
        )
        np.testing.assert_almost_equal(
            exponent_function_basic(a, 2.2, "basicMirrorFwd"), a_p, decimal=7
        )
        np.testing.assert_almost_equal(
            exponent_function_basic(a, 2.2, "basicPassThruFwd"), a_p, decimal=7
        )

        a = 0.0229932049927
        a_p = 0.18

        a = np.tile(a, 6)
        a_p = np.tile(a_p, 6)
        np.testing.assert_almost_equal(
            exponent_function_basic(a, 2.2, "basicRev"), a_p, decimal=7
        )
        np.testing.assert_almost_equal(
            exponent_function_basic(a, 2.2, "basicMirrorRev"), a_p, decimal=7
        )
        np.testing.assert_almost_equal(
            exponent_function_basic(a, 2.2, "basicPassThruRev"), a_p, decimal=7
        )

        a = np.reshape(a, (2, 3))
        a_p = np.reshape(a_p, (2, 3))
        np.testing.assert_almost_equal(
            exponent_function_basic(a, 2.2, "basicRev"), a_p, decimal=7
        )
        np.testing.assert_almost_equal(
            exponent_function_basic(a, 2.2, "basicMirrorRev"), a_p, decimal=7
        )
        np.testing.assert_almost_equal(
            exponent_function_basic(a, 2.2, "basicPassThruRev"), a_p, decimal=7
        )

        a = np.reshape(a, (2, 3, 1))
        a_p = np.reshape(a_p, (2, 3, 1))
        np.testing.assert_almost_equal(
            exponent_function_basic(a, 2.2, "basicRev"), a_p, decimal=7
        )
        np.testing.assert_almost_equal(
            exponent_function_basic(a, 2.2, "basicMirrorRev"), a_p, decimal=7
        )
        np.testing.assert_almost_equal(
            exponent_function_basic(a, 2.2, "basicPassThruRev"), a_p, decimal=7
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


class TestExponentFunctionMonitorCurve(unittest.TestCase):
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
        self.assertAlmostEqual(
            exponent_function_monitor_curve(a, 2.2, 0.001), a_p, places=7
        )

        self.assertAlmostEqual(
            exponent_function_monitor_curve(
                a, 2.2, 0.001, "monCurveMirrorFwd"
            ),
            a_p,
            places=7,
        )

        a = 0.0232240466001
        a_p = 0.18
        self.assertAlmostEqual(
            exponent_function_monitor_curve(a, 2.2, 0.001, "monCurveRev"),
            a_p,
            places=7,
        )

        self.assertAlmostEqual(
            exponent_function_monitor_curve(
                a, 2.2, 0.001, "monCurveMirrorRev"
            ),
            a_p,
            places=7,
        )

        a = -0.18
        self.assertAlmostEqual(
            exponent_function_monitor_curve(a, 2.2, 0.001),
            -0.000205413951,
            places=7,
        )

        self.assertAlmostEqual(
            exponent_function_monitor_curve(
                a, 2.2, 0.001, "monCurveMirrorFwd"
            ),
            -0.0232240466001,
            places=7,
        )

        a = -0.000205413951
        self.assertAlmostEqual(
            exponent_function_monitor_curve(a, 2.2, 0.001, "monCurveRev"),
            -0.18,
            places=7,
        )

        self.assertAlmostEqual(
            exponent_function_monitor_curve(
                a, 2.2, 0.001, "monCurveMirrorRev"
            ),
            -0.0201036111565,
            places=7,
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
        np.testing.assert_almost_equal(
            exponent_function_monitor_curve(a, 2.2, 0.001), a_p, decimal=7
        )
        np.testing.assert_almost_equal(
            exponent_function_monitor_curve(
                a, 2.2, 0.001, "monCurveMirrorFwd"
            ),
            a_p,
            decimal=7,
        )

        a = np.reshape(a, (2, 3))
        a_p = np.reshape(a_p, (2, 3))
        np.testing.assert_almost_equal(
            exponent_function_monitor_curve(a, 2.2, 0.001), a_p, decimal=7
        )
        np.testing.assert_almost_equal(
            exponent_function_monitor_curve(
                a, 2.2, 0.001, "monCurveMirrorFwd"
            ),
            a_p,
            decimal=7,
        )

        a = np.reshape(a, (2, 3, 1))
        a_p = np.reshape(a_p, (2, 3, 1))
        np.testing.assert_almost_equal(
            exponent_function_monitor_curve(a, 2.2, 0.001), a_p, decimal=7
        )
        np.testing.assert_almost_equal(
            exponent_function_monitor_curve(
                a, 2.2, 0.001, "monCurveMirrorFwd"
            ),
            a_p,
            decimal=7,
        )

        a = 0.0232240466001
        a_p = 0.18

        a = np.tile(a, 6)
        a_p = np.tile(a_p, 6)
        np.testing.assert_almost_equal(
            exponent_function_monitor_curve(a, 2.2, 0.001, "monCurveRev"),
            a_p,
            decimal=7,
        )
        np.testing.assert_almost_equal(
            exponent_function_monitor_curve(
                a, 2.2, 0.001, "monCurveMirrorRev"
            ),
            a_p,
            decimal=7,
        )

        a = np.reshape(a, (2, 3))
        a_p = np.reshape(a_p, (2, 3))
        np.testing.assert_almost_equal(
            exponent_function_monitor_curve(a, 2.2, 0.001, "monCurveRev"),
            a_p,
            decimal=7,
        )
        np.testing.assert_almost_equal(
            exponent_function_monitor_curve(
                a, 2.2, 0.001, "monCurveMirrorRev"
            ),
            a_p,
            decimal=7,
        )

        a = np.reshape(a, (2, 3, 1))
        a_p = np.reshape(a_p, (2, 3, 1))
        np.testing.assert_almost_equal(
            exponent_function_monitor_curve(a, 2.2, 0.001, "monCurveRev"),
            a_p,
            decimal=7,
        )
        np.testing.assert_almost_equal(
            exponent_function_monitor_curve(
                a, 2.2, 0.001, "monCurveMirrorRev"
            ),
            a_p,
            decimal=7,
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


if __name__ == "__main__":
    unittest.main()
