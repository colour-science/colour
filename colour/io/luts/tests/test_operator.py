"""Defines the unit tests for the :mod:`colour.io.luts.operator` module."""

import numpy as np
import textwrap
import unittest

from colour.io.luts import AbstractLUTSequenceOperator, LUTOperatorMatrix
from colour.utilities import tstack, zeros

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestAbstractLUTSequenceOperator",
    "TestLUTOperatorMatrix",
]


class TestAbstractLUTSequenceOperator(unittest.TestCase):
    """
    Define :class:`colour.io.luts.operator.AbstractLUTSequenceOperator` class
    unit tests methods.
    """

    def test_required_attributes(self):
        """Test the presence of required attributes."""

        required_attributes = ("name", "comments")

        for method in required_attributes:
            self.assertIn(method, dir(AbstractLUTSequenceOperator))

    def test_required_methods(self):
        """Test the presence of required methods."""

        required_methods = ("apply",)

        for method in required_methods:
            self.assertIn(method, dir(AbstractLUTSequenceOperator))


class TestLUTOperatorMatrix(unittest.TestCase):
    """
    Define :class:`colour.io.luts.operator.LUTOperatorMatrix` class unit tests
    methods.
    """

    def setUp(self):
        """Initialise the common tests attributes."""

        self._lut_operator_matrix = LUTOperatorMatrix(
            np.linspace(0, 1, 16).reshape([4, 4]),
            offset=np.array([0.25, 0.5, 0.75, 1.0]),
            name="Nemo Matrix",
            comments=["A first comment.", "A second comment."],
        )

    def test_required_attributes(self):
        """Test the presence of required attributes."""

        required_attributes = ("matrix", "offset")

        for method in required_attributes:
            self.assertIn(method, dir(LUTOperatorMatrix))

    def test_required_methods(self):
        """Test the presence of required methods."""

        required_methods = ("__str__", "__repr__", "__eq__", "__ne__", "apply")

        for method in required_methods:
            self.assertIn(method, dir(LUTOperatorMatrix))

    def test_matrix(self):
        """
        Test :class:`colour.io.luts.operator.LUTOperatorMatrix.matrix`
        property.
        """

        M = np.identity(3)

        lut_operator_matrix = LUTOperatorMatrix(M)
        np.testing.assert_array_equal(
            lut_operator_matrix.matrix, np.identity(4)
        )

    def test_offset(self):
        """
        Test :class:`colour.io.luts.operator.LUTOperatorMatrix.offset`
        property.
        """

        offset = zeros(3)

        lut_operator_matrix = LUTOperatorMatrix(np.identity(3), offset)
        np.testing.assert_array_equal(lut_operator_matrix.offset, zeros(4))

    def test__str__(self):
        """
        Test :class:`colour.io.luts.operator.LUTOperatorMatrix.__str__`
        method.
        """

        self.assertEqual(
            str(self._lut_operator_matrix),
            textwrap.dedent(
                """
            LUTOperatorMatrix - Nemo Matrix
            -------------------------------

            Matrix     : [[ 0.          0.06666667  0.13333333  0.2       ]
                          [ 0.26666667  0.33333333  0.4         0.46666667]
                          [ 0.53333333  0.6         0.66666667  0.73333333]
                          [ 0.8         0.86666667  0.93333333  1.        ]]
            Offset     : [ 0.25  0.5   0.75  1.  ]

            A first comment.
            A second comment."""
            )[1:],
        )

    def test__repr__(self):
        """
        Test :class:`colour.io.luts.operator.LUTOperatorMatrix.__repr__`
        method.
        """

        self.assertEqual(
            repr(self._lut_operator_matrix),
            textwrap.dedent(
                """
LUTOperatorMatrix([[ 0.        ,  0.06666667,  0.13333333,  0.2       ],
                   [ 0.26666667,  0.33333333,  0.4       ,  0.46666667],
                   [ 0.53333333,  0.6       ,  0.66666667,  0.73333333],
                   [ 0.8       ,  0.86666667,  0.93333333,  1.        ]],
                  [ 0.25,  0.5 ,  0.75,  1.  ],
                  name='Nemo Matrix',
                  comments=['A first comment.', 'A second comment.'])"""[
                    1:
                ]
            ),
        )

    def test__eq__(self):
        """Test :class:`colour.io.luts.operator.LUTOperatorMatrix.__eq__` method."""

        matrix = LUTOperatorMatrix(
            np.linspace(0, 1, 16).reshape([4, 4]),
            np.array([0.25, 0.5, 0.75, 1.0]),
        )

        self.assertEqual(self._lut_operator_matrix, matrix)

    def test__neq__(self):
        """
        Test :class:`colour.io.luts.operator.LUTOperatorMatrix.__neq__`
        method.
        """

        matrix = LUTOperatorMatrix(
            np.linspace(0, 1, 16).reshape([4, 4]) * 0.75
        )

        self.assertNotEqual(self._lut_operator_matrix, matrix)

    def test_apply(self):
        """Test :class:`colour.io.luts.operator.LUTOperatorMatrix.apply` method."""

        samples = np.linspace(0, 1, 5)
        RGB = tstack([samples, samples, samples])

        np.testing.assert_array_equal(LUTOperatorMatrix().apply(RGB), RGB)

        np.testing.assert_almost_equal(
            self._lut_operator_matrix.apply(RGB),
            np.array(
                [
                    [0.25000000, 0.50000000, 0.75000000],
                    [0.30000000, 0.75000000, 1.20000000],
                    [0.35000000, 1.00000000, 1.65000000],
                    [0.40000000, 1.25000000, 2.10000000],
                    [0.45000000, 1.50000000, 2.55000000],
                ]
            ),
        )

        np.testing.assert_almost_equal(
            self._lut_operator_matrix.apply(RGB, apply_offset_first=True),
            np.array(
                [
                    [0.13333333, 0.53333333, 0.93333333],
                    [0.18333333, 0.78333333, 1.38333333],
                    [0.23333333, 1.03333333, 1.83333333],
                    [0.28333333, 1.28333333, 2.28333333],
                    [0.33333333, 1.53333333, 2.73333333],
                ]
            ),
        )

        RGBA = tstack([samples, samples, samples, samples])

        np.testing.assert_array_equal(LUTOperatorMatrix().apply(RGBA), RGBA)

        np.testing.assert_almost_equal(
            self._lut_operator_matrix.apply(RGBA),
            np.array(
                [
                    [0.25000000, 0.50000000, 0.75000000, 1.00000000],
                    [0.35000000, 0.86666667, 1.38333333, 1.90000000],
                    [0.45000000, 1.23333333, 2.01666667, 2.80000000],
                    [0.55000000, 1.60000000, 2.65000000, 3.70000000],
                    [0.65000000, 1.96666667, 3.28333333, 4.60000000],
                ]
            ),
        )

        np.testing.assert_almost_equal(
            self._lut_operator_matrix.apply(RGBA, apply_offset_first=True),
            np.array(
                [
                    [0.33333333, 1.00000000, 1.66666667, 2.33333333],
                    [0.43333333, 1.36666667, 2.30000000, 3.23333333],
                    [0.53333333, 1.73333333, 2.93333333, 4.13333333],
                    [0.63333333, 2.10000000, 3.56666667, 5.03333333],
                    [0.73333333, 2.46666667, 4.20000000, 5.93333333],
                ],
            ),
        )


if __name__ == "__main__":
    unittest.main()
