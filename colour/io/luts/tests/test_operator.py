# -*- coding: utf-8 -*-
"""
Defines the unit tests for the :mod:`colour.io.luts.operator` module.
"""

import numpy as np
import textwrap
import unittest

from colour.io.luts import AbstractLUTSequenceOperator, Matrix
from colour.utilities import tstack

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['TestAbstractLUTSequenceOperator', 'TestMatrix']


class TestAbstractLUTSequenceOperator(unittest.TestCase):
    """
    Defines :class:`colour.io.luts.operator.AbstractLUTSequenceOperator` class
    unit tests methods.
    """

    def test_required_attributes(self):
        """
        Tests presence of required attributes.
        """

        required_attributes = ('name', 'comments')

        for method in required_attributes:
            self.assertIn(method, dir(AbstractLUTSequenceOperator))

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ('apply', )

        for method in required_methods:
            self.assertIn(method, dir(AbstractLUTSequenceOperator))


class TestMatrix(unittest.TestCase):
    """
    Defines :class:`colour.io.luts.operator.Matrix` class unit tests methods.
    """

    def setUp(self):
        """
        Initialises common tests attributes.
        """

        self._matrix = Matrix(
            np.linspace(0, 1, 12).reshape([3, 4]),
            name='Nemo Matrix',
            comments=['A first comment.', 'A second comment.'])

    def test_required_attributes(self):
        """
        Tests presence of required attributes.
        """

        required_attributes = ('array', )

        for method in required_attributes:
            self.assertIn(method, dir(Matrix))

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ('__str__', '__repr__', '__eq__', '__ne__', 'apply')

        for method in required_methods:
            self.assertIn(method, dir(Matrix))

    def test__str__(self):
        """
        Tests :class:`colour.io.luts.operator.Matrix.__str__` method.
        """

        self.assertEqual(
            str(self._matrix),
            textwrap.dedent("""
            Matrix - Nemo Matrix
            --------------------

            Dimensions : (3, 4)
            Matrix     : [[ 0.          0.09090909  0.18181818  0.27272727]
                          [ 0.36363636  0.45454545  0.54545455  0.63636364]
                          [ 0.72727273  0.81818182  0.90909091  1.        ]]

            A first comment.
            A second comment.""")[1:])

    def test__repr__(self):
        """
        Tests :class:`colour.io.luts.operator.Matrix.__repr__` method.
        """

        self.assertEqual(
            repr(self._matrix),
            textwrap.dedent("""
        Matrix([[ 0.        ,  0.09090909,  0.18181818,  0.27272727],
                [ 0.36363636,  0.45454545,  0.54545455,  0.63636364],
                [ 0.72727273,  0.81818182,  0.90909091,  1.        ]],
               name='Nemo Matrix',
               comments=['A first comment.', 'A second comment.'])""" [1:]))

    def test__eq__(self):
        """
        Tests :class:`colour.io.luts.operator.Matrix.__eq__` method.
        """

        matrix = Matrix(np.linspace(0, 1, 12).reshape([3, 4]))

        self.assertEqual(self._matrix, matrix)

    def test__neq__(self):
        """
        Tests :class:`colour.io.luts.operator.Matrix.__neq__` method.
        """

        matrix = Matrix(np.linspace(0, 1, 12).reshape([3, 4]) * 0.75)

        self.assertNotEqual(self._matrix, matrix)

    def test_apply(self):
        """
        Tests :class:`colour.io.luts.operator.Matrix.apply` method.
        """

        samples = np.linspace(0, 1, 5)
        RGB = tstack([samples, samples, samples])

        np.testing.assert_almost_equal(
            self._matrix.apply(RGB),
            np.array([
                [0.27272727, 0.63636364, 1.00000000],
                [0.34090909, 0.97727273, 1.61363636],
                [0.40909091, 1.31818182, 2.22727273],
                [0.47727273, 1.65909091, 2.84090909],
                [0.54545455, 2.00000000, 3.45454545],
            ]))


if __name__ == '__main__':
    unittest.main()
