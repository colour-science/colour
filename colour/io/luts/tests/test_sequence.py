# -*- coding: utf-8 -*-
"""
Defines the unit tests for the :mod:`colour.io.luts.sequence` module.
"""

from __future__ import annotations

import numpy as np
import textwrap
import unittest

from colour.io.luts import (
    AbstractLUTSequenceOperator,
    LUT1D,
    LUT3x1D,
    LUT3D,
    LUTSequence,
)
from colour.hints import FloatingOrNDArray
from colour.models import gamma_function
from colour.utilities import tstack

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestLUTSequence',
]


class TestLUTSequence(unittest.TestCase):
    """
    Defines :class:`colour.io.luts.sequence.LUTSequence` class unit tests
    methods.
    """

    def setUp(self):
        """
        Initialises common tests attributes.
        """

        self._LUT_1 = LUT1D(LUT1D.linear_table(16) + 0.125, 'Nemo 1D')
        self._LUT_2 = LUT3D(LUT3D.linear_table(16) ** (1 / 2.2), 'Nemo 3D')
        self._LUT_3 = LUT3x1D(LUT3x1D.linear_table(16) * 0.750, 'Nemo 3x1D')
        self._LUT_sequence = LUTSequence(self._LUT_1, self._LUT_2, self._LUT_3)

        samples = np.linspace(0, 1, 5)

        self._RGB = tstack([samples, samples, samples])

    def test_required_attributes(self):
        """
        Tests presence of required attributes.
        """

        required_attributes = ('sequence', )

        for attribute in required_attributes:
            self.assertIn(attribute, dir(LUTSequence))

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ('__init__', '__getitem__', '__setitem__',
                            '__delitem__', '__len__', '__str__', '__repr__',
                            '__eq__', '__ne__', 'insert', 'apply', 'copy')

        for method in required_methods:
            self.assertIn(method, dir(LUTSequence))

    def test_sequence(self):
        """
        Tests :class:`colour.io.luts.sequence.LUTSequence.sequence` property.
        """

        sequence = [self._LUT_1, self._LUT_2, self._LUT_3]
        LUT_sequence = LUTSequence()
        LUT_sequence.sequence = sequence
        self.assertListEqual(self._LUT_sequence.sequence, sequence)

    def test__init__(self):
        """
        Tests :class:`colour.io.luts.sequence.LUTSequence.__init__` method.
        """

        self.assertEqual(
            LUTSequence(self._LUT_1, self._LUT_2, self._LUT_3),
            self._LUT_sequence)

    def test__getitem__(self):
        """
        Tests :class:`colour.io.luts.sequence.LUTSequence.__getitem__` method.
        """

        self.assertEqual(self._LUT_sequence[0], self._LUT_1)
        self.assertEqual(self._LUT_sequence[1], self._LUT_2)
        self.assertEqual(self._LUT_sequence[2], self._LUT_3)

    def test__setitem__(self):
        """
        Tests :class:`colour.io.luts.sequence.LUTSequence.__setitem__` method.
        """

        LUT_sequence = self._LUT_sequence.copy()
        LUT_sequence[0] = self._LUT_3
        LUT_sequence[1] = self._LUT_1
        LUT_sequence[2] = self._LUT_2

        self.assertEqual(LUT_sequence[1], self._LUT_1)
        self.assertEqual(LUT_sequence[2], self._LUT_2)
        self.assertEqual(LUT_sequence[0], self._LUT_3)

    def test__delitem__(self):
        """
        Tests :class:`colour.io.luts.sequence.LUTSequence.__delitem__` method.
        """

        LUT_sequence = self._LUT_sequence.copy()

        del LUT_sequence[0]
        del LUT_sequence[0]

        self.assertEqual(LUT_sequence[0], self._LUT_3)

    def test__len__(self):
        """
        Tests :class:`colour.io.luts.sequence.LUTSequence.__len__` method.
        """

        self.assertEqual(len(self._LUT_sequence), 3)

    def test__str__(self):
        """
        Tests :class:`colour.io.luts.sequence.LUTSequence.__str__` method.
        """

        self.assertEqual(
            str(self._LUT_sequence),
            textwrap.dedent("""
            LUT Sequence
            ------------

            Overview

                LUT1D --> LUT3D --> LUT3x1D

            Operations

                LUT1D - Nemo 1D
                ---------------

                Dimensions : 1
                Domain     : [ 0.  1.]
                Size       : (16,)

                LUT3D - Nemo 3D
                ---------------

                Dimensions : 3
                Domain     : [[ 0.  0.  0.]
                              [ 1.  1.  1.]]
                Size       : (16, 16, 16, 3)

                LUT3x1D - Nemo 3x1D
                -------------------

                Dimensions : 2
                Domain     : [[ 0.  0.  0.]
                              [ 1.  1.  1.]]
                Size       : (16, 3)""")[1:])

    def test__repr__(self):
        """
        Tests :class:`colour.io.luts.sequence.LUTSequence.__repr__` method.
        """

        LUT_sequence = self._LUT_sequence.copy()
        LUT_sequence[1].table = LUT3D.linear_table(5)

        self.assertEqual(
            repr(LUT_sequence),
            textwrap.dedent("""
            LUTSequence(
                LUT1D([ 0.125     ,  0.19166667,  0.25833333,  0.325     ,  \
0.39166667,
                        0.45833333,  0.525     ,  0.59166667,  0.65833333,  \
0.725     ,
                        0.79166667,  0.85833333,  0.925     ,  0.99166667,  \
1.05833333,
                        1.125     ],
                      name='Nemo 1D',
                      domain=[ 0.,  1.]),
                LUT3D([[[[ 0.  ,  0.  ,  0.  ],
                         [ 0.  ,  0.  ,  0.25],
                         [ 0.  ,  0.  ,  0.5 ],
                         [ 0.  ,  0.  ,  0.75],
                         [ 0.  ,  0.  ,  1.  ]],

                        [[ 0.  ,  0.25,  0.  ],
                         [ 0.  ,  0.25,  0.25],
                         [ 0.  ,  0.25,  0.5 ],
                         [ 0.  ,  0.25,  0.75],
                         [ 0.  ,  0.25,  1.  ]],

                        [[ 0.  ,  0.5 ,  0.  ],
                         [ 0.  ,  0.5 ,  0.25],
                         [ 0.  ,  0.5 ,  0.5 ],
                         [ 0.  ,  0.5 ,  0.75],
                         [ 0.  ,  0.5 ,  1.  ]],

                        [[ 0.  ,  0.75,  0.  ],
                         [ 0.  ,  0.75,  0.25],
                         [ 0.  ,  0.75,  0.5 ],
                         [ 0.  ,  0.75,  0.75],
                         [ 0.  ,  0.75,  1.  ]],

                        [[ 0.  ,  1.  ,  0.  ],
                         [ 0.  ,  1.  ,  0.25],
                         [ 0.  ,  1.  ,  0.5 ],
                         [ 0.  ,  1.  ,  0.75],
                         [ 0.  ,  1.  ,  1.  ]]],

                       [[[ 0.25,  0.  ,  0.  ],
                         [ 0.25,  0.  ,  0.25],
                         [ 0.25,  0.  ,  0.5 ],
                         [ 0.25,  0.  ,  0.75],
                         [ 0.25,  0.  ,  1.  ]],

                        [[ 0.25,  0.25,  0.  ],
                         [ 0.25,  0.25,  0.25],
                         [ 0.25,  0.25,  0.5 ],
                         [ 0.25,  0.25,  0.75],
                         [ 0.25,  0.25,  1.  ]],

                        [[ 0.25,  0.5 ,  0.  ],
                         [ 0.25,  0.5 ,  0.25],
                         [ 0.25,  0.5 ,  0.5 ],
                         [ 0.25,  0.5 ,  0.75],
                         [ 0.25,  0.5 ,  1.  ]],

                        [[ 0.25,  0.75,  0.  ],
                         [ 0.25,  0.75,  0.25],
                         [ 0.25,  0.75,  0.5 ],
                         [ 0.25,  0.75,  0.75],
                         [ 0.25,  0.75,  1.  ]],

                        [[ 0.25,  1.  ,  0.  ],
                         [ 0.25,  1.  ,  0.25],
                         [ 0.25,  1.  ,  0.5 ],
                         [ 0.25,  1.  ,  0.75],
                         [ 0.25,  1.  ,  1.  ]]],

                       [[[ 0.5 ,  0.  ,  0.  ],
                         [ 0.5 ,  0.  ,  0.25],
                         [ 0.5 ,  0.  ,  0.5 ],
                         [ 0.5 ,  0.  ,  0.75],
                         [ 0.5 ,  0.  ,  1.  ]],

                        [[ 0.5 ,  0.25,  0.  ],
                         [ 0.5 ,  0.25,  0.25],
                         [ 0.5 ,  0.25,  0.5 ],
                         [ 0.5 ,  0.25,  0.75],
                         [ 0.5 ,  0.25,  1.  ]],

                        [[ 0.5 ,  0.5 ,  0.  ],
                         [ 0.5 ,  0.5 ,  0.25],
                         [ 0.5 ,  0.5 ,  0.5 ],
                         [ 0.5 ,  0.5 ,  0.75],
                         [ 0.5 ,  0.5 ,  1.  ]],

                        [[ 0.5 ,  0.75,  0.  ],
                         [ 0.5 ,  0.75,  0.25],
                         [ 0.5 ,  0.75,  0.5 ],
                         [ 0.5 ,  0.75,  0.75],
                         [ 0.5 ,  0.75,  1.  ]],

                        [[ 0.5 ,  1.  ,  0.  ],
                         [ 0.5 ,  1.  ,  0.25],
                         [ 0.5 ,  1.  ,  0.5 ],
                         [ 0.5 ,  1.  ,  0.75],
                         [ 0.5 ,  1.  ,  1.  ]]],

                       [[[ 0.75,  0.  ,  0.  ],
                         [ 0.75,  0.  ,  0.25],
                         [ 0.75,  0.  ,  0.5 ],
                         [ 0.75,  0.  ,  0.75],
                         [ 0.75,  0.  ,  1.  ]],

                        [[ 0.75,  0.25,  0.  ],
                         [ 0.75,  0.25,  0.25],
                         [ 0.75,  0.25,  0.5 ],
                         [ 0.75,  0.25,  0.75],
                         [ 0.75,  0.25,  1.  ]],

                        [[ 0.75,  0.5 ,  0.  ],
                         [ 0.75,  0.5 ,  0.25],
                         [ 0.75,  0.5 ,  0.5 ],
                         [ 0.75,  0.5 ,  0.75],
                         [ 0.75,  0.5 ,  1.  ]],

                        [[ 0.75,  0.75,  0.  ],
                         [ 0.75,  0.75,  0.25],
                         [ 0.75,  0.75,  0.5 ],
                         [ 0.75,  0.75,  0.75],
                         [ 0.75,  0.75,  1.  ]],

                        [[ 0.75,  1.  ,  0.  ],
                         [ 0.75,  1.  ,  0.25],
                         [ 0.75,  1.  ,  0.5 ],
                         [ 0.75,  1.  ,  0.75],
                         [ 0.75,  1.  ,  1.  ]]],

                       [[[ 1.  ,  0.  ,  0.  ],
                         [ 1.  ,  0.  ,  0.25],
                         [ 1.  ,  0.  ,  0.5 ],
                         [ 1.  ,  0.  ,  0.75],
                         [ 1.  ,  0.  ,  1.  ]],

                        [[ 1.  ,  0.25,  0.  ],
                         [ 1.  ,  0.25,  0.25],
                         [ 1.  ,  0.25,  0.5 ],
                         [ 1.  ,  0.25,  0.75],
                         [ 1.  ,  0.25,  1.  ]],

                        [[ 1.  ,  0.5 ,  0.  ],
                         [ 1.  ,  0.5 ,  0.25],
                         [ 1.  ,  0.5 ,  0.5 ],
                         [ 1.  ,  0.5 ,  0.75],
                         [ 1.  ,  0.5 ,  1.  ]],

                        [[ 1.  ,  0.75,  0.  ],
                         [ 1.  ,  0.75,  0.25],
                         [ 1.  ,  0.75,  0.5 ],
                         [ 1.  ,  0.75,  0.75],
                         [ 1.  ,  0.75,  1.  ]],

                        [[ 1.  ,  1.  ,  0.  ],
                         [ 1.  ,  1.  ,  0.25],
                         [ 1.  ,  1.  ,  0.5 ],
                         [ 1.  ,  1.  ,  0.75],
                         [ 1.  ,  1.  ,  1.  ]]]],
                      name='Nemo 3D',
                      domain=[[ 0.,  0.,  0.],
                              [ 1.,  1.,  1.]]),
                LUT3x1D([[ 0.  ,  0.  ,  0.  ],
                         [ 0.05,  0.05,  0.05],
                         [ 0.1 ,  0.1 ,  0.1 ],
                         [ 0.15,  0.15,  0.15],
                         [ 0.2 ,  0.2 ,  0.2 ],
                         [ 0.25,  0.25,  0.25],
                         [ 0.3 ,  0.3 ,  0.3 ],
                         [ 0.35,  0.35,  0.35],
                         [ 0.4 ,  0.4 ,  0.4 ],
                         [ 0.45,  0.45,  0.45],
                         [ 0.5 ,  0.5 ,  0.5 ],
                         [ 0.55,  0.55,  0.55],
                         [ 0.6 ,  0.6 ,  0.6 ],
                         [ 0.65,  0.65,  0.65],
                         [ 0.7 ,  0.7 ,  0.7 ],
                         [ 0.75,  0.75,  0.75]],
                        name='Nemo 3x1D',
                        domain=[[ 0.,  0.,  0.],
                                [ 1.,  1.,  1.]])
            )""" [1:]))

    def test__eq__(self):
        """
        Tests :class:`colour.io.luts.sequence.LUTSequence.__eq__` method.
        """

        LUT_sequence_1 = LUTSequence(self._LUT_1, self._LUT_2, self._LUT_3)
        LUT_sequence_2 = LUTSequence(self._LUT_1, self._LUT_2)

        self.assertEqual(self._LUT_sequence, LUT_sequence_1)

        self.assertNotEqual(self._LUT_sequence, self._LUT_sequence[0])

        self.assertNotEqual(LUT_sequence_1, LUT_sequence_2)

    def test__neq__(self):
        """
        Tests :class:`colour.io.luts.sequence.LUTSequence.__neq__` method.
        """

        self.assertNotEqual(
            self._LUT_sequence,
            LUTSequence(self._LUT_1,
                        self._LUT_2.copy() * 0.75, self._LUT_3))

    def test_insert(self):
        """
        Tests :class:`colour.io.luts.sequence.LUTSequence.insert` method.
        """

        LUT_sequence = self._LUT_sequence.copy()

        LUT_sequence.insert(1, self._LUT_2.copy())

        self.assertEqual(
            LUT_sequence,
            LUTSequence(
                self._LUT_1,
                self._LUT_2,
                self._LUT_2,
                self._LUT_3,
            ))

    def test_apply(self):
        """
        Tests :class:`colour.io.luts.sequence.LUTSequence.apply` method.
        """

        class GammaOperator(AbstractLUTSequenceOperator):
            """
            Gamma operator for unit tests.

            Parameters
            ----------
            gamma
                Gamma value.
            """

            def __init__(self, gamma: FloatingOrNDArray = 1.0):
                self._gamma = gamma

            def apply(self, RGB, **kwargs):
                """
                Applies the *LUT* sequence operator to given *RGB* colourspace
                array.

                Parameters
                ----------
                RGB
                    *RGB* colourspace array to apply the *LUT* sequence
                    operator onto.

                Returns
                -------
                :class:`numpy.ndarray`
                    Processed *RGB* colourspace array.
                """

                direction = kwargs.get('direction', 'Forward')

                gamma = (self._gamma
                         if direction == 'Forward' else 1 / self._gamma)

                return gamma_function(RGB, gamma)

        LUT_sequence = self._LUT_sequence.copy()
        LUT_sequence.insert(1, GammaOperator(1 / 2.2))
        samples = np.linspace(0, 1, 5)
        RGB = tstack([samples, samples, samples])

        np.testing.assert_almost_equal(
            LUT_sequence.apply(RGB, GammaOperator={'direction': 'Inverse'}),
            np.array([
                [0.03386629, 0.03386629, 0.03386629],
                [0.27852298, 0.27852298, 0.27852298],
                [0.46830881, 0.46830881, 0.46830881],
                [0.65615595, 0.65615595, 0.65615595],
                [0.75000000, 0.75000000, 0.75000000],
            ]))


if __name__ == '__main__':
    unittest.main()
