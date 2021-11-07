# -*- coding: utf-8 -*-
"""
LUT Operator
============

Defines the *LUT* sequence class:

-   :class:`colour.LUTSequence`
"""

import re
from collections.abc import MutableSequence
from copy import deepcopy

from colour.algebra import LinearInterpolator, table_interpolation_trilinear
from colour.io.luts import AbstractLUTSequenceOperator, LUT1D, LUT3x1D, LUT3D
from colour.utilities import attest

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['LUTSequence']


class LUTSequence(MutableSequence):
    """
    Defines the base class for a *LUT* sequence, i.e. a series of *LUTs*.

    The `colour.LUTSequence` class can be used to model series of *LUTs* such
    as when a shaper *LUT* is combined with a 3D *LUT*.

    Other Parameters
    ----------------
    \\*args : list, optional
        Sequence of `colour.LUT1D`, `colour.LUT3x1D`, `colour.LUT3D` or
        `colour.io.lut.l.AbstractLUTSequenceOperator` class instances.

    Attributes
    ----------
    -   :attr:`~colour.LUTSequence.sequence`

    Methods
    -------
    -   :meth:`~colour.LUTSequence.__init__`
    -   :meth:`~colour.LUTSequence.__getitem__`
    -   :meth:`~colour.LUTSequence.__setitem__`
    -   :meth:`~colour.LUTSequence.__delitem__`
    -   :meth:`~colour.LUTSequence.__len__`
    -   :meth:`~colour.LUTSequence.__str__`
    -   :meth:`~colour.LUTSequence.__repr__`
    -   :meth:`~colour.LUTSequence.__eq__`
    -   :meth:`~colour.LUTSequence.__ne__`
    -   :meth:`~colour.LUTSequence.insert`
    -   :meth:`~colour.LUTSequence.apply`
    -   :meth:`~colour.LUTSequence.copy`

    Examples
    --------
    >>> LUT_1 = LUT1D()
    >>> LUT_2 = LUT3D(size=3)
    >>> LUT_3 = LUT3x1D()
    >>> print(LUTSequence(LUT_1, LUT_2, LUT_3))
    LUT Sequence
    ------------
    <BLANKLINE>
    Overview
    <BLANKLINE>
        LUT1D ---> LUT3D ---> LUT3x1D
    <BLANKLINE>
    Operations
    <BLANKLINE>
        LUT1D - Unity 10
        ----------------
    <BLANKLINE>
        Dimensions : 1
        Domain     : [ 0.  1.]
        Size       : (10,)
    <BLANKLINE>
        LUT3D - Unity 3
        ---------------
    <BLANKLINE>
        Dimensions : 3
        Domain     : [[ 0.  0.  0.]
                      [ 1.  1.  1.]]
        Size       : (3, 3, 3, 3)
    <BLANKLINE>
        LUT3x1D - Unity 10
        ------------------
    <BLANKLINE>
        Dimensions : 2
        Domain     : [[ 0.  0.  0.]
                      [ 1.  1.  1.]]
        Size       : (10, 3)
    """

    def __init__(self, *args):
        for arg in args:
            attest(
                isinstance(
                    arg, (LUT1D, LUT3x1D, LUT3D, AbstractLUTSequenceOperator)),
                '"args" elements must be instances of "LUT1D", '
                '"LUT3x1D", "LUT3D" or "AbstractLUTSequenceOperator"!')

        self._sequence = list(args)

    @property
    def sequence(self):
        """
        Getter and setter property for the underlying *LUT* sequence.

        Parameters
        ----------
        value : list
            Value to set the the underlying *LUT* sequence with.

        Returns
        -------
        list
            Underlying *LUT* sequence.
        """

        return self._sequence

    @sequence.setter
    def sequence(self, value):
        """
        Setter for **self.sequence** property.
        """

        if value is not None:
            self._sequence = list(value)

    def __getitem__(self, index):
        """
        Returns the *LUT* sequence item at given index.

        Parameters
        ----------
        index : int
            *LUT* sequence item index.

        Returns
        -------
        LUT1D or LUT3x1D or LUT3D or AbstractLUTSequenceOperator
            *LUT* sequence item at given index.
        """

        return self._sequence[index]

    def __setitem__(self, index, value):
        """
        Sets given the *LUT* sequence item at given index with given value.

        Parameters
        ----------
        index : int
            *LUT* sequence item index.
        value : LUT1D or LUT3x1D or LUT3D or AbstractLUTSequenceOperator
            Value.
        """

        self._sequence[index] = value

    def __delitem__(self, index):
        """
        Deletes the *LUT* sequence item at given index.

        Parameters
        ----------
        index : int
            *LUT* sequence item index.
        """

        del self._sequence[index]

    def __len__(self):
        """
        Returns the *LUT* sequence items count.

        Returns
        -------
        int
            *LUT* sequence items count.
        """

        return len(self._sequence)

    def __str__(self):
        """
        Returns a formatted string representation of the *LUT* sequence.

        Returns
        -------
        str
            Formatted string representation.
        """

        operations = re.sub(
            '^',
            ' ' * 4,
            '\n\n'.join([str(a) for a in self._sequence]),
            flags=re.MULTILINE)
        operations = re.sub('^\\s+$', '', operations, flags=re.MULTILINE)

        return ('LUT Sequence\n'
                '------------\n\n'
                'Overview\n\n'
                '    {0}\n\n'
                'Operations\n\n'
                '{1}').format(
                    ' ---> '.join(
                        [a.__class__.__name__ for a in self._sequence]),
                    operations)

    def __repr__(self):
        """
        Returns an evaluable string representation of the *LUT* sequence.

        Returns
        -------
        str
            Evaluable string representation.
        """

        operations = re.sub(
            '^',
            ' ' * 4,
            ',\n'.join([repr(a) for a in self._sequence]),
            flags=re.MULTILINE)
        operations = re.sub('^\\s+$', '', operations, flags=re.MULTILINE)

        return '{0}(\n{1}\n)'.format(self.__class__.__name__, operations)

    def __eq__(self, other):
        """
        Returns whether the *LUT* sequence is equal to given other object.

        Parameters
        ----------
        other : object
            Object to test whether it is equal to the *LUT* sequence.

        Returns
        -------
        bool
            Is given object equal to the *LUT* sequence.
        """

        if not isinstance(other, LUTSequence):
            return False

        if len(self) != len(other):
            return False

        # pylint: disable=C0200
        for i in range(len(self)):
            if self[i] != other[i]:
                return False

        return True

    def __ne__(self, other):
        """
        Returns whether the *LUT* sequence is not equal to given other object.

        Parameters
        ----------
        other : object
            Object to test whether it is not equal to the *LUT* sequence.

        Returns
        -------
        bool
            Is given object not equal to the *LUT* sequence.
        """

        return not (self == other)

    # pylint: disable=W0221
    def insert(self, index, LUT):
        """
        Inserts given *LUT* at given index into the *LUT* sequence.

        Parameters
        ----------
        index : index
            Index to insert the *LUT* at into the *LUT* sequence.
        LUT : LUT1D or LUT3x1D or LUT3D or AbstractLUTSequenceOperator
            *LUT* to insert into the *LUT* sequence.
        """

        attest(
            isinstance(LUT,
                       (LUT1D, LUT3x1D, LUT3D, AbstractLUTSequenceOperator)),
            '"LUT" must be an instance of "LUT1D", "LUT3x1D", "LUT3D" or '
            '"AbstractLUTSequenceOperator"!')

        self._sequence.insert(index, LUT)

    def apply(self,
              RGB,
              interpolator_1D=LinearInterpolator,
              interpolator_1D_kwargs=None,
              interpolator_3D=table_interpolation_trilinear,
              interpolator_3D_kwargs=None):
        """
        Applies the *LUT* sequence sequentially to given *RGB* colourspace
        array.

        Parameters
        ----------
        RGB : array_like
            *RGB* colourspace array to apply the *LUT* sequence sequentially
            onto.
        interpolator_1D : object, optional
            Interpolator object to use as interpolating function for
            :class:`colour.LUT1D` (and :class:`colour.LUT3x1D`) class
            instances.
        interpolator_1D_kwargs : dict_like, optional
            Arguments to use when calling the interpolating function for
            :class:`colour.LUT1D` (and :class:`colour.LUT3x1D`) class
            instances.
        interpolator_3D : object, optional
            Interpolator object to use as interpolating function for
            :class:`colour.LUT3D` class instances.
        interpolator_3D_kwargs : dict_like, optional
            Arguments to use when calling the interpolating function for
            :class:`colour.LUT3D` class instances.

        Returns
        -------
        ndarray
            Processed *RGB* colourspace array.

        Examples
        --------
        >>> import numpy as np
        >>> from colour.utilities import tstack
        >>> LUT_1 = LUT1D(LUT1D.linear_table(16) + 0.125)
        >>> LUT_2 = LUT3D(LUT3D.linear_table(16) ** (1 / 2.2))
        >>> LUT_3 = LUT3x1D(LUT3x1D.linear_table(16) * 0.750)
        >>> LUT_sequence = LUTSequence(LUT_1, LUT_2, LUT_3)
        >>> samples = np.linspace(0, 1, 5)
        >>> RGB = tstack([samples, samples, samples])
        >>> LUT_sequence.apply(RGB)  # doctest: +ELLIPSIS
        array([[ 0.2899886...,  0.2899886...,  0.2899886...],
               [ 0.4797662...,  0.4797662...,  0.4797662...],
               [ 0.6055328...,  0.6055328...,  0.6055328...],
               [ 0.7057779...,  0.7057779...,  0.7057779...],
               [ 0.75     ...,  0.75     ...,  0.75     ...]])
        """

        for operation in self:
            if isinstance(operation, (LUT1D, LUT3x1D)):
                RGB = operation.apply(RGB, interpolator_1D,
                                      interpolator_1D_kwargs)
            elif isinstance(operation, LUT3D):
                RGB = operation.apply(RGB, interpolator_3D,
                                      interpolator_3D_kwargs)
            else:
                RGB = operation.apply(RGB)

        return RGB

    def copy(self):
        """
        Returns a copy of the *LUT* sequence.

        Returns
        -------
        LUTSequence
            *LUT* sequence copy.
        """

        return deepcopy(self)
