# -*- coding: utf-8 -*-
"""
LUT Operator
============

Defines the *LUT* operator classes:

-   :class:`colour.io.AbstractLUTSequenceOperator`
-   :class:`colour.io.LUTOperatorMatrix`
"""

import numpy as np
from abc import ABC, abstractmethod

from colour.algebra import vector_dot
from colour.utilities import (as_float_array, is_iterable, is_string, ones,
                              zeros)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['AbstractLUTSequenceOperator', 'LUTOperatorMatrix']


class AbstractLUTSequenceOperator(ABC):
    """
    Defines the base class for *LUT* sequence operators.

    This is an :class:`ABCMeta` abstract class that must be inherited by
    sub-classes.

    Parameters
    ----------
    name : unicode, optional
        *LUT* sequence operator name.
    comments : array_like, optional
        Comments to add to the *LUT* sequence operator.

    Attributes
    ----------
    -   :attr:`~colour.io.AbstractLUTSequenceOperator.name`
    -   :attr:`~colour.io.AbstractLUTSequenceOperator.comments`

    Methods
    -------
    -   :meth:`~colour.io.AbstractLUTSequenceOperator.apply`
    """

    def __init__(self, name=None, comments=None):
        self._name = 'LUT Sequence Operator {0}'.format(id(self))
        self.name = name
        self._comments = []
        self.comments = comments

    @property
    def name(self):
        """
        Getter and setter property for the *LUT* name.

        Parameters
        ----------
        value : unicode
            Value to set the *LUT* name with.

        Returns
        -------
        unicode
            *LUT* name.
        """

        return self._name

    @name.setter
    def name(self, value):
        """
        Setter for **self.name** property.
        """

        if value is not None:
            assert is_string(value), (
                ('"{0}" attribute: "{1}" type is not "str" or "unicode"!'
                 ).format('name', value))

            self._name = value

    @property
    def comments(self):
        """
        Getter and setter property for the *LUT* comments.

        Parameters
        ----------
        value : unicode
            Value to set the *LUT* comments with.

        Returns
        -------
        unicode
            *LUT* comments.
        """

        return self._comments

    @comments.setter
    def comments(self, value):
        """
        Setter for **self.comments** property.
        """

        if value is not None:
            assert is_iterable(value), ((
                '"{0}" attribute: "{1}" must be an array like!').format(
                    'comments', value))

            self._comments = value

    @abstractmethod
    def apply(self, RGB, *args, **kwargs):
        """
        Applies the *LUT* sequence operator to given *RGB* colourspace array.

        Parameters
        ----------
        RGB : array_like
            *RGB* colourspace array to apply the *LUT* sequence operator onto.

        Other Parameters
        ----------------
        \\*args : list, optional
            Arguments.
        \\**kwargs : dict, optional
            Keywords arguments.

        Returns
        -------
        ndarray
            Processed *RGB* colourspace array.
        """

        pass


class LUTOperatorMatrix(AbstractLUTSequenceOperator):
    """
    Defines the *LUT* operator supporting a 3x3 or 4x4 matrix and an offset
    vector.

    Parameters
    ----------
    matrix : array_like, optional
        3x3 or 4x4 matrix for the operator.
    offset : array_like, optional
        Offset for the operator.
    name : unicode, optional
        *LUT* operator name.
    comments : array_like, optional
        Comments to add to the *LUT* operator.

    Attributes
    ----------
     -   :meth:`~colour.io.LUTOperatorMatrix.matrix`
     -   :meth:`~colour.io.LUTOperatorMatrix.offset`

    Methods
    -------
     -   :meth:`~colour.io.LUTOperatorMatrix.__str__`
     -   :meth:`~colour.io.LUTOperatorMatrix.__repr__`
     -   :meth:`~colour.io.LUTOperatorMatrix.__eq__`
     -   :meth:`~colour.io.LUTOperatorMatrix.__ne__`
     -   :meth:`~colour.io.LUTOperatorMatrix.apply`

    Notes
    -----
    -    The internal :attr:`colour.io.Matrix.matrix` and
        :attr:`colour.io.Matrix.offset` attributes are reshaped to (4, 4) and
        (4, ) respectively.

    Examples
    --------
    Instantiating an identity matrix:

    >>> print(LUTOperatorMatrix(name='Identity'))
    LUTOperatorMatrix - Identity
    ----------------------------
    <BLANKLINE>
    Matrix     : [[ 1.  0.  0.  0.]
                  [ 0.  1.  0.  0.]
                  [ 0.  0.  1.  0.]
                  [ 0.  0.  0.  1.]]
    Offset     : [ 0.  0.  0.  0.]

    Instantiating a matrix with comments:

    >>> matrix = np.array([[ 1.45143932, -0.23651075, -0.21492857],
    ...                    [-0.07655377,  1.1762297 , -0.09967593],
    ...                    [ 0.00831615, -0.00603245,  0.9977163 ]])
    >>> print(LUTOperatorMatrix(
    ...         matrix,
    ...         name='AP0 to AP1',
    ...         comments=['A first comment.', 'A second comment.']))
    LUTOperatorMatrix - AP0 to AP1
    ------------------------------
    <BLANKLINE>
    Matrix     : [[ 1.45143932 -0.23651075 -0.21492857  0.        ]
                  [-0.07655377  1.1762297  -0.09967593  0.        ]
                  [ 0.00831615 -0.00603245  0.9977163   0.        ]
                  [ 0.          0.          0.          1.        ]]
    Offset     : [ 0.  0.  0.  0.]
    <BLANKLINE>
    A first comment.
    A second comment.
    """

    def __init__(self, matrix=None, offset=None, *args, **kwargs):
        super(LUTOperatorMatrix, self).__init__(*args, **kwargs)

        self._matrix = np.diag(ones(4))
        self.matrix = matrix

        self._offset = zeros(4)
        self.offset = offset

    @property
    def matrix(self):
        """
        Getter and setter property for the *LUT* operator matrix.

        Parameters
        ----------
        value : unicode
            Value to set the *LUT* operator matrix with.

        Returns
        -------
        unicode
            Operator matrix.
        """

        return self._matrix

    @matrix.setter
    def matrix(self, value):
        """
        Setter for **self.matrix** property.
        """

        if value is not None:
            value = as_float_array(value)

            shape_t = value.shape[-1]

            value = value.reshape([shape_t, shape_t])

            assert value.shape in [
                (3, 3), (4, 4)
            ], (('"{0}" attribute: "{1}" shape is not (3, 3) or (4, 4)!'
                 ).format('matrix', value))

            M = np.identity(4)
            M[:shape_t, :shape_t] = value

            self._matrix = M

    @property
    def offset(self):
        """
        Getter and setter property for the *LUT* operator offset.

        Parameters
        ----------
        value : unicode
            Value to set the *LUT* operator offset with.

        Returns
        -------
        unicode
            Operator offset.
        """

        return self._offset

    @offset.setter
    def offset(self, value):
        """
        Setter for **self.offset** property.
        """

        if value is not None:
            value = as_float_array(value)

            shape_t = value.shape[-1]

            assert value.shape in [
                (3, ), (4, )
            ], ('"{0}" attribute: "{1}" shape is not (3, ) or (4, )!'.format(
                'offset', value))

            offset = zeros(4)
            offset[:shape_t] = value

            self._offset = offset

    def __str__(self):
        """
        Returns a formatted string representation of the *LUT* operator.

        Returns
        -------
        unicode
            Formatted string representation.

        Examples
        --------
        >>> print(LUTOperatorMatrix())  # doctest: +ELLIPSIS
        LUTOperatorMatrix - LUT Sequence Operator ...
        ------------------------------------------...
        <BLANKLINE>
        Matrix     : [[ 1.  0.  0.  0.]
                      [ 0.  1.  0.  0.]
                      [ 0.  0.  1.  0.]
                      [ 0.  0.  0.  1.]]
        Offset     : [ 0.  0.  0.  0.]
        """

        def _indent_array(a):
            """
            Indents given array string representation.
            """

            return str(a).replace(' [', ' ' * 14 + '[')

        return ('{0} - {1}\n'
                '{2}\n\n'
                'Matrix     : {3}\n'
                'Offset     : {4}'
                '{5}'.format(
                    self.__class__.__name__, self._name,
                    '-' * (len(self.__class__.__name__) + 3 + len(self._name)),
                    _indent_array(self._matrix), _indent_array(self._offset),
                    '\n\n{0}'.format('\n'.join(self._comments))
                    if self._comments else ''))

    def __repr__(self):
        """
        Returns an evaluable string representation of the *LUT* operator.

        Returns
        -------
        unicode
            Evaluable string representation.

        Examples
        --------
        >>> LUTOperatorMatrix(
        ...     comments=['A first comment.', 'A second comment.'])
        ... # doctest: +ELLIPSIS
        LUTOperatorMatrix([[ 1.,  0.,  0.,  0.],
                           [ 0.,  1.,  0.,  0.],
                           [ 0.,  0.,  1.,  0.],
                           [ 0.,  0.,  0.,  1.]],
                          [ 0.,  0.,  0.,  0.],
                          name='LUT Sequence Operator ...',
                          comments=['A first comment.', 'A second comment.'])
        """

        representation = repr(self._matrix)
        representation = representation.replace('array',
                                                self.__class__.__name__)
        representation = representation.replace(
            '       [',
            '{0}['.format(' ' * (len(self.__class__.__name__) + 2)))

        indentation = ' ' * (len(self.__class__.__name__) + 1)
        representation = ('{0},\n'
                          '{1}{2},\n'
                          '{1}name=\'{3}\''
                          '{4})').format(
                              representation[:-1], indentation,
                              repr(self._offset).replace('array(', '').replace(
                                  ')', ''), self._name,
                              ',\n{0}comments={1}'.format(
                                  indentation, repr(self._comments))
                              if self._comments else '')

        return representation

    def __eq__(self, other):
        """
        Returns whether the *LUT* operator is equal to given other object.

        Parameters
        ----------
        other : object
            Object to test whether it is equal to the *LUT* operator.

        Returns
        -------
        bool
            Is given object equal to the *LUT* operator.

        Examples
        --------
        >>> LUTOperatorMatrix() == LUTOperatorMatrix()
        True
        """

        if isinstance(other, LUTOperatorMatrix):
            if all([
                    np.array_equal(self._matrix, other._matrix),
                    np.array_equal(self._offset, other._offset)
            ]):
                return True

        return False

    def __ne__(self, other):
        """
        Returns whether the *LUT* operator is not equal to given other object.

        Parameters
        ----------
        other : object
            Object to test whether it is not equal to the *LUT* operator.

        Returns
        -------
        bool
            Is given object not equal to the *LUT* operator.

        Examples
        --------
        >>> LUTOperatorMatrix() != LUTOperatorMatrix(
        ...     np.linspace(0, 1, 16).reshape([4, 4]))
        True
        """

        return not (self == other)

    def apply(self, RGB, apply_offset_first=False):
        """
        Applies the *LUT* operator to given *RGB* array.

        Parameters
        ----------
        RGB : array_like
            *RGB* array to apply the *LUT* operator transform to.
        apply_offset_first : bool, optional
            Whether to apply the offset first and then the matrix.

        Returns
        -------
        ndarray
            Transformed *RGB* array.

        Examples
        --------
        >>> matrix = np.array([[ 1.45143932, -0.23651075, -0.21492857],
        ...                    [-0.07655377,  1.1762297 , -0.09967593],
        ...                    [ 0.00831615, -0.00603245,  0.9977163 ]])
        >>> M = LUTOperatorMatrix(matrix)
        >>> RGB = np.array([0.3, 0.4, 0.5])
        >>> M.apply(RGB)  # doctest: +ELLIPSIS
        array([ 0.2333632...,  0.3976877...,  0.4989400...])
        """

        RGB = as_float_array(RGB)

        has_alpha_channel = RGB.shape[-1] == 4
        M = self._matrix
        offset = self._offset

        if not has_alpha_channel:
            M = M[:3, :3]
            offset = offset[:3]

        if apply_offset_first:
            RGB += offset

        RGB = vector_dot(M, RGB)

        if not apply_offset_first:
            RGB += offset

        return RGB
