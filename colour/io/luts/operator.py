# -*- coding: utf-8 -*-
"""
LUT Operator
============

Defines the *LUT* operator classes:

-   :class:`colour.io.AbstractLUTSequenceOperator`
-   :class:`colour.io.Matrix`
"""

import numpy as np
from abc import ABC, abstractmethod

from colour.algebra import vector_dot
from colour.utilities import (as_float_array, is_iterable, is_string, ones,
                              tsplit, tstack)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['AbstractLUTSequenceOperator', 'Matrix']


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
    def apply(self, RGB, *args):
        """
        Applies the *LUT* sequence operator to given *RGB* colourspace array.

        Parameters
        ----------
        RGB : array_like
            *RGB* colourspace array to apply the *LUT* sequence operator onto.

        Returns
        -------
        ndarray
            Processed *RGB* colourspace array.
        """

        pass


class Matrix(AbstractLUTSequenceOperator):
    """
    Defines the base class for a *Matrix* transform.

    Parameters
    ----------
    array : array_like, optional
        3x3 or 3x4 matrix for the transform.
    name : unicode, optional
        *Matrix* name.
    comments : array_like, optional
        Comments to add to the *Matrix*.

    Methods
    -------
    __str__
    __repr__
    __eq__
    __ne__
    apply

    Examples
    --------
    Instantiating an identity matrix:

    >>> print(Matrix(name='Identity'))
    Matrix - Identity
    -----------------
    <BLANKLINE>
    Dimensions : (3, 3)
    Matrix     : [[ 1.  0.  0.]
                  [ 0.  1.  0.]
                  [ 0.  0.  1.]]

    Instantiating a matrix with comments:

    >>> array = np.array([[ 1.45143932, -0.23651075, -0.21492857],
    ...                   [-0.07655377,  1.1762297 , -0.09967593],
    ...                   [ 0.00831615, -0.00603245,  0.9977163 ]])
    >>> print(Matrix(
    ...         array,
    ...         name='AP0 to AP1',
    ...         comments=['A first comment.', 'A second comment.']))
    Matrix - AP0 to AP1
    -------------------
    <BLANKLINE>
    Dimensions : (3, 3)
    Matrix     : [[ 1.45143932 -0.23651075 -0.21492857]
                  [-0.07655377  1.1762297  -0.09967593]
                  [ 0.00831615 -0.00603245  0.9977163 ]]
    <BLANKLINE>
    A first comment.
    A second comment.
    """

    def __init__(self, array=None, *args, **kwargs):
        super(Matrix, self).__init__(*args, **kwargs)

        self._array = np.identity(3)
        self.array = array

    @property
    def array(self):
        """
        Getter and setter property for the *Matrix* array.

        Parameters
        ----------
        value : unicode
            Value to set the *Matrix* array with.

        Returns
        -------
        unicode
            *Matrix* array.
        """

        return self._array

    @array.setter
    def array(self, value):
        """
        Setter for **self.array** property.
        """

        if value is not None:
            value = as_float_array(value)

            assert value.shape in [
                (3, 3), (3, 4)
            ], (('"{0}" attribute: "{1}" shape is not (3, 3) or (3, 4)!'
                 ).format('array', value))

            self._array = value

    def __str__(self):
        """
        Returns a formatted string representation of the *Matrix*.

        Returns
        -------
        unicode
            Formatted string representation.

        Examples
        --------
        >>> print(Matrix())  # doctest: +ELLIPSIS
        Matrix - LUT Sequence Operator ...
        -----------------------------------------
        <BLANKLINE>
        Dimensions : (3, 3)
        Matrix     : [[ 1.  0.  0.]
                      [ 0.  1.  0.]
                      [ 0.  0.  1.]]
        """

        def _indent_array(a):
            """
            Indents given array string representation.
            """

            return str(a).replace(' [', ' ' * 14 + '[')

        return ('{0} - {1}\n'
                '{2}\n\n'
                'Dimensions : {3}\n'
                'Matrix     : {4}'
                '{5}'.format(
                    self.__class__.__name__, self.name,
                    '-' * (len(self.__class__.__name__) + 3 + len(self.name)),
                    self.array.shape, _indent_array(
                        self.array), '\n\n{0}'.format('\n'.join(self.comments))
                    if self.comments else ''))

    def __repr__(self):
        """
        Returns an evaluable string representation of the *Matrix*.

        Returns
        -------
        unicode
            Evaluable string representation.

        Examples
        --------
        >>> Matrix(comments=['A first comment.', 'A second comment.'])
        ... # doctest: +ELLIPSIS
        Matrix([[ 1.,  0.,  0.],
                [ 0.,  1.,  0.],
                [ 0.,  0.,  1.]],
               name='LUT Sequence Operator ...',
               comments=['A first comment.', 'A second comment.'])
        """

        representation = repr(self.array)
        representation = representation.replace('array',
                                                self.__class__.__name__)
        representation = representation.replace(
            '       [',
            '{0}['.format(' ' * (len(self.__class__.__name__) + 2)))

        indentation = ' ' * (len(self.__class__.__name__) + 1)
        representation = ('{0},\n'
                          '{1}name=\'{2}\''
                          '{3})').format(
                              representation[:-1], indentation, self.name,
                              ',\n{0}comments={1}'.format(
                                  indentation, repr(self.comments))
                              if self.comments else '')

        return representation

    def __eq__(self, other):
        """
        Returns whether the *Matrix* is equal to given other object.

        Parameters
        ----------
        other : object
            Object to test whether it is equal to the *Matrix*.

        Returns
        -------
        bool
            Is given object equal to the *Matrix*.

        Examples
        --------
        >>> Matrix() == Matrix()
        True
        """

        if isinstance(other, Matrix):
            if np.array_equal(self.array, other.array):
                return True

        return False

    def __ne__(self, other):
        """
        Returns whether the *Matrix* is not equal to given other object.

        Parameters
        ----------
        other : object
            Object to test whether it is not equal to the *Matrix*.

        Returns
        -------
        bool
            Is given object not equal to the *Matrix*.

        Examples
        --------
        >>> Matrix() != Matrix(np.linspace(0, 1, 12).reshape([3, 4]))
        True
        """

        return not (self == other)

    def apply(self, RGB):
        """
        Applies the *Matrix* transform to given *RGB* array.

        Parameters
        ----------
        RGB : array_like
            *RGB* array to apply the *Matrix* transform to.

        Returns
        -------
        ndarray
            Transformed *RGB* array.

        Examples
        --------
        >>> array = np.array([[ 1.45143932, -0.23651075, -0.21492857],
        ...                   [-0.07655377,  1.1762297 , -0.09967593],
        ...                   [ 0.00831615, -0.00603245,  0.9977163 ]])
        >>> M = Matrix(array)
        >>> RGB = np.array([0.3, 0.4, 0.5])
        >>> M.apply(RGB)  # doctest: +ELLIPSIS
        array([ 0.2333632...,  0.3976877...,  0.4989400...])
        """

        RGB = as_float_array(RGB)

        if self.array.shape == (3, 4):
            R, G, B = tsplit(RGB)
            RGB = tstack((R, G, B, ones(R.shape)))

        return vector_dot(self.array, RGB)
