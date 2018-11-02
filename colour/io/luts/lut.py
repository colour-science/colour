# -*- coding: utf-8 -*-
"""
LUT Processing
==============

Defines the classes and definitions handling *LUT* processing:

-   :class:`colour.LUT1D`
-   :class:`colour.LUT2D`
-   :class:`colour.LUT3D`
-   :class:`colour.LUTSequence`
-   :class:`colour.io.LUT_to_LUT`
"""

from __future__ import division, unicode_literals

import numpy as np
import re
from abc import ABCMeta, abstractmethod
from collections import MutableSequence
from copy import deepcopy
# pylint: disable=W0622
from operator import add, mul, pow, sub, iadd, imul, ipow, isub

# Python 3 compatibility.
try:
    from operator import div, idiv
except ImportError:
    from operator import truediv, itruediv

    div = truediv
    idiv = itruediv
from six import add_metaclass

from colour.algebra import LinearInterpolator, table_interpolation_trilinear
from colour.constants import DEFAULT_INT_DTYPE
from colour.utilities import (as_float_array, is_iterable, is_string,
                              linear_conversion, tsplit, tstack)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'AbstractLUT', 'LUT1D', 'LUT2D', 'LUT3D', 'LUT_to_LUT',
    'AbstractLUTSequenceOperator', 'LUTSequence'
]


@add_metaclass(ABCMeta)
class AbstractLUT:
    """
    Defines the base class for *LUT*.

    This is an :class:`ABCMeta` abstract class that must be inherited by
    sub-classes.

    Parameters
    ----------
    table : array_like, optional
        Underlying *LUT* table.
    name : unicode, optional
        *LUT* name.
    dimensions : int, optional
        *LUT* dimensions, typically, 1 for a 1D *LUT*, 2 for a 2D *LUT* and 3
        for a 3D *LUT*.
    domain : unicode, optional
        *LUT* domain, also used to define the instantiation time default table
        domain.
    size : int, optional
        *LUT* size, also used to define the instantiation time default table
        size.
    comments : array_like, optional
        Comments to add to the *LUT*.

    Attributes
    ----------
    table
    name
    dimensions
    domain
    size
    comments

    Methods
    -------
    __str__
    __repr__
    __eq__
    __ne__
    __add__
    __iadd__
    __sub__
    __isub__
    __mul__
    __imul__
    __div__
    __idiv__
    __pow__
    __ipow__
    arithmetical_operation
    linear_table
    apply
    copy
    as_LUT
    """

    def __init__(self,
                 table=None,
                 name=None,
                 dimensions=None,
                 domain=None,
                 size=33,
                 comments=None):
        default_name = ('Unity {0}'.format(size)
                        if table is None else '{0}'.format(id(self)))
        self._name = default_name
        self.name = name

        self._dimensions = dimensions

        self._domain = None
        self.domain = domain
        # pylint: disable=E1121
        self._table = self.linear_table(size, domain)
        self.table = table
        self._comments = []
        self.comments = comments

    @property
    def table(self):
        """
        Getter and setter property for the underlying *LUT* table.

        Parameters
        ----------
        value : unicode
            Value to set the the underlying *LUT* table with.

        Returns
        -------
        unicode
            Underlying *LUT* table.
        """

        return self._table

    @table.setter
    def table(self, value):
        """
        Setter for **self.table** property.
        """

        if value is not None:
            # pylint: disable=E1121
            self._table = self._validate_table(value)

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
    def domain(self):
        """
        Getter and setter property for the *LUT* domain.

        Parameters
        ----------
        value : unicode
            Value to set the *LUT* domain with.

        Returns
        -------
        unicode
            *LUT* domain.
        """

        return self._domain

    @domain.setter
    def domain(self, value):
        """
        Setter for **self.domain** property.
        """

        if value is not None:
            # pylint: disable=E1121
            self._domain = self._validate_domain(value)

    @property
    def dimensions(self):
        """
        Getter and setter property for the *LUT* dimensions.

        Returns
        -------
        unicode
            *LUT* dimensions.
        """

        return self._dimensions

    @property
    def size(self):
        """
        Getter property for the *LUT* size.

        Returns
        -------
        unicode
            *LUT* size.
        """

        return self._table.shape[0]

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

    def __str__(self):
        """
        Returns a formatted string representation of the *LUT*.

        Returns
        -------
        unicode
            Formatted string representation.
        """

        def _indent_array(a):
            """
            Indents given array string representation.
            """

            return str(a).replace(' [', ' ' * 14 + '[')

        comments = [
            'Comment {0} : {1}'.format(str(i + 1).zfill(2), comment)
            for i, comment in enumerate(self.comments)
        ]

        return ('{0} - {1}\n'
                '{2}\n\n'
                'Dimensions : {3}\n'
                'Domain     : {4}\n'
                'Size       : {5!s}{6}').format(
                    self.__class__.__name__, self.name,
                    '-' * (len(self.__class__.__name__) + 3 + len(self.name)),
                    self.dimensions, _indent_array(self.domain),
                    str(self.table.shape).replace("L", ""), '\n{0}'.format(
                        '\n'.join(comments)) if comments else '')

    def __repr__(self):
        """
        Returns an evaluable string representation of the *LUT*.

        Returns
        -------
        unicode
            Evaluable string representation.
        """

        representation = repr(self.table)
        representation = representation.replace('array',
                                                self.__class__.__name__)
        representation = representation.replace('       [', '{0}['.format(
            ' ' * (len(self.__class__.__name__) + 2)))

        domain = repr(self.domain).replace('array(', '').replace(')', '')
        domain = domain.replace('       [', '{0}['.format(
            ' ' * (len(self.__class__.__name__) + 9)))

        indentation = ' ' * (len(self.__class__.__name__) + 1)
        representation = ('{0},\n'
                          '{1}name=\'{2}\',\n'
                          '{1}domain={3}{4})').format(
                              representation[:-1], indentation, self.name,
                              domain, ',\n{0}comments={1}'.format(
                                  indentation, repr(self.comments))
                              if self.comments else '')

        return representation

    def __eq__(self, other):
        """
        Returns whether the *LUT* is equal to given other object.

        Parameters
        ----------
        other : object
            Object to test whether it is equal to the *LUT*.

        Returns
        -------
        bool
            Is given object equal to the *LUT*.
        """

        if isinstance(other, AbstractLUT):
            if all([
                    np.array_equal(self.table, other.table),
                    np.array_equal(self.domain, other.domain)
            ]):
                return True

        return False

    def __ne__(self, other):
        """
        Returns whether the *LUT* is not equal to given other object.

        Parameters
        ----------
        other : object
            Object to test whether it is not equal to the *LUT*.

        Returns
        -------
        bool
            Is given object not equal to the *LUT*.
        """

        return not (self == other)

    def __add__(self, a):
        """
        Implements support for addition.

        Parameters
        ----------
        a : numeric or array_like or AbstractLUT
            :math:`a` variable to add.

        Returns
        -------
        AbstractLUT
            Variable added *LUT*.
        """

        return self.arithmetical_operation(a, '+')

    def __iadd__(self, a):
        """
        Implements support for in-place addition.

        Parameters
        ----------
        a : numeric or array_like or AbstractLUT
            :math:`a` variable to add in-place.

        Returns
        -------
        AbstractLUT
            In-place variable added *LUT*.
        """

        return self.arithmetical_operation(a, '+', True)

    def __sub__(self, a):
        """
        Implements support for subtraction.

        Parameters
        ----------
        a : numeric or array_like or AbstractLUT
            :math:`a` variable to subtract.

        Returns
        -------
        AbstractLUT
            Variable subtracted *LUT*.
        """

        return self.arithmetical_operation(a, '-')

    def __isub__(self, a):
        """
        Implements support for in-place subtraction.

        Parameters
        ----------
        a : numeric or array_like or AbstractLUT
            :math:`a` variable to subtract in-place.

        Returns
        -------
        AbstractLUT
            In-place variable subtracted *LUT*.
        """

        return self.arithmetical_operation(a, '-', True)

    def __mul__(self, a):
        """
        Implements support for multiplication.

        Parameters
        ----------
        a : numeric or array_like or AbstractLUT
            :math:`a` variable to multiply by.

        Returns
        -------
        AbstractLUT
            Variable multiplied *LUT*.
        """

        return self.arithmetical_operation(a, '*')

    def __imul__(self, a):
        """
        Implements support for in-place multiplication.

        Parameters
        ----------
        a : numeric or array_like or AbstractLUT
            :math:`a` variable to multiply by in-place.

        Returns
        -------
        AbstractLUT
            In-place variable multiplied *LUT*.
        """

        return self.arithmetical_operation(a, '*', True)

    def __div__(self, a):
        """
        Implements support for division.

        Parameters
        ----------
        a : numeric or array_like or AbstractLUT
            :math:`a` variable to divide by.

        Returns
        -------
        AbstractLUT
            Variable divided *LUT*.
        """

        return self.arithmetical_operation(a, '/')

    def __idiv__(self, a):
        """
        Implements support for in-place division.

        Parameters
        ----------
        a : numeric or array_like or AbstractLUT
            :math:`a` variable to divide by in-place.

        Returns
        -------
        AbstractLUT
            In-place variable divided *LUT*.
        """

        return self.arithmetical_operation(a, '/', True)

    __itruediv__ = __idiv__
    __truediv__ = __div__

    def __pow__(self, a):
        """
        Implements support for exponentiation.

        Parameters
        ----------
        a : numeric or array_like or AbstractLUT
            :math:`a` variable to exponentiate by.

        Returns
        -------
        AbstractLUT
            Variable exponentiated *LUT*.
        """

        return self.arithmetical_operation(a, '**')

    def __ipow__(self, a):
        """
        Implements support for in-place exponentiation.

        Parameters
        ----------
        a : numeric or array_like or AbstractLUT
            :math:`a` variable to exponentiate by in-place.

        Returns
        -------
        AbstractLUT
            In-place variable exponentiated *LUT*.
        """

        return self.arithmetical_operation(a, '**', True)

    def arithmetical_operation(self, a, operation, in_place=False):
        """
        Performs given arithmetical operation with :math:`a` operand, the
        operation can be either performed on a copy or in-place, must be
        reimplemented by sub-classes.

        Parameters
        ----------
        a : numeric or ndarray or AbstractLUT
            Operand.
        operation : object
            Operation to perform.
        in_place : bool, optional
            Operation happens in place.

        Returns
        -------
        AbstractLUT
            *LUT*.
        """

        operation, ioperator = {
            '+': (add, iadd),
            '-': (sub, isub),
            '*': (mul, imul),
            '/': (div, idiv),
            '**': (pow, ipow)
        }[operation]

        if in_place:
            if isinstance(a, AbstractLUT):
                operand = a.table
            else:
                operand = as_float_array(a)

            self.table = ioperator(self.table, operand)

            return self
        else:
            copy = ioperator(self.copy(), a)

            return copy

    @abstractmethod
    def _validate_table(table):
        """
        Validates given table according to *LUT* dimensions.

        Parameters
        ----------
        table : array_like
            Table to validate.

        Returns
        -------
        ndarray
            Validated table as a :class:`ndarray` instance.
        """

        pass

    @abstractmethod
    def _validate_domain(domain):
        """
        Validates given domain according to *LUT* dimensions.

        Parameters
        ----------
        domain : array_like
            Domain to validate.

        Returns
        -------
        ndarray
            Validated domain as a :class:`ndarray` instance.
        """

        pass

    @abstractmethod
    def linear_table(size=33, domain=None):
        """
        Returns a linear table of given size according to *LUT* dimensions.

        Parameters
        ----------
        size : int, optional
            Expected table size, for a 1D *LUT*, the number of output samples
            :math:`n` is equal to ``size``, for a 2D *LUT* :math:`n` is equal
            to ``size * 3``, for a 3D *LUT* :math:`n` is equal to
            ``size**3 * 3``.
        domain : array_like, optional
            Domain of the table.

        Returns
        -------
        ndarray
            Linear table.
        """

        pass

    @abstractmethod
    def apply(self, RGB, interpolator, interpolator_args):
        """
        Applies the *LUT* to given *RGB* colourspace array using given method.

        Parameters
        ----------
        RGB : array_like
            *RGB* colourspace array to apply the *LUT* onto.
        interpolator : object, optional
            Interpolator class type or object to use as interpolating function.
        interpolator_args : dict_like, optional
            Arguments to use when instantiating or calling the interpolating
            function.

        Returns
        -------
        ndarray
            Interpolated *RGB* colourspace array.
        """

        pass

    def copy(self):
        """
        Returns a copy of the sub-class instance.

        Returns
        -------
        AbstractLUT
            *LUT* copy.
        """

        return deepcopy(self)

    @abstractmethod
    def as_LUT(self, cls, force_conversion, **kwargs):
        """
        Converts the *LUT* to given ``cls`` class instance.

        Parameters
        ----------
        cls : LUT1D or LUT2D or LUT3D
            *LUT* class instance.
        force_conversion : bool, optional
            Whether to force the conversion as it might be destructive.

        Other Parameters
        ----------------
        interpolator : object, optional
            Interpolator class type to use as interpolating function.
        interpolator_args : dict_like, optional
            Arguments to use when instantiating the interpolating function.
        size : int, optional
            Expected table size in case of an upcast to or a downcast from a
            :class:`LUT3D` class instance.

        Returns
        -------
        LUT1D or LUT2D or LUT3D
            Converted *LUT* class instance.

        Warning
        -------
        Some conversions are destructive and raise a :class:`ValueError`
        exception by default.

        Raises
        ------
        ValueError
            If the conversion is destructive.
        """

        pass


class LUT1D(AbstractLUT):
    """
    Defines the base class for a 1D *LUT*.

    Parameters
    ----------
    table : array_like, optional
        Underlying *LUT* table.
    name : unicode, optional
        *LUT* name.
    domain : unicode, optional
        *LUT* domain, also used to define the instantiation time default table
        domain.
    size : int, optional
        Size of the instantiation time default table.
    comments : array_like, optional
        Comments to add to the *LUT*.

    Methods
    -------
    linear_table
    apply
    as_LUT

    Examples
    --------
    Instantiating a unity LUT with a table with 16 elements:

    >>> print(LUT1D(size=16))
    LUT1D - Unity 16
    ----------------
    <BLANKLINE>
    Dimensions : 1
    Domain     : [ 0.  1.]
    Size       : (16,)

    Instantiating a LUT using a custom table with 16 elements:

    >>> print(LUT1D(LUT1D.linear_table(16) ** (1 / 2.2)))  # doctest: +ELLIPSIS
    LUT1D - ...
    --------...
    <BLANKLINE>
    Dimensions : 1
    Domain     : [ 0.  1.]
    Size       : (16,)

    Instantiating a LUT using a custom table with 16 elements, custom name,
    custom domain and comments:

    >>> print(LUT1D(
    ...     LUT1D.linear_table(16) ** (1 / 2.2),
    ...     'My LUT',
    ...     np.array([-0.1, 1.5]),
    ...     comments=['A first comment.', 'A second comment.']))
    LUT1D - My LUT
    --------------
    <BLANKLINE>
    Dimensions : 1
    Domain     : [-0.1  1.5]
    Size       : (16,)
    Comment 01 : A first comment.
    Comment 02 : A second comment.
    """

    def __init__(self,
                 table=None,
                 name=None,
                 domain=None,
                 size=10,
                 comments=None):
        if domain is None:
            domain = np.array([0, 1])

        super(LUT1D, self).__init__(table, name, 1, domain, size, comments)

    # pylint: disable=W0221
    @staticmethod
    def _validate_table(table):
        """
        Validates given table is a 1D array.

        Parameters
        ----------
        table : array_like
            Table to validate.

        Returns
        -------
        ndarray
            Validated table as a :class:`ndarray` instance.
        """

        table = as_float_array(table)

        assert len(table.shape) == 1, 'The table must be a 1D array!'

        return table

    # pylint: disable=W0221
    @staticmethod
    def _validate_domain(domain):
        """
        Validates given domain shape is equal to (2, ).

        Parameters
        ----------
        domain : array_like
            Domain to validate.

        Returns
        -------
        ndarray
            Validated domain as a :class:`ndarray` instance.
        """

        domain = as_float_array(domain)

        assert domain.shape == (2, ), (
            'The domain shape must be equal to (2, )!')

        return domain

    # pylint: disable=W0221
    @staticmethod
    def linear_table(size=10, domain=None):
        """
        Returns a linear table, the number of output samples :math:`n` is equal
        to ``size``.

        Parameters
        ----------
        size : int, optional
            Expected table size.
        domain : array_like, optional
            Domain of the table.

        Returns
        -------
        ndarray
            Linear table with ``size`` samples.

        Examples
        --------
        >>> LUT1D.linear_table(5, np.array([-0.1, 1.5]))
        array([-0.1,  0.3,  0.7,  1.1,  1.5])
        """

        x, y = (0, 1) if domain is None else domain

        return np.linspace(x, y, size)

    def apply(self,
              RGB,
              interpolator=LinearInterpolator,
              interpolator_args=None):
        """
        Applies the *LUT* to given *RGB* colourspace array using given method.

        Parameters
        ----------
        RGB : array_like
            *RGB* colourspace array to apply the *LUT* onto.
        interpolator : object, optional
            Interpolator class type to use as interpolating function.
        interpolator_args : dict_like, optional
            Arguments to use when instantiating the interpolating function.

        Returns
        -------
        ndarray
            Interpolated *RGB* colourspace array.

        Examples
        --------
        >>> LUT = LUT1D(LUT1D.linear_table() ** (1 / 2.2))
        >>> RGB = np.array([0.18, 0.18, 0.18])
        >>> LUT.apply(RGB)  # doctest: +ELLIPSIS
        array([ 0.4529220...,  0.4529220...,  0.4529220...])
        """

        domain_min, domain_max = self.domain

        samples = np.linspace(domain_min, domain_max, self._table.size)

        RGB_interpolator = interpolator(samples, self._table)

        return RGB_interpolator(RGB)

    def as_LUT(self, cls, force_conversion=False, **kwargs):
        """
        Converts the *LUT* to given ``cls`` class instance.

        Parameters
        ----------
        cls : LUT1D or LUT2D or LUT3D
            *LUT* class instance.
        force_conversion : bool, optional
            Whether to force the conversion as it might be destructive.

        Other Parameters
        ----------------
        interpolator : object, optional
            Interpolator class type to use as interpolating function.
        interpolator_args : dict_like, optional
            Arguments to use when instantiating the interpolating function.
        size : int, optional
            Expected table size in case of an upcast to a :class:`LUT3D` class
            instance.

        Returns
        -------
        LUT1D or LUT2D or LUT3D
            Converted *LUT* class instance.

        Warning
        -------
        Some conversions are destructive and raise a :class:`ValueError`
        exception by default.

        Raises
        ------
        ValueError
            If the conversion is destructive.

        Examples
        --------
        >>> LUT = LUT1D()
        >>> print(LUT.as_LUT(LUT1D))
        LUT1D - Unity 10 - Converted 1D to 1D
        -------------------------------------
        <BLANKLINE>
        Dimensions : 1
        Domain     : [ 0.  1.]
        Size       : (10,)
        >>> print(LUT.as_LUT(LUT2D))
        LUT2D - Unity 10 - Converted 1D to 2D
        -------------------------------------
        <BLANKLINE>
        Dimensions : 2
        Domain     : [[ 0.  0.  0.]
                      [ 1.  1.  1.]]
        Size       : (10, 3)
        >>> print(LUT.as_LUT(LUT3D, force_conversion=True))
        LUT3D - Unity 10 - Converted 1D to 3D
        -------------------------------------
        <BLANKLINE>
        Dimensions : 3
        Domain     : [[ 0.  0.  0.]
                      [ 1.  1.  1.]]
        Size       : (33, 33, 33, 3)
        """

        return LUT_to_LUT(self, cls, force_conversion, **kwargs)


class LUT2D(AbstractLUT):
    """
    Defines the base class for a 2D *LUT*.

    Parameters
    ----------
    table : array_like, optional
        Underlying *LUT* table.
    name : unicode, optional
        *LUT* name.
    domain : unicode, optional
        *LUT* domain, also used to define the instantiation time default table
        domain.
    size : int, optional
        Size of the instantiation time default table.
    comments : array_like, optional
        Comments to add to the *LUT*.

    Methods
    -------
    linear_table
    apply
    as_LUT

    Examples
    --------
    Instantiating a unity LUT with a table with 16x3 elements:

    >>> print(LUT2D(size=16))
    LUT2D - Unity 16
    ----------------
    <BLANKLINE>
    Dimensions : 2
    Domain     : [[ 0.  0.  0.]
                  [ 1.  1.  1.]]
    Size       : (16, 3)

    Instantiating a LUT using a custom table with 16x3 elements:

    >>> print(LUT2D(LUT2D.linear_table(16) ** (1 / 2.2)))  # doctest: +ELLIPSIS
    LUT2D - ...
    --------...
    <BLANKLINE>
    Dimensions : 2
    Domain     : [[ 0.  0.  0.]
                  [ 1.  1.  1.]]
    Size       : (16, 3)

    Instantiating a LUT using a custom table with 16x3 elements, custom name,
    custom domain and comments:

    >>> print(LUT2D(
    ...     LUT2D.linear_table(16) ** (1 / 2.2),
    ...     'My LUT',
    ...     np.array([[-0.1, -0.2, -0.4], [1.5, 3.0, 6.0]]),
    ...     comments=['A first comment.', 'A second comment.']))
    LUT2D - My LUT
    --------------
    <BLANKLINE>
    Dimensions : 2
    Domain     : [[-0.1 -0.2 -0.4]
                  [ 1.5  3.   6. ]]
    Size       : (16, 3)
    Comment 01 : A first comment.
    Comment 02 : A second comment.
    """

    def __init__(self,
                 table=None,
                 name=None,
                 domain=None,
                 size=10,
                 comments=None):
        if domain is None:
            domain = np.array([[0, 0, 0], [1, 1, 1]])

        super(LUT2D, self).__init__(table, name, 2, domain, size, comments)

    # pylint: disable=W0221
    @staticmethod
    def _validate_table(table):
        """
        Validates given table is a 2D array.

        Parameters
        ----------
        table : array_like
            Table to validate.

        Returns
        -------
        ndarray
            Validated table as a :class:`ndarray` instance.
        """

        table = as_float_array(table)

        assert len(table.shape) == 2, 'The table must be a 2D array!'

        return table

    # pylint: disable=W0221
    @staticmethod
    def _validate_domain(domain):
        """
        Validates given domain shape is equal to (2, 3).

        Parameters
        ----------
        domain : array_like
            Domain to validate.

        Returns
        -------
        ndarray
            Validated domain as a :class:`ndarray` instance.
        """

        domain = as_float_array(domain)

        assert domain.shape == (2, 3), (
            'The domain shape must be equal to (2, 3)!')

        return domain

    # pylint: disable=W0221
    @staticmethod
    def linear_table(size=10, domain=None):
        """
        Returns a linear table, the number of output samples :math:`n` is equal
        to ``size * 3``.

        Parameters
        ----------
        size : int, optional
            Expected table size.
        domain : array_like, optional
            Domain of the table.

        Returns
        -------
        ndarray
            Linear table with ``size * 3`` samples.

        Examples
        --------
        >>> LUT2D.linear_table(
        ...     5, np.array([[-0.1, -0.2, -0.4], [1.5, 3.0, 6.0]]))
        array([[-0.1, -0.2, -0.4],
               [ 0.3,  0.6,  1.2],
               [ 0.7,  1.4,  2.8],
               [ 1.1,  2.2,  4.4],
               [ 1.5,  3. ,  6. ]])
        """

        if domain is None:
            R = G = B = [0, 1]
        else:
            R, G, B = tsplit(domain)

        samples = [np.linspace(a[0], a[1], size) for a in (R, G, B)]

        return tstack(samples)

    def apply(self,
              RGB,
              interpolator=LinearInterpolator,
              interpolator_args=None):
        """
        Applies the *LUT* to given *RGB* colourspace array using given method.

        Parameters
        ----------
        RGB : array_like
            *RGB* colourspace array to apply the *LUT* onto.
        interpolator : object, optional
            Interpolator class type to use as interpolating function.
        interpolator_args : dict_like, optional
            Arguments to use when instantiating the interpolating function.

        Returns
        -------
        ndarray
            Interpolated *RGB* colourspace array.

        Examples
        --------
        >>> LUT = LUT2D(LUT2D.linear_table() ** (1 / 2.2))
        >>> RGB = np.array([0.18, 0.18, 0.18])
        >>> LUT.apply(RGB)  # doctest: +ELLIPSIS
        array([ 0.4529220...,  0.4529220...,  0.4529220...])
        """

        R, G, B = tsplit(RGB)
        R_t, G_t, B_t = tsplit(self._table)
        domain_min, domain_max = self.domain

        size = DEFAULT_INT_DTYPE(self._table.size / 3)

        RGB_i = [
            interpolator(
                np.linspace(domain_min[i], domain_max[i], size), j[1])(j[0])
            for i, j in enumerate([(R, R_t), (G, G_t), (B, B_t)])
        ]

        return tstack(RGB_i)

    def as_LUT(self, cls, force_conversion=False, **kwargs):
        """
        Converts the *LUT* to given ``cls`` class instance.

        Parameters
        ----------
        cls : LUT1D or LUT2D or LUT3D
            *LUT* class instance.
        force_conversion : bool, optional
            Whether to force the conversion as it might be destructive.

        Other Parameters
        ----------------
        interpolator : object, optional
            Interpolator class type to use as interpolating function.
        interpolator_args : dict_like, optional
            Arguments to use when instantiating the interpolating function.
        size : int, optional
            Expected table size in case of an upcast to a :class:`LUT3D` class
            instance.

        Returns
        -------
        LUT1D or LUT2D or LUT3D
            Converted *LUT* class instance.

        Warning
        -------
        Some conversions are destructive and raise a :class:`ValueError`
        exception by default.

        Raises
        ------
        ValueError
            If the conversion is destructive.

        Examples
        --------
        >>> LUT = LUT2D()
        >>> print(LUT.as_LUT(LUT1D, force_conversion=True))
        LUT1D - Unity 10 - Converted 2D to 1D
        -------------------------------------
        <BLANKLINE>
        Dimensions : 1
        Domain     : [ 0.  1.]
        Size       : (10,)
        >>> print(LUT.as_LUT(LUT2D))
        LUT2D - Unity 10 - Converted 2D to 2D
        -------------------------------------
        <BLANKLINE>
        Dimensions : 2
        Domain     : [[ 0.  0.  0.]
                      [ 1.  1.  1.]]
        Size       : (10, 3)
        >>> print(LUT.as_LUT(LUT3D, force_conversion=True))
        LUT3D - Unity 10 - Converted 2D to 3D
        -------------------------------------
        <BLANKLINE>
        Dimensions : 3
        Domain     : [[ 0.  0.  0.]
                      [ 1.  1.  1.]]
        Size       : (33, 33, 33, 3)
        """

        return LUT_to_LUT(self, cls, force_conversion, **kwargs)


class LUT3D(AbstractLUT):
    """
    Defines the base class for a 3D *LUT*.

    Parameters
    ----------
    table : array_like, optional
        Underlying *LUT* table.
    name : unicode, optional
        *LUT* name.
    domain : unicode, optional
        *LUT* domain, also used to define the instantiation time default table
        domain.
    size : int, optional
        Size of the instantiation time default table.
    comments : array_like, optional
        Comments to add to the *LUT*.

    Methods
    -------
    linear_table
    apply
    as_LUT

    Examples
    --------
    Instantiating a unity LUT with a table with 16x16x16x3 elements:

    >>> print(LUT3D(size=16))
    LUT3D - Unity 16
    ----------------
    <BLANKLINE>
    Dimensions : 3
    Domain     : [[ 0.  0.  0.]
                  [ 1.  1.  1.]]
    Size       : (16, 16, 16, 3)

    Instantiating a LUT using a custom table with 16x16x16x3 elements:

    >>> print(LUT3D(LUT3D.linear_table(16) ** (1 / 2.2)))  # doctest: +ELLIPSIS
    LUT3D - ...
    --------...
    <BLANKLINE>
    Dimensions : 3
    Domain     : [[ 0.  0.  0.]
                  [ 1.  1.  1.]]
    Size       : (16, 16, 16, 3)

    Instantiating a LUT using a custom table with 16x16x16x3 elements, custom
    name, custom domain and comments:

    >>> print(LUT3D(
    ...     LUT3D.linear_table(16) ** (1 / 2.2),
    ...     'My LUT',
    ...     np.array([[-0.1, -0.2, -0.4], [1.5, 3.0, 6.0]]),
    ...     comments=['A first comment.', 'A second comment.']))
    LUT3D - My LUT
    --------------
    <BLANKLINE>
    Dimensions : 3
    Domain     : [[-0.1 -0.2 -0.4]
                  [ 1.5  3.   6. ]]
    Size       : (16, 16, 16, 3)
    Comment 01 : A first comment.
    Comment 02 : A second comment.
    """

    def __init__(self,
                 table=None,
                 name=None,
                 domain=None,
                 size=33,
                 comments=None):
        if domain is None:
            domain = np.array([[0, 0, 0], [1, 1, 1]])

        super(LUT3D, self).__init__(table, name, 3, domain, size, comments)

    # pylint: disable=W0221
    @staticmethod
    def _validate_table(table):
        """
        Validates given table is a 4D array and that its dimensions are equal.

        Parameters
        ----------
        table : array_like
            Table to validate.

        Returns
        -------
        ndarray
            Validated table as a :class:`ndarray` instance.
        """

        table = as_float_array(table)

        assert len(table.shape) == 4, 'The table must be a 4D array!'
        assert len(set(
            table.shape[:-1])) == 1, 'The table dimensions must be equal!'

        return table

    # pylint: disable=W0221
    @staticmethod
    def _validate_domain(domain):
        """
        Validates given domain is equal to (2, 3).

        Parameters
        ----------
        domain : array_like
            Domain to validate.

        Returns
        -------
        ndarray
            Validated domain as a :class:`ndarray` instance.
        """

        domain = as_float_array(domain)

        assert domain.shape == (2, 3), (
            'The domain shape must be equal to (2, 3)!')

        return domain

    # pylint: disable=W0221
    @staticmethod
    def linear_table(size=33, domain=None):
        """
        Returns a linear table, the number of output samples :math:`n` is equal
        to ``size**3 * 3``.

        Parameters
        ----------
        size : int, optional
            Expected table size.
        domain : array_like, optional
            Domain of the table.

        Returns
        -------
        ndarray
            Linear table with ``size**3 * 3`` samples.

        Examples
        --------
        >>> LUT3D.linear_table(
        ...     3, np.array([[-0.1, -0.2, -0.4], [1.5, 3.0, 6.0]]))
        array([[[[-0.1, -0.2, -0.4],
                 [-0.1, -0.2,  2.8],
                 [-0.1, -0.2,  6. ]],
        <BLANKLINE>
                [[-0.1,  1.4, -0.4],
                 [-0.1,  1.4,  2.8],
                 [-0.1,  1.4,  6. ]],
        <BLANKLINE>
                [[-0.1,  3. , -0.4],
                 [-0.1,  3. ,  2.8],
                 [-0.1,  3. ,  6. ]]],
        <BLANKLINE>
        <BLANKLINE>
               [[[ 0.7, -0.2, -0.4],
                 [ 0.7, -0.2,  2.8],
                 [ 0.7, -0.2,  6. ]],
        <BLANKLINE>
                [[ 0.7,  1.4, -0.4],
                 [ 0.7,  1.4,  2.8],
                 [ 0.7,  1.4,  6. ]],
        <BLANKLINE>
                [[ 0.7,  3. , -0.4],
                 [ 0.7,  3. ,  2.8],
                 [ 0.7,  3. ,  6. ]]],
        <BLANKLINE>
        <BLANKLINE>
               [[[ 1.5, -0.2, -0.4],
                 [ 1.5, -0.2,  2.8],
                 [ 1.5, -0.2,  6. ]],
        <BLANKLINE>
                [[ 1.5,  1.4, -0.4],
                 [ 1.5,  1.4,  2.8],
                 [ 1.5,  1.4,  6. ]],
        <BLANKLINE>
                [[ 1.5,  3. , -0.4],
                 [ 1.5,  3. ,  2.8],
                 [ 1.5,  3. ,  6. ]]]])
        """

        if domain is None:
            R = G = B = [0, 1]
        else:
            R, G, B = tsplit(domain)

        samples = [np.linspace(a[0], a[1], size) for a in (B, G, R)]
        table = np.meshgrid(*samples, indexing='ij')
        table = np.transpose(table).reshape([size, size, size, 3])

        return np.flip(table, -1)

    def apply(self,
              RGB,
              interpolator=table_interpolation_trilinear,
              interpolator_args=None):
        """
        Applies the *LUT* to given *RGB* colourspace array using given method.

        Parameters
        ----------
        RGB : array_like
            *RGB* colourspace array to apply the *LUT* onto.
        interpolator : object, optional
            Interpolator object to use as interpolating function.
        interpolator_args : dict_like, optional
            Arguments to use when calling the interpolating function.

        Returns
        -------
        ndarray
            Interpolated *RGB* colourspace array.

        Examples
        --------
        >>> LUT = LUT3D(LUT3D.linear_table() ** (1 / 2.2))
        >>> RGB = np.array([0.18, 0.18, 0.18])
        >>> LUT.apply(RGB)  # doctest: +ELLIPSIS
        array([ 0.4583277...,  0.4583277...,  0.4583277...])
        """

        R, G, B = tsplit(RGB)
        domain_min, domain_max = self.domain

        RGB_l = [
            linear_conversion(j, (domain_min[i], domain_max[i]), (0, 1))
            for i, j in enumerate((R, G, B))
        ]

        return interpolator(tstack(RGB_l), self._table)

    def as_LUT(self, cls, force_conversion=False, **kwargs):
        """
        Converts the *LUT* to given ``cls`` class instance.

        Parameters
        ----------
        cls : LUT1D or LUT2D or LUT3D
            *LUT* class instance.
        force_conversion : bool, optional
            Whether to force the conversion as it might be destructive.

        Other Parameters
        ----------------
        interpolator : object, optional
            Interpolator class type to use as interpolating function.
        interpolator_args : dict_like, optional
            Arguments to use when instantiating the interpolating function.
        size : int, optional
            Expected table size in case of a downcast from a :class:`LUT3D`
            class instance.

        Returns
        -------
        LUT1D or LUT2D or LUT3D
            Converted *LUT* class instance.

        Warning
        -------
        Some conversions are destructive and raise a :class:`ValueError`
        exception by default.

        Raises
        ------
        ValueError
            If the conversion is destructive.

        Examples
        --------
        >>> LUT = LUT3D()
        >>> print(LUT.as_LUT(LUT1D, force_conversion=True))
        LUT1D - Unity 33 - Converted 3D to 1D
        -------------------------------------
        <BLANKLINE>
        Dimensions : 1
        Domain     : [ 0.  1.]
        Size       : (10,)
        >>> print(LUT.as_LUT(LUT2D, force_conversion=True))
        LUT2D - Unity 33 - Converted 3D to 2D
        -------------------------------------
        <BLANKLINE>
        Dimensions : 2
        Domain     : [[ 0.  0.  0.]
                      [ 1.  1.  1.]]
        Size       : (10, 3)
        >>> print(LUT.as_LUT(LUT3D))
        LUT3D - Unity 33 - Converted 3D to 3D
        -------------------------------------
        <BLANKLINE>
        Dimensions : 3
        Domain     : [[ 0.  0.  0.]
                      [ 1.  1.  1.]]
        Size       : (33, 33, 33, 3)
        """

        return LUT_to_LUT(self, cls, force_conversion, **kwargs)


def LUT_to_LUT(LUT, cls, force_conversion=False, **kwargs):
    """
    Converts given *LUT* to given ``cls`` class instance.

    Parameters
    ----------
    cls : LUT1D or LUT2D or LUT3D
        *LUT* class instance.
    force_conversion : bool, optional
        Whether to force the conversion as it might be destructive.

    Other Parameters
    ----------------
    interpolator : object, optional
        Interpolator class type to use as interpolating function.
    interpolator_args : dict_like, optional
        Arguments to use when instantiating the interpolating function.
    size : int, optional
        Expected table size in case of an upcast to or a downcast from a
        :class:`LUT3D` class instance.
    channel_weights : array_like, optional
        Channel weights in case of a downcast from a :class:`LUT2D` or
        :class:`LUT3D` class instance.

    Returns
    -------
    LUT1D or LUT2D or LUT3D
        Converted *LUT* class instance.

    Warning
    -------
    Some conversions are destructive and raise a :class:`ValueError` exception
    by default.

    Raises
    ------
    ValueError
        If the conversion is destructive.

    Examples
    --------
    >>> print(LUT_to_LUT(LUT1D(), LUT3D, force_conversion=True))
    LUT3D - Unity 10 - Converted 1D to 3D
    -------------------------------------
    <BLANKLINE>
    Dimensions : 3
    Domain     : [[ 0.  0.  0.]
                  [ 1.  1.  1.]]
    Size       : (33, 33, 33, 3)
    >>> print(LUT_to_LUT(LUT2D(), LUT1D, force_conversion=True))
    LUT1D - Unity 10 - Converted 2D to 1D
    -------------------------------------
    <BLANKLINE>
    Dimensions : 1
    Domain     : [ 0.  1.]
    Size       : (10,)
    >>> print(LUT_to_LUT(LUT3D(), LUT1D, force_conversion=True))
    LUT1D - Unity 33 - Converted 3D to 1D
    -------------------------------------
    <BLANKLINE>
    Dimensions : 1
    Domain     : [ 0.  1.]
    Size       : (10,)
    """

    ranks = {LUT1D: 1, LUT2D: 2, LUT3D: 3}
    path = (ranks[LUT.__class__], ranks[cls])
    if path in ((1, 3), (2, 1), (2, 3), (3, 1), (3, 2)):
        if not force_conversion:
            raise ValueError(
                'Conversion of a "LUT" {0}D to a "LUT" {1}D is destructive, '
                'please use the "force_conversion" argument to proceed.'.
                format(*path))

    suffix = ' - Converted {0}D to {1}D'.format(*path)
    name = '{0}{1}'.format(LUT.name, suffix)

    # Same dimension conversion, returning a copy.
    if len(set(path)) == 1:
        LUT = LUT.copy()
        LUT.name = name
    else:
        size = kwargs.get('size', 33 if cls is LUT3D else 10)
        if 'size' in kwargs:
            del kwargs['size']

        channel_weights = as_float_array(
            kwargs.get('channel_weights', np.full(3, 1 / 3)))
        if 'channel_weights' in kwargs:
            del kwargs['channel_weights']

        if isinstance(LUT, LUT1D):
            if cls is LUT2D:
                domain = tstack([LUT.domain, LUT.domain, LUT.domain])
                table = tstack([LUT.table, LUT.table, LUT.table])
            elif cls is LUT3D:
                domain = tstack([LUT.domain, LUT.domain, LUT.domain])
                table = LUT3D.linear_table(size, domain)
                table = LUT.apply(table, **kwargs)
        elif isinstance(LUT, LUT2D):
            if cls is LUT1D:
                domain = np.array(
                    [np.max(LUT.domain[0, ...]),
                     np.min(LUT.domain[1, ...])])
                table = np.sum(LUT.table * channel_weights, axis=-1)
            elif cls is LUT3D:
                domain = LUT.domain
                table = LUT3D.linear_table(size, domain)
                table = LUT.apply(table, **kwargs)
        elif isinstance(LUT, LUT3D):
            if cls is LUT1D:
                domain = np.array(
                    [np.max(LUT.domain[0, ...]),
                     np.min(LUT.domain[1, ...])])
                table = LUT1D.linear_table(size, domain)
                table = LUT.apply(tstack([table, table, table]), **kwargs)
                table = np.sum(table * channel_weights, axis=-1)
            elif cls is LUT2D:
                domain = LUT.domain
                table = LUT2D.linear_table(size, domain)
                table = LUT.apply(table, **kwargs)

        LUT = cls(table, name, domain, table.shape[0], LUT.comments)

    return LUT


@add_metaclass(ABCMeta)
class AbstractLUTSequenceOperator:
    """
    Defines the base class for *LUT* sequence operators.

    This is an :class:`ABCMeta` abstract class that must be inherited by
    sub-classes.

    Methods
    -------
    apply
    """

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


class LUTSequence(MutableSequence):
    """
    Defines the base class for a *LUT* sequence, i.e., a series of *LUTs*.

    The `colour.LUTSequence` class can be used to model series of *LUTs* such
    as when a shaper *LUT* is combined with a 3D *LUT*.

    Other Parameters
    ----------------
    \\*args : list, optional
        Sequence of `colour.LUT1D`, `colour.LUT2D`, `colour.LUT3D` or
        `colour.io.lut.l.AbstractLUTSequenceOperator` class instances.

    Attributes
    ----------
    sequence

    Methods
    -------
    __getitem__
    __setitem__
    __delitem__
    __len__
    __str__
    __repr__
    __eq__
    __ne__
    insert
    apply
    copy

    Examples
    --------
    >>> LUT_1 = LUT1D()
    >>> LUT_2 = LUT3D(size=3)
    >>> LUT_3 = LUT2D()
    >>> print(LUTSequence(LUT_1, LUT_2, LUT_3))
    LUT Sequence
    ------------
    <BLANKLINE>
    Overview
    <BLANKLINE>
        LUT1D ---> LUT3D ---> LUT2D
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
        LUT2D - Unity 10
        ----------------
    <BLANKLINE>
        Dimensions : 2
        Domain     : [[ 0.  0.  0.]
                      [ 1.  1.  1.]]
        Size       : (10, 3)
    """

    def __init__(self, *args):
        for arg in args:
            assert isinstance(
                arg, (LUT1D, LUT2D, LUT3D, AbstractLUTSequenceOperator)
            ), ('"args" elements must be instances of '
                '"LUT1D", "LUT2D",  "LUT3D" or "AbstractLUTSequenceOperator"!')

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
        LUT1D or LUT2D or LUT3D or AbstractLUTSequenceOperator
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
        value : LUT1D or LUT2D or LUT3D or AbstractLUTSequenceOperator
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
        unicode
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
                '{1}').format(' ---> '.join(
                    [a.__class__.__name__ for a in self._sequence]),
                              operations)

    def __repr__(self):
        """
        Returns an evaluable string representation of the *LUT* sequence.

        Returns
        -------
        unicode
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

    def insert(self, index, LUT):
        """
        Inserts given *LUT* at given index into the *LUT* sequence.

        Parameters
        ----------
        index : index
            Index to insert the *LUT* at into the *LUT* sequence.
        LUT : LUT1D or LUT2D or LUT3D or AbstractLUTSequenceOperator
            *LUT* to insert into the *LUT* sequence.
        """

        assert isinstance(
            LUT, (LUT1D, LUT2D, LUT3D, AbstractLUTSequenceOperator)), (
                '"LUT" must be an instance of "LUT1D", "LUT2D", "LUT3D" or '
                '"AbstractLUTSequenceOperator"!')

        self._sequence.insert(index, LUT)

    def apply(self,
              RGB,
              interpolator_1D=LinearInterpolator,
              interpolator_1D_args=None,
              interpolator_3D=table_interpolation_trilinear,
              interpolator_3D_args=None):
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
            :class:`colour.LUT1D` (and :class:`colour.LUT2D`) class instances.
        interpolator_1D_args : dict_like, optional
            Arguments to use when calling the interpolating function for
            :class:`colour.LUT1D` (and :class:`colour.LUT2D`) class instances.
        interpolator_3D : object, optional
            Interpolator object to use as interpolating function for
            :class:`colour.LUT3D` class instances.
        interpolator_3D_args : dict_like, optional
            Arguments to use when calling the interpolating function for
            :class:`colour.LUT3D` class instances.

        Returns
        -------
        ndarray
            Processed *RGB* colourspace array.

        Examples
        --------
        >>> LUT_1 = LUT1D(LUT1D.linear_table(16) + 0.125)
        >>> LUT_2 = LUT3D(LUT3D.linear_table(16) ** (1 / 2.2))
        >>> LUT_3 = LUT2D(LUT2D.linear_table(16) * 0.750)
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
            if isinstance(operation, (LUT1D, LUT2D)):
                RGB = operation.apply(RGB, interpolator_1D,
                                      interpolator_1D_args)
            elif isinstance(operation, LUT3D):
                RGB = operation.apply(RGB, interpolator_3D,
                                      interpolator_3D_args)
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
