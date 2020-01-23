# -*- coding: utf-8 -*-
"""
Abstract Continuous Function
============================

Defines the abstract class implementing support for abstract continuous
function:

-   :class:`colour.continuous.AbstractContinuousFunction`.
"""

from __future__ import division, unicode_literals

import numpy as np
from abc import ABCMeta, abstractmethod, abstractproperty
from copy import deepcopy

# Python 3 compatibility.
try:
    from operator import div, idiv
except ImportError:
    from operator import truediv, itruediv

    div = truediv
    idiv = itruediv
from six import add_metaclass

from colour.utilities import as_float, closest, is_uniform, is_string

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['AbstractContinuousFunction']


@add_metaclass(ABCMeta)
class AbstractContinuousFunction:
    """
    Defines the base class for abstract continuous function.

    This is an :class:`ABCMeta` abstract class that must be inherited by
    sub-classes.

    The sub-classes are expected to implement the
    :meth:`colour.continuous.AbstractContinuousFunction.function` method so
    that evaluating the function for any independent domain
    :math:`x \\in\\mathbb{R}` variable returns a corresponding range
    :math:`y \\in\\mathbb{R}` variable. A conventional implementation adopts an
    interpolating function encapsulated inside an extrapolating function.
    The resulting function independent domain, stored as discrete values in the
    :attr:`colour.continuous.AbstractContinuousFunction.domain` attribute
    corresponds with the function dependent and already known range stored in
    the :attr:`colour.continuous.AbstractContinuousFunction.range` attribute.

    Parameters
    ----------
    name : unicode, optional
        Continuous function name.

    Attributes
    ----------
    name
    domain
    range
    interpolator
    interpolator_args
    extrapolator
    extrapolator_args
    function

    Methods
    -------
    __str__
    __repr__
    __hash__
    __getitem__
    __setitem__
    __contains__
    __len__
    __eq__
    __ne__
    __iadd__
    __add__
    __isub__
    __sub__
    __imul__
    __mul__
    __idiv__
    __div__
    __ipow__
    __pow__
    arithmetical_operation
    fill_nan
    domain_distance
    is_uniform
    copy
    """

    def __init__(self, name=None):
        self._name = '{0} ({1})'.format(self.__class__.__name__, id(self))
        self.name = name

    @property
    def name(self):
        """
        Getter and setter property for the abstract continuous function name.

        Parameters
        ----------
        value : unicode
            Value to set the abstract continuous function name with.

        Returns
        -------
        unicode
            Abstract continuous function name.
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

    def _get_dtype(self):
        """
        Getter and setter property for the abstract continuous function dtype,
        must be reimplemented by sub-classes.

        Parameters
        ----------
        value : type
            Value to set the abstract continuous function dtype with.

        Returns
        -------
        type
            Abstract continuous function dtype.
        """

        pass

    def _set_dtype(self, value):
        """
        Setter for **self.dtype** property, must be reimplemented by
        sub-classes.
        """

        pass

    domain = abstractproperty(_get_dtype, _set_dtype)

    def _get_domain(self):
        """
        Getter and setter property for the abstract continuous function
        independent domain :math:`x` variable, must be reimplemented by
        sub-classes.

        Parameters
        ----------
        value : array_like
            Value to set the abstract continuous function independent domain
            :math:`x` variable with.

        Returns
        -------
        ndarray
            Abstract continuous function independent domain
            :math:`x` variable.
        """

        pass

    def _set_domain(self, value):
        """
        Setter for the **self.domain** property, must be reimplemented by
        sub-classes.
        """

        pass

    domain = abstractproperty(_get_domain, _set_domain)

    def _get_range(self):
        """
        Getter and setter property for the abstract continuous function
        corresponding range :math:`y` variable,
        must be reimplemented by sub-classes.

        Parameters
        ----------
        value : array_like
            Value to set the abstract continuous function corresponding range
            :math:`y` variable with.

        Returns
        -------
        ndarray
            Abstract continuous function corresponding range
            :math:`y` variable.
        """

        pass

    def _set_range(self, value):
        """
        Setter for the **self.range** property, must be reimplemented by
        sub-classes.
        """

        pass

    range = abstractproperty(_get_range, _set_range)

    def _get_interpolator(self):
        """
        Getter and setter property for the abstract continuous function
        interpolator type, must be reimplemented by sub-classes.

        Parameters
        ----------
        value : type
            Value to set the abstract continuous function interpolator type
            with.

        Returns
        -------
        type
            Abstract continuous function interpolator type.
        """

        pass

    def _set_interpolator(self, value):
        """
        Setter for the **self.interpolator** property, must be reimplemented by
        sub-classes.
        """

        pass

    interpolator = abstractproperty(_get_interpolator, _set_interpolator)

    def _get_interpolator_args(self):
        """
        Getter and setter property for the abstract continuous function
        interpolator instantiation time arguments, must be reimplemented by
        sub-classes.

        Parameters
        ----------
        value : dict
            Value to set the abstract continuous function interpolator
            instantiation time arguments to.

        Returns
        -------
        dict
            Abstract continuous function interpolator instantiation time
            arguments.
        """

        pass

    def _set_interpolator_args(self, value):
        """
        Setter for the **self.interpolator_args** property, must be
        reimplemented by sub-classes.
        """

        pass

    interpolator_args = abstractproperty(_get_interpolator_args,
                                         _set_interpolator_args)

    def _get_extrapolator(self):
        """
        Getter and setter property for the abstract continuous function
        extrapolator type, must be reimplemented by sub-classes.

        Parameters
        ----------
        value : type
            Value to set the abstract continuous function extrapolator type
            with.

        Returns
        -------
        type
            Abstract continuous function extrapolator type.
        """

        pass

    def _set_extrapolator(self, value):
        """
        Setter for the **self.extrapolator** property, must be reimplemented by
        sub-classes.
        """

        pass

    extrapolator = abstractproperty(_get_extrapolator, _set_extrapolator)

    def _get_extrapolator_args(self):
        """
        Getter and setter property for the abstract continuous function
        extrapolator instantiation time arguments, must be reimplemented by
        sub-classes.

        Parameters
        ----------
        value : dict
            Value to set the abstract continuous function extrapolator
            instantiation time arguments to.

        Returns
        -------
        dict
            Abstract continuous function extrapolator instantiation time
            arguments.
        """

        pass

    def _set_extrapolator_args(self, value):
        """
        Setter for the **self.extrapolator_args** property, must be
        reimplemented by sub-classes.
        """

        pass

    extrapolator_args = abstractproperty(_get_extrapolator_args,
                                         _set_extrapolator_args)

    def _get_function(self):
        """
        Getter and setter property for the abstract continuous function
        callable, must be reimplemented by sub-classes.

        Parameters
        ----------
        value : object
            Attribute value.

        Returns
        -------
        callable
            Abstract continuous function callable.

        Notes
        -----
        -   This property is read only.
        """

        pass

    def _set_function(self, value):
        """
        Setter for the **self.function** property, must be reimplemented by
        sub-classes.
        """

        pass

    function = abstractproperty(_get_function, _set_function)

    @abstractmethod
    def __str__(self):
        """
        Returns a formatted string representation of the abstract continuous
        function, must be reimplemented by sub-classes.

        Returns
        -------
        unicode
            Formatted string representation.
        """

        return '<{0} object at {1}>'.format(self.__class__.__name__, id(self))

    @abstractmethod
    def __repr__(self):
        """
        Returns an evaluable string representation of the abstract continuous
        function, must be reimplemented by sub-classes.

        Returns
        -------
        unicode
            Evaluable string representation.
        """

        return '{0}()'.format(self.__class__.__name__)

    @abstractmethod
    def __hash__(self):
        """
        Returns the abstract continuous function hash.

        Returns
        -------
        int
            Object hash.
        """

        pass

    @abstractmethod
    def __getitem__(self, x):
        """
        Returns the corresponding range :math:`y` variable for independent
        domain :math:`x` variable, must be reimplemented by sub-classes.

        Parameters
        ----------
        x : numeric, array_like or slice
            Independent domain :math:`x` variable.

        Returns
        -------
        numeric or ndarray
            math:`y` range value.
        """

        pass

    @abstractmethod
    def __setitem__(self, x, y):
        """
        Sets the corresponding range :math:`y` variable for independent domain
        :math:`x` variable, must be reimplemented by sub-classes.

        Parameters
        ----------
        x : numeric, array_like or slice
            Independent domain :math:`x` variable.
        y : numeric or ndarray
            Corresponding range :math:`y` variable.
        """

        pass

    @abstractmethod
    def __contains__(self, x):
        """
        Returns whether the abstract continuous function contains given
        independent domain :math:`x` variable, must be reimplemented by
        sub-classes.

        Parameters
        ----------
        x : numeric, array_like or slice
            Independent domain :math:`x` variable.

        Returns
        -------
        bool
            Is :math:`x` domain value contained.
        """

        pass

    def __len__(self):
        """
        Returns the abstract continuous function independent domain :math:`x`
        variable elements count.


        Returns
        -------
        int
            Independent domain :math:`x` variable elements count.
        """

        return len(self.domain)

    @abstractmethod
    def __eq__(self, other):
        """
        Returns whether the abstract continuous function is equal to given
        other object, must be reimplemented by sub-classes.

        Parameters
        ----------
        other : object
            Object to test whether it is equal to the abstract continuous
            function.

        Returns
        -------
        bool
            Is given object equal to the abstract continuous function.
        """

        pass

    @abstractmethod
    def __ne__(self, other):
        """
        Returns whether the abstract continuous function is not equal to given
        other object, must be reimplemented by sub-classes.

        Parameters
        ----------
        other : object
            Object to test whether it is not equal to the abstract continuous
            function.

        Returns
        -------
        bool
            Is given object not equal to the abstract continuous function.
        """

        pass

    def __add__(self, a):
        """
        Implements support for addition.

        Parameters
        ----------
        a : numeric or array_like or AbstractContinuousFunction
            :math:`a` variable to add.

        Returns
        -------
        AbstractContinuousFunction
            Variable added abstract continuous function.
        """

        return self.arithmetical_operation(a, '+')

    def __iadd__(self, a):
        """
        Implements support for in-place addition.

        Parameters
        ----------
        a : numeric or array_like or AbstractContinuousFunction
            :math:`a` variable to add in-place.

        Returns
        -------
        AbstractContinuousFunction
            In-place variable added abstract continuous function.
        """

        return self.arithmetical_operation(a, '+', True)

    def __sub__(self, a):
        """
        Implements support for subtraction.

        Parameters
        ----------
        a : numeric or array_like or AbstractContinuousFunction
            :math:`a` variable to subtract.

        Returns
        -------
        AbstractContinuousFunction
            Variable subtracted abstract continuous function.
        """

        return self.arithmetical_operation(a, '-')

    def __isub__(self, a):
        """
        Implements support for in-place subtraction.

        Parameters
        ----------
        a : numeric or array_like or AbstractContinuousFunction
            :math:`a` variable to subtract in-place.

        Returns
        -------
        AbstractContinuousFunction
            In-place variable subtracted abstract continuous function.
        """

        return self.arithmetical_operation(a, '-', True)

    def __mul__(self, a):
        """
        Implements support for multiplication.

        Parameters
        ----------
        a : numeric or array_like or AbstractContinuousFunction
            :math:`a` variable to multiply by.

        Returns
        -------
        AbstractContinuousFunction
            Variable multiplied abstract continuous function.
        """

        return self.arithmetical_operation(a, '*')

    def __imul__(self, a):
        """
        Implements support for in-place multiplication.

        Parameters
        ----------
        a : numeric or array_like or AbstractContinuousFunction
            :math:`a` variable to multiply by in-place.

        Returns
        -------
        AbstractContinuousFunction
            In-place variable multiplied abstract continuous function.
        """

        return self.arithmetical_operation(a, '*', True)

    def __div__(self, a):
        """
        Implements support for division.

        Parameters
        ----------
        a : numeric or array_like or AbstractContinuousFunction
            :math:`a` variable to divide by.

        Returns
        -------
        AbstractContinuousFunction
            Variable divided abstract continuous function.
        """

        return self.arithmetical_operation(a, '/')

    def __idiv__(self, a):
        """
        Implements support for in-place division.

        Parameters
        ----------
        a : numeric or array_like or AbstractContinuousFunction
            :math:`a` variable to divide by in-place.

        Returns
        -------
        AbstractContinuousFunction
            In-place variable divided abstract continuous function.
        """

        return self.arithmetical_operation(a, '/', True)

    __itruediv__ = __idiv__
    __truediv__ = __div__

    def __pow__(self, a):
        """
        Implements support for exponentiation.

        Parameters
        ----------
        a : numeric or array_like or AbstractContinuousFunction
            :math:`a` variable to exponentiate by.

        Returns
        -------
        AbstractContinuousFunction
            Variable exponentiated abstract continuous function.
        """

        return self.arithmetical_operation(a, '**')

    def __ipow__(self, a):
        """
        Implements support for in-place exponentiation.

        Parameters
        ----------
        a : numeric or array_like or AbstractContinuousFunction
            :math:`a` variable to exponentiate by in-place.

        Returns
        -------
        AbstractContinuousFunction
            In-place variable exponentiated abstract continuous function.
        """

        return self.arithmetical_operation(a, '**', True)

    @abstractmethod
    def arithmetical_operation(self, a, operation, in_place=False):
        """
        Performs given arithmetical operation with :math:`a` operand, the
        operation can be either performed on a copy or in-place, must be
        reimplemented by sub-classes.

        Parameters
        ----------
        a : numeric or ndarray or AbstractContinuousFunction
            Operand.
        operation : object
            Operation to perform.
        in_place : bool, optional
            Operation happens in place.

        Returns
        -------
        AbstractContinuousFunction
            Abstract continuous function.
        """

        pass

    @abstractmethod
    def fill_nan(self, method='Interpolation', default=0):
        """
        Fill NaNs in independent domain :math:`x` variable and corresponding
        range :math:`y` variable using given method, must be reimplemented by
        sub-classes.

        Parameters
        ----------
        method : unicode, optional
            **{'Interpolation', 'Constant'}**,
            *Interpolation* method linearly interpolates through the NaNs,
            *Constant* method replaces NaNs with ``default``.
        default : numeric, optional
            Value to use with the *Constant* method.

        Returns
        -------
        AbstractContinuousFunction
            NaNs filled abstract continuous function.
        """

        pass

    def domain_distance(self, a):
        """
        Returns the euclidean distance between given array and independent
        domain :math:`x` closest element.

        Parameters
        ----------
        a : numeric or array_like
            :math:`a` variable to compute the euclidean distance with
            independent domain :math:`x` variable.

        Returns
        -------
        numeric or array_like
            Euclidean distance between independent domain :math:`x` variable
            and given :math:`a` variable.
        """

        n = closest(self.domain, a)

        return as_float(np.abs(a - n))

    def is_uniform(self):
        """
        Returns if independent domain :math:`x` variable is uniform.

        Returns
        -------
        bool
            Is independent domain :math:`x` variable uniform.
        """

        return is_uniform(self.domain)

    def copy(self):
        """
        Returns a copy of the sub-class instance.

        Returns
        -------
        AbstractContinuousFunction
            Abstract continuous function copy.
        """

        return deepcopy(self)
