# -*- coding: utf-8 -*-
"""
Signal
======

Defines the class implementing support for continuous signal:

-   :class:`colour.continuous.Signal`
"""

from __future__ import division, unicode_literals

import numpy as np
from collections import Iterator, Mapping, OrderedDict, Sequence
from operator import add, mul, pow, sub, iadd, imul, ipow, isub

# Python 3 compatibility.
try:
    from operator import div, idiv
except ImportError:
    from operator import truediv, itruediv

    div = truediv
    idiv = itruediv

from colour.algebra import Extrapolator, KernelInterpolator
from colour.constants import DEFAULT_FLOAT_DTYPE
from colour.continuous import AbstractContinuousFunction
from colour.utilities import (as_array, fill_nan, is_pandas_installed,
                              runtime_warning, tsplit, tstack)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['Signal']


class Signal(AbstractContinuousFunction):
    """
    Defines the base class for continuous signal.

    The class implements the :meth:`Signal.function` method so that evaluating
    the function for any independent domain :math:`x \\in\\mathbb{R}` variable
    returns a corresponding range :math:`y \\in\\mathbb{R}` variable.
    It adopts an interpolating function encapsulated inside an extrapolating
    function. The resulting function independent domain, stored as discrete
    values in the :attr:`colour.continuous.Signal.domain` attribute corresponds
    with the function dependent and already known range stored in the
    :attr:`colour.continuous.Signal.range` attribute.

    Parameters
    ----------
    data : Series or Signal or array_like or dict_like, optional
        Data to be stored in the continuous signal.
    domain : array_like, optional
        Values to initialise the :attr:`colour.continuous.Signal.domain`
        attribute with. If both ``data`` and ``domain`` arguments are defined,
        the latter with be used to initialise the
        :attr:`colour.continuous.Signal.domain` attribute.

    Other Parameters
    ----------------
    name : unicode, optional
        Continuous signal name.
    dtype : type, optional
        **{np.float16, np.float32, np.float64, np.float128}**,
        Floating point data type.
    interpolator : object, optional
        Interpolator class type to use as interpolating function.
    interpolator_args : dict_like, optional
        Arguments to use when instantiating the interpolating function.
    extrapolator : object, optional
        Extrapolator class type to use as extrapolating function.
    extrapolator_args : dict_like, optional
        Arguments to use when instantiating the extrapolating function.

    Attributes
    ----------
    dtype
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
    __eq__
    __ne__
    arithmetical_operation
    signal_unpack_data
    fill_nan
    to_series

    Examples
    --------
    Instantiation with implicit *domain*:

    >>> range_ = np.linspace(10, 100, 10)
    >>> print(Signal(range_))
    [[   0.   10.]
     [   1.   20.]
     [   2.   30.]
     [   3.   40.]
     [   4.   50.]
     [   5.   60.]
     [   6.   70.]
     [   7.   80.]
     [   8.   90.]
     [   9.  100.]]

    Instantiation with explicit *domain*:

    >>> domain = np.arange(100, 1100, 100)
    >>> print(Signal(range_, domain))
    [[  100.    10.]
     [  200.    20.]
     [  300.    30.]
     [  400.    40.]
     [  500.    50.]
     [  600.    60.]
     [  700.    70.]
     [  800.    80.]
     [  900.    90.]
     [ 1000.   100.]]

    Instantiation with a *dict*:

    >>> print(Signal(dict(zip(domain, range_))))
    [[  100.    10.]
     [  200.    20.]
     [  300.    30.]
     [  400.    40.]
     [  500.    50.]
     [  600.    60.]
     [  700.    70.]
     [  800.    80.]
     [  900.    90.]
     [ 1000.   100.]]

    Instantiation with a *Pandas* *Series*:

    >>> if is_pandas_installed():
    ...     from pandas import Series
    ...     print(Signal(  # doctest: +SKIP
    ...         Series(dict(zip(domain, range_)))))
    [[  100.    10.]
     [  200.    20.]
     [  300.    30.]
     [  400.    40.]
     [  500.    50.]
     [  600.    60.]
     [  700.    70.]
     [  800.    80.]
     [  900.    90.]
     [ 1000.   100.]]

    Retrieving domain *y* variable for arbitrary range *x* variable:

    >>> x = 150
    >>> range_ = np.sin(np.linspace(0, 1, 10))
    >>> Signal(range_, domain)[x]  # doctest: +ELLIPSIS
    0.0359701...
    >>> x = np.linspace(100, 1000, 3)
    >>> Signal(range_, domain)[x]  # doctest: +ELLIPSIS
    array([  ...,   4.7669395...e-01,   8.4147098...e-01])

    Using an alternative interpolating function:

    >>> x = 150
    >>> from colour.algebra import CubicSplineInterpolator
    >>> Signal(
    ...     range_,
    ...     domain,
    ...     interpolator=CubicSplineInterpolator)[x]  # doctest: +ELLIPSIS
    0.0555274...
    >>> x = np.linspace(100, 1000, 3)
    >>> Signal(
    ...     range_,
    ...     domain,
    ...     interpolator=CubicSplineInterpolator)[x]  # doctest: +ELLIPSIS
    array([ 0.        ,  0.4794253...,  0.8414709...])
    """

    def __init__(self, data=None, domain=None, **kwargs):
        super(Signal, self).__init__(kwargs.get('name'))

        self._dtype = None
        self._domain = None
        self._range = None
        self._interpolator = KernelInterpolator
        self._interpolator_args = {}
        self._extrapolator = Extrapolator
        self._extrapolator_args = {
            'method': 'Constant',
            'left': np.nan,
            'right': np.nan
        }

        self.domain, self.range = self.signal_unpack_data(data, domain)

        self.dtype = kwargs.get('dtype')

        self.interpolator = kwargs.get('interpolator')
        self.interpolator_args = kwargs.get('interpolator_args')
        self.extrapolator = kwargs.get('extrapolator')
        self.extrapolator_args = kwargs.get('extrapolator_args')

        self._create_function()

    @property
    def dtype(self):
        """
        Getter and setter property for the continuous signal dtype.

        Parameters
        ----------
        value : type
            Value to set the continuous signal dtype with.

        Returns
        -------
        type
            Continuous signal dtype.
        """

        return self._dtype

    @dtype.setter
    def dtype(self, value):
        """
        Setter for **self.dtype** property.
        """

        if value is not None:
            float_dtypes = []
            for float_dtype in ['float16', 'float32', 'float64', 'float128']:
                if hasattr(np, float_dtype):
                    float_dtypes.append(getattr(np, float_dtype))

            assert value in float_dtypes, ((
                '"{0}" attribute: "{1}" type is not in "{2}"!').format(
                    'dtype', value, ', '.join([
                        float_dtype.__name__ for float_dtype in float_dtypes
                    ])))

            self._dtype = value

            # The following self-assignments are written as intended and
            # triggers the rebuild of the underlying function.
            self.domain = self.domain
            self.range = self.range

    @property
    def domain(self):
        """
        Getter and setter property for the continuous signal independent
        domain :math:`x` variable.

        Parameters
        ----------
        value : array_like
            Value to set the continuous signal independent domain
            :math:`x` variable with.

        Returns
        -------
        ndarray
            Continuous signal independent domain :math:`x` variable.
        """

        return np.copy(self._domain)

    @domain.setter
    def domain(self, value):
        """
        Setter for the **self.domain** property.
        """

        if value is not None:
            if not np.all(np.isfinite(value)):
                runtime_warning(
                    '"{0}" new "domain" variable is not finite: {1}, '
                    'unpredictable results may occur!'.format(
                        self.name, value))

            value = np.copy(value).astype(self.dtype)

            if self._range is not None:
                if value.size != self._range.size:
                    runtime_warning(
                        '"{0}" new "domain" and current "range" variables '
                        'have different size, "range" variable will be '
                        'resized to "domain" variable shape!'.format(
                            self.name))
                    self._range = np.resize(self._range, value.shape)

            self._domain = value
            self._create_function()

    @property
    def range(self):
        """
        Getter and setter property for the continuous signal corresponding
        range :math:`y` variable.

        Parameters
        ----------
        value : array_like
            Value to set the continuous signal corresponding range :math:`y`
            variable with.

        Returns
        -------
        ndarray
            Continuous signal corresponding range :math:`y` variable.
        """

        return np.copy(self._range)

    @range.setter
    def range(self, value):
        """
        Setter for the **self.range** property.
        """

        if value is not None:
            if not np.all(np.isfinite(value)):
                runtime_warning(
                    '"{0}" new "range" variable is not finite: {1}, '
                    'unpredictable results may occur!'.format(
                        self.name, value))

            value = np.copy(value).astype(self.dtype)

            if self._domain is not None:
                assert value.size == self._domain.size, (
                    '"domain" and "range" variables must have same size!')

            self._range = value
            self._create_function()

    @property
    def interpolator(self):
        """
        Getter and setter property for the continuous signal interpolator type.

        Parameters
        ----------
        value : type
            Value to set the continuous signal interpolator type
            with.

        Returns
        -------
        type
            Continuous signal interpolator type.
        """

        return self._interpolator

    @interpolator.setter
    def interpolator(self, value):
        """
        Setter for the **self.interpolator** property.
        """

        if value is not None:
            # TODO: Check for interpolator capabilities.
            self._interpolator = value
            self._create_function()

    @property
    def interpolator_args(self):
        """
        Getter and setter property for the continuous signal interpolator
        instantiation time arguments.

        Parameters
        ----------
        value : dict
            Value to set the continuous signal interpolator instantiation
            time arguments to.

        Returns
        -------
        dict
            Continuous signal interpolator instantiation time
            arguments.
        """

        return self._interpolator_args

    @interpolator_args.setter
    def interpolator_args(self, value):
        """
        Setter for the **self.interpolator_args** property.
        """

        if value is not None:
            assert isinstance(value, (dict, OrderedDict)), (
                '"{0}" attribute: "{1}" type is not "dict" or "OrderedDict"!'
            ).format('interpolator_args', value)

            self._interpolator_args = value
            self._create_function()

    @property
    def extrapolator(self):
        """
        Getter and setter property for the continuous signal extrapolator type.

        Parameters
        ----------
        value : type
            Value to set the continuous signal extrapolator type
            with.

        Returns
        -------
        type
            Continuous signal extrapolator type.
        """

        return self._extrapolator

    @extrapolator.setter
    def extrapolator(self, value):
        """
        Setter for the **self.extrapolator** property.
        """

        if value is not None:
            # TODO: Check for extrapolator capabilities.
            self._extrapolator = value
            self._create_function()

    @property
    def extrapolator_args(self):
        """
        Getter and setter property for the continuous signal extrapolator
        instantiation time arguments.

        Parameters
        ----------
        value : dict
            Value to set the continuous signal extrapolator instantiation
            time arguments to.

        Returns
        -------
        dict
            Continuous signal extrapolator instantiation time
            arguments.
        """

        return self._extrapolator_args

    @extrapolator_args.setter
    def extrapolator_args(self, value):
        """
        Setter for the **self.extrapolator_args** property.
        """

        if value is not None:
            assert isinstance(value, (dict, OrderedDict)), (
                '"{0}" attribute: "{1}" type is not "dict" or "OrderedDict"!'.
                format('extrapolator_args', value))

            self._extrapolator_args = value
            self._create_function()

    @property
    def function(self):
        """
        Getter and setter property for the continuous signal callable.

        Parameters
        ----------
        value : object
            Attribute value.

        Returns
        -------
        callable
            Continuous signal callable.

        Notes
        -----
        -   This property is read only.
        """

        return self._function

    def __str__(self):
        """
        Returns a formatted string representation of the continuous signal.

        Returns
        -------
        unicode
            Formatted string representation.

        Examples
        --------
        >>> range_ = np.linspace(10, 100, 10)
        >>> print(Signal(range_))
        [[   0.   10.]
         [   1.   20.]
         [   2.   30.]
         [   3.   40.]
         [   4.   50.]
         [   5.   60.]
         [   6.   70.]
         [   7.   80.]
         [   8.   90.]
         [   9.  100.]]
        """

        try:
            return str(tstack([self.domain, self.range]))
        except TypeError:
            return super(Signal, self).__str__()

    def __repr__(self):
        """
        Returns an evaluable string representation of the continuous signal.

        Returns
        -------
        unicode
            Evaluable string representation.

        Examples
        --------
        >>> range_ = np.linspace(10, 100, 10)
        >>> Signal(range_)  # doctest: +ELLIPSIS
        Signal([[   0.,   10.],
                [   1.,   20.],
                [   2.,   30.],
                [   3.,   40.],
                [   4.,   50.],
                [   5.,   60.],
                [   6.,   70.],
                [   7.,   80.],
                [   8.,   90.],
                [   9.,  100.]],
               interpolator=KernelInterpolator,
               interpolator_args={},
               extrapolator=Extrapolator,
               extrapolator_args={...})
        """

        try:
            representation = repr(tstack([self.domain, self.range]))
            representation = representation.replace('array',
                                                    self.__class__.__name__)
            representation = representation.replace(
                '       [',
                '{0}['.format(' ' * (len(self.__class__.__name__) + 2)))
            representation = ('{0},\n'
                              '{1}interpolator={2},\n'
                              '{1}interpolator_args={3},\n'
                              '{1}extrapolator={4},\n'
                              '{1}extrapolator_args={5})').format(
                                  representation[:-1],
                                  ' ' * (len(self.__class__.__name__) + 1),
                                  self.interpolator.__name__,
                                  repr(self.interpolator_args),
                                  self.extrapolator.__name__,
                                  repr(self.extrapolator_args))

            return representation
        except TypeError:
            # TODO: Discuss what is the most suitable behaviour, either the
            # following or __str__ one.
            return '{0}()'.format(self.__class__.__name__)

    def __hash__(self):
        """
        Returns the abstract continuous function hash.

        Returns
        -------
        int
            Object hash.
        """

        return hash(repr(self))

    def __getitem__(self, x):
        """
        Returns the corresponding range :math:`y` variable for independent
        domain :math:`x` variable.

        Parameters
        ----------
        x : numeric, array_like or slice
            Independent domain :math:`x` variable.

        Returns
        -------
        numeric or ndarray
            math:`y` range value.

        Examples
        --------
        >>> range_ = np.linspace(10, 100, 10)
        >>> signal = Signal(range_)
        >>> print(signal)
        [[   0.   10.]
         [   1.   20.]
         [   2.   30.]
         [   3.   40.]
         [   4.   50.]
         [   5.   60.]
         [   6.   70.]
         [   7.   80.]
         [   8.   90.]
         [   9.  100.]]
        >>> signal[0]
        10.0
        >>> signal[np.array([0, 1, 2])]
        array([ 10.,  20.,  30.])
        >>> signal[0:3]
        array([ 10.,  20.,  30.])
        >>> signal[np.linspace(0, 5, 5)]  # doctest: +ELLIPSIS
        array([ 10.        ,  22.8348902...,  34.8004492...,  \
47.5535392...,  60.        ])
        """

        if isinstance(x, slice):
            return self._range[x]
        else:
            return self._function(x)

    def __setitem__(self, x, y):
        """
        Sets the corresponding range :math:`y` variable for independent domain
        :math:`x` variable.

        Parameters
        ----------
        x : numeric, array_like or slice
            Independent domain :math:`x` variable.
        y : numeric or ndarray
            Corresponding range :math:`y` variable.

        Examples
        --------
        >>> range_ = np.linspace(10, 100, 10)
        >>> signal = Signal(range_)
        >>> print(signal)
        [[   0.   10.]
         [   1.   20.]
         [   2.   30.]
         [   3.   40.]
         [   4.   50.]
         [   5.   60.]
         [   6.   70.]
         [   7.   80.]
         [   8.   90.]
         [   9.  100.]]
        >>> signal[0] = 20
        >>> signal[0]
        20.0
        >>> signal[np.array([0, 1, 2])] = 30
        >>> signal[np.array([0, 1, 2])]
        array([ 30.,  30.,  30.])
        >>> signal[0:3] = 40
        >>> signal[0:3]
        array([ 40.,  40.,  40.])
        >>> signal[np.linspace(0, 5, 5)] = 50
        >>> print(signal)
        [[   0.     50.  ]
         [   1.     40.  ]
         [   1.25   50.  ]
         [   2.     40.  ]
         [   2.5    50.  ]
         [   3.     40.  ]
         [   3.75   50.  ]
         [   4.     50.  ]
         [   5.     50.  ]
         [   6.     70.  ]
         [   7.     80.  ]
         [   8.     90.  ]
         [   9.    100.  ]]
        >>> signal[np.array([0, 1, 2])] = np.array([10, 20, 30])
        >>> print(signal)
        [[   0.     10.  ]
         [   1.     20.  ]
         [   1.25   50.  ]
         [   2.     30.  ]
         [   2.5    50.  ]
         [   3.     40.  ]
         [   3.75   50.  ]
         [   4.     50.  ]
         [   5.     50.  ]
         [   6.     70.  ]
         [   7.     80.  ]
         [   8.     90.  ]
         [   9.    100.  ]]
        """

        if isinstance(x, slice):
            self._range[x] = y
        else:
            x = np.atleast_1d(x).astype(self.dtype)
            y = np.resize(y, x.shape)

            # Matching domain, updating existing `self._range` values.
            mask = np.in1d(x, self._domain)
            x_m = x[mask]
            indexes = np.searchsorted(self._domain, x_m)
            self._range[indexes] = y[mask]

            # Non matching domain, inserting into existing `self.domain`
            # and `self.range`.
            x_nm = x[~mask]
            indexes = np.searchsorted(self._domain, x_nm)
            if indexes.size != 0:
                self._domain = np.insert(self._domain, indexes, x_nm)
                self._range = np.insert(self._range, indexes, y[~mask])

        self._create_function()

    def __contains__(self, x):
        """
        Returns whether the continuous signal contains given independent domain
        :math:`x` variable.

        Parameters
        ----------
        x : numeric, array_like or slice
            Independent domain :math:`x` variable.

        Returns
        -------
        bool
            Is :math:`x` domain value contained.

        Examples
        --------
        >>> range_ = np.linspace(10, 100, 10)
        >>> signal = Signal(range_)
        >>> 0 in signal
        True
        >>> 0.5 in signal
        True
        >>> 1000 in signal
        False
        """

        return np.all(
            np.where(
                np.logical_and(x >= np.min(self._domain),
                               x <= np.max(self._domain)),
                True,
                False,
            ))

    def __eq__(self, other):
        """
        Returns whether the continuous signal is equal to given other object.

        Parameters
        ----------
        other : object
            Object to test whether it is equal to the continuous signal.

        Returns
        -------
        bool
            Is given object equal to the continuous signal.

        Examples
        --------
        >>> range_ = np.linspace(10, 100, 10)
        >>> signal_1 = Signal(range_)
        >>> signal_2 = Signal(range_)
        >>> signal_1 == signal_2
        True
        >>> signal_2[0] = 20
        >>> signal_1 == signal_2
        False
        >>> signal_2[0] = 10
        >>> signal_1 == signal_2
        True
        >>> from colour.algebra import CubicSplineInterpolator
        >>> signal_2.interpolator = CubicSplineInterpolator
        >>> signal_1 == signal_2
        False
        """

        if isinstance(other, Signal):
            if all([
                    np.array_equal(self._domain, other.domain),
                    np.array_equal(self._range, other.range),
                    self._interpolator is other.interpolator,
                    self._interpolator_args == other.interpolator_args,
                    self._extrapolator is other.extrapolator,
                    self._extrapolator_args == other.extrapolator_args
            ]):
                return True

        return False

    def __ne__(self, other):
        """
        Returns whether the continuous signal is not equal to given other
        object.

        Parameters
        ----------
        other : object
            Object to test whether it is not equal to the continuous signal.

        Returns
        -------
        bool
            Is given object not equal to the continuous signal.

        Examples
        --------
        >>> range_ = np.linspace(10, 100, 10)
        >>> signal_1 = Signal(range_)
        >>> signal_2 = Signal(range_)
        >>> signal_1 != signal_2
        False
        >>> signal_2[0] = 20
        >>> signal_1 != signal_2
        True
        >>> signal_2[0] = 10
        >>> signal_1 != signal_2
        False
        >>> from colour.algebra import CubicSplineInterpolator
        >>> signal_2.interpolator = CubicSplineInterpolator
        >>> signal_1 != signal_2
        True
        """

        return not (self == other)

    def _create_function(self):
        """
        Creates the continuous signal underlying function.
        """

        if self._domain is not None and self._range is not None:
            self._function = self._extrapolator(
                self._interpolator(self.domain, self.range,
                                   **self._interpolator_args),
                **self._extrapolator_args)
        else:

            def _undefined_function(*args, **kwargs):
                """
                Raises a :class:`RuntimeError` exception.

                Other Parameters
                ----------------
                \\*args : list, optional
                    Arguments.
                \\**kwargs : dict, optional
                    Keywords arguments.

                Raises
                ------
                RuntimeError
                """

                raise RuntimeError(
                    'Underlying signal interpolator function does not exists, '
                    'please ensure you defined both '
                    '"domain" and "range" variables!')

            self._function = _undefined_function

    def _fill_domain_nan(self, method='Interpolation', default=0):
        """
        Fill NaNs in independent domain :math:`x` variable using given method.

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
        Signal
            NaNs filled continuous signal independent domain :math:`x`
            variable.
        """

        self._domain = fill_nan(self._domain, method, default)
        self._create_function()

    def _fill_range_nan(self, method='Interpolation', default=0):
        """
        Fill NaNs in corresponding range :math:`y` variable using given method.

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
        Signal
            NaNs filled continuous signal i corresponding range :math:`y`
            variable.
        """

        self._range = fill_nan(self._range, method, default)
        self._create_function()

    def arithmetical_operation(self, a, operation, in_place=False):
        """
        Performs given arithmetical operation with :math:`a` operand, the
        operation can be either performed on a copy or in-place.

        Parameters
        ----------
        a : numeric or ndarray or Signal
            Operand.
        operation : object
            Operation to perform.
        in_place : bool, optional
            Operation happens in place.

        Returns
        -------
        Signal
            Continuous signal.

        Examples
        --------
        Adding a single *numeric* variable:

        >>> range_ = np.linspace(10, 100, 10)
        >>> signal_1 = Signal(range_)
        >>> print(signal_1)
        [[   0.   10.]
         [   1.   20.]
         [   2.   30.]
         [   3.   40.]
         [   4.   50.]
         [   5.   60.]
         [   6.   70.]
         [   7.   80.]
         [   8.   90.]
         [   9.  100.]]
        >>> print(signal_1.arithmetical_operation(10, '+', True))
        [[   0.   20.]
         [   1.   30.]
         [   2.   40.]
         [   3.   50.]
         [   4.   60.]
         [   5.   70.]
         [   6.   80.]
         [   7.   90.]
         [   8.  100.]
         [   9.  110.]]

        Adding an *array_like* variable:

        >>> a = np.linspace(10, 100, 10)
        >>> print(signal_1.arithmetical_operation(a, '+', True))
        [[   0.   30.]
         [   1.   50.]
         [   2.   70.]
         [   3.   90.]
         [   4.  110.]
         [   5.  130.]
         [   6.  150.]
         [   7.  170.]
         [   8.  190.]
         [   9.  210.]]

        Adding a :class:`colour.continuous.Signal` class:

        >>> signal_2 = Signal(range_)
        >>> print(signal_1.arithmetical_operation(signal_2, '+', True))
        [[   0.   40.]
         [   1.   70.]
         [   2.  100.]
         [   3.  130.]
         [   4.  160.]
         [   5.  190.]
         [   6.  220.]
         [   7.  250.]
         [   8.  280.]
         [   9.  310.]]
        """

        operation, ioperator = {
            '+': (add, iadd),
            '-': (sub, isub),
            '*': (mul, imul),
            '/': (div, idiv),
            '**': (pow, ipow)
        }[operation]

        if in_place:
            if isinstance(a, Signal):
                self[self._domain] = operation(self._range, a[self._domain])
                exclusive_or = np.setxor1d(self._domain, a.domain)
                self[exclusive_or] = np.full(exclusive_or.shape, np.nan)
            else:
                self.range = ioperator(self.range, a)

            return self
        else:
            copy = ioperator(self.copy(), a)

            return copy

    @staticmethod
    def signal_unpack_data(data=None, domain=None, dtype=DEFAULT_FLOAT_DTYPE):
        """
        Unpack given data for continuous signal instantiation.

        Parameters
        ----------
        data : Series or Signal or array_like or dict_like, optional
            Data to unpack for continuous signal instantiation.
        domain : array_like, optional
            Values to initialise the :attr:`colour.continuous.Signal.domain`
            attribute with. If both ``data`` and ``domain`` arguments are
            defined, the latter will be used to initialise the
            :attr:`colour.continuous.Signal.domain` attribute.
        dtype : type, optional
            **{np.float16, np.float32, np.float64, np.float128}**,
            Floating point data type.

        Returns
        -------
        tuple
            Independent domain :math:`x` variable and corresponding range
            :math:`y` variable unpacked for continuous signal instantiation.

        Examples
        --------
        Unpacking using implicit *domain*:

        >>> range_ = np.linspace(10, 100, 10)
        >>> domain, range_ = Signal.signal_unpack_data(range_)
        >>> print(domain)
        [ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9.]
        >>> print(range_)
        [  10.   20.   30.   40.   50.   60.   70.   80.   90.  100.]

        Unpacking using explicit *domain*:

        >>> domain = np.arange(100, 1100, 100)
        >>> domain, range = Signal.signal_unpack_data(range_, domain)
        >>> print(domain)
        [  100.   200.   300.   400.   500.   600.   700.   800.   900.  1000.]
        >>> print(range_)
        [  10.   20.   30.   40.   50.   60.   70.   80.   90.  100.]

        Unpacking using a *dict*:

        >>> domain, range_ = Signal.signal_unpack_data(
        ...     dict(zip(domain, range_)))
        >>> print(domain)
        [  100.   200.   300.   400.   500.   600.   700.   800.   900.  1000.]
        >>> print(range_)
        [  10.   20.   30.   40.   50.   60.   70.   80.   90.  100.]

        Unpacking using a *Pandas* *Series*:

        >>> if is_pandas_installed():
        ...     from pandas import Series
        ...     domain, range = Signal.signal_unpack_data(
        ...         Series(dict(zip(domain, range_))))
        ... # doctest: +ELLIPSIS
        >>> print(domain)  # doctest: +SKIP
        [  100.   200.   300.   400.   500.   600.   700.   800.   900.  1000.]
        >>> print(range_)  # doctest: +SKIP
        [  10.   20.   30.   40.   50.   60.   70.   80.   90.  100.]

        Unpacking using a :class:`colour.continuous.Signal` class:

        >>> domain, range_ = Signal.signal_unpack_data(
        ...     Signal(range_, domain))
        >>> print(domain)
        [  100.   200.   300.   400.   500.   600.   700.   800.   900.  1000.]
        >>> print(range_)
        [  10.   20.   30.   40.   50.   60.   70.   80.   90.  100.]
        """

        assert dtype in np.sctypes['float'], (
            '"dtype" must be one of the following types: {0}'.format(
                np.sctypes['float']))

        domain_u, range_u = None, None
        if isinstance(data, Signal):
            domain_u = data.domain
            range_u = data.range
        elif (issubclass(type(data), Sequence) or
              isinstance(data, (tuple, list, np.ndarray, Iterator))):
            data = tsplit(list(data) if isinstance(data, Iterator) else data)
            assert data.ndim == 1, 'User "data" must be 1-dimensional!'
            domain_u, range_u = np.arange(0, data.size, dtype=dtype), data
        elif (issubclass(type(data), Mapping) or
              isinstance(data, (dict, OrderedDict))):
            domain_u, range_u = tsplit(sorted(data.items()))
        elif is_pandas_installed():
            from pandas import Series

            if isinstance(data, Series):
                domain_u = data.index.values
                range_u = data.values

        if domain is not None and range_u is not None:
            assert len(domain) == len(range_u), (
                'User "domain" is not compatible with unpacked range!')
            domain_u = as_array(domain, dtype)

        if range_u is not None:
            range_u = as_array(range_u, dtype)

        return domain_u, range_u

    def fill_nan(self, method='Interpolation', default=0):
        """
        Fill NaNs in independent domain :math:`x` variable and corresponding
        range :math:`y` variable using given method.

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
        Signal
            NaNs filled continuous signal.

        Examples
        --------
        >>> range_ = np.linspace(10, 100, 10)
        >>> signal = Signal(range_)
        >>> signal[3:7] = np.nan
        >>> print(signal)
        [[   0.   10.]
         [   1.   20.]
         [   2.   30.]
         [   3.   nan]
         [   4.   nan]
         [   5.   nan]
         [   6.   nan]
         [   7.   80.]
         [   8.   90.]
         [   9.  100.]]
        >>> print(signal.fill_nan())
        [[   0.   10.]
         [   1.   20.]
         [   2.   30.]
         [   3.   40.]
         [   4.   50.]
         [   5.   60.]
         [   6.   70.]
         [   7.   80.]
         [   8.   90.]
         [   9.  100.]]
        >>> signal[3:7] = np.nan
        >>> print(signal.fill_nan(method='Constant'))
        [[   0.   10.]
         [   1.   20.]
         [   2.   30.]
         [   3.    0.]
         [   4.    0.]
         [   5.    0.]
         [   6.    0.]
         [   7.   80.]
         [   8.   90.]
         [   9.  100.]]
        """

        self._fill_domain_nan(method, default)
        self._fill_range_nan(method, default)

        return self

    def to_series(self):
        """
        Converts the continuous signal to a *Pandas* :class:`Series` class
        instance.

        Returns
        -------
        Series
            Continuous signal as a *Pandas* :class:`Series` class instance.

        Examples
        --------
        >>> if is_pandas_installed():
        ...     range_ = np.linspace(10, 100, 10)
        ...     signal = Signal(range_)
        ...     print(signal.to_series())  # doctest: +SKIP
        0.0     10.0
        1.0     20.0
        2.0     30.0
        3.0     40.0
        4.0     50.0
        5.0     60.0
        6.0     70.0
        7.0     80.0
        8.0     90.0
        9.0    100.0
        Name: Signal (...), dtype: float64
        """

        if is_pandas_installed():
            from pandas import Series

            return Series(data=self._range, index=self._domain, name=self.name)
