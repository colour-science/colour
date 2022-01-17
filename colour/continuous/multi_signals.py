# -*- coding: utf-8 -*-
"""
Multi Signals
=============

Defines the class implementing support for multi-continuous signals:

-   :class:`colour.continuous.MultiSignals`
"""

import numpy as np
from collections.abc import Iterator, KeysView, Mapping, Sequence, ValuesView

from colour.constants import DEFAULT_FLOAT_DTYPE
from colour.continuous import AbstractContinuousFunction, Signal
from colour.utilities import (
    as_float_array,
    attest,
    first_item,
    is_iterable,
    is_pandas_installed,
    required,
    tsplit,
    tstack,
    validate_method,
)
from colour.utilities.documentation import is_documentation_building

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'MultiSignals',
]


class MultiSignals(AbstractContinuousFunction):
    """
    Defines the base class for multi-continuous signals, a container for
    multiple :class:`colour.continuous.Signal` sub-class instances.

    .. important::

        Specific documentation about getting, setting, indexing and slicing the
        multi-continuous signals values is available in the
        :ref:`spectral-representation-and-continuous-signal` section.

    Parameters
    ----------
    data : Series or Dataframe or Signal or MultiSignals or array_like or \
dict_like, optional
        Data to be stored in the multi-continuous signals.
    domain : array_like, optional
        Values to initialise the multiple :class:`colour.continuous.Signal`
        sub-class instances :attr:`colour.continuous.Signal.domain` attribute
        with. If both ``data`` and ``domain`` arguments are defined, the latter
        will be used to initialise the :attr:`colour.continuous.Signal.domain`
        attribute.
    labels : array_like, optional
        Names to use for the :class:`colour.continuous.Signal` sub-class
        instances.

    Other Parameters
    ----------------
    name : str, optional
        multi-continuous signals name.
    dtype : type, optional
        **{np.float16, np.float32, np.float64, np.float128}**,
        Floating point data type.
    interpolator : object, optional
        Interpolator class type to use as interpolating function for the
        :class:`colour.continuous.Signal` sub-class instances.
    interpolator_kwargs : dict_like, optional
        Arguments to use when instantiating the interpolating function
        of the :class:`colour.continuous.Signal` sub-class instances.
    extrapolator : object, optional
        Extrapolator class type to use as extrapolating function for the
        :class:`colour.continuous.Signal` sub-class instances.
    extrapolator_kwargs : dict_like, optional
        Arguments to use when instantiating the extrapolating function
        of the :class:`colour.continuous.Signal` sub-class instances.
    signal_type : type, optional
        The :class:`colour.continuous.Signal` sub-class type used for
        instances.

    Attributes
    ----------
    -   :attr:`~colour.continuous.MultiSignals.dtype`
    -   :attr:`~colour.continuous.MultiSignals.domain`
    -   :attr:`~colour.continuous.MultiSignals.range`
    -   :attr:`~colour.continuous.MultiSignals.interpolator`
    -   :attr:`~colour.continuous.MultiSignals.interpolator_kwargs`
    -   :attr:`~colour.continuous.MultiSignals.extrapolator`
    -   :attr:`~colour.continuous.MultiSignals.extrapolator_kwargs`
    -   :attr:`~colour.continuous.MultiSignals.function`
    -   :attr:`~colour.continuous.MultiSignals.signals`
    -   :attr:`~colour.continuous.MultiSignals.labels`
    -   :attr:`~colour.continuous.MultiSignals.signal_type`

    Methods
    -------
    -   :meth:`~colour.continuous.MultiSignals.__init__`
    -   :meth:`~colour.continuous.MultiSignals.__str__`
    -   :meth:`~colour.continuous.MultiSignals.__repr__`
    -   :meth:`~colour.continuous.MultiSignals.__hash__`
    -   :meth:`~colour.continuous.MultiSignals.__getitem__`
    -   :meth:`~colour.continuous.MultiSignals.__setitem__`
    -   :meth:`~colour.continuous.MultiSignals.__contains__`
    -   :meth:`~colour.continuous.MultiSignals.__eq__`
    -   :meth:`~colour.continuous.MultiSignals.__ne__`
    -   :meth:`~colour.continuous.MultiSignals.arithmetical_operation`
    -   :meth:`~colour.continuous.MultiSignals.multi_signals_unpack_data`
    -   :meth:`~colour.continuous.MultiSignals.fill_nan`
    -   :meth:`~colour.continuous.MultiSignals.to_dataframe`

    Examples
    --------
    Instantiation with implicit *domain* and a single signal:

    >>> range_ = np.linspace(10, 100, 10)
    >>> print(MultiSignals(range_))
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

    Instantiation with explicit *domain* and a single signal:

    >>> domain = np.arange(100, 1100, 100)
    >>> print(MultiSignals(range_, domain))
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

    Instantiation with multiple signals:

    >>> range_ = tstack([np.linspace(10, 100, 10)] * 3)
    >>> range_ += np.array([0, 10, 20])
    >>> print(MultiSignals(range_, domain))
    [[  100.    10.    20.    30.]
     [  200.    20.    30.    40.]
     [  300.    30.    40.    50.]
     [  400.    40.    50.    60.]
     [  500.    50.    60.    70.]
     [  600.    60.    70.    80.]
     [  700.    70.    80.    90.]
     [  800.    80.    90.   100.]
     [  900.    90.   100.   110.]
     [ 1000.   100.   110.   120.]]

    Instantiation with a *dict*:

    >>> print(MultiSignals(dict(zip(domain, range_))))
    [[  100.    10.    20.    30.]
     [  200.    20.    30.    40.]
     [  300.    30.    40.    50.]
     [  400.    40.    50.    60.]
     [  500.    50.    60.    70.]
     [  600.    60.    70.    80.]
     [  700.    70.    80.    90.]
     [  800.    80.    90.   100.]
     [  900.    90.   100.   110.]
     [ 1000.   100.   110.   120.]]

    Instantiation using a *Signal* sub-class:

    >>> class NotSignal(Signal):
    ...     pass

    >>> multi_signals = MultiSignals(range_, domain, signal_type=NotSignal)
    >>> print(multi_signals)
    [[  100.    10.    20.    30.]
     [  200.    20.    30.    40.]
     [  300.    30.    40.    50.]
     [  400.    40.    50.    60.]
     [  500.    50.    60.    70.]
     [  600.    60.    70.    80.]
     [  700.    70.    80.    90.]
     [  800.    80.    90.   100.]
     [  900.    90.   100.   110.]
     [ 1000.   100.   110.   120.]]
     >>> type(multi_signals.signals[0])  # doctest: +SKIP
     <class 'multi_signals.NotSignal'>

    Instantiation with a *Pandas* `Series`:

    >>> if is_pandas_installed():
    ...     from pandas import Series
    ...     print(MultiSignals(  # doctest: +SKIP
    ...         Series(dict(zip(domain, np.linspace(10, 100, 10))))))
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

    Instantiation with a *Pandas* `DataFrame`:

    >>> if is_pandas_installed():
    ...     from pandas import DataFrame
    ...     data = dict(zip(['a', 'b', 'c'], tsplit(range_)))
    ...     print(MultiSignals(  # doctest: +SKIP
    ...         DataFrame(data, domain)))
    [[  100.    10.    20.    30.]
     [  200.    20.    30.    40.]
     [  300.    30.    40.    50.]
     [  400.    40.    50.    60.]
     [  500.    50.    60.    70.]
     [  600.    60.    70.    80.]
     [  700.    70.    80.    90.]
     [  800.    80.    90.   100.]
     [  900.    90.   100.   110.]
     [ 1000.   100.   110.   120.]]

    Retrieving domain *y* variable for arbitrary range *x* variable:

    >>> x = 150
    >>> range_ = tstack([np.sin(np.linspace(0, 1, 10))] * 3)
    >>> range_ += np.array([0.0, 0.25, 0.5])
    >>> MultiSignals(range_, domain)[x]  # doctest: +ELLIPSIS
    array([ 0.0359701...,  0.2845447...,  0.5331193...])
    >>> x = np.linspace(100, 1000, 3)
    >>> MultiSignals(range_, domain)[x]  # doctest: +ELLIPSIS
    array([[  4.4085384...e-20,   2.5000000...e-01,   5.0000000...e-01],
           [  4.7669395...e-01,   7.2526859...e-01,   9.7384323...e-01],
           [  8.4147098...e-01,   1.0914709...e+00,   1.3414709...e+00]])

    Using an alternative interpolating function:

    >>> x = 150
    >>> from colour.algebra import CubicSplineInterpolator
    >>> MultiSignals(
    ...     range_,
    ...     domain,
    ...     interpolator=CubicSplineInterpolator)[x]  # doctest: +ELLIPSIS
    array([ 0.0555274...,  0.3055274...,  0.5555274...])
    >>> x = np.linspace(100, 1000, 3)
    >>> MultiSignals(
    ...     range_,
    ...     domain,
    ...     interpolator=CubicSplineInterpolator)[x]  # doctest: +ELLIPSIS
    array([[ 0.       ...,  0.25     ...,  0.5      ...],
           [ 0.4794253...,  0.7294253...,  0.9794253...],
           [ 0.8414709...,  1.0914709...,  1.3414709...]])
    """

    def __init__(self, data=None, domain=None, labels=None, **kwargs):
        super(MultiSignals, self).__init__(kwargs.get('name'))

        self._signal_type = kwargs.get('signal_type', Signal)

        self._signals = self.multi_signals_unpack_data(data, domain, labels,
                                                       **kwargs)

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

        if self._signals:
            return first_item(self._signals.values()).dtype

    @dtype.setter
    def dtype(self, value):
        """
        Setter for **self.dtype** property.
        """

        if value is not None:
            for signal in self._signals.values():
                signal.dtype = value

    @property
    def domain(self):
        """
        Getter and setter property for the :class:`colour.continuous.Signal`
        sub-class instances independent domain :math:`x` variable.

        Parameters
        ----------
        value : array_like
            Value to set the :class:`colour.continuous.Signal` sub-class
            instances independent domain :math:`x` variable with.

        Returns
        -------
        ndarray
            :class:`colour.continuous.Signal` sub-class instances independent
            domain :math:`x` variable.
        """

        if self._signals:
            return first_item(self._signals.values()).domain

    @domain.setter
    def domain(self, value):
        """
        Setter for the **self.domain** property.
        """

        if value is not None:
            for signal in self._signals.values():
                signal.domain = value

    @property
    def range(self):
        """
        Getter and setter property for the :class:`colour.continuous.Signal`
        sub-class instances corresponding range :math:`y` variable.

        Parameters
        ----------
        value : array_like
            Value to set the :class:`colour.continuous.Signal` sub-class
            instances corresponding range :math:`y` variable with.

        Returns
        -------
        ndarray
            :class:`colour.continuous.Signal` sub-class instances corresponding
            range :math:`y` variable.
        """

        if self._signals:
            return tstack([signal.range for signal in self._signals.values()])

    @range.setter
    def range(self, value):
        """
        Setter for the **self.range** property.
        """

        if value is not None:
            value = as_float_array(value)

            if value.ndim in (0, 1):
                for signal in self._signals.values():
                    signal.range = value
            else:
                attest(
                    value.shape[-1] == len(self._signals),
                    'Corresponding "y" variable columns must have '
                    'same count than underlying "Signal" components!')

                for signal, y in zip(self._signals.values(), tsplit(value)):
                    signal.range = y

    @property
    def interpolator(self):
        """
        Getter and setter property for the :class:`colour.continuous.Signal`
        sub-class instances interpolator type.

        Parameters
        ----------
        value : type
            Value to set the :class:`colour.continuous.Signal` sub-class
            instances interpolator type with.

        Returns
        -------
        type
            :class:`colour.continuous.Signal` sub-class instances interpolator
            type.
        """

        if self._signals:
            return first_item(self._signals.values()).interpolator

    @interpolator.setter
    def interpolator(self, value):
        """
        Setter for the **self.interpolator** property.
        """

        if value is not None:
            for signal in self._signals.values():
                signal.interpolator = value

    @property
    def interpolator_kwargs(self):
        """
        Getter and setter property for the :class:`colour.continuous.Signal`
        sub-class instances interpolator instantiation time arguments.

        Parameters
        ----------
        value : dict
            Value to set the :class:`colour.continuous.Signal` sub-class
            instances interpolator instantiation time arguments to.

        Returns
        -------
        dict
            :class:`colour.continuous.Signal` sub-class instances interpolator
            instantiation time arguments.
        """

        if self._signals:
            return first_item(self._signals.values()).interpolator_kwargs

    @interpolator_kwargs.setter
    def interpolator_kwargs(self, value):
        """
        Setter for the **self.interpolator_kwargs** property.
        """

        if value is not None:
            for signal in self._signals.values():
                signal.interpolator_kwargs = value

    @property
    def extrapolator(self):
        """
        Getter and setter property for the :class:`colour.continuous.Signal`
        sub-class instances extrapolator type.

        Parameters
        ----------
        value : type
            Value to set the :class:`colour.continuous.Signal` sub-class
            instances extrapolator type with.

        Returns
        -------
        type
            :class:`colour.continuous.Signal` sub-class instances extrapolator
            type.
        """

        if self._signals:
            return first_item(self._signals.values()).extrapolator

    @extrapolator.setter
    def extrapolator(self, value):
        """
        Setter for the **self.extrapolator** property.
        """

        if value is not None:
            for signal in self._signals.values():
                signal.extrapolator = value

    @property
    def extrapolator_kwargs(self):
        """
        Getter and setter property for the :class:`colour.continuous.Signal`
        sub-class instances extrapolator instantiation time arguments.

        Parameters
        ----------
        value : dict
            Value to set the :class:`colour.continuous.Signal` sub-class
            instances extrapolator instantiation time arguments to.

        Returns
        -------
        dict
            :class:`colour.continuous.Signal` sub-class instances extrapolator
            instantiation time arguments.
        """

        if self._signals:
            return first_item(self._signals.values()).extrapolator_kwargs

    @extrapolator_kwargs.setter
    def extrapolator_kwargs(self, value):
        """
        Setter for the **self.extrapolator_kwargs** property.
        """

        if value is not None:
            for signal in self._signals.values():
                signal.extrapolator_kwargs = value

    @property
    def function(self):
        """
        Getter property for the :class:`colour.continuous.Signal` sub-class
        instances callable.

        Returns
        -------
        callable
            :class:`colour.continuous.Signal` sub-class instances callable.
        """

        if self._signals:
            return first_item(self._signals.values()).function

    @property
    def signals(self):
        """
        Getter and setter property for the :class:`colour.continuous.Signal`
        sub-class instances.

        Parameters
        ----------
        value : Series or Dataframe or Signal or MultiSignals or array_like \
or dict_like
            Attribute value.

        Returns
        -------
        dict
            :class:`colour.continuous.Signal` sub-class instances.
        """

        return self._signals

    @signals.setter
    def signals(self, value):
        """
        Setter for the **self.signals** property.
        """

        if value is not None:
            self._signals = self.multi_signals_unpack_data(
                value, signal_type=self._signal_type)

    @property
    def labels(self):
        """
        Getter and setter property for the :class:`colour.continuous.Signal`
        sub-class instances name.

        Parameters
        ----------
        value : array_like
            Value to set the :class:`colour.continuous.Signal` sub-class
            instances name.

        Returns
        -------
        dict
            :class:`colour.continuous.Signal` sub-class instance name.
        """

        if self._signals:
            return list(self._signals.keys())

    @labels.setter
    def labels(self, value):
        """
        Setter for the **self.labels** property.
        """

        if value is not None:
            attest(
                is_iterable(value),
                '"{0}" attribute: "{1}" is not an "iterable" like object!'.
                format('labels', value))

            attest(
                len(set(value)) == len(value),
                '"{0}" attribute: values must be unique!'.format('labels'))

            attest(
                len(value) == len(self.labels),
                '"{0}" attribute: length must be "{1}"!'.format(
                    'labels', len(self._signals)))

            self._signals = {
                value[i]: signal
                for i, (_key, signal) in enumerate(self._signals.items())
            }

    @property
    def signal_type(self):
        """
        Getter property for the :class:`colour.continuous.Signal` sub-class
        instances type.

        Returns
        -------
        type
            :class:`colour.continuous.Signal` sub-class instances type.
        """

        return self._signal_type

    def __str__(self):
        """
        Returns a formatted string representation of the multi-continuous
        signals.

        Returns
        -------
        str
            Formatted string representation.

        Examples
        --------
        >>> domain = np.arange(0, 10, 1)
        >>> range_ = tstack([np.linspace(10, 100, 10)] * 3)
        >>> range_ += np.array([0, 10, 20])
        >>> print(MultiSignals(range_))
        [[   0.   10.   20.   30.]
         [   1.   20.   30.   40.]
         [   2.   30.   40.   50.]
         [   3.   40.   50.   60.]
         [   4.   50.   60.   70.]
         [   5.   60.   70.   80.]
         [   6.   70.   80.   90.]
         [   7.   80.   90.  100.]
         [   8.   90.  100.  110.]
         [   9.  100.  110.  120.]]
        """

        try:
            return str(np.hstack([self.domain[:, np.newaxis], self.range]))
        except TypeError:
            return super(MultiSignals, self).__str__()

    def __repr__(self):
        """
        Returns an evaluable string representation of the multi-continuous
        signals.

        Returns
        -------
        str
            Evaluable string representation.

        Examples
        --------
        >>> domain = np.arange(0, 10, 1)
        >>> range_ = tstack([np.linspace(10, 100, 10)] * 3)
        >>> range_ += np.array([0, 10, 20])
        >>> MultiSignals(range_)  # doctest: +ELLIPSIS
        MultiSignals([[   0.,   10.,   20.,   30.],
                      [   1.,   20.,   30.,   40.],
                      [   2.,   30.,   40.,   50.],
                      [   3.,   40.,   50.,   60.],
                      [   4.,   50.,   60.,   70.],
                      [   5.,   60.,   70.,   80.],
                      [   6.,   70.,   80.,   90.],
                      [   7.,   80.,   90.,  100.],
                      [   8.,   90.,  100.,  110.],
                      [   9.,  100.,  110.,  120.]],
                     labels=[0, 1, 2],
                     interpolator=KernelInterpolator,
                     interpolator_kwargs={},
                     extrapolator=Extrapolator,
                     extrapolator_kwargs={...)
        """

        if is_documentation_building():  # pragma: no cover
            return "{0}(name='{1}', ...)".format(self.__class__.__name__,
                                                 self.name)

        try:
            representation = repr(
                np.hstack([self.domain[:, np.newaxis], self.range]))
            representation = representation.replace('array',
                                                    self.__class__.__name__)
            representation = representation.replace(
                '       [',
                '{0}['.format(' ' * (len(self.__class__.__name__) + 2)))
            representation = ('{0},\n'
                              '{1}labels={2},\n'
                              '{1}interpolator={3},\n'
                              '{1}interpolator_kwargs={4},\n'
                              '{1}extrapolator={5},\n'
                              '{1}extrapolator_kwargs={6})').format(
                                  representation[:-1],
                                  ' ' * (len(self.__class__.__name__) + 1),
                                  repr(self.labels), self.interpolator.__name__
                                  if self.interpolator is not None else
                                  self.interpolator,
                                  repr(self.interpolator_kwargs),
                                  self.extrapolator.__name__
                                  if self.extrapolator is not None else
                                  self.extrapolator,
                                  repr(self.extrapolator_kwargs))

            return representation
        except TypeError:
            return super(MultiSignals, self).__repr__()

    def __hash__(self):
        """
        Returns the abstract continuous function hash.

        Returns
        -------
        int
            Object hash.
        """

        return hash((
            self.domain.tobytes(),
            self.range.tobytes(),
            self.interpolator.__name__,
            repr(self.interpolator_kwargs),
            self.extrapolator.__name__,
            repr(self.extrapolator_kwargs),
        ))

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
        >>> range_ = tstack([np.linspace(10, 100, 10)] * 3)
        >>> range_ += np.array([0, 10, 20])
        >>> multi_signals = MultiSignals(range_)
        >>> print(multi_signals)
        [[   0.   10.   20.   30.]
         [   1.   20.   30.   40.]
         [   2.   30.   40.   50.]
         [   3.   40.   50.   60.]
         [   4.   50.   60.   70.]
         [   5.   60.   70.   80.]
         [   6.   70.   80.   90.]
         [   7.   80.   90.  100.]
         [   8.   90.  100.  110.]
         [   9.  100.  110.  120.]]
        >>> multi_signals[0]
        array([ 10.,  20.,  30.])
        >>> multi_signals[np.array([0, 1, 2])]
        array([[ 10.,  20.,  30.],
               [ 20.,  30.,  40.],
               [ 30.,  40.,  50.]])
        >>> multi_signals[np.linspace(0, 5, 5)]  # doctest: +ELLIPSIS
        array([[ 10.       ...,  20.       ...,  30.       ...],
               [ 22.8348902...,  32.8046056...,  42.774321 ...],
               [ 34.8004492...,  44.7434347...,  54.6864201...],
               [ 47.5535392...,  57.5232546...,  67.4929700...],
               [ 60.       ...,  70.       ...,  80.       ...]])
        >>> multi_signals[0:3]
        array([[ 10.,  20.,  30.],
               [ 20.,  30.,  40.],
               [ 30.,  40.,  50.]])
        >>> multi_signals[:, 0:2]
        array([[  10.,   20.],
               [  20.,   30.],
               [  30.,   40.],
               [  40.,   50.],
               [  50.,   60.],
               [  60.,   70.],
               [  70.,   80.],
               [  80.,   90.],
               [  90.,  100.],
               [ 100.,  110.]])
        """

        x_r, x_c = (x[0], x[1]) if isinstance(x, tuple) else (x, slice(None))

        if self._signals:
            return tstack(
                [signal[x_r] for signal in self._signals.values()])[..., x_c]
        else:
            raise RuntimeError('No underlying "Signal" defined!')

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
        >>> domain = np.arange(0, 10, 1)
        >>> range_ = tstack([np.linspace(10, 100, 10)] * 3)
        >>> range_ += np.array([0, 10, 20])
        >>> multi_signals = MultiSignals(range_)
        >>> print(multi_signals)
        [[   0.   10.   20.   30.]
         [   1.   20.   30.   40.]
         [   2.   30.   40.   50.]
         [   3.   40.   50.   60.]
         [   4.   50.   60.   70.]
         [   5.   60.   70.   80.]
         [   6.   70.   80.   90.]
         [   7.   80.   90.  100.]
         [   8.   90.  100.  110.]
         [   9.  100.  110.  120.]]
        >>> multi_signals[0] = 20
        >>> multi_signals[0]
        array([ 20.,  20.,  20.])
        >>> multi_signals[np.array([0, 1, 2])] = 30
        >>> multi_signals[np.array([0, 1, 2])]
        array([[ 30.,  30.,  30.],
               [ 30.,  30.,  30.],
               [ 30.,  30.,  30.]])
        >>> multi_signals[np.linspace(0, 5, 5)] = 50
        >>> print(multi_signals)
        [[   0.     50.     50.     50.  ]
         [   1.     30.     30.     30.  ]
         [   1.25   50.     50.     50.  ]
         [   2.     30.     30.     30.  ]
         [   2.5    50.     50.     50.  ]
         [   3.     40.     50.     60.  ]
         [   3.75   50.     50.     50.  ]
         [   4.     50.     60.     70.  ]
         [   5.     50.     50.     50.  ]
         [   6.     70.     80.     90.  ]
         [   7.     80.     90.    100.  ]
         [   8.     90.    100.    110.  ]
         [   9.    100.    110.    120.  ]]
        >>> multi_signals[np.array([0, 1, 2])] = np.array([10, 20, 30])
        >>> print(multi_signals)
        [[   0.     10.     20.     30.  ]
         [   1.     10.     20.     30.  ]
         [   1.25   50.     50.     50.  ]
         [   2.     10.     20.     30.  ]
         [   2.5    50.     50.     50.  ]
         [   3.     40.     50.     60.  ]
         [   3.75   50.     50.     50.  ]
         [   4.     50.     60.     70.  ]
         [   5.     50.     50.     50.  ]
         [   6.     70.     80.     90.  ]
         [   7.     80.     90.    100.  ]
         [   8.     90.    100.    110.  ]
         [   9.    100.    110.    120.  ]]
        >>> y = np.arange(1, 10, 1).reshape(3, 3)
        >>> multi_signals[np.array([0, 1, 2])] = y
        >>> print(multi_signals)
        [[   0.      1.      2.      3.  ]
         [   1.      4.      5.      6.  ]
         [   1.25   50.     50.     50.  ]
         [   2.      7.      8.      9.  ]
         [   2.5    50.     50.     50.  ]
         [   3.     40.     50.     60.  ]
         [   3.75   50.     50.     50.  ]
         [   4.     50.     60.     70.  ]
         [   5.     50.     50.     50.  ]
         [   6.     70.     80.     90.  ]
         [   7.     80.     90.    100.  ]
         [   8.     90.    100.    110.  ]
         [   9.    100.    110.    120.  ]]
        >>> multi_signals[0:3] = 40
        >>> multi_signals[0:3]
        array([[ 40.,  40.,  40.],
               [ 40.,  40.,  40.],
               [ 40.,  40.,  40.]])
        >>> multi_signals[:, 0:2] = 50
        >>> print(multi_signals)
        [[   0.     50.     50.     40.  ]
         [   1.     50.     50.     40.  ]
         [   1.25   50.     50.     40.  ]
         [   2.     50.     50.      9.  ]
         [   2.5    50.     50.     50.  ]
         [   3.     50.     50.     60.  ]
         [   3.75   50.     50.     50.  ]
         [   4.     50.     50.     70.  ]
         [   5.     50.     50.     50.  ]
         [   6.     50.     50.     90.  ]
         [   7.     50.     50.    100.  ]
         [   8.     50.     50.    110.  ]
         [   9.     50.     50.    120.  ]]
        """

        y = as_float_array(y)

        x_r, x_c = (x[0], x[1]) if isinstance(x, tuple) else (x, slice(None))

        attest(
            y.ndim in range(3),
            'Corresponding "y" variable must be a numeric or a 1-dimensional '
            'or 2-dimensional array!')

        if y.ndim == 0:
            y = np.tile(y, len(self._signals))
        elif y.ndim == 1:
            y = y[np.newaxis, :]

        attest(
            y.shape[-1] == len(self._signals),
            'Corresponding "y" variable columns must have same count than '
            'underlying "Signal" components!')

        for signal, y in list(zip(self._signals.values(), tsplit(y)))[x_c]:
            signal[x_r] = y

    def __contains__(self, x):
        """
        Returns whether the multi-continuous signals contains given independent
        domain :math:`x` variable.

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
        >>> multi_signals = MultiSignals(range_)
        >>> 0 in multi_signals
        True
        >>> 0.5 in multi_signals
        True
        >>> 1000 in multi_signals
        False
        """

        if self._signals:
            return x in first_item(self._signals.values())
        else:
            raise RuntimeError('No underlying "Signal" defined!')

    def __eq__(self, other):
        """
        Returns whether the multi-continuous signals is equal to given other
        object.

        Parameters
        ----------
        other : object
            Object to test whether it is equal to the multi-continuous signals.

        Returns
        -------
        bool
            Is given object equal to the multi-continuous signals.

        Examples
        --------
        >>> range_ = np.linspace(10, 100, 10)
        >>> multi_signals_1 = MultiSignals(range_)
        >>> multi_signals_2 = MultiSignals(range_)
        >>> multi_signals_1 == multi_signals_2
        True
        >>> multi_signals_2[0] = 20
        >>> multi_signals_1 == multi_signals_2
        False
        >>> multi_signals_2[0] = 10
        >>> multi_signals_1 == multi_signals_2
        True
        >>> from colour.algebra import CubicSplineInterpolator
        >>> multi_signals_2.interpolator = CubicSplineInterpolator
        >>> multi_signals_1 == multi_signals_2
        False
        """

        if isinstance(other, MultiSignals):
            if all([
                    np.array_equal(self.domain, other.domain),
                    np.array_equal(
                        self.range,
                        other.range), self.interpolator is other.interpolator,
                    self.interpolator_kwargs == other.interpolator_kwargs,
                    self.extrapolator is other.extrapolator,
                    self.extrapolator_kwargs == other.extrapolator_kwargs,
                    self.labels == other.labels
            ]):
                return True

        return False

    def __ne__(self, other):
        """
        Returns whether the multi-continuous signals is not equal to given
        other object.

        Parameters
        ----------
        other : object
            Object to test whether it is not equal to the multi-continuous
            signals.

        Returns
        -------
        bool
            Is given object not equal to the multi-continuous signals.

        Examples
        --------
        >>> range_ = np.linspace(10, 100, 10)
        >>> multi_signals_1 = MultiSignals(range_)
        >>> multi_signals_2 = MultiSignals(range_)
        >>> multi_signals_1 != multi_signals_2
        False
        >>> multi_signals_2[0] = 20
        >>> multi_signals_1 != multi_signals_2
        True
        >>> multi_signals_2[0] = 10
        >>> multi_signals_1 != multi_signals_2
        False
        >>> from colour.algebra import CubicSplineInterpolator
        >>> multi_signals_2.interpolator = CubicSplineInterpolator
        >>> multi_signals_1 != multi_signals_2
        True
        """

        return not (self == other)

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
        MultiSignals
            multi-continuous signals.

        Examples
        --------
        Adding a single *numeric* variable:

        >>> domain = np.arange(0, 10, 1)
        >>> range_ = tstack([np.linspace(10, 100, 10)] * 3)
        >>> range_ += np.array([0, 10, 20])
        >>> multi_signals_1 = MultiSignals(range_)
        >>> print(multi_signals_1)
        [[   0.   10.   20.   30.]
         [   1.   20.   30.   40.]
         [   2.   30.   40.   50.]
         [   3.   40.   50.   60.]
         [   4.   50.   60.   70.]
         [   5.   60.   70.   80.]
         [   6.   70.   80.   90.]
         [   7.   80.   90.  100.]
         [   8.   90.  100.  110.]
         [   9.  100.  110.  120.]]
        >>> print(multi_signals_1.arithmetical_operation(10, '+', True))
        [[   0.   20.   30.   40.]
         [   1.   30.   40.   50.]
         [   2.   40.   50.   60.]
         [   3.   50.   60.   70.]
         [   4.   60.   70.   80.]
         [   5.   70.   80.   90.]
         [   6.   80.   90.  100.]
         [   7.   90.  100.  110.]
         [   8.  100.  110.  120.]
         [   9.  110.  120.  130.]]

        Adding an *array_like* variable:

        >>> a = np.linspace(10, 100, 10)
        >>> print(multi_signals_1.arithmetical_operation(a, '+', True))
        [[   0.   30.   40.   50.]
         [   1.   50.   60.   70.]
         [   2.   70.   80.   90.]
         [   3.   90.  100.  110.]
         [   4.  110.  120.  130.]
         [   5.  130.  140.  150.]
         [   6.  150.  160.  170.]
         [   7.  170.  180.  190.]
         [   8.  190.  200.  210.]
         [   9.  210.  220.  230.]]

        >>> a = np.array([[10, 20, 30]])
        >>> print(multi_signals_1.arithmetical_operation(a, '+', True))
        [[   0.   40.   60.   80.]
         [   1.   60.   80.  100.]
         [   2.   80.  100.  120.]
         [   3.  100.  120.  140.]
         [   4.  120.  140.  160.]
         [   5.  140.  160.  180.]
         [   6.  160.  180.  200.]
         [   7.  180.  200.  220.]
         [   8.  200.  220.  240.]
         [   9.  220.  240.  260.]]

        >>> a = np.arange(0, 30, 1).reshape([10, 3])
        >>> print(multi_signals_1.arithmetical_operation(a, '+', True))
        [[   0.   40.   61.   82.]
         [   1.   63.   84.  105.]
         [   2.   86.  107.  128.]
         [   3.  109.  130.  151.]
         [   4.  132.  153.  174.]
         [   5.  155.  176.  197.]
         [   6.  178.  199.  220.]
         [   7.  201.  222.  243.]
         [   8.  224.  245.  266.]
         [   9.  247.  268.  289.]]

        Adding a :class:`colour.continuous.Signal` sub-class:

        >>> multi_signals_2 = MultiSignals(range_)
        >>> print(multi_signals_1.arithmetical_operation(
        ...     multi_signals_2, '+', True))
        [[   0.   50.   81.  112.]
         [   1.   83.  114.  145.]
         [   2.  116.  147.  178.]
         [   3.  149.  180.  211.]
         [   4.  182.  213.  244.]
         [   5.  215.  246.  277.]
         [   6.  248.  279.  310.]
         [   7.  281.  312.  343.]
         [   8.  314.  345.  376.]
         [   9.  347.  378.  409.]]
        """

        multi_signals = self if in_place else self.copy()

        if isinstance(a, MultiSignals):
            attest(
                len(self.signals) == len(a.signals),
                '"MultiSignals" operands must have same count than '
                'underlying "Signal" components!')

            for signal_a, signal_b in zip(multi_signals.signals.values(),
                                          a.signals.values()):
                signal_a.arithmetical_operation(signal_b, operation, True)
        else:
            a = as_float_array(a)

            attest(
                a.ndim in range(3),
                'Operand "a" variable must be a numeric or a 1-dimensional or '
                '2-dimensional array!')

            if a.ndim in (0, 1):
                for signal in multi_signals.signals.values():
                    signal.arithmetical_operation(a, operation, True)
            else:
                attest(
                    a.shape[-1] == len(multi_signals.signals),
                    'Operand "a" variable columns must have same count than '
                    'underlying "Signal" components!')

                for signal, y in zip(multi_signals.signals.values(),
                                     tsplit(a)):
                    signal.arithmetical_operation(y, operation, True)

        return multi_signals

    @staticmethod
    def multi_signals_unpack_data(data=None,
                                  domain=None,
                                  labels=None,
                                  dtype=None,
                                  signal_type=Signal,
                                  **kwargs):
        """
        Unpack given data for multi-continuous signals instantiation.

        Parameters
        ----------
        data : Series or Dataframe or Signal or MultiSignals or array_like or \
dict_like, optional
            Data to unpack for multi-continuous signals instantiation.
        domain : array_like, optional
            Values to initialise the multiple :class:`colour.continuous.Signal`
            sub-class instances :attr:`colour.continuous.Signal.domain`
            attribute with. If both ``data`` and ``domain`` arguments are
            defined, the latter will be used to initialise the
            :attr:`colour.continuous.Signal.domain` attribute.
        dtype : type, optional
            **{np.float16, np.float32, np.float64, np.float128}**,
            Floating point data type.
        signal_type : type, optional
            A :class:`colour.continuous.Signal` sub-class type.

        Other Parameters
        ----------------
        name : str, optional
            multi-continuous signals name.
        interpolator : object, optional
            Interpolator class type to use as interpolating function for the
            :class:`colour.continuous.Signal` sub-class instances.
        interpolator_kwargs : dict_like, optional
            Arguments to use when instantiating the interpolating function
            of the :class:`colour.continuous.Signal` sub-class instances.
        extrapolator : object, optional
            Extrapolator class type to use as extrapolating function for the
            :class:`colour.continuous.Signal` sub-class instances.
        extrapolator_kwargs : dict_like, optional
            Arguments to use when instantiating the extrapolating function
            of the :class:`colour.continuous.Signal` sub-class instances.

        Returns
        -------
        dict
            Mapping of labeled :class:`colour.continuous.Signal` sub-class
            instances.

        Examples
        --------
        Unpacking using implicit *domain* and data for a single signal:

        >>> range_ = np.linspace(10, 100, 10)
        >>> signals = MultiSignals.multi_signals_unpack_data(range_)
        >>> list(signals.keys())
        [0]
        >>> print(signals[0])
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

        Unpacking using explicit *domain* and data for a single signal:

        >>> domain = np.arange(100, 1100, 100)
        >>> signals = MultiSignals.multi_signals_unpack_data(range_, domain)
        >>> list(signals.keys())
        [0]
        >>> print(signals[0])
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

        Unpacking using data for multiple signals:

        >>> range_ = tstack([np.linspace(10, 100, 10)] * 3)
        >>> range_ += np.array([0, 10, 20])
        >>> signals = MultiSignals.multi_signals_unpack_data(range_, domain)
        >>> list(signals.keys())
        [0, 1, 2]
        >>> print(signals[2])
        [[  100.    30.]
         [  200.    40.]
         [  300.    50.]
         [  400.    60.]
         [  500.    70.]
         [  600.    80.]
         [  700.    90.]
         [  800.   100.]
         [  900.   110.]
         [ 1000.   120.]]

        Unpacking using a *dict*:

        >>> signals = MultiSignals.multi_signals_unpack_data(
        ...     dict(zip(domain, range_)))
        >>> list(signals.keys())
        [0, 1, 2]
        >>> print(signals[2])
        [[  100.    30.]
         [  200.    40.]
         [  300.    50.]
         [  400.    60.]
         [  500.    70.]
         [  600.    80.]
         [  700.    90.]
         [  800.   100.]
         [  900.   110.]
         [ 1000.   120.]]

        Unpacking using a sequence of *Signal* instances, note how the keys
        are :class:`str` instances because the *Signal* names are used:

        >>> signals = MultiSignals.multi_signals_unpack_data(
        ...     dict(zip(domain, range_))).values()
        >>> signals = MultiSignals.multi_signals_unpack_data(signals)
        >>> list(signals.keys())
        ['0', '1', '2']
        >>> print(signals['2'])
        [[  100.    30.]
         [  200.    40.]
         [  300.    50.]
         [  400.    60.]
         [  500.    70.]
         [  600.    80.]
         [  700.    90.]
         [  800.   100.]
         [  900.   110.]
         [ 1000.   120.]]

        Unpacking using *MultiSignals.multi_signals_unpack_data* method output:

        >>> signals = MultiSignals.multi_signals_unpack_data(
        ...     dict(zip(domain, range_)))
        >>> signals = MultiSignals.multi_signals_unpack_data(signals)
        >>> list(signals.keys())
        [0, 1, 2]
        >>> print(signals[2])
        [[  100.    30.]
         [  200.    40.]
         [  300.    50.]
         [  400.    60.]
         [  500.    70.]
         [  600.    80.]
         [  700.    90.]
         [  800.   100.]
         [  900.   110.]
         [ 1000.   120.]]

        Unpacking using a *Pandas* `Series`:

        >>> if is_pandas_installed():
        ...     from pandas import Series
        ...     signals = MultiSignals.multi_signals_unpack_data(
        ...         Series(dict(zip(domain, np.linspace(10, 100, 10)))))
        ...     print(signals[0])  # doctest: +SKIP
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

        Unpacking using a *Pandas* `DataFrame`:

        >>> if is_pandas_installed():
        ...     from pandas import DataFrame
        ...     data = dict(zip(['a', 'b', 'c'], tsplit(range_)))
        ...     signals = MultiSignals.multi_signals_unpack_data(
        ...         DataFrame(data, domain))
        ...     print(signals['c'])  # doctest: +SKIP
        [[  100.    30.]
         [  200.    40.]
         [  300.    50.]
         [  400.    60.]
         [  500.    70.]
         [  600.    80.]
         [  700.    90.]
         [  800.   100.]
         [  900.   110.]
         [ 1000.   120.]]
        """

        if dtype is None:
            dtype = DEFAULT_FLOAT_DTYPE

        settings = {}
        settings.update(kwargs)
        settings.update({
            'dtype': dtype,
            'strict_name': None,
        })

        # domain_u, range_u, signals = None, None, None

        signals = {}

        domain = list(domain) if isinstance(domain, KeysView) else domain

        # TODO: Implement support for Signal class passing.
        if isinstance(data, MultiSignals):
            signals = data.signals
        elif (issubclass(type(data), Sequence) or
              isinstance(data,
                         (tuple, list, np.ndarray, Iterator, ValuesView))):

            is_signal = all(
                [True if isinstance(i, Signal) else False for i in data])

            if is_signal:
                for signal in data:
                    signals[signal.name] = signal_type(
                        signal.range, signal.domain, **settings)
            else:
                data = tsplit(
                    list(data) if isinstance(data, (Iterator,
                                                    ValuesView)) else data)
                attest(
                    data.ndim in (1, 2),
                    'User "data" must be 1-dimensional or 2-dimensional!')

                if data.ndim == 1:
                    data = data[np.newaxis, :]

                for i, range_u in enumerate(data):
                    signals[i] = signal_type(range_u, domain, **settings)
        elif issubclass(type(data), Mapping) or isinstance(data, dict):

            is_signal = all([
                True if isinstance(i, Signal) else False
                for i in data.values()
            ])

            if is_signal:
                for label, signal in data.items():
                    signals[label] = signal_type(signal.range, signal.domain,
                                                 **settings)
            else:
                domain_u, range_u = zip(*sorted(data.items()))
                for i, range_u in enumerate(tsplit(range_u)):
                    signals[i] = signal_type(range_u, domain_u, **settings)
        elif is_pandas_installed():
            from pandas import DataFrame, Series

            if isinstance(data, Series):
                signals[0] = signal_type(data, **settings)
            elif isinstance(data, DataFrame):
                domain_u = data.index.values
                signals = {
                    label: signal_type(data[label], domain_u, **settings)
                    for label in data
                }

                for label in data:
                    signals[label].name = label

        if domain is not None and signals is not None:
            for signal in signals.values():
                attest(
                    len(domain) == len(signal.domain),
                    'User "domain" is not compatible with unpacked signals!')

                signal.domain = domain

        if labels is not None and signals is not None:
            attest(
                len(labels) == len(signals),
                'User "labels" is not compatible with unpacked signals!')

            signals = {
                labels[i]: signal
                for i, (_key, signal) in enumerate(signals.items())
            }

        for label in signals:
            signals[label].name = str(label)

        return signals

    def fill_nan(self, method='Interpolation', default=0):
        """
        Fill NaNs in independent domain :math:`x` variable and corresponding
        range :math:`y` variable using given method.

        Parameters
        ----------
        method : str, optional
            **{'Interpolation', 'Constant'}**,
            *Interpolation* method linearly interpolates through the NaNs,
            *Constant* method replaces NaNs with ``default``.
        default : numeric, optional
            Value to use with the *Constant* method.

        Returns
        -------
        Signal
            NaNs filled multi-continuous signals.

        >>> domain = np.arange(0, 10, 1)
        >>> range_ = tstack([np.linspace(10, 100, 10)] * 3)
        >>> range_ += np.array([0, 10, 20])
        >>> multi_signals = MultiSignals(range_)
        >>> multi_signals[3:7] = np.nan
        >>> print(multi_signals)
        [[   0.   10.   20.   30.]
         [   1.   20.   30.   40.]
         [   2.   30.   40.   50.]
         [   3.   nan   nan   nan]
         [   4.   nan   nan   nan]
         [   5.   nan   nan   nan]
         [   6.   nan   nan   nan]
         [   7.   80.   90.  100.]
         [   8.   90.  100.  110.]
         [   9.  100.  110.  120.]]
        >>> print(multi_signals.fill_nan())
        [[   0.   10.   20.   30.]
         [   1.   20.   30.   40.]
         [   2.   30.   40.   50.]
         [   3.   40.   50.   60.]
         [   4.   50.   60.   70.]
         [   5.   60.   70.   80.]
         [   6.   70.   80.   90.]
         [   7.   80.   90.  100.]
         [   8.   90.  100.  110.]
         [   9.  100.  110.  120.]]
        >>> multi_signals[3:7] = np.nan
        >>> print(multi_signals.fill_nan(method='Constant'))
        [[   0.   10.   20.   30.]
         [   1.   20.   30.   40.]
         [   2.   30.   40.   50.]
         [   3.    0.    0.    0.]
         [   4.    0.    0.    0.]
         [   5.    0.    0.    0.]
         [   6.    0.    0.    0.]
         [   7.   80.   90.  100.]
         [   8.   90.  100.  110.]
         [   9.  100.  110.  120.]]
        """

        method = validate_method(method, ['Interpolation', 'Constant'])

        for signal in self._signals.values():
            signal.fill_nan(method, default)

        return self

    @required('Pandas')
    def to_dataframe(self):
        """
        Converts the continuous signal to a *Pandas* :class:`DataFrame` class
        instance.

        Returns
        -------
        DataFrame
            Continuous signal as a *Pandas* :class:`DataFrame` class instance.

        Examples
        --------
        >>> if is_pandas_installed():
        ...     domain = np.arange(0, 10, 1)
        ...     range_ = tstack([np.linspace(10, 100, 10)] * 3)
        ...     range_ += np.array([0, 10, 20])
        ...     multi_signals = MultiSignals(range_)
        ...     print(multi_signals.to_dataframe())  # doctest: +SKIP
                 0      1      2
        0.0   10.0   20.0   30.0
        1.0   20.0   30.0   40.0
        2.0   30.0   40.0   50.0
        3.0   40.0   50.0   60.0
        4.0   50.0   60.0   70.0
        5.0   60.0   70.0   80.0
        6.0   70.0   80.0   90.0
        7.0   80.0   90.0  100.0
        8.0   90.0  100.0  110.0
        9.0  100.0  110.0  120.0
        """

        from pandas import DataFrame

        return DataFrame(
            data=self.range, index=self.domain, columns=self.labels)
