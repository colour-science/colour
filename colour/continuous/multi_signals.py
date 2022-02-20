"""
Multi Signals
=============

Defines the class implementing support for multi-continuous signals:

-   :class:`colour.continuous.MultiSignals`
"""

from __future__ import annotations

import numpy as np
from collections.abc import Iterator, Mapping, ValuesView

from colour.constants import DEFAULT_FLOAT_DTYPE
from colour.continuous import AbstractContinuousFunction, Signal
from colour.hints import (
    Any,
    ArrayLike,
    Boolean,
    Callable,
    Dict,
    DTypeFloating,
    FloatingOrArrayLike,
    FloatingOrNDArray,
    Integer,
    List,
    Literal,
    NDArray,
    Number,
    Optional,
    Sequence,
    Type,
    TypeExtrapolator,
    TypeInterpolator,
    Union,
    cast,
)
from colour.utilities import (
    as_float_array,
    attest,
    first_item,
    is_iterable,
    is_pandas_installed,
    optional,
    required,
    tsplit,
    tstack,
    validate_method,
)
from colour.utilities.documentation import is_documentation_building

if is_pandas_installed():
    from pandas import DataFrame, Series
else:  # pragma: no cover
    from unittest import mock

    DataFrame = mock.MagicMock()
    Series = mock.MagicMock()

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "MultiSignals",
]


class MultiSignals(AbstractContinuousFunction):
    """
    Define the base class for multi-continuous signals, a container for
    multiple :class:`colour.continuous.Signal` sub-class instances.

    .. important::

        Specific documentation about getting, setting, indexing and slicing the
        multi-continuous signals values is available in the
        :ref:`spectral-representation-and-continuous-signal` section.

    Parameters
    ----------
    data
        Data to be stored in the multi-continuous signals.
    domain
        Values to initialise the multiple :class:`colour.continuous.Signal`
        sub-class instances :attr:`colour.continuous.Signal.domain` attribute
        with. If both ``data`` and ``domain`` arguments are defined, the latter
        will be used to initialise the :attr:`colour.continuous.Signal.domain`
        attribute.
    labels
        Names to use for the :class:`colour.continuous.Signal` sub-class
        instances.

    Other Parameters
    ----------------
    dtype
        Floating point data type.
    extrapolator
        Extrapolator class type to use as extrapolating function for the
        :class:`colour.continuous.Signal` sub-class instances.
    extrapolator_kwargs
        Arguments to use when instantiating the extrapolating function
        of the :class:`colour.continuous.Signal` sub-class instances.
    interpolator
        Interpolator class type to use as interpolating function for the
        :class:`colour.continuous.Signal` sub-class instances.
    interpolator_kwargs
        Arguments to use when instantiating the interpolating function
        of the :class:`colour.continuous.Signal` sub-class instances.
    name
        multi-continuous signals name.
    signal_type
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

    Instantiation with a *Pandas* :class:`pandas.DataFrame`:

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

    def __init__(
        self,
        data: Optional[
            Union[
                ArrayLike,
                DataFrame,
                dict,
                MultiSignals,
                Sequence,
                Series,
                Signal,
            ]
        ] = None,
        domain: Optional[ArrayLike] = None,
        labels: Optional[Sequence] = None,
        **kwargs: Any,
    ):
        super().__init__(kwargs.get("name"))

        self._signal_type: Type[Signal] = kwargs.get("signal_type", Signal)

        self._signals: Dict[str, Signal] = self.multi_signals_unpack_data(
            data, domain, labels, **kwargs
        )

    @property
    def dtype(self) -> Type[DTypeFloating]:
        """
        Getter and setter property for the continuous signal dtype.

        Parameters
        ----------
        value
            Value to set the continuous signal dtype with.

        Returns
        -------
        Type[DTypeFloating]
            Continuous signal dtype.
        """

        return first_item(self._signals.values()).dtype

    @dtype.setter
    def dtype(self, value: Type[DTypeFloating]):
        """Setter for the **self.dtype** property."""

        for signal in self._signals.values():
            signal.dtype = value

    @property
    def domain(self) -> NDArray:
        """
        Getter and setter property for the :class:`colour.continuous.Signal`
        sub-class instances independent domain variable :math:`x`.

        Parameters
        ----------
        value
            Value to set the :class:`colour.continuous.Signal` sub-class
            instances independent domain variable :math:`x` with.

        Returns
        -------
        :class:`numpy.ndarray`
            :class:`colour.continuous.Signal` sub-class instances independent
            domain variable :math:`x`.
        """

        return first_item(self._signals.values()).domain

    @domain.setter
    def domain(self, value: ArrayLike):
        """Setter for the **self.domain** property."""

        for signal in self._signals.values():
            signal.domain = as_float_array(value, self.dtype)

    @property
    def range(self) -> NDArray:
        """
        Getter and setter property for the :class:`colour.continuous.Signal`
        sub-class instances corresponding range variable :math:`y`.

        Parameters
        ----------
        value
            Value to set the :class:`colour.continuous.Signal` sub-class
            instances corresponding range variable :math:`y` with.

        Returns
        -------
        :class:`numpy.ndarray`
            :class:`colour.continuous.Signal` sub-class instances corresponding
            range variable :math:`y`.
        """

        return tstack([signal.range for signal in self._signals.values()])

    @range.setter
    def range(self, value: ArrayLike):
        """Setter for the **self.range** property."""

        value = as_float_array(value)

        if value.ndim in (0, 1):
            for signal in self._signals.values():
                signal.range = value
        else:
            attest(
                value.shape[-1] == len(self._signals),
                'Corresponding "y" variable columns must have '
                'same count than underlying "Signal" components!',
            )

            for signal, y in zip(self._signals.values(), tsplit(value)):
                signal.range = y

    @property
    def interpolator(self) -> Type[TypeInterpolator]:
        """
        Getter and setter property for the :class:`colour.continuous.Signal`
        sub-class instances interpolator type.

        Parameters
        ----------
        value
            Value to set the :class:`colour.continuous.Signal` sub-class
            instances interpolator type with.

        Returns
        -------
        Type[TypeInterpolator]
            :class:`colour.continuous.Signal` sub-class instances interpolator
            type.
        """

        return first_item(self._signals.values()).interpolator

    @interpolator.setter
    def interpolator(self, value: Type[TypeInterpolator]):
        """Setter for the **self.interpolator** property."""

        if value is not None:
            for signal in self._signals.values():
                signal.interpolator = value

    @property
    def interpolator_kwargs(self) -> Dict:
        """
        Getter and setter property for the :class:`colour.continuous.Signal`
        sub-class instances interpolator instantiation time arguments.

        Parameters
        ----------
        value
            Value to set the :class:`colour.continuous.Signal` sub-class
            instances interpolator instantiation time arguments to.

        Returns
        -------
        :class:`dict`
            :class:`colour.continuous.Signal` sub-class instances interpolator
            instantiation time arguments.
        """

        return first_item(self._signals.values()).interpolator_kwargs

    @interpolator_kwargs.setter
    def interpolator_kwargs(self, value: dict):
        """Setter for the **self.interpolator_kwargs** property."""

        for signal in self._signals.values():
            signal.interpolator_kwargs = value

    @property
    def extrapolator(self) -> Type[TypeExtrapolator]:
        """
        Getter and setter property for the :class:`colour.continuous.Signal`
        sub-class instances extrapolator type.

        Parameters
        ----------
        value
            Value to set the :class:`colour.continuous.Signal` sub-class
            instances extrapolator type with.

        Returns
        -------
        Type[TypeExtrapolator]
            :class:`colour.continuous.Signal` sub-class instances extrapolator
            type.
        """

        return first_item(self._signals.values()).extrapolator

    @extrapolator.setter
    def extrapolator(self, value: Type[TypeExtrapolator]):
        """Setter for the **self.extrapolator** property."""

        for signal in self._signals.values():
            signal.extrapolator = value

    @property
    def extrapolator_kwargs(self) -> Dict:
        """
        Getter and setter property for the :class:`colour.continuous.Signal`
        sub-class instances extrapolator instantiation time arguments.

        Parameters
        ----------
        value
            Value to set the :class:`colour.continuous.Signal` sub-class
            instances extrapolator instantiation time arguments to.

        Returns
        -------
        :class:`dict`
            :class:`colour.continuous.Signal` sub-class instances extrapolator
            instantiation time arguments.
        """

        return first_item(self._signals.values()).extrapolator_kwargs

    @extrapolator_kwargs.setter
    def extrapolator_kwargs(self, value: dict):
        """Setter for the **self.extrapolator_kwargs** property."""

        for signal in self._signals.values():
            signal.extrapolator_kwargs = value

    @property
    def function(self) -> Callable:
        """
        Getter property for the :class:`colour.continuous.Signal` sub-class
        instances callable.

        Returns
        -------
        Callable
            :class:`colour.continuous.Signal` sub-class instances callable.
        """

        return first_item(self._signals.values()).function

    @property
    def signals(self) -> Dict[str, Signal]:
        """
        Getter and setter property for the :class:`colour.continuous.Signal`
        sub-class instances.

        Parameters
        ----------
        value
            Attribute value.

        Returns
        -------
        :class:`dict`
            :class:`colour.continuous.Signal` sub-class instances.
        """

        return self._signals

    @signals.setter
    def signals(
        self,
        value: Optional[
            Union[ArrayLike, DataFrame, dict, MultiSignals, Signal, Series]
        ],
    ):
        """Setter for the **self.signals** property."""

        self._signals = self.multi_signals_unpack_data(
            value, signal_type=self._signal_type
        )

    @property
    def labels(self) -> List[str]:
        """
        Getter and setter property for the :class:`colour.continuous.Signal`
        sub-class instance names.

        Parameters
        ----------
        value
            Value to set the :class:`colour.continuous.Signal` sub-class
            instance names.

        Returns
        -------
        :class:`list`
            :class:`colour.continuous.Signal` sub-class instance names.
        """

        return [str(key) for key in self._signals.keys()]

    @labels.setter
    def labels(self, value: Sequence):
        """Setter for the **self.labels** property."""

        attest(
            is_iterable(value),
            f'"labels" property: "{value}" is not an "iterable" like object!',
        )

        attest(
            len(set(value)) == len(value),
            '"labels" property: values must be unique!',
        )

        attest(
            len(value) == len(self.labels),
            f'"labels" property: length must be "{len(self._signals)}"!',
        )

        self._signals = {
            str(value[i]): signal
            for i, signal in enumerate(self._signals.values())
        }

    @property
    def signal_type(self) -> Type[Signal]:
        """
        Getter property for the :class:`colour.continuous.Signal` sub-class
        instances type.

        Returns
        -------
        Type[Signal]
            :class:`colour.continuous.Signal` sub-class instances type.
        """

        return self._signal_type

    def __str__(self) -> str:
        """
        Return a formatted string representation of the multi-continuous
        signals.

        Returns
        -------
        :class:`str`
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
            return super().__str__()

    def __repr__(self) -> str:
        """
        Return an evaluable string representation of the multi-continuous
        signals.

        Returns
        -------
        :class:`str`
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
                     labels=['0', '1', '2'],
                     interpolator=KernelInterpolator,
                     interpolator_kwargs={},
                     extrapolator=Extrapolator,
                     extrapolator_kwargs={...)
        """

        if is_documentation_building():  # pragma: no cover
            return f"{self.__class__.__name__}(name='{self.name}', ...)"

        try:
            representation = repr(
                np.hstack([self.domain[:, np.newaxis], self.range])
            )
            representation = representation.replace(
                "array", self.__class__.__name__
            )
            representation = representation.replace(
                "       [",
                f"{' ' * (len(self.__class__.__name__) + 2)}[",
            )
            indentation = " " * (len(self.__class__.__name__) + 1)
            interpolator = (
                self.interpolator.__name__
                if self.interpolator is not None
                else self.interpolator
            )
            extrapolator = (
                self.extrapolator.__name__
                if self.extrapolator is not None
                else self.extrapolator
            )
            representation = (
                f"{representation[:-1]},\n"
                f"{indentation}labels={repr(self.labels)},\n"
                f"{indentation}interpolator={interpolator},\n"
                f"{indentation}interpolator_kwargs="
                f"{repr(self.interpolator_kwargs)},\n"
                f"{indentation}extrapolator={extrapolator},\n"
                f"{indentation}extrapolator_kwargs="
                f"{repr(self.extrapolator_kwargs)})"
            )

            return representation
        except TypeError:
            return super().__repr__()

    def __hash__(self) -> Integer:
        """
        Return the abstract continuous function hash.

        Returns
        -------
        :class:`numpy.integer`
            Object hash.
        """

        return hash(
            (
                self.domain.tobytes(),
                self.range.tobytes(),
                self.interpolator.__name__,
                repr(self.interpolator_kwargs),
                self.extrapolator.__name__,
                repr(self.extrapolator_kwargs),
            )
        )

    def __getitem__(
        self, x: Union[FloatingOrArrayLike, slice]
    ) -> FloatingOrNDArray:
        """
        Return the corresponding range variable :math:`y` for independent
        domain variable :math:`x`.

        Parameters
        ----------
        x
            Independent domain variable :math:`x`.

        Returns
        -------
        :class:`numpy.floating` or :class:`numpy.ndarray`
            Variable :math:`y` range value.

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

        return tstack([signal[x_r] for signal in self._signals.values()])[
            ..., x_c
        ]

    def __setitem__(
        self, x: Union[FloatingOrArrayLike, slice], y: FloatingOrArrayLike
    ):
        """
        Set the corresponding range variable :math:`y` for independent domain
        variable :math:`x`.

        Parameters
        ----------
        x
            Independent domain variable :math:`x`.
        y
            Corresponding range variable :math:`y`.

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
            "or 2-dimensional array!",
        )

        if y.ndim == 0:
            y = np.tile(y, len(self._signals))
        elif y.ndim == 1:
            y = y[np.newaxis, :]

        attest(
            y.shape[-1] == len(self._signals),
            'Corresponding "y" variable columns must have same count than '
            'underlying "Signal" components!',
        )

        for signal, y in list(zip(self._signals.values(), tsplit(y)))[x_c]:
            signal[x_r] = y

    def __contains__(self, x: Union[FloatingOrArrayLike, slice]) -> bool:
        """
        Return whether the multi-continuous signals contains given independent
        domain variable :math:`x`.

        Parameters
        ----------
        x
            Independent domain variable :math:`x`.

        Returns
        -------
        :class:`bool`
            Whether :math:`x` domain value is contained.

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

        return x in first_item(self._signals.values())

    def __eq__(self, other: Any) -> bool:
        """
        Return whether the multi-continuous signals is equal to given other
        object.

        Parameters
        ----------
        other
            Object to test whether it is equal to the multi-continuous signals.

        Returns
        -------
        :class:`bool`
            Whether given object is equal to the multi-continuous signals.

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
            return all(
                [
                    np.array_equal(self.domain, other.domain),
                    np.array_equal(self.range, other.range),
                    self.interpolator is other.interpolator,
                    self.interpolator_kwargs == other.interpolator_kwargs,
                    self.extrapolator is other.extrapolator,
                    self.extrapolator_kwargs == other.extrapolator_kwargs,
                    self.labels == other.labels,
                ]
            )
        else:
            return False

    def __ne__(self, other: Any) -> bool:
        """
        Return whether the multi-continuous signals is not equal to given
        other object.

        Parameters
        ----------
        other
            Object to test whether it is not equal to the multi-continuous
            signals.

        Returns
        -------
        :class:`bool`
            Whether given object is not equal to the multi-continuous signals.

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

    def arithmetical_operation(
        self,
        a: Union[FloatingOrArrayLike, AbstractContinuousFunction],
        operation: Literal["+", "-", "*", "/", "**"],
        in_place: Boolean = False,
    ) -> AbstractContinuousFunction:
        """
        Perform given arithmetical operation with operand :math:`a`, the
        operation can be either performed on a copy or in-place.

        Parameters
        ----------
        a
            Operand :math:`a`.
        operation
            Operation to perform.
        in_place
            Operation happens in place.

        Returns
        -------
        :class:`colour.continuous.MultiSignals`
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

        Adding an `ArrayLike` variable:

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

        multi_signals = cast(MultiSignals, self if in_place else self.copy())

        if isinstance(a, MultiSignals):
            attest(
                len(self.signals) == len(a.signals),
                '"MultiSignals" operands must have same count than '
                'underlying "Signal" components!',
            )

            for signal_a, signal_b in zip(
                multi_signals.signals.values(), a.signals.values()
            ):
                signal_a.arithmetical_operation(signal_b, operation, True)
        else:
            a = as_float_array(a)  # type: ignore[arg-type]

            attest(
                a.ndim in range(3),
                'Operand "a" variable must be a numeric or a 1-dimensional or '
                "2-dimensional array!",
            )

            if a.ndim in (0, 1):
                for signal in multi_signals.signals.values():
                    signal.arithmetical_operation(a, operation, True)
            else:
                attest(
                    a.shape[-1] == len(multi_signals.signals),
                    'Operand "a" variable columns must have same count than '
                    'underlying "Signal" components!',
                )

                for signal, y in zip(
                    multi_signals.signals.values(), tsplit(a)
                ):
                    signal.arithmetical_operation(y, operation, True)

        return multi_signals

    @staticmethod
    def multi_signals_unpack_data(
        data: Optional[
            Union[
                ArrayLike,
                DataFrame,
                dict,
                MultiSignals,
                Sequence,
                Series,
                Signal,
            ]
        ] = None,
        domain: Optional[ArrayLike] = None,
        labels: Optional[Sequence] = None,
        dtype: Optional[Type[DTypeFloating]] = None,
        signal_type: Type[Signal] = Signal,
        **kwargs: Any,
    ) -> Dict[str, Signal]:
        """
        Unpack given data for multi-continuous signals instantiation.

        Parameters
        ----------
        data
            Data to unpack for multi-continuous signals instantiation.
        domain
            Values to initialise the multiple :class:`colour.continuous.Signal`
            sub-class instances :attr:`colour.continuous.Signal.domain`
            attribute with. If both ``data`` and ``domain`` arguments are
            defined, the latter will be used to initialise the
            :attr:`colour.continuous.Signal.domain` property.
        labels
            Names to use for the :class:`colour.continuous.Signal` sub-class
            instances.
        dtype
            Floating point data type.
        signal_type
            A :class:`colour.continuous.Signal` sub-class type.

        Other Parameters
        ----------------
        extrapolator
            Extrapolator class type to use as extrapolating function for the
            :class:`colour.continuous.Signal` sub-class instances.
        extrapolator_kwargs
            Arguments to use when instantiating the extrapolating function
            of the :class:`colour.continuous.Signal` sub-class instances.
        interpolator
            Interpolator class type to use as interpolating function for the
            :class:`colour.continuous.Signal` sub-class instances.
        interpolator_kwargs
            Arguments to use when instantiating the interpolating function
            of the :class:`colour.continuous.Signal` sub-class instances.
        name
            multi-continuous signals name.

        Returns
        -------
        :class:`dict`
            Mapping of labeled :class:`colour.continuous.Signal` sub-class
            instances.

        Examples
        --------
        Unpacking using implicit *domain* and data for a single signal:

        >>> range_ = np.linspace(10, 100, 10)
        >>> signals = MultiSignals.multi_signals_unpack_data(range_)
        >>> list(signals.keys())
        ['0']
        >>> print(signals['0'])
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
        ['0']
        >>> print(signals['0'])
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

        Unpacking using a *dict*:

        >>> signals = MultiSignals.multi_signals_unpack_data(
        ...     dict(zip(domain, range_)))
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

        Unpacking using a *Pandas* :class:`pandas.DataFrame`:

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

        dtype = cast(Type[DTypeFloating], optional(dtype, DEFAULT_FLOAT_DTYPE))

        settings = {}
        settings.update(kwargs)
        settings.update({"dtype": dtype})

        # domain_unpacked, range_unpacked, signals = (
        #   np.array([]), np.array([]), {})

        signals = {}

        if isinstance(data, Signal):
            signals[data.name] = data
        elif isinstance(data, MultiSignals):
            signals = data.signals
        elif issubclass(type(data), Sequence) or isinstance(
            data, (tuple, list, np.ndarray, Iterator, ValuesView)
        ):
            data_sequence = list(data)  # type: ignore[arg-type]

            is_signal = all(
                [
                    True if isinstance(i, Signal) else False
                    for i in data_sequence
                ]
            )

            if is_signal:
                for signal in data_sequence:
                    signals[signal.name] = signal_type(
                        signal.range, signal.domain, **settings
                    )
            else:
                data_array = tsplit(data_sequence)
                attest(
                    data_array.ndim in (1, 2),
                    'User "data" must be 1-dimensional or 2-dimensional!',
                )

                if data_array.ndim == 1:
                    data_array = data_array[np.newaxis, :]

                for i, range_unpacked in enumerate(data_array):
                    signals[str(i)] = signal_type(
                        range_unpacked, domain, **settings
                    )
        elif issubclass(type(data), Mapping) or isinstance(data, dict):
            data_mapping = dict(data)  # type: ignore[arg-type]

            is_signal = all(
                [
                    True if isinstance(i, Signal) else False
                    for i in data_mapping.values()
                ]
            )

            if is_signal:
                for label, signal in data_mapping.items():
                    signals[label] = signal_type(
                        signal.range, signal.domain, **settings
                    )
            else:
                domain_unpacked, range_unpacked = zip(
                    *sorted(data_mapping.items())
                )
                for i, range_unpacked in enumerate(tsplit(range_unpacked)):
                    signals[str(i)] = signal_type(
                        range_unpacked, domain_unpacked, **settings
                    )
        elif is_pandas_installed():
            if isinstance(data, Series):
                signals["0"] = signal_type(data, **settings)
            elif isinstance(data, DataFrame):
                domain_unpacked = data.index.values
                signals = {
                    label: signal_type(
                        data[label], domain_unpacked, **settings
                    )
                    for label in data
                }

        if domain is not None:
            domain_array = as_float_array(list(domain), dtype)  # type: ignore[arg-type]

            for signal in signals.values():
                attest(
                    len(domain_array) == len(signal.domain),
                    'User "domain" length is not compatible with unpacked '
                    '"signals"!',
                )

                signal.domain = domain_array

        signals = {str(label): signal for label, signal in signals.items()}

        if labels is not None:
            attest(
                len(labels) == len(signals),
                'User "labels" length is not compatible with unpacked '
                '"signals"!',
            )

            signals = {
                str(labels[i]): signal
                for i, signal in enumerate(signals.values())
            }

        for label in signals:
            signals[label].name = label

        if not signals:
            signals = {"Undefined": Signal(name="Undefined")}

        return signals

    def fill_nan(
        self,
        method: Union[
            Literal["Constant", "Interpolation"], str
        ] = "Interpolation",
        default: Number = 0,
    ) -> AbstractContinuousFunction:
        """
        Fill NaNs in independent domain variable :math:`x` and corresponding
        range variable :math:`y` using given method.

        Parameters
        ----------
        method
            *Interpolation* method linearly interpolates through the NaNs,
            *Constant* method replaces NaNs with ``default``.
        default
            Value to use with the *Constant* method.

        Returns
        -------
        :class:`colour.continuous.MultiSignals`
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

        method = validate_method(method, ["Interpolation", "Constant"])

        for signal in self._signals.values():
            signal.fill_nan(method, default)

        return self

    @required("Pandas")
    def to_dataframe(self) -> DataFrame:
        """
        Convert the continuous signal to a *Pandas* :class:`pandas.DataFrame`
        class instance.

        Returns
        -------
        :class:`pandas.DataFrame`
            Continuous signal as a *Pandas* :class:`pandas.DataFrame` class
            instance.

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

        return DataFrame(
            data=self.range, index=self.domain, columns=self.labels
        )
