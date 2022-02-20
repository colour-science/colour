"""
Signal
======

Defines the class implementing support for continuous signal:

-   :class:`colour.continuous.Signal`
"""

from __future__ import annotations

import numpy as np
from operator import (
    add,
    mul,
    pow,
    sub,
    truediv,
    iadd,
    imul,
    ipow,
    isub,
    itruediv,
)
from collections.abc import Iterator, Mapping, Sequence, ValuesView
from colour.algebra import Extrapolator, KernelInterpolator
from colour.constants import DEFAULT_FLOAT_DTYPE
from colour.continuous import AbstractContinuousFunction
from colour.hints import (
    Any,
    ArrayLike,
    Boolean,
    Callable,
    DTypeFloating,
    Dict,
    FloatingOrArrayLike,
    FloatingOrNDArray,
    Integer,
    Literal,
    NDArray,
    Number,
    Optional,
    Tuple,
    Type,
    TypeExtrapolator,
    TypeInterpolator,
    Union,
    cast,
)
from colour.utilities import (
    as_float_array,
    attest,
    fill_nan,
    full,
    is_pandas_installed,
    optional,
    required,
    runtime_warning,
    tsplit,
    tstack,
    validate_method,
)
from colour.utilities.documentation import is_documentation_building

if is_pandas_installed():
    from pandas import Series
else:  # pragma: no cover
    from unittest import mock

    Series = mock.MagicMock()

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "Signal",
]


class Signal(AbstractContinuousFunction):
    """
    Define the base class for continuous signal.

    The class implements the :meth:`Signal.function` method so that evaluating
    the function for any independent domain variable :math:`x \\in\\mathbb{R}`
    returns a corresponding range variable :math:`y \\in\\mathbb{R}`.
    It adopts an interpolating function encapsulated inside an extrapolating
    function. The resulting function independent domain, stored as discrete
    values in the :attr:`colour.continuous.Signal.domain` property corresponds
    with the function dependent and already known range stored in the
    :attr:`colour.continuous.Signal.range` property.

    .. important::

        Specific documentation about getting, setting, indexing and slicing the
        continuous signal values is available in the
        :ref:`spectral-representation-and-continuous-signal` section.

    Parameters
    ----------
    data
        Data to be stored in the continuous signal.
    domain
        Values to initialise the :attr:`colour.continuous.Signal.domain`
        attribute with. If both ``data`` and ``domain`` arguments are defined,
        the latter with be used to initialise the
        :attr:`colour.continuous.Signal.domain` property.

    Other Parameters
    ----------------
    dtype
        Floating point data type.
    extrapolator
        Extrapolator class type to use as extrapolating function.
    extrapolator_kwargs
        Arguments to use when instantiating the extrapolating function.
    interpolator
        Interpolator class type to use as interpolating function.
    interpolator_kwargs
        Arguments to use when instantiating the interpolating function.
    name
        Continuous signal name.

    Attributes
    ----------
    -   :attr:`~colour.continuous.Signal.dtype`
    -   :attr:`~colour.continuous.Signal.domain`
    -   :attr:`~colour.continuous.Signal.range`
    -   :attr:`~colour.continuous.Signal.interpolator`
    -   :attr:`~colour.continuous.Signal.interpolator_kwargs`
    -   :attr:`~colour.continuous.Signal.extrapolator`
    -   :attr:`~colour.continuous.Signal.extrapolator_kwargs`
    -   :attr:`~colour.continuous.Signal.function`

    Methods
    -------
    -   :meth:`~colour.continuous.Signal.__init__`
    -   :meth:`~colour.continuous.Signal.__str__`
    -   :meth:`~colour.continuous.Signal.__repr__`
    -   :meth:`~colour.continuous.Signal.__hash__`
    -   :meth:`~colour.continuous.Signal.__getitem__`
    -   :meth:`~colour.continuous.Signal.__setitem__`
    -   :meth:`~colour.continuous.Signal.__contains__`
    -   :meth:`~colour.continuous.Signal.__eq__`
    -   :meth:`~colour.continuous.Signal.__ne__`
    -   :meth:`~colour.continuous.Signal.arithmetical_operation`
    -   :meth:`~colour.continuous.Signal.signal_unpack_data`
    -   :meth:`~colour.continuous.Signal.fill_nan`
    -   :meth:`~colour.continuous.Signal.to_series`

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

    Instantiation with a *Pandas* :class:`pandas.Series`:

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

    def __init__(
        self,
        data: Optional[Union[ArrayLike, dict, Series, Signal]] = None,
        domain: Optional[ArrayLike] = None,
        **kwargs: Any,
    ):
        super().__init__(kwargs.get("name"))

        self._dtype: Type[DTypeFloating] = DEFAULT_FLOAT_DTYPE
        self._domain: NDArray = np.array([])
        self._range: NDArray = np.array([])
        self._interpolator: Type[TypeInterpolator] = KernelInterpolator
        self._interpolator_kwargs: Dict = {}
        self._extrapolator: Type[TypeExtrapolator] = Extrapolator
        self._extrapolator_kwargs: Dict = {
            "method": "Constant",
            "left": np.nan,
            "right": np.nan,
        }

        self.domain, self.range = self.signal_unpack_data(data, domain)

        self.dtype = kwargs.get("dtype", self._dtype)

        self.interpolator = kwargs.get("interpolator", self._interpolator)
        self.interpolator_kwargs = kwargs.get(
            "interpolator_kwargs", self._interpolator_kwargs
        )
        self.extrapolator = kwargs.get("extrapolator", self._extrapolator)
        self.extrapolator_kwargs = kwargs.get(
            "extrapolator_kwargs", self._extrapolator_kwargs
        )

        self._create_function()

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
        DTypeFloating
            Continuous signal dtype.
        """

        return self._dtype

    @dtype.setter
    def dtype(self, value: Type[DTypeFloating]):
        """Setter for the **self.dtype** property."""

        attest(
            value in np.sctypes["float"],
            f'"dtype" must be one of the following types: '
            f"{np.sctypes['float']}",
        )

        self._dtype = value

        # The following self-assignments are written as intended and
        # triggers the rebuild of the underlying function.
        self.domain = self.domain
        self.range = self.range

    @property
    def domain(self) -> NDArray:
        """
        Getter and setter property for the continuous signal independent
        domain variable :math:`x`.

        Parameters
        ----------
        value
            Value to set the continuous signal independent domain
            variable :math:`x` with.

        Returns
        -------
        :class:`numpy.ndarray`
            Continuous signal independent domain variable :math:`x`.
        """

        return np.copy(self._domain)

    @domain.setter
    def domain(self, value: ArrayLike):
        """Setter for the **self.domain** property."""

        value = as_float_array(value, self.dtype)

        if not np.all(np.isfinite(value)):
            runtime_warning(
                f'"{self.name}" new "domain" variable is not finite: {value}, '
                f"unpredictable results may occur!"
            )

        if value.size != self._range.size:
            self._range = np.resize(self._range, value.shape)

        self._domain = value
        self._create_function()

    @property
    def range(self) -> NDArray:
        """
        Getter and setter property for the continuous signal corresponding
        range variable :math:`y`.

        Parameters
        ----------
        value
            Value to set the continuous signal corresponding range :math:`y`
            variable with.

        Returns
        -------
        :class:`numpy.ndarray`
            Continuous signal corresponding range variable :math:`y`.
        """

        return np.copy(self._range)

    @range.setter
    def range(self, value: ArrayLike):
        """Setter for the **self.range** property."""

        value = as_float_array(value, self.dtype)

        if not np.all(np.isfinite(value)):
            runtime_warning(
                f'"{self.name}" new "range" variable is not finite: {value}, '
                f"unpredictable results may occur!"
            )

        attest(
            value.size == self._domain.size,
            '"domain" and "range" variables must have same size!',
        )

        self._range = value
        self._create_function()

    @property
    def interpolator(self) -> Type[TypeInterpolator]:
        """
        Getter and setter property for the continuous signal interpolator type.

        Parameters
        ----------
        value
            Value to set the continuous signal interpolator type
            with.

        Returns
        -------
        Type[TypeInterpolator]
            Continuous signal interpolator type.
        """

        return self._interpolator

    @interpolator.setter
    def interpolator(self, value: Type[TypeInterpolator]):
        """Setter for the **self.interpolator** property."""

        # TODO: Check for interpolator compatibility.
        self._interpolator = value
        self._create_function()

    @property
    def interpolator_kwargs(self) -> Dict:
        """
        Getter and setter property for the continuous signal interpolator
        instantiation time arguments.

        Parameters
        ----------
        value
            Value to set the continuous signal interpolator instantiation
            time arguments to.

        Returns
        -------
        :class:`dict`
            Continuous signal interpolator instantiation time
            arguments.
        """

        return self._interpolator_kwargs

    @interpolator_kwargs.setter
    def interpolator_kwargs(self, value: dict):
        """Setter for the **self.interpolator_kwargs** property."""

        attest(
            isinstance(value, dict),
            f'"interpolator_kwargs" property: "{value}" type is not "dict"!',
        )

        self._interpolator_kwargs = value
        self._create_function()

    @property
    def extrapolator(self) -> Type[TypeExtrapolator]:
        """
        Getter and setter property for the continuous signal extrapolator type.

        Parameters
        ----------
        value
            Value to set the continuous signal extrapolator type
            with.

        Returns
        -------
        Type[TypeExtrapolator]
            Continuous signal extrapolator type.
        """

        return self._extrapolator

    @extrapolator.setter
    def extrapolator(self, value: Type[TypeExtrapolator]):
        """Setter for the **self.extrapolator** property."""

        # TODO: Check for extrapolator compatibility.
        self._extrapolator = value
        self._create_function()

    @property
    def extrapolator_kwargs(self) -> Dict:
        """
        Getter and setter property for the continuous signal extrapolator
        instantiation time arguments.

        Parameters
        ----------
        value
            Value to set the continuous signal extrapolator instantiation
            time arguments to.

        Returns
        -------
        :class:`dict`
            Continuous signal extrapolator instantiation time
            arguments.
        """

        return self._extrapolator_kwargs

    @extrapolator_kwargs.setter
    def extrapolator_kwargs(self, value: dict):
        """Setter for the **self.extrapolator_kwargs** property."""

        attest(
            isinstance(value, dict),
            f'"extrapolator_kwargs" property: "{value}" type is not "dict"!',
        )

        self._extrapolator_kwargs = value
        self._create_function()

    @property
    def function(self) -> Callable:
        """
        Getter property for the continuous signal callable.

        Returns
        -------
        Callable
            Continuous signal callable.
        """

        return self._function

    def __str__(self) -> str:
        """
        Return a formatted string representation of the continuous signal.

        Returns
        -------
        :class:`str`
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
            return super().__str__()

    def __repr__(self) -> str:
        """
        Return an evaluable string representation of the continuous signal.

        Returns
        -------
        :class:`str`
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
               interpolator_kwargs={},
               extrapolator=Extrapolator,
               extrapolator_kwargs={...})
        """

        if is_documentation_building():  # pragma: no cover
            return f"{self.__class__.__name__}(name='{self.name}', ...)"

        try:
            representation = repr(tstack([self.domain, self.range]))
            representation = representation.replace(
                "array", self.__class__.__name__
            )
            representation = representation.replace(
                "       [",
                f"{' ' * (len(self.__class__.__name__) + 2)}[",
            )
            indentation = " " * (len(self.__class__.__name__) + 1)
            representation = (
                f"{representation[:-1]},\n"
                f"{indentation}interpolator={self.interpolator.__name__},\n"
                f"{indentation}interpolator_kwargs="
                f"{repr(self.interpolator_kwargs)},\n"
                f"{indentation}extrapolator={self.extrapolator.__name__},\n"
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

    def __contains__(self, x: Union[FloatingOrArrayLike, slice]) -> bool:
        """
        Return whether the continuous signal contains given independent domain
        variable :math:`x`.

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
        >>> signal = Signal(range_)
        >>> 0 in signal
        True
        >>> 0.5 in signal
        True
        >>> 1000 in signal
        False
        """

        return bool(
            np.all(
                np.where(
                    np.logical_and(
                        x >= np.min(self._domain), x <= np.max(self._domain)
                    ),
                    True,
                    False,
                )
            )
        )

    def __eq__(self, other: Any) -> bool:
        """
        Return whether the continuous signal is equal to given other object.

        Parameters
        ----------
        other
            Object to test whether it is equal to the continuous signal.

        Returns
        -------
        :class:`bool`
            Whether given object is equal to the continuous signal.

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
            return all(
                [
                    np.array_equal(self._domain, other.domain),
                    np.array_equal(self._range, other.range),
                    self._interpolator is other.interpolator,
                    self._interpolator_kwargs == other.interpolator_kwargs,
                    self._extrapolator is other.extrapolator,
                    self._extrapolator_kwargs == other.extrapolator_kwargs,
                ]
            )
        else:
            return False

    def __ne__(self, other: Any) -> bool:
        """
        Return whether the continuous signal is not equal to given other
        object.

        Parameters
        ----------
        other
            Object to test whether it is not equal to the continuous signal.

        Returns
        -------
        :class:`bool`
            Whether given object is not equal to the continuous signal.

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
        """Create the continuous signal underlying function."""

        if self._domain.size != 0 and self._range.size != 0:
            self._function = self._extrapolator(
                self._interpolator(
                    self.domain, self.range, **self._interpolator_kwargs
                ),
                **self._extrapolator_kwargs,
            )
        else:

            def _undefined_function(*args: Any, **kwargs: Any):
                """
                Raise a :class:`ValueError` exception.

                Other Parameters
                ----------------
                args
                    Arguments.
                kwargs
                    Keywords arguments.

                Raises
                ------
                ValueError
                """

                raise ValueError(
                    "Underlying signal interpolator function does not exists, "
                    "please ensure you defined both "
                    '"domain" and "range" variables!'
                )

            self._function = _undefined_function

    def _fill_domain_nan(
        self,
        method: Union[
            Literal["Constant", "Interpolation"], str
        ] = "Interpolation",
        default: Number = 0,
    ):
        """
        Fill NaNs in independent domain variable :math:`x` using given method.

        Parameters
        ----------
        method
            *Interpolation* method linearly interpolates through the NaNs,
            *Constant* method replaces NaNs with ``default``.
        default
            Value to use with the *Constant* method.

        Returns
        -------
        :class:`colour.continuous.Signal`
            NaNs filled continuous signal independent domain :math:`x`
            variable.
        """

        self._domain = fill_nan(self._domain, method, default)
        self._create_function()

    def _fill_range_nan(
        self,
        method: Union[
            Literal["Constant", "Interpolation"], str
        ] = "Interpolation",
        default: Number = 0,
    ):
        """
        Fill NaNs in corresponding range variable :math:`y` using given method.

        Parameters
        ----------
        method
            *Interpolation* method linearly interpolates through the NaNs,
            *Constant* method replaces NaNs with ``default``.
        default
            Value to use with the *Constant* method.

        Returns
        -------
        :class:`colour.continuous.Signal`
            NaNs filled continuous signal i corresponding range :math:`y`
            variable.
        """

        self._range = fill_nan(self._range, method, default)
        self._create_function()

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
        :class:`colour.continuous.Signal`
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

        Adding an `ArrayLike` variable:

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

        operator, ioperator = {
            "+": (add, iadd),
            "-": (sub, isub),
            "*": (mul, imul),
            "/": (truediv, itruediv),
            "**": (pow, ipow),
        }[operation]

        if in_place:
            if isinstance(a, Signal):
                self[self._domain] = operator(self._range, a[self._domain])
                exclusive_or = np.setxor1d(self._domain, a.domain)
                self[exclusive_or] = full(exclusive_or.shape, np.nan)
            else:
                self.range = ioperator(self.range, a)

            return self
        else:
            copy = ioperator(self.copy(), a)

            return copy

    @staticmethod
    def signal_unpack_data(
        data=Optional[Union[ArrayLike, dict, Series, "Signal"]],
        domain: Optional[ArrayLike] = None,
        dtype: Optional[Type[DTypeFloating]] = None,
    ) -> Tuple:
        """
        Unpack given data for continuous signal instantiation.

        Parameters
        ----------
        data
            Data to unpack for continuous signal instantiation.
        domain
            Values to initialise the :attr:`colour.continuous.Signal.domain`
            attribute with. If both ``data`` and ``domain`` arguments are
            defined, the latter will be used to initialise the
            :attr:`colour.continuous.Signal.domain` property.
        dtype
            Floating point data type.

        Returns
        -------
        :class:`tuple`
            Independent domain variable :math:`x` and corresponding range
            variable :math:`y` unpacked for continuous signal instantiation.

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

        Unpacking using a *Pandas* :class:`pandas.Series`:

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

        dtype = cast(Type[DTypeFloating], optional(dtype, DEFAULT_FLOAT_DTYPE))

        domain_unpacked: NDArray = np.array([])
        range_unpacked: NDArray = np.array([])

        if isinstance(data, Signal):
            domain_unpacked = data.domain
            range_unpacked = data.range
        elif issubclass(type(data), Sequence) or isinstance(
            data, (tuple, list, np.ndarray, Iterator, ValuesView)
        ):
            data_array = tsplit(list(data))

            attest(data_array.ndim == 1, 'User "data" must be 1-dimensional!')

            domain_unpacked, range_unpacked = (
                np.arange(0, data_array.size, dtype=dtype),
                data_array,
            )
        elif issubclass(type(data), Mapping) or isinstance(data, dict):
            domain_unpacked, range_unpacked = tsplit(sorted(data.items()))
        elif is_pandas_installed():
            if isinstance(data, Series):
                domain_unpacked = data.index.values
                range_unpacked = data.values

        if domain is not None:
            domain_array = as_float_array(list(domain), dtype)  # type: ignore[arg-type]

            attest(
                len(domain_array) == len(range_unpacked),
                'User "domain" length is not compatible with unpacked "range"!',
            )

            domain_unpacked = domain_array

        if range_unpacked is not None:
            range_unpacked = as_float_array(range_unpacked, dtype)

        return domain_unpacked, range_unpacked

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
        :class:`colour.continuous.Signal`
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

        method = validate_method(method, ["Interpolation", "Constant"])

        self._fill_domain_nan(method, default)
        self._fill_range_nan(method, default)

        return self

    @required("Pandas")
    def to_series(self) -> Series:
        """
        Convert the continuous signal to a *Pandas* :class:`pandas.Series`
        class instance.

        Returns
        -------
        :class:`pandas.Series`
            Continuous signal as a *Pandas*:class:`pandas.Series` class
            instance.

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

        return Series(data=self._range, index=self._domain, name=self.name)
