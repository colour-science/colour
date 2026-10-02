"""
Multi-Signals
=============

Define multi-continuous signal support for colour science computations.

This module provides the :class:`colour.continuous.MultiSignals` class for
representing and operating on multiple continuous signals simultaneously,
supporting interpolation and extrapolation operations.

-   :class:`colour.continuous.MultiSignals`
"""

from __future__ import annotations

import typing
from collections.abc import Iterator, KeysView, Mapping, ValuesView
from operator import pow  # noqa: A004
from operator import add, iadd, imul, ipow, isub, itruediv, mul, sub, truediv

import numpy as np

from colour.algebra import Extrapolator, LinearInterpolator
from colour.constants import DTYPE_FLOAT_DEFAULT
from colour.continuous import AbstractContinuousFunction, Signal

if typing.TYPE_CHECKING:
    from colour.hints import (
        Any,
        List,
        Literal,
        NDArrayFloat,
        ProtocolArrayNamespace,
        ProtocolExtrapolator,
        ProtocolInterpolator,
        Real,
        Self,
        Sequence,
        Type,
    )

from colour.hints import ArrayLike, Callable, DTypeFloat, Sequence, cast
from colour.utilities import (
    array_namespace,
    as_float_array,
    as_ndarray,
    attest,
    fill_nan,
    full,
    int_digest,
    is_iterable,
    is_non_ndarray,
    is_pandas_installed,
    multiline_repr,
    ndarray_copy,
    ndarray_copy_enable,
    optional,
    required,
    runtime_warning,
    tstack,
    validate_method,
    xp_as_array,
    xp_as_float_array,
    xp_astype,
    xp_atleast_1d,
    xp_broadcast_to,
    xp_insert,
    xp_isin,
    xp_reshape,
    xp_resize,
    xp_setxor1d,
)
from colour.utilities.documentation import is_documentation_building

if typing.TYPE_CHECKING or is_pandas_installed():
    from pandas import DataFrame, Series  # pragma: no cover
else:  # pragma: no cover
    from unittest import mock

    DataFrame = mock.MagicMock()
    Series = mock.MagicMock()

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "MultiSignals",
]


class MultiSignals(AbstractContinuousFunction):
    """
    Define the base class for multi-signals, a container for
    multiple :class:`colour.continuous.Signal` sub-class instances.

    .. important::

        Specific documentation about getting, setting, indexing and slicing
        the multi-signals values is available in the
        :ref:`spectral-representation-and-continuous-signal` section.

    Parameters
    ----------
    data
        Data to be stored in the multi-signals.
    domain
        Values to initialise the multiple :class:`colour.continuous.Signal`
        sub-class instances :attr:`colour.continuous.Signal.domain`
        attribute with. If both ``data`` and ``domain`` arguments are
        defined, the latter will be used to initialise the
        :attr:`colour.continuous.Signal.domain` attribute.
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
        Arguments to use when instantiating the extrapolating function of
        the :class:`colour.continuous.Signal` sub-class instances.
    interpolator
        Interpolator class type to use as interpolating function for the
        :class:`colour.continuous.Signal` sub-class instances.
    interpolator_kwargs
        Arguments to use when instantiating the interpolating function of
        the :class:`colour.continuous.Signal` sub-class instances.
    name
        Multi-signals name.
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
    [[  0.  10.]
     [  1.  20.]
     [  2.  30.]
     [  3.  40.]
     [  4.  50.]
     [  5.  60.]
     [  6.  70.]
     [  7.  80.]
     [  8.  90.]
     [  9. 100.]]

    Instantiation with explicit *domain* and a single signal:

    >>> domain = np.arange(100, 1100, 100)
    >>> print(MultiSignals(range_, domain))
    [[ 100.   10.]
     [ 200.   20.]
     [ 300.   30.]
     [ 400.   40.]
     [ 500.   50.]
     [ 600.   60.]
     [ 700.   70.]
     [ 800.   80.]
     [ 900.   90.]
     [1000.  100.]]

    Instantiation with multiple signals:

    >>> range_ = tstack([np.linspace(10, 100, 10)] * 3)
    >>> range_ += np.array([0, 10, 20])
    >>> print(MultiSignals(range_, domain))
    [[ 100.   10.   20.   30.]
     [ 200.   20.   30.   40.]
     [ 300.   30.   40.   50.]
     [ 400.   40.   50.   60.]
     [ 500.   50.   60.   70.]
     [ 600.   60.   70.   80.]
     [ 700.   70.   80.   90.]
     [ 800.   80.   90.  100.]
     [ 900.   90.  100.  110.]
     [1000.  100.  110.  120.]]

    Instantiation with a *dict*:

    >>> print(MultiSignals(dict(zip(domain, range_))))
    [[ 100.   10.   20.   30.]
     [ 200.   20.   30.   40.]
     [ 300.   30.   40.   50.]
     [ 400.   40.   50.   60.]
     [ 500.   50.   60.   70.]
     [ 600.   60.   70.   80.]
     [ 700.   70.   80.   90.]
     [ 800.   80.   90.  100.]
     [ 900.   90.  100.  110.]
     [1000.  100.  110.  120.]]

    Instantiation using a *Signal* sub-class:

    >>> class NotSignal(Signal):
    ...     pass

    >>> multi_signals = MultiSignals(range_, domain, signal_type=NotSignal)
    >>> print(multi_signals)
    [[ 100.   10.   20.   30.]
     [ 200.   20.   30.   40.]
     [ 300.   30.   40.   50.]
     [ 400.   40.   50.   60.]
     [ 500.   50.   60.   70.]
     [ 600.   60.   70.   80.]
     [ 700.   70.   80.   90.]
     [ 800.   80.   90.  100.]
     [ 900.   90.  100.  110.]
     [1000.  100.  110.  120.]]
     >>> type(multi_signals.signals[0])  # doctest: +SKIP
     <class 'multi_signals.NotSignal'>

    Instantiation with a *Pandas* `Series`:

    >>> if is_pandas_installed():
    ...     from pandas import Series
    ...
    ...     print(
    ...         MultiSignals(  # doctest: +SKIP
    ...             Series(dict(zip(domain, np.linspace(10, 100, 10))))
    ...         )
    ...     )
    [[ 100.   10.]
     [ 200.   20.]
     [ 300.   30.]
     [ 400.   40.]
     [ 500.   50.]
     [ 600.   60.]
     [ 700.   70.]
     [ 800.   80.]
     [ 900.   90.]
     [1000.  100.]]

    Instantiation with a *Pandas* :class:`pandas.DataFrame`:

    >>> if is_pandas_installed():
    ...     from pandas import DataFrame
    ...
    ...     data = dict(zip(["a", "b", "c"], tsplit(range_)))
    ...     print(MultiSignals(DataFrame(data, domain)))  # doctest: +SKIP
    [[ 100.   10.   20.   30.]
     [ 200.   20.   30.   40.]
     [ 300.   30.   40.   50.]
     [ 400.   40.   50.   60.]
     [ 500.   50.   60.   70.]
     [ 600.   60.   70.   80.]
     [ 700.   70.   80.   90.]
     [ 800.   80.   90.  100.]
     [ 900.   90.  100.  110.]
     [1000.  100.  110.  120.]]

    Retrieving domain *y* variable for arbitrary range *x* variable:

    >>> x = 150
    >>> range_ = tstack([np.sin(np.linspace(0, 1, 10))] * 3)
    >>> range_ += np.array([0.0, 0.25, 0.5])
    >>> MultiSignals(range_, domain)[x]  # doctest: +ELLIPSIS
    array([0.0554413..., 0.3054413..., 0.5554413...])
    >>> x = np.linspace(100, 1000, 3)
    >>> MultiSignals(range_, domain)[x]  # doctest: +ELLIPSIS
    array([[0.        , 0.25      , 0.5       ],
           [0.4786858..., 0.7286858..., 0.9786858...],
           [0.8414709..., 1.0914709..., 1.3414709...]])

    Using an alternative interpolating function:

    >>> x = 150
    >>> from colour.algebra import CubicSplineInterpolator
    >>> MultiSignals(range_, domain, interpolator=CubicSplineInterpolator)[
    ...     x
    ... ]  # doctest: +ELLIPSIS
    array([0.0555274..., 0.3055274..., 0.5555274...])
    >>> x = np.linspace(100, 1000, 3)
    >>> MultiSignals(range_, domain, interpolator=CubicSplineInterpolator)[
    ...     x
    ... ]  # doctest: +ELLIPSIS
    array([[0.       ..., 0.25     ..., 0.5      ...],
           [0.4794253..., 0.7294253..., 0.9794253...],
           [0.8414709..., 1.0914709..., 1.3414709...]])
    """

    def __init__(
        self,
        data: (
            ArrayLike
            | DataFrame
            | dict
            | Self
            | Sequence
            | Series
            | Signal
            | ValuesView
            | None
        ) = None,
        domain: ArrayLike | KeysView | None = None,
        labels: Sequence | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(kwargs.get("name"))

        self._signal_type: Type[Signal] = kwargs.get("signal_type", Signal)
        self._dtype: Type[DTypeFloat] = kwargs.get("dtype", DTYPE_FLOAT_DEFAULT)

        # Canonical storage owned at the parent level.
        self._domain: NDArrayFloat = as_float_array([], self._dtype)
        self._range: NDArrayFloat = np.zeros((0, 0), dtype=self._dtype)
        self._labels: List[str] = []

        # Interpolator / extrapolator parameters mirror :class:`Signal`'s
        # defaults; sub-classes (e.g. :class:`MultiSpectralDistributions`)
        # override via ``kwargs``.
        self._interpolator: Type[ProtocolInterpolator] = kwargs.get(
            "interpolator", LinearInterpolator
        )
        self._interpolator_kwargs: dict = kwargs.get("interpolator_kwargs", {})
        self._extrapolator: Type[ProtocolExtrapolator] = kwargs.get(
            "extrapolator", Extrapolator
        )
        self._extrapolator_kwargs: dict = kwargs.get(
            "extrapolator_kwargs",
            {"method": "Constant", "left": float("nan"), "right": float("nan")},
        )

        # ``multi_signals_unpack_data`` normalises every supported input
        # format to the canonical ``(domain, range, labels)`` triple; one
        # ``isfinite`` check on the full 2-D ``range`` replaces the prior
        # per-child fan-out validation.
        self._domain, self._range, self._labels = self.multi_signals_unpack_data(
            data, domain, labels, dtype=self._dtype, **kwargs
        )

        if len(self._range) > 0 and self._range.shape[-1] > 0:
            xp = array_namespace(self._range)
            # Promote ``_domain`` onto ``_range``'s backend so they live on
            # the same device for downstream evaluation; mirrors the per-
            # signal promotion in :class:`Signal.function`.
            self._domain = xp_as_float_array(self._domain, xp=xp, like=self._range)
            if not bool(xp.all(xp.isfinite(self._range))):
                runtime_warning(
                    f'"{self.name}" new "range" variable is not finite: '
                    f"{self._range}, unpredictable results may occur!"
                )

        # Per-column :class:`Signal` copies are materialised on demand by
        # the :attr:`signals` property; the canonical state lives at the
        # parent level.
        self._function: Callable | None = None

    @property
    def dtype(self) -> Type[DTypeFloat]:
        """
        Getter and setter for the multi-signals dtype.

        Parameters
        ----------
        value
            Value to set the multi-signals dtype with.

        Returns
        -------
        Type[DTypeFloat]
            Multi-signals dtype.
        """

        return self._dtype

    @dtype.setter
    def dtype(self, value: Type[DTypeFloat]) -> None:
        """Setter for the **self.dtype** property."""

        attest(
            value in DTypeFloat.__args__,
            f'"dtype" must be one of the following types: {DTypeFloat.__args__}',
        )

        self._dtype = value

        # The following self-assignments are written as intended and
        # triggers the rebuild of the underlying function.
        if self.domain.dtype != value or self.range.dtype != value:
            self.domain = self.domain
            self.range = self.range

    @property
    def domain(self) -> NDArrayFloat:
        """
        Getter and setter for the multi-signals' independent
        domain variable :math:`x`.

        Parameters
        ----------
        value
            Value to set the multi-signals independent domain
            variable :math:`x` with.

        Returns
        -------
        :class:`numpy.ndarray`
            Multi-signals independent domain variable
            :math:`x`.
        """

        return ndarray_copy(self._domain)

    @domain.setter
    def domain(self, value: ArrayLike) -> None:
        """Setter for the **self.domain** property."""

        value = as_float_array(value, self.dtype)

        xp = array_namespace(value)

        if not xp.all(xp.isfinite(value)):
            runtime_warning(
                f'"{self.name}" new "domain" variable is not finite: {value}, '
                f"unpredictable results may occur!"
            )
        else:
            attest(
                xp.all(value[:-1] <= value[1:]),
                "The new domain value is not monotonic! ",
            )

        if len(value) != self._range.shape[0]:
            xp = array_namespace(self._range)

            self._range = xp_resize(
                self._range, (len(value), self._range.shape[1]), xp=xp
            )

        self._domain = value
        self._function = None  # Invalidate the underlying continuous function.

    @property
    def range(self) -> NDArrayFloat:
        """
        Getter and setter for the multi-signals' range
        variable :math:`y`.

        Parameters
        ----------
        value
            Value to set the multi-signals' range variable
            :math:`y` with.

        Returns
        -------
        :class:`numpy.ndarray`
            Multi-signals' range variable :math:`y`.
        """

        return ndarray_copy(self._range)

    @range.setter
    def range(self, value: ArrayLike) -> None:
        """Setter for the **self.range** property."""

        value = as_float_array(value, self.dtype)

        xp = array_namespace(value)

        if not xp.all(xp.isfinite(value)):
            runtime_warning(
                f'"{self.name}" new "range" variable is not finite: {value}, '
                f"unpredictable results may occur!"
            )

        if value.ndim in (0, 1):
            value = xp_broadcast_to(
                value[..., None] if value.ndim == 1 else value,
                self._range.shape,
                xp=xp,
            )
            value = as_float_array(value, self.dtype)
        else:
            attest(
                value.shape[-1] == self._range.shape[1],
                'Corresponding "y" variable columns must have '
                'same count than underlying "Signal" components!',
            )

        self._range = value
        self._function = None  # Invalidate the underlying continuous function.

    @property
    def interpolator(self) -> Type[ProtocolInterpolator]:
        """
        Getter and setter for the multi-signals interpolator
        type.

        Parameters
        ----------
        value
            Value to set the multi-signals interpolator type
            with.

        Returns
        -------
        Type[ProtocolInterpolator]
            Multi-signals interpolator type.
        """

        return self._interpolator

    @interpolator.setter
    def interpolator(self, value: Type[ProtocolInterpolator]) -> None:
        """Setter for the **self.interpolator** property."""

        if value is not None and value is not self._interpolator:
            self._interpolator = value
            self._function = None  # Invalidate the underlying continuous function.

    @property
    def interpolator_kwargs(self) -> dict:
        """
        Getter and setter for the interpolator instantiation time arguments.

        Parameters
        ----------
        value
            Value to set the multi-signals interpolator
            instantiation time arguments to.

        Returns
        -------
        :class:`dict`
            Multi-signals interpolator instantiation time
            arguments.
        """

        return self._interpolator_kwargs

    @interpolator_kwargs.setter
    def interpolator_kwargs(self, value: dict) -> None:
        """Setter for the **self.interpolator_kwargs** property."""

        self._interpolator_kwargs = value
        self._function = None  # Invalidate the underlying continuous function.

    @property
    def extrapolator(self) -> Type[ProtocolExtrapolator]:
        """
        Getter and setter for the multi-signals extrapolator
        type.

        Parameters
        ----------
        value
            Value to set the multi-signals extrapolator type
            with.

        Returns
        -------
        Type[ProtocolExtrapolator]
            Multi-signals extrapolator type.
        """

        return self._extrapolator

    @extrapolator.setter
    def extrapolator(self, value: Type[ProtocolExtrapolator]) -> None:
        """Setter for the **self.extrapolator** property."""

        if value is not None and value is not self._extrapolator:
            self._extrapolator = value
            self._function = None  # Invalidate the underlying continuous function.

    @property
    def extrapolator_kwargs(self) -> dict:
        """
        Getter and setter for the multi-signals extrapolator
        instantiation time arguments.

        Parameters
        ----------
        value
            Value to set the multi-signals extrapolator
            instantiation time arguments to.

        Returns
        -------
        :class:`dict`
            Multi-signals extrapolator instantiation time
            arguments.
        """

        return self._extrapolator_kwargs

    @extrapolator_kwargs.setter
    def extrapolator_kwargs(self, value: dict) -> None:
        """Setter for the **self.extrapolator_kwargs** property."""

        self._extrapolator_kwargs = value
        self._function = None  # Invalidate the underlying continuous function.

    @property
    @ndarray_copy_enable(False)
    def function(self) -> Callable:
        """
        Getter for the multi-signals callable.

        Returns
        -------
        Callable
            Multi-signals callable.
        """

        if self._function is None:
            # Create the underlying continuous function. Each interpolator
            # owns its own input conversion, so backend tensors flow
            # straight through array-aware ones (Sprague / Linear / Kernel
            # / Null) and only get coerced to *NumPy* by the scipy-bound
            # ones (CubicSpline / Pchip). ``self.domain`` is promoted to
            # ``self.range``'s backend so the interpolator sees both
            # variables on the same device, and downstream ``like=``
            # references can canonically use the stored ``x`` axis.
            if len(self.domain) != 0 and len(self.range) != 0:
                xp = array_namespace(self.domain, self.range)
                domain = xp_as_float_array(self.domain, xp=xp, like=self.range)
                self._function = self.extrapolator(
                    self.interpolator(
                        domain,
                        self.range,
                        **self.interpolator_kwargs,
                    ),
                    **self.extrapolator_kwargs,
                )
            else:

                def _undefined_function(
                    *args: Any,  # noqa: ARG001
                    **kwargs: Any,  # noqa: ARG001
                ) -> None:
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

                    error = (
                        "Underlying multi-signals interpolator function "
                        'does not exists, please ensure that both "domain" '
                        'and "range" variables are defined!'
                    )

                    raise ValueError(error)

                self._function = cast("Callable", _undefined_function)

        return cast("Callable", self._function)

    @property
    def signals(self) -> Mapping[str, Signal]:
        """
        Getter and setter for the dictionary of
        :class:`colour.continuous.Signal` sub-class instances.

        The canonical state lives at the parent level, in ``self._domain``
        and ``self._range``: a new :class:`colour.continuous.Signal`
        sub-class instance is materialised per label on each access,
        wrapping a column of ``self._range``.

        The returned instances are therefore copies, not views: mutating
        one, e.g. ``multi_signals.signals["a"][560] = 1``, does not
        propagate to the parent. Assign to the parent instead, e.g.
        ``multi_signals[560] = 1``, or set the
        :attr:`colour.continuous.MultiSignals.signals` property.

        Parameters
        ----------
        value
            Dictionary of :class:`colour.continuous.Signal` sub-class
            instances to set.

        Returns
        -------
        :class:`dict`
            Dictionary mapping signal names to their corresponding
            :class:`colour.continuous.Signal` sub-class instances.
        """

        return {
            label: self._signal_type(
                self._range[:, i],
                self._domain,
                name=label,
                dtype=self._dtype,
                interpolator=self._interpolator,
                interpolator_kwargs=self._interpolator_kwargs,
                extrapolator=self._extrapolator,
                extrapolator_kwargs=self._extrapolator_kwargs,
            )
            for i, label in enumerate(self._labels)
        }

    @signals.setter
    def signals(
        self,
        value: ArrayLike | DataFrame | dict | Self | Series | Signal | None,
    ) -> None:
        """Setter for the **self.signals** property."""

        self._domain, self._range, self._labels = self.multi_signals_unpack_data(
            value, dtype=self._dtype
        )
        self._function = None  # Invalidate the underlying continuous function.

    @property
    def labels(self) -> List[str]:
        """
        Getter and setter for the :class:`colour.continuous.Signal` sub-class
        instance names.

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

        return list(self._labels)

    @labels.setter
    def labels(self, value: Sequence) -> None:
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
            len(value) == len(self._labels),
            f'"labels" property: length must be "{len(self._labels)}"!',
        )

        self._labels = [str(label) for label in value]
        self._function = None  # Invalidate the underlying continuous function.

    @property
    def signal_type(self) -> Type[Signal]:
        """
        Getter for the type of :class:`colour.continuous.Signal`
        sub-class instances.

        Returns
        -------
        Type[Signal]
            Type of :class:`colour.continuous.Signal` sub-class
            instances used in this multi-signal collection.
        """

        return self._signal_type

    def __str__(self) -> str:
        """
        Return a formatted string representation of the multi-signals.

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
        [[  0.  10.  20.  30.]
         [  1.  20.  30.  40.]
         [  2.  30.  40.  50.]
         [  3.  40.  50.  60.]
         [  4.  50.  60.  70.]
         [  5.  60.  70.  80.]
         [  6.  70.  80.  90.]
         [  7.  80.  90. 100.]
         [  8.  90. 100. 110.]
         [  9. 100. 110. 120.]]
        """

        xp = array_namespace(self.domain, self.range)

        return str(xp.concat([self.domain[:, None], self.range], axis=1))

    def __repr__(self) -> str:
        """
        Return an evaluable string representation of the multi-signals.

        Returns
        -------
        :class:`str`
            Evaluable string representation.

        Examples
        --------
        >>> domain = np.arange(0, 10, 1)
        >>> range_ = tstack([np.linspace(10, 100, 10)] * 3)
        >>> range_ += np.array([0, 10, 20])
        >>> MultiSignals(range_)
        MultiSignals([[  0.,  10.,  20.,  30.],
                      [  1.,  20.,  30.,  40.],
                      [  2.,  30.,  40.,  50.],
                      [  3.,  40.,  50.,  60.],
                      [  4.,  50.,  60.,  70.],
                      [  5.,  60.,  70.,  80.],
                      [  6.,  70.,  80.,  90.],
                      [  7.,  80.,  90., 100.],
                      [  8.,  90., 100., 110.],
                      [  9., 100., 110., 120.]],
                     ['0', '1', '2'],
                     LinearInterpolator,
                     {},
                     Extrapolator,
                     {'method': 'Constant', 'left': nan, 'right': nan})
        """

        if is_documentation_building():  # pragma: no cover
            return f"{self.__class__.__name__}(name='{self.name}', ...)"

        return multiline_repr(
            self,
            [
                {
                    "formatter": lambda x: repr(  # noqa: ARG005
                        array_namespace(self.domain, self.range).concat(
                            [self.domain[:, None], self.range], axis=1
                        )
                    ),
                },
                {"name": "labels"},
                {
                    "name": "interpolator",
                    "formatter": lambda x: (  # noqa: ARG005
                        self.interpolator.__name__
                    ),
                },
                {"name": "interpolator_kwargs"},
                {
                    "name": "extrapolator",
                    "formatter": lambda x: (  # noqa: ARG005
                        self.extrapolator.__name__
                    ),
                },
                {"name": "extrapolator_kwargs"},
            ],
        )

    def __hash__(self) -> int:
        """
        Compute the hash of the multi-signals.

        Returns
        -------
        :class:`int`
            Object hash.
        """

        # See :meth:`Signal.__hash__` for the host-bytes-plus-namespace rationale.
        return hash(
            (
                int_digest(as_ndarray(self._domain).tobytes()),
                int_digest(as_ndarray(self._range).tobytes()),
                array_namespace(self._domain, self._range).__name__,
                self._interpolator.__name__,
                repr(self._interpolator_kwargs),
                self._extrapolator.__name__,
                repr(self._extrapolator_kwargs),
            )
        )

    @ndarray_copy_enable(False)
    def __getitem__(self, x: ArrayLike | slice) -> NDArrayFloat:
        """
        Return the corresponding range variable :math:`y` for the specified
        independent domain variable :math:`x`.

        Parameters
        ----------
        x
            Independent domain variable :math:`x`.

        Returns
        -------
        :class:`numpy.ndarray`
            Variable :math:`y` range value.

        Examples
        --------
        >>> range_ = tstack([np.linspace(10, 100, 10)] * 3)
        >>> range_ += np.array([0, 10, 20])
        >>> multi_signals = MultiSignals(range_)
        >>> print(multi_signals)
        [[  0.  10.  20.  30.]
         [  1.  20.  30.  40.]
         [  2.  30.  40.  50.]
         [  3.  40.  50.  60.]
         [  4.  50.  60.  70.]
         [  5.  60.  70.  80.]
         [  6.  70.  80.  90.]
         [  7.  80.  90. 100.]
         [  8.  90. 100. 110.]
         [  9. 100. 110. 120.]]
        >>> multi_signals[0]
        array([10., 20., 30.])
        >>> multi_signals[np.array([0, 1, 2])]
        array([[10., 20., 30.],
               [20., 30., 40.],
               [30., 40., 50.]])
        >>> multi_signals[np.linspace(0, 5, 5)]  # doctest: +ELLIPSIS
        array([[10. , 20. , 30. ],
               [22.5, 32.5, 42.5],
               [35. , 45. , 55. ],
               [47.5, 57.5, 67.5],
               [60. , 70. , 80. ]])
        >>> multi_signals[0:3]
        array([[10., 20., 30.],
               [20., 30., 40.],
               [30., 40., 50.]])
        >>> multi_signals[:, 0:2]
        array([[ 10.,  20.],
               [ 20.,  30.],
               [ 30.,  40.],
               [ 40.,  50.],
               [ 50.,  60.],
               [ 60.,  70.],
               [ 70.,  80.],
               [ 80.,  90.],
               [ 90., 100.],
               [100., 110.]])
        """

        x_r, x_c = (x[0], x[1]) if isinstance(x, tuple) else (x, slice(None))

        # The slice path serves directly from the cached 2-D ``_range``
        # aggregate; the non-slice path routes through ``self.function`` so
        # one shared interpolator / extrapolator chain over the 2-D
        # aggregate replaces the prior per-child fan-out.
        values = self.range[x_r] if isinstance(x_r, slice) else self.function(x_r)

        return values[..., x_c]  # pyright: ignore

    def __setitem__(self, x: ArrayLike | slice, y: ArrayLike) -> None:
        """
        Set the corresponding range variable :math:`y` for the specified
        independent domain variable :math:`x`.

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
        [[  0.  10.  20.  30.]
         [  1.  20.  30.  40.]
         [  2.  30.  40.  50.]
         [  3.  40.  50.  60.]
         [  4.  50.  60.  70.]
         [  5.  60.  70.  80.]
         [  6.  70.  80.  90.]
         [  7.  80.  90. 100.]
         [  8.  90. 100. 110.]
         [  9. 100. 110. 120.]]
        >>> multi_signals[0] = 20
        >>> multi_signals[0]
        array([20., 20., 20.])
        >>> multi_signals[np.array([0, 1, 2])] = 30
        >>> multi_signals[np.array([0, 1, 2])]
        array([[30., 30., 30.],
               [30., 30., 30.],
               [30., 30., 30.]])
        >>> multi_signals[np.linspace(0, 5, 5)] = 50
        >>> print(multi_signals)
        [[  0.    50.    50.    50.  ]
         [  1.    30.    30.    30.  ]
         [  1.25  50.    50.    50.  ]
         [  2.    30.    30.    30.  ]
         [  2.5   50.    50.    50.  ]
         [  3.    40.    50.    60.  ]
         [  3.75  50.    50.    50.  ]
         [  4.    50.    60.    70.  ]
         [  5.    50.    50.    50.  ]
         [  6.    70.    80.    90.  ]
         [  7.    80.    90.   100.  ]
         [  8.    90.   100.   110.  ]
         [  9.   100.   110.   120.  ]]
        >>> multi_signals[np.array([0, 1, 2])] = np.array([10, 20, 30])
        >>> print(multi_signals)
        [[  0.    10.    20.    30.  ]
         [  1.    10.    20.    30.  ]
         [  1.25  50.    50.    50.  ]
         [  2.    10.    20.    30.  ]
         [  2.5   50.    50.    50.  ]
         [  3.    40.    50.    60.  ]
         [  3.75  50.    50.    50.  ]
         [  4.    50.    60.    70.  ]
         [  5.    50.    50.    50.  ]
         [  6.    70.    80.    90.  ]
         [  7.    80.    90.   100.  ]
         [  8.    90.   100.   110.  ]
         [  9.   100.   110.   120.  ]]
        >>> y = np.reshape(np.arange(1, 10, 1), (3, 3))
        >>> multi_signals[np.array([0, 1, 2])] = y
        >>> print(multi_signals)
        [[  0.     1.     2.     3.  ]
         [  1.     4.     5.     6.  ]
         [  1.25  50.    50.    50.  ]
         [  2.     7.     8.     9.  ]
         [  2.5   50.    50.    50.  ]
         [  3.    40.    50.    60.  ]
         [  3.75  50.    50.    50.  ]
         [  4.    50.    60.    70.  ]
         [  5.    50.    50.    50.  ]
         [  6.    70.    80.    90.  ]
         [  7.    80.    90.   100.  ]
         [  8.    90.   100.   110.  ]
         [  9.   100.   110.   120.  ]]
        >>> multi_signals[0:3] = 40
        >>> multi_signals[0:3]
        array([[40., 40., 40.],
               [40., 40., 40.],
               [40., 40., 40.]])
        >>> multi_signals[:, 0:2] = 50
        >>> print(multi_signals)
        [[  0.    50.    50.    40.  ]
         [  1.    50.    50.    40.  ]
         [  1.25  50.    50.    40.  ]
         [  2.    50.    50.     9.  ]
         [  2.5   50.    50.    50.  ]
         [  3.    50.    50.    60.  ]
         [  3.75  50.    50.    50.  ]
         [  4.    50.    50.    70.  ]
         [  5.    50.    50.    50.  ]
         [  6.    50.    50.    90.  ]
         [  7.    50.    50.   100.  ]
         [  8.    50.    50.   110.  ]
         [  9.    50.    50.   120.  ]]
        """

        xp = array_namespace(self._range)

        y = xp_as_float_array(y, xp=xp, like=self._range)

        x_r, x_c = (x[0], x[1]) if isinstance(x, tuple) else (x, slice(None))

        attest(
            y.ndim in range(3),
            'Corresponding "y" variable must be a numeric or a 1-dimensional '
            "or 2-dimensional array!",
        )

        n_signals = self._range.shape[1]
        if y.ndim == 0:
            y = xp_broadcast_to(y, (1, n_signals), xp=xp)
        elif y.ndim == 1:
            y = y[None, :]

        attest(
            y.shape[-1] == n_signals,
            'Corresponding "y" variable columns must have same count than '
            'underlying "Signal" components!',
        )

        def set_range(
            index: ArrayLike | slice, values: ArrayLike, xp: ProtocolArrayNamespace
        ) -> None:
            """
            Set ``self._range`` at ``[index, x_c]`` mutably, round-tripping
            through numpy for immutable backends.
            """

            sliced = values[..., x_c]  # pyright: ignore
            if not is_non_ndarray(self._range):
                self._range[index, x_c] = sliced  # pyright: ignore
            else:
                range_ = np.array(as_ndarray(self._range))
                range_[  # pyright: ignore
                    index if isinstance(index, slice) else as_ndarray(index), x_c
                ] = as_ndarray(sliced)
                self._range = xp_as_array(range_, xp=xp, like=self._range)

        if isinstance(x_r, slice):
            set_range(x_r, y, xp)
        else:
            x_r = xp_astype(
                xp_atleast_1d(xp_as_float_array(x_r, xp=xp, like=self._range), xp=xp),
                self.dtype,
                xp=xp,
            )
            y = xp_resize(y, (x_r.shape[0], n_signals), xp=xp)
            domain = xp_as_array(self._domain, xp=xp, like=self._range)

            mask = xp_isin(x_r, domain, xp=xp)
            x_m = x_r[mask]
            if len(x_m) != 0:
                set_range(xp.searchsorted(domain, x_m), y[mask], xp)

            x_nm = x_r[~mask]
            if len(x_nm) != 0:
                indexes = xp.searchsorted(domain, x_nm)
                self._domain = as_ndarray(xp_insert(domain, indexes, x_nm, xp=xp))
                self._range = xp_insert(self._range, indexes, y[~mask], axis=0, xp=xp)

        self._function = None  # Invalidate the underlying continuous function.

    def __contains__(self, x: ArrayLike | slice) -> bool:
        """
        Determine whether the multi-signals contains the
        specified independent domain variable :math:`x`.

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

        xp = array_namespace(self._domain)

        return bool(
            xp.all(
                xp.where(
                    xp.logical_and(
                        x >= xp.min(self._domain),
                        x <= xp.max(self._domain),
                    ),
                    True,
                    False,
                )
            )
        )

    def __eq__(self, other: object) -> bool:
        """
        Determine whether the multi-signals equals the specified
        object.

        Parameters
        ----------
        other
            Object to determine for equality with the multi-signals.

        Returns
        -------
        :class:`bool`
            Whether the specified object is equal to the multi-signals.

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

        # ``interpolator_kwargs`` / ``extrapolator_kwargs`` compared as repr to
        # handle NaNs. Different-backend operands are treated as unequal.
        if isinstance(other, MultiSignals):
            xp = array_namespace(self._domain, self._range)
            if xp is not array_namespace(other.domain, other.range):
                return False

            return all(
                [
                    self._domain.shape == other.domain.shape
                    and bool(xp.all(self._domain == other.domain)),
                    self._range.shape == other.range.shape
                    and bool(xp.all(self._range == other.range)),
                    self._interpolator is other.interpolator,
                    repr(self._interpolator_kwargs) == repr(other.interpolator_kwargs),
                    self._extrapolator is other.extrapolator,
                    repr(self._extrapolator_kwargs) == repr(other.extrapolator_kwargs),
                    self._labels == other.labels,
                ]
            )

        return False

    def __ne__(self, other: object) -> bool:
        """
        Determine whether the multi-signals is not equal to the
        specified object.

        Parameters
        ----------
        other
            Object to test whether it is not equal to the multi-signals.

        Returns
        -------
        :class:`bool`
            Whether the specified object is not equal to the multi-signals.

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
        a: ArrayLike | AbstractContinuousFunction,
        operation: Literal["+", "-", "*", "/", "**"],
        in_place: bool = False,
    ) -> MultiSignals:
        """
        Perform the specified arithmetical operation with operand :math:`a`,
        either on a copy or in-place.

        Parameters
        ----------
        a
            Operand :math:`a`. Can be a numeric value, array-like object, or
            another continuous function instance.
        operation
            Operation to perform.
        in_place
            Operation happens in place.

        Returns
        -------
        :class:`colour.continuous.MultiSignals`
            Multi-signals.

        Examples
        --------
        Adding a single *numeric* variable:

        >>> domain = np.arange(0, 10, 1)
        >>> range_ = tstack([np.linspace(10, 100, 10)] * 3)
        >>> range_ += np.array([0, 10, 20])
        >>> multi_signals_1 = MultiSignals(range_)
        >>> print(multi_signals_1)
        [[  0.  10.  20.  30.]
         [  1.  20.  30.  40.]
         [  2.  30.  40.  50.]
         [  3.  40.  50.  60.]
         [  4.  50.  60.  70.]
         [  5.  60.  70.  80.]
         [  6.  70.  80.  90.]
         [  7.  80.  90. 100.]
         [  8.  90. 100. 110.]
         [  9. 100. 110. 120.]]
        >>> print(multi_signals_1.arithmetical_operation(10, "+", True))
        [[  0.  20.  30.  40.]
         [  1.  30.  40.  50.]
         [  2.  40.  50.  60.]
         [  3.  50.  60.  70.]
         [  4.  60.  70.  80.]
         [  5.  70.  80.  90.]
         [  6.  80.  90. 100.]
         [  7.  90. 100. 110.]
         [  8. 100. 110. 120.]
         [  9. 110. 120. 130.]]

        Adding an `ArrayLike` variable:

        >>> a = np.linspace(10, 100, 10)
        >>> print(multi_signals_1.arithmetical_operation(a, "+", True))
        [[  0.  30.  40.  50.]
         [  1.  50.  60.  70.]
         [  2.  70.  80.  90.]
         [  3.  90. 100. 110.]
         [  4. 110. 120. 130.]
         [  5. 130. 140. 150.]
         [  6. 150. 160. 170.]
         [  7. 170. 180. 190.]
         [  8. 190. 200. 210.]
         [  9. 210. 220. 230.]]

        >>> a = np.array([[10, 20, 30]])
        >>> print(multi_signals_1.arithmetical_operation(a, "+", True))
        [[  0.  40.  60.  80.]
         [  1.  60.  80. 100.]
         [  2.  80. 100. 120.]
         [  3. 100. 120. 140.]
         [  4. 120. 140. 160.]
         [  5. 140. 160. 180.]
         [  6. 160. 180. 200.]
         [  7. 180. 200. 220.]
         [  8. 200. 220. 240.]
         [  9. 220. 240. 260.]]

        >>> a = np.reshape(np.arange(0, 30, 1), (10, 3))
        >>> print(multi_signals_1.arithmetical_operation(a, "+", True))
        [[  0.  40.  61.  82.]
         [  1.  63.  84. 105.]
         [  2.  86. 107. 128.]
         [  3. 109. 130. 151.]
         [  4. 132. 153. 174.]
         [  5. 155. 176. 197.]
         [  6. 178. 199. 220.]
         [  7. 201. 222. 243.]
         [  8. 224. 245. 266.]
         [  9. 247. 268. 289.]]

        Adding a :class:`colour.continuous.Signal` sub-class:

        >>> multi_signals_2 = MultiSignals(range_)
        >>> print(multi_signals_1.arithmetical_operation(multi_signals_2, "+", True))
        [[  0.  50.  81. 112.]
         [  1.  83. 114. 145.]
         [  2. 116. 147. 178.]
         [  3. 149. 180. 211.]
         [  4. 182. 213. 244.]
         [  5. 215. 246. 277.]
         [  6. 248. 279. 310.]
         [  7. 281. 312. 343.]
         [  8. 314. 345. 376.]
         [  9. 347. 378. 409.]]
        """

        operator, ioperator = {
            "+": (add, iadd),
            "-": (sub, isub),
            "*": (mul, imul),
            "/": (truediv, itruediv),
            "**": (pow, ipow),
        }[operation]

        n_signals = self._range.shape[1]

        if in_place:
            if isinstance(a, MultiSignals):
                attest(
                    n_signals == a.range.shape[1],
                    '"MultiSignals" operands must have same count than '
                    'underlying "Signal" components!',
                )

                # The operation is on the ranges, so the namespace is resolved
                # from both ranges (the domains are always *NumPy*) and both
                # operands are promoted to it.
                a_range = a[self._domain]
                xp = array_namespace(self._range, a_range)

                self[self._domain] = operator(
                    xp_as_float_array(self._range, xp=xp, like=a_range),
                    xp_as_float_array(a_range, xp=xp, like=self._range),
                )
                exclusive_or = xp_setxor1d(self._domain, a.domain)
                self[exclusive_or] = full(
                    (exclusive_or.shape[0], n_signals), float("nan")
                )
            else:
                operand = as_float_array(cast("ArrayLike", a))

                attest(
                    operand.ndim in range(3),
                    'Operand "a" variable must be a numeric or a 1-dimensional '
                    "or 2-dimensional array!",
                )

                if operand.ndim == 0 or (
                    operand.ndim == 1 and operand.shape[0] == self._range.shape[0]
                ):
                    # Scalar or 1-D operand of length ``n_wavelengths`` broadcasts
                    # over the signal axis.
                    operand = operand[..., None] if operand.ndim == 1 else operand
                else:
                    attest(
                        operand.shape[-1] == n_signals,
                        'Operand "a" variable columns must have same count than '
                        'underlying "Signal" components!',
                    )

                xp = array_namespace(self._range, operand)
                self_range = xp_as_float_array(self._range, xp=xp, like=operand)
                operand = xp_as_array(operand, xp=xp, like=self_range)
                self.range = ioperator(self_range, operand)

            return self

        return ioperator(self.copy(), a)

    @staticmethod
    @ndarray_copy_enable(True)
    def multi_signals_unpack_data(
        data: (
            ArrayLike
            | DataFrame
            | dict
            | MultiSignals
            | Sequence
            | Series
            | Signal
            | ValuesView
            | None
        ) = None,
        domain: ArrayLike | KeysView | None = None,
        labels: Sequence | None = None,
        dtype: Type[DTypeFloat] | None = None,
        **kwargs: Any,  # noqa: ARG004
    ) -> tuple[NDArrayFloat, NDArrayFloat, List[str]]:
        """
        Unpack specified data for multi-signals instantiation.

        Parameters
        ----------
        data
            Data to unpack for multi-signals instantiation.
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
            Multi-signals name.

        Returns
        -------
        :class:`tuple`
            Tuple of ``(domain, range, labels)`` where ``domain`` is the
            unpacked independent variable as an ``(N,)`` array, ``range`` is
            the unpacked dependent variable as an ``(N, M)`` array with one
            column per signal, and ``labels`` is the list of signal labels
            of length ``M``.

        Examples
        --------
        Unpacking using implicit *domain* and data for a single signal:

        >>> range_ = np.linspace(10, 100, 10)
        >>> domain, range_unpacked, labels = MultiSignals.multi_signals_unpack_data(
        ...     range_
        ... )
        >>> labels
        ['0']
        >>> print(np.column_stack([domain, range_unpacked]))
        [[  0.  10.]
         [  1.  20.]
         [  2.  30.]
         [  3.  40.]
         [  4.  50.]
         [  5.  60.]
         [  6.  70.]
         [  7.  80.]
         [  8.  90.]
         [  9. 100.]]

        Unpacking using explicit *domain* and data for a single signal:

        >>> domain_ = np.arange(100, 1100, 100)
        >>> domain, range_unpacked, labels = MultiSignals.multi_signals_unpack_data(
        ...     range_, domain_
        ... )
        >>> labels
        ['0']
        >>> print(np.column_stack([domain, range_unpacked]))
        [[ 100.   10.]
         [ 200.   20.]
         [ 300.   30.]
         [ 400.   40.]
         [ 500.   50.]
         [ 600.   60.]
         [ 700.   70.]
         [ 800.   80.]
         [ 900.   90.]
         [1000.  100.]]

        Unpacking using data for multiple signals:

        >>> range_ = tstack([np.linspace(10, 100, 10)] * 3)
        >>> range_ += np.array([0, 10, 20])
        >>> domain, range_unpacked, labels = MultiSignals.multi_signals_unpack_data(
        ...     range_, domain_
        ... )
        >>> labels
        ['0', '1', '2']
        >>> print(np.column_stack([domain, range_unpacked[:, 2]]))
        [[ 100.   30.]
         [ 200.   40.]
         [ 300.   50.]
         [ 400.   60.]
         [ 500.   70.]
         [ 600.   80.]
         [ 700.   90.]
         [ 800.  100.]
         [ 900.  110.]
         [1000.  120.]]

        Unpacking using a *dict*:

        >>> domain, range_unpacked, labels = MultiSignals.multi_signals_unpack_data(
        ...     dict(zip(domain_, range_))
        ... )
        >>> labels
        ['0', '1', '2']
        >>> print(np.column_stack([domain, range_unpacked[:, 2]]))
        [[ 100.   30.]
         [ 200.   40.]
         [ 300.   50.]
         [ 400.   60.]
         [ 500.   70.]
         [ 600.   80.]
         [ 700.   90.]
         [ 800.  100.]
         [ 900.  110.]
         [1000.  120.]]

        Unpacking using a sequence of *Signal* instances:

        >>> from colour.continuous import Signal
        >>> signals_seq = [Signal(range_[:, i], domain_, name=str(i)) for i in range(3)]
        >>> domain, range_unpacked, labels = MultiSignals.multi_signals_unpack_data(
        ...     signals_seq
        ... )
        >>> labels
        ['0', '1', '2']
        >>> print(np.column_stack([domain, range_unpacked[:, 2]]))
        [[ 100.   30.]
         [ 200.   40.]
         [ 300.   50.]
         [ 400.   60.]
         [ 500.   70.]
         [ 600.   80.]
         [ 700.   90.]
         [ 800.  100.]
         [ 900.  110.]
         [1000.  120.]]

        Unpacking from an existing :class:`MultiSignals` instance:

        >>> multi_signals = MultiSignals(range_, domain_)
        >>> domain, range_unpacked, labels = MultiSignals.multi_signals_unpack_data(
        ...     multi_signals
        ... )
        >>> labels
        ['0', '1', '2']
        >>> print(np.column_stack([domain, range_unpacked[:, 2]]))
        [[ 100.   30.]
         [ 200.   40.]
         [ 300.   50.]
         [ 400.   60.]
         [ 500.   70.]
         [ 600.   80.]
         [ 700.   90.]
         [ 800.  100.]
         [ 900.  110.]
         [1000.  120.]]

        Unpacking using a *Pandas* `Series`:

        >>> if is_pandas_installed():
        ...     from pandas import Series
        ...
        ...     domain, range_unpacked, labels = MultiSignals.multi_signals_unpack_data(
        ...         Series(dict(zip(domain_, np.linspace(10, 100, 10))))
        ...     )
        ...     print(np.column_stack([domain, range_unpacked]))  # doctest: +SKIP
        [[ 100.   10.]
         [ 200.   20.]
         [ 300.   30.]
         [ 400.   40.]
         [ 500.   50.]
         [ 600.   60.]
         [ 700.   70.]
         [ 800.   80.]
         [ 900.   90.]
         [1000.  100.]]

        Unpacking using a *Pandas* :class:`pandas.DataFrame`:

        >>> if is_pandas_installed():
        ...     from pandas import DataFrame
        ...
        ...     data = dict(zip(["a", "b", "c"], tsplit(range_)))
        ...     domain, range_unpacked, labels = MultiSignals.multi_signals_unpack_data(
        ...         DataFrame(data, domain_)
        ...     )
        ...     print(np.column_stack([domain, range_unpacked[:, 2]]))  # doctest: +SKIP
        [[ 100.   30.]
         [ 200.   40.]
         [ 300.   50.]
         [ 400.   60.]
         [ 500.   70.]
         [ 600.   80.]
         [ 700.   90.]
         [ 800.  100.]
         [ 900.  110.]
         [1000.  120.]]
        """

        dtype = optional(dtype, DTYPE_FLOAT_DEFAULT)

        domain_unpacked: NDArrayFloat = as_float_array([], dtype)
        range_unpacked: NDArrayFloat = np.zeros((0, 0), dtype=dtype)
        labels_unpacked: List[str] = []

        if data is None:
            pass
        elif (
            isinstance(data, tuple)
            and len(data) == 3
            and is_non_ndarray(data[0]) is False
            and isinstance(data[2], (list, tuple))
            and all(isinstance(label, str) for label in data[2])
        ):
            # The unpacked ``(domain, range, labels)`` triple is accepted back
            # so that the definition is idempotent: assigning its own output to
            # the :attr:`signals` property, as in
            # ``msds.signals = MultiSignals.multi_signals_unpack_data(...)``,
            # round-trips rather than being re-unpacked as a sequence.
            domain_unpacked = as_float_array(data[0], dtype)
            range_unpacked = as_float_array(data[1], dtype)
            if range_unpacked.ndim == 1:
                range_unpacked = xp_reshape(range_unpacked, (-1, 1))
            labels_unpacked = [str(label) for label in data[2]]
        elif isinstance(data, Signal):
            domain_unpacked = as_float_array(data.domain, dtype)
            range_unpacked = xp_reshape(as_float_array(data.range, dtype), (-1, 1))
            labels_unpacked = [str(data.name)]
        elif isinstance(data, MultiSignals):
            domain_unpacked = as_float_array(data.domain, dtype)
            range_unpacked = as_float_array(data.range, dtype)
            if range_unpacked.ndim == 1:
                range_unpacked = xp_reshape(range_unpacked, (-1, 1))
            labels_unpacked = list(data.labels)
        elif is_non_ndarray(data):
            range_unpacked = as_float_array(data, dtype)  # pyright: ignore

            attest(
                range_unpacked.ndim in (1, 2),
                'User "data" must be 1-dimensional or 2-dimensional!',
            )

            if range_unpacked.ndim == 1:
                range_unpacked = xp_reshape(range_unpacked, (-1, 1))
            labels_unpacked = [str(i) for i in range(range_unpacked.shape[1])]
        elif issubclass(type(data), Sequence) or isinstance(
            data, (tuple, list, np.ndarray, Iterator, ValuesView)
        ):
            data_sequence = list(cast("Sequence", data))

            is_signal = bool(data_sequence) and all(
                isinstance(i, Signal) for i in data_sequence
            )

            if is_signal:
                domain_unpacked = as_float_array(data_sequence[0].domain, dtype)
                range_unpacked = tstack(
                    [as_float_array(signal.range, dtype) for signal in data_sequence]
                )
                if range_unpacked.ndim == 1:
                    range_unpacked = xp_reshape(range_unpacked, (-1, 1))
                labels_unpacked = [str(signal.name) for signal in data_sequence]
            else:
                range_unpacked = as_float_array(np.asarray(data_sequence), dtype)
                attest(
                    range_unpacked.ndim in (1, 2),
                    'User "data" must be 1-dimensional or 2-dimensional!',
                )

                if range_unpacked.ndim == 1:
                    range_unpacked = xp_reshape(range_unpacked, (-1, 1))
                labels_unpacked = [str(i) for i in range(range_unpacked.shape[1])]
        elif issubclass(type(data), Mapping) or isinstance(data, dict):
            data_mapping = dict(cast("Mapping", data))

            is_signal = bool(data_mapping) and all(
                isinstance(i, Signal) for i in data_mapping.values()
            )

            if is_signal:
                first_signal = next(iter(data_mapping.values()))
                domain_unpacked = as_float_array(first_signal.domain, dtype)
                range_unpacked = tstack(
                    [
                        as_float_array(signal.range, dtype)
                        for signal in data_mapping.values()
                    ]
                )
                if range_unpacked.ndim == 1:
                    range_unpacked = xp_reshape(range_unpacked, (-1, 1))
                labels_unpacked = [str(label) for label in data_mapping]
            else:
                keys, values = zip(*sorted(data_mapping.items()), strict=True)
                domain_unpacked = as_float_array(keys, dtype)
                range_unpacked = as_float_array(np.asarray(values), dtype)
                if range_unpacked.ndim == 1:
                    range_unpacked = xp_reshape(range_unpacked, (-1, 1))
                labels_unpacked = [str(i) for i in range(range_unpacked.shape[1])]
        elif is_pandas_installed():
            if isinstance(data, Series):
                domain_unpacked = as_float_array(data.index.values, dtype)  # pyright: ignore
                range_unpacked = xp_reshape(
                    as_float_array(data.to_numpy(), dtype), (-1, 1)
                )
                labels_unpacked = ["0"]
            elif isinstance(data, DataFrame):
                domain_unpacked = as_float_array(data.index.values, dtype)  # pyright: ignore
                range_unpacked = as_float_array(data.to_numpy(), dtype)
                if range_unpacked.ndim == 1:
                    range_unpacked = xp_reshape(range_unpacked, (-1, 1))
                labels_unpacked = [str(label) for label in data.columns]

        if domain is not None:
            if isinstance(domain, KeysView):
                domain = list(domain)

            domain_array = as_float_array(domain, dtype)

            if len(range_unpacked) > 0:
                attest(
                    len(domain_array) == len(range_unpacked),
                    'User "domain" length is not compatible with unpacked "range"!',
                )

            domain_unpacked = domain_array

        if len(domain_unpacked) == 0 and len(range_unpacked) > 0:
            domain_unpacked = np.arange(range_unpacked.shape[0], dtype=dtype)

        if labels is not None:
            attest(
                len(labels) == len(labels_unpacked),
                'User "labels" length is not compatible with unpacked "labels"!',
            )

            if len(labels) != len(set(labels)):
                labels = [f"{label} - {i}" for i, label in enumerate(labels)]

            labels_unpacked = [str(label) for label in labels]

        if not labels_unpacked:
            labels_unpacked = ["Undefined"]
            if range_unpacked.size == 0:
                range_unpacked = np.zeros((0, 1), dtype=dtype)

        return (
            ndarray_copy(domain_unpacked),
            ndarray_copy(range_unpacked),
            labels_unpacked,
        )

    def fill_nan(
        self,
        method: Literal["Constant", "Interpolation"] | str = "Interpolation",
        default: Real = 0,
    ) -> MultiSignals:
        """
        Fill NaNs in independent domain variable :math:`x` and corresponding
        range variable :math:`y` using the specified method.

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
            Multi-signals with NaN values filled.

        Examples
        --------
        >>> domain = np.arange(0, 10, 1)
        >>> range_ = tstack([np.linspace(10, 100, 10)] * 3)
        >>> range_ += np.array([0, 10, 20])
        >>> multi_signals = MultiSignals(range_)
        >>> multi_signals[3:7] = np.nan
        >>> print(multi_signals)
        [[  0.  10.  20.  30.]
         [  1.  20.  30.  40.]
         [  2.  30.  40.  50.]
         [  3.  nan  nan  nan]
         [  4.  nan  nan  nan]
         [  5.  nan  nan  nan]
         [  6.  nan  nan  nan]
         [  7.  80.  90. 100.]
         [  8.  90. 100. 110.]
         [  9. 100. 110. 120.]]
        >>> print(multi_signals.fill_nan())
        [[  0.  10.  20.  30.]
         [  1.  20.  30.  40.]
         [  2.  30.  40.  50.]
         [  3.  40.  50.  60.]
         [  4.  50.  60.  70.]
         [  5.  60.  70.  80.]
         [  6.  70.  80.  90.]
         [  7.  80.  90. 100.]
         [  8.  90. 100. 110.]
         [  9. 100. 110. 120.]]
        >>> multi_signals[3:7] = np.nan
        >>> print(multi_signals.fill_nan(method="Constant"))
        [[  0.  10.  20.  30.]
         [  1.  20.  30.  40.]
         [  2.  30.  40.  50.]
         [  3.   0.   0.   0.]
         [  4.   0.   0.   0.]
         [  5.   0.   0.   0.]
         [  6.   0.   0.   0.]
         [  7.  80.  90. 100.]
         [  8.  90. 100. 110.]
         [  9. 100. 110. 120.]]
        """

        method = validate_method(method, ("Interpolation", "Constant"))

        if self._labels:
            self.domain = fill_nan(self._domain, method, default)
            # ``fill_nan`` is 1-D; iterate over the trailing signal axis so
            # each column's NaN pattern resolves independently.
            self.range = tstack(
                [
                    fill_nan(self._range[..., i], method, default)
                    for i in range(self._range.shape[-1])
                ]
            )

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
            data=self.range,
            index=self.domain,
            columns=self.labels,  # pyright: ignore
        )
