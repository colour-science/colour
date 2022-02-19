"""
Array Utilities
===============

Defines array utilities objects.

References
----------
-   :cite:`Castro2014a` : Castro, S. (2014). Numpy: Fastest way of computing
    diagonal for each row of a 2d array. Retrieved August 22, 2014, from
    http://stackoverflow.com/questions/26511401/\
numpy-fastest-way-of-computing-diagonal-for-each-row-of-a-2d-array/\
26517247#26517247
-   :cite:`Yorke2014a` : Yorke, R. (2014). Python: Change format of np.array or
    allow tolerance in in1d function. Retrieved March 27, 2015, from
    http://stackoverflow.com/a/23521245/931625
"""

from __future__ import annotations

import functools
import numpy as np
import sys
from collections.abc import KeysView, ValuesView
from contextlib import contextmanager
from dataclasses import fields, is_dataclass, replace
from operator import add, mul, pow, sub, truediv

from colour.constants import DEFAULT_FLOAT_DTYPE, DEFAULT_INT_DTYPE, EPSILON
from colour.hints import (
    Any,
    ArrayLike,
    Boolean,
    Callable,
    Dataclass,
    DType,
    DTypeBoolean,
    DTypeFloating,
    DTypeInteger,
    DTypeNumber,
    Floating,
    FloatingOrArrayLike,
    FloatingOrNDArray,
    Generator,
    Integer,
    IntegerOrArrayLike,
    IntegerOrNDArray,
    Literal,
    NDArray,
    NestedSequence,
    Number,
    NumberOrArrayLike,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
)
from colour.utilities import (
    attest,
    optional,
    suppress_warnings,
    validate_method,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "MixinDataclassFields",
    "MixinDataclassIterable",
    "MixinDataclassArray",
    "MixinDataclassArithmetic",
    "as_array",
    "as_int",
    "as_float",
    "as_int_array",
    "as_float_array",
    "as_int_scalar",
    "as_float_scalar",
    "set_default_int_dtype",
    "set_default_float_dtype",
    "get_domain_range_scale",
    "set_domain_range_scale",
    "domain_range_scale",
    "to_domain_1",
    "to_domain_10",
    "to_domain_100",
    "to_domain_degrees",
    "to_domain_int",
    "from_range_1",
    "from_range_10",
    "from_range_100",
    "from_range_degrees",
    "from_range_int",
    "closest_indexes",
    "closest",
    "interval",
    "is_uniform",
    "in_array",
    "tstack",
    "tsplit",
    "row_as_diagonal",
    "orient",
    "centroid",
    "fill_nan",
    "has_only_nan",
    "ndarray_write",
    "zeros",
    "ones",
    "full",
    "index_along_last_axis",
]


class MixinDataclassFields:
    """
    A mixin providing fields introspection for the :class:`dataclass`-like
    class fields.

    Attributes
    ----------
    -   :meth:`~colour.utilities.MixinDataclassFields.fields`
    """

    @property
    def fields(self) -> Tuple:
        """
        Getter property for the fields of the :class:`dataclass`-like class.

        Returns
        -------
        :class:`tuple`
           Tuple of :class:`dataclass`-like class fields.
        """

        return fields(self)


class MixinDataclassIterable(MixinDataclassFields):
    """
    A mixin providing iteration capabilities over the :class:`dataclass`-like
    class fields.

    Attributes
    ----------
    -   :meth:`~colour.utilities.MixinDataclassIterable.keys`
    -   :meth:`~colour.utilities.MixinDataclassIterable.values`
    -   :meth:`~colour.utilities.MixinDataclassIterable.items`

    Methods
    -------
    -   :meth:`~colour.utilities.MixinDataclassIterable.__iter__`

    Notes
    -----
    -   The :class:`colour.utilities.MixinDataclassIterable` class inherits the
        methods from the following class:

        -   :class:`colour.utilities.MixinDataclassFields`
    """

    @property
    def keys(self) -> Tuple:
        """
        Getter property for the :class:`dataclass`-like class keys, i.e. the
        field names.

        Returns
        -------
        :class:`tuple`
           :class:`dataclass`-like class keys.
        """

        return tuple(field for field, _value in self)

    @property
    def values(self) -> Tuple:
        """
        Getter property for the :class:`dataclass`-like class values, i.e. the
        field values.

        Returns
        -------
        :class:`tuple`
           :class:`dataclass`-like class values.
        """

        return tuple(value for _field, value in self)

    @property
    def items(self) -> Tuple:
        """
        Getter property for  the :class:`dataclass`-like class items, i.e. the
        field names and values.

        Returns
        -------
        :class:`tuple`
           :class:`dataclass`-like class items.
        """

        return tuple((field, value) for field, value in self)

    def __iter__(self) -> Generator:
        """
        Return a generator for the :class:`dataclass`-like class fields.

        Yields
        ------
        Generator
           :class:`dataclass`-like class field generator.
        """

        yield from {
            field.name: getattr(self, field.name) for field in self.fields
        }.items()


class MixinDataclassArray(MixinDataclassIterable):
    """
    A mixin providing conversion methods for :class:`dataclass`-like class
    conversion to :class:`numpy.ndarray` class.

    Methods
    -------
    -   :meth:`~colour.utilities.MixinDataclassArray.__array__`

    Notes
    -----
    -   The :class:`colour.utilities.MixinDataclassArray` class inherits the
        methods from the following classes:

        -   :class:`colour.utilities.MixinDataclassIterable`
        -   :class:`colour.utilities.MixinDataclassFields`
    """

    def __array__(self, dtype: Optional[Type[DTypeNumber]] = None) -> NDArray:
        """
        Implement support for :class:`dataclass`-like class conversion to
        :class:`numpy.ndarray` class.

        A field set to *None* will be filled with `np.nan` according to the
        shape of the first field not set with *None*.

        Parameters
        ----------
        dtype
            :class:`numpy.dtype` to use for conversion to `np.ndarray`, default
            to the :class:`numpy.dtype` defined by
            :attr:`colour.constant.DEFAULT_FLOAT_DTYPE` attribute.

        Returns
        -------
        :class:`numpy.ndarray`
            :class:`dataclass`-like class converted to :class:`numpy.ndarray`.
        """

        dtype = cast(Type[DTypeNumber], optional(dtype, DEFAULT_FLOAT_DTYPE))

        default = None
        for _field, value in self:
            if value is not None:
                default = full(as_float_array(value).shape, np.nan)
                break

        return tstack(
            [value if value is not None else default for value in self.values],
            dtype=dtype,
        )


class MixinDataclassArithmetic(MixinDataclassArray):
    """
    A mixin providing mathematical operations for :class:`dataclass`-like
    class.

    Methods
    -------
    -   :meth:`~colour.utilities.MixinDataclassArray.__iadd__`
    -   :meth:`~colour.utilities.MixinDataclassArray.__add__`
    -   :meth:`~colour.utilities.MixinDataclassArray.__isub__`
    -   :meth:`~colour.utilities.MixinDataclassArray.__sub__`
    -   :meth:`~colour.utilities.MixinDataclassArray.__imul__`
    -   :meth:`~colour.utilities.MixinDataclassArray.__mul__`
    -   :meth:`~colour.utilities.MixinDataclassArray.__idiv__`
    -   :meth:`~colour.utilities.MixinDataclassArray.__div__`
    -   :meth:`~colour.utilities.MixinDataclassArray.__ipow__`
    -   :meth:`~colour.utilities.MixinDataclassArray.__pow__`
    -   :meth:`~colour.utilities.MixinDataclassArray.arithmetical_operation`

    Notes
    -----
    -   The :class:`colour.utilities.MixinDataclassArithmetic` class inherits
        the methods from the following classes:

        -   :class:`colour.utilities.MixinDataclassArray`
        -   :class:`colour.utilities.MixinDataclassIterable`
        -   :class:`colour.utilities.MixinDataclassFields`
    """

    def __add__(self, a: Any) -> Dataclass:
        """
        Implement support for addition.

        Parameters
        ----------
        a
            Variable :math:`a` to add.

        Returns
        -------
        :class:`dataclass`
            Variable added :class:`dataclass`-like class.
        """

        return self.arithmetical_operation(a, "+")

    def __iadd__(self, a: Any) -> Dataclass:
        """
        Implement support for in-place addition.

        Parameters
        ----------
        a
            Variable :math:`a` to add in-place.

        Returns
        -------
        :class:`dataclass`
            In-place variable added :class:`dataclass`-like class.
        """

        return self.arithmetical_operation(a, "+", True)

    def __sub__(self, a: Any) -> Dataclass:
        """
        Implement support for subtraction.

        Parameters
        ----------
        a
            Variable :math:`a` to subtract.

        Returns
        -------
        :class:`dataclass`
            Variable subtracted :class:`dataclass`-like class.
        """

        return self.arithmetical_operation(a, "-")

    def __isub__(self, a: Any) -> Dataclass:
        """
        Implement support for in-place subtraction.

        Parameters
        ----------
        a
            Variable :math:`a` to subtract in-place.

        Returns
        -------
        :class:`dataclass`
            In-place variable subtracted :class:`dataclass`-like class.
        """

        return self.arithmetical_operation(a, "-", True)

    def __mul__(self, a: Any) -> Dataclass:
        """
        Implement support for multiplication.

        Parameters
        ----------
        a
            Variable :math:`a` to multiply by.

        Returns
        -------
        :class:`dataclass`
            Variable multiplied :class:`dataclass`-like class.
        """

        return self.arithmetical_operation(a, "*")

    def __imul__(self, a: Any) -> Dataclass:
        """
        Implement support for in-place multiplication.

        Parameters
        ----------
        a
            Variable :math:`a` to multiply by in-place.

        Returns
        -------
        :class:`dataclass`
            In-place variable multiplied :class:`dataclass`-like class.
        """

        return self.arithmetical_operation(a, "*", True)

    def __div__(self, a: Any) -> Dataclass:
        """
        Implement support for division.

        Parameters
        ----------
        a
            Variable :math:`a` to divide by.

        Returns
        -------
        :class:`dataclass`
            Variable divided :class:`dataclass`-like class.
        """

        return self.arithmetical_operation(a, "/")

    def __idiv__(self, a: Any) -> Dataclass:
        """
        Implement support for in-place division.

        Parameters
        ----------
        a
            Variable :math:`a` to divide by in-place.

        Returns
        -------
        :class:`dataclass`
            In-place variable divided :class:`dataclass`-like class.
        """

        return self.arithmetical_operation(a, "/", True)

    __itruediv__ = __idiv__
    __truediv__ = __div__

    def __pow__(self, a: Any) -> Dataclass:
        """
        Implement support for exponentiation.

        Parameters
        ----------
        a
            Variable :math:`a` to exponentiate by.

        Returns
        -------
        :class:`dataclass`
            Variable exponentiated :class:`dataclass`-like class.
        """

        return self.arithmetical_operation(a, "**")

    def __ipow__(self, a: Any) -> Dataclass:
        """
        Implement support for in-place exponentiation.

        Parameters
        ----------
        a
            Variable :math:`a` to exponentiate by in-place.

        Returns
        -------
        :class:`dataclass`
            In-place variable exponentiated :class:`dataclass`-like class.
        """

        return self.arithmetical_operation(a, "**", True)

    def arithmetical_operation(
        self, a: Any, operation: str, in_place: Boolean = False
    ) -> Dataclass:
        """
        Perform given arithmetical operation with :math:`a` operand on the
        :class:`dataclass`-like class.

        Parameters
        ----------
        a
            Operand.
        operation
            Operation to perform.
        in_place
            Operation happens in place.

        Returns
        -------
        :class:`dataclass`
            :class:`dataclass`-like class with arithmetical operation
            performed.
        """

        callable_operation = {
            "+": add,
            "-": sub,
            "*": mul,
            "/": truediv,
            "**": pow,
        }[operation]

        if is_dataclass(a):
            a = as_float_array(a)

        values = tsplit(callable_operation(as_float_array(self), a))
        field_values = {field: values[i] for i, field in enumerate(self.keys)}
        field_values.update(
            {field: None for field, value in self if value is None}
        )

        dataclass = replace(self, **field_values)

        if in_place:
            for field in self.keys:
                setattr(self, field, getattr(dataclass, field))
            return self
        else:
            return dataclass


def as_array(
    a: Union[NumberOrArrayLike, NestedSequence[NumberOrArrayLike]],
    dtype: Optional[Type[DType]] = None,
) -> NDArray:
    """
    Convert given variable :math:`a` to :class:`numpy.ndarray` using given
    :class:`numpy.dtype`.

    Parameters
    ----------
    a
        Variable to convert.
    dtype
        :class:`numpy.dtype` to use for conversion, default to the
        :class:`numpy.dtype` defined by the
        :attr:`colour.constant.DEFAULT_FLOAT_DTYPE` attribute.

    Returns
    -------
    :class:`numpy.ndarray`
        Variable :math:`a` converted to :class:`numpy.ndarray`.

    Examples
    --------
    >>> as_array([1, 2, 3])  # doctest: +ELLIPSIS
    array([1, 2, 3]...)
    >>> as_array([1, 2, 3], dtype=DEFAULT_FLOAT_DTYPE)
    array([ 1.,  2.,  3.])
    """

    # TODO: Remove when https://github.com/numpy/numpy/issues/5718 is
    # addressed.
    if isinstance(a, (KeysView, ValuesView)):
        a = list(a)

    return np.asarray(a, dtype)


def as_int(
    a: NumberOrArrayLike, dtype: Optional[Type[DTypeInteger]] = None
) -> IntegerOrNDArray:
    """
    Attempt to convert given variable :math:`a` to :class:`numpy.integer`
    using given :class:`numpy.dtype`. If variable :math:`a` is not a scalar or
    0-dimensional, it is converted to :class:`numpy.ndarray`.

    Parameters
    ----------
    a
        Variable to convert.
    dtype
        :class:`numpy.dtype` to use for conversion, default to the
        :class:`numpy.dtype` defined by the
        :attr:`colour.constant.DEFAULT_INT_DTYPE` attribute.

    Returns
    -------
    :class:`numpy.integer` or :class:`numpy.ndarray`
        Variable :math:`a` converted to :class:`numpy.integer`.

    Examples
    --------
    >>> as_int(np.array([1]))
    1
    >>> as_int(np.arange(10))  # doctest: +ELLIPSIS
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]...)
    """

    dtype = cast(Type[DTypeInteger], optional(dtype, DEFAULT_INT_DTYPE))

    args = getattr(DTypeInteger, "__args__")
    attest(
        dtype in args,
        f'"dtype" must be one of the following types: {args}',
    )

    # TODO: Reassess implementation when and if
    # https://github.com/numpy/numpy/issues/11956 is addressed.
    return dtype(np.squeeze(a))  # type: ignore[return-value]


def as_float(
    a: NumberOrArrayLike, dtype: Optional[Type[DTypeFloating]] = None
) -> FloatingOrNDArray:
    """
    Attempt to convert given variable :math:`a` to :class:`numpy.floating`
    using given :class:`numpy.dtype`. If variable :math:`a` is not a scalar or
    0-dimensional, it is converted to :class:`numpy.ndarray`.

    Parameters
    ----------
    a
        Variable to convert.
    dtype
        :class:`numpy.dtype` to use for conversion, default to the
        :class:`numpy.dtype` defined by the
        :attr:`colour.constant.DEFAULT_FLOAT_DTYPE` attribute.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Variable :math:`a` converted to :class:`numpy.floating`.

    Examples
    --------
    >>> as_float(np.array([1]))
    1.0
    >>> as_float(np.arange(10))
    array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.])
    """

    dtype = cast(Type[DTypeFloating], optional(dtype, DEFAULT_FLOAT_DTYPE))

    args = getattr(DTypeFloating, "__args__")
    attest(
        dtype in args,
        f'"dtype" must be one of the following types: {args}',
    )

    return dtype(a)  # type: ignore[arg-type, return-value]


def as_int_array(
    a: NumberOrArrayLike, dtype: Optional[Type[DTypeInteger]] = None
) -> NDArray:
    """
    Convert given variable :math:`a` to :class:`numpy.ndarray` using given
    :class:`numpy.dtype`.

    Parameters
    ----------
    a
        Variable to convert.
    dtype
        :class:`numpy.dtype` to use for conversion, default to the
        :class:`numpy.dtype` defined by the
        :attr:`colour.constant.DEFAULT_INT_DTYPE` attribute.

    Returns
    -------
    :class:`numpy.ndarray`
        Variable :math:`a` converted to :class:`numpy.ndarray`.

    Examples
    --------
    >>> as_int_array([1.0, 2.0, 3.0])  # doctest: +ELLIPSIS
    array([1, 2, 3]...)
    """

    dtype = cast(Type[DTypeInteger], optional(dtype, DEFAULT_INT_DTYPE))

    args = getattr(DTypeInteger, "__args__")
    attest(
        dtype in args,
        f'"dtype" must be one of the following types: {args}',
    )

    return as_array(a, dtype)


def as_float_array(
    a: NumberOrArrayLike, dtype: Optional[Type[DTypeFloating]] = None
) -> NDArray:
    """
    Convert given variable :math:`a` to :class:`numpy.ndarray` using given
    :class:`numpy.dtype`.

    Parameters
    ----------
    a
        Variable to convert.
    dtype
        :class:`numpy.dtype` to use for conversion, default to the
        :class:`numpy.dtype` defined by the
        :attr:`colour.constant.DEFAULT_FLOAT_DTYPE` attribute.

    Returns
    -------
    :class:`numpy.ndarray`
        Variable :math:`a` converted to :class:`numpy.ndarray`.

    Examples
    --------
    >>> as_float_array([1, 2, 3])
    array([ 1.,  2.,  3.])
    """

    dtype = cast(Type[DTypeFloating], optional(dtype, DEFAULT_FLOAT_DTYPE))

    attest(
        dtype in np.sctypes["float"],
        f"\"dtype\" must be one of the following types: {np.sctypes['float']}",
    )

    return as_array(a, dtype)


def as_int_scalar(
    a: NumberOrArrayLike, dtype: Optional[Type[DTypeInteger]] = None
) -> Integer:
    """
    Convert given :math:`a` variable to :class:`numpy.integer` using given
    :class:`numpy.dtype`.

    Parameters
    ----------
    a
        Variable to convert.
    dtype
        :class:`numpy.dtype` to use for conversion, default to the
        :class:`numpy.dtype` defined by the
        :attr:`colour.constant.DEFAULT_INT_DTYPE` attribute.

    Returns
    -------
    :class:`numpy.integer`
        :math:`a` variable converted to :class:`numpy.integer`.

    Examples
    --------
    >>> as_int_scalar(np.array(1))
    1
    """

    a = np.squeeze(as_int_array(a, dtype))

    attest(a.ndim == 0, f'"{a}" cannot be converted to "int" scalar!')

    return cast(Integer, as_int(a, dtype))


def as_float_scalar(
    a: NumberOrArrayLike, dtype: Optional[Type[DTypeFloating]] = None
) -> Floating:
    """
    Convert given :math:`a` variable to :class:`numpy.floating` using given
    :class:`numpy.dtype`.

    Parameters
    ----------
    a
        Variable to convert.
    dtype
        :class:`numpy.dtype` to use for conversion, default to the
        :class:`numpy.dtype` defined by the
        :attr:`colour.constant.DEFAULT_FLOAT_DTYPE` attribute.

    Returns
    -------
    :class:`numpy.floating`
        :math:`a` variable converted to :class:`numpy.floating`.

    Examples
    --------
    >>> as_float_scalar(np.array(1))
    1.0
    """

    a = np.squeeze(as_float_array(a, dtype))

    attest(a.ndim == 0, f'"{a}" cannot be converted to "float" scalar!')

    return cast(Floating, as_float(a, dtype))


def set_default_int_dtype(
    dtype: Type[DTypeInteger] = DEFAULT_INT_DTYPE,
) -> None:
    """
    Set *Colour* default :class:`numpy.integer` precision by setting
    :attr:`colour.constant.DEFAULT_INT_DTYPE` attribute with given
    :class:`numpy.dtype` wherever the attribute is imported.

    Parameters
    ----------
    dtype
        :class:`numpy.dtype` to set :attr:`colour.constant.DEFAULT_INT_DTYPE`
        with.

    Notes
    -----
    -   It is possible to define the int precision at import time by setting
        the *COLOUR_SCIENCE__DEFAULT_INT_DTYPE* environment variable, for
        example `set COLOUR_SCIENCE__DEFAULT_INT_DTYPE=int32`.

    Warnings
    --------
    This definition is mostly given for consistency purposes with
    :func:`colour.utilities.set_default_float_dtype` definition but contrary to the
    latter, changing *integer* precision will almost certainly completely break
    *Colour*. With great power comes great responsibility.

    Examples
    --------
    >>> as_int_array(np.ones(3)).dtype  # doctest: +SKIP
    dtype('int64')
    >>> set_default_int_dtype(np.int32)
    >>> as_int_array(np.ones(3)).dtype  # doctest: +SKIP
    dtype('int32')
    >>> set_default_int_dtype(np.int64)
    >>> as_int_array(np.ones(3)).dtype  # doctest: +SKIP
    dtype('int64')
    """

    # TODO: Investigate behaviour on Windows.
    with suppress_warnings(colour_usage_warnings=True):
        for module in sys.modules.values():
            if not hasattr(module, "DEFAULT_INT_DTYPE"):
                continue

            setattr(module, "DEFAULT_INT_DTYPE", dtype)


def set_default_float_dtype(
    dtype: Type[DTypeFloating] = DEFAULT_FLOAT_DTYPE,
) -> None:
    """
    Set *Colour* default :class:`numpy.floating` precision by setting
    :attr:`colour.constant.DEFAULT_FLOAT_DTYPE` attribute with given
    :class:`numpy.dtype` wherever the attribute is imported.

    Parameters
    ----------
    dtype
        :class:`numpy.dtype` to set :attr:`colour.constant.DEFAULT_FLOAT_DTYPE`
        with.

    Warnings
    --------
    Changing *float* precision might result in various *Colour* functionality
    breaking entirely: https://github.com/numpy/numpy/issues/6860. With great
    power comes great responsibility.

    Notes
    -----
    -   It is possible to define the *float* precision at import time by
        setting the *COLOUR_SCIENCE__DEFAULT_FLOAT_DTYPE* environment variable,
        for example `set COLOUR_SCIENCE__DEFAULT_FLOAT_DTYPE=float32`.
    -   Some definition returning a single-scalar ndarray might not honour the
        given *float* precision: https://github.com/numpy/numpy/issues/16353

    Examples
    --------
    >>> as_float_array(np.ones(3)).dtype
    dtype('float64')
    >>> set_default_float_dtype(np.float16)
    >>> as_float_array(np.ones(3)).dtype
    dtype('float16')
    >>> set_default_float_dtype(np.float64)
    >>> as_float_array(np.ones(3)).dtype
    dtype('float64')
    """

    with suppress_warnings(colour_usage_warnings=True):
        for module in sys.modules.values():
            if not hasattr(module, "DEFAULT_FLOAT_DTYPE"):
                continue

            setattr(module, "DEFAULT_FLOAT_DTYPE", dtype)


# TODO: Annotate with "Union[Literal['ignore', 'reference', '1', '100'], str]"
# when Python 3.7 is dropped.
_DOMAIN_RANGE_SCALE = "reference"
"""
Global variable storing the current *Colour* domain-range scale.

_DOMAIN_RANGE_SCALE
"""


def get_domain_range_scale() -> Union[
    Literal["ignore", "reference", "1", "100"], str
]:
    """
    Return the current *Colour* domain-range scale. The following scales are
    available:

    -   **'Reference'**, the default *Colour* domain-range scale which varies
        depending on the referenced algorithm, e.g. [0, 1], [0, 10], [0, 100],
        [0, 255], etc...
    -   **'1'**, a domain-range scale normalised to [0, 1], it is important to
        acknowledge that this is a soft normalisation and it is possible to
        use negative out of gamut values or high dynamic range data exceeding
        1.

    Returns
    -------
    :class:`str`
        *Colour* domain-range scale.

    Warnings
    --------
    -   The **'Ignore'** and **'100'** domain-range scales are for internal
        usage only!
    """

    return _DOMAIN_RANGE_SCALE


def set_domain_range_scale(
    scale: Union[
        Literal["ignore", "reference" "Ignore", "Reference", "1", "100"], str
    ] = "reference"
):
    """
    Set the current *Colour* domain-range scale. The following scales are
    available:

    -   **'Reference'**, the default *Colour* domain-range scale which varies
        depending on the referenced algorithm, e.g. [0, 1], [0, 10], [0, 100],
        [0, 255], etc...
    -   **'1'**, a domain-range scale normalised to [0, 1], it is important to
        acknowledge that this is a soft normalisation and it is possible to
        use negative out of gamut values or high dynamic range data exceeding
        1.

    Parameters
    ----------
    scale
        *Colour* domain-range scale to set.

    Warnings
    --------
    -   The **'Ignore'** and **'100'** domain-range scales are for internal
        usage only!
    """

    global _DOMAIN_RANGE_SCALE

    _DOMAIN_RANGE_SCALE = validate_method(
        str(scale),
        ("ignore", "reference", "1", "100"),
        '"{0}" scale is invalid, it must be one of {1}!',
    )


class domain_range_scale:
    """
    Define context manager and decorator temporarily setting *Colour*
    domain-range scale.

    The following scales are available:

    -   **'Reference'**, the default *Colour* domain-range scale which varies
        depending on the referenced algorithm, e.g. [0, 1], [0, 10], [0, 100],
        [0, 255], etc...
    -   **'1'**, a domain-range scale normalised to [0, 1], it is important to
        acknowledge that this is a soft normalisation and it is possible to
        use negative out of gamut values or high dynamic range data exceeding
        1.

    Parameters
    ----------
    scale
        *Colour* domain-range scale to set.

    Warnings
    --------
    -   The **'Ignore'** and **'100'** domain-range scales are for internal
        usage only!

    Examples
    --------
    With *Colour* domain-range scale set to **'Reference'**:

    >>> with domain_range_scale('1'):
    ...     to_domain_1(1)
    array(1.0)
    >>> with domain_range_scale('Reference'):
    ...     from_range_1(1)
    array(1.0)

    With *Colour* domain-range scale set to **'1'**:

    >>> with domain_range_scale('1'):
    ...     to_domain_1(1)
    array(1.0)
    >>> with domain_range_scale('1'):
    ...     from_range_1(1)
    array(1.0)

    With *Colour* domain-range scale set to **'100'** (unsupported):

    >>> with domain_range_scale('100'):
    ...     to_domain_1(1)
    array(0.01)
    >>> with domain_range_scale('100'):
    ...     from_range_1(1)
    array(100.0)
    """

    def __init__(
        self,
        scale: Union[
            Literal["ignore", "reference" "Ignore", "Reference", "1", "100"],
            str,
        ],
    ):
        self._scale = scale
        self._previous_scale = get_domain_range_scale()

    def __enter__(self) -> domain_range_scale:
        """Set the new domain-range scale upon entering the context manager."""

        set_domain_range_scale(self._scale)

        return self

    def __exit__(self, *args: Any):
        """Set the previous domain-range scale upon exiting the context manager."""

        set_domain_range_scale(self._previous_scale)

    def __call__(self, function: Callable) -> Any:
        """Call the wrapped definition."""

        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with self:
                return function(*args, **kwargs)

        return wrapper


def to_domain_1(
    a: ArrayLike,
    scale_factor: FloatingOrArrayLike = 100,
    dtype: Optional[Type[DTypeFloating]] = None,
) -> NDArray:
    """
    Scale given array :math:`a` to domain **'1'**. The behaviour is as
    follows:

    -   If *Colour* domain-range scale is **'Reference'** or **'1'**, the
        definition is almost entirely by-passed and will conveniently convert
        array :math:`a` to :class:`np.ndarray`.
    -   If *Colour* domain-range scale is **'100'** (currently unsupported
        private value only used for unit tests), array :math:`a` is divided by
        ``scale_factor``, typically 100.

    Parameters
    ----------
    a
        Array :math:`a` to scale to domain **'1'**.
    scale_factor
        Scale factor, usually *numeric* but can be a :class:`numpy.ndarray` if
        some axis need different scaling to be brought to domain **'1'**.
    dtype
        Data type used for the conversion to :class:`np.ndarray`.

    Returns
    -------
    :class:`numpy.ndarray`
        Array :math:`a` scaled to domain **'1'**.

    Examples
    --------
    With *Colour* domain-range scale set to **'Reference'**:

    >>> with domain_range_scale('Reference'):
    ...     to_domain_1(1)
    array(1.0)

    With *Colour* domain-range scale set to **'1'**:

    >>> with domain_range_scale('1'):
    ...     to_domain_1(1)
    array(1.0)

    With *Colour* domain-range scale set to **'100'** (unsupported):

    >>> with domain_range_scale('100'):
    ...     to_domain_1(1)
    array(0.01)
    """

    dtype = cast(Type[DTypeFloating], optional(dtype, DEFAULT_FLOAT_DTYPE))

    a = as_float_array(a, dtype).copy()

    if _DOMAIN_RANGE_SCALE == "100":
        a /= as_float_array(scale_factor)

    return a


def to_domain_10(
    a: ArrayLike,
    scale_factor: FloatingOrArrayLike = 10,
    dtype: Optional[Type[DTypeFloating]] = None,
) -> NDArray:
    """
    Scale given array :math:`a` to domain **'10'**, used by
    *Munsell Renotation System*. The behaviour is as follows:

    -   If *Colour* domain-range scale is **'Reference'**, the
        definition is almost entirely by-passed and will conveniently convert
        array :math:`a` to :class:`np.ndarray`.
    -   If *Colour* domain-range scale is **'1'**, array :math:`a` is
        multiplied by ``scale_factor``, typically 10.
    -   If *Colour* domain-range scale is **'100'** (currently unsupported
        private value only used for unit tests), array :math:`a` is
        divided by ``scale_factor``, typically 10.

    Parameters
    ----------
    a
        Array :math:`a` to scale to domain **'10'**.
    scale_factor
        Scale factor, usually *numeric* but can be a :class:`numpy.ndarray` if
        some axis need different scaling to be brought to domain **'10'**.
    dtype
        Data type used for the conversion to :class:`np.ndarray`.

    Returns
    -------
    :class:`numpy.ndarray`
        Array :math:`a` scaled to domain **'10'**.

    Examples
    --------
    With *Colour* domain-range scale set to **'Reference'**:

    >>> with domain_range_scale('Reference'):
    ...     to_domain_10(1)
    array(1.0)

    With *Colour* domain-range scale set to **'1'**:

    >>> with domain_range_scale('1'):
    ...     to_domain_10(1)
    array(10.0)

    With *Colour* domain-range scale set to **'100'** (unsupported):

    >>> with domain_range_scale('100'):
    ...     to_domain_10(1)
    array(0.1)
    """

    dtype = cast(Type[DTypeFloating], optional(dtype, DEFAULT_FLOAT_DTYPE))

    a = as_float_array(a, dtype).copy()

    if _DOMAIN_RANGE_SCALE == "1":
        a *= as_float_array(scale_factor)

    if _DOMAIN_RANGE_SCALE == "100":
        a /= as_float_array(scale_factor)

    return a


def to_domain_100(
    a: ArrayLike,
    scale_factor: FloatingOrArrayLike = 100,
    dtype: Optional[Type[DTypeFloating]] = None,
) -> NDArray:
    """
    Scale given array :math:`a` to domain **'100'**. The behaviour is as
    follows:

    -   If *Colour* domain-range scale is **'Reference'** or **'100'**
        (currently unsupported private value only used for unit tests), the
        definition is almost entirely by-passed and will conveniently convert
        array :math:`a` to :class:`np.ndarray`.
    -   If *Colour* domain-range scale is **'1'**, array :math:`a` is
        multiplied by ``scale_factor``, typically 100.

    Parameters
    ----------
    a
        Array :math:`a` to scale to domain **'100'**.
    scale_factor
        Scale factor, usually *numeric* but can be a :class:`numpy.ndarray` if
        some axis need different scaling to be brought to domain **'100'**.
    dtype
        Data type used for the conversion to :class:`np.ndarray`.

    Returns
    -------
    :class:`numpy.ndarray`
        Array :math:`a` scaled to domain **'100'**.

    Examples
    --------
    With *Colour* domain-range scale set to **'Reference'**:

    >>> with domain_range_scale('Reference'):
    ...     to_domain_100(1)
    array(1.0)

    With *Colour* domain-range scale set to **'1'**:

    >>> with domain_range_scale('1'):
    ...     to_domain_100(1)
    array(100.0)

    With *Colour* domain-range scale set to **'100'** (unsupported):

    >>> with domain_range_scale('100'):
    ...     to_domain_100(1)
    array(1.0)
    """

    dtype = cast(Type[DTypeFloating], optional(dtype, DEFAULT_FLOAT_DTYPE))

    a = as_float_array(a, dtype).copy()

    if _DOMAIN_RANGE_SCALE == "1":
        a *= as_float_array(scale_factor)

    return a


def to_domain_degrees(
    a: ArrayLike,
    scale_factor: FloatingOrArrayLike = 360,
    dtype: Optional[Type[DTypeFloating]] = None,
) -> NDArray:
    """
    Scale given array :math:`a` to degrees domain. The behaviour is as
    follows:

    -   If *Colour* domain-range scale is **'Reference'**, the
        definition is almost entirely by-passed and will conveniently convert
        array :math:`a` to :class:`np.ndarray`.
    -   If *Colour* domain-range scale is **'1'**, array :math:`a` is
        multiplied by ``scale_factor``, typically 360.
    -   If *Colour* domain-range scale is **'100'** (currently unsupported
        private value only used for unit tests), array :math:`a` is
        multiplied by ``scale_factor`` / 100, typically 360 / 100.

    Parameters
    ----------
    a
        Array :math:`a` to scale to degrees domain.
    scale_factor
        Scale factor, usually *numeric* but can be a :class:`numpy.ndarray` if
        some axis need different scaling to be brought to degrees domain.
    dtype
        Data type used for the conversion to :class:`np.ndarray`.

    Returns
    -------
    :class:`numpy.ndarray`
        Array :math:`a` scaled to degrees domain.

    Examples
    --------
    With *Colour* domain-range scale set to **'Reference'**:

    >>> with domain_range_scale('Reference'):
    ...     to_domain_degrees(1)
    array(1.0)

    With *Colour* domain-range scale set to **'1'**:

    >>> with domain_range_scale('1'):
    ...     to_domain_degrees(1)
    array(360.0)

    With *Colour* domain-range scale set to **'100'** (unsupported):

    >>> with domain_range_scale('100'):
    ...     to_domain_degrees(1)
    array(3.6)
    """

    dtype = cast(Type[DTypeFloating], optional(dtype, DEFAULT_FLOAT_DTYPE))

    a = as_float_array(a, dtype).copy()

    if _DOMAIN_RANGE_SCALE == "1":
        a *= as_float_array(scale_factor)

    if _DOMAIN_RANGE_SCALE == "100":
        a *= as_float_array(scale_factor) / 100

    return a


def to_domain_int(
    a: ArrayLike,
    bit_depth: IntegerOrArrayLike = 8,
    dtype: Optional[Type[DTypeFloating]] = None,
) -> NDArray:
    """
    Scale given array :math:`a` to int domain. The behaviour is as follows:

    -   If *Colour* domain-range scale is **'Reference'**, the
        definition is almost entirely by-passed and will conveniently convert
        array :math:`a` to :class:`np.ndarray`.
    -   If *Colour* domain-range scale is **'1'**, array :math:`a` is
        multiplied by :math:`2^{bit\\_depth} - 1`.
    -   If *Colour* domain-range scale is **'100'** (currently unsupported
        private value only used for unit tests), array :math:`a` is
        multiplied by :math:`2^{bit\\_depth} - 1`.

    Parameters
    ----------
    a
        Array :math:`a` to scale to int domain.
    bit_depth
        Bit depth, usually *integer* but can be a :class:`numpy.ndarray` if
        some axis need different scaling to be brought to int domain.
    dtype
        Data type used for the conversion to :class:`np.ndarray`.

    Returns
    -------
    :class:`numpy.ndarray`
        Array :math:`a` scaled to int domain.

    Notes
    -----
    -   To avoid precision issues and rounding, the scaling is performed on
        *float* numbers.

    Examples
    --------
    With *Colour* domain-range scale set to **'Reference'**:

    >>> with domain_range_scale('Reference'):
    ...     to_domain_int(1)
    array(1.0)

    With *Colour* domain-range scale set to **'1'**:

    >>> with domain_range_scale('1'):
    ...     to_domain_int(1)
    array(255.0)

    With *Colour* domain-range scale set to **'100'** (unsupported):

    >>> with domain_range_scale('100'):
    ...     to_domain_int(1)
    array(2.55)
    """

    dtype = cast(Type[DTypeFloating], optional(dtype, DEFAULT_FLOAT_DTYPE))

    a = as_float_array(a, dtype).copy()

    maximum_code_value = np.power(2, bit_depth) - 1
    if _DOMAIN_RANGE_SCALE == "1":
        a *= maximum_code_value

    if _DOMAIN_RANGE_SCALE == "100":
        a *= maximum_code_value / 100

    return a


def from_range_1(
    a: ArrayLike,
    scale_factor: FloatingOrArrayLike = 100,
    dtype: Optional[Type[DTypeFloating]] = None,
) -> NDArray:
    """
    Scale given array :math:`a` from range **'1'**. The behaviour is as
    follows:

    -   If *Colour* domain-range scale is **'Reference'** or **'1'**, the
        definition is entirely by-passed.
    -   If *Colour* domain-range scale is **'100'** (currently unsupported
        private value only used for unit tests), array :math:`a` is multiplied
        by ``scale_factor``, typically 100.

    Parameters
    ----------
    a
        Array :math:`a` to scale from range **'1'**.
    scale_factor
        Scale factor, usually *numeric* but can be a :class:`numpy.ndarray` if
        some axis need different scaling to be brought from range **'1'**.
    dtype
        Data type used for the conversion to :class:`np.ndarray`.

    Returns
    -------
    :class:`numpy.ndarray`
        Array :math:`a` scaled from range **'1'**.

    Warnings
    --------
    The scale conversion of variable :math:`a` happens in-place, i.e. :math:`a`
    will be mutated!

    Examples
    --------
    With *Colour* domain-range scale set to **'Reference'**:

    >>> with domain_range_scale('Reference'):
    ...     from_range_1(1)
    array(1.0)

    With *Colour* domain-range scale set to **'1'**:

    >>> with domain_range_scale('1'):
    ...     from_range_1(1)
    array(1.0)

    With *Colour* domain-range scale set to **'100'** (unsupported):

    >>> with domain_range_scale('100'):
    ...     from_range_1(1)
    array(100.0)
    """

    dtype = cast(Type[DTypeFloating], optional(dtype, DEFAULT_FLOAT_DTYPE))

    a = as_float_array(a, dtype)

    if _DOMAIN_RANGE_SCALE == "100":
        a *= as_float_array(scale_factor)

    return a


def from_range_10(
    a: ArrayLike,
    scale_factor: FloatingOrArrayLike = 10,
    dtype: Optional[Type[DTypeFloating]] = None,
) -> NDArray:
    """
    Scale given array :math:`a` from range **'10'**, used by
    *Munsell Renotation System*. The behaviour is as follows:

    -   If *Colour* domain-range scale is **'Reference'**, the
        definition is entirely by-passed.
    -   If *Colour* domain-range scale is **'1'**, array :math:`a` is
        divided by ``scale_factor``, typically 10.
    -   If *Colour* domain-range scale is **'100'** (currently unsupported
        private value only used for unit tests), array :math:`a` is
        multiplied by ``scale_factor``, typically 10.

    Parameters
    ----------
    a
        Array :math:`a` to scale from range **'10'**.
    scale_factor
        Scale factor, usually *numeric* but can be a :class:`numpy.ndarray` if
        some axis need different scaling to be brought from range **'10'**.
    dtype
        Data type used for the conversion to :class:`np.ndarray`.

    Returns
    -------
    :class:`numpy.ndarray`
        Array :math:`a` scaled from range **'10'**.

    Warnings
    --------
    The scale conversion of variable :math:`a` happens in-place, i.e. :math:`a`
    will be mutated!

    Examples
    --------
    With *Colour* domain-range scale set to **'Reference'**:

    >>> with domain_range_scale('Reference'):
    ...     from_range_10(1)
    array(1.0)

    With *Colour* domain-range scale set to **'1'**:

    >>> with domain_range_scale('1'):
    ...     from_range_10(1)
    array(0.1)

    With *Colour* domain-range scale set to **'100'** (unsupported):

    >>> with domain_range_scale('100'):
    ...     from_range_10(1)
    array(10.0)
    """

    dtype = cast(Type[DTypeFloating], optional(dtype, DEFAULT_FLOAT_DTYPE))

    a = as_float_array(a, dtype)

    if _DOMAIN_RANGE_SCALE == "1":
        a /= as_float_array(scale_factor)

    if _DOMAIN_RANGE_SCALE == "100":
        a *= as_float_array(scale_factor)

    return a


def from_range_100(
    a: ArrayLike,
    scale_factor: FloatingOrArrayLike = 100,
    dtype: Optional[Type[DTypeFloating]] = None,
) -> NDArray:
    """
    Scale given array :math:`a` from range **'100'**. The behaviour is as
    follows:

    -   If *Colour* domain-range scale is **'Reference'** or **'100'**
        (currently unsupported private value only used for unit tests), the
        definition is entirely by-passed.
    -   If *Colour* domain-range scale is **'1'**, array :math:`a` is
        divided by ``scale_factor``, typically 100.

    Parameters
    ----------
    a
        Array :math:`a` to scale from range **'100'**.
    scale_factor
        Scale factor, usually *numeric* but can be a :class:`numpy.ndarray` if
        some axis need different scaling to be brought from range **'100'**.
    dtype
        Data type used for the conversion to :class:`np.ndarray`.

    Returns
    -------
    :class:`numpy.ndarray`
        Array :math:`a` scaled from range **'100'**.

    Warnings
    --------
    The scale conversion of variable :math:`a` happens in-place, i.e. :math:`a`
    will be mutated!

    Examples
    --------
    With *Colour* domain-range scale set to **'Reference'**:

    >>> with domain_range_scale('Reference'):
    ...     from_range_100(1)
    array(1.0)

    With *Colour* domain-range scale set to **'1'**:

    >>> with domain_range_scale('1'):
    ...     from_range_100(1)
    array(0.01)

    With *Colour* domain-range scale set to **'100'** (unsupported):

    >>> with domain_range_scale('100'):
    ...     from_range_100(1)
    array(1.0)
    """

    dtype = cast(Type[DTypeFloating], optional(dtype, DEFAULT_FLOAT_DTYPE))

    a = as_float_array(a, dtype)

    if _DOMAIN_RANGE_SCALE == "1":
        a /= as_float_array(scale_factor)

    return a


def from_range_degrees(
    a: ArrayLike,
    scale_factor: FloatingOrArrayLike = 360,
    dtype: Optional[Type[DTypeFloating]] = None,
) -> NDArray:
    """
    Scale given array :math:`a` from degrees range. The behaviour is as
    follows:

    -   If *Colour* domain-range scale is **'Reference'**, the
        definition is entirely by-passed.
    -   If *Colour* domain-range scale is **'1'**, array :math:`a` is
        divided by ``scale_factor``, typically 360.
    -   If *Colour* domain-range scale is **'100'** (currently unsupported
        private value only used for unit tests), array :math:`a` is
        divided by ``scale_factor`` / 100, typically 360 / 100.

    Parameters
    ----------
    a
        Array :math:`a` to scale from degrees range.
    scale_factor
        Scale factor, usually *numeric* but can be a :class:`numpy.ndarray` if
        some axis need different scaling to be brought from degrees range.
    dtype
        Data type used for the conversion to :class:`np.ndarray`.

    Returns
    -------
    :class:`numpy.ndarray`
        Array :math:`a` scaled from degrees range.

    Warnings
    --------
    The scale conversion of variable :math:`a` happens in-place, i.e. :math:`a`
    will be mutated!

    Examples
    --------
    With *Colour* domain-range scale set to **'Reference'**:

    >>> with domain_range_scale('Reference'):
    ...     from_range_degrees(1)
    array(1.0)

    With *Colour* domain-range scale set to **'1'**:

    >>> with domain_range_scale('1'):
    ...     from_range_degrees(1)  # doctest: +ELLIPSIS
    array(0.0027777...)

    With *Colour* domain-range scale set to **'100'** (unsupported):

    >>> with domain_range_scale('100'):
    ...     from_range_degrees(1)  # doctest: +ELLIPSIS
    array(0.2777777...)
    """

    dtype = cast(Type[DTypeFloating], optional(dtype, DEFAULT_FLOAT_DTYPE))

    a = as_float_array(a, dtype)

    if _DOMAIN_RANGE_SCALE == "1":
        a /= as_float_array(scale_factor)

    if _DOMAIN_RANGE_SCALE == "100":
        a /= as_float_array(scale_factor) / 100

    return a


def from_range_int(
    a: ArrayLike,
    bit_depth: IntegerOrArrayLike = 8,
    dtype: Optional[Type[DTypeFloating]] = None,
) -> NDArray:
    """
    Scale given array :math:`a` from int range. The behaviour is as follows:

    -   If *Colour* domain-range scale is **'Reference'**, the
        definition is entirely by-passed.
    -   If *Colour* domain-range scale is **'1'**, array :math:`a` is converted
        to :class:`np.ndarray` and divided by :math:`2^{bit\\_depth} - 1`.
    -   If *Colour* domain-range scale is **'100'** (currently unsupported
        private value only used for unit tests), array :math:`a` is converted
        to :class:`np.ndarray` and divided by :math:`2^{bit\\_depth} - 1`.

    Parameters
    ----------
    a
        Array :math:`a` to scale from int range.
    bit_depth
        Bit depth, usually *integer* but can be a :class:`numpy.ndarray` if
        some axis need different scaling to be brought from int range.
    dtype
        Data type used for the conversion to :class:`np.ndarray`.

    Returns
    -------
    :class:`numpy.ndarray`
        Array :math:`a` scaled from int range.

    Warnings
    --------
    The scale conversion of variable :math:`a` happens in-place, i.e. :math:`a`
    will be mutated!

    Notes
    -----
    -   To avoid precision issues and rounding, the scaling is performed on
        *float* numbers.

    Examples
    --------
    With *Colour* domain-range scale set to **'Reference'**:

    >>> with domain_range_scale('Reference'):
    ...     from_range_int(1)
    array(1.0)

    With *Colour* domain-range scale set to **'1'**:

    >>> with domain_range_scale('1'):
    ...     from_range_int(1)  # doctest: +ELLIPSIS
    array(0.0039215...)

    With *Colour* domain-range scale set to **'100'** (unsupported):

    >>> with domain_range_scale('100'):
    ...     from_range_int(1)  # doctest: +ELLIPSIS
    array(0.3921568...)
    """

    dtype = cast(Type[DTypeFloating], optional(dtype, DEFAULT_FLOAT_DTYPE))

    a = as_float_array(a, dtype)

    maximum_code_value = np.power(2, bit_depth) - 1
    if _DOMAIN_RANGE_SCALE == "1":
        a /= maximum_code_value

    if _DOMAIN_RANGE_SCALE == "100":
        a /= maximum_code_value / 100

    return a


def closest_indexes(a: ArrayLike, b: ArrayLike) -> NDArray:
    """
    Return the array :math:`a` closest element indexes to the reference array
    :math:`b` elements.

    Parameters
    ----------
    a
        Array :math:`a` to search for the closest element indexes.
    b
        Reference array :math:`b`.

    Returns
    -------
    :class:`numpy.ndarray`
        Closest array :math:`a` element indexes.

    Examples
    --------
    >>> a = np.array([24.31357115, 63.62396289, 55.71528816,
    ...               62.70988028, 46.84480573, 25.40026416])
    >>> print(closest_indexes(a, 63))
    [3]
    >>> print(closest_indexes(a, [63, 25]))
    [3 5]
    """

    a = np.ravel(a)[:, np.newaxis]
    b = np.ravel(b)[np.newaxis, :]

    return np.abs(a - b).argmin(axis=0)


def closest(a: ArrayLike, b: ArrayLike) -> NDArray:
    """
    Return the array :math:`a` closest elements to the reference array
    :math:`b` elements.

    Parameters
    ----------
    a
        Array :math:`a` to search for the closest element.
    b
        Reference array :math:`b`.

    Returns
    -------
    :class:`numpy.ndarray`
        Closest array :math:`a` elements.

    Examples
    --------
    >>> a = np.array([24.31357115, 63.62396289, 55.71528816,
    ...               62.70988028, 46.84480573, 25.40026416])
    >>> closest(a, 63)
    array([ 62.70988028])
    >>> closest(a, [63, 25])
    array([ 62.70988028,  25.40026416])
    """

    a = np.array(a)

    return a[closest_indexes(a, b)]


def interval(distribution: ArrayLike, unique: Boolean = True) -> NDArray:
    """
    Return the interval size of given distribution.

    Parameters
    ----------
    distribution
        Distribution to retrieve the interval.
    unique
        Whether to return unique intervals if  the distribution is
        non-uniformly spaced or the complete intervals

    Returns
    -------
    :class:`numpy.ndarray`
        Distribution interval.

    Examples
    --------
    Uniformly spaced variable:

    >>> y = np.array([1, 2, 3, 4, 5])
    >>> interval(y)
    array([ 1.])
    >>> interval(y, False)
    array([ 1.,  1.,  1.,  1.])

    Non-uniformly spaced variable:

    >>> y = np.array([1, 2, 3, 4, 8])
    >>> interval(y)
    array([ 1.,  4.])
    >>> interval(y, False)
    array([ 1.,  1.,  1.,  4.])
    """

    distribution = as_float_array(distribution)
    i = np.arange(distribution.size - 1)

    differences = np.abs(distribution[i + 1] - distribution[i])
    if unique:
        return np.unique(differences)
    else:
        return differences


def is_uniform(distribution: ArrayLike) -> Boolean:
    """
    Return whether given distribution is uniform.

    Parameters
    ----------
    distribution
        Distribution to check the uniformity of.

    Returns
    -------
    :class:`bool`
        Whether distribution uniform.

    Examples
    --------
    Uniformly spaced variable:

    >>> a = np.array([1, 2, 3, 4, 5])
    >>> is_uniform(a)
    True

    Non-uniformly spaced variable:

    >>> a = np.array([1, 2, 3.1415, 4, 5])
    >>> is_uniform(a)
    False
    """

    return True if interval(distribution).size == 1 else False


def in_array(
    a: ArrayLike, b: ArrayLike, tolerance: Number = EPSILON
) -> NDArray:
    """
    Return whether each element of the array :math:`a` is also present in the
    array :math:`b` within given tolerance.

    Parameters
    ----------
    a
        Array :math:`a` to test the elements from.
    b
        The array :math:`b` against which to test the elements of array
        :math:`a`.
    tolerance
        Tolerance value.

    Returns
    -------
    :class:`numpy.ndarray`
        A boolean array with array :math:`a` shape describing whether an
        element of array :math:`a` is present in array :math:`b` within given
        tolerance.

    References
    ----------
    :cite:`Yorke2014a`

    Examples
    --------
    >>> a = np.array([0.50, 0.60])
    >>> b = np.linspace(0, 10, 101)
    >>> np.in1d(a, b)
    array([ True, False], dtype=bool)
    >>> in_array(a, b)
    array([ True,  True], dtype=bool)
    """

    a = as_float_array(a)
    b = as_float_array(b)

    d = np.abs(np.ravel(a) - b[..., np.newaxis])

    return np.reshape(np.any(d <= tolerance, axis=0), a.shape)


def tstack(
    a: Union[ArrayLike, NestedSequence[NumberOrArrayLike]],
    dtype: Optional[Union[Type[DTypeBoolean], Type[DTypeNumber]]] = None,
) -> NDArray:
    """
    Stack given array of arrays :math:`a` along the last axis (tail) to
    produce a stacked array.

    It is used to stack an array of arrays produced by the
    :func:`colour.utilities.tsplit` definition.

    Parameters
    ----------
    a
        Array of arrays :math:`a` to stack along the last axis.
    dtype
        :class:`numpy.dtype` to use for initial conversion to
        :class:`numpy.ndarray`, default to the :class:`numpy.dtype` defined by
        :attr:`colour.constant.DEFAULT_FLOAT_DTYPE` attribute.

    Returns
    -------
    :class:`numpy.ndarray`
        Stacked array.

    Examples
    --------
    >>> a = 0
    >>> tstack([a, a, a])
    array([ 0.,  0.,  0.])
    >>> a = np.arange(0, 6)
    >>> tstack([a, a, a])
    array([[ 0.,  0.,  0.],
           [ 1.,  1.,  1.],
           [ 2.,  2.,  2.],
           [ 3.,  3.,  3.],
           [ 4.,  4.,  4.],
           [ 5.,  5.,  5.]])
    >>> a = np.reshape(a, (1, 6))
    >>> tstack([a, a, a])
    array([[[ 0.,  0.,  0.],
            [ 1.,  1.,  1.],
            [ 2.,  2.,  2.],
            [ 3.,  3.,  3.],
            [ 4.,  4.,  4.],
            [ 5.,  5.,  5.]]])
    >>> a = np.reshape(a, (1, 1, 6))
    >>> tstack([a, a, a])
    array([[[[ 0.,  0.,  0.],
             [ 1.,  1.,  1.],
             [ 2.,  2.,  2.],
             [ 3.,  3.,  3.],
             [ 4.,  4.,  4.],
             [ 5.,  5.,  5.]]]])
    """

    dtype = cast(Type[DTypeFloating], optional(dtype, DEFAULT_FLOAT_DTYPE))

    a = as_array(a, dtype)

    return np.concatenate([x[..., np.newaxis] for x in a], axis=-1)


def tsplit(
    a: Union[ArrayLike, NestedSequence[NumberOrArrayLike]],
    dtype: Optional[Union[Type[DTypeBoolean], Type[DTypeNumber]]] = None,
) -> NDArray:
    """
    Split given stacked array :math:`a` along the last axis (tail) to produce
    an array of arrays.

    It is used to split a stacked array produced by the
    :func:`colour.utilities.tstack` definition.

    Parameters
    ----------
    a
        Stacked array :math:`a` to split.
    dtype
        :class:`numpy.dtype` to use for initial conversion to
        :class:`numpy.ndarray`, default to the :class:`numpy.dtype` defined by
        :attr:`colour.constant.DEFAULT_FLOAT_DTYPE` attribute.

    Returns
    -------
    :class:`numpy.ndarray`
        Array of arrays.

    Examples
    --------
    >>> a = np.array([0, 0, 0])
    >>> tsplit(a)
    array([ 0.,  0.,  0.])
    >>> a = np.array(
    ...     [[0, 0, 0],
    ...      [1, 1, 1],
    ...      [2, 2, 2],
    ...      [3, 3, 3],
    ...      [4, 4, 4],
    ...      [5, 5, 5]]
    ... )
    >>> tsplit(a)
    array([[ 0.,  1.,  2.,  3.,  4.,  5.],
           [ 0.,  1.,  2.,  3.,  4.,  5.],
           [ 0.,  1.,  2.,  3.,  4.,  5.]])
    >>> a = np.array(
    ...     [[[0, 0, 0],
    ...       [1, 1, 1],
    ...       [2, 2, 2],
    ...       [3, 3, 3],
    ...       [4, 4, 4],
    ...       [5, 5, 5]]]
    ... )
    >>> tsplit(a)
    array([[[ 0.,  1.,  2.,  3.,  4.,  5.]],
    <BLANKLINE>
           [[ 0.,  1.,  2.,  3.,  4.,  5.]],
    <BLANKLINE>
           [[ 0.,  1.,  2.,  3.,  4.,  5.]]])
    """

    dtype = cast(Type[DTypeFloating], optional(dtype, DEFAULT_FLOAT_DTYPE))

    a = as_array(a, dtype)

    return np.array([a[..., x] for x in range(a.shape[-1])])


def row_as_diagonal(a: ArrayLike) -> NDArray:
    """
    Return the rows of given array :math:`a` as diagonal matrices.

    Parameters
    ----------
    a
        Array :math:`a` to returns the rows of as diagonal matrices.

    Returns
    -------
    :class:`numpy.ndarray`
        Array :math:`a` rows as diagonal matrices.

    References
    ----------
    :cite:`Castro2014a`

    Examples
    --------
    >>> a = np.array(
    ...     [[0.25891593, 0.07299478, 0.36586996],
    ...       [0.30851087, 0.37131459, 0.16274825],
    ...       [0.71061831, 0.67718718, 0.09562581],
    ...       [0.71588836, 0.76772047, 0.15476079],
    ...       [0.92985142, 0.22263399, 0.88027331]]
    ... )
    >>> row_as_diagonal(a)
    array([[[ 0.25891593,  0.        ,  0.        ],
            [ 0.        ,  0.07299478,  0.        ],
            [ 0.        ,  0.        ,  0.36586996]],
    <BLANKLINE>
           [[ 0.30851087,  0.        ,  0.        ],
            [ 0.        ,  0.37131459,  0.        ],
            [ 0.        ,  0.        ,  0.16274825]],
    <BLANKLINE>
           [[ 0.71061831,  0.        ,  0.        ],
            [ 0.        ,  0.67718718,  0.        ],
            [ 0.        ,  0.        ,  0.09562581]],
    <BLANKLINE>
           [[ 0.71588836,  0.        ,  0.        ],
            [ 0.        ,  0.76772047,  0.        ],
            [ 0.        ,  0.        ,  0.15476079]],
    <BLANKLINE>
           [[ 0.92985142,  0.        ,  0.        ],
            [ 0.        ,  0.22263399,  0.        ],
            [ 0.        ,  0.        ,  0.88027331]]])
    """

    d = as_array(a)

    d = np.expand_dims(d, -2)

    return np.eye(d.shape[-1]) * d


def orient(
    a: ArrayLike,
    orientation: Union[Literal["Flip", "Flop", "90 CW", "90 CCW", "180"], str],
) -> Union[NDArray, None]:
    """
    Orient given array :math:`a` according to given orientation.

    Parameters
    ----------
    a
        Array :math:`a` to orient.
    orientation
        Orientation to perform.

    Returns
    -------
    :class:`numpy.ndarray`
        Oriented array.

    Examples
    --------
    >>> a = np.tile(np.arange(5), (5, 1))
    >>> a
    array([[0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4]])
    >>> orient(a, '90 CW')
    array([[0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1],
           [2, 2, 2, 2, 2],
           [3, 3, 3, 3, 3],
           [4, 4, 4, 4, 4]])
    >>> orient(a, 'Flip')
    array([[4, 3, 2, 1, 0],
           [4, 3, 2, 1, 0],
           [4, 3, 2, 1, 0],
           [4, 3, 2, 1, 0],
           [4, 3, 2, 1, 0]])
    """

    orientation = validate_method(
        orientation, ["Flip", "Flop", "90 CW", "90 CCW", "180"]
    )

    if orientation == "flip":
        return np.fliplr(a)
    elif orientation == "flop":
        return np.flipud(a)
    elif orientation == "90 cw":
        return np.rot90(a, 3)
    elif orientation == "90 ccw":
        return np.rot90(a)
    elif orientation == "180":
        return np.rot90(a, 2)
    else:  # pragma: no cover
        return None


def centroid(a: ArrayLike) -> NDArray:
    """
    Return the centroid indexes of given array :math:`a`.

    Parameters
    ----------
    a
        Array :math:`a` to returns the centroid indexes of.

    Returns
    -------
    :class:`numpy.ndarray`
        Array :math:`a` centroid indexes.

    Examples
    --------
    >>> a = np.tile(np.arange(0, 5), (5, 1))
    >>> centroid(a)  # doctest: +ELLIPSIS
    array([2, 3]...)
    """

    a = as_float_array(a)

    a_s = np.sum(a)

    ranges = [np.arange(0, a.shape[i]) for i in range(a.ndim)]
    coordinates = np.meshgrid(*ranges)

    a_ci = []
    for axis in coordinates:
        axis = np.transpose(axis)
        # Aligning axis for N-D arrays where N is normalised to
        # range [3, :math:`\\\infty`]
        for i in range(axis.ndim - 2, 0, -1):
            axis = np.rollaxis(axis, i - 1, axis.ndim)

        a_ci.append(np.sum(axis * a) // a_s)

    return np.array(a_ci).astype(DEFAULT_INT_DTYPE)


def fill_nan(
    a: ArrayLike,
    method: Union[Literal["Interpolation", "Constant"], str] = "Interpolation",
    default: Number = 0,
) -> NDArray:
    """
    Fill given array :math:`a` NaN values according to given method.

    Parameters
    ----------
    a
        Array :math:`a` to fill the NaNs of.
    method
        *Interpolation* method linearly interpolates through the NaN values,
        *Constant* method replaces NaN values with ``default``.
    default
        Value to use with the *Constant* method.

    Returns
    -------
    :class:`numpy.ndarray`
        NaNs filled array :math:`a`.

    Examples
    --------
    >>> a = np.array([0.1, 0.2, np.nan, 0.4, 0.5])
    >>> fill_nan(a)
    array([ 0.1,  0.2,  0.3,  0.4,  0.5])
    >>> fill_nan(a, method='Constant')
    array([ 0.1,  0.2,  0. ,  0.4,  0.5])
    """

    a = np.array(a, copy=True)
    method = validate_method(method, ["Interpolation", "Constant"])

    mask = np.isnan(a)

    if method == "interpolation":
        a[mask] = np.interp(
            np.flatnonzero(mask), np.flatnonzero(~mask), a[~mask]
        )
    elif method == "constant":
        a[mask] = default

    return a


def has_only_nan(a: ArrayLike) -> Boolean:
    """
    Return whether given array :math:`a` contains only NaN values.

    Parameters
    ----------
    a
        Array :math:`a` to check whether it contains only NaN values.

    Returns
    -------
    :class:`bool`
        Whether array :math:`a` contains only NaN values.

    Examples
    --------
    >>> has_only_nan(None)
    True
    >>> has_only_nan([None, None])
    True
    >>> has_only_nan([True, None])
    False
    >>> has_only_nan([0.1, np.nan, 0.3])
    False
    """

    a = as_float_array(a)

    return bool(np.all(np.isnan(a)))


@contextmanager
def ndarray_write(a: ArrayLike) -> Generator:
    """
    Define a context manager setting given array :math:`a` writeable to
    operate one and then read-only.

    Parameters
    ----------
    a
        Array :math:`a` to operate on.

    Yields
    ------
    Generator
        Array :math:`a` operated.

    Examples
    --------
    >>> a = np.linspace(0, 1, 10)
    >>> a.setflags(write=False)
    >>> try:
    ...     a += 1
    ... except ValueError:
    ...     pass
    >>> with ndarray_write(a):
    ...     a +=1
    """

    a = as_float_array(a)

    a.setflags(write=True)

    try:
        yield a
    finally:
        a.setflags(write=False)


def zeros(
    shape: Union[Integer, Tuple[int, ...]],
    dtype: Optional[Type[DTypeNumber]] = None,
    order: Literal["C", "F"] = "C",
) -> NDArray:
    """
    Wrap :func:`np.zeros` definition to create an array with the active
    :class:`numpy.dtype` defined by the
    :attr:`colour.constant.DEFAULT_FLOAT_DTYPE` attribute.

    Parameters
    ----------
    shape
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    dtype
        :class:`numpy.dtype` to use for conversion, default to the
        :class:`numpy.dtype` defined by the
        :attr:`colour.constant.DEFAULT_FLOAT_DTYPE` attribute.
    order
        Whether to store multi-dimensional data in row-major (C-style) or
        column-major (Fortran-style) order in memory.

    Returns
    -------
    :class:`numpy.ndarray`
        Array of given shape and :class:`numpy.dtype`, filled with zeros.

    Examples
    --------
    >>> zeros(3)
    array([ 0.,  0.,  0.])
    """

    dtype = cast(Type[DTypeFloating], optional(dtype, DEFAULT_FLOAT_DTYPE))

    return np.zeros(shape, dtype, order)


def ones(
    shape: Union[Integer, Tuple[int, ...]],
    dtype: Optional[Type[DTypeNumber]] = None,
    order: Literal["C", "F"] = "C",
) -> NDArray:
    """
    Wrap :func:`np.ones` definition to create an array with the active
    :class:`numpy.dtype` defined by the
    :attr:`colour.constant.DEFAULT_FLOAT_DTYPE` attribute.

    Parameters
    ----------
    shape
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    dtype
        :class:`numpy.dtype` to use for conversion, default to the
        :class:`numpy.dtype` defined by the
        :attr:`colour.constant.DEFAULT_FLOAT_DTYPE` attribute.
    order
        Whether to store multi-dimensional data in row-major (C-style) or
        column-major (Fortran-style) order in memory.

    Returns
    -------
    :class:`numpy.ndarray`
        Array of given shape and type, filled with ones.

    Examples
    --------
    >>> ones(3)
    array([ 1.,  1.,  1.])
    """

    dtype = cast(Type[DTypeFloating], optional(dtype, DEFAULT_FLOAT_DTYPE))

    return np.ones(shape, dtype, order)


def full(
    shape: Union[Integer, Tuple[int, ...]],
    fill_value: Number,
    dtype: Optional[Type[DTypeNumber]] = None,
    order: Literal["C", "F"] = "C",
) -> NDArray:
    """
    Wrap :func:`np.full` definition to create an array with the active type
    defined by the:attr:`colour.constant.DEFAULT_FLOAT_DTYPE` attribute.

    Parameters
    ----------
    shape
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    fill_value
        Fill value.
    dtype
        :class:`numpy.dtype` to use for conversion, default to the
        :class:`numpy.dtype` defined by the
        :attr:`colour.constant.DEFAULT_FLOAT_DTYPE` attribute.
    order
        Whether to store multi-dimensional data in row-major (C-style) or
        column-major (Fortran-style) order in memory.

    Returns
    -------
    :class:`numpy.ndarray`
        Array of given shape and :class:`numpy.dtype`, filled with given value.

    Examples
    --------
    >>> ones(3)
    array([ 1.,  1.,  1.])
    """

    dtype = cast(Type[DTypeFloating], optional(dtype, DEFAULT_FLOAT_DTYPE))

    return np.full(shape, fill_value, dtype, order)


def index_along_last_axis(a: ArrayLike, indexes: ArrayLike) -> NDArray:
    """
    Reduce the dimension of array :math:`a` by one, by using an array of
    indexes to pick elements off the last axis.

    Parameters
    ----------
    a
        Array :math:`a` to be indexed.
    indexes
        *Integer* array with the same shape as `a` but with one dimension
        fewer, containing indices to the last dimension of `a`. All elements
        must be numbers between `0` and `m` - 1.

    Returns
    -------
    :class:`numpy.ndarray`
        Indexed array :math:`a`.

    Raises
    ------
    :class:`ValueError`
        If the array :math:`a` and ``indexes`` have incompatible shapes.
    :class:`IndexError`
        If ``indexes`` has elements outside of the allowed range of 0 to
        `m` - 1 or if it's not an *integer* array.

    Examples
    --------
    >>> a = np.array(
    ...     [[[0.3, 0.5, 6.9],
    ...       [3.3, 4.4, 1.6],
    ...       [4.4, 7.5, 2.3],
    ...       [2.3, 1.6, 7.4]],
    ...      [[2. , 5.9, 2.8],
    ...       [6.2, 4.9, 8.6],
    ...       [3.7, 9.7, 7.3],
    ...       [6.3, 4.3, 3.2]],
    ...      [[0.8, 1.9, 0.7],
    ...       [5.6, 4. , 1.7],
    ...       [6.7, 8.2, 1.7],
    ...       [1.2, 7.1, 1.4]],
    ...      [[4. , 4.8, 8.9],
    ...       [4. , 0.3, 6.9],
    ...       [3.5, 7.1, 4.5],
    ...       [1.4, 1.9, 1.6]]]
    ... )
    >>> indexes = np.array(
    ...     [[2, 0, 1, 1],
    ...      [2, 1, 1, 0],
    ...      [0, 0, 1, 2],
    ...      [0, 0, 1, 2]]
    ... )
    >>> index_along_last_axis(a, indexes)
    array([[ 6.9,  3.3,  7.5,  1.6],
           [ 2.8,  4.9,  9.7,  6.3],
           [ 0.8,  5.6,  8.2,  1.4],
           [ 4. ,  4. ,  7.1,  1.6]])

    This function can be used to compute the result of :func:`np.min` along
    the last axis given the corresponding :func:`np.argmin` indexes.

    >>> indexes = np.argmin(a, axis=-1)
    >>> np.array_equal(
    ...     index_along_last_axis(a, indexes),
    ...     np.min(a, axis=-1)
    ... )
    True

    In particular, this can be used to manipulate the indexes given by
    functions like :func:`np.min` before indexing the array. For example, to
    get elements directly following the smallest elements:

    >>> index_along_last_axis(a, (indexes + 1) % 3)
    array([[ 0.5,  3.3,  4.4,  7.4],
           [ 5.9,  8.6,  9.7,  6.3],
           [ 0.8,  5.6,  6.7,  7.1],
           [ 4.8,  6.9,  7.1,  1.9]])
    """

    a = np.array(a)
    indexes = np.array(indexes)

    if a.shape[:-1] != indexes.shape:
        raise ValueError(
            f"Array and indexes have incompatible shapes: {a.shape} and {indexes.shape}"
        )

    return np.take_along_axis(a, indexes[..., np.newaxis], axis=-1).squeeze(
        axis=-1
    )
