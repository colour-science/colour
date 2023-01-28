"""
Annotation Type Hints
=====================

Defines the annotation type hints, the module exposes many aliases from
:mod:`typing` and :mod:`numpy.typing` to avoid having to handle multiple
imports.
"""

from __future__ import annotations

import numpy as np
import re
from numpy.typing import ArrayLike, NDArray
from types import ModuleType
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Literal,
    Mapping,
    NewType,
    Optional,
    Protocol,
    Sequence,
    SupportsIndex,
    TYPE_CHECKING,
    TextIO,
    Tuple,
    Type,
    TypeVar,
    TypedDict,
    Union,
    cast,
    overload,
    runtime_checkable,
)
from typing_extensions import Self

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "ModuleType",
    "Any",
    "Callable",
    "Dict",
    "Generator",
    "Iterable",
    "Iterator",
    "List",
    "Literal",
    "Mapping",
    "NewType",
    "Optional",
    "Protocol",
    "Sequence",
    "SupportsIndex",
    "TYPE_CHECKING",
    "TextIO",
    "Tuple",
    "Type",
    "TypeVar",
    "TypedDict",
    "Union",
    "cast",
    "overload",
    "runtime_checkable",
    "Self",
    "ArrayLike",
    "NDArray",
    "RegexFlag",
    "DTypeInt",
    "DTypeFloat",
    "DTypeReal",
    "DTypeComplex",
    "DTypeBoolean",
    "DType",
    "Real",
    "Dataclass",
    "NDArrayInt",
    "NDArrayFloat",
    "NDArrayReal",
    "NDArrayComplex",
    "NDArrayBoolean",
    "NDArrayStr",
    "ProtocolInterpolator",
    "ProtocolExtrapolator",
    "ProtocolLUTSequenceItem",
    "LiteralWarning",
]

RegexFlag = NewType("RegexFlag", re.RegexFlag)

DTypeInt = Union[
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]
DTypeFloat = Union[np.float16, np.float32, np.float64]
DTypeReal = Union[DTypeInt, DTypeFloat]
DTypeComplex = Union[np.csingle, np.cdouble]
DTypeBoolean = np.bool_
DType = Union[DTypeBoolean, DTypeReal, DTypeComplex]

Real = Union[int, float]

# TODO: Revisit to use Protocol.
Dataclass = Any

NDArrayInt = NDArray[DTypeInt]
NDArrayFloat = NDArray[DTypeFloat]
NDArrayReal = NDArray[Union[DTypeInt, DTypeFloat]]
NDArrayComplex = NDArray[DTypeComplex]
NDArrayBoolean = NDArray[DTypeBoolean]
NDArrayStr = NDArray[np.str_]


class ProtocolInterpolator(Protocol):  # noqa: D101
    @property
    def x(self) -> NDArray:  # noqa: D102
        ...

    @x.setter
    def x(self, value: ArrayLike):  # noqa: D102
        ...

    @property
    def y(self) -> NDArray:  # noqa: D102
        ...

    @y.setter
    def y(self, value: ArrayLike):  # noqa: D102
        ...

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D102
        ...  # pragma: no cover

    def __call__(self, x: ArrayLike) -> NDArray:  # noqa: D102
        ...  # pragma: no cover


class ProtocolExtrapolator(Protocol):  # noqa: D101
    @property
    def interpolator(self) -> ProtocolInterpolator:  # noqa: D102
        ...

    @interpolator.setter
    def interpolator(self, value: ProtocolInterpolator):  # noqa: D102
        ...

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D102
        ...  # pragma: no cover

    def __call__(self, x: ArrayLike) -> NDArray:  # noqa: D102
        ...  # pragma: no cover


@runtime_checkable
class ProtocolLUTSequenceItem(Protocol):  # noqa: D101
    def apply(self, RGB: ArrayLike, **kwargs: Any) -> NDArray:  # noqa: D102
        ...  # pragma: no cover


LiteralWarning = Literal[
    "default", "error", "ignore", "always", "module", "once"
]


def arraylike(a: ArrayLike) -> NDArray:
    ...


def number_or_arraylike(a: ArrayLike) -> NDArray:
    ...


a: DTypeFloat = np.float64(1)
b: float = 1
c: float = 1
d: ArrayLike = [c, c]
e: ArrayLike = d
s_a: Sequence[DTypeFloat] = [a, a]
s_b: Sequence[float] = [b, b]
s_c: Sequence[float] = [c, c]

arraylike(a)
arraylike(b)
arraylike(c)
arraylike(d)
arraylike([d, d])
arraylike(e)
arraylike([e, e])
arraylike(s_a)
arraylike(s_b)
arraylike(s_c)

number_or_arraylike(a)
number_or_arraylike(b)
number_or_arraylike(c)
number_or_arraylike(d)
number_or_arraylike([d, d])
number_or_arraylike(e)
number_or_arraylike([e, e])
number_or_arraylike(s_a)
number_or_arraylike(s_b)
number_or_arraylike(s_c)

np.atleast_1d(a)
np.atleast_1d(b)
np.atleast_1d(c)
np.atleast_1d(arraylike(d))
np.atleast_1d(arraylike([d, d]))
np.atleast_1d(arraylike(e))
np.atleast_1d(arraylike([e, e]))
np.atleast_1d(s_a)
np.atleast_1d(s_b)
np.atleast_1d(s_c)

del a, b, c, d, e, s_a, s_b, s_c


# ----------------------------------------------------------------------------#
# ---                API Changes and Deprecation Management                ---#
# ----------------------------------------------------------------------------#
if not TYPE_CHECKING:
    DTypeFloating = DTypeFloat
    DTypeInteger = DTypeInt
    DTypeNumber = DTypeReal
    Boolean = bool
    Floating = float
    Integer = int
    Number = Real
    FloatingOrArrayLike = ArrayLike
    FloatingOrNDArray = NDArrayFloat
