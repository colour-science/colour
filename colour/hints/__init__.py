"""
Annotation Type Hints
=====================

Defines the annotation type hints, the module exposes many aliases from
:mod:`typing` and :mod:`numpy.typing` to avoid having to handle multiple
imports.
"""

from __future__ import annotations

import numpy as np

# TODO: Drop mocking when minimal "Numpy" version is 1.20.x.
try:
    import numpy.typing as npt
except ImportError:  # pragma: no cover
    from unittest import mock

    npt = mock.MagicMock()
import re
from types import ModuleType
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Mapping,
    NewType,
    Optional,
    Union,
    Sequence,
    TextIO,
    Tuple,
    TYPE_CHECKING,
    Type,
    TypeVar,
    cast,
    overload,
)

try:
    from typing import (
        Literal,
        Protocol,
        SupportsIndex,
        TypedDict,
        runtime_checkable,
    )
# TODO: Drop "typing_extensions" when "Google Colab" uses Python >= 3.8.
# Remove exclusion in ".pre-commit-config.yaml" file for "pyupgrade".
except ImportError:  # pragma: no cover
    from typing_extensions import (  # type: ignore[misc]
        Literal,
        Protocol,
        SupportsIndex,
        TypedDict,
        runtime_checkable,
    )

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "Any",
    "Callable",
    "Dict",
    "Generator",
    "Iterable",
    "Iterator",
    "List",
    "Mapping",
    "ModuleType",
    "Optional",
    "Union",
    "Sequence",
    "SupportsIndex",
    "TextIO",
    "Tuple",
    "Type",
    "TypedDict",
    "TypeVar",
    "RegexFlag",
    "DTypeBoolean",
    "DTypeInteger",
    "DTypeFloating",
    "DTypeNumber",
    "DTypeComplex",
    "DType",
    "Integer",
    "Floating",
    "Number",
    "Complex",
    "Boolean",
    "Literal",
    "Dataclass",
    "NestedSequence",
    "ArrayLike",
    "IntegerOrArrayLike",
    "FloatingOrArrayLike",
    "NumberOrArrayLike",
    "ComplexOrArrayLike",
    "BooleanOrArrayLike",
    "ScalarType",
    "StrOrArrayLike",
    "NDArray",
    "IntegerOrNDArray",
    "FloatingOrNDArray",
    "NumberOrNDArray",
    "ComplexOrNDArray",
    "BooleanOrNDArray",
    "StrOrNDArray",
    "TypeInterpolator",
    "TypeExtrapolator",
    "TypeLUTSequenceItem",
    "LiteralWarning",
    "cast",
]

Any = Any
Callable = Callable
Dict = Dict
Generator = Generator
Iterable = Iterable
Iterator = Iterator
List = List
Mapping = Mapping
ModuleType = ModuleType
Optional = Optional
Union = Union
Sequence = Sequence
SupportsIndex = SupportsIndex
TextIO = TextIO
Tuple = Tuple
Type = Type
TypedDict = TypedDict
TypeVar = TypeVar

RegexFlag = NewType("RegexFlag", re.RegexFlag)

DTypeInteger = Union[
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]
DTypeFloating = Union[np.float16, np.float32, np.float64]
DTypeNumber = Union[DTypeInteger, DTypeFloating]
DTypeComplex = Union[np.csingle, np.cdouble]
DTypeBoolean = np.bool_
DType = Union[DTypeBoolean, DTypeNumber, DTypeComplex]

Integer = int
Floating = float
Number = Union[Integer, Floating]
Complex = complex
Boolean = bool

# TODO: Use "typing.Literal" when minimal Python version is raised to 3.8.
Literal = Literal

# TODO: Revisit to use Protocol.
Dataclass = Any

# TODO: Drop mocking when minimal "Numpy" version is 1.21.x.
_T_co = TypeVar("_T_co", covariant=True)


class NestedSequence(Protocol[_T_co]):
    """A protocol for representing nested sequences.

    Warning
    -------
    `NestedSequence` currently does not work in combination with typevars,
    *e.g.* ``def func(a: _NestedSequnce[T]) -> T: ...``.

    See Also
    --------
    collections.abc.Sequence
        ABCs for read-only and mutable :term:`sequences`.

    Examples
    --------
    >>> from __future__ import annotations

    >>> from typing import TYPE_CHECKING
    >>> import numpy as np

    >>> def get_dtype(seq: NestedSequence[float]) -> np.dtype[np.float64]:
    ...     return np.asarray(seq).dtype

    >>> a = get_dtype([1.0])
    >>> b = get_dtype([[1.0]])
    >>> c = get_dtype([[[1.0]]])
    >>> d = get_dtype([[[[1.0]]]])

    >>> if TYPE_CHECKING:
    ...     reveal_locals()
    ...     # note: Revealed local types are:
    ...     # note:     a: numpy.dtype[numpy.floating[numpy._typing._64Bit]]
    ...     # note:     b: numpy.dtype[numpy.floating[numpy._typing._64Bit]]
    ...     # note:     c: numpy.dtype[numpy.floating[numpy._typing._64Bit]]
    ...     # note:     d: numpy.dtype[numpy.floating[numpy._typing._64Bit]]
    ...

    """

    def __len__(self) -> int:
        """Implement ``len(self)``."""
        raise NotImplementedError

    @overload
    def __getitem__(
        self, index: int
    ) -> _T_co | NestedSequence[_T_co]:  # noqa: D105
        ...

    @overload
    def __getitem__(self, index: slice) -> NestedSequence[_T_co]:  # noqa: D105
        ...

    def __getitem__(self, index):
        """Implement ``self[x]``."""
        raise NotImplementedError

    def __contains__(self, x: object) -> bool:
        """Implement ``x in self``."""
        raise NotImplementedError

    def __iter__(self) -> Iterator[_T_co | NestedSequence[_T_co]]:
        """Implement ``iter(self)``."""
        raise NotImplementedError

    def __reversed__(self) -> Iterator[_T_co | NestedSequence[_T_co]]:
        """Implement ``reversed(self)``."""
        raise NotImplementedError

    def count(self, value: Any) -> int:
        """Return the number of occurrences of `value`."""
        raise NotImplementedError

    def index(self, value: Any) -> int:
        """Return the first index of `value`."""
        raise NotImplementedError


ArrayLike = npt.ArrayLike
IntegerOrArrayLike = Union[Integer, ArrayLike]
FloatingOrArrayLike = Union[Floating, ArrayLike]
NumberOrArrayLike = Union[Number, ArrayLike]
ComplexOrArrayLike = Union[Complex, ArrayLike]

BooleanOrArrayLike = Union[Boolean, ArrayLike]

StrOrArrayLike = Union[str, ArrayLike]

ScalarType = TypeVar("ScalarType", bound=np.generic, covariant=True)

# TODO: Use "numpy.typing.NDArray" when minimal Numpy version is raised to
# 1.21.
if TYPE_CHECKING:  # pragma: no cover
    NDArray = np.ndarray[Any, np.dtype[ScalarType]]
else:
    NDArray = np.ndarray

# TODO: Drop when minimal Python is raised to 3.9.
if TYPE_CHECKING:  # pragma: no cover
    IntegerOrNDArray = Union[Integer, NDArray[DTypeInteger]]
    FloatingOrNDArray = Union[Floating, NDArray[DTypeFloating]]
    NumberOrNDArray = Union[
        Number, NDArray[Union[DTypeInteger, DTypeFloating]]
    ]
    ComplexOrNDArray = Union[Complex, NDArray[DTypeComplex]]

    BooleanOrNDArray = Union[Boolean, NDArray[DTypeBoolean]]

    StrOrNDArray = Union[str, NDArray[np.str_]]

else:
    IntegerOrNDArray = Union[Integer, NDArray]
    FloatingOrNDArray = Union[Floating, NDArray]
    NumberOrNDArray = Union[Number, NDArray]
    ComplexOrNDArray = Union[Complex, NDArray]

    BooleanOrNDArray = Union[Boolean, NDArray]

    StrOrNDArray = Union[str, NDArray]


class TypeInterpolator(Protocol):  # noqa: D101
    x: NDArray
    y: NDArray

    def __init__(self, *args: Any, **kwargs: Any):  # noqa: D102
        ...  # pragma: no cover

    def __call__(
        self, x: FloatingOrArrayLike
    ) -> FloatingOrNDArray:  # noqa: D102
        ...  # pragma: no cover


class TypeExtrapolator(Protocol):  # noqa: D101
    interpolator: TypeInterpolator

    def __init__(self, *args: Any, **kwargs: Any):  # noqa: D102
        ...  # pragma: no cover

    def __call__(
        self, x: FloatingOrArrayLike
    ) -> FloatingOrNDArray:  # noqa: D102
        ...  # pragma: no cover


@runtime_checkable
class TypeLUTSequenceItem(Protocol):  # noqa: D101
    def apply(self, RGB: ArrayLike, **kwargs: Any) -> NDArray:  # noqa: D102
        ...  # pragma: no cover


LiteralWarning = Literal[
    "default", "error", "ignore", "always", "module", "once"
]

cast = cast


def arraylike(a: ArrayLike | NestedSequence[ArrayLike]) -> NDArray:
    ...


def number_or_arraylike(
    a: NumberOrArrayLike | NestedSequence[ArrayLike],
) -> NDArray:
    ...


a: DTypeFloating = np.float64(1)
b: float = 1
c: Floating = 1
d: ArrayLike = [c, c]
e: FloatingOrArrayLike = d
s_a: Sequence[DTypeFloating] = [a, a]
s_b: Sequence[float] = [b, b]
s_c: Sequence[Floating] = [c, c]

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
