"""
LUT Processing
==============

Defines the classes and definitions handling *LUT* processing:

-   :class:`colour.LUT1D`
-   :class:`colour.LUT3x1D`
-   :class:`colour.LUT3D`
-   :class:`colour.io.LUT_to_LUT`
-   :class:`colour.io.Range`
-   :class:`colour.io.Matrix`
-   :class:`colour.io.ASC_CDL`
-   :class:`colour.io.Exponent`
-   :class:`colour.io.Log`
"""

from __future__ import annotations

import abc
import re
from abc import ABC, abstractmethod
from copy import deepcopy
from operator import (
    add,
    iadd,
    imul,
    ipow,
    isub,
    itruediv,
    mul,
    pow,
    sub,
    truediv,
)

from colour.models import (
    exponent_function_basic,
    exponent_function_monitor_curve,
    logarithmic_function_basic,
    logarithmic_function_camera,
    logarithmic_function_quasilog,
)

try:
    from collections import MutableSequence
except ImportError:
    from collections.abc import MutableSequence

from functools import partial

import numpy as np
from scipy.spatial import KDTree
from six import add_metaclass

from colour.algebra import (
    Extrapolator,
    LinearInterpolator,
    linear_conversion,
    table_interpolation_trilinear,
    vector_dot,
)
from colour.hints import (
    Any,
    ArrayLike,
    List,
    Literal,
    NDArrayFloat,
    Sequence,
    Type,
    cast,
)
from colour.utilities import (
    as_array,
    as_float_array,
    as_int,
    as_int_array,
    as_int_scalar,
    attest,
    full,
    is_iterable,
    is_numeric,
    is_string,
    multiline_repr,
    multiline_str,
    optional,
    runtime_warning,
    tsplit,
    tstack,
    usage_warning,
    validate_method,
)
from colour.utilities.deprecation import ObjectRenamed, handle_arguments_deprecation

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "AbstractLUT",
    "LUT1D",
    "LUT3x1D",
    "LUT3D",
    "LUT_to_LUT",
    "AbstractLUTSequenceOperator",
    "LUTSequence",
    "Range",
    "Matrix",
    "Exponent",
    "Log",
]


class AbstractLUT(ABC):
    """
    Define the base class for *LUT*.

    This is an :class:`ABCMeta` abstract class that must be inherited by
    sub-classes.

    Parameters
    ----------
    table
        Underlying *LUT* table.
    name
        *LUT* name.
    dimensions
        *LUT* dimensions, typically, 1 for a 1D *LUT*, 2 for a 3x1D *LUT* and 3
        for a 3D *LUT*.
    domain
        *LUT* domain, also used to define the instantiation time default table
        domain.
    size
        *LUT* size, also used to define the instantiation time default table
        size.
    comments
        Comments to add to the *LUT*.

    Attributes
    ----------
    -   :attr:`~colour.io.luts.lut.AbstractLUT.table`
    -   :attr:`~colour.io.luts.lut.AbstractLUT.name`
    -   :attr:`~colour.io.luts.lut.AbstractLUT.dimensions`
    -   :attr:`~colour.io.luts.lut.AbstractLUT.domain`
    -   :attr:`~colour.io.luts.lut.AbstractLUT.size`
    -   :attr:`~colour.io.luts.lut.AbstractLUT.comments`

    Methods
    -------
    -   :meth:`~colour.io.luts.lut.AbstractLUT.__init__`
    -   :meth:`~colour.io.luts.lut.AbstractLUT.__str__`
    -   :meth:`~colour.io.luts.lut.AbstractLUT.__repr__`
    -   :meth:`~colour.io.luts.lut.AbstractLUT.__eq__`
    -   :meth:`~colour.io.luts.lut.AbstractLUT.__ne__`
    -   :meth:`~colour.io.luts.lut.AbstractLUT.__add__`
    -   :meth:`~colour.io.luts.lut.AbstractLUT.__iadd__`
    -   :meth:`~colour.io.luts.lut.AbstractLUT.__sub__`
    -   :meth:`~colour.io.luts.lut.AbstractLUT.__isub__`
    -   :meth:`~colour.io.luts.lut.AbstractLUT.__mul__`
    -   :meth:`~colour.io.luts.lut.AbstractLUT.__imul__`
    -   :meth:`~colour.io.luts.lut.AbstractLUT.__div__`
    -   :meth:`~colour.io.luts.lut.AbstractLUT.__idiv__`
    -   :meth:`~colour.io.luts.lut.AbstractLUT.__pow__`
    -   :meth:`~colour.io.luts.lut.AbstractLUT.__ipow__`
    -   :meth:`~colour.io.luts.lut.AbstractLUT.arithmetical_operation`
    -   :meth:`~colour.io.luts.lut.AbstractLUT.is_domain_explicit`
    -   :meth:`~colour.io.luts.lut.AbstractLUT.linear_table`
    -   :meth:`~colour.io.luts.lut.AbstractLUT.copy`
    -   :meth:`~colour.io.luts.lut.AbstractLUT.invert`
    -   :meth:`~colour.io.luts.lut.AbstractLUT.apply`
    -   :meth:`~colour.io.luts.lut.AbstractLUT.convert`
    """

    def __init__(
        self,
        table: ArrayLike | None = None,
        name: str | None = None,
        dimensions: int | None = None,
        domain: ArrayLike | None = None,
        size: ArrayLike | None = None,
        comments: Sequence | None = None,
    ) -> None:
        self._name: str = f"Unity {size!r}" if table is None else f"{id(self)}"
        self.name = optional(name, self._name)
        self._dimensions = optional(dimensions, 0)
        self._table: NDArrayFloat = self.linear_table(
            optional(size, 0), optional(domain, np.array([]))
        )
        self.table = optional(table, self._table)
        self._domain: NDArrayFloat = np.array([])
        self.domain = optional(domain, self._domain)
        self._comments: list = []
        self.comments = cast(list, optional(comments, self._comments))

    @property
    def table(self) -> NDArrayFloat:
        """
        Getter and setter property for the underlying *LUT* table.

        Parameters
        ----------
        value
            Value to set the underlying *LUT* table with.

        Returns
        -------
        :class:`numpy.ndarray`
            Underlying *LUT* table.
        """

        return self._table

    @table.setter
    def table(self, value: ArrayLike):
        """Setter for the **self.table** property."""

        self._table = self._validate_table(value)

    @property
    def name(self) -> str:
        """
        Getter and setter property for the *LUT* name.

        Parameters
        ----------
        value
            Value to set the *LUT* name with.

        Returns
        -------
        :class:`str`
            *LUT* name.
        """

        return self._name

    @name.setter
    def name(self, value: str):
        """Setter for the **self.name** property."""

        attest(
            is_string(value),
            f'"name" property: "{value}" type is not "str"!',
        )

        self._name = value

    @property
    def domain(self) -> NDArrayFloat:
        """
        Getter and setter property for the *LUT* domain.

        Parameters
        ----------
        value
            Value to set the *LUT* domain with.

        Returns
        -------
        :class:`numpy.ndarray`
            *LUT* domain.
        """

        return self._domain

    @domain.setter
    def domain(self, value: ArrayLike):
        """Setter for the **self.domain** property."""

        self._domain = self._validate_domain(value)

    @property
    def dimensions(self) -> int:
        """
        Getter property for the *LUT* dimensions.

        Returns
        -------
        :class:`int`
            *LUT* dimensions.
        """

        return self._dimensions

    @property
    def size(self) -> int:
        """
        Getter property for the *LUT* size.

        Returns
        -------
        :class:`int`
            *LUT* size.
        """

        return self._table.shape[0]

    @property
    def comments(self) -> list:
        """
        Getter and setter property for the *LUT* comments.

        Parameters
        ----------
        value
            Value to set the *LUT* comments with.

        Returns
        -------
        :class:`list`
            *LUT* comments.
        """

        return self._comments

    @comments.setter
    def comments(self, value: Sequence):
        """Setter for the **self.comments** property."""

        attest(
            is_iterable(value),
            f'"comments" property: "{value}" must be a sequence!',
        )

        self._comments = list(value)

    def __str__(self) -> str:
        """
        Return a formatted string representation of the *LUT*.

        Returns
        -------
        :class:`str`
            Formatted string representation.
        """
        attributes = [
            {
                "formatter": lambda x: (  # noqa: ARG005
                    f"{self.__class__.__name__} - {self.name}"
                ),
                "section": True,
            },
            {"line_break": True},
            {"name": "dimensions", "label": "Dimensions"},
            {"name": "domain", "label": "Domain"},
            {
                "label": "Size",
                "formatter": lambda x: str(self.table.shape),  # noqa: ARG005
            },
        ]

        if self.comments:
            attributes.append(
                {
                    "formatter": lambda x: "\n".join(  # noqa: ARG005
                        [
                            f"Comment {str(i + 1).zfill(2)} : {comment}"
                            for i, comment in enumerate(self.comments)
                        ]
                    ),
                }
            )

        return multiline_str(self, cast(List[dict], attributes))

    def __repr__(self) -> str:
        """
        Return an evaluable string representation of the *LUT*.

        Returns
        -------
        :class:`str`
            Evaluable string representation.
        """

        attributes = [
            {"name": "table"},
            {"name": "name"},
            {"name": "domain"},
            {"name": "size"},
        ]

        if self.comments:
            attributes.append({"name": "comments"})

        return multiline_repr(self, attributes)

    def __eq__(self, other: Any) -> bool:
        """
        Return whether the *LUT* is equal to given other object.

        Parameters
        ----------
        other
            Object to test whether it is equal to the *LUT*.

        Returns
        -------
        :class:`bool`
            Whether given object is equal to the *LUT*.
        """

        if isinstance(other, AbstractLUT) and all(
            [
                np.array_equal(self.table, other.table),
                np.array_equal(self.domain, other.domain),
            ]
        ):
            return True

        return False

    def __ne__(self, other: Any) -> bool:
        """
        Return whether the *LUT* is not equal to given other object.

        Parameters
        ----------
        other
            Object to test whether it is not equal to the *LUT*.

        Returns
        -------
        :class:`bool`
            Whether given object is not equal to the *LUT*.
        """

        return not (self == other)

    def __add__(self, a: ArrayLike | AbstractLUT) -> AbstractLUT:
        """
        Implement support for addition.

        Parameters
        ----------
        a
            :math:`a` variable to add.

        Returns
        -------
        :class:`colour.io.luts.lut.AbstractLUT`
            Variable added *LUT*.
        """

        return self.arithmetical_operation(a, "+")

    def __iadd__(self, a: ArrayLike | AbstractLUT) -> AbstractLUT:
        """
        Implement support for in-place addition.

        Parameters
        ----------
        a
            :math:`a` variable to add in-place.

        Returns
        -------
        :class:`colour.io.luts.lut.AbstractLUT`
            In-place variable added *LUT*.
        """

        return self.arithmetical_operation(a, "+", True)

    def __sub__(self, a: ArrayLike | AbstractLUT) -> AbstractLUT:
        """
        Implement support for subtraction.

        Parameters
        ----------
        a
            :math:`a` variable to subtract.

        Returns
        -------
        :class:`colour.io.luts.lut.AbstractLUT`
            Variable subtracted *LUT*.
        """

        return self.arithmetical_operation(a, "-")

    def __isub__(self, a: ArrayLike | AbstractLUT) -> AbstractLUT:
        """
        Implement support for in-place subtraction.

        Parameters
        ----------
        a
            :math:`a` variable to subtract in-place.

        Returns
        -------
        :class:`colour.io.luts.lut.AbstractLUT`
            In-place variable subtracted *LUT*.
        """

        return self.arithmetical_operation(a, "-", True)

    def __mul__(self, a: ArrayLike | AbstractLUT) -> AbstractLUT:
        """
        Implement support for multiplication.

        Parameters
        ----------
        a
            :math:`a` variable to multiply by.

        Returns
        -------
        :class:`colour.io.luts.lut.AbstractLUT`
            Variable multiplied *LUT*.
        """

        return self.arithmetical_operation(a, "*")

    def __imul__(self, a: ArrayLike | AbstractLUT) -> AbstractLUT:
        """
        Implement support for in-place multiplication.

        Parameters
        ----------
        a
            :math:`a` variable to multiply by in-place.

        Returns
        -------
        :class:`colour.io.luts.lut.AbstractLUT`
            In-place variable multiplied *LUT*.
        """

        return self.arithmetical_operation(a, "*", True)

    def __div__(self, a: ArrayLike | AbstractLUT) -> AbstractLUT:
        """
        Implement support for division.

        Parameters
        ----------
        a
            :math:`a` variable to divide by.

        Returns
        -------
        :class:`colour.io.luts.lut.AbstractLUT`
            Variable divided *LUT*.
        """

        return self.arithmetical_operation(a, "/")

    def __idiv__(self, a: ArrayLike | AbstractLUT) -> AbstractLUT:
        """
        Implement support for in-place division.

        Parameters
        ----------
        a
            :math:`a` variable to divide by in-place.

        Returns
        -------
        :class:`colour.io.luts.lut.AbstractLUT`
            In-place variable divided *LUT*.
        """

        return self.arithmetical_operation(a, "/", True)

    __itruediv__ = __idiv__
    __truediv__ = __div__

    def __pow__(self, a: ArrayLike | AbstractLUT) -> AbstractLUT:
        """
        Implement support for exponentiation.

        Parameters
        ----------
        a
            :math:`a` variable to exponentiate by.

        Returns
        -------
        :class:`colour.io.luts.lut.AbstractLUT`
            Variable exponentiated *LUT*.
        """

        return self.arithmetical_operation(a, "**")

    def __ipow__(self, a: ArrayLike | AbstractLUT) -> AbstractLUT:
        """
        Implement support for in-place exponentiation.

        Parameters
        ----------
        a
            :math:`a` variable to exponentiate by in-place.

        Returns
        -------
        :class:`colour.io.luts.lut.AbstractLUT`
            In-place variable exponentiated *LUT*.
        """

        return self.arithmetical_operation(a, "**", True)

    def arithmetical_operation(
        self,
        a: ArrayLike | AbstractLUT,
        operation: Literal["+", "-", "*", "/", "**"],
        in_place: bool = False,
    ) -> AbstractLUT:
        """
        Perform given arithmetical operation with :math:`a` operand, the
        operation can be either performed on a copy or in-place, must be
        reimplemented by sub-classes.

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
        :class:`colour.io.luts.lut.AbstractLUT`
            *LUT*.
        """

        operator, ioperator = {
            "+": (add, iadd),
            "-": (sub, isub),
            "*": (mul, imul),
            "/": (truediv, itruediv),
            "**": (pow, ipow),
        }[operation]

        if in_place:
            operand = a.table if isinstance(a, AbstractLUT) else as_float_array(a)

            self.table = operator(self.table, operand)

            return self
        else:
            copy = ioperator(self.copy(), a)

            return copy

    @abstractmethod
    def _validate_table(self, table: ArrayLike) -> NDArrayFloat:
        """
        Validate given table according to *LUT* dimensions.

        Parameters
        ----------
        table
            Table to validate.

        Returns
        -------
        :class:`numpy.ndarray`
            Validated table as a :class:`ndarray` instance.
        """

    @abstractmethod
    def _validate_domain(self, domain: ArrayLike) -> NDArrayFloat:
        """
        Validate given domain according to *LUT* dimensions.

        Parameters
        ----------
        domain
            Domain to validate.

        Returns
        -------
        :class:`numpy.ndarray`
            Validated domain as a :class:`ndarray` instance.
        """

    @abstractmethod
    def is_domain_explicit(self) -> bool:
        """
        Return whether the *LUT* domain is explicit (or implicit).

        An implicit domain is defined by its shape only::

            [[0 1]
             [0 1]
             [0 1]]

        While an explicit domain defines every single discrete samples::

            [[0.0 0.0 0.0]
             [0.1 0.1 0.1]
             [0.2 0.2 0.2]
             [0.3 0.3 0.3]
             [0.4 0.4 0.4]
             [0.8 0.8 0.8]
             [1.0 1.0 1.0]]

        Returns
        -------
        :class:`bool`
            Is *LUT* domain explicit.
        """

    @staticmethod
    @abstractmethod
    def linear_table(
        size: ArrayLike | None = None,
        domain: ArrayLike | None = None,
    ) -> NDArrayFloat:
        """
        Return a linear table of given size according to *LUT* dimensions.

        Parameters
        ----------
        size
            Expected table size, for a 1D *LUT*, the number of output samples
            :math:`n` is equal to ``size``, for a 3x1D *LUT* :math:`n` is equal
            to ``size * 3`` or ``size[0] + size[1] + size[2]``, for a 3D *LUT*
            :math:`n` is equal to ``size**3 * 3`` or
            ``size[0] * size[1] * size[2] * 3``.
        domain
            Domain of the table.

        Returns
        -------
        :class:`numpy.ndarray`
            Linear table.
        """

    def copy(self) -> AbstractLUT:
        """
        Return a copy of the sub-class instance.

        Returns
        -------
        :class:`colour.io.luts.lut.AbstractLUT`
            *LUT* copy.
        """

        return deepcopy(self)

    @abstractmethod
    def invert(self, **kwargs: Any) -> AbstractLUT:
        """
        Compute and returns an inverse copy of the *LUT*.

        Other Parameters
        ----------------
        kwargs
            Keywords arguments.

        Returns
        -------
        :class:`colour.io.luts.lut.AbstractLUT`
            Inverse *LUT* class instance.
        """

    @abstractmethod
    def apply(self, RGB: ArrayLike, **kwargs: Any) -> NDArrayFloat:
        """
        Apply the *LUT* to given *RGB* colourspace array using given method.

        Parameters
        ----------
        RGB
            *RGB* colourspace array to apply the *LUT* onto.

        Other Parameters
        ----------------
        direction
            Whether the *LUT* should be applied in the forward or inverse
            direction.
        extrapolator
            Extrapolator class type or object to use as extrapolating function.
        extrapolator_kwargs
            Arguments to use when instantiating or calling the extrapolating
            function.
        interpolator
            Interpolator class type or object to use as interpolating function.
        interpolator_args : dict_like, optional
            Arguments to use when instantiating or calling the interpolating
            function.

        Returns
        -------
        :class:`numpy.ndarray`
            Interpolated *RGB* colourspace array.
        """

    def convert(
        self,
        cls: Type[AbstractLUT],
        force_conversion: bool = False,
        **kwargs: Any,
    ) -> AbstractLUT:
        """
        Convert the *LUT* to given ``cls`` class instance.

        Parameters
        ----------
        cls
            *LUT* class instance.
        force_conversion
            Whether to force the conversion as it might be destructive.

        Other Parameters
        ----------------
        interpolator
            Interpolator class type to use as interpolating function.
        interpolator_kwargs
            Arguments to use when instantiating the interpolating function.
        size
            Expected table size in case of an upcast to or a downcast from a
            :class:`LUT3D` class instance.

        Returns
        -------
        :class:`colour.io.luts.lut.AbstractLUT`
            Converted *LUT* class instance.

        Warnings
        --------
        Some conversions are destructive and raise a :class:`ValueError`
        exception by default.

        Raises
        ------
        ValueError
            If the conversion is destructive.
        """

        return LUT_to_LUT(
            self,
            cls,
            force_conversion,
            **kwargs,  # pyright: ignore
        )


class LUT1D(AbstractLUT):
    """
    Define the base class for a 1D *LUT*.

    Parameters
    ----------
    table
        Underlying *LUT* table.
    name
        *LUT* name.
    domain
        *LUT* domain, also used to define the instantiation time default table
        domain.
    size
        Size of the instantiation time default table, default to 10.
    comments
        Comments to add to the *LUT*.

    Methods
    -------
    -   :meth:`~colour.LUT1D.__init__`
    -   :meth:`~colour.LUT1D.is_domain_explicit`
    -   :meth:`~colour.LUT1D.linear_table`
    -   :meth:`~colour.LUT1D.invert`
    -   :meth:`~colour.LUT1D.apply`

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

    >>> from colour.algebra import spow
    >>> domain = np.array([-0.1, 1.5])
    >>> print(
    ...     LUT1D(
    ...         spow(LUT1D.linear_table(16, domain), 1 / 2.2),
    ...         "My LUT",
    ...         domain,
    ...         comments=["A first comment.", "A second comment."],
    ...     )
    ... )
    LUT1D - My LUT
    --------------
    <BLANKLINE>
    Dimensions : 1
    Domain     : [-0.1  1.5]
    Size       : (16,)
    Comment 01 : A first comment.
    Comment 02 : A second comment.
    """

    def __init__(
        self,
        table: ArrayLike | None = None,
        name: str | None = None,
        domain: ArrayLike | None = None,
        size: ArrayLike | None = None,
        comments: Sequence | None = None,
    ) -> None:
        domain = as_float_array(optional(domain, np.array([0, 1])))
        size = optional(size, 10)

        super().__init__(table, name, 1, domain, size, comments)

    def _validate_table(self, table: ArrayLike) -> NDArrayFloat:
        """
        Validate given table is a 1D array.

        Parameters
        ----------
        table
            Table to validate.

        Returns
        -------
        :class:`numpy.ndarray`
            Validated table as a :class:`ndarray` instance.
        """

        table = as_float_array(table)

        attest(len(table.shape) == 1, "The table must be a 1D array!")

        return table

    def _validate_domain(self, domain: ArrayLike) -> NDArrayFloat:
        """
        Validate given domain.

        Parameters
        ----------
        domain
            Domain to validate.

        Returns
        -------
        :class:`numpy.ndarray`
            Validated domain as a :class:`ndarray` instance.
        """

        domain = as_float_array(domain)

        attest(len(domain.shape) == 1, "The domain must be a 1D array!")

        attest(
            domain.shape[0] >= 2,
            "The domain column count must be equal or greater than 2!",
        )

        return domain

    def is_domain_explicit(self) -> bool:
        """
        Return whether the *LUT* domain is explicit (or implicit).

        An implicit domain is defined by its shape only::

            [0 1]

        While an explicit domain defines every single discrete samples::

            [0.0 0.1 0.2 0.4 0.8 1.0]

        Returns
        -------
        :class:`bool`
            Is *LUT* domain explicit.

        Examples
        --------
        >>> LUT1D().is_domain_explicit()
        False
        >>> table = domain = np.linspace(0, 1, 10)
        >>> LUT1D(table, domain=domain).is_domain_explicit()
        True
        """

        return len(self.domain) != 2

    @staticmethod
    def linear_table(
        size: ArrayLike | None = None,
        domain: ArrayLike | None = None,
    ) -> NDArrayFloat:
        """
        Return a linear table, the number of output samples :math:`n` is equal
        to ``size``.

        Parameters
        ----------
        size
            Expected table size, default to 10.
        domain
            Domain of the table.

        Returns
        -------
        :class:`numpy.ndarray`
            Linear table with ``size`` samples.

        Examples
        --------
        >>> LUT1D.linear_table(5, np.array([-0.1, 1.5]))
        array([-0.1,  0.3,  0.7,  1.1,  1.5])
        >>> LUT1D.linear_table(domain=np.linspace(-0.1, 1.5, 5))
        array([-0.1,  0.3,  0.7,  1.1,  1.5])
        """

        size = optional(size, 10)
        domain = as_float_array(optional(domain, np.array([0, 1])))

        if len(domain) != 2:
            return domain
        else:
            attest(is_numeric(size),
                   f"Linear table size must be a numeric but is {size} instead!")

            return np.linspace(domain[0], domain[1], as_int_scalar(size))

    def invert(self, **kwargs: Any) -> LUT1D:  # noqa: ARG002
        """
        Compute and returns an inverse copy of the *LUT*.

        Other Parameters
        ----------------
        kwargs
            Keywords arguments, only given for signature compatibility with
            the :meth:`AbstractLUT.invert` method.

        Other Parameters
        ----------------
        \\**kwargs : dict, optional
            Keywords arguments for deprecation management.

        Returns
        -------
        :class:`colour.LUT1D`
            Inverse *LUT* class instance.

        Examples
        --------
        >>> LUT = LUT1D(LUT1D.linear_table() ** (1 / 2.2))
        >>> print(LUT.table)  # doctest: +ELLIPSIS
        [ 0.       ...  0.3683438...  0.5047603...  0.6069133...  \
0.6916988...  0.7655385...
          0.8316843...  0.8920493...  0.9478701...  1.        ]
        >>> print(LUT.invert())  # doctest: +ELLIPSIS
        LUT1D - ... - Inverse
        --------...----------
        <BLANKLINE>
        Dimensions : 1
        Domain     : [ 0.          0.3683438...  0.5047603...  0.6069133...  \
0.6916988...  0.7655385...
                       0.8316843...  0.8920493...  0.9478701...  1.        ]
        Size       : (10,)
        >>> print(LUT.invert().table)  # doctest: +ELLIPSIS
        [ 0.       ...  0.1111111...  0.2222222...  0.3333333...  \
0.4444444...  0.5555555...
          0.6666666...  0.7777777...  0.8888888...  1.        ]
        """

        if self.is_domain_explicit():
            domain = self.domain
        else:
            domain_min, domain_max = self.domain
            domain = np.linspace(domain_min, domain_max, self.size)

        LUT_i = LUT1D(
            table=domain,
            name=f"{self.name} - Inverse",
            domain=self.table,
        )

        return LUT_i

    def apply(self, RGB: ArrayLike, **kwargs: Any) -> NDArrayFloat:
        """
        Apply the *LUT* to given *RGB* colourspace array using given method.

        Parameters
        ----------
        RGB
            *RGB* colourspace array to apply the *LUT* onto.

        Other Parameters
        ----------------
        direction
            Whether the *LUT* should be applied in the forward or inverse
            direction.
        extrapolator
            Extrapolator class type or object to use as extrapolating function.
        extrapolator_kwargs
            Arguments to use when instantiating or calling the extrapolating
            function.
        interpolator
            Interpolator class type to use as interpolating function.
        interpolator_kwargs
            Arguments to use when instantiating the interpolating function.

        Returns
        -------
        :class:`numpy.ndarray`
            Interpolated *RGB* colourspace array.

        Examples
        --------
        >>> LUT = LUT1D(LUT1D.linear_table() ** (1 / 2.2))
        >>> RGB = np.array([0.18, 0.18, 0.18])

        *LUT* applied to the given *RGB* colourspace in the forward direction:

        >>> LUT.apply(RGB)  # doctest: +ELLIPSIS
        array([ 0.4529220...,  0.4529220...,  0.4529220...])

        *LUT* applied to the modified *RGB* colourspace in the inverse
        direction:

        >>> LUT.apply(LUT.apply(RGB), direction="Inverse")
        ... # doctest: +ELLIPSIS
        array([ 0.18...,  0.18...,  0.18...])
        """

        direction = validate_method(
            kwargs.get("direction", "Forward"), ("Forward", "Inverse")
        )

        interpolator = kwargs.get("interpolator", LinearInterpolator)
        interpolator_kwargs = kwargs.get("interpolator_kwargs", {})
        extrapolator = kwargs.get("extrapolator", Extrapolator)
        extrapolator_kwargs = kwargs.get("extrapolator_kwargs", {})

        LUT = self.invert() if direction == "inverse" else self

        if LUT.is_domain_explicit():
            samples = LUT.domain
        else:
            domain_min, domain_max = LUT.domain
            samples = np.linspace(domain_min, domain_max, LUT.size)

        RGB_interpolator = extrapolator(
            interpolator(samples, LUT.table, **interpolator_kwargs),
            **extrapolator_kwargs,
        )

        return RGB_interpolator(RGB)

    # ------------------------------------------------------------------------#
    # ---              API Changes and Deprecation Management              ---#
    # ------------------------------------------------------------------------#
    def as_LUT(  # noqa: D102
        self,
        cls: Type[AbstractLUT],
        force_conversion: bool = False,
        **kwargs: Any,
    ) -> AbstractLUT:  # pragma: no cover
        # Docstrings are omitted for documentation purposes.
        usage_warning(
            str(
                ObjectRenamed(
                    "LUT1D.as_LUT",
                    "LUT1D.convert",
                )
            )
        )

        return self.convert(cls, force_conversion, **kwargs)  # pyright: ignore


class LUT3x1D(AbstractLUT):
    """
    Define the base class for a 3x1D *LUT*.

    Parameters
    ----------
    table
        Underlying *LUT* table.
    name
        *LUT* name.
    domain
        *LUT* domain, also used to define the instantiation time default table
        domain.
    size
        Size of the instantiation time default table, default to 10.
    comments
        Comments to add to the *LUT*.

    Methods
    -------
    -   :meth:`~colour.LUT3x1D.__init__`
    -   :meth:`~colour.LUT3x1D.is_domain_explicit`
    -   :meth:`~colour.LUT3x1D.linear_table`
    -   :meth:`~colour.LUT3x1D.invert`
    -   :meth:`~colour.LUT3x1D.apply`

    Examples
    --------
    Instantiating a unity LUT with a table with 16x3 elements:

    >>> print(LUT3x1D(size=16))
    LUT3x1D - Unity 16
    ------------------
    <BLANKLINE>
    Dimensions : 2
    Domain     : [[ 0.  0.  0.]
                  [ 1.  1.  1.]]
    Size       : (16, 3)

    Instantiating a LUT using a custom table with 16x3 elements:

    >>> print(LUT3x1D(LUT3x1D.linear_table(16) ** (1 / 2.2)))
    ... # doctest: +ELLIPSIS
    LUT3x1D - ...
    ----------...
    <BLANKLINE>
    Dimensions : 2
    Domain     : [[ 0.  0.  0.]
                  [ 1.  1.  1.]]
    Size       : (16, 3)

    Instantiating a LUT using a custom table with 16x3 elements, custom name,
    custom domain and comments:

    >>> from colour.algebra import spow
    >>> domain = np.array([[-0.1, -0.2, -0.4], [1.5, 3.0, 6.0]])
    >>> print(
    ...     LUT3x1D(
    ...         spow(LUT3x1D.linear_table(16), 1 / 2.2),
    ...         "My LUT",
    ...         domain,
    ...         comments=["A first comment.", "A second comment."],
    ...     )
    ... )
    LUT3x1D - My LUT
    ----------------
    <BLANKLINE>
    Dimensions : 2
    Domain     : [[-0.1 -0.2 -0.4]
                  [ 1.5  3.   6. ]]
    Size       : (16, 3)
    Comment 01 : A first comment.
    Comment 02 : A second comment.
    """

    def __init__(
        self,
        table: ArrayLike | None = None,
        name: str | None = None,
        domain: ArrayLike | None = None,
        size: ArrayLike | None = None,
        comments: Sequence | None = None,
    ) -> None:
        domain = as_float_array(optional(domain, [[0, 0, 0], [1, 1, 1]]))
        size = optional(size, 10)

        super().__init__(table, name, 2, domain, size, comments)

    def _validate_table(self, table: ArrayLike) -> NDArrayFloat:
        """
        Validate given table is a 3x1D array.

        Parameters
        ----------
        table
            Table to validate.

        Returns
        -------
        :class:`numpy.ndarray`
            Validated table as a :class:`ndarray` instance.
        """

        table = as_float_array(table)

        attest(len(table.shape) == 2, "The table must be a 2D array!")

        return table

    def _validate_domain(self, domain: ArrayLike) -> NDArrayFloat:
        """
        Validate given domain.

        Parameters
        ----------
        domain
            Domain to validate.

        Returns
        -------
        :class:`numpy.ndarray`
            Validated domain as a :class:`ndarray` instance.
        """

        domain = as_float_array(domain)

        attest(len(domain.shape) == 2, "The domain must be a 2D array!")

        attest(
            domain.shape[0] >= 2,
            "The domain row count must be equal or greater than 2!",
        )

        attest(domain.shape[1] == 3, "The domain column count must be equal to 3!")

        return domain

    def is_domain_explicit(self) -> bool:
        """
        Return whether the *LUT* domain is explicit (or implicit).

        An implicit domain is defined by its shape only::

            [[0 1]
             [0 1]
             [0 1]]

        While an explicit domain defines every single discrete samples::

            [[0.0 0.0 0.0]
             [0.1 0.1 0.1]
             [0.2 0.2 0.2]
             [0.3 0.3 0.3]
             [0.4 0.4 0.4]
             [0.8 0.8 0.8]
             [1.0 1.0 1.0]]

        Returns
        -------
        :class:`bool`
            Is *LUT* domain explicit.

        Examples
        --------
        >>> LUT3x1D().is_domain_explicit()
        False
        >>> samples = np.linspace(0, 1, 10)
        >>> table = domain = tstack([samples, samples, samples])
        >>> LUT3x1D(table, domain=domain).is_domain_explicit()
        True
        """

        return self.domain.shape != (2, 3)

    @staticmethod
    def linear_table(
        size: ArrayLike | None = None,
        domain: ArrayLike | None = None,
    ) -> NDArrayFloat:
        """
        Return a linear table, the number of output samples :math:`n` is equal
        to ``size * 3`` or ``size[0] + size[1] + size[2]``.

        Parameters
        ----------
        size
            Expected table size, default to 10.
        domain
            Domain of the table.

        Returns
        -------
        :class:`numpy.ndarray`
            Linear table with ``size * 3`` or ``size[0] + size[1] + size[2]``
            samples.

        Warnings
        --------
        If ``size`` is non uniform, the linear table will be padded
        accordingly.

        Examples
        --------
        >>> LUT3x1D.linear_table(5, np.array([[-0.1, -0.2, -0.4], [1.5, 3.0, 6.0]]))
        array([[-0.1, -0.2, -0.4],
               [ 0.3,  0.6,  1.2],
               [ 0.7,  1.4,  2.8],
               [ 1.1,  2.2,  4.4],
               [ 1.5,  3. ,  6. ]])
        >>> LUT3x1D.linear_table(
        ...     np.array([5, 3, 2]),
        ...     np.array([[-0.1, -0.2, -0.4], [1.5, 3.0, 6.0]]),
        ... )
        array([[-0.1, -0.2, -0.4],
               [ 0.3,  1.4,  6. ],
               [ 0.7,  3. ,  nan],
               [ 1.1,  nan,  nan],
               [ 1.5,  nan,  nan]])
        >>> domain = np.array(
        ...     [
        ...         [-0.1, -0.2, -0.4],
        ...         [0.3, 1.4, 6.0],
        ...         [0.7, 3.0, np.nan],
        ...         [1.1, np.nan, np.nan],
        ...         [1.5, np.nan, np.nan],
        ...     ]
        ... )
        >>> LUT3x1D.linear_table(domain=domain)
        array([[-0.1, -0.2, -0.4],
               [ 0.3,  1.4,  6. ],
               [ 0.7,  3. ,  nan],
               [ 1.1,  nan,  nan],
               [ 1.5,  nan,  nan]])
        """

        size = optional(size, 10)
        domain = as_float_array(optional(domain, [[0, 0, 0], [1, 1, 1]]))

        if domain.shape != (2, 3):
            return domain
        else:
            size_array = np.tile(size, 3) if is_numeric(size) else as_int_array(size)

            R, G, B = tsplit(domain)

            samples = [
                np.linspace(a[0], a[1], size_array[i]) for i, a in enumerate([R, G, B])
            ]

            if len(np.unique(size_array)) != 1:
                runtime_warning(
                    "Table is non uniform, axis will be "
                    'padded with "NaNs" accordingly!'
                )

                samples = [
                    np.pad(
                        axis,
                        (0, np.max(size_array) - len(axis)),  # pyright: ignore
                        mode="constant",
                        constant_values=np.nan,
                    )
                    for axis in samples
                ]

            return tstack(samples)

    def invert(self, **kwargs: Any) -> LUT3x1D:  # noqa: ARG002
        """
        Compute and returns an inverse copy of the *LUT*.

        Other Parameters
        ----------------
        kwargs
            Keywords arguments, only given for signature compatibility with
            the :meth:`AbstractLUT.invert` method.

        Returns
        -------
        :class:`colour.LUT3x1D`
            Inverse *LUT* class instance.

        Examples
        --------
        >>> LUT = LUT3x1D(LUT3x1D.linear_table() ** (1 / 2.2))
        >>> print(LUT.table)
        [[ 0.          0.          0.        ]
         [ 0.36834383  0.36834383  0.36834383]
         [ 0.50476034  0.50476034  0.50476034]
         [ 0.60691337  0.60691337  0.60691337]
         [ 0.69169882  0.69169882  0.69169882]
         [ 0.76553851  0.76553851  0.76553851]
         [ 0.83168433  0.83168433  0.83168433]
         [ 0.89204934  0.89204934  0.89204934]
         [ 0.94787016  0.94787016  0.94787016]
         [ 1.          1.          1.        ]]
        >>> print(LUT.invert())  # doctest: +ELLIPSIS
        LUT3x1D - ... - Inverse
        ----------...----------
        <BLANKLINE>
        Dimensions : 2
        Domain     : [[ 0.       ...  0.       ...  0.       ...]
                      [ 0.3683438...  0.3683438...  0.3683438...]
                      [ 0.5047603...  0.5047603...  0.5047603...]
                      [ 0.6069133...  0.6069133...  0.6069133...]
                      [ 0.6916988...  0.6916988...  0.6916988...]
                      [ 0.7655385...  0.7655385...  0.7655385...]
                      [ 0.8316843...  0.8316843...  0.8316843...]
                      [ 0.8920493...  0.8920493...  0.8920493...]
                      [ 0.9478701...  0.9478701...  0.9478701...]
                      [ 1.       ...  1.       ...  1.       ...]]
        Size       : (10, 3)
        >>> print(LUT.invert().table)  # doctest: +ELLIPSIS
        [[ 0.       ...  0.       ...  0.       ...]
         [ 0.1111111...  0.1111111...  0.1111111...]
         [ 0.2222222...  0.2222222...  0.2222222...]
         [ 0.3333333...  0.3333333...  0.3333333...]
         [ 0.4444444...  0.4444444...  0.4444444...]
         [ 0.5555555...  0.5555555...  0.5555555...]
         [ 0.6666666...  0.6666666...  0.6666666...]
         [ 0.7777777...  0.7777777...  0.7777777...]
         [ 0.8888888...  0.8888888...  0.8888888...]
         [ 1.       ...  1.       ...  1.       ...]]
        """

        size = self.table.size // 3
        if self.is_domain_explicit():
            domain = [
                axes[: (~np.isnan(axes)).cumsum().argmax() + 1]
                for axes in np.transpose(self.domain)
            ]
        else:
            domain_min, domain_max = self.domain
            domain = [np.linspace(domain_min[i], domain_max[i], size) for i in range(3)]

        LUT_i = LUT3x1D(
            table=tstack(domain),
            name=f"{self.name} - Inverse",
            domain=self.table,
        )

        return LUT_i

    def apply(self, RGB: ArrayLike, **kwargs: Any) -> NDArrayFloat:
        """
        Apply the *LUT* to given *RGB* colourspace array using given method.

        Parameters
        ----------
        RGB
            *RGB* colourspace array to apply the *LUT* onto.

        Other Parameters
        ----------------
        direction
            Whether the *LUT* should be applied in the forward or inverse
            direction.
        extrapolator
            Extrapolator class type or object to use as extrapolating function.
        extrapolator_kwargs
            Arguments to use when instantiating or calling the extrapolating
            function.
        interpolator
            Interpolator class type to use as interpolating function.
        interpolator_kwargs
            Arguments to use when instantiating the interpolating function.

        Other Parameters
        ----------------
        \\**kwargs : dict, optional
            Keywords arguments for deprecation management.

        Returns
        -------
        :class:`numpy.ndarray`
            Interpolated *RGB* colourspace array.

        Examples
        --------
        >>> LUT = LUT3x1D(LUT3x1D.linear_table() ** (1 / 2.2))
        >>> RGB = np.array([0.18, 0.18, 0.18])
        >>> LUT.apply(RGB)  # doctest: +ELLIPSIS
        array([ 0.4529220...,  0.4529220...,  0.4529220...])
        >>> LUT.apply(LUT.apply(RGB), direction="Inverse")
        ... # doctest: +ELLIPSIS
        array([ 0.18...,  0.18...,  0.18...])
        >>> from colour.algebra import spow
        >>> domain = np.array([[-0.1, -0.2, -0.4], [1.5, 3.0, 6.0]])
        >>> table = spow(LUT3x1D.linear_table(domain=domain), 1 / 2.2)
        >>> LUT = LUT3x1D(table, domain=domain)
        >>> RGB = np.array([0.18, 0.18, 0.18])
        >>> LUT.apply(RGB)  # doctest: +ELLIPSIS
        array([ 0.4423903...,  0.4503801...,  0.3581625...])
        >>> domain = np.array(
        ...     [
        ...         [-0.1, -0.2, -0.4],
        ...         [0.3, 1.4, 6.0],
        ...         [0.7, 3.0, np.nan],
        ...         [1.1, np.nan, np.nan],
        ...         [1.5, np.nan, np.nan],
        ...     ]
        ... )
        >>> table = spow(LUT3x1D.linear_table(domain=domain), 1 / 2.2)
        >>> LUT = LUT3x1D(table, domain=domain)
        >>> RGB = np.array([0.18, 0.18, 0.18])
        >>> LUT.apply(RGB)  # doctest: +ELLIPSIS
        array([ 0.2996370..., -0.0901332..., -0.3949770...])
        """

        direction = validate_method(
            kwargs.get("direction", "Forward"), ("Forward", "Inverse")
        )

        interpolator = kwargs.get("interpolator", LinearInterpolator)
        interpolator_kwargs = kwargs.get("interpolator_kwargs", {})
        extrapolator = kwargs.get("extrapolator", Extrapolator)
        extrapolator_kwargs = kwargs.get("extrapolator_kwargs", {})

        R, G, B = tsplit(RGB)

        LUT = self.invert() if direction == "inverse" else self

        size = LUT.table.size // 3
        if LUT.is_domain_explicit():
            samples = [
                axes[: (~np.isnan(axes)).cumsum().argmax() + 1]
                for axes in np.transpose(LUT.domain)
            ]
            R_t, G_t, B_t = (
                axes[: len(samples[i])]
                for i, axes in enumerate(np.transpose(LUT.table))
            )
        else:
            domain_min, domain_max = LUT.domain
            samples = [
                np.linspace(domain_min[i], domain_max[i], size) for i in range(3)
            ]
            R_t, G_t, B_t = tsplit(LUT.table)

        s_R, s_G, s_B = samples

        RGB_i = [
            extrapolator(
                interpolator(a[0], a[1], **interpolator_kwargs),
                **extrapolator_kwargs,
            )(a[2])
            for a in zip((s_R, s_G, s_B), (R_t, G_t, B_t), (R, G, B))
        ]

        return tstack(RGB_i)

    # ------------------------------------------------------------------------#
    # ---              API Changes and Deprecation Management              ---#
    # ------------------------------------------------------------------------#
    def as_LUT(  # noqa: D102
        self,
        cls: Type[AbstractLUT],
        force_conversion: bool = False,
        **kwargs: Any,
    ) -> AbstractLUT:  # pragma: no cover
        # Docstrings are omitted for documentation purposes.
        usage_warning(
            str(
                ObjectRenamed(
                    "LUT3x1D.as_LUT",
                    "LUT3x1D.convert",
                )
            )
        )

        return self.convert(cls, force_conversion, **kwargs)  # pyright: ignore


class LUT3D(AbstractLUT):
    """
    Define the base class for a 3D *LUT*.

    Parameters
    ----------
    table
        Underlying *LUT* table.
    name
        *LUT* name.
    domain
        *LUT* domain, also used to define the instantiation time default table
        domain.
    size
        Size of the instantiation time default table, default to 33.
    comments
        Comments to add to the *LUT*.

    Methods
    -------
    -   :meth:`~colour.LUT3D.__init__`
    -   :meth:`~colour.LUT3D.is_domain_explicit`
    -   :meth:`~colour.LUT3D.linear_table`
    -   :meth:`~colour.LUT3D.invert`
    -   :meth:`~colour.LUT3D.apply`

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

    >>> from colour.algebra import spow
    >>> domain = np.array([[-0.1, -0.2, -0.4], [1.5, 3.0, 6.0]])
    >>> print(
    ...     LUT3D(
    ...         spow(LUT3D.linear_table(16), 1 / 2.2),
    ...         "My LUT",
    ...         domain,
    ...         comments=["A first comment.", "A second comment."],
    ...     )
    ... )
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

    def __init__(
        self,
        table: ArrayLike | None = None,
        name: str | None = None,
        domain: ArrayLike | None = None,
        size: ArrayLike | None = None,
        comments: Sequence | None = None,
    ) -> None:
        domain = as_float_array(optional(domain, [[0, 0, 0], [1, 1, 1]]))
        size = optional(size, 33)

        super().__init__(table, name, 3, domain, size, comments)

    def _validate_table(self, table: ArrayLike) -> NDArrayFloat:
        """
        Validate given table is a 4D array and that its dimensions are equal.

        Parameters
        ----------
        table
            Table to validate.

        Returns
        -------
        :class:`numpy.ndarray`
            Validated table as a :class:`ndarray` instance.
        """

        table = as_float_array(table)

        attest(len(table.shape) == 4, "The table must be a 4D array!")

        return table

    def _validate_domain(self, domain: ArrayLike) -> NDArrayFloat:
        """
        Validate given domain.

        Parameters
        ----------
        domain
            Domain to validate.

        Returns
        -------
        :class:`numpy.ndarray`
            Validated domain as a :class:`ndarray` instance.

        Notes
        -----
        -   A :class:`LUT3D` class instance must use an implicit domain.
        """

        domain = as_float_array(domain)

        attest(len(domain.shape) == 2, "The domain must be a 2D array!")

        attest(
            domain.shape[0] >= 2,
            "The domain row count must be equal or greater than 2!",
        )

        attest(domain.shape[1] == 3, "The domain column count must be equal to 3!")

        return domain

    def is_domain_explicit(self) -> bool:
        """
        Return whether the *LUT* domain is explicit (or implicit).

        An implicit domain is defined by its shape only::

            [[0 0 0]
             [1 1 1]]

        While an explicit domain defines every single discrete samples::

            [[0.0 0.0 0.0]
             [0.1 0.1 0.1]
             [0.2 0.2 0.2]
             [0.3 0.3 0.3]
             [0.4 0.4 0.4]
             [0.8 0.8 0.8]
             [1.0 1.0 1.0]]

        Returns
        -------
        :class:`bool`
            Is *LUT* domain explicit.

        Examples
        --------
        >>> LUT3D().is_domain_explicit()
        False
        >>> domain = np.array([[-0.1, -0.2, -0.4], [0.7, 1.4, 6.0], [1.5, 3.0, np.nan]])
        >>> LUT3D(domain=domain).is_domain_explicit()
        True
        """

        return self.domain.shape != (2, 3)

    @staticmethod
    def linear_table(
        size: ArrayLike | None = None,
        domain: ArrayLike | None = None,
    ) -> NDArrayFloat:
        """
        Return a linear table, the number of output samples :math:`n` is equal
        to ``size**3 * 3`` or ``size[0] * size[1] * size[2] * 3``.

        Parameters
        ----------
        size
            Expected table size, default to 33.
        domain
            Domain of the table.

        Returns
        -------
        :class:`numpy.ndarray`
            Linear table with ``size**3 * 3`` or
            ``size[0] * size[1] * size[2] * 3`` samples.

        Examples
        --------
        >>> LUT3D.linear_table(3, np.array([[-0.1, -0.2, -0.4], [1.5, 3.0, 6.0]]))
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
        >>> LUT3D.linear_table(
        ...     np.array([3, 3, 2]),
        ...     np.array([[-0.1, -0.2, -0.4], [1.5, 3.0, 6.0]]),
        ... )
        array([[[[-0.1, -0.2, -0.4],
                 [-0.1, -0.2,  6. ]],
        <BLANKLINE>
                [[-0.1,  1.4, -0.4],
                 [-0.1,  1.4,  6. ]],
        <BLANKLINE>
                [[-0.1,  3. , -0.4],
                 [-0.1,  3. ,  6. ]]],
        <BLANKLINE>
        <BLANKLINE>
               [[[ 0.7, -0.2, -0.4],
                 [ 0.7, -0.2,  6. ]],
        <BLANKLINE>
                [[ 0.7,  1.4, -0.4],
                 [ 0.7,  1.4,  6. ]],
        <BLANKLINE>
                [[ 0.7,  3. , -0.4],
                 [ 0.7,  3. ,  6. ]]],
        <BLANKLINE>
        <BLANKLINE>
               [[[ 1.5, -0.2, -0.4],
                 [ 1.5, -0.2,  6. ]],
        <BLANKLINE>
                [[ 1.5,  1.4, -0.4],
                 [ 1.5,  1.4,  6. ]],
        <BLANKLINE>
                [[ 1.5,  3. , -0.4],
                 [ 1.5,  3. ,  6. ]]]])
        >>> domain = np.array([[-0.1, -0.2, -0.4], [0.7, 1.4, 6.0], [1.5, 3.0, np.nan]])
        >>> LUT3D.linear_table(domain=domain)
        array([[[[-0.1, -0.2, -0.4],
                 [-0.1, -0.2,  6. ]],
        <BLANKLINE>
                [[-0.1,  1.4, -0.4],
                 [-0.1,  1.4,  6. ]],
        <BLANKLINE>
                [[-0.1,  3. , -0.4],
                 [-0.1,  3. ,  6. ]]],
        <BLANKLINE>
        <BLANKLINE>
               [[[ 0.7, -0.2, -0.4],
                 [ 0.7, -0.2,  6. ]],
        <BLANKLINE>
                [[ 0.7,  1.4, -0.4],
                 [ 0.7,  1.4,  6. ]],
        <BLANKLINE>
                [[ 0.7,  3. , -0.4],
                 [ 0.7,  3. ,  6. ]]],
        <BLANKLINE>
        <BLANKLINE>
               [[[ 1.5, -0.2, -0.4],
                 [ 1.5, -0.2,  6. ]],
        <BLANKLINE>
                [[ 1.5,  1.4, -0.4],
                 [ 1.5,  1.4,  6. ]],
        <BLANKLINE>
                [[ 1.5,  3. , -0.4],
                 [ 1.5,  3. ,  6. ]]]])
        """

        size = optional(size, 33)
        domain = as_float_array(optional(domain, [[0, 0, 0], [1, 1, 1]]))

        if domain.shape != (2, 3):
            samples = list(
                np.flip(
                    # NOTE: "dtype=object" is required for ragged array support
                    # in "Numpy" 1.24.0.
                    as_array(
                        [
                            axes[: (~np.isnan(axes)).cumsum().argmax() + 1]
                            for axes in np.transpose(domain)
                        ],
                        dtype=object,  # pyright: ignore
                    ),
                    -1,
                )
            )
            size_array = as_int_array([len(axes) for axes in samples])
        else:
            size_array = np.tile(size, 3) if is_numeric(size) else as_int_array(size)

            R, G, B = tsplit(domain)

            size_array = np.flip(size_array, -1)
            samples = [
                np.linspace(a[0], a[1], size_array[i]) for i, a in enumerate([B, G, R])
            ]

        table = np.flip(
            np.reshape(
                np.transpose(np.meshgrid(*samples, indexing="ij")),
                np.hstack([np.flip(size_array, -1), 3]),
            ),
            -1,
        )

        return table

    def invert(self, **kwargs: Any) -> LUT3D:
        """
        Compute and returns an inverse copy of the *LUT*.

        Other Parameters
        ----------------
        extrapolate
            Whether to extrapolate the *LUT* when computing its inverse.
            Extrapolation is performed by reflecting the *LUT* cube along its 8
            faces. Note that the domain is extended beyond [0, 1], thus the
            *LUT* might not be handled properly in other software.
        interpolator
            Interpolator class type or object to use as interpolating function.
        query_size
            Number of points to query in the KDTree, their mean is computed,
            resulting in a smoother result.
        size
            Size of the inverse *LUT*. With the given implementation, it is
            good practise to double the size of the inverse *LUT* to provide a
            smoother result. If ``size`` is not given,
            :math:`2^{\\sqrt{size_{LUT}} + 1} + 1` will be used instead.

        Returns
        -------
        :class:`colour.LUT3D`
            Inverse *LUT* class instance.

        Examples
        --------
        >>> LUT = LUT3D()
        >>> print(LUT)
        LUT3D - Unity 33
        ----------------
        <BLANKLINE>
        Dimensions : 3
        Domain     : [[ 0.  0.  0.]
                      [ 1.  1.  1.]]
        Size       : (33, 33, 33, 3)
        >>> print(LUT.invert())
        LUT3D - Unity 33 - Inverse
        --------------------------
        <BLANKLINE>
        Dimensions : 3
        Domain     : [[ 0.  0.  0.]
                      [ 1.  1.  1.]]
        Size       : (108, 108, 108, 3)
        """

        if self.is_domain_explicit():
            raise NotImplementedError(
                'Inverting a "LUT3D" with an explicit domain is not implemented!'
            )

        interpolator = kwargs.get("interpolator", table_interpolation_trilinear)
        extrapolate = kwargs.get("extrapolate", False)
        query_size = kwargs.get("query_size", 3)

        LUT = self.copy()
        source_size = LUT.size
        target_size = kwargs.get("size", (as_int(2 ** (np.sqrt(source_size) + 1) + 1)))

        if target_size > 129:  # pragma: no cover
            usage_warning("LUT3D inverse computation time could be excessive!")

        if extrapolate:
            LUT.table = np.pad(
                LUT.table,
                [(1, 1), (1, 1), (1, 1), (0, 0)],
                "reflect",
                reflect_type="odd",
            )

            LUT.domain[0] -= 1 / (source_size - 1)
            LUT.domain[1] += 1 / (source_size - 1)

        # "LUT_t" is an intermediate LUT with a size equal to that of the
        # final inverse LUT which is usually larger than the input LUT.
        # The intent is to smooth the inverse LUT's table by increasing the
        # resolution of the KDTree.
        LUT_t = LUT3D(size=target_size, domain=LUT.domain)
        table = np.reshape(LUT_t.table, (-1, 3))
        LUT_t.table = LUT.apply(LUT_t.table, interpolator=interpolator)

        tree = KDTree(np.reshape(LUT_t.table, (-1, 3)))

        # "LUT_q" stores the indexes of the KDTree query, i.e. the closest
        # entry of "LUT_t" for any searched table sample.
        LUT_q = LUT3D(size=target_size, domain=LUT.domain)
        query = tree.query(table, query_size)[-1]
        if query_size == 1:
            LUT_q.table = np.reshape(
                table[query], (target_size, target_size, target_size, 3)
            )
        else:
            LUT_q.table = np.reshape(
                np.mean(table[query], axis=-2),
                (target_size, target_size, target_size, 3),
            )

        LUT_q.name = f"{self.name} - Inverse"

        return LUT_q

    def apply(self, RGB: ArrayLike, **kwargs: Any) -> NDArrayFloat:
        """
        Apply the *LUT* to given *RGB* colourspace array using given method.

        Parameters
        ----------
        RGB
            *RGB* colourspace array to apply the *LUT* onto.

        Other Parameters
        ----------------
        direction
            Whether the *LUT* should be applied in the forward or inverse
            direction.
        extrapolate
            Whether to extrapolate the *LUT* when computing its inverse.
            Extrapolation is performed by reflecting the *LUT* cube along its 8
            faces.
        interpolator
            Interpolator object to use as interpolating function.
        interpolator_kwargs
            Arguments to use when calling the interpolating function.
        query_size
            Number of points to query in the KDTree, their mean is computed,
            resulting in a smoother result.
        size
            Size of the inverse *LUT*. With the given implementation, it is
            good practise to double the size of the inverse *LUT* to provide a
            smoother result. If ``size`` is not given,
            :math:`2^{\\sqrt{size_{LUT}} + 1} + 1` will be used instead.

        Other Parameters
        ----------------
        \\**kwargs : dict, optional
            Keywords arguments for deprecation management.

        Returns
        -------
        :class:`numpy.ndarray`
            Interpolated *RGB* colourspace array.

        Examples
        --------
        >>> LUT = LUT3D(LUT3D.linear_table() ** (1 / 2.2))
        >>> RGB = np.array([0.18, 0.18, 0.18])
        >>> LUT.apply(RGB)  # doctest: +ELLIPSIS
        array([ 0.4583277...,  0.4583277...,  0.4583277...])
        >>> LUT.apply(LUT.apply(RGB), direction="Inverse")
        ... # doctest: +ELLIPSIS
        array([ 0.1781995...,  0.1809414...,  0.1809513...])
        >>> from colour.algebra import spow
        >>> domain = np.array(
        ...     [
        ...         [-0.1, -0.2, -0.4],
        ...         [0.3, 1.4, 6.0],
        ...         [0.7, 3.0, np.nan],
        ...         [1.1, np.nan, np.nan],
        ...         [1.5, np.nan, np.nan],
        ...     ]
        ... )
        >>> table = spow(LUT3D.linear_table(domain=domain), 1 / 2.2)
        >>> LUT = LUT3D(table, domain=domain)
        >>> RGB = np.array([0.18, 0.18, 0.18])
        >>> LUT.apply(RGB)  # doctest: +ELLIPSIS
        array([ 0.2996370..., -0.0901332..., -0.3949770...])
        """

        direction = validate_method(
            kwargs.get("direction", "Forward"), ("Forward", "Inverse")
        )

        interpolator = kwargs.get("interpolator", table_interpolation_trilinear)
        interpolator_kwargs = kwargs.get("interpolator_kwargs", {})

        R, G, B = tsplit(RGB)

        settings = {"interpolator": interpolator}
        settings.update(**kwargs)
        LUT = self.invert(**settings) if direction == "inverse" else self

        if LUT.is_domain_explicit():
            domain_min = LUT.domain[0, ...]
            domain_max = [
                axes[: (~np.isnan(axes)).cumsum().argmax() + 1][-1]
                for axes in np.transpose(LUT.domain)
            ]
            usage_warning(
                f'"LUT" was defined with an explicit domain but requires an '
                f"implicit domain to be applied. The following domain will be "
                f"used: {np.vstack([domain_min, domain_max])}"
            )
        else:
            domain_min, domain_max = LUT.domain

        RGB_l = [
            linear_conversion(j, (domain_min[i], domain_max[i]), (0, 1))
            for i, j in enumerate((R, G, B))
        ]

        RGB_i = interpolator(tstack(RGB_l), LUT.table, **interpolator_kwargs)

        return RGB_i

    # ------------------------------------------------------------------------#
    # ---              API Changes and Deprecation Management              ---#
    # ------------------------------------------------------------------------#
    def as_LUT(  # noqa: D102
        self,
        cls: Type[AbstractLUT],
        force_conversion: bool = False,
        **kwargs: Any,
    ) -> AbstractLUT:  # pragma: no cover
        # Docstrings are omitted for documentation purposes.
        usage_warning(
            str(
                ObjectRenamed(
                    "LUT3D.as_LUT",
                    "LUT3D.convert",
                )
            )
        )

        return self.convert(cls, force_conversion, **kwargs)  # pyright: ignore


def LUT_to_LUT(
    LUT,
    cls: Type[AbstractLUT],
    force_conversion: bool = False,
    **kwargs: Any,
) -> AbstractLUT:
    """
    Convert given *LUT* to given ``cls`` class instance.

    Parameters
    ----------
    cls
        *LUT* class instance.
    force_conversion
        Whether to force the conversion if destructive.

    Other Parameters
    ----------------
    channel_weights
        Channel weights in case of a downcast from a :class:`LUT3x1D` or
        :class:`LUT3D` class instance.
    interpolator
        Interpolator class type to use as interpolating function.
    interpolator_kwargs
        Arguments to use when instantiating the interpolating function.
    size
        Expected table size in case of an upcast to or a downcast from a
        :class:`LUT3D` class instance.

    Returns
    -------
    :class:`colour.LUT1D` or :class:`colour.LUT3x1D` or :class:`colour.LUT3D`
        Converted *LUT* class instance.

    Warnings
    --------
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
    >>> print(LUT_to_LUT(LUT3x1D(), LUT1D, force_conversion=True))
    LUT1D - Unity 10 - Converted 3x1D to 1D
    ---------------------------------------
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

    ranks = {LUT1D: 1, LUT3x1D: 2, LUT3D: 3}
    path = (ranks[LUT.__class__], ranks[cls])  # pyright: ignore
    path_verbose = [f"{element}D" if element != 2 else "3x1D" for element in path]
    if path in ((1, 3), (2, 1), (2, 3), (3, 1), (3, 2)) and not force_conversion:
        raise ValueError(
            f'Conversion of a "LUT" {path_verbose[0]} to a "LUT" '
            f"{path_verbose[1]} is destructive, please use the "
            f'"force_conversion" argument to proceed!'
        )

    suffix = f" - Converted {path_verbose[0]} to {path_verbose[1]}"
    name = f"{LUT.name}{suffix}"

    # Same dimension conversion, returning a copy.
    if len(set(path)) == 1:
        LUT = LUT.copy()
        LUT.name = name
    else:
        size = kwargs.get("size", 33 if cls is LUT3D else 10)
        if "size" in kwargs:
            del kwargs["size"]

        channel_weights = as_float_array(kwargs.get("channel_weights", full(3, 1 / 3)))
        if "channel_weights" in kwargs:
            del kwargs["channel_weights"]

        if isinstance(LUT, LUT1D):
            if cls is LUT3x1D:
                domain = tstack([LUT.domain, LUT.domain, LUT.domain])
                table = tstack([LUT.table, LUT.table, LUT.table])
            elif cls is LUT3D:
                domain = tstack([LUT.domain, LUT.domain, LUT.domain])
                table = LUT3D.linear_table(size, domain)
                table = LUT.apply(table, **kwargs)
        elif isinstance(LUT, LUT3x1D):
            if cls is LUT1D:
                domain = np.sum(LUT.domain * channel_weights, axis=-1)
                table = np.sum(LUT.table * channel_weights, axis=-1)
            elif cls is LUT3D:
                domain = LUT.domain
                table = LUT3D.linear_table(size, domain)
                table = LUT.apply(table, **kwargs)
        elif isinstance(LUT, LUT3D):
            if cls is LUT1D:
                domain = np.sum(LUT.domain * channel_weights, axis=-1)
                table = LUT1D.linear_table(size, domain)
                table = LUT.apply(tstack([table, table, table]), **kwargs)
                table = np.sum(table * channel_weights, axis=-1)
            elif cls is LUT3x1D:
                domain = LUT.domain
                table = LUT3x1D.linear_table(size, domain)
                table = LUT.apply(table, **kwargs)

        LUT = cls(
            table=table,
            name=name,
            domain=domain,
            size=table.shape[0],
            comments=LUT.comments,
        )  # pyright: ignore

    return LUT


@add_metaclass(abc.ABCMeta)
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


class LUTSequence(MutableSequence):
    """
    Defines the base class for a *LUT* sequence, i.e. a series of *LUTs*.

    The `colour.LUTSequence` class can be used to model series of *LUTs* such
    as when a shaper *LUT* is combined with a 3D *LUT*.

    Other Parameters
    ----------------
    \\*args : list, optional
        Sequence of `colour.LUT1D`, `colour.LUT3x1D`, `colour.LUT3D` or
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
    >>> LUT_3 = LUT3x1D()
    >>> print(LUTSequence(LUT_1, LUT_2, LUT_3))
    LUT Sequence
    ------------
    <BLANKLINE>
    Overview
    <BLANKLINE>
        LUT1D ---> LUT3D ---> LUT3x1D
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
        LUT3x1D - Unity 10
        ------------------
    <BLANKLINE>
        Dimensions : 2
        Domain     : [[ 0.  0.  0.]
                      [ 1.  1.  1.]]
        Size       : (10, 3)
    """

    def __init__(self, *args):
        for arg in args:
            assert isinstance(
                arg, (LUT1D, LUT3x1D, LUT3D, AbstractLUTSequenceOperator)
            ), (
                '"args" elements must be instances of "LUT1D", '
                '"LUT3x1D", "LUT3D" or "AbstractLUTSequenceOperator"!'
            )

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
        LUT1D or LUT3x1D or LUT3D or AbstractLUTSequenceOperator
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
        value : LUT1D or LUT3x1D or LUT3D or AbstractLUTSequenceOperator
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
            " " * 4, "\n\n".join([str(a) for a in self._sequence]), flags=re.MULTILINE
        )
        operations = re.sub("^\\s+$", "", operations, flags=re.MULTILINE)

        return f"LUT Sequence\n------------\n\nOverview\n\n    {' ---> '.join([a.__class__.__name__ for a in self._sequence])}\n\nOperations\n\n{operations}"

    def __repr__(self):
        """
        Returns an evaluable string representation of the *LUT* sequence.

        Returns
        -------
        unicode
            Evaluable string representation.
        """

        operations = re.sub(
            "^",
            " " * 4,
            ",\n".join([repr(a) for a in self._sequence]),
            flags=re.MULTILINE,
        )
        operations = re.sub("^\\s+$", "", operations, flags=re.MULTILINE)

        return f"{self.__class__.__name__}(\n{operations}\n)"

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

        # pylint: disable=C0200
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

    # pylint: disable=W0221
    def insert(self, index: int, value):
        """
        Inserts given *LUT* at given index into the *LUT* sequence.

        Parameters
        ----------
        index : index
            Index to insert the *LUT* at into the *LUT* sequence.
        value : LUT1D or LUT3x1D or LUT3D or AbstractLUTSequenceOperator
            *LUT* to insert into the *LUT* sequence.
        """

        assert isinstance(
            value, (LUT1D, LUT3x1D, LUT3D, AbstractLUTSequenceOperator)
        ), (
            '"LUT" must be an instance of "LUT1D", "LUT3x1D", "LUT3D" or '
            '"AbstractLUTSequenceOperator"!'
        )

        self._sequence.insert(index, value)

    def apply(
        self,
        RGB,
        interpolator_1D=LinearInterpolator,
        interpolator_1D_kwargs=None,
        interpolator_3D=table_interpolation_trilinear,
        clip_input_to_domain=False,
        interpolator_3D_kwargs=None,
        **kwargs,
    ):
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
            :class:`colour.LUT1D` (and :class:`colour.LUT3x1D`) class
            instances.
        interpolator_1D_kwargs : dict_like, optional
            Arguments to use when calling the interpolating function for
            :class:`colour.LUT1D` (and :class:`colour.LUT3x1D`) class
            instances.
        interpolator_3D : object, optional
            Interpolator object to use as interpolating function for
            :class:`colour.LUT3D` class instances.
        interpolator_3D_kwargs : dict_like, optional
            Arguments to use when calling the interpolating function for
            :class:`colour.LUT3D` class instances.

        Other Parameters
        ----------------
        \\**kwargs : dict, optional
            Keywords arguments for deprecation management.

        Returns
        -------
        ndarray
            Processed *RGB* colourspace array.

        Examples
        --------
        >>> LUT_1 = LUT1D(LUT1D.linear_table(16) + 0.125)
        >>> LUT_2 = LUT3D(LUT3D.linear_table(16) ** (1 / 2.2))
        >>> LUT_3 = LUT3x1D(LUT3x1D.linear_table(16) * 0.750)
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

        interpolator_1D_kwargs = handle_arguments_deprecation(
            {
                "ArgumentRenamed": [["interpolator_1D_args", "interpolator_1D_kwargs"]],
            },
            **kwargs,
        ).get("interpolator_1D_kwargs", interpolator_1D_kwargs)

        interpolator_3D_kwargs = handle_arguments_deprecation(
            {
                "ArgumentRenamed": [["interpolator_3D_args", "interpolator_3D_kwargs"]],
            },
            **kwargs,
        ).get("interpolator_3D_kwargs", interpolator_3D_kwargs)

        for operation in self:
            if clip_input_to_domain:
                if isinstance(operation, LUT1D):
                    RGB = np.clip(
                        RGB, np.nanmin(operation.domain), np.nanmax(operation.domain)
                    )
                elif isinstance(operation, (LUT3x1D, LUT3D)):
                    r, g, b = tsplit(RGB)
                    domain_r, domain_g, domain_b = tsplit(operation.domain)
                    r = np.clip(r, np.nanmin(domain_r), np.nanmax(domain_r))
                    g = np.clip(g, np.nanmin(domain_g), np.nanmax(domain_g))
                    b = np.clip(b, np.nanmin(domain_b), np.nanmax(domain_b))
                    RGB = tstack((r, g, b))
            if isinstance(operation, (LUT1D, LUT3x1D)):
                RGB = operation.apply(RGB, interpolator_1D, interpolator_1D_kwargs)
            elif isinstance(operation, LUT3D):
                RGB = operation.apply(RGB, interpolator_3D, interpolator_3D_kwargs)
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


class Range(AbstractLUTSequenceOperator):
    """
    Defines the class for a *Range* scale.

    Parameters
    ----------
    min_in_value : numeric, optional
        Input value which will be mapped to min_out_value.
    max_in_value : numeric, optional
        Input value which will be mapped to max_out_value.
    min_out_value : numeric, optional
        Output value to which min_in_value will be mapped.
    max_out_value : numeric, optional
        Output value to which max_in_value will be mapped.
    no_clamp : boolean, optional
        Whether to not clamp the output values.
    name : unicode, optional
        *Range* name.
    comments : array_like, optional
        Comments to add to the *Range*.

    Methods
    -------
    apply

    Examples
    --------
    A full to legal scale:

    >>> print(Range(name='Full to Legal',
                    min_out_value=64./1023,
                    max_out_value=940./1023))
    Range - Full to Legal
    ---------------------
    <BLANKLINE>
    Input      : 0.0 - 1.0
    Output     : 0.0625610948192 - 0.918866080156
    <BLANKLINE>
    Clamping   : No
    """

    def __init__(
        self,
        min_in_value=0.0,
        max_in_value=1.0,
        min_out_value=0.0,
        max_out_value=1.0,
        no_clamp=True,
        name="",
        comments=None,
    ):
        self.min_in_value = min_in_value
        self.max_in_value = max_in_value
        self.min_out_value = min_out_value
        self.max_out_value = max_out_value
        self.no_clamp = no_clamp
        self.name = name
        self.comments = comments

    def apply(self, RGB, *args):
        """
        Applies the *Range* scale to given *RGB* array.

        Parameters
        ----------
        RGB : array_like
            *RGB* array to apply the *Range* scale to.

        Returns
        -------
        ndarray
            Scaled *RGB* array.

        Examples
        --------
        >>> R = Range(
        ...     name="Legal to Full",
        ...     min_in_value=64.0 / 1023,
        ...     max_in_value=940.0 / 1023,
        ...     no_clamp=False,
        ... )
        >>> RGB = np.array([0.8, 0.9, 1.0])
        >>> R.apply(RGB)
        array([ 0.86118721,  0.97796804,  1.        ])
        """
        RGB = np.asarray(RGB)

        scale = (self.max_out_value - self.min_out_value) / (
            self.max_in_value - self.min_in_value
        )
        RGB_out = RGB * scale + self.min_out_value - self.min_in_value * scale

        if not self.no_clamp:
            RGB_out = np.clip(RGB_out, self.min_out_value, self.max_out_value)

        return RGB_out

    def __str__(self):
        """
        Returns a formatted string representation of the *Range* operation.

        Returns
        -------
        unicode
            Formatted string representation.
        """

        return (
            "{} - {}\n"
            "{}\n\n"
            "Input      : {} - {}\n"
            "Output     : {} - {}\n\n"
            "Clamping   : {}"
            "{}".format(
                self.__class__.__name__,
                self.name,
                "-" * (len(self.__class__.__name__) + 3 + len(self.name)),
                self.min_in_value,
                self.max_in_value,
                self.min_out_value,
                self.max_out_value,
                "No" if self.no_clamp else "Yes",
                "\n\n{}".format("\n".join(self.comments)) if self.comments else "",
            )
        )


class Matrix(AbstractLUTSequenceOperator):
    """
    Defines the base class for a *Matrix* transform.

    Parameters
    ----------
    array : array_like, optional
        3x3 or 3x4 matrix for the transform.
    name : unicode, optional
        *Matrix* name.
    comments : array_like, optional
        Comments to add to the *Matrix*.

    Methods
    -------
    apply

    Examples
    --------
    Instantiating an identity matrix:

    >>> print(Matrix(name="Identity"))
    Matrix - Identity
    -----------------
    <BLANKLINE>
    Dimensions : (3, 3)
    Matrix     : [[ 1.  0.  0.]
                  [ 0.  1.  0.]
                  [ 0.  0.  1.]]

    Instantiating a matrix with comments:

    >>> array = np.array([[ 1.45143932, -0.23651075, -0.21492857],
        ...                   [-0.07655377,  1.1762297 , -0.09967593],
        ...                   [ 0.00831615, -0.00603245,  0.9977163 ]])
    >>> print(
    ...     Matrix(
    ...         array=array,
    ...         name="AP0 to AP1",
    ...         comments=["A first comment.", "A second comment."],
    ...     )
    ... )
    Matrix - AP0 to AP1
    -------------------
    <BLANKLINE>
    Dimensions : (3, 3)
    Matrix     : [[ 1.45143932 -0.23651075 -0.21492857]
                  [-0.07655377  1.1762297  -0.09967593]
                  [ 0.00831615 -0.00603245  0.9977163 ]]
    <BLANKLINE>
    A first comment.
    A second comment.
    """

    def __init__(self, array=np.identity(3), name="", comments=None):
        self.array = array
        self.name = name
        self.comments = comments

    @staticmethod
    def _validate_array(array):
        assert array.shape in [(3, 4), (3, 3)], "Matrix shape error!"

        return array

    def apply(self, RGB, *args):
        """
        Applies the *Matrix* transform to given *RGB* array.

        Parameters
        ----------
        RGB : array_like
            *RGB* array to apply the *Matrix* transform to.

        Returns
        -------
        ndarray
            Transformed *RGB* array.

        Examples
        --------
        >>> array = np.array(
        ...     [
        ...         [1.45143932, -0.23651075, -0.21492857],
        ...         [-0.07655377, 1.1762297, -0.09967593],
        ...         [0.00831615, -0.00603245, 0.9977163],
        ...     ]
        ... )
        >>> M = Matrix(array=array)
        >>> RGB = [0.3, 0.4, 0.5]
        >>> M.apply(RGB)
        array([ 0.23336321,  0.39768778,  0.49894002])
        """
        RGB = np.asarray(RGB)

        if self.array.shape == (3, 4):
            R, G, B = tsplit(RGB)
            RGB = tstack([R, G, B, np.ones(R.shape)])

        return vector_dot(self.array, RGB)

    def __str__(self):
        """
        Returns a formatted string representation of the *Matrix*.

        Returns
        -------
        unicode
            Formatted string representation.
        """

        def _indent_array(a):
            """
            Indents given array string representation.
            """

            return str(a).replace(" [", " " * 14 + "[")

        return (
            "{} - {}\n"
            "{}\n\n"
            "Dimensions : {}\n"
            "Matrix     : {}"
            "{}".format(
                self.__class__.__name__,
                self.name,
                "-" * (len(self.__class__.__name__) + 3 + len(self.name)),
                self.array.shape,
                _indent_array(self.array),
                "\n\n{}".format("\n".join(self.comments)) if self.comments else "",
            )
        )


class Exponent(AbstractLUTSequenceOperator):
    def __init__(
        self,
        exponent=None,
        offset=None,  # ignored for basic
        style="basicFwd",
        name="",
        comments=None,
    ):
        if offset is None:
            offset = [0, 0, 0]
        if exponent is None:
            exponent = [1, 1, 1]
        self.exponent = exponent
        self.offset = offset
        self.style = style
        self.name = name
        self.comments = comments

    def apply(self, RGB, *args):
        if as_float_array(RGB).size == 3 or (
            isinstance(RGB, np.ndarray) and RGB.shape[-1] == 3
        ):
            r, g, b = tsplit(np.asarray(RGB))

        else:
            r = g = b = np.asarray(RGB)

        if self.style.lower()[:5] == "basic":
            r = exponent_function_basic(r, self.exponent[0], self.style)
            g = exponent_function_basic(g, self.exponent[1], self.style)
            b = exponent_function_basic(b, self.exponent[2], self.style)

            return tstack((r, g, b))

        if self.style.lower()[:8] == "moncurve":
            r = exponent_function_monitor_curve(
                r, self.exponent[0], self.offset[0], self.style
            )
            g = exponent_function_monitor_curve(
                g, self.exponent[1], self.offset[1], self.style
            )
            b = exponent_function_monitor_curve(
                b, self.exponent[2], self.offset[2], self.style
            )

            return tstack((r, g, b))

    def __str__(self):
        return (
            "{} - {}\n"
            "{}\n\n"
            "Exponent.r : {}\n"
            "Exponent.g : {}\n"
            "Exponent.b : {}\n"
            "{}"
            "Style : {}\n"
            "{}".format(
                self.__class__.__name__,
                self.name,
                "-" * (len(self.__class__.__name__) + 3 + len(self.name)),
                self.exponent[0],
                self.exponent[1],
                self.exponent[2],
                f"Offset.r : {self.offset[0]}\nOffset.g : {self.offset[1]}\nOffset.b : {self.offset[2]}\n"
                if self.style.lower()[:8] == "moncurve"
                else "",
                self.style,
                "\n\n{}".format("\n".join(self.comments)) if self.comments else "",
            )
        )


class Log(AbstractLUTSequenceOperator):
    def __init__(
        self,
        base=2,
        logSideSlope=1,
        logSideOffset=0,
        linSideSlope=1,
        linSideOffset=0,
        linSideBreak=0,
        linearSlope=1,
        style="cameraLinToLog",
        name="",
        comments=None,
    ):
        self.name = name
        self.style = style
        self.comments = comments or []
        self.base = base
        self.linSideOffset = linSideOffset
        self.linSideSlope = linSideSlope
        self.logSideSlope = logSideSlope
        self.logSideOffset = logSideOffset
        self.linSideBreak = linSideBreak
        self.linearSlope = linearSlope

    @property
    def lin_to_log_styles(self):
        return ["log2", "log10", "linToLog", "cameraLinToLog"]

    @property
    def log_to_lin_styles(self):
        return ["antiLog2", "antiLog10", "logToLin", "cameraLogToLin"]

    @property
    def style(self):
        style = self._style
        if style.startswith("camera") and self.lin_side_break is None:
            style = style.replace("cameraL", "l")
        return style

    @style.setter
    def style(self, value):
        if value not in self.log_styles:
            raise ValueError(f"Invalid Log style: {value}")

        if value.endswith("2"):
            self.base = 2
        elif value.endswith("10"):
            self.base = 10
        else:
            self._style = value

    @property
    def log_styles(self):
        return self.log_to_lin_styles + self.lin_to_log_styles

    @property
    def log_side_slope(self):
        return self.logSideSlope

    @log_side_slope.setter
    def log_side_slope(self, *value):
        self.logSideSlope = value

    @property
    def log_side_offset(self):
        return self.logSideOffset

    @log_side_offset.setter
    def log_side_offset(self, *value):
        self.logSideOffset = value

    @property
    def lin_side_slope(self):
        return self.linSideSlope

    @lin_side_slope.setter
    def lin_side_slope(self, *value):
        self.linSideSlope = value

    @property
    def lin_side_offset(self):
        return self.linSideOffset

    @lin_side_offset.setter
    def lin_side_offset(self, *value):
        self.linSideOffset = value

    @property
    def lin_side_break(self):
        return self.linSideBreak

    @lin_side_break.setter
    def lin_side_break(self, *value):
        if value is None:
            self.linSideBreak = None
        self.linSideBreak = value

    @property
    def linear_slope(self):
        return self.linearSlope

    @linear_slope.setter
    def linear_slope(self, *value):
        if value is None:
            self.linearSlope = None
        self.linearSlope = value

    def is_encoding_style(self, style=None):
        style = style or self.style
        return style.lower() in [s.lower() for s in self.lin_to_log_styles]

    def is_decoding_style(self, style=None):
        style = style or self.style
        return style.lower() in [s.lower() for s in self.log_to_lin_styles]

    def _logarithmic_function_factory(
        self,
        lin_side_slope=None,
        lin_side_offset=None,
        log_side_slope=None,
        log_side_offset=None,
        lin_side_break=None,
        linear_slope=None,
        base=None,
        style="log10",
    ):
        # TODO: promote to module level? Make static?
        def _is_decoding_style(s):
            s = style.lower()
            return s.startswith("anti") or s.endswith("lin")

        function_kwargs = {}
        if style[-1] in ["2", "0"]:
            __function = partial(
                logarithmic_function_basic, base=int(style[-1]), style=style
            )

        elif style.startswith("anti") or any(
            x is None
                for x in [
                    lin_side_slope,
                    lin_side_offset,
                    log_side_slope,
                    log_side_offset,
                ]
        ):
            style = "logB"
            if style.lower().startswith("anti"):
                style = "antiLogB"

            __function = partial(logarithmic_function_basic, base=base, style=style)

        else:
            function_kwargs = {
                "log_side_slope": log_side_slope,
                "log_side_offset": log_side_offset,
                "lin_side_slope": lin_side_slope,
                "lin_side_offset": lin_side_offset,
            }

            if lin_side_break is not None:
                function_kwargs.update(lin_side_break=lin_side_break)
                style = (
                    "cameraLogToLin" if _is_decoding_style(style) else "cameraLinToLog"
                )
                __function = partial(
                    logarithmic_function_camera, base=base, style=style
                )

            else:
                style = "logToLin" if _is_decoding_style(style) else "linToLog"
                __function = partial(
                    logarithmic_function_quasilog, base=base, style=style
                )

            if any(as_float_array(v).size > 1 for v in function_kwargs.values()):
                function_kwargs = {
                    k: v * np.ones(3) for k, v in function_kwargs.items()
                }

        return partial(__function, **function_kwargs)

    def _apply_directed(self, RGB, inverse=False):
        RGB_out = as_float_array(RGB)

        inverse_styles = dict(zip(
                self.lin_to_log_styles + self.log_to_lin_styles,
                self.log_to_lin_styles + self.lin_to_log_styles,
            ))
        style = inverse_styles[self.style] if inverse else self.style
        logarithmic_function = self._logarithmic_function_factory(
            style=style,
            base=self.base,
            lin_side_slope=self.lin_side_slope,
            lin_side_offset=self.lin_side_offset,
            log_side_slope=self.log_side_slope,
            log_side_offset=self.log_side_offset,
            lin_side_break=self.lin_side_break,
            linear_slope=self.linear_slope,
        )

        return logarithmic_function(RGB_out)

    def apply(self, RGB, *args):
        return self._apply_directed(RGB, inverse=False)

    def reverse(self, RGB):
        return self._apply_directed(RGB, inverse=True)

    def __str__(self):
        direction = (
            "Log to Linear" if self.style in self.log_to_lin_styles else "Linear to Log"
        )
        title = f"{f'{self.name} - ' if self.name else ''}{direction}"
        basic_style = self.style[-1] in "20"
        return (
            "{} - {}\n"
            "{}\n\n"
            "style          : {}\n"
            "base           : {}"
            "{}{}{}{}{}{}{}"
        ).format(
            self.__class__.__name__,
            title,
            "-" * (len(self.__class__.__name__) + 3 + len(title)),
            self.style,
            self.base,
            f"\nlogSideSlope   : {self.log_side_slope}" if not basic_style else "",
            f"\nlogSideOffset  : {self.log_side_offset}" if not basic_style else "",
            f"\nlinSideSlope   : {self.lin_side_slope}" if not basic_style else "",
            f"\nlinSideOffset  : {self.lin_side_offset}" if not basic_style else "",
            f"\nlinearSlope    : {self.linear_slope}"
            if not basic_style and self.linear_slope is not None
            else "",
            f"\nlinSideBreak   : {self.lin_side_break}"
            if not basic_style and self.lin_side_break is not None
            else "",
            "\n\n{}".format("\n".join(self.comments)) if self.comments else "",
        )

    def __repr__(self):
        # TODO: show only the used parameters (see __str__ method)
        return f"{self.__class__.__name__}(base={self.base}, logSideSlope={self.log_side_slope}, logSideOffset={self.log_side_offset}, linSideSlope={self.lin_side_slope}, linSideOffset={self.lin_side_offset}, linearSlope={self.linear_slope}, linSideBreak={self.lin_side_break}, style=\"{self.style}\"{f' name={self.name}' if self.name else ''})"
