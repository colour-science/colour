"""
LUT Processing
==============

Define the classes and definitions for *Look-Up Table* (*LUT*) processing
operations.

-   :class:`colour.LUT1D`: One-dimensional lookup table for single-channel
    transformations
-   :class:`colour.LUT3x1D`: Three parallel one-dimensional lookup tables
    for independent RGB channel processing
-   :class:`colour.LUT3D`: Three-dimensional lookup table for complex colour
    space transformations
-   :class:`colour.io.LUT_to_LUT`: Utility for converting between different
    LUT formats and types
"""

from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from copy import deepcopy
from operator import pow  # noqa: A004
from operator import add, iadd, imul, ipow, isub, itruediv, mul, sub, truediv

import numpy as np

from colour.algebra import (
    Extrapolator,
    LinearInterpolator,
    linear_conversion,
    table_interpolation_trilinear,
)
from colour.constants import EPSILON

if typing.TYPE_CHECKING:
    from colour.hints import (
        Any,
        ArrayLike,
        Literal,
        NDArrayFloat,
        Self,
        Sequence,
        Type,
    )

from colour.hints import List, cast
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
    multiline_repr,
    multiline_str,
    optional,
    required,
    runtime_warning,
    tsplit,
    tstack,
    usage_warning,
    validate_method,
)

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
]


class AbstractLUT(ABC):
    """
    Define the base class for *LUT* (Look-Up Table).

    This is an abstract base class (:class:`ABCMeta`) that must be inherited
    by concrete *LUT* implementations to provide common functionality and
    interface specifications.

    Parameters
    ----------
    table
        Underlying *LUT* table array containing the lookup values.
    name
        *LUT* identifying name.
    dimensions
        *LUT* dimensionality: typically 1 for a 1D *LUT*, 2 for a 3x1D *LUT*,
        and 3 for a 3D *LUT*.
    domain
        *LUT* input domain boundaries, also used to define the instantiation
        time default table domain.
    size
        *LUT* resolution or sampling density, also used to define the
        instantiation time default table size.
    comments
        Additional comments or metadata to associate with the *LUT*.

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
        self.comments = cast("list", optional(comments, self._comments))

    @property
    def table(self) -> NDArrayFloat:
        """
        Getter and setter for the underlying *LUT* table.

        Access or modify the lookup table data structure that defines the
        transformation mapping for this LUT instance.

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
    def table(self, value: ArrayLike) -> None:
        """Setter for the **self.table** property."""

        self._table = self._validate_table(value)

    @property
    def name(self) -> str:
        """
        Getter and setter for the *LUT* name.

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
    def name(self, value: str) -> None:
        """Setter for the **self.name** property."""

        attest(
            isinstance(value, str),
            f'"name" property: "{value}" type is not "str"!',
        )

        self._name = value

    @property
    def domain(self) -> NDArrayFloat:
        """
        Getter and setter for the *LUT* domain.

        The domain defines the input coordinate space for the lookup table,
        specifying the valid range of input values that can be interpolated.

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
    def domain(self, value: ArrayLike) -> None:
        """Setter for the **self.domain** property."""

        self._domain = self._validate_domain(value)

    @property
    def dimensions(self) -> int:
        """
        Getter for the *LUT* dimensions.

        Returns
        -------
        :class:`int`
            *LUT* dimensions.
        """

        return self._dimensions

    @property
    def size(self) -> int:
        """
        Getter for the *LUT* size.

        Returns
        -------
        :class:`int`
            *LUT* size.
        """

        return self._table.shape[0]

    @property
    def comments(self) -> list:
        """
        Getter and setter for the *LUT* comments.

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
    def comments(self, value: Sequence) -> None:
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

        return multiline_str(self, cast("List[dict]", attributes))

    def __repr__(self) -> str:
        """
        Return an evaluable string representation of the *LUT*.

        This method provides a string that, when evaluated, recreates the
        *LUT* object with its current state and configuration.

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

    __hash__ = None  # pyright: ignore

    def __eq__(self, other: object) -> bool:
        """
        Return whether the *LUT* is equal to the specified other object.

        Parameters
        ----------
        other
            Object to test whether it is equal to the *LUT*.

        Returns
        -------
        :class:`bool`
            Whether the specified object is equal to the *LUT*.
        """

        return isinstance(other, AbstractLUT) and all(
            [
                np.array_equal(self.table, other.table),
                np.array_equal(self.domain, other.domain),
            ]
        )

    def __ne__(self, other: object) -> bool:
        """
        Determine whether the *LUT* is not equal to the specified other
        object.

        Parameters
        ----------
        other
            Object to test for inequality with the *LUT*.

        Returns
        -------
        :class:`bool`
            Whether the specified object is not equal to the *LUT*.
        """

        return not (self == other)

    def __add__(self, a: ArrayLike | AbstractLUT) -> AbstractLUT:
        """
        Implement support for addition.

        Parameters
        ----------
        a
            *a* variable to add.

        Returns
        -------
        :class:`colour.io.luts.lut.AbstractLUT`
            Variable added *LUT*.
        """

        return self.arithmetical_operation(a, "+")

    def __iadd__(self, a: ArrayLike | AbstractLUT) -> Self:
        """
        Implement support for in-place addition.

        Add the specified operand to this *LUT* in-place, modifying the
        current instance rather than creating a new one.

        Parameters
        ----------
        a
            Operand to add in-place. Can be a numeric array or another
            *LUT* instance with compatible dimensions.

        Returns
        -------
        :class:`colour.io.luts.lut.AbstractLUT`
            Current *LUT* instance with the addition applied in-place.
        """

        return self.arithmetical_operation(a, "+", True)

    def __sub__(self, a: ArrayLike | AbstractLUT) -> Self:
        """
        Implement support for subtraction.

        Parameters
        ----------
        a
            Variable, array or *LUT* to subtract from the current *LUT*.

        Returns
        -------
        :class:`colour.io.luts.lut.AbstractLUT`
            Variable subtracted *LUT*.
        """

        return self.arithmetical_operation(a, "-")

    def __isub__(self, a: ArrayLike | AbstractLUT) -> Self:
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

    def __mul__(self, a: ArrayLike | AbstractLUT) -> Self:
        """
        Implement support for multiplication.

        Parameters
        ----------
        a
            Variable to multiply with the *LUT*. Can be a numeric array or
            another *LUT* instance.

        Returns
        -------
        :class:`colour.io.luts.lut.AbstractLUT`
            Variable multiplied *LUT*.
        """

        return self.arithmetical_operation(a, "*")

    def __imul__(self, a: ArrayLike | AbstractLUT) -> Self:
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

    def __div__(self, a: ArrayLike | AbstractLUT) -> Self:
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

    def __idiv__(self, a: ArrayLike | AbstractLUT) -> Self:
        """
        Perform in-place division of the *LUT* by the specified operand.

        Parameters
        ----------
        a
            Operand to divide the *LUT* by in-place.

        Returns
        -------
        :class:`colour.io.luts.lut.AbstractLUT`
            Current *LUT* instance with the division applied in-place.
        """

        return self.arithmetical_operation(a, "/", True)

    __itruediv__ = __idiv__
    __truediv__ = __div__

    def __pow__(self, a: ArrayLike | AbstractLUT) -> Self:
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

    def __ipow__(self, a: ArrayLike | AbstractLUT) -> Self:
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
    ) -> Self:
        """
        Perform the specified arithmetical operation with the :math:`a`
        operand.

        Execute the requested mathematical operation between this *LUT*
        instance and the specified operand. The operation can be performed
        either on a copy of the *LUT* or in-place on the current instance.
        This method must be reimplemented by sub-classes to handle their
        specific table structures.

        Parameters
        ----------
        a
            Operand for the arithmetical operation. Can be either a numeric
            array or another *LUT* instance with compatible dimensions.
        operation
            Arithmetical operation to perform. Supported operations are
            addition (``+``), subtraction (``-``), multiplication (``*``),
            division (``/``), and exponentiation (``**``).
        in_place
            Whether to perform the operation in-place on the current *LUT*
            instance (``True``) or on a copy (``False``).

        Returns
        -------
        :class:`colour.io.luts.lut.AbstractLUT`
            Modified *LUT* instance. If ``in_place`` is ``True``, returns
            the current instance after modification. If ``False``, returns
            a new modified copy.
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

        return ioperator(self.copy(), a)

    @abstractmethod
    def _validate_table(self, table: ArrayLike) -> NDArrayFloat:
        """
        Validate the specified table according to *LUT* dimensions.

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
        Validate specified domain according to *LUT* dimensions.

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

        While an explicit domain defines every single discrete sample::

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
        Generate a linear table of the specified size according to LUT dimensions.

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
            Copy of the LUT instance.
        """

        return deepcopy(self)

    @abstractmethod
    def invert(self, **kwargs: Any) -> AbstractLUT:
        """
        Compute and return an inverse copy of the *LUT*.

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
        Apply the *LUT* to the specified *RGB* colourspace array using the
        specified method.

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
            Extrapolator class type or object to use as extrapolating
            function.
        extrapolator_kwargs
            Arguments to use when instantiating or calling the extrapolating
            function.
        interpolator
            Interpolator class type or object to use as interpolating
            function.
        interpolator_kwargs
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
        Convert the *LUT* to the specified ``cls`` class instance.

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

        return LUT_to_LUT(self, cls, force_conversion, **kwargs)


class LUT1D(AbstractLUT):
    """
    Define the base class for a 1D *LUT*.

    A 1D (one-dimensional) lookup table provides a mapping function from
    input values to output values through interpolation of discrete table
    entries. This class is commonly used for tone mapping, gamma correction,
    and other single-channel transformations where the output depends solely
    on the input value.

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
        Validate that the specified table is a 1D array.

        Parameters
        ----------
        table
            Table to validate.

        Returns
        -------
        :class:`numpy.ndarray`
            Validated table as a :class:`numpy.ndarray` instance.
        """

        table = as_float_array(table)

        attest(len(table.shape) == 1, "The table must be a 1D array!")

        return table

    def _validate_domain(self, domain: ArrayLike) -> NDArrayFloat:
        """
        Validate specified domain.

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
        Generate a linear table with the specified number of output samples
        :math:`n`.

        The table contains linearly spaced values across the specified domain.
        If no domain is provided, the default domain [0, 1] is used.

        Parameters
        ----------
        size
            Number of samples in the output table. Default is 10.
        domain
            Domain boundaries of the table as a 2-element array [min, max]
            or an array of values whose minimum and maximum define the
            domain. Default is [0, 1].

        Returns
        -------
        :class:`numpy.ndarray`
            Linear table containing ``size`` evenly spaced samples across
            the specified domain.

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

        attest(is_numeric(size), "Linear table size must be a numeric!")

        return np.linspace(domain[0], domain[1], as_int_scalar(size))

    def invert(self, **kwargs: Any) -> LUT1D:  # noqa: ARG002
        """
        Compute and return an inverse copy of the *LUT*.

        Other Parameters
        ----------------
        kwargs
            Keywords arguments, only specified for signature compatibility
            with the :meth:`AbstractLUT.invert` method.

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

        return LUT1D(
            table=domain,
            name=f"{self.name} - Inverse",
            domain=self.table,
        )

    def apply(self, RGB: ArrayLike, **kwargs: Any) -> NDArrayFloat:
        """
        Apply the *LUT* to the specified *RGB* colourspace array using the
        specified method.

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
            Extrapolator class type or object to use as extrapolating
            function.
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

        *LUT* applied to the specified *RGB* colourspace in the forward
        direction:

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


class LUT3x1D(AbstractLUT):
    """
    Define the base class for a 3x1D *LUT*.

    A 3x1D (three-by-one-dimensional) lookup table applies independent
    transformations to each channel of a three-channel input. Each channel
    has its own 1D lookup table, enabling per-channel colour corrections
    and tone mapping operations.

    Parameters
    ----------
    table
        Underlying *LUT* table.
    name
        *LUT* name.
    domain
        *LUT* domain, also used to define the instantiation time default
        table domain.
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

    Instantiating a LUT using a custom table with 16x3 elements, custom
    name, custom domain and comments:

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
        Validate specified table is a 3x1D array.

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
        Validate the specified domain for the lookup table.

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
        Generate a linear table with the specified size and domain.

        The number of output samples :math:`n` is equal to ``size * 3`` or
        ``size[0] + size[1] + size[2]``.

        Parameters
        ----------
        size
            Expected table size, default to 10.
        domain
            Domain of the table.

        Returns
        -------
        :class:`numpy.ndarray`
            Linear table with ``size * 3`` or ``size[0] + size[1] +
            size[2]`` samples.

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

        size_array = np.tile(size, 3) if is_numeric(size) else as_int_array(size)

        R, G, B = tsplit(domain)

        samples = [
            np.linspace(a[0], a[1], size_array[i]) for i, a in enumerate([R, G, B])
        ]

        if len(np.unique(size_array)) != 1:
            runtime_warning(
                'Table is non uniform, axis will be padded with "NaNs" accordingly!'
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
        Compute and return an inverse copy of the *LUT*.

        Other Parameters
        ----------------
        kwargs
            Keywords arguments, only specified for signature compatibility with
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

        return LUT3x1D(
            table=tstack(domain),
            name=f"{self.name} - Inverse",
            domain=self.table,
        )

    def apply(self, RGB: ArrayLike, **kwargs: Any) -> NDArrayFloat:
        """
        Apply the *LUT* to the specified *RGB* colourspace array using the
        specified method.

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
            Extrapolator class type or object to use as extrapolating
            function.
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
            for a in zip((s_R, s_G, s_B), (R_t, G_t, B_t), (R, G, B), strict=True)
        ]

        return tstack(RGB_i)


class LUT3D(AbstractLUT):
    """
    Define the base class for a 3-dimensional lookup table (3D *LUT*).

    This class provides a foundation for working with 3D lookup tables,
    which map input colour values through a discretized 3D grid to output
    colour values. The table operates on three input channels
    simultaneously, making it suitable for RGB-to-RGB colour
    transformations and other tristimulus colour space operations.

    Parameters
    ----------
    table
        Underlying *LUT* table.
    name
        *LUT* name.
    domain
        *LUT* domain, also used to define the instantiation time default
        table domain.
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

    Instantiating a LUT using a custom table with 16x16x16x3 elements,
    custom name, custom domain and comments:

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
        Validate that the specified table is a 4D array with equal
        dimensions.

        Parameters
        ----------
        table
            Table to validate.

        Returns
        -------
        :class:`numpy.ndarray`
            Validated table as a :class:`numpy.ndarray` instance.
        """

        table = as_float_array(table)

        attest(len(table.shape) == 4, "The table must be a 4D array!")

        return table

    def _validate_domain(self, domain: ArrayLike) -> NDArrayFloat:
        """
        Validate the specified domain for the 3D lookup table.

        Parameters
        ----------
        domain
            Domain array to validate. Must be a 2D array with at least 2
            rows and exactly 3 columns.

        Returns
        -------
        :class:`numpy.ndarray`
            Validated domain as a :class:`numpy.ndarray` instance.

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

        While an explicit domain defines every single discrete sample::

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
        Generate a linear table with the specified size and domain.

        The number of output samples :math:`n` is equal to ``size**3 * 3`` or
        ``size[0] * size[1] * size[2] * 3``.

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

        return np.flip(
            np.reshape(
                np.transpose(np.meshgrid(*samples, indexing="ij")),
                np.hstack([np.flip(size_array, -1), 3]),
            ),
            -1,
        )

    @required("SciPy")
    def invert(self, **kwargs: Any) -> LUT3D:
        """
        Compute and return an inverse copy of the *LUT*.

        Other Parameters
        ----------------
        interpolator
            Interpolator class type or object to use as interpolating
            function.
        query_size
            Number of nearest neighbors to use for Shepard interpolation
            (inverse distance weighting). Default is 8, optimized for speed and
            quality. Higher values (16-32) may slightly improve smoothness but
            significantly increase computation time.
        gamma
            Gradient smoothness parameter for Shepard interpolation. Default is
            3.0 (optimized for smoothness). Controls the weight falloff rate in
            inverse distance weighting (:math:`w_i = 1/d_i^{1/gamma}`). Higher
            gamma values produce smoother gradients.

            - Default (3.0): Optimal smoothness with minimal artifacts
            - Lower values (1.5-2.0): Sharper transitions, faster computation,
              may increase banding artifacts
            - Very low values (0.5-1.0): Maximum sharpness, more localized
              interpolation, higher banding risk
        sigma
            Gaussian blur sigma for iterative adaptive smoothing.
            Default is 0.7. Smoothing is applied iteratively only to
            high-gradient regions (banding artifacts) identified using the
            percentile threshold, preserving quality in smooth regions.

            - Default (0.7): Optimal smoothing - reduces banding by ~38%
              (26 → 16 artifacts) while preserving corners
            - Higher values (0.8-0.9): More aggressive, may increase corner shift
            - Lower values (0.5-0.6): Gentler smoothing, better corner preservation
            - Set to 0.0 to disable adaptive smoothing entirely

            The iterative adaptive approach with gradient recomputation ensures
            clean LUTs remain unaffected while problematic regions receive
            targeted smoothing.
        tau
            Percentile threshold for identifying high-gradient regions (0-1).
            Default is 0.75 (75th percentile). Higher values mean fewer regions
            are smoothed (more selective), lower values mean more regions are
            smoothed (more aggressive).

            - Default (0.75): Smooths top 25% of gradient regions
            - Higher values (0.85-0.95): Very selective, minimal smoothing
            - Lower values (0.50-0.65): More aggressive, smooths more regions

            Only used when sigma > 0.
        iterations
            Number of iterative smoothing passes. Default is 10.
            Each iteration recomputes gradients and adapts smoothing to the
            evolving LUT state, providing better artifact reduction than a
            single strong blur.

            - Default (10): Optimal balance of quality and performance
            - Higher values (12-15): Slightly better artifact reduction, slower
            - Lower values (5-7): Faster, but fewer artifacts removed

            Only used when sigma > 0.
        oversampling
            Oversampling factor for building the KDTree. Default is 1.2.
            The optimal value is based on Jacobian analysis of the LUT
            transformation: the Jacobian matrix
            :math:`J = \\partial(output)/\\partial(input)` measures local
            volume distortion. When :math:`|J| < 1`, the LUT compresses space,
            requiring higher sampling density for accurate inversion.
            The factor 1.2 captures approximately 80% of the theoretical
            accuracy benefit at 30% of the computational cost. Values between
            1.0 (no oversampling) and 2.0 (diminishing returns) are supported.
        size
            Size of the inverse *LUT*. With the specified implementation,
            it is good practise to double the size of the inverse *LUT* to
            provide a smoother result. If ``size`` is not specified,
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

        from scipy.ndimage import gaussian_filter  # noqa: PLC0415
        from scipy.spatial import KDTree  # noqa: PLC0415

        if self.is_domain_explicit():
            error = 'Inverting a "LUT3D" with an explicit domain is not implemented!'

            raise NotImplementedError(error)

        interpolator = kwargs.get("interpolator", table_interpolation_trilinear)
        query_size = kwargs.get("query_size", 8)
        gamma = kwargs.get("gamma", 3.0)
        sigma = kwargs.get("sigma", 0.7)
        tau = kwargs.get("tau", 0.75)
        oversampling = kwargs.get("oversampling", 1.2)

        LUT = self.copy()
        source_size = LUT.size
        target_size = kwargs.get("size", (as_int(2 ** (np.sqrt(source_size) + 1) + 1)))
        sampling_size = int(target_size * oversampling)

        if target_size > 129:  # pragma: no cover
            usage_warning("LUT3D inverse computation time could be excessive!")

        # "LUT_t" is an intermediate LUT with oversampling to better capture
        # the LUT's transformation, especially in regions with high compression.
        # Sampling factor of 1.2 is based on Jacobian analysis: captures 80%
        # of theoretical benefit at 30% of computational cost.
        LUT_t = LUT3D(size=sampling_size, domain=LUT.domain)
        table = np.reshape(LUT_t.table, (-1, 3))
        LUT_t.table = LUT.apply(LUT_t.table, interpolator=interpolator)

        tree = KDTree(np.reshape(LUT_t.table, (-1, 3)))

        # "LUT_q" stores the inverse LUT with improved interpolation.
        # Query at the target resolution (output size).
        LUT_q = LUT3D(size=target_size, domain=LUT.domain)
        query_points = np.reshape(LUT_q.table, (-1, 3))

        distances, indices = tree.query(query_points, query_size)

        if query_size == 1:
            # Single nearest neighbor - no interpolation needed
            LUT_q.table = np.reshape(
                table[indices], (target_size, target_size, target_size, 3)
            )
        else:
            # Shepard's method (inverse distance weighting) for smooth interpolation.
            # Uses w_i = 1 / d_i^(1/gamma) where gamma controls the falloff rate.
            # Higher gamma (e.g., 2.0-4.0) creates smoother gradients by blending more
            # globally, while lower gamma (e.g., 0.25-0.5) creates sharper transitions.
            power = 1.0 / gamma
            distances = cast("NDArrayFloat", distances)
            weights = 1.0 / (distances + EPSILON) ** power
            weights = weights / np.sum(weights, axis=1, keepdims=True)

            # Weighted average: sum over neighbors dimension
            weighted_table = np.sum(table[indices] * weights[..., np.newaxis], axis=1)

            LUT_q.table = np.reshape(
                weighted_table,
                (target_size, target_size, target_size, 3),
            )

        # Apply iterative adaptive smoothing based on gradient magnitude.
        # Smooths only high-gradient regions (banding artifacts) while preserving
        # quality in smooth regions. Multiple iterations with gradient recomputation
        # allow smoothing to adapt as the LUT evolves.
        if sigma > 0:

            def extrapolate(data_3d: NDArrayFloat, pad_width: int) -> NDArrayFloat:
                """
                Pad the 3D array with linear extrapolation based on edge gradients.

                For each axis, extrapolate using:
                value[edge + i] = value[edge] + i * gradient

                This preserves boundary values much better than reflect/mirror modes.
                """

                result = data_3d

                for axis in range(3):
                    # Compute edge gradients
                    edge_lo = np.take(result, [0], axis=axis)
                    edge_hi = np.take(result, [-1], axis=axis)
                    grad_lo = edge_lo - np.take(result, [1], axis=axis)
                    grad_hi = edge_hi - np.take(result, [-2], axis=axis)

                    # Create padding using linear extrapolation
                    pad_lo = [edge_lo + (i + 1) * grad_lo for i in range(pad_width)]
                    pad_hi = [edge_hi + (i + 1) * grad_hi for i in range(pad_width)]

                    # Concatenate (reverse low padding)
                    result = np.concatenate([*pad_lo[::-1], result, *pad_hi], axis=axis)

                return result

            # Iterative smoothing: apply multiple passes with gradient recomputation.
            # Each iteration adapts to the evolving LUT state, providing better
            # artifact reduction than a single strong blur.
            iterations = kwargs.get("iterations", 10)
            pad_width = 10

            for _ in range(iterations):
                # Recompute gradient magnitude at each iteration to adapt
                # to the current LUT state
                gradient_magnitude = np.zeros(LUT_q.table.shape[:3])

                for i in range(3):
                    gx = np.gradient(LUT_q.table[..., i], axis=0)
                    gy = np.gradient(LUT_q.table[..., i], axis=1)
                    gz = np.gradient(LUT_q.table[..., i], axis=2)

                    gradient_magnitude += np.sqrt(gx**2 + gy**2 + gz**2)

                gradient_magnitude /= 3.0

                # Identify high-gradient regions using percentile threshold
                threshold = np.percentile(gradient_magnitude, tau * 100)

                # Apply Gaussian blur with linear extrapolation padding
                for i in range(3):
                    # Pad with linear extrapolation (recomputed each iteration)
                    table_p = extrapolate(LUT_q.table[..., i], pad_width)
                    # Filter the padded data
                    table_f = gaussian_filter(table_p, sigma=sigma)
                    # Un-pad
                    table_e = table_f[
                        pad_width:-pad_width,
                        pad_width:-pad_width,
                        pad_width:-pad_width,
                    ]
                    # Apply selectively to high-gradient regions only
                    LUT_q.table[..., i] = np.where(
                        gradient_magnitude > threshold,
                        table_e,
                        LUT_q.table[..., i],
                    )

        LUT_q.name = f"{self.name} - Inverse"

        return LUT_q

    def apply(self, RGB: ArrayLike, **kwargs: Any) -> NDArrayFloat:
        """
        Apply the *LUT* to the specified *RGB* colourspace array using the
        specified interpolation method.

        Parameters
        ----------
        RGB
            *RGB* colourspace array to apply the *LUT* onto.

        Other Parameters
        ----------------
        direction
            Whether the *LUT* should be applied in the forward or inverse
            direction.
        interpolator
            Interpolator object to use as the interpolating function.
        interpolator_kwargs
            Arguments to use when calling the interpolating function.
        query_size
            Number of points to query in the KDTree, with their mean
            computed to produce a smoother result.
        size
            Size of the inverse *LUT*. With the specified implementation,
            it is recommended to double the size of the inverse *LUT* to
            provide a smoother result. If ``size`` is not specified,
            :math:`2^{\\sqrt{size_{LUT}} + 1} + 1` will be used instead.

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
        ... # doctest: +ELLIPSIS +SKIP
        array([ 0.1799897...,  0.1796077...,  0.1795868...])
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

        return interpolator(tstack(RGB_l), LUT.table, **interpolator_kwargs)


def LUT_to_LUT(
    LUT: AbstractLUT,
    cls: Type[AbstractLUT],
    force_conversion: bool = False,
    **kwargs: Any,
) -> AbstractLUT:
    """
    Convert a specified *LUT* to the specified ``cls`` class instance.

    This function facilitates conversion between different LUT class types,
    including LUT1D, LUT3x1D, and LUT3D instances. Some conversions may be
    destructive and require explicit force conversion.

    Parameters
    ----------
    LUT
        *LUT* to convert.
    cls
        Target *LUT* class type for conversion.
    force_conversion
        Whether to force the conversion if it would be destructive.

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
    path = (ranks[LUT.__class__], ranks[cls])
    path_verbose = [f"{element}D" if element != 2 else "3x1D" for element in path]
    if path in ((1, 3), (2, 1), (2, 3), (3, 1), (3, 2)) and not force_conversion:
        error = (
            f'Conversion of a "LUT" {path_verbose[0]} to a "LUT" '
            f"{path_verbose[1]} is destructive, please use the "
            f'"force_conversion" argument to proceed!'
        )

        raise ValueError(error)

    suffix = f" - Converted {path_verbose[0]} to {path_verbose[1]}"
    name = f"{LUT.name}{suffix}"

    # Same dimension conversion, returning a copy.
    if len(set(path)) == 1:
        LUT = LUT.copy()
        LUT.name = name
    else:
        size = kwargs.get("size", 33 if cls is LUT3D else 10)
        kwargs.pop("size", None)

        channel_weights = as_float_array(kwargs.get("channel_weights", full(3, 1 / 3)))
        kwargs.pop("channel_weights", None)

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
        )

    return LUT
