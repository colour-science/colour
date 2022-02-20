"""
LUT Processing
==============

Defines the classes and definitions handling *LUT* processing:

-   :class:`colour.LUT1D`
-   :class:`colour.LUT3x1D`
-   :class:`colour.LUT3D`
-   :class:`colour.io.LUT_to_LUT`
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from copy import deepcopy
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

from colour.algebra import (
    Extrapolator,
    LinearInterpolator,
    linear_conversion,
    table_interpolation_trilinear,
)
from colour.hints import (
    Any,
    ArrayLike,
    Boolean,
    FloatingOrArrayLike,
    Integer,
    IntegerOrArrayLike,
    List,
    Literal,
    NDArray,
    Optional,
    Sequence,
    Type,
    Union,
    cast,
)
from colour.utilities import (
    as_float_array,
    as_int,
    as_int_array,
    as_int_scalar,
    attest,
    is_numeric,
    is_iterable,
    is_string,
    full,
    optional,
    required,
    runtime_warning,
    tsplit,
    tstack,
    usage_warning,
    validate_method,
)
from colour.utilities.deprecation import ObjectRenamed

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
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
        table: Optional[ArrayLike] = None,
        name: Optional[str] = None,
        dimensions: Optional[Integer] = None,
        domain: Optional[ArrayLike] = None,
        size: Optional[IntegerOrArrayLike] = None,
        comments: Optional[Sequence] = None,
    ):
        self._name: str = f"Unity {size!r}" if table is None else f"{id(self)}"
        self.name = optional(name, self._name)
        self._dimensions = optional(dimensions, 0)
        self._table: NDArray = self.linear_table(
            cast(ArrayLike, optional(size, 0)),
            cast(ArrayLike, optional(domain, np.array([]))),
        )
        self.table = cast(
            ArrayLike, optional(table, self._table)
        )  # type: ignore[assignment]
        # TODO: Remove pragma when https://github.com/python/mypy/issues/3004
        # is resolved.
        self._domain: NDArray = np.array([])
        self.domain = cast(
            ArrayLike, optional(domain, self._domain)
        )  # type: ignore[assignment]
        self._comments: List = []
        self.comments = cast(
            ArrayLike, optional(comments, self._comments)
        )  # type: ignore[assignment]

    @property
    def table(self) -> NDArray:
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
    def domain(self) -> NDArray:
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

        # pylint: disable=E1121
        self._domain = self._validate_domain(value)

    @property
    def dimensions(self) -> Integer:
        """
        Getter property for the *LUT* dimensions.

        Returns
        -------
        :class:`numpy.integer`
            *LUT* dimensions.
        """

        return self._dimensions

    @property
    def size(self) -> Integer:
        """
        Getter property for the *LUT* size.

        Returns
        -------
        :class:`numpy.integer`
            *LUT* size.
        """

        return self._table.shape[0]

    @property
    def comments(self) -> List:
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

        def _indent_array(a: ArrayLike) -> str:
            """Indent given array string representation."""

            return str(a).replace(" [", " " * 14 + "[")

        comments = "\n".join(
            [
                f"Comment {str(i + 1).zfill(2)} : {comment}"
                for i, comment in enumerate(self.comments)
            ]
        )
        comments = f"\n{comments}" if comments else ""

        underline = "-" * (len(self.__class__.__name__) + 3 + len(self.name))

        return (
            f"{self.__class__.__name__} - {self.name}\n"
            f"{underline}\n\n"
            f"Dimensions : {self.dimensions}\n"
            f"Domain     : {_indent_array(self.domain)}\n"
            f'Size       : {str(self.table.shape).replace("L", "")}'
            f"{comments}"
        )

    def __repr__(self) -> str:
        """
        Return an evaluable string representation of the *LUT*.

        Returns
        -------
        :class:`str`
            Evaluable string representation.
        """

        representation = repr(self.table)
        representation = representation.replace(
            "array", self.__class__.__name__
        )
        representation = representation.replace(
            "       [", f"{' ' * (len(self.__class__.__name__) + 2)}["
        )

        domain = repr(self.domain).replace("array(", "").replace(")", "")
        domain = domain.replace(
            "       [", f"{' ' * (len(self.__class__.__name__) + 9)}["
        )

        indentation = " " * (len(self.__class__.__name__) + 1)
        comments = (
            f",\n\n{indentation}comments={repr(self.comments)}"
            if self.comments
            else ""
        )

        return (
            f"{representation[:-1]},\n"
            f"{indentation}name='{self.name}',\n"
            f"{indentation}domain={domain}"
            f"{comments})"
        )

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

        if isinstance(other, AbstractLUT):
            if all(
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

    def __add__(
        self, a: Union[FloatingOrArrayLike, AbstractLUT]
    ) -> AbstractLUT:
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

    def __iadd__(
        self, a: Union[FloatingOrArrayLike, AbstractLUT]
    ) -> AbstractLUT:
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

    def __sub__(
        self, a: Union[FloatingOrArrayLike, AbstractLUT]
    ) -> AbstractLUT:
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

    def __isub__(
        self, a: Union[FloatingOrArrayLike, AbstractLUT]
    ) -> AbstractLUT:
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

    def __mul__(
        self, a: Union[FloatingOrArrayLike, AbstractLUT]
    ) -> AbstractLUT:
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

    def __imul__(
        self, a: Union[FloatingOrArrayLike, AbstractLUT]
    ) -> AbstractLUT:
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

    def __div__(
        self, a: Union[FloatingOrArrayLike, AbstractLUT]
    ) -> AbstractLUT:
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

    def __idiv__(
        self, a: Union[FloatingOrArrayLike, AbstractLUT]
    ) -> AbstractLUT:
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

    def __pow__(
        self, a: Union[FloatingOrArrayLike, AbstractLUT]
    ) -> AbstractLUT:
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

    def __ipow__(
        self, a: Union[FloatingOrArrayLike, AbstractLUT]
    ) -> AbstractLUT:
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
        a: Union[FloatingOrArrayLike, AbstractLUT],
        operation: Literal["+", "-", "*", "/", "**"],
        in_place: Boolean = False,
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
            if isinstance(a, AbstractLUT):
                operand = a.table
            else:
                operand = as_float_array(a)

            self.table = operator(self.table, operand)

            return self
        else:
            copy = ioperator(self.copy(), a)

            return copy

    @abstractmethod
    def _validate_table(self, table: ArrayLike) -> NDArray:
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

        pass

    @abstractmethod
    def _validate_domain(self, domain: ArrayLike) -> NDArray:
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

        pass

    @abstractmethod
    def is_domain_explicit(self) -> Boolean:
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

        pass

    @staticmethod
    @abstractmethod
    def linear_table(
        size: Optional[IntegerOrArrayLike] = None,
        domain: Optional[ArrayLike] = None,
    ) -> NDArray:
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

        pass

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

        pass

    @abstractmethod
    def apply(self, RGB: ArrayLike, **kwargs: Any) -> NDArray:
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
        interpolator_kwargs
            Arguments to use when instantiating or calling the interpolating
            function.

        Returns
        -------
        :class:`numpy.ndarray`
            Interpolated *RGB* colourspace array.
        """

        pass

    @abstractmethod
    def convert(
        self,
        cls: Type[AbstractLUT],
        force_conversion: Boolean = False,
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

        pass


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
    -   :meth:`~colour.LUT1D.convert`

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
    >>> print(LUT1D(
    ...     spow(LUT1D.linear_table(16, domain), 1 / 2.2),
    ...     'My LUT',
    ...     domain,
    ...     comments=['A first comment.', 'A second comment.']))
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
        table: Optional[ArrayLike] = None,
        name: Optional[str] = None,
        domain: Optional[ArrayLike] = None,
        size: Optional[IntegerOrArrayLike] = None,
        comments: Optional[Sequence] = None,
    ):

        domain = as_float_array(
            cast(ArrayLike, optional(domain, np.array([0, 1])))
        )
        size = cast(Integer, optional(size, 10))

        super().__init__(table, name, 1, domain, size, comments)

    def _validate_table(self, table: ArrayLike) -> NDArray:
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

    def _validate_domain(self, domain: ArrayLike) -> NDArray:
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

    def is_domain_explicit(self) -> Boolean:
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
        size: Optional[IntegerOrArrayLike] = None,
        domain: Optional[ArrayLike] = None,
    ) -> NDArray:
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

        size = cast(Integer, optional(size, 10))
        domain = as_float_array(
            cast(ArrayLike, optional(domain, np.array([0, 1])))
        )

        if len(domain) != 2:
            return domain
        else:
            attest(is_numeric(size), "Linear table size must be a numeric!")

            return np.linspace(domain[0], domain[1], as_int_scalar(size))

    def invert(self, **kwargs: Any) -> LUT1D:
        """
        Compute and returns an inverse copy of the *LUT*.

        Other Parameters
        ----------------
        kwargs
            Keywords arguments, only given for signature compatibility with
            the :meth:`AbstractLUT.invert` method.

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

    def apply(self, RGB: ArrayLike, **kwargs: Any) -> NDArray:
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

        >>> LUT.apply(LUT.apply(RGB), direction='Inverse')
        ... # doctest: +ELLIPSIS
        array([ 0.18...,  0.18...,  0.18...])
        """

        direction = validate_method(
            kwargs.get("direction", "Forward"), ["Forward", "Inverse"]
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

    def convert(
        self,
        cls: Type[AbstractLUT],
        force_conversion: Boolean = False,
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
            Expected table size in case of an upcast to a :class:`LUT3D` class
            instance.

        Returns
        -------
        :class:`colour.LUT1D` or :class:`colour.LUT3x1D` or \
:class:`colour.LUT3D`
            Converted *LUT* class instance.

        Warnings
        --------
        Some conversions are destructive and raise a :class:`ValueError`
        exception by default.

        Raises
        ------
        ValueError
            If the conversion is destructive.

        Examples
        --------
        >>> LUT = LUT1D()
        >>> print(LUT.convert(LUT1D))
        LUT1D - Unity 10 - Converted 1D to 1D
        -------------------------------------
        <BLANKLINE>
        Dimensions : 1
        Domain     : [ 0.  1.]
        Size       : (10,)
        >>> print(LUT.convert(LUT3x1D))
        LUT3x1D - Unity 10 - Converted 1D to 3x1D
        -----------------------------------------
        <BLANKLINE>
        Dimensions : 2
        Domain     : [[ 0.  0.  0.]
                      [ 1.  1.  1.]]
        Size       : (10, 3)
        >>> print(LUT.convert(LUT3D, force_conversion=True))
        LUT3D - Unity 10 - Converted 1D to 3D
        -------------------------------------
        <BLANKLINE>
        Dimensions : 3
        Domain     : [[ 0.  0.  0.]
                      [ 1.  1.  1.]]
        Size       : (33, 33, 33, 3)
        """

        return LUT_to_LUT(self, cls, force_conversion, **kwargs)

    # ------------------------------------------------------------------------#
    # ---              API Changes and Deprecation Management              ---#
    # ------------------------------------------------------------------------#
    def as_LUT(
        self,
        cls: Type[AbstractLUT],
        force_conversion: Boolean = False,
        **kwargs: Any,
    ) -> AbstractLUT:  # pragma: no cover  # noqa: D102
        # Docstrings are omitted for documentation purposes.
        usage_warning(
            str(
                ObjectRenamed(
                    "LUT1D.as_LUT",
                    "LUT1D.convert",
                )
            )
        )

        return self.convert(cls, force_conversion, **kwargs)


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
    -   :meth:`~colour.LUT3x1D.convert`

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
    >>> print(LUT3x1D(
    ...     spow(LUT3x1D.linear_table(16), 1 / 2.2),
    ...     'My LUT',
    ...     domain,
    ...     comments=['A first comment.', 'A second comment.']))
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
        table: Optional[ArrayLike] = None,
        name: Optional[str] = None,
        domain: Optional[ArrayLike] = None,
        size: Optional[IntegerOrArrayLike] = None,
        comments: Optional[Sequence] = None,
    ):
        domain = cast(
            ArrayLike, optional(domain, np.array([[0, 0, 0], [1, 1, 1]]))
        )
        size = cast(Integer, optional(size, 10))

        super().__init__(table, name, 2, domain, size, comments)

    def _validate_table(self, table: ArrayLike) -> NDArray:
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

    def _validate_domain(self, domain: ArrayLike) -> NDArray:
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

        attest(
            domain.shape[1] == 3, "The domain column count must be equal to 3!"
        )

        return domain

    def is_domain_explicit(self) -> Boolean:
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
        size: Optional[IntegerOrArrayLike] = None,
        domain: Optional[ArrayLike] = None,
    ) -> NDArray:
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
        >>> LUT3x1D.linear_table(
        ...     5, np.array([[-0.1, -0.2, -0.4], [1.5, 3.0, 6.0]]))
        array([[-0.1, -0.2, -0.4],
               [ 0.3,  0.6,  1.2],
               [ 0.7,  1.4,  2.8],
               [ 1.1,  2.2,  4.4],
               [ 1.5,  3. ,  6. ]])
        >>> LUT3x1D.linear_table(
        ...     np.array([5, 3, 2]),
        ...     np.array([[-0.1, -0.2, -0.4], [1.5, 3.0, 6.0]]))
        array([[-0.1, -0.2, -0.4],
               [ 0.3,  1.4,  6. ],
               [ 0.7,  3. ,  nan],
               [ 1.1,  nan,  nan],
               [ 1.5,  nan,  nan]])
        >>> domain = np.array([[-0.1, -0.2, -0.4],
        ...                    [0.3, 1.4, 6.0],
        ...                    [0.7, 3.0, np.nan],
        ...                    [1.1, np.nan, np.nan],
        ...                    [1.5, np.nan, np.nan]])
        >>> LUT3x1D.linear_table(domain=domain)
        array([[-0.1, -0.2, -0.4],
               [ 0.3,  1.4,  6. ],
               [ 0.7,  3. ,  nan],
               [ 1.1,  nan,  nan],
               [ 1.5,  nan,  nan]])
        """

        size = cast(Integer, optional(size, 10))
        domain = as_float_array(
            cast(ArrayLike, optional(domain, np.array([[0, 0, 0], [1, 1, 1]])))
        )

        if domain.shape != (2, 3):
            return domain
        else:
            if is_numeric(size):
                size_array = np.tile(size, 3)
            else:
                size_array = as_int_array(size)

            R, G, B = tsplit(domain)

            samples = [
                np.linspace(a[0], a[1], size_array[i])
                for i, a in enumerate([R, G, B])
            ]

            if not len(np.unique(size_array)) == 1:
                runtime_warning(
                    "Table is non uniform, axis will be "
                    'padded with "NaNs" accordingly!'
                )

                samples = [
                    np.pad(
                        axis,
                        (0, np.max(size_array) - len(axis)),
                        mode="constant",
                        constant_values=np.nan,
                    )
                    for axis in samples
                ]

            return tstack(samples)

    def invert(self, **kwargs: Any) -> LUT3x1D:
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
            domain = [
                np.linspace(domain_min[i], domain_max[i], size)
                for i in range(3)
            ]

        LUT_i = LUT3x1D(
            table=tstack(domain),
            name=f"{self.name} - Inverse",
            domain=self.table,
        )

        return LUT_i

    def apply(self, RGB: ArrayLike, **kwargs: Any) -> NDArray:
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
        >>> LUT = LUT3x1D(LUT3x1D.linear_table() ** (1 / 2.2))
        >>> RGB = np.array([0.18, 0.18, 0.18])
        >>> LUT.apply(RGB)  # doctest: +ELLIPSIS
        array([ 0.4529220...,  0.4529220...,  0.4529220...])
        >>> LUT.apply(LUT.apply(RGB), direction='Inverse')
        ... # doctest: +ELLIPSIS
        array([ 0.18...,  0.18...,  0.18...])
        >>> from colour.algebra import spow
        >>> domain = np.array([[-0.1, -0.2, -0.4], [1.5, 3.0, 6.0]])
        >>> table = spow(LUT3x1D.linear_table(domain=domain), 1 / 2.2)
        >>> LUT = LUT3x1D(table, domain=domain)
        >>> RGB = np.array([0.18, 0.18, 0.18])
        >>> LUT.apply(RGB)  # doctest: +ELLIPSIS
        array([ 0.4423903...,  0.4503801...,  0.3581625...])
        >>> domain = np.array([[-0.1, -0.2, -0.4],
        ...                    [0.3, 1.4, 6.0],
        ...                    [0.7, 3.0, np.nan],
        ...                    [1.1, np.nan, np.nan],
        ...                    [1.5, np.nan, np.nan]])
        >>> table = spow(LUT3x1D.linear_table(domain=domain), 1 / 2.2)
        >>> LUT = LUT3x1D(table, domain=domain)
        >>> RGB = np.array([0.18, 0.18, 0.18])
        >>> LUT.apply(RGB)  # doctest: +ELLIPSIS
        array([ 0.2996370..., -0.0901332..., -0.3949770...])
        """

        direction = validate_method(
            kwargs.get("direction", "Forward"), ["Forward", "Inverse"]
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
                np.linspace(domain_min[i], domain_max[i], size)
                for i in range(3)
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

    def convert(
        self,
        cls: Type[AbstractLUT],
        force_conversion: Boolean = False,
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
            Expected table size in case of an upcast to a :class:`LUT3D` class
            instance.

        Returns
        -------
        :class:`colour.LUT1D` or :class:`colour.LUT3x1D` or \
:class:`colour.LUT3D`
            Converted *LUT* class instance.

        Warnings
        --------
        Some conversions are destructive and raise a :class:`ValueError`
        exception by default.

        Raises
        ------
        ValueError
            If the conversion is destructive.

        Examples
        --------
        >>> LUT = LUT3x1D()
        >>> print(LUT.convert(LUT1D, force_conversion=True))
        LUT1D - Unity 10 - Converted 3x1D to 1D
        ---------------------------------------
        <BLANKLINE>
        Dimensions : 1
        Domain     : [ 0.  1.]
        Size       : (10,)
        >>> print(LUT.convert(LUT3x1D))
        LUT3x1D - Unity 10 - Converted 3x1D to 3x1D
        -------------------------------------------
        <BLANKLINE>
        Dimensions : 2
        Domain     : [[ 0.  0.  0.]
                      [ 1.  1.  1.]]
        Size       : (10, 3)
        >>> print(LUT.convert(LUT3D, force_conversion=True))
        LUT3D - Unity 10 - Converted 3x1D to 3D
        ---------------------------------------
        <BLANKLINE>
        Dimensions : 3
        Domain     : [[ 0.  0.  0.]
                      [ 1.  1.  1.]]
        Size       : (33, 33, 33, 3)
        """

        return LUT_to_LUT(self, cls, force_conversion, **kwargs)

    # ------------------------------------------------------------------------#
    # ---              API Changes and Deprecation Management              ---#
    # ------------------------------------------------------------------------#
    def as_LUT(
        self,
        cls: Type[AbstractLUT],
        force_conversion: Boolean = False,
        **kwargs: Any,
    ) -> AbstractLUT:  # pragma: no cover  # noqa: D102
        # Docstrings are omitted for documentation purposes.
        usage_warning(
            str(
                ObjectRenamed(
                    "LUT3x1D.as_LUT",
                    "LUT3x1D.convert",
                )
            )
        )

        return self.convert(cls, force_conversion, **kwargs)


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
    -   :meth:`~colour.LUT3D.convert`

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
    >>> print(LUT3D(
    ...     spow(LUT3D.linear_table(16), 1 / 2.2),
    ...     'My LUT',
    ...     domain,
    ...     comments=['A first comment.', 'A second comment.']))
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
        table: Optional[ArrayLike] = None,
        name: Optional[str] = None,
        domain: Optional[ArrayLike] = None,
        size: Optional[IntegerOrArrayLike] = None,
        comments: Optional[Sequence] = None,
    ):
        domain = cast(
            ArrayLike, optional(domain, np.array([[0, 0, 0], [1, 1, 1]]))
        )
        size = cast(Integer, optional(size, 33))

        super().__init__(table, name, 3, domain, size, comments)

    def _validate_table(self, table: ArrayLike) -> NDArray:
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

    def _validate_domain(self, domain: ArrayLike) -> NDArray:
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

        attest(
            domain.shape[1] == 3, "The domain column count must be equal to 3!"
        )

        return domain

    def is_domain_explicit(self) -> Boolean:
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
        >>> domain = np.array([[-0.1, -0.2, -0.4],
        ...                    [0.7, 1.4, 6.0],
        ...                    [1.5, 3.0, np.nan]])
        >>> LUT3D(domain=domain).is_domain_explicit()
        True
        """

        return self.domain.shape != (2, 3)

    @staticmethod
    def linear_table(
        size: Optional[IntegerOrArrayLike] = None,
        domain: Optional[ArrayLike] = None,
    ) -> NDArray:
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
        >>> LUT3D.linear_table(
        ...     3, np.array([[-0.1, -0.2, -0.4], [1.5, 3.0, 6.0]]))
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
        ...     np.array([[-0.1, -0.2, -0.4], [1.5, 3.0, 6.0]]))
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
        >>> domain = np.array([[-0.1, -0.2, -0.4],
        ...                    [0.7, 1.4, 6.0],
        ...                    [1.5, 3.0, np.nan]])
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

        size = cast(Integer, optional(size, 33))
        domain = as_float_array(
            cast(ArrayLike, optional(domain, np.array([[0, 0, 0], [1, 1, 1]])))
        )

        if domain.shape != (2, 3):
            samples = list(
                np.flip(
                    [
                        axes[: (~np.isnan(axes)).cumsum().argmax() + 1]
                        for axes in np.transpose(domain)
                    ],
                    -1,
                )
            )
            size_array = as_int_array([len(axes) for axes in samples])
        else:
            if is_numeric(size):
                size_array = np.tile(size, 3)
            else:
                size_array = as_int_array(size)

            R, G, B = tsplit(domain)

            size_array = np.flip(size_array, -1)
            samples = [
                np.linspace(a[0], a[1], size_array[i])
                for i, a in enumerate([B, G, R])
            ]

        table = np.flip(
            np.transpose(np.meshgrid(*samples, indexing="ij")).reshape(
                np.hstack([np.flip(size_array, -1), 3])
            ),
            -1,
        )

        return table

    @required("Scikit-Learn")
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

        # TODO: Drop "sklearn" requirement whenever "Scipy" 1.7 can be
        # defined as the minimal version.
        from sklearn.neighbors import KDTree

        interpolator = kwargs.get(
            "interpolator", table_interpolation_trilinear
        )
        extrapolate = kwargs.get("extrapolate", False)
        query_size = kwargs.get("query_size", 3)

        LUT = self.copy()
        source_size = LUT.size
        target_size = kwargs.get(
            "size", (as_int(2 ** (np.sqrt(source_size) + 1) + 1))
        )

        if target_size > 129:
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
            LUT_q.table = table[query].reshape(
                [target_size, target_size, target_size, 3]
            )
        else:
            LUT_q.table = np.mean(table[query], axis=-2).reshape(
                [target_size, target_size, target_size, 3]
            )

        # "LUT_i" is the final inverse LUT generated by applying "LUT_q" on
        # an identity LUT at the target size.
        LUT_i = LUT3D(size=target_size, domain=LUT.domain)
        LUT_i.table = LUT_q.apply(LUT_i.table, interpolator=interpolator)

        LUT_i.name = f"{self.name} - Inverse"

        return LUT_i

    def apply(self, RGB: ArrayLike, **kwargs: Any) -> NDArray:
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
        >>> LUT.apply(LUT.apply(RGB), direction='Inverse')
        ... # doctest: +ELLIPSIS
        array([ 0.1781995...,  0.1809414...,  0.1809513...])
        >>> from colour.algebra import spow
        >>> domain = np.array([[-0.1, -0.2, -0.4],
        ...                    [0.3, 1.4, 6.0],
        ...                    [0.7, 3.0, np.nan],
        ...                    [1.1, np.nan, np.nan],
        ...                    [1.5, np.nan, np.nan]])
        >>> table = spow(LUT3D.linear_table(domain=domain), 1 / 2.2)
        >>> LUT = LUT3D(table, domain=domain)
        >>> RGB = np.array([0.18, 0.18, 0.18])
        >>> LUT.apply(RGB)  # doctest: +ELLIPSIS
        array([ 0.2996370..., -0.0901332..., -0.3949770...])
        """

        direction = validate_method(
            kwargs.get("direction", "Forward"), ["Forward", "Inverse"]
        )

        interpolator = kwargs.get(
            "interpolator", table_interpolation_trilinear
        )
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

    def convert(
        self,
        cls: Type[AbstractLUT],
        force_conversion: Boolean = False,
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
            Expected table size in case of a downcast from a :class:`LUT3D`
            class instance.

        Returns
        -------
        :class:`colour.LUT1D` or :class:`colour.LUT3x1D` or \
:class:`colour.LUT3D`
            Converted *LUT* class instance.

        Warnings
        --------
        Some conversions are destructive and raise a :class:`ValueError`
        exception by default.

        Raises
        ------
        ValueError
            If the conversion is destructive.

        Examples
        --------
        >>> LUT = LUT3D()
        >>> print(LUT.convert(LUT1D, force_conversion=True))
        LUT1D - Unity 33 - Converted 3D to 1D
        -------------------------------------
        <BLANKLINE>
        Dimensions : 1
        Domain     : [ 0.  1.]
        Size       : (10,)
        >>> print(LUT.convert(LUT3x1D, force_conversion=True))
        LUT3x1D - Unity 33 - Converted 3D to 3x1D
        -----------------------------------------
        <BLANKLINE>
        Dimensions : 2
        Domain     : [[ 0.  0.  0.]
                      [ 1.  1.  1.]]
        Size       : (10, 3)
        >>> print(LUT.convert(LUT3D))
        LUT3D - Unity 33 - Converted 3D to 3D
        -------------------------------------
        <BLANKLINE>
        Dimensions : 3
        Domain     : [[ 0.  0.  0.]
                      [ 1.  1.  1.]]
        Size       : (33, 33, 33, 3)
        """

        return LUT_to_LUT(self, cls, force_conversion, **kwargs)

    # ------------------------------------------------------------------------#
    # ---              API Changes and Deprecation Management              ---#
    # ------------------------------------------------------------------------#
    def as_LUT(
        self,
        cls: Type[AbstractLUT],
        force_conversion: Boolean = False,
        **kwargs: Any,
    ) -> AbstractLUT:  # pragma: no cover  # noqa: D102
        # Docstrings are omitted for documentation purposes.
        usage_warning(
            str(
                ObjectRenamed(
                    "LUT3D.as_LUT",
                    "LUT3D.convert",
                )
            )
        )

        return self.convert(cls, force_conversion, **kwargs)


def LUT_to_LUT(
    LUT,
    cls: Type[AbstractLUT],
    force_conversion: Boolean = False,
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
    path = (ranks[LUT.__class__], ranks[cls])
    path_verbose = [
        f"{element}D" if element != 2 else "3x1D" for element in path
    ]
    if path in ((1, 3), (2, 1), (2, 3), (3, 1), (3, 2)):
        if not force_conversion:
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

        channel_weights = as_float_array(
            kwargs.get("channel_weights", full(3, 1 / 3))
        )
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
        )

    return LUT
