"""
Abstract Continuous Function
============================

Define an abstract base class for continuous mathematical functions with
support for interpolation, extrapolation, and arithmetical operations:

-   :class:`colour.continuous.AbstractContinuousFunction`
"""

from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np

if typing.TYPE_CHECKING:
    from colour.hints import (
        ArrayLike,
        Callable,
        DTypeFloat,
        Generator,
        Literal,
        NDArrayFloat,
        ProtocolExtrapolator,
        ProtocolInterpolator,
        Real,
        Self,
        Type,
    )

from colour.utilities import (
    MixinCallback,
    as_float,
    attest,
    closest,
    is_uniform,
    optional,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "AbstractContinuousFunction",
]


class AbstractContinuousFunction(ABC, MixinCallback):
    """
    Define the base class for an abstract continuous function.

    This is an :class:`ABCMeta` abstract class that must be inherited by
    sub-classes.

    The sub-classes are expected to implement the
    :meth:`colour.continuous.AbstractContinuousFunction.function` method so
    that evaluating the function for any independent domain variable
    :math:`x \\in\\mathbb{R}` returns a corresponding range variable
    :math:`y \\in\\mathbb{R}`. A conventional implementation adopts an
    interpolating function encapsulated inside an extrapolating function.
    The resulting function independent domain, stored as discrete values in
    the :attr:`colour.continuous.AbstractContinuousFunction.domain` attribute,
    corresponds with the function dependent and already known range stored in
    the :attr:`colour.continuous.AbstractContinuousFunction.range` property.

    Parameters
    ----------
    name
        Continuous function name.

    Attributes
    ----------
    -   :attr:`~colour.continuous.AbstractContinuousFunction.name`
    -   :attr:`~colour.continuous.AbstractContinuousFunction.dtype`
    -   :attr:`~colour.continuous.AbstractContinuousFunction.domain`
    -   :attr:`~colour.continuous.AbstractContinuousFunction.range`
    -   :attr:`~colour.continuous.AbstractContinuousFunction.interpolator`
    -   :attr:`~colour.continuous.\
AbstractContinuousFunction.interpolator_kwargs`
    -   :attr:`~colour.continuous.AbstractContinuousFunction.extrapolator`
    -   :attr:`~colour.continuous.\
AbstractContinuousFunction.extrapolator_kwargs`
    -   :attr:`~colour.continuous.AbstractContinuousFunction.function`

    Methods
    -------
    -   :meth:`~colour.continuous.AbstractContinuousFunction.__init__`
    -   :meth:`~colour.continuous.AbstractContinuousFunction.__str__`
    -   :meth:`~colour.continuous.AbstractContinuousFunction.__repr__`
    -   :meth:`~colour.continuous.AbstractContinuousFunction.__hash__`
    -   :meth:`~colour.continuous.AbstractContinuousFunction.__getitem__`
    -   :meth:`~colour.continuous.AbstractContinuousFunction.__setitem__`
    -   :meth:`~colour.continuous.AbstractContinuousFunction.__contains__`
    -   :meth:`~colour.continuous.AbstractContinuousFunction.__iter__`
    -   :meth:`~colour.continuous.AbstractContinuousFunction.__len__`
    -   :meth:`~colour.continuous.AbstractContinuousFunction.__eq__`
    -   :meth:`~colour.continuous.AbstractContinuousFunction.__ne__`
    -   :meth:`~colour.continuous.AbstractContinuousFunction.__iadd__`
    -   :meth:`~colour.continuous.AbstractContinuousFunction.__add__`
    -   :meth:`~colour.continuous.AbstractContinuousFunction.__isub__`
    -   :meth:`~colour.continuous.AbstractContinuousFunction.__sub__`
    -   :meth:`~colour.continuous.AbstractContinuousFunction.__imul__`
    -   :meth:`~colour.continuous.AbstractContinuousFunction.__mul__`
    -   :meth:`~colour.continuous.AbstractContinuousFunction.__idiv__`
    -   :meth:`~colour.continuous.AbstractContinuousFunction.__div__`
    -   :meth:`~colour.continuous.AbstractContinuousFunction.__ipow__`
    -   :meth:`~colour.continuous.AbstractContinuousFunction.__pow__`
    -   :meth:`~colour.continuous.AbstractContinuousFunction.\
arithmetical_operation`
    -   :meth:`~colour.continuous.AbstractContinuousFunction.fill_nan`
    -   :meth:`~colour.continuous.AbstractContinuousFunction.domain_distance`
    -   :meth:`~colour.continuous.AbstractContinuousFunction.is_uniform`
    -   :meth:`~colour.continuous.AbstractContinuousFunction.copy`
    """

    def __init__(self, name: str | None = None) -> None:
        super().__init__()

        self._name: str = f"{self.__class__.__name__} ({id(self)})"
        self.name = optional(name, self._name)

    @property
    def name(self) -> str:
        """
        Getter and setter for the abstract continuous function name.

        Parameters
        ----------
        value
            Value to set the abstract continuous function name with.

        Returns
        -------
        :class:`str`
            Abstract continuous function name.
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
    @abstractmethod
    def dtype(self) -> Type[DTypeFloat]:
        """
        Getter and setter for the abstract continuous function dtype.

        This property must be reimplemented by sub-classes.

        Parameters
        ----------
        value
            Value to set the abstract continuous function dtype with.

        Returns
        -------
        Type[DTypeFloat]
            Abstract continuous function dtype.
        """

        ...  # pragma: no cover

    @dtype.setter
    @abstractmethod
    def dtype(self, value: Type[DTypeFloat]) -> None:
        """
        Setter for the **self.dtype** property, must be reimplemented by
        sub-classes.
        """

        ...  # pragma: no cover

    @property
    @abstractmethod
    def domain(self) -> NDArrayFloat:
        """
        Getter and setter for the abstract continuous function's independent
        domain variable :math:`x`.

        This property must be reimplemented by sub-classes.

        Parameters
        ----------
        value
            Value to set the abstract continuous function independent domain
            variable :math:`x` with.

        Returns
        -------
        :class:`numpy.ndarray`
            Abstract continuous function independent domain variable
            :math:`x`.
        """

        ...  # pragma: no cover

    @domain.setter
    @abstractmethod
    def domain(self, value: ArrayLike) -> None:
        """
        Setter for the **self.domain** property, must be reimplemented by
        sub-classes.
        """

        ...  # pragma: no cover

    @property
    @abstractmethod
    def range(self) -> NDArrayFloat:
        """
        Getter and setter for the abstract continuous function's range
        variable :math:`y`.

        This property must be reimplemented by sub-classes.

        Parameters
        ----------
        value
            Value to set the abstract continuous function's range variable
            :math:`y` with.

        Returns
        -------
        :class:`numpy.ndarray`
            Abstract continuous function's range variable :math:`y`.
        """

        ...  # pragma: no cover

    @range.setter
    @abstractmethod
    def range(self, value: ArrayLike) -> None:
        """
        Setter for the **self.range** property, must be reimplemented by
        sub-classes.
        """

        ...  # pragma: no cover

    @property
    @abstractmethod
    def interpolator(self) -> Type[ProtocolInterpolator]:
        """
        Getter and setter for the abstract continuous function interpolator
        type.

        This property must be reimplemented by sub-classes.

        Parameters
        ----------
        value
            Value to set the abstract continuous function interpolator type
            with.

        Returns
        -------
        Type[ProtocolInterpolator]
            Abstract continuous function interpolator type.
        """

        ...  # pragma: no cover

    @interpolator.setter
    @abstractmethod
    def interpolator(self, value: Type[ProtocolInterpolator]) -> None:
        """
        Setter for the **self.interpolator** property, must be reimplemented by
        sub-classes.
        """

        ...  # pragma: no cover

    @property
    @abstractmethod
    def interpolator_kwargs(self) -> dict:
        """
        Getter and setter for the interpolator instantiation time arguments.

        This property must be reimplemented by sub-classes.

        Parameters
        ----------
        value
            Value to set the abstract continuous function interpolator
            instantiation time arguments to.

        Returns
        -------
        :class:`dict`
            Abstract continuous function interpolator instantiation time
            arguments.
        """

        ...  # pragma: no cover

    @interpolator_kwargs.setter
    @abstractmethod
    def interpolator_kwargs(self, value: dict) -> None:
        """
        Setter for the **self.interpolator_kwargs** property, must be
        reimplemented by sub-classes.
        """

        ...  # pragma: no cover

    @property
    @abstractmethod
    def extrapolator(self) -> Type[ProtocolExtrapolator]:
        """
        Getter and setter for the abstract continuous function extrapolator
        type.

        This property must be reimplemented by sub-classes.

        Parameters
        ----------
        value
            Value to set the abstract continuous function extrapolator type
            with.

        Returns
        -------
        Type[ProtocolExtrapolator]
            Abstract continuous function extrapolator type.
        """

        ...  # pragma: no cover

    @extrapolator.setter
    @abstractmethod
    def extrapolator(self, value: Type[ProtocolExtrapolator]) -> None:
        """
        Setter for the **self.extrapolator** property, must be reimplemented by
        sub-classes.
        """

        ...  # pragma: no cover

    @property
    @abstractmethod
    def extrapolator_kwargs(self) -> dict:
        """
        Getter and setter for the abstract continuous function extrapolator
        instantiation time arguments.

        This property must be reimplemented by sub-classes.

        Parameters
        ----------
        value
            Value to set the abstract continuous function extrapolator
            instantiation time arguments to.

        Returns
        -------
        :class:`dict`
            Abstract continuous function extrapolator instantiation time
            arguments.
        """

        ...  # pragma: no cover

    @extrapolator_kwargs.setter
    @abstractmethod
    def extrapolator_kwargs(self, value: dict) -> None:
        """
        Setter for the **self.extrapolator_kwargs** property, must be
        reimplemented by sub-classes.
        """

        ...  # pragma: no cover

    @property
    @abstractmethod
    def function(self) -> Callable:
        """
        Getter for the abstract continuous function callable.

        This property must be reimplemented by sub-classes.

        Returns
        -------
        Callable
            Abstract continuous function callable.
        """

        ...  # pragma: no cover

    @abstractmethod
    def __str__(self) -> str:
        """
        Return a formatted string representation of the abstract continuous
        function.

        This method must be reimplemented by sub-classes.

        Returns
        -------
        :class:`str`
            Formatted string representation.
        """

        return super().__repr__()  # pragma: no cover

    @abstractmethod
    def __repr__(self) -> str:
        """
        Return an evaluable string representation of the abstract continuous
        function, must be reimplemented by sub-classes.

        Returns
        -------
        :class:`str`
            Evaluable string representation.
        """

        return super().__repr__()  # pragma: no cover

    @abstractmethod
    def __hash__(self) -> int:
        """
        Compute the hash of the abstract continuous function.

        Returns
        -------
        :class:`int`
            Object hash.
        """

        ...  # pragma: no cover

    @abstractmethod
    def __getitem__(self, x: ArrayLike | slice) -> NDArrayFloat:
        """
        Return the corresponding range variable :math:`y` for the specified
        independent domain variable :math:`x`.

        This abstract method must be reimplemented by sub-classes.

        Parameters
        ----------
        x
            Independent domain variable :math:`x`.

        Returns
        -------
        :class:`numpy.ndarray`
            Variable :math:`y` range value.
        """

        ...  # pragma: no cover

    @abstractmethod
    def __setitem__(self, x: ArrayLike | slice, y: ArrayLike) -> None:
        """
        Set the corresponding range variable :math:`y` for the specified
        independent domain variable :math:`x`.

        This abstract method must be reimplemented by sub-classes.

        Parameters
        ----------
        x
            Independent domain variable :math:`x`.
        y
            Corresponding range variable :math:`y`.
        """

        ...  # pragma: no cover

    @abstractmethod
    def __contains__(self, x: ArrayLike | slice) -> bool:
        """
        Determine whether the abstract continuous function contains the
        specified independent domain variable :math:`x`.

        This abstract method must be reimplemented by sub-classes.

        Parameters
        ----------
        x
            Independent domain variable :math:`x`.

        Returns
        -------
        :class:`bool`
            Whether :math:`x` domain value is contained.
        """

        ...  # pragma: no cover

    def __iter__(self) -> Generator:
        """
        Return a generator for the abstract continuous function.

        Yields
        ------
        Generator
            Abstract continuous function generator.
        """

        yield from np.column_stack([self.domain, self.range])

    def __len__(self) -> int:
        """
        Return the number of elements in the abstract continuous function's
        independent domain variable :math:`x`.

        Returns
        -------
        :class:`int`
            Number of elements in the independent domain variable :math:`x`.
        """

        return len(self.domain)

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """
        Determine whether the abstract continuous function equals the specified
        object.

        This abstract method must be reimplemented by sub-classes.

        Parameters
        ----------
        other
            Object to determine for equality with the abstract continuous function.

        Returns
        -------
        :class:`bool`
            Whether the specified object is equal to the abstract continuous
            function.
        """

        ...  # pragma: no cover

    @abstractmethod
    def __ne__(self, other: object) -> bool:
        """
        Determine whether the abstract continuous function is not equal to the
        specified object.

        This method must be reimplemented by sub-classes.

        Parameters
        ----------
        other
            Object to determine whether it is not equal to the abstract continuous
            function.

        Returns
        -------
        :class:`bool`
            Whether the specified object is not equal to the abstract
            continuous function.
        """

        ...  # pragma: no cover

    def __add__(self, a: ArrayLike | Self) -> Self:
        """
        Implement support for addition.

        Parameters
        ----------
        a
            Variable :math:`a` to add to the continuous function.

        Returns
        -------
        :class:`colour.continuous.AbstractContinuousFunction`
            Abstract continuous function with the specified variable
            added.
        """

        return self.arithmetical_operation(a, "+")

    def __iadd__(self, a: ArrayLike | Self) -> Self:
        """
        Implement support for in-place addition.

        Parameters
        ----------
        a
            Variable :math:`a` to add in-place to the abstract continuous
            function.

        Returns
        -------
        :class:`colour.continuous.AbstractContinuousFunction`
            Abstract continuous function with in-place addition applied.
        """

        return self.arithmetical_operation(a, "+", True)

    def __sub__(self, a: ArrayLike | Self) -> Self:
        """
        Implement support for subtraction.

        Parameters
        ----------
        a
            Variable :math:`a` to subtract from the continuous function.

        Returns
        -------
        :class:`colour.continuous.AbstractContinuousFunction`
            Abstract continuous function with the specified variable
            subtracted.
        """

        return self.arithmetical_operation(a, "-")

    def __isub__(self, a: ArrayLike | Self) -> Self:
        """
        Implement support for in-place subtraction.

        Parameters
        ----------
        a
            Variable :math:`a` to subtract in-place from the abstract continuous
            function.

        Returns
        -------
        :class:`colour.continuous.AbstractContinuousFunction`
            Abstract continuous function with in-place subtraction applied.
        """

        return self.arithmetical_operation(a, "-", True)

    def __mul__(self, a: ArrayLike | Self) -> Self:
        """
        Implement support for multiplication.

        Parameters
        ----------
        a
            Variable :math:`a` to multiply the continuous function by.

        Returns
        -------
        :class:`colour.continuous.AbstractContinuousFunction`
            Abstract continuous function with the specified variable
            multiplied.
        """

        return self.arithmetical_operation(a, "*")

    def __imul__(self, a: ArrayLike | Self) -> Self:
        """
        Implement support for in-place multiplication.

        Parameters
        ----------
        a
            Variable :math:`a` to multiply in-place by the abstract continuous
            function.

        Returns
        -------
        :class:`colour.continuous.AbstractContinuousFunction`
            Abstract continuous function with in-place multiplication applied.
        """

        return self.arithmetical_operation(a, "*", True)

    def __div__(self, a: ArrayLike | Self) -> Self:
        """
        Implement support for division.

        Parameters
        ----------
        a
            Variable :math:`a` to divide the continuous function by.

        Returns
        -------
        :class:`colour.continuous.AbstractContinuousFunction`
            Abstract continuous function with the specified variable
            divided.
        """

        return self.arithmetical_operation(a, "/")

    def __idiv__(self, a: ArrayLike | Self) -> Self:
        """
        Implement support for in-place division.

        Parameters
        ----------
        a
            Variable :math:`a` to divide in-place by the abstract continuous
            function.

        Returns
        -------
        :class:`colour.continuous.AbstractContinuousFunction`
            Abstract continuous function with in-place division applied.
        """

        return self.arithmetical_operation(a, "/", True)

    __itruediv__ = __idiv__
    __truediv__ = __div__

    def __pow__(self, a: ArrayLike | Self) -> Self:
        """
        Implement support for exponentiation.

        Parameters
        ----------
        a
            Variable :math:`a` to raise the continuous function to the power of.

        Returns
        -------
        :class:`colour.continuous.AbstractContinuousFunction`
            Abstract continuous function with the specified variable
            exponentiated.
        """

        return self.arithmetical_operation(a, "**")

    def __ipow__(self, a: ArrayLike | Self) -> Self:
        """
        Implement support for in-place exponentiation.

        Parameters
        ----------
        a
            Variable :math:`a` to raise in-place the abstract continuous
            function to the power of.

        Returns
        -------
        :class:`colour.continuous.AbstractContinuousFunction`
            Abstract continuous function with in-place exponentiation applied.
        """

        return self.arithmetical_operation(a, "**", True)

    @abstractmethod
    def arithmetical_operation(
        self,
        a: ArrayLike | Self,
        operation: Literal["+", "-", "*", "/", "**"],
        in_place: bool = False,
    ) -> Self:
        """
        Perform the specified arithmetical operation with operand :math:`a`,
        either on a copy or in-place.

        This method must be reimplemented by sub-classes.

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
        :class:`colour.continuous.AbstractContinuousFunction`
            Abstract continuous function.
        """

        ...  # pragma: no cover

    @abstractmethod
    def fill_nan(
        self,
        method: Literal["Constant", "Interpolation"] | str = "Interpolation",
        default: Real = 0,
    ) -> Self:
        """
        Fill NaNs in independent domain variable :math:`x` and corresponding
        range variable :math:`y` using the specified method.

        This abstract method must be reimplemented by sub-classes.

        Parameters
        ----------
        method
            *Interpolation* method linearly interpolates through the NaNs,
            *Constant* method replaces NaNs with ``default``.
        default
            Value to use with the *Constant* method.

        Returns
        -------
        :class:`colour.continuous.AbstractContinuousFunction`
            NaNs filled abstract continuous function.
        """

        ...  # pragma: no cover

    def domain_distance(self, a: ArrayLike) -> NDArrayFloat:
        """
        Return the Euclidean distance between specified array and the closest
        element of the independent domain :math:`x`.

        Parameters
        ----------
        a
            Variable :math:`a` to compute the Euclidean distance with the
            independent domain variable :math:`x`.

        Returns
        -------
        :class:`numpy.ndarray`
            Euclidean distance between independent domain variable :math:`x`
            and specified variable :math:`a`.
        """

        n = closest(self.domain, a)

        return as_float(np.abs(a - n))

    def is_uniform(self) -> bool:
        """
        Return whether the independent domain variable :math:`x` is uniform.

        Returns
        -------
        :class:`bool`
            Whether the independent domain variable :math:`x` is uniform.
        """

        return is_uniform(self.domain)

    def copy(self) -> Self:
        """
        Return a copy of the sub-class instance.

        Returns
        -------
        :class:`colour.continuous.AbstractContinuousFunction`
            Copy of the abstract continuous function.
        """

        return deepcopy(self)
