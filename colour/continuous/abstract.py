"""
Abstract Continuous Function
============================

Defines the abstract class implementing support for abstract continuous
function:

-   :class:`colour.continuous.AbstractContinuousFunction`.
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from copy import deepcopy

from colour.hints import (
    Any,
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
    is_string,
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
    Define the base class for abstract continuous function.

    This is an :class:`ABCMeta` abstract class that must be inherited by
    sub-classes.

    The sub-classes are expected to implement the
    :meth:`colour.continuous.AbstractContinuousFunction.function` method so
    that evaluating the function for any independent domain
    variable :math:`x \\in\\mathbb{R}` returns a corresponding range variable
    :math:`y \\in\\mathbb{R}`. A conventional implementation adopts an
    interpolating function encapsulated inside an extrapolating function.
    The resulting function independent domain, stored as discrete values in the
    :attr:`colour.continuous.AbstractContinuousFunction.domain` attribute
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
        Getter and setter property for the abstract continuous function name.

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
    def name(self, value: str):
        """Setter for the **self.name** property."""

        attest(
            is_string(value),
            f'"name" property: "{value}" type is not "str"!',
        )

        self._name = value

    @property
    @abstractmethod
    def dtype(self) -> Type[DTypeFloat]:
        """
        Getter and setter property for the abstract continuous function dtype,
        must be reimplemented by sub-classes.

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
    def dtype(self, value: Type[DTypeFloat]):
        """
        Setter for the **self.dtype** property, must be reimplemented by
        sub-classes.
        """

        ...  # pragma: no cover

    @property
    @abstractmethod
    def domain(self) -> NDArrayFloat:
        """
        Getter and setter property for the abstract continuous function
        independent domain variable :math:`x`, must be reimplemented by
        sub-classes.

        Parameters
        ----------
        value
            Value to set the abstract continuous function independent domain
            variable :math:`x` with.

        Returns
        -------
        :class:`numpy.ndarray`
            Abstract continuous function independent domain variable :math:`x`.
        """

        ...  # pragma: no cover

    @domain.setter
    @abstractmethod
    def domain(self, value: ArrayLike):
        """
        Setter for the **self.domain** property, must be reimplemented by
        sub-classes.
        """

        ...  # pragma: no cover

    @property
    @abstractmethod
    def range(self) -> NDArrayFloat:  # noqa: A003
        """
        Getter and setter property for the abstract continuous function
        corresponding range variable :math:`y`, must be reimplemented by
        sub-classes.

        Parameters
        ----------
        value
            Value to set the abstract continuous function corresponding range
            variable :math:`y` with.

        Returns
        -------
        :class:`numpy.ndarray`
            Abstract continuous function corresponding range variable
            :math:`y`.
        """

        ...  # pragma: no cover

    @range.setter
    @abstractmethod
    def range(self, value: ArrayLike):  # noqa: A003
        """
        Setter for the **self.range** property, must be reimplemented by
        sub-classes.
        """

        ...  # pragma: no cover

    @property
    @abstractmethod
    def interpolator(self) -> Type[ProtocolInterpolator]:
        """
        Getter and setter property for the abstract continuous function
        interpolator type, must be reimplemented by sub-classes.

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
    def interpolator(self, value: Type[ProtocolInterpolator]):
        """
        Setter for the **self.interpolator** property, must be reimplemented by
        sub-classes.
        """

        ...  # pragma: no cover

    @property
    @abstractmethod
    def interpolator_kwargs(self) -> dict:
        """
        Getter and setter property for the abstract continuous function
        interpolator instantiation time arguments, must be reimplemented by
        sub-classes.

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
    def interpolator_kwargs(self, value: dict):
        """
        Setter for the **self.interpolator_kwargs** property, must be
        reimplemented by sub-classes.
        """

        ...  # pragma: no cover

    @property
    @abstractmethod
    def extrapolator(self) -> Type[ProtocolExtrapolator]:
        """
        Getter and setter property for the abstract continuous function
        extrapolator type, must be reimplemented by sub-classes.

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
    def extrapolator(self, value: Type[ProtocolExtrapolator]):
        """
        Setter for the **self.extrapolator** property, must be reimplemented by
        sub-classes.
        """

        ...  # pragma: no cover

    @property
    @abstractmethod
    def extrapolator_kwargs(self) -> dict:
        """
        Getter and setter property for the abstract continuous function
        extrapolator instantiation time arguments, must be reimplemented by
        sub-classes.

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
    def extrapolator_kwargs(self, value: dict):
        """
        Setter for the **self.extrapolator_kwargs** property, must be
        reimplemented by sub-classes.
        """

        ...  # pragma: no cover

    @property
    @abstractmethod
    def function(self) -> Callable:
        """
        Getter property for the abstract continuous function callable, must be
        reimplemented by sub-classes.

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
        function, must be reimplemented by sub-classes.

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
        Return the abstract continuous function hash.

        Returns
        -------
        :class:`int`
            Object hash.
        """

        ...  # pragma: no cover

    @abstractmethod
    def __getitem__(self, x: ArrayLike | slice) -> NDArrayFloat:
        """
        Return the corresponding range variable :math:`y` for independent
        domain variable :math:`x`, must be reimplemented by sub-classes.

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
    def __setitem__(self, x: ArrayLike | slice, y: ArrayLike):
        """
        Set the corresponding range variable :math:`y` for independent domain
        variable :math:`x`, must be reimplemented by sub-classes.

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
        Return whether the abstract continuous function contains given
        independent domain variable :math:`x`, must be reimplemented by
        sub-classes.

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
        Return the abstract continuous function independent domain :math:`x`
        variable elements count.


        Returns
        -------
        :class:`int`
            Independent domain variable :math:`x` elements count.
        """

        return len(self.domain)

    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        """
        Return whether the abstract continuous function is equal to given
        other object, must be reimplemented by sub-classes.

        Parameters
        ----------
        other
            Object to test whether it is equal to the abstract continuous
            function.

        Returns
        -------
        :class:`bool`
            Whether given object is equal to the abstract continuous function.
        """

        ...  # pragma: no cover

    @abstractmethod
    def __ne__(self, other: Any) -> bool:
        """
        Return whether the abstract continuous function is not equal to given
        other object, must be reimplemented by sub-classes.

        Parameters
        ----------
        other
            Object to test whether it is not equal to the abstract continuous
            function.

        Returns
        -------
        :class:`bool`
            Whether given object is not equal to the abstract continuous
            function.
        """

        ...  # pragma: no cover

    def __add__(self, a: ArrayLike | Self) -> Self:
        """
        Implement support for addition.

        Parameters
        ----------
        a
            Variable :math:`a` to add.

        Returns
        -------
        :class:`colour.continuous.AbstractContinuousFunction`
            Variable added abstract continuous function.
        """

        return self.arithmetical_operation(a, "+")

    def __iadd__(self, a: ArrayLike | Self) -> Self:
        """
        Implement support for in-place addition.

        Parameters
        ----------
        a
            Variable :math:`a` to add in-place.

        Returns
        -------
        :class:`colour.continuous.AbstractContinuousFunction`
            In-place variable added abstract continuous function.
        """

        return self.arithmetical_operation(a, "+", True)

    def __sub__(self, a: ArrayLike | Self) -> Self:
        """
        Implement support for subtraction.

        Parameters
        ----------
        a
            Variable :math:`a` to subtract.

        Returns
        -------
        :class:`colour.continuous.AbstractContinuousFunction`
            Variable subtracted abstract continuous function.
        """

        return self.arithmetical_operation(a, "-")

    def __isub__(self, a: ArrayLike | Self) -> Self:
        """
        Implement support for in-place subtraction.

        Parameters
        ----------
        a
            Variable :math:`a` to subtract in-place.

        Returns
        -------
        :class:`colour.continuous.AbstractContinuousFunction`
            In-place variable subtracted abstract continuous function.
        """

        return self.arithmetical_operation(a, "-", True)

    def __mul__(self, a: ArrayLike | Self) -> Self:
        """
        Implement support for multiplication.

        Parameters
        ----------
        a
            Variable :math:`a` to multiply by.

        Returns
        -------
        :class:`colour.continuous.AbstractContinuousFunction`
            Variable multiplied abstract continuous function.
        """

        return self.arithmetical_operation(a, "*")

    def __imul__(self, a: ArrayLike | Self) -> Self:
        """
        Implement support for in-place multiplication.

        Parameters
        ----------
        a
            Variable :math:`a` to multiply by in-place.

        Returns
        -------
        :class:`colour.continuous.AbstractContinuousFunction`
            In-place variable multiplied abstract continuous function.
        """

        return self.arithmetical_operation(a, "*", True)

    def __div__(self, a: ArrayLike | Self) -> Self:
        """
        Implement support for division.

        Parameters
        ----------
        a
            Variable :math:`a` to divide by.

        Returns
        -------
        :class:`colour.continuous.AbstractContinuousFunction`
            Variable divided abstract continuous function.
        """

        return self.arithmetical_operation(a, "/")

    def __idiv__(self, a: ArrayLike | Self) -> Self:
        """
        Implement support for in-place division.

        Parameters
        ----------
        a
            Variable :math:`a` to divide by in-place.

        Returns
        -------
        :class:`colour.continuous.AbstractContinuousFunction`
            In-place variable divided abstract continuous function.
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
            Variable :math:`a` to exponentiate by.

        Returns
        -------
        :class:`colour.continuous.AbstractContinuousFunction`
            Variable exponentiated abstract continuous function.
        """

        return self.arithmetical_operation(a, "**")

    def __ipow__(self, a: ArrayLike | Self) -> Self:
        """
        Implement support for in-place exponentiation.

        Parameters
        ----------
        a
            Variable :math:`a` to exponentiate by in-place.

        Returns
        -------
        :class:`colour.continuous.AbstractContinuousFunction`
            In-place variable exponentiated abstract continuous function.
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
        Perform given arithmetical operation with operand :math:`a`, the
        operation can be either performed on a copy or in-place, must be
        reimplemented by sub-classes.

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
        range variable :math:`y` using given method, must be reimplemented by
        sub-classes.

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
        Return the euclidean distance between given array and independent
        domain :math:`x` closest element.

        Parameters
        ----------
        a
            Variable :math:`a` to compute the euclidean distance with
            independent domain variable :math:`x`.

        Returns
        -------
        :class:`numpy.ndarray`
            Euclidean distance between independent domain variable :math:`x`
            and given variable :math:`a`.
        """

        n = closest(self.domain, a)

        return as_float(np.abs(a - n))

    def is_uniform(self) -> bool:
        """
        Return if independent domain variable :math:`x` is uniform.

        Returns
        -------
        :class:`bool`
            Is independent domain variable :math:`x` uniform.
        """

        return is_uniform(self.domain)

    def copy(self) -> Self:
        """
        Return a copy of the sub-class instance.

        Returns
        -------
        :class:`colour.continuous.AbstractContinuousFunction`
            Abstract continuous function copy.
        """

        return deepcopy(self)
