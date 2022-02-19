"""
LUT Operator
============

Defines the *LUT* sequence class:

-   :class:`colour.LUTSequence`
"""

from __future__ import annotations

import re
from collections.abc import MutableSequence
from copy import deepcopy

from colour.hints import (
    Any,
    ArrayLike,
    Integer,
    List,
    NDArray,
    Sequence,
    TypeLUTSequenceItem,
    Union,
)
from colour.utilities import as_float_array, attest, is_iterable

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "LUTSequence",
]


class LUTSequence(MutableSequence):
    """
    Define the base class for a *LUT* sequence, i.e. a series of *LUTs*,
    *LUT* operators or objects implementing the
    :class:`colour.hints.TypeLUTSequenceItem` protocol.

    The `colour.LUTSequence` class can be used to model series of *LUTs* such
    as when a shaper *LUT* is combined with a 3D *LUT*.

    Other Parameters
    ----------------
    args
        Sequence of objects implementing the
        :class:`colour.hints.TypeLUTSequenceItem` protocol.

    Attributes
    ----------
    -   :attr:`~colour.LUTSequence.sequence`

    Methods
    -------
    -   :meth:`~colour.LUTSequence.__init__`
    -   :meth:`~colour.LUTSequence.__getitem__`
    -   :meth:`~colour.LUTSequence.__setitem__`
    -   :meth:`~colour.LUTSequence.__delitem__`
    -   :meth:`~colour.LUTSequence.__len__`
    -   :meth:`~colour.LUTSequence.__str__`
    -   :meth:`~colour.LUTSequence.__repr__`
    -   :meth:`~colour.LUTSequence.__eq__`
    -   :meth:`~colour.LUTSequence.__ne__`
    -   :meth:`~colour.LUTSequence.insert`
    -   :meth:`~colour.LUTSequence.apply`
    -   :meth:`~colour.LUTSequence.copy`

    Examples
    --------
    >>> from colour.io.luts import  LUT1D, LUT3x1D, LUT3D
    >>> LUT_1 = LUT1D()
    >>> LUT_2 = LUT3D(size=3)
    >>> LUT_3 = LUT3x1D()
    >>> print(LUTSequence(LUT_1, LUT_2, LUT_3))
    LUT Sequence
    ------------
    <BLANKLINE>
    Overview
    <BLANKLINE>
        LUT1D --> LUT3D --> LUT3x1D
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

    def __init__(self, *args: TypeLUTSequenceItem):
        # TODO: Remove pragma when https://github.com/python/mypy/issues/3004
        # is resolved.
        self._sequence: List[TypeLUTSequenceItem] = []
        self.sequence = args  # type: ignore[assignment]

    @property
    def sequence(self) -> List[TypeLUTSequenceItem]:
        """
        Getter and setter property for the underlying *LUT* sequence.

        Parameters
        ----------
        value
            Value to set the underlying *LUT* sequence with.

        Returns
        -------
        :class:`list`
            Underlying *LUT* sequence.
        """

        return self._sequence

    @sequence.setter
    def sequence(self, value: Sequence[TypeLUTSequenceItem]):
        """Setter for the **self.sequence** property."""

        for item in value:
            attest(
                isinstance(item, TypeLUTSequenceItem),
                '"value" items must implement the "TypeLUTSequenceItem" '
                "protocol!",
            )

        self._sequence = list(value)

    def __getitem__(self, index: Union[Integer, slice]) -> Any:
        """
        Return the *LUT* sequence item(s) at given index (or slice).

        Parameters
        ----------
        index
            Index (or slice) to return the *LUT* sequence item(s) at.

        Returns
        -------
        TypeLUTSequenceItem
            *LUT* sequence item(s) at given index (or slice).
        """

        return self._sequence[index]

    def __setitem__(self, index: Union[Integer, slice], value: Any):
        """
        Set the *LUT* sequence at given index (or slice) with given value.

        Parameters
        ----------
        index
            Index (or slice) to set the *LUT* sequence value at.
        value
            Value to set the *LUT* sequence with.
        """

        for item in value if is_iterable(value) else [value]:
            attest(
                isinstance(item, TypeLUTSequenceItem),
                '"value" items must implement the "TypeLUTSequenceItem" '
                "protocol!",
            )

        self._sequence[index] = value

    def __delitem__(self, index: Union[Integer, slice]):
        """
        Delete the *LUT* sequence item(s) at given index (or slice).

        Parameters
        ----------
        index
            Index (or slice) to delete the *LUT* sequence items at.
        """

        del self._sequence[index]

    def __len__(self) -> Integer:
        """
        Return the *LUT* sequence items count.

        Returns
        -------
        :class:`numpy.integer`
            *LUT* sequence items count.
        """

        return len(self._sequence)

    def __str__(self) -> str:
        """
        Return a formatted string representation of the *LUT* sequence.

        Returns
        -------
        :class:`str`
            Formatted string representation.
        """

        sequence = " --> ".join([a.__class__.__name__ for a in self._sequence])

        operations = re.sub(
            "^",
            " " * 4,
            "\n\n".join([str(a) for a in self._sequence]),
            flags=re.MULTILINE,
        )
        operations = re.sub("^\\s+$", "", operations, flags=re.MULTILINE)

        return (
            "LUT Sequence\n"
            "------------\n\n"
            "Overview\n\n"
            f"    {sequence}\n\n"
            "Operations\n\n"
            f"{operations}"
        )

    def __repr__(self) -> str:
        """
        Return an evaluable string representation of the *LUT* sequence.

        Returns
        -------
        :class:`str`
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

    def __eq__(self, other) -> bool:
        """
        Return whether the *LUT* sequence is equal to given other object.

        Parameters
        ----------
        other
            Object to test whether it is equal to the *LUT* sequence.

        Returns
        -------
        :class:`bool`
            Whether given object is equal to the *LUT* sequence.
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

    def __ne__(self, other) -> bool:
        """
        Return whether the *LUT* sequence is not equal to given other object.

        Parameters
        ----------
        other
            Object to test whether it is not equal to the *LUT* sequence.

        Returns
        -------
        :class:`bool`
            Whether given object is not equal to the *LUT* sequence.
        """

        return not (self == other)

    def insert(self, index: Integer, item: TypeLUTSequenceItem):
        """
        Insert given *LUT* at given index into the *LUT* sequence.

        Parameters
        ----------
        index
            Index to insert the item at into the *LUT* sequence.
        item
            *LUT* to insert into the *LUT* sequence.
        """

        attest(
            isinstance(item, TypeLUTSequenceItem),
            '"value" items must implement the "TypeLUTSequenceItem" '
            "protocol!",
        )

        self._sequence.insert(index, item)

    def apply(self, RGB: ArrayLike, **kwargs: Any) -> NDArray:
        """
        Apply the *LUT* sequence sequentially to given *RGB* colourspace
        array.

        Parameters
        ----------
        RGB
            *RGB* colourspace array to apply the *LUT* sequence sequentially
            onto.

        Other Parameters
        ----------------
        kwargs
            Keywords arguments, the keys must be the class type names for which
            they are intended to be used with. There is no implemented way to
            discriminate which class instance the keyword arguments should be
            used with, thus if many class instances of the same type are
            members of the sequence, any matching keyword arguments will be
            used with all the class instances.

        Returns
        -------
        :class:`numpy.ndarray`
            Processed *RGB* colourspace array.

        Examples
        --------
        >>> import numpy as np
        >>> from colour.io.luts import  LUT1D, LUT3x1D, LUT3D
        >>> from colour.utilities import tstack
        >>> LUT_1 = LUT1D(LUT1D.linear_table(16) + 0.125)
        >>> LUT_2 = LUT3D(LUT3D.linear_table(16) ** (1 / 2.2))
        >>> LUT_3 = LUT3x1D(LUT3x1D.linear_table(16) * 0.750)
        >>> LUT_sequence = LUTSequence(LUT_1, LUT_2, LUT_3)
        >>> samples = np.linspace(0, 1, 5)
        >>> RGB = tstack([samples, samples, samples])
        >>> LUT_sequence.apply(RGB, LUT1D={'direction': 'Inverse'})
        ... # doctest: +ELLIPSIS
        array([[ 0.       ...,  0.       ...,  0.       ...],
               [ 0.2899886...,  0.2899886...,  0.2899886...],
               [ 0.4797662...,  0.4797662...,  0.4797662...],
               [ 0.6055328...,  0.6055328...,  0.6055328...],
               [ 0.7057779...,  0.7057779...,  0.7057779...]])
        """

        RGB = as_float_array(RGB)

        RGB_o = RGB
        for operator in self:
            RGB_o = operator.apply(
                RGB_o, **kwargs.get(operator.__class__.__name__, {})
            )

        return RGB_o

    def copy(self) -> LUTSequence:
        """
        Return a copy of the *LUT* sequence.

        Returns
        -------
        :class:`colour.LUTSequence`
            *LUT* sequence copy.
        """

        return deepcopy(self)
