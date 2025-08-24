"""
LUT Operator
============

Define operator classes for Look-Up Table (LUT) transformations within the
colour processing pipeline.

-   :class:`colour.io.AbstractLUTSequenceOperator`
-   :class:`colour.LUTOperatorMatrix`
"""

from __future__ import annotations

import typing
from abc import ABC, abstractmethod

import numpy as np

from colour.algebra import vecmul

if typing.TYPE_CHECKING:
    from colour.hints import (
        Any,
        ArrayLike,
        List,
        NDArrayFloat,
        Sequence,
    )

from colour.utilities import as_float_array, attest, is_iterable, ones, optional, zeros

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "AbstractLUTSequenceOperator",
    "LUTOperatorMatrix",
]


class AbstractLUTSequenceOperator(ABC):
    """
    Define the base class for *LUT* sequence operators.

    Provide an abstract base class that establishes the interface for all
    *LUT* sequence operator implementations. This :class:`ABCMeta` abstract
    class must be inherited by concrete sub-classes that implement specific
    operator functionality within *LUT* processing pipelines.

    Parameters
    ----------
    name
        *LUT* sequence operator name.
    comments
        Comments to add to the *LUT* sequence operator.

    Attributes
    ----------
    -   :attr:`~colour.io.AbstractLUTSequenceOperator.name`
    -   :attr:`~colour.io.AbstractLUTSequenceOperator.comments`

    Methods
    -------
    -   :meth:`~colour.io.AbstractLUTSequenceOperator.apply`
    """

    def __init__(
        self,
        name: str | None = None,
        comments: Sequence[str] | None = None,
    ) -> None:
        self._name = f"LUT Sequence Operator {id(self)}"
        self.name = optional(name, self._name)
        self._comments: List[str] = []
        self.comments = optional(comments, self._comments)

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
    def comments(self) -> List[str]:
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
    def comments(self, value: Sequence[str]) -> None:
        """Setter for the **self.comments** property."""

        attest(
            is_iterable(value),
            f'"comments" property: "{value}" must be a sequence!',
        )

        self._comments = list(value)

    @abstractmethod
    def apply(self, RGB: ArrayLike, *args: Any, **kwargs: Any) -> NDArrayFloat:
        """
        Apply the *LUT* sequence operator to the specified *RGB* colourspace
        array.

        Parameters
        ----------
        RGB
            *RGB* colourspace array to apply the *LUT* sequence operator onto.

        Other Parameters
        ----------------
        args
            Arguments.
        kwargs
            Keywords arguments.

        Returns
        -------
        :class:`numpy.ndarray`
            Processed *RGB* colourspace array.
        """


class LUTOperatorMatrix(AbstractLUTSequenceOperator):
    """
    Define the *LUT* operator that applies matrix transformations and offset
    vectors for colour space conversions.

    Support 3x3 or 4x4 matrix operations with optional offset vectors to
    perform affine transformations on *RGB* colourspace data. The operator
    internally reshapes matrices to 4x4 and offsets to 4-element vectors to
    maintain computational consistency.

    Parameters
    ----------
    matrix
        3x3 or 4x4 matrix for the operator.
    offset
        Offset for the operator.
    name
        *LUT* operator name.
    comments
        Comments to add to the *LUT* operator.

    Attributes
    ----------
    -   :meth:`~colour.LUTOperatorMatrix.matrix`
    -   :meth:`~colour.LUTOperatorMatrix.offset`

    Methods
    -------
    -   :meth:`~colour.LUTOperatorMatrix.__str__`
    -   :meth:`~colour.LUTOperatorMatrix.__repr__`
    -   :meth:`~colour.LUTOperatorMatrix.__eq__`
    -   :meth:`~colour.LUTOperatorMatrix.__ne__`
    -   :meth:`~colour.LUTOperatorMatrix.apply`

    Notes
    -----
    -   The internal :attr:`colour.io.Matrix.matrix` and
        :attr:`colour.io.Matrix.offset` properties are reshaped to (4, 4) and
        (4, ) respectively.

    Examples
    --------
    Instantiating an identity matrix:

    >>> print(LUTOperatorMatrix(name="Identity"))
    LUTOperatorMatrix - Identity
    ----------------------------
    <BLANKLINE>
    Matrix     : [[ 1.  0.  0.  0.]
                  [ 0.  1.  0.  0.]
                  [ 0.  0.  1.  0.]
                  [ 0.  0.  0.  1.]]
    Offset     : [ 0.  0.  0.  0.]

    Instantiating a matrix with comments:

    >>> matrix = np.array(
    ...     [
    ...         [1.45143932, -0.23651075, -0.21492857],
    ...         [-0.07655377, 1.1762297, -0.09967593],
    ...         [0.00831615, -0.00603245, 0.9977163],
    ...     ]
    ... )
    >>> print(
    ...     LUTOperatorMatrix(
    ...         matrix,
    ...         name="AP0 to AP1",
    ...         comments=["A first comment.", "A second comment."],
    ...     )
    ... )
    LUTOperatorMatrix - AP0 to AP1
    ------------------------------
    <BLANKLINE>
    Matrix     : [[ 1.45143932 -0.23651075 -0.21492857  0.        ]
                  [-0.07655377  1.1762297  -0.09967593  0.        ]
                  [ 0.00831615 -0.00603245  0.9977163   0.        ]
                  [ 0.          0.          0.          1.        ]]
    Offset     : [ 0.  0.  0.  0.]
    <BLANKLINE>
    A first comment.
    A second comment.
    """

    def __init__(
        self,
        matrix: ArrayLike | None = None,
        offset: ArrayLike | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self._matrix: NDArrayFloat = np.diag(ones(4))
        self.matrix = optional(matrix, self._matrix)
        self._offset: NDArrayFloat = zeros(4)
        self.offset = optional(offset, self._offset)

    @property
    def matrix(self) -> NDArrayFloat:
        """
        Getter and setter for the *LUT* operator matrix.

        Parameters
        ----------
        value
            Value to set the *LUT* operator matrix with.

        Returns
        -------
        :class:`numpy.ndarray`
            Operator matrix.
        """

        return self._matrix

    @matrix.setter
    def matrix(self, value: ArrayLike) -> None:
        """Setter for the **self.matrix** property."""

        value = as_float_array(value)

        shape_t = value.shape[-1]

        value = np.reshape(value, (shape_t, shape_t))

        attest(
            value.shape in [(3, 3), (4, 4)],
            f'"matrix" property: "{value}" shape is not (3, 3) or (4, 4)!',
        )

        M = np.identity(4)
        M[:shape_t, :shape_t] = value

        self._matrix = M

    @property
    def offset(self) -> NDArrayFloat:
        """
        Getter and setter for the *LUT* operator offset vector.

        The offset vector is applied after the matrix transformation in the LUT
        operator, enabling translation operations in the colour space.

        Parameters
        ----------
        value
            Value to set the *LUT* operator offset with.

        Returns
        -------
        :class:`numpy.ndarray`
            Operator offset vector.
        """

        return self._offset

    @offset.setter
    def offset(self, value: ArrayLike) -> None:
        """Setter for the **self.offset** property."""

        value = as_float_array(value)

        shape_t = value.shape[-1]

        attest(
            value.shape in [(3,), (4,)],
            f'"offset" property: "{value}" shape is not (3, ) or (4, )!',
        )

        offset = zeros(4)
        offset[:shape_t] = value

        self._offset = offset

    def __str__(self) -> str:
        """
        Return a formatted string representation of the *LUT* operator.

        Returns
        -------
        :class:`str`
            Formatted string representation.

        Examples
        --------
        >>> print(LUTOperatorMatrix())  # doctest: +ELLIPSIS
        LUTOperatorMatrix - LUT Sequence Operator ...
        ------------------------------------------...
        <BLANKLINE>
        Matrix     : [[ 1.  0.  0.  0.]
                      [ 0.  1.  0.  0.]
                      [ 0.  0.  1.  0.]
                      [ 0.  0.  0.  1.]]
        Offset     : [ 0.  0.  0.  0.]
        """

        def _format(a: ArrayLike) -> str:
            """Format specified array string representation."""

            return str(a).replace(" [", " " * 14 + "[")

        comments = "\n".join(self._comments)
        comments = f"\n\n{comments}" if self._comments else ""

        underline = "-" * (len(self.__class__.__name__) + 3 + len(self._name))

        return "\n".join(
            [
                f"{self.__class__.__name__} - {self._name}",
                f"{underline}",
                "",
                f"Matrix     : {_format(self._matrix)}",
                f"Offset     : {_format(self._offset)}{comments}",
            ]
        )

    def __repr__(self) -> str:
        """
        Return an evaluable string representation of the *LUT* operator.

        Returns
        -------
        :class:`str`
            Evaluable string representation.

        Examples
        --------
        >>> LUTOperatorMatrix(comments=["A first comment.", "A second comment."])
        ... # doctest: +ELLIPSIS
        LUTOperatorMatrix([[ 1.,  0.,  0.,  0.],
                           [ 0.,  1.,  0.,  0.],
                           [ 0.,  0.,  1.,  0.],
                           [ 0.,  0.,  0.,  1.]],
                          [ 0.,  0.,  0.,  0.],
                          name='LUT Sequence Operator ...',
                          comments=['A first comment.', 'A second comment.'])
        """

        representation = repr(self._matrix)
        representation = representation.replace("array", self.__class__.__name__)
        representation = representation.replace(
            "       [", f"{' ' * (len(self.__class__.__name__) + 2)}["
        )

        indentation = " " * (len(self.__class__.__name__) + 1)

        comments = (
            f",\n{indentation}comments={self._comments!r}" if self._comments else ""
        )

        return "\n".join(
            [
                f"{representation[:-1]},",
                f"{indentation}"
                f"{repr(self._offset).replace('array(', '').replace(')', '')},",
                f"{indentation}name='{self._name}'{comments})",
            ]
        )

    __hash__ = None  # pyright: ignore

    def __eq__(self, other: object) -> bool:
        """
        Determine whether the *LUT* operator is equal to the specified other
        object.

        Parameters
        ----------
        other
            Object to test whether it is equal to the *LUT* operator.

        Returns
        -------
        :class:`bool`
            Whether specified object equal to the *LUT* operator.

        Examples
        --------
        >>> LUTOperatorMatrix() == LUTOperatorMatrix()
        True
        """

        return isinstance(other, LUTOperatorMatrix) and all(
            [
                np.array_equal(self._matrix, other._matrix),
                np.array_equal(self._offset, other._offset),
            ]
        )

    def __ne__(self, other: object) -> bool:
        """
        Determine whether the *LUT* operator is not equal to the specified other
        object.

        Parameters
        ----------
        other
            Object to test whether it is not equal to the *LUT* operator.

        Returns
        -------
        :class:`bool`
            Whether the specified object is not equal to the *LUT* operator.

        Examples
        --------
        >>> LUTOperatorMatrix() != LUTOperatorMatrix(
        ...     np.reshape(np.linspace(0, 1, 16), (4, 4))
        ... )
        True
        """

        return not (self == other)

    def apply(
        self,
        RGB: ArrayLike,
        *args: Any,  # noqa: ARG002
        **kwargs: Any,
    ) -> NDArrayFloat:
        """
        Apply the *LUT* operator to the specified *RGB* array.

        Parameters
        ----------
        RGB
            *RGB* array to apply the *LUT* operator transform to.

        Other Parameters
        ----------------
        apply_offset_first
            Whether to apply the offset first and then the matrix.

        Returns
        -------
        :class:`numpy.ndarray`
            Transformed *RGB* array.

        Examples
        --------
        >>> matrix = np.array(
        ...     [
        ...         [1.45143932, -0.23651075, -0.21492857],
        ...         [-0.07655377, 1.1762297, -0.09967593],
        ...         [0.00831615, -0.00603245, 0.9977163],
        ...     ]
        ... )
        >>> M = LUTOperatorMatrix(matrix)
        >>> RGB = np.array([0.3, 0.4, 0.5])
        >>> M.apply(RGB)  # doctest: +ELLIPSIS
        array([ 0.2333632...,  0.3976877...,  0.4989400...])
        """

        RGB = as_float_array(RGB)
        apply_offset_first = kwargs.get("apply_offset_first", False)

        has_alpha_channel = RGB.shape[-1] == 4
        M = self._matrix
        offset = self._offset

        if not has_alpha_channel:
            M = M[:3, :3]
            offset = offset[:3]

        if apply_offset_first:
            RGB += offset

        RGB = vecmul(M, RGB)

        if not apply_offset_first:
            RGB += offset

        return RGB
