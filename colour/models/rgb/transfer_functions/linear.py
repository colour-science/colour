"""
Linear Colour Component Transfer Function
=========================================

Define the linear encoding / decoding colour component transfer function
related objects.

- :func:`colour.linear_function`
"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ArrayLike, DTypeFloat, NDArray, NDArrayFloat

from colour.utilities import as_float

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "linear_function",
]


@typing.overload
def linear_function(a: float | DTypeFloat) -> DTypeFloat: ...
@typing.overload
def linear_function(a: NDArray) -> NDArrayFloat: ...
@typing.overload
def linear_function(a: ArrayLike) -> DTypeFloat | NDArrayFloat: ...
def linear_function(a: ArrayLike) -> DTypeFloat | NDArrayFloat:
    """
    Perform pass-through linear encoding/decoding transformation.

    Implement an identity transformation where the output equals the input,
    commonly used as a reference or default encoding/decoding function in
    colour science workflows.

    Parameters
    ----------
    a
        Array to encode/decode.

    Returns
    -------
    :class:`numpy.ndarray`
        Encoded/decoded array, identical to input.

    Examples
    --------
    >>> linear_function(0.18)  # doctest: +ELLIPSIS
    0.1799999...
    """

    return as_float(a)
