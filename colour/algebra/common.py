"""
Common Utilities
================

Define common algebra utility objects that do not fall within any specific
category.

The *Common* sub-package provides general-purpose mathematical and
computational utilities used throughout the colour science library.
"""

from __future__ import annotations

import functools
import typing

import numpy as np

if typing.TYPE_CHECKING:
    from colour.hints import (
        Any,
        ArrayLike,
        Callable,
        DTypeFloat,
        NDArray,
        NDArrayFloat,
        Self,
        Tuple,
    )

from colour.constants import EPSILON
from colour.hints import Literal, cast
from colour.utilities import (
    as_float,
    as_float_array,
    optional,
    runtime_warning,
    tsplit,
    validate_method,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "get_sdiv_mode",
    "set_sdiv_mode",
    "sdiv_mode",
    "sdiv",
    "is_spow_enabled",
    "set_spow_enable",
    "spow_enable",
    "spow",
    "normalise_vector",
    "normalise_maximum",
    "vecmul",
    "euclidean_distance",
    "manhattan_distance",
    "linear_conversion",
    "linstep_function",
    "lerp",
    "smoothstep_function",
    "smooth",
    "is_identity",
    "eigen_decomposition",
]

_SDIV_MODE: Literal[
    "Numpy",
    "Ignore",
    "Warning",
    "Raise",
    "Ignore Zero Conversion",
    "Warning Zero Conversion",
    "Ignore Limit Conversion",
    "Warning Limit Conversion",
    "Replace With Epsilon",
    "Warning Replace With Epsilon",
] = "Ignore Zero Conversion"
"""
Global variable storing the current *Colour* safe division function mode.
"""


def get_sdiv_mode() -> Literal[
    "Numpy",
    "Ignore",
    "Warning",
    "Raise",
    "Ignore Zero Conversion",
    "Warning Zero Conversion",
    "Ignore Limit Conversion",
    "Warning Limit Conversion",
    "Replace With Epsilon",
    "Warning Replace With Epsilon",
]:
    """
    Return the current *Colour* safe division mode.

    Returns
    -------
    :class:`str`
        Current *Colour* safe division mode. See
        :func:`colour.algebra.sdiv` definition for an explanation of
        the possible modes.

    Examples
    --------
    >>> with sdiv_mode("Numpy"):
    ...     get_sdiv_mode()
    'numpy'
    >>> with sdiv_mode("Ignore Zero Conversion"):
    ...     get_sdiv_mode()
    'ignore zero conversion'
    """

    return _SDIV_MODE


def set_sdiv_mode(
    mode: (
        Literal[
            "Numpy",
            "Ignore",
            "Warning",
            "Raise",
            "Ignore Zero Conversion",
            "Warning Zero Conversion",
            "Ignore Limit Conversion",
            "Warning Limit Conversion",
            "Replace With Epsilon",
            "Warning Replace With Epsilon",
        ]
        | str
    ),
) -> None:
    """
    Set the *Colour* safe division function mode.

    Parameters
    ----------
    mode
        *Colour* safe division mode. See :func:`colour.algebra.sdiv`
        definition for an explanation of the possible modes.

    Examples
    --------
    >>> with sdiv_mode(get_sdiv_mode()):
    ...     print(get_sdiv_mode())
    ...     set_sdiv_mode("Raise")
    ...     print(get_sdiv_mode())
    ignore zero conversion
    raise
    """

    global _SDIV_MODE  # noqa: PLW0603

    _SDIV_MODE = cast(
        "Literal['Numpy', 'Ignore', 'Warning', 'Raise', "  # pyright: ignore
        "'Ignore Zero Conversion', 'Warning Zero Conversion', "
        "'Ignore Limit Conversion', 'Warning Limit Conversion', "
        "'Replace With Epsilon', 'Warning Replace With Epsilon']",
        validate_method(
            mode,
            (
                "Numpy",
                "Ignore",
                "Warning",
                "Raise",
                "Ignore Zero Conversion",
                "Warning Zero Conversion",
                "Ignore Limit Conversion",
                "Warning Limit Conversion",
                "Replace With Epsilon",
                "Warning Replace With Epsilon",
            ),
        ),
    )


class sdiv_mode:
    """
    Context manager and decorator for temporarily modifying *Colour* safe
    division function mode.

    This utility enables temporary modification of the safe division behavior
    in *Colour* computations, allowing control over how division operations
    handle edge cases such as division by zero or near-zero values. The
    context manager ensures automatic restoration of the original mode upon
    exit.

    Parameters
    ----------
    mode
        *Colour* safe division function mode, see :func:`colour.algebra.sdiv`
        definition for an explanation about the possible modes.
    """

    def __init__(
        self,
        mode: (
            Literal[
                "Numpy",
                "Ignore",
                "Warning",
                "Raise",
                "Ignore Zero Conversion",
                "Warning Zero Conversion",
                "Ignore Limit Conversion",
                "Warning Limit Conversion",
                "Replace With Epsilon",
                "Warning Replace With Epsilon",
            ]
            | None
        ) = None,
    ) -> None:
        self._mode = optional(mode, get_sdiv_mode())
        self._previous_mode = get_sdiv_mode()

    def __enter__(self) -> Self:
        """
        Set the *Colour* safe/symmetrical power function state to the
        specified value upon entering the context manager.
        """

        set_sdiv_mode(self._mode)

        return self

    def __exit__(self, *args: Any) -> None:
        """
        Restore the *Colour* safe / symmetrical power function enabled state
        upon exiting the context manager.
        """

        set_sdiv_mode(self._previous_mode)

    def __call__(self, function: Callable) -> Callable:
        """
        Call the wrapped definition.

        The decorator applies the specified spectral power distribution
        state to the wrapped function during its execution.
        """

        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with self:
                return function(*args, **kwargs)

        return wrapper


def sdiv(a: ArrayLike, b: ArrayLike) -> NDArrayFloat:
    """
    Perform safe division of array :math:`a` by array :math:`b` while
    handling zero-division cases.

    Avoid NaN and +/- inf generation when array :math:`b` contains zero
    values. The zero-division handling behaviour is controlled by the
    :func:`colour.algebra.set_sdiv_mode` definition or the
    :func:`sdiv_mode` context manager. The following modes are available:

    -   ``Numpy``: The current *Numpy* zero-division handling occurs.
    -   ``Ignore``: Zero-division occurs silently.
    -   ``Warning``: Zero-division occurs with a warning.
    -   ``Ignore Zero Conversion``: Zero-division occurs silently and NaNs
        or +/- infs values are converted to zeros. See
        :func:`numpy.nan_to_num` definition for more details.
    -   ``Warning Zero Conversion``: Zero-division occurs with a warning
        and NaNs or +/- infs values are converted to zeros. See
        :func:`numpy.nan_to_num` definition for more details.
    -   ``Ignore Limit Conversion``: Zero-division occurs silently and
        NaNs or +/- infs values are converted to zeros or the largest +/-
        finite floating point values representable by the division result
        :class:`numpy.dtype`. See :func:`numpy.nan_to_num` definition for
        more details.
    -   ``Warning Limit Conversion``: Zero-division occurs with a warning
        and NaNs or +/- infs values are converted to zeros or the largest
        +/- finite floating point values representable by the division
        result :class:`numpy.dtype`.
    -   ``Replace With Epsilon``: Zero-division is avoided by replacing
        zero denominators with the machine epsilon value from
        :attr:`colour.constants.EPSILON`.
    -   ``Warning Replace With Epsilon``: Zero-division is avoided by
        replacing zero denominators with the machine epsilon value from
        :attr:`colour.constants.EPSILON` with a warning.

    Parameters
    ----------
    a
        Numerator array :math:`a`.
    b
        Denominator array :math:`b`.

    Returns
    -------
    :class:`np.float` or :class:`numpy.ndarray`
        Array :math:`a` safely divided by :math:`b`.

    Examples
    --------
    >>> a = np.array([0, 1, 2])
    >>> b = np.array([2, 1, 0])
    >>> sdiv(a, b)
    array([ 0.,  1.,  0.])
    >>> try:
    ...     with sdiv_mode("Raise"):
    ...         sdiv(a, b)
    ... except Exception as error:
    ...     error  # doctest: +ELLIPSIS
    FloatingPointError('divide by zero encountered in...divide')
    >>> with sdiv_mode("Ignore Zero Conversion"):
    ...     sdiv(a, b)
    array([ 0.,  1.,  0.])
    >>> with sdiv_mode("Warning Zero Conversion"):
    ...     sdiv(a, b)
    array([ 0.,  1.,  0.])
    >>> with sdiv_mode("Ignore Limit Conversion"):
    ...     sdiv(a, b)  # doctest: +SKIP
    array([  0.00000000e+000,   1.00000000e+000,   1.79769313e+308])
    >>> with sdiv_mode("Warning Limit Conversion"):
    ...     sdiv(a, b)  # doctest: +SKIP
    array([  0.00000000e+000,   1.00000000e+000,   1.79769313e+308])
    >>> with sdiv_mode("Replace With Epsilon"):
    ...     sdiv(a, b)  # doctest: +ELLIPSIS
    array([  0.00000000e+00,   1.00000000e+00,  ...])
    >>> with sdiv_mode("Warning Replace With Epsilon"):
    ...     sdiv(a, b)  # doctest: +ELLIPSIS
    array([  0.00000000e+00,   1.00000000e+00,  ...])
    """

    a = as_float_array(a)
    b = as_float_array(b)

    mode = validate_method(
        _SDIV_MODE,
        (
            "Numpy",
            "Ignore",
            "Warning",
            "Raise",
            "Ignore Zero Conversion",
            "Warning Zero Conversion",
            "Ignore Limit Conversion",
            "Warning Limit Conversion",
            "Replace With Epsilon",
            "Warning Replace With Epsilon",
        ),
    )

    if mode == "numpy":
        c = a / b
    elif mode == "ignore":
        with np.errstate(divide="ignore", invalid="ignore"):
            c = a / b
    elif mode == "warning":
        with np.errstate(divide="warn", invalid="warn"):
            c = a / b
    elif mode == "raise":
        with np.errstate(divide="raise", invalid="raise"):
            c = a / b
    elif mode == "ignore zero conversion":
        with np.errstate(divide="ignore", invalid="ignore"):
            c = np.nan_to_num(a / b, nan=0, posinf=0, neginf=0)
    elif mode == "warning zero conversion":
        with np.errstate(divide="warn", invalid="warn"):
            c = np.nan_to_num(a / b, nan=0, posinf=0, neginf=0)
    elif mode == "ignore limit conversion":
        with np.errstate(divide="ignore", invalid="ignore"):
            c = np.nan_to_num(a / b)
    elif mode == "warning limit conversion":
        with np.errstate(divide="warn", invalid="warn"):
            c = np.nan_to_num(a / b)
    elif mode == "replace with epsilon":
        b = np.where(b == 0, EPSILON, b)
        c = a / b
    elif mode == "warning replace with epsilon":
        if np.any(b == 0):
            runtime_warning("Zero(s) detected in denominator, replacing with EPSILON.")
        b = np.where(b == 0, EPSILON, b)
        c = a / b

    return c


_SPOW_ENABLED: bool = True
"""
Global variable storing the current *Colour* safe / symmetrical power function
enabled state.
"""


def is_spow_enabled() -> bool:
    """
    Return whether *Colour* safe / symmetrical power function is enabled.

    Returns
    -------
    :class:`bool`
        Whether *Colour* safe / symmetrical power function is enabled.

    Examples
    --------
    >>> with spow_enable(False):
    ...     is_spow_enabled()
    False
    >>> with spow_enable(True):
    ...     is_spow_enabled()
    True
    """

    return _SPOW_ENABLED


def set_spow_enable(enable: bool) -> None:
    """
    Set the *Colour* safe/symmetrical power function enabled state.

    Parameters
    ----------
    enable
        Whether to enable the *Colour* safe/symmetrical power function.

    Examples
    --------
    >>> with spow_enable(is_spow_enabled()):
    ...     print(is_spow_enabled())
    ...     set_spow_enable(False)
    ...     print(is_spow_enabled())
    True
    False
    """

    global _SPOW_ENABLED  # noqa: PLW0603

    _SPOW_ENABLED = enable


class spow_enable:
    """
    Context manager and decorator for temporarily setting the state of *Colour*
    safe/symmetrical power function.

    This utility provides both context manager and decorator functionality to
    temporarily enable or disable the safe/symmetrical power function used
    throughout the *Colour* library. When enabled, power operations use a
    symmetrical implementation that handles negative values appropriately for
    colour science computations.

    Parameters
    ----------
    enable
        Whether to enable or disable the *Colour* safe/symmetrical power
        function for the duration of the context or decorated function.
    """

    def __init__(self, enable: bool) -> None:
        self._enable = enable
        self._previous_state = is_spow_enabled()

    def __enter__(self) -> Self:
        """
        Set the *Colour* safe / symmetrical power function enabled state
        upon entering the context manager.
        """

        set_spow_enable(self._enable)

        return self

    def __exit__(self, *args: Any) -> None:
        """
        Set the *Colour* safe / symmetrical power function enabled state
        upon exiting the context manager.
        """

        set_spow_enable(self._previous_state)

    def __call__(self, function: Callable) -> Callable:
        """Call the wrapped definition."""

        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with self:
                return function(*args, **kwargs)

        return wrapper


@typing.overload
def spow(a: float | DTypeFloat, p: float | DTypeFloat) -> DTypeFloat: ...
@typing.overload
def spow(a: NDArray, p: ArrayLike) -> NDArrayFloat: ...
@typing.overload
def spow(a: ArrayLike, p: NDArray) -> NDArrayFloat: ...
@typing.overload
def spow(a: ArrayLike, p: ArrayLike) -> DTypeFloat | NDArrayFloat: ...
def spow(a: ArrayLike, p: ArrayLike) -> DTypeFloat | NDArrayFloat:
    """
    Raise specified array :math:`a` to the power :math:`p` as follows:
    :math:`\\text{sign}(a) \\cdot |a|^p`.

    This definition avoids NaN generation when array :math:`a` is negative
    and power :math:`p` is fractional. This behaviour can be enabled or
    disabled with the :func:`colour.algebra.set_spow_enable` definition or
    with the :func:`spow_enable` context manager.

    Parameters
    ----------
    a
        Array :math:`a`.
    p
        Power :math:`p`.

    Returns
    -------
    :class:`np.float` or :class:`numpy.ndarray`
        Array :math:`a` safely raised to the power :math:`p`.

    Examples
    --------
    >>> np.power(-2, 0.15)
    nan
    >>> spow(-2, 0.15)  # doctest: +ELLIPSIS
    -1.1095694...
    >>> spow(0, 0)
    0.0
    """

    if not _SPOW_ENABLED:
        return np.power(a, p)

    a = as_float_array(a)
    p = as_float_array(p)

    a_p = np.sign(a) * np.abs(a) ** p

    return as_float(0 if a_p.ndim == 0 and np.isnan(a_p) else a_p)


def normalise_vector(a: ArrayLike) -> NDArrayFloat:
    """
    Normalise the specified vector :math:`a`.

    The normalisation process scales the vector to have unit length, ensuring
    that the magnitude of the resulting vector equals 1.

    Parameters
    ----------
    a
        Vector :math:`a` to normalise.

    Returns
    -------
    :class:`numpy.ndarray`
        Normalised vector :math:`a` with unit length.

    Examples
    --------
    >>> a = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> normalise_vector(a)  # doctest: +ELLIPSIS
    array([ 0.8419703...,  0.4972256...,  0.2094102...])
    """

    a = as_float_array(a)

    with sdiv_mode():
        return sdiv(a, np.linalg.norm(a))


def normalise_maximum(
    a: ArrayLike,
    axis: int | None = None,
    factor: float = 1,
    clip: bool = True,
) -> NDArrayFloat:
    """
    Normalise specified array :math:`a` values by :math:`a` maximum value
    and optionally clip them between [0, factor].

    Parameters
    ----------
    a
        Array :math:`a` to normalise.
    axis
        Normalization axis.
    factor
        Normalization factor.
    clip
        Clip values to domain [0, 'factor'].

    Returns
    -------
    :class:`numpy.ndarray`
        Maximum normalised array :math:`a`.

    Examples
    --------
    >>> a = np.array([0.48222001, 0.31654775, 0.22070353])
    >>> normalise_maximum(a)  # doctest: +ELLIPSIS
    array([ 1.        ,  0.6564384...,  0.4576822...])
    """

    a = as_float_array(a)

    maximum = np.max(a, axis=axis)

    with sdiv_mode():
        a = a * sdiv(1, maximum[..., None]) * factor

    return np.clip(a, 0, factor) if clip else a


def vecmul(m: ArrayLike, v: ArrayLike) -> NDArrayFloat:
    """
    Perform batched multiplication between the matrix array :math:`m` and
    vector array :math:`v`.

    This function is equivalent to :func:`numpy.matmul` but specifically
    designed for vector multiplication by a matrix. Vector dimensionality is
    automatically increased to enable broadcasting. The operation can be
    expressed using :func:`numpy.einsum` with subscripts
    *'...ij,...j->...i'*.

    Parameters
    ----------
    m
        Matrix array :math:`m`.
    v
        Vector array :math:`v`.

    Returns
    -------
    :class:`numpy.ndarray`
        Multiplied vector array :math:`v`.

    Examples
    --------
    >>> m = np.array(
    ...     [
    ...         [0.7328, 0.4296, -0.1624],
    ...         [-0.7036, 1.6975, 0.0061],
    ...         [0.0030, 0.0136, 0.9834],
    ...     ]
    ... )
    >>> m = np.reshape(np.tile(m, (6, 1)), (6, 3, 3))
    >>> v = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> v = np.tile(v, (6, 1))
    >>> vecmul(m, v)  # doctest: +ELLIPSIS
    array([[ 0.1954094...,  0.0620396...,  0.0527952...],
           [ 0.1954094...,  0.0620396...,  0.0527952...],
           [ 0.1954094...,  0.0620396...,  0.0527952...],
           [ 0.1954094...,  0.0620396...,  0.0527952...],
           [ 0.1954094...,  0.0620396...,  0.0527952...],
           [ 0.1954094...,  0.0620396...,  0.0527952...]])
    """

    return np.matmul(as_float_array(m), as_float_array(v)[..., None]).squeeze(-1)


def euclidean_distance(a: ArrayLike, b: ArrayLike) -> NDArrayFloat:
    """
    Calculate the *Euclidean* distance between the specified point arrays
    :math:`a` and :math:`b`.

    For a two-dimensional space, the metric is as follows:

    :math:`E_D = [(x_a - x_b)^2 + (y_a - y_b)^2]^{1/2}`

    Parameters
    ----------
    a
        Point array :math:`a`.
    b
        Point array :math:`b`.

    Returns
    -------
    :class:`numpy.float64` or :class:`numpy.ndarray`
        *Euclidean* distance between the two point arrays.

    Examples
    --------
    >>> a = np.array([100.00000000, 21.57210357, 272.22819350])
    >>> b = np.array([100.00000000, 426.67945353, 72.39590835])
    >>> euclidean_distance(a, b)  # doctest: +ELLIPSIS
    451.7133019...
    """

    return as_float(np.linalg.norm(as_float_array(a) - as_float_array(b), axis=-1))


def manhattan_distance(a: ArrayLike, b: ArrayLike) -> NDArrayFloat:
    """
    Compute the *Manhattan* (or *City-Block*) distance between point array
    :math:`a` and point array :math:`b`.

    For a two-dimensional space, the metric is defined as:

    :math:`M_D = |x_a - x_b| + |y_a - y_b|`

    Parameters
    ----------
    a
        Point array :math:`a`.
    b
        Point array :math:`b`.

    Returns
    -------
    :class:`np.float` or :class:`numpy.ndarray`
        *Manhattan* distance.

    Examples
    --------
    >>> a = np.array([100.00000000, 21.57210357, 272.22819350])
    >>> b = np.array([100.00000000, 426.67945353, 72.39590835])
    >>> manhattan_distance(a, b)  # doctest: +ELLIPSIS
    604.9396351...
    """

    return as_float(np.sum(np.abs(as_float_array(a) - as_float_array(b)), axis=-1))


def linear_conversion(
    a: ArrayLike, old_range: ArrayLike, new_range: ArrayLike
) -> NDArrayFloat:
    """
    Perform simple linear conversion of the specified array :math:`a` between the
    old and new ranges.

    Parameters
    ----------
    a
        Array :math:`a` to perform the linear conversion onto.
    old_range
        Old range.
    new_range
        New range.

    Returns
    -------
    :class:`numpy.ndarray`
        Linear conversion result.

    Examples
    --------
    >>> a = np.linspace(0, 1, 10)
    >>> linear_conversion(a, np.array([0, 1]), np.array([1, 10]))
    array([  1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.])
    """

    a = as_float_array(a)

    in_min, in_max = tsplit(old_range)
    out_min, out_max = tsplit(new_range)

    return ((a - in_min) / (in_max - in_min)) * (out_max - out_min) + out_min


def linstep_function(
    x: ArrayLike,
    a: ArrayLike = 0,
    b: ArrayLike = 1,
    clip: bool = False,
) -> NDArrayFloat:
    """
    Perform linear interpolation between specified arrays :math:`a` and
    :math:`b` using array :math:`x`.

    Parameters
    ----------
    x
        Array :math:`x` containing values to use for interpolation between
        array :math:`a` and array :math:`b`.
    a
        Array :math:`a`, the start of the interpolation range.
    b
        Array :math:`b`, the end of the interpolation range.
    clip
        Whether to clip the output values to range [:math:`a`, :math:`b`].

    Returns
    -------
    :class:`numpy.ndarray`
        Linear interpolation result.

    Examples
    --------
    >>> a = 0
    >>> b = 2
    >>> linstep_function(0.5, a, b)
    1.0
    """

    x = as_float_array(x)
    a = as_float_array(a)
    b = as_float_array(b)

    y = (1.0 - x) * a + x * b

    return np.clip(y, a, b) if clip else y


lerp = linstep_function


def smoothstep_function(
    x: ArrayLike,
    a: ArrayLike = 0,
    b: ArrayLike = 1,
    clip: bool = False,
) -> NDArrayFloat:
    """
    Apply the *smoothstep* cubic Hermite interpolation function to
    array :math:`x`.

    The *smoothstep* function creates a smooth S-shaped curve between
    specified edge values, commonly used for smooth transitions in
    colour interpolation and rendering operations.

    Parameters
    ----------
    x
        Input array :math:`x` containing values to be transformed.
    a
        Lower edge value for the interpolation domain.
    b
        Upper edge value for the interpolation domain.
    clip
        Whether to normalize and constrain input values to the domain
        [:math:`a`, :math:`b`] before applying the *smoothstep* function.

    Returns
    -------
    :class:`numpy.ndarray`
        Transformed array with values smoothly interpolated using the
        cubic Hermite polynomial :math:`3x^2 - 2x^3`.

    Examples
    --------
    >>> x = np.linspace(-2, 2, 5)
    >>> smoothstep_function(x, -2, 2, clip=True)
    array([ 0.     ,  0.15625,  0.5    ,  0.84375,  1.     ])
    """

    x = as_float_array(x)
    a = as_float_array(a)
    b = as_float_array(b)

    i = np.clip((x - a) / (b - a), 0, 1) if clip else x

    return (i**2) * (3.0 - 2.0 * i)


smooth = smoothstep_function


def is_identity(a: ArrayLike) -> bool:
    """
    Determine whether the specified array :math:`a` is an identity matrix.

    An identity matrix is a square matrix with ones on the main diagonal
    and zeros elsewhere, satisfying :math:`I \\cdot A = A \\cdot I = A`
    for any compatible matrix :math:`A`.

    Parameters
    ----------
    a
        Array :math:`a` to test for identity matrix properties.

    Returns
    -------
    :class:`bool`
        Whether the specified array :math:`a` is an identity matrix.

    Examples
    --------
    >>> is_identity(np.reshape(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]), (3, 3)))
    True
    >>> is_identity(np.reshape(np.array([1, 2, 0, 0, 1, 0, 0, 0, 1]), (3, 3)))
    False
    """

    return np.array_equal(np.identity(len(np.diag(a))), a)


def eigen_decomposition(
    a: ArrayLike,
    eigen_w_v_count: int | None = None,
    descending_order: bool = True,
    covariance_matrix: bool = False,
) -> Tuple[NDArrayFloat, NDArrayFloat]:
    """
    Compute the eigenvalues :math:`w` and eigenvectors :math:`v` of the
    specified array :math:`a` in the specified order.

    Parameters
    ----------
    a
        Array to compute the eigenvalues :math:`w` and eigenvectors :math:`v`
        for.
    eigen_w_v_count
        Number of eigenvalues :math:`w` and eigenvectors :math:`v` to return.
    descending_order
        Whether to return the eigenvalues :math:`w` and eigenvectors :math:`v`
        in descending order.
    covariance_matrix
        Whether to compute the eigenvalues :math:`w` and eigenvectors
        :math:`v` of the array :math:`a` covariance matrix
        :math:`A = a^T \\cdot a`.

    Returns
    -------
    :class:`tuple`
        Tuple of eigenvalues :math:`w` and eigenvectors :math:`v`. The
        eigenvalues are in the specified order, each repeated according to
        its multiplicity. The column ``v[:, i]`` is the normalized eigenvector
        corresponding to the eigenvalue ``w[i]``.

    Examples
    --------
    >>> a = np.diag([1, 2, 3])
    >>> w, v = eigen_decomposition(a)
    >>> w
    array([ 3.,  2.,  1.])
    >>> v
    array([[ 0.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 1.,  0.,  0.]])
    >>> w, v = eigen_decomposition(a, 1)
    >>> w
    array([ 3.])
    >>> v
    array([[ 0.],
           [ 0.],
           [ 1.]])
    >>> w, v = eigen_decomposition(a, descending_order=False)
    >>> w
    array([ 1.,  2.,  3.])
    >>> v
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    >>> w, v = eigen_decomposition(a, covariance_matrix=True)
    >>> w
    array([ 9.,  4.,  1.])
    >>> v
    array([[ 0.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 1.,  0.,  0.]])
    """

    A = as_float_array(a)

    if covariance_matrix:
        A = np.dot(np.transpose(A), A)

    w, v = np.linalg.eigh(A)

    if eigen_w_v_count is not None:
        w = w[-eigen_w_v_count:]
        v = v[..., -eigen_w_v_count:]

    if descending_order:
        w = np.flipud(w)
        v = np.fliplr(v)

    return w, v
