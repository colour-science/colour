"""
Common Utilities
================

Defines the common algebra utilities objects that don't fall in any specific
category.
"""

from __future__ import annotations

import functools
import numpy as np

from colour.hints import (
    Any,
    ArrayLike,
    Callable,
    Literal,
    NDArrayFloat,
    Tuple,
    cast,
)
from colour.utilities import (
    as_float_array,
    as_float,
    optional,
    tsplit,
    validate_method,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
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
    "vector_dot",
    "matrix_dot",
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
] = "Ignore Zero Conversion"
"""
Global variable storing the current *Colour* safe division function mode.
"""


def get_sdiv_mode() -> (
    Literal[
        "Numpy",
        "Ignore",
        "Warning",
        "Raise",
        "Ignore Zero Conversion",
        "Warning Zero Conversion",
        "Ignore Limit Conversion",
        "Warning Limit Conversion",
    ]
):
    """
    Return *Colour* safe division mode.

    Returns
    -------
    :class:`str`
        *Colour* safe division mode, see :func:`colour.algebra.sdiv` definition
        for an explanation about the possible modes.

    Examples
    --------
    >>> with sdiv_mode("Numpy"):
    ...     get_sdiv_mode()
    ...
    'numpy'
    >>> with sdiv_mode("Ignore Zero Conversion"):
    ...     get_sdiv_mode()
    ...
    'ignore zero conversion'
    """

    return _SDIV_MODE


def set_sdiv_mode(
    mode: Literal[
        "Numpy",
        "Ignore",
        "Warning",
        "Raise",
        "Ignore Zero Conversion",
        "Warning Zero Conversion",
        "Ignore Limit Conversion",
        "Warning Limit Conversion",
    ]
    | str
):
    """
    Set *Colour* safe division function mode.

    Parameters
    ----------
    mode
        *Colour* safe division mode, see :func:`colour.algebra.sdiv` definition
        for an explanation about the possible modes.

    Examples
    --------
    >>> with sdiv_mode(get_sdiv_mode()):
    ...     print(get_sdiv_mode())
    ...     set_sdiv_mode("Raise")
    ...     print(get_sdiv_mode())
    ...
    ignore zero conversion
    raise
    """

    global _SDIV_MODE

    _SDIV_MODE = cast(
        Literal[
            "Numpy",
            "Ignore",
            "Warning",
            "Raise",
            "Ignore Zero Conversion",
            "Warning Zero Conversion",
            "Ignore Limit Conversion",
            "Warning Limit Conversion",
        ],
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
            ),
        ),
    )


class sdiv_mode:
    """
    Define a context manager and decorator temporarily setting *Colour* safe
    division function mode.

    Parameters
    ----------
    mode
       *Colour* safe division function mode, see :func:`colour.algebra.sdiv`
       definition for an explanation about the possible modes.
    """

    def __init__(
        self,
        mode: Literal[
            "Numpy",
            "Ignore",
            "Warning",
            "Raise",
            "Ignore Zero Conversion",
            "Warning Zero Conversion",
            "Ignore Limit Conversion",
            "Warning Limit Conversion",
        ]
        | None = None,
    ) -> None:
        self._mode = optional(mode, get_sdiv_mode())
        self._previous_mode = get_sdiv_mode()

    def __enter__(self) -> sdiv_mode:
        """
        Set the *Colour* safe division function mode upon entering the context
        manager.
        """

        set_sdiv_mode(self._mode)

        return self

    def __exit__(self, *args: Any):
        """
        Set the *Colour* safe division function mode upon exiting the context
        manager.
        """

        set_sdiv_mode(self._previous_mode)

    def __call__(self, function: Callable) -> Callable:
        """Call the wrapped definition."""

        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with self:
                return function(*args, **kwargs)

        return wrapper


def sdiv(a: ArrayLike, b: ArrayLike) -> NDArrayFloat:
    """
    Divide given array :math:`b` with array :math:`b` while handling
    zero-division.

    This definition avoids NaNs and +/- infs generation when array :math:`b`
    is equal to zero. This behaviour can be controlled with the
    :func:`colour.algebra.set_sdiv_mode` definition or with the
    :func:`sdiv_mode` context manager. The following modes are available:

    -   ``Numpy``: The current *Numpy* zero-division handling occurs.
    -   ``Ignore``: Zero-division occurs silently.
    -   ``Warning``: Zero-division occurs with a warning.
    -   ``Ignore Zero Conversion``: Zero-division occurs silently and NaNs or
        +/- infs values are converted to zeros. See :func:`numpy.nan_to_num`
        definition for more details.
    -   ``Warning Zero Conversion``: Zero-division occurs with a warning and
        NaNs or +/- infs values are converted to zeros. See
        :func:`numpy.nan_to_num` definition for more details.
    -   ``Ignore Limit Conversion``: Zero-division occurs silently and
        NaNs or +/- infs values are converted to zeros or the largest +/-
        finite floating point values representable by the division result
        :class:`numpy.dtype`. See :func:`numpy.nan_to_num` definition for more
        details.
    -   ``Warning Limit Conversion``: Zero-division occurs  with a warning and
        NaNs or +/- infs values are converted to zeros or the largest +/-
        finite floating point values representable by the division result
        :class:`numpy.dtype`.

    Parameters
    ----------
    a
        Numerator array :math:`a`.
    b
        Denominator array :math:`b`.

    Returns
    -------
    :class:`np.float` or :class:`numpy.ndarray`
        Array :math:`b` safely divided by :math:`a`.

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
    ...
    FloatingPointError('divide by zero encountered in...divide')
    >>> with sdiv_mode("Ignore Zero Conversion"):
    ...     sdiv(a, b)
    ...
    array([ 0.,  1.,  0.])
    >>> with sdiv_mode("Warning Zero Conversion"):
    ...     sdiv(a, b)
    ...
    array([ 0.,  1.,  0.])
    >>> with sdiv_mode("Ignore Limit Conversion"):
    ...     sdiv(a, b)  # doctest: +SKIP
    ...
    array([  0.00000000e+000,   1.00000000e+000,   1.79769313e+308])
    >>> with sdiv_mode("Warning Limit Conversion"):
    ...     sdiv(a, b)  # doctest: +SKIP
    ...
    array([  0.00000000e+000,   1.00000000e+000,   1.79769313e+308])
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
    ...
    False
    >>> with spow_enable(True):
    ...     is_spow_enabled()
    ...
    True
    """

    return _SPOW_ENABLED


def set_spow_enable(enable: bool):
    """
    Set *Colour* safe / symmetrical power function enabled state.

    Parameters
    ----------
    enable
        Whether to enable *Colour* safe / symmetrical power function.

    Examples
    --------
    >>> with spow_enable(is_spow_enabled()):
    ...     print(is_spow_enabled())
    ...     set_spow_enable(False)
    ...     print(is_spow_enabled())
    ...
    True
    False
    """

    global _SPOW_ENABLED

    _SPOW_ENABLED = enable


class spow_enable:
    """
    Define a context manager and decorator temporarily setting *Colour* safe /
    symmetrical power function enabled state.

    Parameters
    ----------
    enable
        Whether to enable or disable *Colour* safe / symmetrical power
        function.
    """

    def __init__(self, enable: bool) -> None:
        self._enable = enable
        self._previous_state = is_spow_enabled()

    def __enter__(self) -> spow_enable:
        """
        Set the *Colour* safe / symmetrical power function enabled state
        upon entering the context manager.
        """

        set_spow_enable(self._enable)

        return self

    def __exit__(self, *args: Any):
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


def spow(a: ArrayLike, p: ArrayLike) -> NDArrayFloat:
    """
    Raise given array :math:`a` to the power :math:`p` as follows:
    :math:`sign(a) * |a|^p`.

    This definition avoids NaNs generation when array :math:`a` is negative and
    the power :math:`p` is fractional. This behaviour can be enabled or
    disabled with the :func:`colour.algebra.set_spow_enable` definition or with
    the :func:`spow_enable` context manager.

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
    Normalise given vector :math:`a`.

    Parameters
    ----------
    a
        Vector :math:`a` to normalise.

    Returns
    -------
    :class:`numpy.ndarray`
        Normalised vector :math:`a`.

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
    Normalise given array :math:`a` values by :math:`a` maximum value and
    optionally clip them between.

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


def vector_dot(m: ArrayLike, v: ArrayLike) -> NDArrayFloat:
    """
    Perform the dot product of the matrix array :math:`m` with the vector
    array :math:`v`.

    This definition is a convenient wrapper around :func:`np.einsum` with the
    following subscripts: *'...ij,...j->...i'*.

    Parameters
    ----------
    m
        Matrix array :math:`m`.
    v
        Vector array :math:`v`.

    Returns
    -------
    :class:`numpy.ndarray`
        Transformed vector array :math:`v`.

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
    >>> vector_dot(m, v)  # doctest: +ELLIPSIS
    array([[ 0.1954094...,  0.0620396...,  0.0527952...],
           [ 0.1954094...,  0.0620396...,  0.0527952...],
           [ 0.1954094...,  0.0620396...,  0.0527952...],
           [ 0.1954094...,  0.0620396...,  0.0527952...],
           [ 0.1954094...,  0.0620396...,  0.0527952...],
           [ 0.1954094...,  0.0620396...,  0.0527952...]])
    """

    return np.einsum("...ij,...j->...i", as_float_array(m), as_float_array(v))


def matrix_dot(a: ArrayLike, b: ArrayLike) -> NDArrayFloat:
    """
    Perform the dot product of the matrix array :math:`a` with the matrix
    array :math:`b`.

    This definition is a convenient wrapper around :func:`np.einsum` with the
    following subscripts: *'...ij,...jk->...ik'*.

    Parameters
    ----------
    a
        Matrix array :math:`a`.
    b
        Matrix array :math:`b`.

    Returns
    -------
    :class:`numpy.ndarray`

    Examples
    --------
    >>> a = np.array(
    ...     [
    ...         [0.7328, 0.4296, -0.1624],
    ...         [-0.7036, 1.6975, 0.0061],
    ...         [0.0030, 0.0136, 0.9834],
    ...     ]
    ... )
    >>> a = np.reshape(np.tile(a, (6, 1)), (6, 3, 3))
    >>> b = a
    >>> matrix_dot(a, b)  # doctest: +ELLIPSIS
    array([[[ 0.2342420...,  1.0418482..., -0.2760903...],
            [-1.7099407...,  2.5793226...,  0.1306181...],
            [-0.0044203...,  0.0377490...,  0.9666713...]],
    <BLANKLINE>
           [[ 0.2342420...,  1.0418482..., -0.2760903...],
            [-1.7099407...,  2.5793226...,  0.1306181...],
            [-0.0044203...,  0.0377490...,  0.9666713...]],
    <BLANKLINE>
           [[ 0.2342420...,  1.0418482..., -0.2760903...],
            [-1.7099407...,  2.5793226...,  0.1306181...],
            [-0.0044203...,  0.0377490...,  0.9666713...]],
    <BLANKLINE>
           [[ 0.2342420...,  1.0418482..., -0.2760903...],
            [-1.7099407...,  2.5793226...,  0.1306181...],
            [-0.0044203...,  0.0377490...,  0.9666713...]],
    <BLANKLINE>
           [[ 0.2342420...,  1.0418482..., -0.2760903...],
            [-1.7099407...,  2.5793226...,  0.1306181...],
            [-0.0044203...,  0.0377490...,  0.9666713...]],
    <BLANKLINE>
           [[ 0.2342420...,  1.0418482..., -0.2760903...],
            [-1.7099407...,  2.5793226...,  0.1306181...],
            [-0.0044203...,  0.0377490...,  0.9666713...]]])
    """

    return np.einsum(
        "...ij,...jk->...ik", as_float_array(a), as_float_array(b)
    )


def euclidean_distance(a: ArrayLike, b: ArrayLike) -> NDArrayFloat:
    """
    Return the *Euclidean* distance between point array :math:`a` and point
    array :math:`b`.

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
    :class:`np.float` or :class:`numpy.ndarray`
        *Euclidean* distance.

    Examples
    --------
    >>> a = np.array([100.00000000, 21.57210357, 272.22819350])
    >>> b = np.array([100.00000000, 426.67945353, 72.39590835])
    >>> euclidean_distance(a, b)  # doctest: +ELLIPSIS
    451.7133019...
    """

    return as_float(
        np.linalg.norm(as_float_array(a) - as_float_array(b), axis=-1)
    )


def manhattan_distance(a: ArrayLike, b: ArrayLike) -> NDArrayFloat:
    """
    Return the *Manhattan* (or *City-Block*) distance between point array
    :math:`a` and point array :math:`b`.

    For a two-dimensional space, the metric is as follows:

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

    return as_float(
        np.sum(np.abs(as_float_array(a) - as_float_array(b)), axis=-1)
    )


def linear_conversion(
    a: ArrayLike, old_range: ArrayLike, new_range: ArrayLike
) -> NDArrayFloat:
    """
    Perform a simple linear conversion of given array :math:`a` between the
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
    Perform a simple linear interpolation between given array :math:`a` and
    array :math:`b` using :math:`x` array.

    Parameters
    ----------
    x
        Array :math:`x` value to use to interpolate between array :math:`a` and
        array :math:`b`.
    a
        Array :math:`a`, the start of the range in which to interpolate.
    b
        Array :math:`b`, the end of the range in which to interpolate.
    clip
        Whether to clip the output values to range [``a``, ``b``].

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
    Evaluate the *smoothstep* sigmoid-like function on array :math:`x`.

    Parameters
    ----------
    x
        Array :math:`x`.
    a
        Low input domain limit, i.e. the left edge.
    b
        High input domain limit, i.e. the right edge.
    clip
        Whether to scale, bias and clip input values to domain [``a``, ``b``].

    Returns
    -------
    :class:`numpy.ndarray`
        Array :math:`x` after *smoothstep* sigmoid-like function evaluation.

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
    Return whether :math:`a` array is an identity matrix.

    Parameters
    ----------
    a
        Array :math:`a` to test.

    Returns
    -------
    :class:`bool`
        Whether :math:`a` array is an identity matrix.

    Examples
    --------
    >>> is_identity(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).reshape(3, 3))
    True
    >>> is_identity(np.array([1, 2, 0, 0, 1, 0, 0, 0, 1]).reshape(3, 3))
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
    Return the eigen-values :math:`w` and eigen-vectors :math:`v` of given
    array :math:`a` in given order.

    Parameters
    ----------
    a
        Array to return the eigen-values :math:`w` and eigen-vectors :math:`v`
        for
    eigen_w_v_count
        Eigen-values :math:`w` and eigen-vectors :math:`v` count.
    descending_order
        Whether to return the eigen-values :math:`w` and eigen-vectors :math:`v`
        in descending order.
    covariance_matrix
        Whether to compute the eigen-values :math:`w` and eigen-vectors
        :math:`v` of the array :math:`a` covariance matrix
        :math:`A =a^T\\cdot a`.

    Returns
    -------
    :class:`tuple`
        Tuple of eigen-values :math:`w` and eigen-vectors :math:`v`. The
        eigenv-alues are in given order, each repeated according to
        its multiplicity. The column ``v[:, i]`` is the normalized eigen-vector
        corresponding to the eige-nvalue ``w[i]``.

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
