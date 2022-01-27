# -*- coding: utf-8 -*-
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
    Boolean,
    Callable,
    Floating,
    FloatingOrArrayLike,
    FloatingOrNDArray,
    Integer,
    NDArray,
    Optional,
)
from colour.utilities import as_float_array, as_float, tsplit

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'is_spow_enabled',
    'set_spow_enable',
    'spow_enable',
    'spow',
    'normalise_maximum',
    'vector_dot',
    'matrix_dot',
    'linear_conversion',
    'linstep_function',
    'lerp',
    'smoothstep_function',
    'smooth',
    'is_identity',
]

# TODO: Annotate with "bool" when Python 3.7 is dropped.
_SPOW_ENABLED = True
"""
Global variable storing the current *Colour* safe / symmetrical power function
enabled state.
"""


def is_spow_enabled() -> bool:
    """
    Returns whether *Colour* safe / symmetrical power function is enabled.

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


def set_spow_enable(enable: bool):
    """
    Sets *Colour* safe / symmetrical power function enabled state.

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
    True
    False
    """

    global _SPOW_ENABLED

    _SPOW_ENABLED = enable


class spow_enable:
    """
    A context manager and decorator temporarily setting *Colour* safe /
    symmetrical power function enabled state.

    Parameters
    ----------
    enable
        Whether to enable or disable *Colour* safe / symmetrical power
        function.
    """

    def __init__(self, enable: bool):
        self._enable = enable
        self._previous_state = is_spow_enabled()

    def __enter__(self) -> spow_enable:
        """
        Called upon entering the context manager and decorator.
        """

        set_spow_enable(self._enable)

        return self

    def __exit__(self, *args: Any):
        """
        Called upon exiting the context manager and decorator.
        """

        set_spow_enable(self._previous_state)

    def __call__(self, function: Callable) -> Callable:
        """
        Calls the wrapped definition.
        """

        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with self:
                return function(*args, **kwargs)

        return wrapper


def spow(a: FloatingOrArrayLike, p: FloatingOrArrayLike) -> FloatingOrNDArray:
    """
    Raises given array :math:`a` to the power :math:`p` as follows:
    :math:`sign(a) * |a|^p`.

    This definition avoids NaNs generation when array :math:`a` is negative and
    the power :math:`p` is fractional. This behaviour can be enabled or
    disabled with the :func:`colour.algebra.set_spow_enable` definition or with
    the :func:`spow_enable` context manager.

    Parameters
    ----------------
    a
        Array :math:`a`.
    p
        Power :math:`p`.

    Returns
    -------
    :class:`np.floating` or :class:`numpy.ndarray`
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

    a = np.atleast_1d(a)
    p = as_float_array(p)

    a_p = np.sign(a) * np.abs(a) ** p

    a_p[np.isnan(a_p)] = 0

    return as_float(a_p)


def normalise_maximum(a: ArrayLike,
                      axis: Optional[Integer] = None,
                      factor: Floating = 1,
                      clip: Boolean = True) -> NDArray:
    """
    Normalises given array :math:`a` values by :math:`a` maximum value and
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
    a = a * (1 / maximum[..., np.newaxis]) * factor

    return np.clip(a, 0, factor) if clip else a


def vector_dot(m: ArrayLike, v: ArrayLike) -> NDArray:
    """
    Convenient wrapper around :func:`np.einsum` with the following subscripts:
    *'...ij,...j->...i'*.

    It performs the dot product of the matrix array :math:`m` with the vector
    array :math:`v`.

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
    ...     [[0.7328, 0.4296, -0.1624],
    ...      [-0.7036, 1.6975, 0.0061],
    ...      [0.0030, 0.0136, 0.9834]]
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

    return np.einsum('...ij,...j->...i', as_float_array(m), as_float_array(v))


def matrix_dot(a: ArrayLike, b: ArrayLike) -> NDArray:
    """
    Convenient wrapper around :func:`np.einsum` with the following subscripts:
    *'...ij,...jk->...ik'*.

    It performs the dot product of the matrix array :math:`a` with the matrix
    array :math:`b`.

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
    ...     [[0.7328, 0.4296, -0.1624],
    ...      [-0.7036, 1.6975, 0.0061],
    ...      [0.0030, 0.0136, 0.9834]]
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

    return np.einsum('...ij,...jk->...ik', as_float_array(a),
                     as_float_array(b))


def linear_conversion(a: ArrayLike, old_range: ArrayLike,
                      new_range: ArrayLike) -> NDArray:
    """
    Performs a simple linear conversion of given array :math:`a` between the
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


def linstep_function(x: FloatingOrArrayLike,
                     a: FloatingOrArrayLike = 0,
                     b: FloatingOrArrayLike = 1,
                     clip: Boolean = False) -> NDArray:
    """
    Performs a simple linear interpolation between given array :math:`a` and
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

    y = (1 - x) * a + x * b

    return np.clip(y, a, b) if clip else y


lerp = linstep_function


def smoothstep_function(x: FloatingOrArrayLike,
                        a: FloatingOrArrayLike = 0,
                        b: FloatingOrArrayLike = 1,
                        clip: Boolean = False) -> NDArray:
    """
    Evaluates the *smoothstep* sigmoid-like function on array :math:`x`.

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

    return (i ** 2) * (3 - 2 * i)


smooth = smoothstep_function


def is_identity(a: ArrayLike) -> Boolean:
    """
    Returns whether :math:`a` array is an identity matrix.

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
