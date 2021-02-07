# -*- coding: utf-8 -*-
"""
Common Utilities
================

Defines common algebra utilities objects that don't fall in any specific
category.
"""

import functools
import numpy as np

from colour.utilities import as_float_array, as_float, tsplit

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'is_spow_enabled', 'set_spow_enable', 'spow_enable', 'spow',
    'smoothstep_function', 'normalise_maximum', 'vector_dot', 'matrix_dot',
    'linear_conversion', 'linstep_function', 'is_identity'
]

_SPOW_ENABLED = True
"""
Global variable storing the current *Colour* safe / symmetrical power function
enabled state.

_SPOW_ENABLED : bool
"""


def is_spow_enabled():
    """
    Returns whether *Colour* safe / symmetrical power function is enabled.

    Returns
    -------
    bool
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


def set_spow_enable(enable):
    """
    Sets *Colour* safe / symmetrical power function enabled state.

    Parameters
    ----------
    enable : bool
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
    enable : bool
        Whether to enable or disable *Colour* safe / symmetrical power
        function.
    """

    def __init__(self, enable):
        self._enable = enable
        self._previous_state = is_spow_enabled()

    def __enter__(self):
        """
        Called upon entering the context manager and decorator.
        """

        set_spow_enable(self._enable)

        return self

    def __exit__(self, *args):
        """
        Called upon exiting the context manager and decorator.
        """

        set_spow_enable(self._previous_state)

    def __call__(self, function):
        """
        Calls the wrapped definition.
        """

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            with self:
                return function(*args, **kwargs)

        return wrapper


def spow(a, p):
    """
    Raises given array :math:`a` to the power :math:`p` as follows:
    :math:`sign(a) * |a|^p`.

    This definition avoids NaNs generation when array :math:`a` is negative and
    the power :math:`p` is fractional. This behaviour can be enabled or
    disabled with the :func:`colour.algebra.set_spow_enable` definition or with
    the :func:`spow_enable` context manager.

    Parameters
    ----------------
    a : numeric or array_like
        Array :math:`a`.
    p : numeric or array_like
        Power :math:`p`.

    Returns
    -------
    numeric or ndarray
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


def normalise_maximum(a, axis=None, factor=1, clip=True):
    """
    Normalises given *array_like* :math:`a` variable values by :math:`a`
    variable maximum value and optionally clip them between.

    Parameters
    ----------
    a : array_like
        :math:`a` variable to normalise.
    axis : numeric, optional
        Normalization axis.
    factor : numeric, optional
        Normalization factor.
    clip : bool, optional
        Clip values to domain [0, 'factor'].

    Returns
    -------
    ndarray
        Maximum normalised :math:`a` variable.

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


def vector_dot(m, v):
    """
    Convenient wrapper around :func:`np.einsum` with the following subscripts:
    *'...ij,...j->...i'*.

    It performs the dot product of two arrays where *m* parameter is expected
    to be an array of 3x3 matrices and parameter *v* an array of vectors.

    Parameters
    ----------
    m : array_like
        Array of 3x3 matrices.
    v : array_like
        Array of vectors.

    Returns
    -------
    ndarray

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

    m = as_float_array(m)
    v = as_float_array(v)

    return np.einsum('...ij,...j->...i', m, v)


def matrix_dot(a, b):
    """
    Convenient wrapper around :func:`np.einsum` with the following subscripts:
    *'...ij,...jk->...ik'*.

    It performs the dot product of two arrays where *a* parameter is expected
    to be an array of 3x3 matrices and parameter *b* another array of of 3x3
    matrices.

    Parameters
    ----------
    a : array_like
        Array of 3x3 matrices.
    b : array_like
        Array of 3x3 matrices.
    dtype : object
        Type to use for conversion, default to the type defined by the
        :attr:`colour.constant.DEFAULT_FLOAT_DTYPE` attribute.

    Returns
    -------
    ndarray

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

    a = as_float_array(a)
    b = as_float_array(b)

    return np.einsum('...ij,...jk->...ik', a, b)


def linear_conversion(a, old_range, new_range):
    """
    Performs a simple linear conversion of given array between the old and new
    ranges.

    Parameters
    ----------
    a : array_like
        Array to perform the linear conversion onto.
    old_range : array_like
        Old range.
    new_range : array_like
        New range.

    Returns
    -------
    ndarray
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


def linstep_function(x, a=0, b=1, clip=False):
    """
    Performs a simple linear interpolation between given array :math:`a` and
    array :math:`b` using :math:`x` array.

    Parameters
    ----------
    x : array_like
        Array :math:`x` value to use to interpolate between array :math:`a` and
        array :math:`b`.
    a : array_like
        Array :math:`a`, the start of the range in which to interpolate.
    b : array_like
        Array :math:`b`, the end of the range in which to interpolate.
    clip : bool, optional
        Whether to clip the output values to range [a, b].

    Returns
    -------
    ndarray
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


def smoothstep_function(x, a=0, b=1, clip=False):
    """
    Evaluates the *smoothstep* sigmoid-like function on array :math:`x`.

    Parameters
    ----------
    x : numeric or array_like
        Array :math:`x`.
    a : numeric, optional
        Low input domain limit, i.e. the left edge.
    b : numeric, optional
        High input domain limit, i.e. the right edge.
    clip : bool, optional
        Whether to scale, bias and clip input values to domain [0, 1].

    Returns
    -------
    array_like
        Array :math:`x` after *smoothstep* sigmoid-like function evaluation.

    Examples
    --------
    >>> x = np.linspace(-2, 2, 5)
    >>> smoothstep_function(x, -2, 2, clip=True)
    array([ 0.     ,  0.15625,  0.5    ,  0.84375,  1.     ])
    """

    x = as_float_array(x)

    i = np.clip((x - a) / (b - a), 0, 1) if clip else x

    return (i ** 2) * (3 - 2 * i)


smooth = smoothstep_function


def is_identity(a, n=3):
    """
    Returns if :math:`a` array is an identity matrix.

    Parameters
    ----------
    a : array_like, (N)
        Variable :math:`a` to test.
    n : int, optional
        Matrix dimension.

    Returns
    -------
    bool
        Is identity matrix.

    Examples
    --------
    >>> is_identity(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).reshape(3, 3))
    True
    >>> is_identity(np.array([1, 2, 0, 0, 1, 0, 0, 0, 1]).reshape(3, 3))
    False
    """

    return np.array_equal(np.identity(n), a)
