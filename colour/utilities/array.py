# -*- coding: utf-8 -*-
"""
Array Utilities
===============

Defines array utilities objects.

References
----------
-   :cite:`Castro2014a` : Castro, S. (2014). Numpy: Fastest way of computing
    diagonal for each row of a 2d array. Retrieved August 22, 2014, from
    http://stackoverflow.com/questions/26511401/\
numpy-fastest-way-of-computing-diagonal-for-each-row-of-a-2d-array/\
26517247#26517247
-   :cite:`Yorke2014a` : Yorke, R. (2014). Python: Change format of np.array or
    allow tolerance in in1d function. Retrieved March 27, 2015, from
    http://stackoverflow.com/a/23521245/931625
"""

from __future__ import division, unicode_literals

import numpy as np
import six
import sys
try:  # pragma: no cover
    from collections import Mapping
except ImportError:  # pragma: no cover
    from collections.abc import Mapping

from contextlib import contextmanager

from colour.constants import DEFAULT_FLOAT_DTYPE, DEFAULT_INT_DTYPE, EPSILON
from colour.utilities import suppress_warnings

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'as_array', 'as_int_array', 'as_float_array', 'as_numeric', 'as_int',
    'as_float', 'set_float_precision', 'set_int_precision', 'as_namedtuple',
    'closest_indexes', 'closest', 'normalise_maximum', 'interval',
    'is_uniform', 'in_array', 'tstack', 'tsplit', 'row_as_diagonal',
    'vector_dot', 'matrix_dot', 'orient', 'centroid', 'linear_conversion',
    'lerp', 'fill_nan', 'ndarray_write', 'zeros', 'ones', 'full',
    'index_along_last_axis'
]


def as_array(a, dtype=None):
    """
    Converts given :math:`a` variable to *ndarray* using given type.

    Parameters
    ----------
    a : object
        Variable to convert.
    dtype : object
        Type to use for conversion, default to the type defined by the
        :attr:`colour.constant.DEFAULT_FLOAT_DTYPE` attribute.

    Returns
    -------
    ndarray
        :math:`a` variable converted to *ndarray*.

    Examples
    --------
    >>> as_array([1, 2, 3])
    array([ 1.,  2.,  3.])
    >>> as_array([1, 2, 3], dtype=DEFAULT_INT_DTYPE)  # doctest: +ELLIPSIS
    array([1, 2, 3]...)
    """

    if dtype is None:
        dtype = DEFAULT_FLOAT_DTYPE

    if six.PY3:  # pragma: no cover
        # TODO: Remove when https://github.com/numpy/numpy/issues/5718 is
        # addressed.
        from collections.abc import ValuesView

        if isinstance(a, ValuesView):
            a = list(a)

    return np.asarray(a, dtype)


def as_int_array(a, dtype=None):
    """
    Converts given :math:`a` variable to *ndarray* using given type.

    Parameters
    ----------
    a : object
        Variable to convert.
    dtype : object
        Type to use for conversion, default to the type defined by the
        :attr:`colour.constant.DEFAULT_INT_DTYPE` attribute.

    Returns
    -------
    ndarray
        :math:`a` variable converted to *ndarray*.

    Examples
    --------
    >>> as_int_array([1.0, 2.0, 3.0])  # doctest: +ELLIPSIS
    array([1, 2, 3]...)
    """

    if dtype is None:
        dtype = DEFAULT_INT_DTYPE

    assert dtype in np.sctypes['int'], (
        '"dtype" must be one of the following types: {0}'.format(
            np.sctypes['int']))

    return as_array(a, dtype)


def as_float_array(a, dtype=None):
    """
    Converts given :math:`a` variable to *ndarray* using given type.

    Parameters
    ----------
    a : object
        Variable to convert.
    dtype : object
        Type to use for conversion, default to the type defined by the
        :attr:`colour.constant.DEFAULT_FLOAT_DTYPE` attribute.

    Returns
    -------
    ndarray
        :math:`a` variable converted to *ndarray*.

    Examples
    --------
    >>> as_float_array([1, 2, 3])
    array([ 1.,  2.,  3.])
    """

    if dtype is None:
        dtype = DEFAULT_FLOAT_DTYPE

    assert dtype in np.sctypes['float'], (
        '"dtype" must be one of the following types: {0}'.format(
            np.sctypes['float']))

    return as_array(a, dtype)


def as_numeric(a, dtype=None):
    """
    Converts given :math:`a` variable to *numeric*. In the event where
    :math:`a` cannot be converted, it is passed as is.

    Parameters
    ----------
    a : object
        Variable to convert.
    dtype : object
        Type to use for conversion, default to the type defined by the
        :attr:`colour.constant.DEFAULT_FLOAT_DTYPE` attribute.

    Returns
    -------
    ndarray
        :math:`a` variable converted to *numeric*.

    Examples
    --------
    >>> as_numeric(np.array([1]))
    1.0
    >>> as_numeric(np.arange(10))
    array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.])
    """

    if dtype is None:
        dtype = DEFAULT_FLOAT_DTYPE

    try:
        return dtype(a)
    except (TypeError, ValueError):
        return a


def as_int(a, dtype=None):
    """
    Attempts to converts given :math:`a` variable to *int* using given type.

    Parameters
    ----------
    a : object
        Variable to convert.
    dtype : object
        Type to use for conversion, default to the type defined by the
        :attr:`colour.constant.DEFAULT_INT_DTYPE` attribute. In the event where
        :math:`a` cannot be converted, it is converted to *ndarray* using the
        type defined by :attr:`colour.constant.DEFAULT_INT_DTYPE` attribute.

    Returns
    -------
    ndarray
        :math:`a` variable converted to *numeric*.

    Warnings
    --------
    The behaviour of this definition is different than
    :func:`colour.utilities.as_numeric` definition when it comes to conversion
    failure: the former will forcibly convert :math:`a` variable to *ndarray*
    using the type defined by :attr:`colour.constant.DEFAULT_INT_DTYPE`
    attribute while the later will pass the :math:`a` variable as is.

    Examples
    --------
    >>> as_int(np.array([1]))
    1
    >>> as_int(np.arange(10))  # doctest: +ELLIPSIS
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]...)
    """

    if dtype is None:
        dtype = DEFAULT_INT_DTYPE

    assert dtype in np.sctypes['int'], (
        '"dtype" must be one of the following types: {0}'.format(
            np.sctypes['int']))
    try:
        # TODO: Change to "DEFAULT_INT_DTYPE" when and if
        # https://github.com/numpy/numpy/issues/11956 is addressed.
        return int(a)
    except TypeError:
        return as_int_array(a, dtype)


def as_float(a, dtype=None):
    """
    Converts given :math:`a` variable to *numeric* using given type.

    Parameters
    ----------
    a : object
        Variable to convert.
    dtype : object
        Type to use for conversion, default to the type defined by the
        :attr:`colour.constant.DEFAULT_INT_DTYPE` attribute. In the event where
        :math:`a` cannot be converted, it is converted to *ndarray* using the
        type defined by :attr:`colour.constant.DEFAULT_FLOAT_DTYPE` attribute.

    Returns
    -------
    ndarray
        :math:`a` variable converted to *numeric*.

    Warnings
    --------
    The behaviour of this definition is different than
    :func:`colour.utilities.as_numeric` definition when it comes to conversion
    failure: the former will forcibly convert :math:`a` variable to *ndarray*
    using the type defined by :attr:`colour.constant.DEFAULT_FLOAT_DTYPE`
    attribute while the later will pass the :math:`a` variable as is.

    Examples
    --------
    >>> as_float(np.array([1]))
    1.0
    >>> as_float(np.arange(10))
    array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.])
    """

    if dtype is None:
        dtype = DEFAULT_FLOAT_DTYPE

    assert dtype in np.sctypes['float'], (
        '"dtype" must be one of the following types: {0}'.format(
            np.sctypes['float']))

    return dtype(a)


def set_float_precision(dtype=DEFAULT_FLOAT_DTYPE):
    """
    Sets *Colour* float precision by setting
    :attr:`colour.constant.DEFAULT_FLOAT_DTYPE` attribute with given type
    wherever the attribute is imported.

    Parameters
    ----------
    dtype : object
        Type to set :attr:`colour.constant.DEFAULT_FLOAT_DTYPE` with.

    Warnings
    --------
    Changing float precision might result in various *Colour* functionality
    breaking entirely: https://github.com/numpy/numpy/issues/6860. With great
    power comes great responsibility.

    Notes
    -----
    -   It is possible to define the float precision at import time by setting
        the *COLOUR_SCIENCE__FLOAT_PRECISION* environment variable, for example
        `set COLOUR_SCIENCE__FLOAT_PRECISION=float32`.
    -   Some definition returning a single-scalar ndarray might not honour the
        given float precision: https://github.com/numpy/numpy/issues/16353

    Examples
    --------
    >>> as_float_array(np.ones(3)).dtype
    dtype('float64')
    >>> set_float_precision(np.float16)
    >>> as_float_array(np.ones(3)).dtype
    dtype('float16')
    >>> set_float_precision(np.float64)
    >>> as_float_array(np.ones(3)).dtype
    dtype('float64')
    """

    with suppress_warnings(colour_usage_warnings=True):
        for name, module in sys.modules.items():
            if not name.startswith(name):
                continue

            if not hasattr(module, 'DEFAULT_FLOAT_DTYPE'):
                continue

            setattr(module, 'DEFAULT_FLOAT_DTYPE', dtype)


def set_int_precision(dtype=DEFAULT_INT_DTYPE):
    """
    Sets *Colour* integer precision by setting
    :attr:`colour.constant.DEFAULT_INT_DTYPE` attribute with given type
    wherever the attribute is imported.

    Parameters
    ----------
    dtype : object
        Type to set :attr:`colour.constant.DEFAULT_INT_DTYPE` with.

    Notes
    -----
    -   It is possible to define the int precision at import time by setting
        the *COLOUR_SCIENCE__INT_PRECISION* environment variable, for example
        `set COLOUR_SCIENCE__INT_PRECISION=int32`.

    Warnings
    --------
    This definition is mostly given for consistency purposes with
    :func:`colour.utilities.set_float_precision` definition but contrary to the
    latter, changing integer precision will almost certainly completely break
    *Colour*. With great power comes great responsibility.

    Examples
    --------
    >>> as_int_array(np.ones(3)).dtype  # doctest: +SKIP
    dtype('int64')
    >>> set_int_precision(np.int32)
    >>> as_int_array(np.ones(3)).dtype  # doctest: +SKIP
    dtype('int32')
    >>> set_int_precision(np.int64)
    >>> as_int_array(np.ones(3)).dtype  # doctest: +SKIP
    dtype('int64')
    """

    # TODO: Investigate behaviour on Windows.
    with suppress_warnings(colour_usage_warnings=True):
        for name, module in sys.modules.items():
            if not name.startswith(name):
                continue

            if not hasattr(module, 'DEFAULT_INT_DTYPE'):
                continue

            setattr(module, 'DEFAULT_INT_DTYPE', dtype)


def as_namedtuple(a, named_tuple):
    """
    Converts given :math:`a` variable to given *namedtuple* class instance.

    :math:`a` can be either a *Numpy* structured array, a *namedtuple*,
    a *mapping*, or an *array_like* object. The definition will attempt to
    convert it to given *namedtuple*.

    Parameters
    ----------
    a : object
        Variable to convert.
    named_tuple : namedtuple
        *namedtuple* class.

    Returns
    -------
    namedtuple
        math:`a` variable converted to *namedtuple*.

    Examples
    --------
    >>> from collections import namedtuple
    >>> a_a = 1
    >>> a_b = 2
    >>> a_c = 3
    >>> NamedTuple = namedtuple('NamedTuple', 'a b c')
    >>> as_namedtuple(NamedTuple(a=1, b=2, c=3), NamedTuple)
    NamedTuple(a=1, b=2, c=3)
    >>> as_namedtuple({'a': a_a, 'b': a_b, 'c': a_c}, NamedTuple)
    NamedTuple(a=1, b=2, c=3)
    >>> as_namedtuple([a_a, a_b, a_c], NamedTuple)
    NamedTuple(a=1, b=2, c=3)
    """

    if isinstance(a, np.ndarray):
        if a.dtype.fields is not None:
            a = {field: a[field] for field in a.dtype.fields}

    if isinstance(a, named_tuple):
        return a
    elif isinstance(a, Mapping):
        return named_tuple(**a)
    else:
        return named_tuple(*a)


def closest_indexes(a, b):
    """
    Returns the :math:`a` variable closest element indexes to reference
    :math:`b` variable elements.

    Parameters
    ----------
    a : array_like
        Variable to search for the closest element indexes.
    b : numeric
        Reference variable.

    Returns
    -------
    numeric
        Closest :math:`a` variable element indexes.

    Examples
    --------
    >>> a = np.array([24.31357115, 63.62396289, 55.71528816,
    ...               62.70988028, 46.84480573, 25.40026416])
    >>> print(closest_indexes(a, 63))
    [3]
    >>> print(closest_indexes(a, [63, 25]))
    [3 5]
    """

    a = np.ravel(a)[:, np.newaxis]
    b = np.ravel(b)[np.newaxis, :]

    return np.abs(a - b).argmin(axis=0)


def closest(a, b):
    """
    Returns the :math:`a` variable closest elements to reference :math:`b`
    variable elements.

    Parameters
    ----------
    a : array_like
        Variable to search for the closest elements.
    b : numeric
        Reference variable.

    Returns
    -------
    numeric
        Closest :math:`a` variable elements.

    Examples
    --------
    >>> a = np.array([24.31357115, 63.62396289, 55.71528816,
    ...               62.70988028, 46.84480573, 25.40026416])
    >>> closest(a, 63)
    array([ 62.70988028])
    >>> closest(a, [63, 25])
    array([ 62.70988028,  25.40026416])
    """

    a = np.array(a)

    return a[closest_indexes(a, b)]


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


def interval(distribution, unique=True):
    """
    Returns the interval size of given distribution.

    Parameters
    ----------
    distribution : array_like
        Distribution to retrieve the interval.
    unique : bool, optional
        Whether to return unique intervals if  the distribution is
        non-uniformly spaced or the complete intervals

    Returns
    -------
    ndarray
        Distribution interval.

    Examples
    --------
    Uniformly spaced variable:

    >>> y = np.array([1, 2, 3, 4, 5])
    >>> interval(y)
    array([ 1.])
    >>> interval(y, False)
    array([ 1.,  1.,  1.,  1.])

    Non-uniformly spaced variable:

    >>> y = np.array([1, 2, 3, 4, 8])
    >>> interval(y)
    array([ 1.,  4.])
    >>> interval(y, False)
    array([ 1.,  1.,  1.,  4.])
    """

    distribution = as_float_array(distribution)
    i = np.arange(distribution.size - 1)

    differences = np.abs(distribution[i + 1] - distribution[i])
    if unique:
        return np.unique(differences)
    else:
        return differences


def is_uniform(distribution):
    """
    Returns if given distribution is uniform.

    Parameters
    ----------
    distribution : array_like
        Distribution to check for uniformity.

    Returns
    -------
    bool
        Is distribution uniform.

    Examples
    --------
    Uniformly spaced variable:

    >>> a = np.array([1, 2, 3, 4, 5])
    >>> is_uniform(a)
    True

    Non-uniformly spaced variable:

    >>> a = np.array([1, 2, 3.1415, 4, 5])
    >>> is_uniform(a)
    False
    """

    return True if interval(distribution).size == 1 else False


def in_array(a, b, tolerance=EPSILON):
    """
    Tests whether each element of an array is also present in a second array
    within given tolerance.

    Parameters
    ----------
    a : array_like
        Array to test the elements from.
    b : array_like
        The values against which to test each value of array *a*.
    tolerance : numeric, optional
        Tolerance value.

    Returns
    -------
    ndarray
        A boolean array with *a* shape describing whether an element of *a* is
        present in *b* within given tolerance.

    References
    ----------
    :cite:`Yorke2014a`

    Examples
    --------
    >>> a = np.array([0.50, 0.60])
    >>> b = np.linspace(0, 10, 101)
    >>> np.in1d(a, b)
    array([ True, False], dtype=bool)
    >>> in_array(a, b)
    array([ True,  True], dtype=bool)
    """

    a = as_float_array(a)
    b = as_float_array(b)

    d = np.abs(np.ravel(a) - b[..., np.newaxis])

    return np.any(d <= tolerance, axis=0).reshape(a.shape)


def tstack(a, dtype=None):
    """
    Stacks arrays in sequence along the last axis (tail).

    Rebuilds arrays divided by :func:`colour.utilities.tsplit`.

    Parameters
    ----------
    a : array_like
        Array to perform the stacking.
    dtype : object
        Type to use for initial conversion to *ndarray*, default to the type
        defined by :attr:`colour.constant.DEFAULT_FLOAT_DTYPE` attribute.

    Returns
    -------
    ndarray

    Examples
    --------
    >>> a = 0
    >>> tstack([a, a, a])
    array([ 0.,  0.,  0.])
    >>> a = np.arange(0, 6)
    >>> tstack([a, a, a])
    array([[ 0.,  0.,  0.],
           [ 1.,  1.,  1.],
           [ 2.,  2.,  2.],
           [ 3.,  3.,  3.],
           [ 4.,  4.,  4.],
           [ 5.,  5.,  5.]])
    >>> a = np.reshape(a, (1, 6))
    >>> tstack([a, a, a])
    array([[[ 0.,  0.,  0.],
            [ 1.,  1.,  1.],
            [ 2.,  2.,  2.],
            [ 3.,  3.,  3.],
            [ 4.,  4.,  4.],
            [ 5.,  5.,  5.]]])
    >>> a = np.reshape(a, (1, 1, 6))
    >>> tstack([a, a, a])
    array([[[[ 0.,  0.,  0.],
             [ 1.,  1.,  1.],
             [ 2.,  2.,  2.],
             [ 3.,  3.,  3.],
             [ 4.,  4.,  4.],
             [ 5.,  5.,  5.]]]])
    """

    if dtype is None:
        dtype = DEFAULT_FLOAT_DTYPE

    a = as_array(a, dtype)

    return np.concatenate([x[..., np.newaxis] for x in a], axis=-1)


def tsplit(a, dtype=None):
    """
    Splits arrays in sequence along the last axis (tail).

    Parameters
    ----------
    a : array_like
        Array to perform the splitting.
    dtype : object
        Type to use for initial conversion to *ndarray*, default to the type
        defined by :attr:`colour.constant.DEFAULT_FLOAT_DTYPE` attribute.

    Returns
    -------
    ndarray

    Examples
    --------
    >>> a = np.array([0, 0, 0])
    >>> tsplit(a)
    array([ 0.,  0.,  0.])
    >>> a = np.array(
    ...     [[0, 0, 0],
    ...      [1, 1, 1],
    ...      [2, 2, 2],
    ...      [3, 3, 3],
    ...      [4, 4, 4],
    ...      [5, 5, 5]]
    ... )
    >>> tsplit(a)
    array([[ 0.,  1.,  2.,  3.,  4.,  5.],
           [ 0.,  1.,  2.,  3.,  4.,  5.],
           [ 0.,  1.,  2.,  3.,  4.,  5.]])
    >>> a = np.array(
    ...     [[[0, 0, 0],
    ...       [1, 1, 1],
    ...       [2, 2, 2],
    ...       [3, 3, 3],
    ...       [4, 4, 4],
    ...       [5, 5, 5]]]
    ... )
    >>> tsplit(a)
    array([[[ 0.,  1.,  2.,  3.,  4.,  5.]],
    <BLANKLINE>
           [[ 0.,  1.,  2.,  3.,  4.,  5.]],
    <BLANKLINE>
           [[ 0.,  1.,  2.,  3.,  4.,  5.]]])
    """

    if dtype is None:
        dtype = DEFAULT_FLOAT_DTYPE

    a = as_array(a, dtype)

    return np.array([a[..., x] for x in range(a.shape[-1])])


def row_as_diagonal(a):
    """
    Returns the per row diagonal matrices of the given array.

    Parameters
    ----------
    a : array_like
        Array to perform the diagonal matrices computation.

    Returns
    -------
    ndarray

    References
    ----------
    :cite:`Castro2014a`

    Examples
    --------
    >>> a = np.array(
    ...     [[0.25891593, 0.07299478, 0.36586996],
    ...       [0.30851087, 0.37131459, 0.16274825],
    ...       [0.71061831, 0.67718718, 0.09562581],
    ...       [0.71588836, 0.76772047, 0.15476079],
    ...       [0.92985142, 0.22263399, 0.88027331]]
    ... )
    >>> row_as_diagonal(a)
    array([[[ 0.25891593,  0.        ,  0.        ],
            [ 0.        ,  0.07299478,  0.        ],
            [ 0.        ,  0.        ,  0.36586996]],
    <BLANKLINE>
           [[ 0.30851087,  0.        ,  0.        ],
            [ 0.        ,  0.37131459,  0.        ],
            [ 0.        ,  0.        ,  0.16274825]],
    <BLANKLINE>
           [[ 0.71061831,  0.        ,  0.        ],
            [ 0.        ,  0.67718718,  0.        ],
            [ 0.        ,  0.        ,  0.09562581]],
    <BLANKLINE>
           [[ 0.71588836,  0.        ,  0.        ],
            [ 0.        ,  0.76772047,  0.        ],
            [ 0.        ,  0.        ,  0.15476079]],
    <BLANKLINE>
           [[ 0.92985142,  0.        ,  0.        ],
            [ 0.        ,  0.22263399,  0.        ],
            [ 0.        ,  0.        ,  0.88027331]]])
    """

    a = np.expand_dims(a, -2)

    return np.eye(a.shape[-1]) * a


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


def orient(a, orientation):
    """
    Orient given array according to given ``orientation`` value.

    Parameters
    ----------
    a : array_like
        Array to perform the orientation onto.
    orientation : unicode, optional
        **{'Flip', 'Flop', '90 CW', '90 CCW', '180'}**
        Orientation to perform.

    Returns
    -------
    ndarray
        Oriented array.

    Examples
    --------
    >>> a = np.tile(np.arange(5), (5, 1))
    >>> a
    array([[0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4]])
    >>> orient(a, '90 CW')
    array([[0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1],
           [2, 2, 2, 2, 2],
           [3, 3, 3, 3, 3],
           [4, 4, 4, 4, 4]])
    >>> orient(a, 'Flip')
    array([[4, 3, 2, 1, 0],
           [4, 3, 2, 1, 0],
           [4, 3, 2, 1, 0],
           [4, 3, 2, 1, 0],
           [4, 3, 2, 1, 0]])
    """

    if orientation.lower() == 'flip':
        return np.fliplr(a)
    elif orientation.lower() == 'flop':
        return np.flipud(a)
    elif orientation.lower() == '90 cw':
        return np.rot90(a, 3)
    elif orientation.lower() == '90 ccw':
        return np.rot90(a)
    elif orientation.lower() == '180':
        return np.rot90(a, 2)
    else:
        return a


def centroid(a):
    """
    Computes the centroid indexes of given :math:`a` array.

    Parameters
    ----------
    a : array_like
        :math:`a` array to compute the centroid indexes.

    Returns
    -------
    ndarray
        :math:`a` array centroid indexes.

    Examples
    --------
    >>> a = np.tile(np.arange(0, 5), (5, 1))
    >>> centroid(a)  # doctest: +ELLIPSIS
    array([2, 3]...)
    """

    a = as_float_array(a)

    a_s = np.sum(a)

    ranges = [np.arange(0, a.shape[i]) for i in range(a.ndim)]
    coordinates = np.meshgrid(*ranges)

    a_ci = []
    for axis in coordinates:
        axis = np.transpose(axis)
        # Aligning axis for N-D arrays where N is normalised to
        # range [3, :math:`\\\infty`]
        for i in range(axis.ndim - 2, 0, -1):
            axis = np.rollaxis(axis, i - 1, axis.ndim)

        a_ci.append(np.sum(axis * a) // a_s)

    return np.array(a_ci).astype(DEFAULT_INT_DTYPE)


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


def lerp(a, b, c):
    """
    Performs a simple linear interpolation between given array :math:`a` and
    array :math:`b` using :math:`c` value.

    Parameters
    ----------
    a : array_like
        Array :math:`a`, the start of the range in which to interpolate.
    b : array_like
        Array :math:`b`, the end of the range in which to interpolate.
    c : array_like
        Array :math:`c` value to use to interpolate between array :math:`a` and
        array :math:`b`.

    Returns
    -------
    ndarray
        Linear interpolation result.
    Examples
    --------
    >>> a = 0
    >>> b = 2
    >>> lerp(a, b, 0.5)
    1.0
    """

    a = as_float_array(a)
    b = as_float_array(b)
    c = as_float_array(c)

    return (1 - c) * a + c * b


def fill_nan(a, method='Interpolation', default=0):
    """
    Fills given array NaNs according to given method.

    Parameters
    ----------
    a : array_like
        Array to fill the NaNs of.
    method : unicode
        **{'Interpolation', 'Constant'}**,
        *Interpolation* method linearly interpolates through the NaNs,
        *Constant* method replaces NaNs with ``default``.
    default : numeric
        Value to use with the *Constant* method.

    Returns
    -------
    ndarray
        NaNs filled array.

    Examples
    --------
    >>> a = np.array([0.1, 0.2, np.nan, 0.4, 0.5])
    >>> fill_nan(a)
    array([ 0.1,  0.2,  0.3,  0.4,  0.5])
    >>> fill_nan(a, method='Constant')
    array([ 0.1,  0.2,  0. ,  0.4,  0.5])
    """

    a = np.copy(a)

    mask = np.isnan(a)

    if method.lower() == 'interpolation':
        a[mask] = np.interp(
            np.flatnonzero(mask), np.flatnonzero(~mask), a[~mask])
    elif method.lower() == 'constant':
        a[mask] = default

    return a


@contextmanager
def ndarray_write(a):
    """
    A context manager setting given array writeable to perform an operation
    and then read-only.

    Parameters
    ----------
    a : array_like
        Array to perform an operation.

    Returns
    -------
    ndarray
        Array.

    Examples
    --------
    >>> a = np.linspace(0, 1, 10)
    >>> a.setflags(write=False)
    >>> try:
    ...     a += 1
    ... except ValueError:
    ...     pass
    >>> with ndarray_write(a):
    ...     a +=1
    """

    a = as_float_array(a)

    a.setflags(write=True)

    try:
        yield a
    finally:
        a.setflags(write=False)


def zeros(shape, dtype=None, order='C'):
    """
    Simple wrapper around :func:`np.zeros` definition to create arrays with
    the active type defined by the:attr:`colour.constant.DEFAULT_FLOAT_DTYPE`
    attribute.

    Parameters
    ----------
    shape : int or array_like
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    dtype : object
        Type to use for conversion, default to the type defined by the
        :attr:`colour.constant.DEFAULT_FLOAT_DTYPE` attribute.
    order : unicode, optional
        {'C', 'F'},
        Whether to store multi-dimensional data in row-major
        (C-style) or column-major (Fortran-style) order in
        memory.

    Returns
    -------
    ndarray
        Array of given shape and type, filled with zeros.

    Examples
    --------
    >>> zeros(3)
    array([ 0.,  0.,  0.])
    """

    if dtype is None:
        dtype = DEFAULT_FLOAT_DTYPE

    return np.zeros(shape, dtype, order)


def ones(shape, dtype=None, order='C'):
    """
    Simple wrapper around :func:`np.ones` definition to create arrays with
    the active type defined by the:attr:`colour.constant.DEFAULT_FLOAT_DTYPE`
    attribute.

    Parameters
    ----------
    shape : int or array_like
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    dtype : object
        Type to use for conversion, default to the type defined by the
        :attr:`colour.constant.DEFAULT_FLOAT_DTYPE` attribute.
    order : unicode, optional
        {'C', 'F'},
        Whether to store multi-dimensional data in row-major
        (C-style) or column-major (Fortran-style) order in
        memory.

    Returns
    -------
    ndarray
        Array of given shape and type, filled with ones.

    Examples
    --------
    >>> ones(3)
    array([ 1.,  1.,  1.])
    """

    if dtype is None:
        dtype = DEFAULT_FLOAT_DTYPE

    return np.ones(shape, dtype, order)


def full(shape, fill_value, dtype=None, order='C'):
    """
    Simple wrapper around :func:`np.full` definition to create arrays with
    the active type defined by the:attr:`colour.constant.DEFAULT_FLOAT_DTYPE`
    attribute.

    Parameters
    ----------
    shape : int or array_like
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    fill_value : numeric
        Fill value.
    dtype : object
        Type to use for conversion, default to the type defined by the
        :attr:`colour.constant.DEFAULT_FLOAT_DTYPE` attribute.
    order : unicode, optional
        {'C', 'F'},
        Whether to store multi-dimensional data in row-major
        (C-style) or column-major (Fortran-style) order in
        memory.

    Returns
    -------
    ndarray
        Array of given shape and type, filled with given value.

    Examples
    --------
    >>> ones(3)
    array([ 1.,  1.,  1.])
    """

    if dtype is None:
        dtype = DEFAULT_FLOAT_DTYPE

    return np.full(shape, fill_value, dtype, order)


def index_along_last_axis(a, indexes):
    """
    Reduces the dimension of an array by one, by using an array of indexes to
    to pick elements off the last axis.

    Parameters
    ----------
    a : ndarray, (Ni..., m)
        Array to be indexed.
    indexes : ndarray, (Ni...)
        Integer array with the same shape as `a` but with one dimension fewer,
        containing indixes to the last dimension of `a`. All elements must be
        numbers between `0` and `m` - 1.

    Returns
    -------
    ndarray, (Ni...)
        Result of the operation.

    Raises
    ------
    ValueError
        If the arrays have incompatible shapes.
    IndexError
        If `indexes` has elements outside of the allowed range of 0 to `m` - 1
        or if it's not an integer array.

    Examples
    --------
    >>> a = np.array(
    ...     [[[0.3, 0.5, 6.9],
    ...       [3.3, 4.4, 1.6],
    ...       [4.4, 7.5, 2.3],
    ...       [2.3, 1.6, 7.4]],
    ...      [[2. , 5.9, 2.8],
    ...       [6.2, 4.9, 8.6],
    ...       [3.7, 9.7, 7.3],
    ...       [6.3, 4.3, 3.2]],
    ...      [[0.8, 1.9, 0.7],
    ...       [5.6, 4. , 1.7],
    ...       [6.7, 8.2, 1.7],
    ...       [1.2, 7.1, 1.4]],
    ...      [[4. , 4.8, 8.9],
    ...       [4. , 0.3, 6.9],
    ...       [3.5, 7.1, 4.5],
    ...       [1.4, 1.9, 1.6]]]
    ... )
    >>> indexes = np.array(
    ...     [[2, 0, 1, 1],
    ...      [2, 1, 1, 0],
    ...      [0, 0, 1, 2],
    ...      [0, 0, 1, 2]]
    ... )
    >>> index_along_last_axis(a, indexes)
    array([[ 6.9,  3.3,  7.5,  1.6],
           [ 2.8,  4.9,  9.7,  6.3],
           [ 0.8,  5.6,  8.2,  1.4],
           [ 4. ,  4. ,  7.1,  1.6]])

    This function can be used to compute the result of :func:`np.min` along
    the last axis given the corresponding :func:`np.argmin` indexes.

    >>> indexes = np.argmin(a, axis=-1)
    >>> np.array_equal(
    ...     index_along_last_axis(a, indexes),
    ...     np.min(a, axis=-1)
    ... )
    True

    In particular, this can be used to manipulate the indices given by
    functions like :func:`np.min` before indexing the array. For example, to
    get elements directly following the smallest elements:

    >>> index_along_last_axis(a, (indexes + 1) % 3)
    array([[ 0.5,  3.3,  4.4,  7.4],
           [ 5.9,  8.6,  9.7,  6.3],
           [ 0.8,  5.6,  6.7,  7.1],
           [ 4.8,  6.9,  7.1,  1.9]])
    """

    if a.shape[:-1] != indexes.shape:
        raise ValueError('Arrays have incompatible shapes: {0} and {1}'.format(
            a.shape, indexes.shape))

    return np.take_along_axis(
        a, np.expand_dims(indexes, axis=-1), axis=-1).squeeze(axis=-1)
