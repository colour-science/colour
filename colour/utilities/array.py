#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Array Utilities
===============

Defines array utilities objects.
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.constants import EPSILON

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['as_numeric',
           'closest',
           'normalise',
           'steps',
           'is_uniform',
           'in_array',
           'tstack',
           'tsplit',
           'row_as_diagonal',
           'dot_vector',
           'dot_matrix']


def as_numeric(x):
    """
    Converts given :math:`x` variable to *numeric*. In the event where
    :math:`x` cannot be converted, it is passed as is.

    Parameters
    ----------
    x : object
        Variable to convert.

    Returns
    -------
    ndarray
        :math:`x` variable converted to *numeric*.

    See Also
    --------
    as_stack, as_shape, auto_axis

    Examples
    --------
    >>> as_numeric(np.array([1]))
    1.0
    >>> as_numeric(np.arange(10))
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    """

    try:
        return float(x)
    except TypeError:
        return x


def closest(y, x):
    """
    Returns closest :math:`y` variable element to reference :math:`x` variable.

    Parameters
    ----------
    y : array_like
        Variable to search for the closest element.
    x : numeric
        Reference variable.

    Returns
    -------
    numeric
        Closest :math:`y` variable element.

    Examples
    --------
    >>> y = np.array([24.31357115,
    ...               63.62396289,
    ...               55.71528816,
    ...               62.70988028,
    ...               46.84480573,
    ...               25.40026416])
    >>> closest(y, 63)
    62.70988028
    """

    return y[(np.abs(np.array(y) - x)).argmin()]


def normalise(x, axis=None, factor=1, clip=True):
    """
    Normalises given *array_like* :math:`x` variable values and optionally clip
    them between.

    Parameters
    ----------
    x : array_like
        :math:`x` variable to normalise.
    axis : numeric, optional
        Normalization axis.
    factor : numeric, optional
        Normalization factor.
    clip : bool, optional
        Clip values between in domain [0, 'factor'].

    Returns
    -------
    ndarray
        Normalised :math:`x` variable.

    Examples
    --------
    >>> x = np.array([0.48224885, 0.31651974, 0.22070513])
    >>> normalise(x)  # doctest: +ELLIPSIS
    array([ 1.        ,  0.6563411...,  0.4576581...])
    """

    x = np.asarray(x)

    maximum = np.max(x, axis=axis)
    x *= (1 / maximum[..., np.newaxis]) * factor

    return np.clip(x, 0, factor) if clip else x


def steps(distribution):
    """
    Returns the steps of given distribution.

    Parameters
    ----------
    distribution : array_like
        Distribution to retrieve the steps.

    Returns
    -------
    ndarray
        Distribution steps.

    Examples
    --------
    Uniformly spaced variable:

    >>> y = np.array([1, 2, 3, 4, 5])
    >>> steps(y)
    array([1])

    Non-uniformly spaced variable:

    >>> y = np.array([1, 2, 3, 4, 8])
    >>> steps(y)
    array([1, 4])
    """

    distribution = sorted(distribution)

    return np.unique([distribution[i + 1] - distribution[i]
                      for i in range(len(distribution) - 1)])


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

    >>> y = np.array([1, 2, 3, 4, 5])
    >>> is_uniform(y)
    True

    Non-uniformly spaced variable:

    >>> y = np.array([1, 2, 3.1415, 4, 5])
    >>> is_uniform(y)
    False
    """

    return True if len(steps(distribution)) == 1 else False


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
    .. [1]  Yorke, R. (2014). Python: Change format of np.array or allow
            tolerance in in1d function. Retrieved March 27, 2015, from
            http://stackoverflow.com/a/23521245/931625

    Examples
    --------
    >>> a = np.array([0.50, 0.60])
    >>> b = np.linspace(0, 10, 101)
    >>> np.in1d(a, b)
    array([ True, False], dtype=bool)
    >>> in_array(a, b)
    array([ True,  True], dtype=bool)
    """

    a = np.asarray(a)
    b = np.asarray(b)

    d = np.abs(np.ravel(a) - b[..., np.newaxis])

    return np.any(d <= tolerance, axis=0).reshape(a.shape)


def tstack(a):
    """
    Stacks arrays in sequence along the last axis (tail).

    Rebuilds arrays divided by :func:`tsplit`.

    Parameters
    ----------
    a : array_like
        Array to perform the stacking.

    Returns
    -------
    ndarray

    See Also
    --------
    tsplit

    Examples
    --------
    >>> a = 0
    >>> tstack((a, a, a))
    array([0, 0, 0])
    >>> a = np.arange(0, 6)
    >>> tstack((a, a, a))
    array([[0, 0, 0],
           [1, 1, 1],
           [2, 2, 2],
           [3, 3, 3],
           [4, 4, 4],
           [5, 5, 5]])
    >>> a = np.reshape(a, (1, 6))
    >>> tstack((a, a, a))
    array([[[0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [4, 4, 4],
            [5, 5, 5]]])
    >>> a = np.reshape(a, (1, 1, 6))
    >>> tstack((a, a, a))
    array([[[[0, 0, 0],
             [1, 1, 1],
             [2, 2, 2],
             [3, 3, 3],
             [4, 4, 4],
             [5, 5, 5]]]])
    """

    a = np.asarray(a)

    return np.concatenate([x[..., np.newaxis] for x in a], axis=-1)


def tsplit(a):
    """
    Splits arrays in sequence along the last axis (tail).

    Parameters
    ----------
    a : array_like
        Array to perform the splitting.

    Returns
    -------
    ndarray

    See Also
    --------
    tstack

    Examples
    --------
    >>> a = np.array([0, 0, 0])
    >>> tsplit(a)
    array([0, 0, 0])
    >>> a = np.array([[0, 0, 0],
    ...               [1, 1, 1],
    ...               [2, 2, 2],
    ...               [3, 3, 3],
    ...               [4, 4, 4],
    ...               [5, 5, 5]])
    >>> tsplit(a)
    array([[0, 1, 2, 3, 4, 5],
           [0, 1, 2, 3, 4, 5],
           [0, 1, 2, 3, 4, 5]])
    >>> a = np.array([[[0, 0, 0],
    ...                [1, 1, 1],
    ...                [2, 2, 2],
    ...                [3, 3, 3],
    ...                [4, 4, 4],
    ...                [5, 5, 5]]])
    >>> tsplit(a)
    array([[[0, 1, 2, 3, 4, 5]],
    <BLANKLINE>
           [[0, 1, 2, 3, 4, 5]],
    <BLANKLINE>
           [[0, 1, 2, 3, 4, 5]]])
    """

    a = np.asarray(a)

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
    .. [1]  Castro, S. (2014). Numpy: Fastest way of computing diagonal for
            each row of a 2d array. Retrieved August 22, 2014, from
            http://stackoverflow.com/questions/26511401/\
numpy-fastest-way-of-computing-diagonal-for-each-row-of-a-2d-array/\
26517247#26517247

    Examples
    --------
    >>> a = np.array([[0.25891593, 0.07299478, 0.36586996],
    ...               [0.30851087, 0.37131459, 0.16274825],
    ...               [0.71061831, 0.67718718, 0.09562581],
    ...               [0.71588836, 0.76772047, 0.15476079],
    ...               [0.92985142, 0.22263399, 0.88027331]])
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


def dot_vector(m, v):
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

    See Also
    --------
    dot_matrix

    Examples
    --------
    >>> m = np.array([[0.7328, 0.4296, -0.1624],
    ...               [-0.7036, 1.6975, 0.0061],
    ...               [0.0030, 0.0136, 0.9834]])
    >>> m = np.reshape(np.tile(m, (6, 1)), (6, 3, 3))
    >>> v = np.array([0.07049534, 0.10080000, 0.09558313])
    >>> v = np.tile(v, (6, 1))
    >>> dot_vector(m, v)  # doctest: +ELLIPSIS
    array([[ 0.0794399...,  0.1220905...,  0.0955788...],
           [ 0.0794399...,  0.1220905...,  0.0955788...],
           [ 0.0794399...,  0.1220905...,  0.0955788...],
           [ 0.0794399...,  0.1220905...,  0.0955788...],
           [ 0.0794399...,  0.1220905...,  0.0955788...],
           [ 0.0794399...,  0.1220905...,  0.0955788...]])
    """

    return np.einsum('...ij,...j->...i', m, v)


def dot_matrix(a, b):
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

    Returns
    -------
    ndarray

    See Also
    --------
    dot_matrix

    Examples
    --------
    >>> a = np.array([[0.7328, 0.4296, -0.1624],
    ...               [-0.7036, 1.6975, 0.0061],
    ...               [0.0030, 0.0136, 0.9834]])
    >>> a = np.reshape(np.tile(a, (6, 1)), (6, 3, 3))
    >>> b = a
    >>> dot_matrix(a, b)  # doctest: +ELLIPSIS
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

    return np.einsum('...ij,...jk->...ik', a, b)
