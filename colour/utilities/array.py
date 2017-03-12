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
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['as_numeric',
           'closest',
           'normalise_maximum',
           'interval',
           'is_uniform',
           'in_array',
           'tstack',
           'tsplit',
           'row_as_diagonal',
           'dot_vector',
           'dot_matrix',
           'orient',
           'centroid',
           'linear_conversion']


def as_numeric(a, type_=np.float_):
    """
    Converts given :math:`a` variable to *numeric*. In the event where
    :math:`a` cannot be converted, it is passed as is.

    Parameters
    ----------
    a : object
        Variable to convert.
    type_ : object
        Type to use for conversion.

    Returns
    -------
    ndarray
        :math:`a` variable converted to *numeric*.

    See Also
    --------
    as_stack, as_shape, auto_axis

    Examples
    --------
    >>> as_numeric(np.array([1]))
    1.0
    >>> as_numeric(np.arange(10))
    array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.])
    """

    try:
        return type_(a)
    except TypeError:
        return a


def closest(a, b):
    """
    Returns closest :math:`a` variable element to reference :math:`b` variable.

    Parameters
    ----------
    a : array_like
        Variable to search for the closest element.
    b : numeric
        Reference variable.

    Returns
    -------
    numeric
        Closest :math:`a` variable element.

    Examples
    --------
    >>> a = np.array([24.31357115,
    ...               63.62396289,
    ...               55.71528816,
    ...               62.70988028,
    ...               46.84480573,
    ...               25.40026416])
    >>> closest(a, 63)
    62.70988028
    """

    return a[(np.abs(np.array(a) - b)).argmin()]


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
        Clip values between in domain [0, 'factor'].

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

    a = np.asarray(a)

    maximum = np.max(a, axis=axis)
    a *= (1 / maximum[..., np.newaxis]) * factor

    return np.clip(a, 0, factor) if clip else a


def interval(distribution):
    """
    Returns the interval size of given distribution.

    Parameters
    ----------
    distribution : array_like
        Distribution to retrieve the interval.

    Returns
    -------
    ndarray
        Distribution interval.

    Examples
    --------
    Uniformly spaced variable:

    >>> y = np.array([1, 2, 3, 4, 5])
    >>> interval(y)
    array([1])

    Non-uniformly spaced variable:

    >>> y = np.array([1, 2, 3, 4, 8])
    >>> interval(y)
    array([1, 4])
    """

    distribution = np.sort(distribution)
    i = np.arange(distribution.size - 1)

    return np.unique(distribution[i + 1] - distribution[i])


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


def orient(a, orientation):
    """
    Orient given array accordingly to given `orientation` value.

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
    >>> centroid(a)
    array([2, 3])
    """

    a = np.asarray(a)

    a_s = np.sum(a)

    ranges = [np.arange(0, a.shape[i]) for i in range(a.ndim)]
    coordinates = np.meshgrid(*ranges)

    a_ci = []
    for axis in coordinates:
        axis = np.transpose(axis)
        # Aligning axis for N-D arrays where N is in range [3, :math:`\infty`]
        for i in range(axis.ndim - 2, 0, -1):
            axis = np.rollaxis(axis, i - 1, axis.ndim)

        a_ci.append(np.sum(axis * a) // a_s)

    return np.array(a_ci).astype(np.int_)


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

    Examples
    --------
    >>> a = np.linspace(0, 1, 10)
    >>> linear_conversion(a, np.array([0, 1]), np.array([1, 10]))
    array([  1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.])
    """

    a = np.asarray(a)

    in_min, in_max = tsplit(old_range)
    out_min, out_max = tsplit(new_range)

    return (((a - in_min) / (in_max - in_min)) *
            (out_max - out_min) + out_min)
