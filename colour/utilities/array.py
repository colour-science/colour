#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Array Utilities
===============

Defines array utilities objects.
"""

from __future__ import division, unicode_literals

import numpy as np

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
           'tstack',
           'tsplit',
           'row_as_diagonal']


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
    except TypeError as error:
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
    >>> y = np.array([24.31357115, 63.62396289, 55.71528816, 62.70988028, 46.84480573, 25.40026416])  # noqa
    >>> closest(y, 63)
    62.70988028
    """

    return y[(np.abs(np.array(y) - x)).argmin()]


def normalise(x, factor=1, clip=True):
    """
    Normalises given *array_like* :math:`x` variable values and optionally clip
    them between.

    Parameters
    ----------
    x : array_like
        :math:`x` variable to normalise.
    factor : numeric, optional
        Normalization factor
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

    maximum = np.max(x)
    x *= (1 / maximum) * factor
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
    tuple
        Distribution steps.

    Examples
    --------
    Uniformly spaced variable:

    >>> y = np.array([1, 2, 3, 4, 5])
    >>> steps(y)
    (1,)

    Non-uniformly spaced variable:

    >>> y = np.array([1, 2, 3, 4, 8])
    >>> steps(y)
    (1, 4)
    """

    return tuple(set([distribution[i + 1] - distribution[i]
                      for i in range(len(distribution) - 1)]))


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
            http://stackoverflow.com/questions/26511401/numpy-fastest-way-of-computing-diagonal-for-each-row-of-a-2d-array/26517247#26517247  # noqa

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

