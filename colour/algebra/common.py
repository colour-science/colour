#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Algebra Common Utilities
========================

Defines algebra common utilities objects that don't belong to any algebra
specific category.
"""

from __future__ import division, unicode_literals

import numpy as np

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['FLOATING_POINT_NUMBER_PATTERN',
           'INTEGER_THRESHOLD',
           'steps',
           'closest',
           'as_array',
           'is_uniform',
           'is_iterable',
           'is_numeric',
           'is_integer',
           'normalise']

FLOATING_POINT_NUMBER_PATTERN = '[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?'

INTEGER_THRESHOLD = 0.001
"""
Integer threshold value.

INTEGER_THRESHOLD : numeric
"""


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


def as_array(x, shape=None, data_type=np.float_):
    """
    Converts given :math:`x` variable to *ndarray*.

    Parameters
    ----------
    x : object
        Variable to convert.
    shape : tuple, optional
        *ndarray* shape.
    data_type : dtype, optional
        *ndarray* data type.

    Returns
    -------
    ndarray
        :math:`x` variable converted to *ndarray*.

    Examples
    --------
    >>> as_array(1)
    array([ 1.])
    """

    array = (np.asarray(x, dtype=data_type)
             if is_iterable(x) else
             np.asarray((x,), dtype=data_type))

    if shape is not None:
        array = array.reshape(shape)

    return array


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


def is_iterable(x):
    """
    Returns if given :math:`x` variable is iterable.

    Parameters
    ----------
    x : object
        Variable to check the iterability.

    Returns
    -------
    bool
        :math:`x` variable iterability.

    Examples
    --------
    >>> is_iterable([1, 2, 3])
    True
    >>> is_iterable(1)
    False
    """

    try:
        for _ in x:
            break
        return True
    except TypeError:
        return False


def is_numeric(x):
    """
    Returns if given :math:`x` variable is a number.

    Parameters
    ----------
    x : object
        Variable to check.

    Returns
    -------
    bool
        Is :math:`x` variable a number.

    See Also
    --------
    is_integer

    Examples
    --------
    >>> is_numeric(1)
    True
    >>> is_numeric((1,))
    False
    """

    return isinstance(x, (int, float, complex,
                          np.integer, np.floating, np.complex))


def is_integer(x):
    """
    Returns if given :math:`x` variable is an integer under given threshold.

    Parameters
    ----------
    x : object
        Variable to check.

    Returns
    -------
    bool
        Is :math:`x` variable an integer.

    Notes
    -----
    -   The determination threshold is defined by the
        :attr:`colour.algebra.common.INTEGER_THRESHOLD` attribute.

    See Also
    --------
    is_numeric

    Examples
    --------
    >>> is_integer(1)
    True
    >>> is_integer(1.01)
    False
    """

    return abs(x - round(x)) <= INTEGER_THRESHOLD


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

    x = as_array(x)
    maximum = np.max(x)
    x *= (1 / maximum) * factor
    return np.clip(x, 0, factor) if clip else x
