#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Algebra Common Utilities
========================

Defines algebra common utilities objects that don"t belong to any algebra
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
           'get_steps',
           'get_closest',
           'to_ndarray',
           'is_uniform',
           'is_iterable',
           'is_number',
           'is_integer',
           'normalise']

FLOATING_POINT_NUMBER_PATTERN = '[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?'

INTEGER_THRESHOLD = 0.001
"""
Integer threshold value.

INTEGER_THRESHOLD : float
"""


def get_steps(distribution):
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
    >>> y = np.array([1, 2, 3, 4, 5])
    >>> colour.get_steps(y)
    (1,)
    """

    return tuple(set([distribution[i + 1] - distribution[i]
                      for i in range(len(distribution) - 1)]))


def get_closest(y, x):
    """
    Returns closest :math:`y` variable element to reference :math:`x` variable.

    Parameters
    ----------
    y : array_like
        Variable to search for the closest element.
    x : int or float
        Reference variable.

    Returns
    -------
    int or float
        Closest :math:`y` variable element.

    Examples
    --------
    >>> y = np.array([24.31357115, 63.62396289, 55.71528816, 62.70988028, 46.84480573, 25.40026416])
    >>> get_closest(63, y)
    62.70988028
    """

    return y[(np.abs(np.array(y) - x)).argmin()]


def to_ndarray(x, data_type=np.float_):
    """
    Converts given :math:`x` variable to *ndarray*.

    Parameters
    ----------
    x : object
        Variable to convert.
    data_type : dtype
        *ndarray* data type.
    Returns
    -------
    ndarray
        :math:`x` variable converted to *ndarray*.

    Examples
    --------
    >>> to_ndarray(1)
    [1]
    """

    return (np.array(x, dtype=data_type)
            if is_iterable(x) else
            np.array((x,), dtype=data_type))


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
    >>> y = np.array([1, 2, 3, 4, 5])
    >>> colour.is_uniform(y)
    True

    >>> y = np.array([1, 2, 3.1415, 4, 5])
    >>> colour.is_uniform(y)
    False
    """

    return True if len(get_steps(distribution)) == 1 else False


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


def is_number(x):
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
    >>> is_number(1)
    True
    >>> is_number((1,))
    False
    """

    return isinstance(x, (int, float, complex))


def is_integer(x):
    """
    Returns if given :math:`x` variable is an integer through thresholding.

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
    The determination threshold is defined by the
    :attr:`colour.algebra.common.INTEGER_THRESHOLD` attribute.

    See Also
    --------
    is_number

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
    factor : float, optional
        Normalization factor
    clip : bool, optional
        Clip values between in domain [0, 'factor'].

    Returns
    -------
    ndarray
        Normalised :math:`x` variable.

    Examples
    --------
    >>> colour.normalise(np.array(0.48224885, 0.31651974, 0.22070513]))
    array([ 1.        ,  0.6563411 ,  0.45765818])
    """

    x = to_ndarray(x)
    maximum = np.max(x)
    x *= (1 / maximum) * factor
    return np.clip(x, 0, factor) if clip else x