# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**common.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Color** package algebra common utilities objects that don't fall in any specific category.

**Others:**

"""

from __future__ import unicode_literals

import numpy

import foundations.verbose

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["get_steps",
           "is_uniform",
           "get_closest"]

LOGGER = foundations.verbose.install_logger()


def get_steps(distribution):
    """
    Returns the steps of given distribution.

    :param distribution: Distribution to retrieve the steps.
    :type distribution: tuple or list or Array or Matrix
    :return: steps.
    :rtype: tuple
    """

    return tuple(set([distribution[i + 1] - distribution[i] for i in range(len(distribution) - 1)]))


def is_uniform(distribution):
    """
    Returns if given distribution is uniform.

    :param distribution: Distribution to check for uniformity.
    :type distribution: tuple or list or Array or Matrix
    :return: Is uniform.
    :rtype: bool
    """

    return True if len(get_steps(distribution)) == 1 else False


def get_closest(y, x):
    """
    Returns closest *y* variable element to reference *x* variable.

    Usage::

        >>> y = numpy.array([24.31357115, 63.62396289, 55.71528816, 62.70988028, 46.84480573, 25.40026416])
        >>> get_closest(63, y)
        62.70988028

    :param y: Variable to search for the closest element.
    :type y: matrix or ndarray
    :param x: Reference variable.
    :type x: int or float
    :return: Closest *y* variable element.
    :rtype: int or float
    """

    return y[(numpy.abs(numpy.array(y) - x)).argmin()]