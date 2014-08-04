# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**matrix.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package matrix helper objects.

**Others:**

"""

from __future__ import unicode_literals

import numpy as np

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["is_identity"]


def is_identity(x, n=3):
    """
    Returns if given *array_like* variable *x* is an identity matrix.

    Usage::

        >>> is_identity(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).reshape(3, 3))
        True
        >>> is_identity(np.array([1, 2, 0, 0, 1, 0, 0, 0, 1]).reshape(3, 3))
        False

    :param x: *array_like* variable *x*.
    :type x: array_like (N)
    :param n: Matrix dimension.
    :type n: int
    :return: Is identity matrix.
    :rtype: bool
    """

    return np.array_equal(np.identity(n), x)