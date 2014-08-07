# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Matrix Helpers
==============

Defines matrices computation helpers objects.
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
    Returns if given *array_like* variable :math:`x` is an identity matrix.

    Parameters
    ----------
    x : array_like (N)
        Variable :math:`x` to test.
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

    return np.array_equal(np.identity(n), x)