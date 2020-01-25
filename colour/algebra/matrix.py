# -*- coding: utf-8 -*-
"""
Matrix Helpers
==============

Defines matrices computation helpers objects.
"""

from __future__ import division, unicode_literals

import numpy as np

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['is_identity']


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
