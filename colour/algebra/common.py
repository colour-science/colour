# -*- coding: utf-8 -*-
"""
Common Utilities
================

Defines common algebra utilities objects that don't fall in any specific
category:

-   :func:`colour.algebra.spow`: Safe (symmetrical) power.
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.utilities import as_numeric

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['spow']


def spow(a, p):
    """
    Raises given array :math:`a` to the power :math:`p` as follows:
    :math:`sign(a) * |a|^p`.

    This avoids NaNs generation when array :math:`a` is negative and the power
    :math:`p` is fractional.

    Parameters
    ----------------
    a : numeric or array_like
        Array :math:`a`.
    p : numeric or array_like
        Power :math:`p`.

    Returns
    -------
    numeric or ndarray
        Array :math:`a` safely raised to the power :math:`p`.

    Examples
    --------
    >>> np.power(-2, 0.15)
    nan
    >>> spow(-2, 0.15)  # doctest: +ELLIPSIS
    -1.1095694...
    >>> spow(0, 0)
    0.0
    """

    a = np.atleast_1d(a)
    p = np.asarray(p)

    a_p = np.sign(a) * np.abs(a) ** p

    a_p[np.isnan(a_p)] = 0

    return as_numeric(a_p)
