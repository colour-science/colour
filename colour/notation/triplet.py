#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hexadecimal Triplet Notation
============================

Defines objects for hexadecimal triplet notation:

-   :func:`RGB_to_HEX`
-   :func:`HEX_to_RGB`
"""

from __future__ import division, unicode_literals

import numpy as np

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['RGB_to_HEX',
           'HEX_to_RGB']


def RGB_to_HEX(RGB):
    """
    Converts from *RGB* colourspace to hexadecimal triplet representation.

    Parameters
    ----------
    RGB : array_like, (3,)
        *RGB* colourspace matrix.

    Returns
    -------
    unicode
        Hexadecimal triplet representation.

    Notes
    -----
    -   Input *RGB* colourspace matrix is in domain [0, 1].

    Examples
    --------
    >>> RGB = np.array([0.66666667, 0.86666667, 1])
    >>> # Doctests skip for Python 2.x compatibility.
    >>> RGB_to_HEX(RGB)  # doctest: +SKIP
    '#aaddff'
    """

    RGB = np.ravel(RGB)
    R, G, B = map(int, RGB * 255)
    return '#{0:02x}{1:02x}{2:02x}'.format(R, G, B)


def HEX_to_RGB(HEX):
    """
    Converts from hexadecimal triplet representation to *RGB* colourspace.

    Parameters
    ----------
    HEX : unicode
        Hexadecimal triplet representation.

    Returns
    -------
    ndarray, (3,)
        *RGB* colourspace matrix.

    Notes
    -----
    -   Output *RGB* colourspace matrix is in domain [0, 1].

    Examples
    --------
    >>> HEX = '#aaddff'
    >>> HEX_to_RGB(HEX)  # doctest: +ELLIPSIS
    array([ 0.6666666...,  0.8666666...,  1.        ])
    """

    HEX = HEX.lstrip('#')
    length = len(HEX)
    return np.array([int(HEX[i:i + length // 3], 16) for i in
                     range(0, length, length // 3)]) / 255
