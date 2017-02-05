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
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
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
    RGB : array_like
        *RGB* colourspace array.

    Returns
    -------
    unicode
        Hexadecimal triplet representation.

    Notes
    -----
    -   Input *RGB* colourspace array is in domain [0, 1].

    Examples
    --------
    >>> RGB = np.array([0.66666667, 0.86666667, 1.00000000])
    >>> # Doctests skip for Python 2.x compatibility.
    >>> RGB_to_HEX(RGB)  # doctest: +SKIP
    '#aaddff'
    """

    RGB = np.asarray(RGB)

    to_HEX = np.vectorize('{0:02x}'.format)

    HEX = to_HEX((RGB * 255).astype(np.uint8)).astype(object)
    HEX = np.asarray('#') + HEX[..., 0] + HEX[..., 1] + HEX[..., 2]

    return HEX


def HEX_to_RGB(HEX):
    """
    Converts from hexadecimal triplet representation to *RGB* colourspace.

    Parameters
    ----------
    HEX : unicode or array_like
        Hexadecimal triplet representation.

    Returns
    -------
    ndarray
        *RGB* colourspace array.

    Notes
    -----
    -   Output *RGB* colourspace array is in range [0, 1].

    Examples
    --------
    >>> HEX = '#aaddff'
    >>> HEX_to_RGB(HEX)  # doctest: +ELLIPSIS
    array([ 0.6666666...,  0.8666666...,  1.        ])
    """

    HEX = np.core.defchararray.lstrip(HEX, '#')

    def to_RGB(x):
        """
        Converts given hexadecimal representation to *RGB*.
        """

        length = len(x)
        return [int(x[i:i + length // 3], 16)
                for i in range(0, length, length // 3)]

    to_RGB_v = np.vectorize(to_RGB, otypes=[np.ndarray])

    RGB = np.asarray(to_RGB_v(HEX).tolist()) / 255

    return RGB
