# -*- coding: utf-8 -*-
"""
Hexadecimal Notation
====================

Defines objects for hexadecimal notation:

-   :func:`colour.notation.RGB_to_HEX`
-   :func:`colour.notation.HEX_to_RGB`
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.models import eotf_inverse_sRGB, eotf_sRGB
from colour.utilities import (as_float_array, from_range_1, normalise_maximum,
                              to_domain_1, usage_warning)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['RGB_to_HEX', 'HEX_to_RGB']


def RGB_to_HEX(RGB):
    """
    Converts from *RGB* colourspace to hexadecimal representation.

    Parameters
    ----------
    RGB : array_like
        *RGB* colourspace array.

    Returns
    -------
    unicode
        Hexadecimal representation.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``RGB``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    Examples
    --------
    >>> RGB = np.array([0.66666667, 0.86666667, 1.00000000])
    >>> # Doctests skip for Python 2.x compatibility.
    >>> RGB_to_HEX(RGB)  # doctest: +SKIP
    '#aaddff'
    """

    RGB = to_domain_1(RGB)

    if np.any(RGB < 0):
        usage_warning(
            '"RGB" array contains negative values, those will be clipped, '
            'unpredictable results may occur!')

        RGB = np.clip(RGB, 0, np.inf)

    if np.any(RGB > 1):
        usage_warning(
            '"RGB" array contains values over 1 and will be normalised, '
            'unpredictable results may occur!')

        RGB = eotf_inverse_sRGB(normalise_maximum(eotf_sRGB(RGB)))

    to_HEX = np.vectorize('{0:02x}'.format)

    HEX = to_HEX((RGB * 255).astype(np.uint8)).astype(object)
    HEX = np.asarray('#') + HEX[..., 0] + HEX[..., 1] + HEX[..., 2]

    return HEX


def HEX_to_RGB(HEX):
    """
    Converts from hexadecimal representation to *RGB* colourspace.

    Parameters
    ----------
    HEX : unicode or array_like
        Hexadecimal representation.

    Returns
    -------
    ndarray
        *RGB* colourspace array.

    Notes
    -----

    +-----------+-----------------------+---------------+
    | **Range** | **Scale - Reference** | **Scale - 1** |
    +===========+=======================+===============+
    | ``RGB``   | [0, 1]                | [0, 1]        |
    +-----------+-----------------------+---------------+

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

        l_x = len(x)
        return [int(x[i:i + l_x // 3], 16) for i in range(0, l_x, l_x // 3)]

    to_RGB_v = np.vectorize(to_RGB, otypes=[np.ndarray])

    RGB = as_float_array(to_RGB_v(HEX).tolist()) / 255

    return from_range_1(RGB)
