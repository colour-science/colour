# -*- coding: utf-8 -*-
"""
YCoCg Colour Encoding
======================

Defines the *YCoCg* colour encoding related transformations:

-   :func:`colour.RGB_to_YCoCg`
-   :func:`colour.YCoCg_to_RGB`

References
----------
-   :cite:`Malvar2003` : Malvar, H., & Sullivan, G. (2003). YCoCg-R: A Color
    Space with RGB Reversibility and Low Dynamic Range.
    https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/\
Malvar_Sullivan_YCoCg-R_JVT-I014r3-2.pdf
"""

from __future__ import annotations

import numpy as np

from colour.algebra import vector_dot
from colour.hints import ArrayLike, NDArray

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Development'

__all__ = [
    'MATRIX_RGB_TO_YCOCG',
    'MATRIX_YCOCG_TO_RGB',
    'RGB_to_YCoCg',
    'YCoCg_to_RGB',
]

MATRIX_RGB_TO_YCOCG: NDArray = np.array([
    [1 / 4, 1 / 2, 1 / 4],
    [1 / 2, 0, -1 / 2],
    [-1 / 4, 1 / 2, -1 / 4],
])
"""
*R'G'B'* colourspace to *YCoCg* colour encoding matrix.
"""

MATRIX_YCOCG_TO_RGB: NDArray = np.array([
    [1, 1, -1],
    [1, 0, 1],
    [1, -1, -1],
])
"""
*YCoCg* colour encoding to *R'G'B'* colourspace matrix.
"""


def RGB_to_YCoCg(RGB: ArrayLike) -> NDArray:
    """
    Converts an array of *R'G'B'* values to the corresponding *YCoCg* colour
    encoding values array.

    Parameters
    ----------
    RGB
        Input *R'G'B'* array.

    Returns
    -------
    :class:`numpy.ndarray`
        *YCoCg* colour encoding array.

    References
    ----------
    :cite:`Malvar2003`

    Examples
    --------
    >>> RGB_to_YCoCg(np.array([1.0, 1.0, 1.0]))
    array([ 1.,  0.,  0.])
    >>> RGB_to_YCoCg(np.array([0.75, 0.5, 0.5]))
    array([ 0.5625,  0.125 , -0.0625])
    """

    return vector_dot(MATRIX_RGB_TO_YCOCG, RGB)


def YCoCg_to_RGB(YCoCg: ArrayLike) -> NDArray:
    """
    Converts an array of *YCoCg* colour encoding values to the corresponding
    *R'G'B'* values array.

    Parameters
    ----------
    YCoCg
        *YCoCg* colour encoding array.

    Returns
    -------
    :class:`numpy.ndarray`
        Output *R'G'B'* array.

    References
    ----------
    :cite:`Malvar2003`

    Examples
    --------
    >>> YCoCg_to_RGB(np.array([1.0, 0.0, 0.0]))
    array([ 1.,  1.,  1.])
    >>> YCoCg_to_RGB(np.array([0.5625, 0.125, -0.0625]))
    array([ 0.75,  0.5 ,  0.5 ])
    """

    return vector_dot(MATRIX_YCOCG_TO_RGB, YCoCg)
