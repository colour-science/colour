# -*- coding: utf-8 -*-
"""
Linear Colour Component Transfer Function
=========================================

Defines the linear encoding / decoding colour component transfer function
related objects:

- :func:`colour.linear_function`
"""

from __future__ import annotations

from colour.hints import FloatingOrArrayLike, FloatingOrNDArray
from colour.utilities import as_float

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'linear_function',
]


def linear_function(a: FloatingOrArrayLike) -> FloatingOrNDArray:
    """
    Defines a typical linear encoding / decoding function, essentially a
    pass-through function.

    Parameters
    ----------
    a
        Array to encode / decode.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Encoded / decoded array.

    Examples
    --------
    >>> linear_function(0.18)  # doctest: +ELLIPSIS
    0.1799999...
    """

    return as_float(a)
