# -*- coding: utf-8 -*-
"""
Kodak Cineon Encoding
=====================

Defines the *Kodak Cineon* encoding:

-   :func:`colour.models.log_encoding_Cineon`
-   :func:`colour.models.log_decoding_Cineon`

References
----------
-   :cite:`SonyImageworks2012a` : Sony Imageworks. (2012). make.py. Retrieved
    November 27, 2014, from
    https://github.com/imageworks/OpenColorIO-Configs/blob/master/\
nuke-default/make.py
"""

from __future__ import annotations

import numpy as np

from colour.hints import FloatingOrArrayLike, FloatingOrNDArray
from colour.utilities import (
    as_float,
    as_float_array,
    from_range_1,
    to_domain_1,
)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'log_encoding_Cineon',
    'log_decoding_Cineon',
]


def log_encoding_Cineon(x: FloatingOrArrayLike,
                        black_offset: FloatingOrArrayLike = 10
                        ** ((95 - 685) / 300)) -> FloatingOrNDArray:
    """
    Defines the *Cineon* log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    x
        Linear data :math:`x`.
    black_offset
        Black offset.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Non-linear data :math:`y`.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``y``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`SonyImageworks2012a`

    Examples
    --------
    >>> log_encoding_Cineon(0.18)  # doctest: +ELLIPSIS
    0.4573196...
    """

    x = to_domain_1(x)
    black_offset = as_float_array(black_offset)

    y = ((685 + 300 * np.log10(x * (1 - black_offset) + black_offset)) / 1023)

    return as_float(from_range_1(y))


def log_decoding_Cineon(y: FloatingOrArrayLike,
                        black_offset: FloatingOrArrayLike = 10
                        ** ((95 - 685) / 300)) -> FloatingOrNDArray:
    """
    Defines the *Cineon* log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    y
        Non-linear data :math:`y`.
    black_offset
        Black offset.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Linear data :math:`x`.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``y``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`SonyImageworks2012a`

    Examples
    --------
    >>> log_decoding_Cineon(0.457319613085418)  # doctest: +ELLIPSIS
    0.1799999...
    """

    y = to_domain_1(y)
    black_offset = as_float_array(black_offset)

    x = ((10 ** ((1023 * y - 685) / 300) - black_offset) / (1 - black_offset))

    return as_float(from_range_1(x))
