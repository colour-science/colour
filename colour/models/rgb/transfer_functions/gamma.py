# -*- coding: utf-8 -*-
"""
Gamma Colour Component Transfer Function
========================================

Defines the gamma encoding / decoding colour component transfer function
related objects:

- :func:`colour.gamma_function`
"""

from __future__ import annotations

import numpy as np

from colour.algebra import spow
from colour.hints import (
    FloatingOrArrayLike,
    FloatingOrNDArray,
    Literal,
    Union,
)
from colour.utilities import as_float_array, as_float, validate_method

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'gamma_function',
]


def gamma_function(a: FloatingOrArrayLike,
                   exponent: FloatingOrArrayLike = 1,
                   negative_number_handling: Union[
                       Literal['Clamp', 'Indeterminate', 'Mirror', 'Preserve'],
                       str] = 'Indeterminate') -> FloatingOrNDArray:
    """
    Defines a typical gamma encoding / decoding function.

    Parameters
    ----------
    a
        Array to encode / decode.
    exponent
        Encoding / decoding exponent.
    negative_number_handling
        Defines the behaviour for ``a`` negative numbers and / or the
        definition return value:

        -   *Indeterminate*: The behaviour will be indeterminate and
            definition return value might contain *nans*.
        -   *Mirror*: The definition return value will be mirrored around
            abscissa and ordinate axis, i.e. Blackmagic Design: Davinci Resolve
            behaviour.
        -   *Preserve*: The definition will preserve any negative number in
            ``a``, i.e. The Foundry Nuke behaviour.
        -   *Clamp*: The definition will clamp any negative number in ``a`` to
            0.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Encoded / decoded array.

    Examples
    --------
    >>> gamma_function(0.18, 2.2)  # doctest: +ELLIPSIS
    0.0229932...
    >>> gamma_function(-0.18, 2.0)  # doctest: +ELLIPSIS
    0.0323999...
    >>> gamma_function(-0.18, 2.2)
    nan
    >>> gamma_function(-0.18, 2.2, 'Mirror')  # doctest: +ELLIPSIS
    -0.0229932...
    >>> gamma_function(-0.18, 2.2, 'Preserve')  # doctest: +ELLIPSIS
    -0.1...
    >>> gamma_function(-0.18, 2.2, 'Clamp')  # doctest: +ELLIPSIS
    0.0
    """

    a = as_float_array(a)
    exponent = as_float_array(exponent)
    negative_number_handling = validate_method(
        negative_number_handling,
        ['Indeterminate', 'Mirror', 'Preserve', 'Clamp'],
        '"{0}" negative number handling is invalid, it must be one of {1}!')

    if negative_number_handling == 'indeterminate':
        return as_float(a ** exponent)
    elif negative_number_handling == 'mirror':
        return spow(a, exponent)
    elif negative_number_handling == 'preserve':
        return as_float(np.where(a <= 0, a, a ** exponent))
    else:  # negative_number_handling == 'clamp':
        return as_float(np.where(a <= 0, 0, a ** exponent))
