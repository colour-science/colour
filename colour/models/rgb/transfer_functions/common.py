# -*- coding: utf-8 -*-
"""
Common Transfer Functions Utilities
===================================

Defines various transfer functions common utilities.
"""

from __future__ import annotations

import numpy as np

from colour.hints import (
    Boolean,
    FloatingOrArrayLike,
    FloatingOrNDArray,
    Integer,
    IntegerOrArrayLike,
    IntegerOrNDArray,
    NDArray,
    Union,
)
from colour.utilities import as_float, as_int, as_float_array, as_int_array

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'CV_range',
    'legal_to_full',
    'full_to_legal',
]


def CV_range(bit_depth: Integer = 10,
             is_legal: Boolean = False,
             is_int: Boolean = False) -> NDArray:
    """
    Returns the code value :math:`CV` range for given bit depth, range legality
    and representation.

    Parameters
    ----------
    bit_depth
        Bit depth of the code value :math:`CV` range.
    is_legal
        Whether the code value :math:`CV` range is legal.
    is_int
        Whether the code value :math:`CV` range represents integer code values.

    Returns
    -------
    :class:`numpy.ndarray`
        Code value :math:`CV` range.

    Examples
    --------
    >>> CV_range(8, True, True)
    array([ 16, 235])
    >>> CV_range(8, True, False)  # doctest: +ELLIPSIS
    array([ 0.0627451...,  0.9215686...])
    >>> CV_range(10, False, False)
    array([ 0.,  1.])
    """

    if is_legal:
        ranges = np.array([16, 235])
        ranges *= 2 ** (bit_depth - 8)
    else:
        ranges = np.array([0, 2 ** bit_depth - 1])

    if not is_int:
        ranges = as_float_array(ranges) / (2 ** bit_depth - 1)

    return ranges


def legal_to_full(CV: Union[FloatingOrArrayLike, IntegerOrArrayLike],
                  bit_depth: Integer = 10,
                  in_int: Boolean = False,
                  out_int: Boolean = False
                  ) -> Union[FloatingOrNDArray, IntegerOrNDArray]:
    """
    Converts given code value :math:`CV` or float equivalent of a code value at
    a given bit depth from legal range (studio swing) to full range
    (full swing).

    Parameters
    ----------
    CV
        Legal range code value :math:`CV` or float equivalent of a code value
        at a given bit depth.
    bit_depth
        Bit depth used for conversion.
    in_int
        Whether to treat the input value as integer code value or float
        equivalent of a code value at a given bit depth.
    out_int
        Whether to return value as integer code value or float equivalent of a
        code value at a given bit depth.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.integer` or :class:`numpy.ndarray`
        Full range code value :math:`CV` or float equivalent of a code value
        at a given bit depth.

    Examples
    --------
    >>> legal_to_full(64 / 1023)
    0.0
    >>> legal_to_full(940 / 1023)
    1.0
    >>> legal_to_full(64 / 1023, out_int=True)
    0
    >>> legal_to_full(940 / 1023, out_int=True)
    1023
    >>> legal_to_full(64, in_int=True)
    0.0
    >>> legal_to_full(940, in_int=True)
    1.0
    >>> legal_to_full(64, in_int=True, out_int=True)
    0
    >>> legal_to_full(940, in_int=True, out_int=True)
    1023
    """

    CV = as_float_array(CV)

    MV = 2 ** bit_depth - 1

    CV_full = as_int_array(np.round(CV)) if in_int else CV * MV

    B, W = CV_range(bit_depth, True, True)

    CV_full = (CV_full - B) / (W - B)

    if out_int:
        return as_int(np.round(CV_full * MV))
    else:
        return as_float(CV_full)


def full_to_legal(CV: Union[FloatingOrArrayLike, IntegerOrArrayLike],
                  bit_depth: Integer = 10,
                  in_int: Boolean = False,
                  out_int: Boolean = False
                  ) -> Union[FloatingOrNDArray, IntegerOrNDArray]:
    """
    Converts given code value :math:`CV` or float equivalent of a code value at
    a given bit depth from full range (full swing) to legal range
    (studio swing).

    Parameters
    ----------
    CV
        Full range code value :math:`CV` or float equivalent of a code value at
        a given bit depth.
    bit_depth
        Bit depth used for conversion.
    in_int
        Whether to treat the input value as integer code value or float
        equivalent of a code value at a given bit depth.
    out_int
        Whether to return value as integer code value or float equivalent of a
        code value at a given bit depth.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.integer` or :class:`numpy.ndarray`
        Legal range code value :math:`CV` or float equivalent of a code value
        at a given bit depth.

    Examples
    --------
    >>> full_to_legal(0.0)  # doctest: +ELLIPSIS
    0.0625610...
    >>> full_to_legal(1.0)  # doctest: +ELLIPSIS
    0.9188660...
    >>> full_to_legal(0.0, out_int=True)
    64
    >>> full_to_legal(1.0, out_int=True)
    940
    >>> full_to_legal(0, in_int=True)  # doctest: +ELLIPSIS
    0.0625610...
    >>> full_to_legal(1023, in_int=True)  # doctest: +ELLIPSIS
    0.9188660...
    >>> full_to_legal(0, in_int=True, out_int=True)
    64
    >>> full_to_legal(1023, in_int=True, out_int=True)
    940
    """

    CV = as_float_array(CV)

    MV = 2 ** bit_depth - 1

    CV_legal = as_int_array(np.round(CV / MV)) if in_int else CV

    B, W = CV_range(bit_depth, True, True)

    CV_legal = (W - B) * CV_legal + B

    if out_int:
        return as_int(np.round(CV_legal))
    else:
        return as_float(CV_legal / MV)
