# -*- coding: utf-8 -*-
"""
Common Transfer Functions Utilities
===================================

Defines various transfer functions common utilities.

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.constants import DEFAULT_FLOAT_DTYPE, DEFAULT_INT_DTYPE
from colour.utilities import as_float_array

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['CV_range', 'legal_to_full', 'full_to_legal']


def CV_range(bit_depth=10, is_legal=False, is_int=False):
    """
    Returns the code value :math:`CV` range for given bit depth, range legality
    and representation.

    Parameters
    ----------
    bit_depth : int, optional
        Bit depth of the code value :math:`CV` range.
    is_legal : bool, optional
        Whether the code value :math:`CV` range is legal.
    is_int : bool, optional
        Whether the code value :math:`CV` range represents integer code values.

    Returns
    -------
    ndarray
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
        ranges = ranges.astype(DEFAULT_FLOAT_DTYPE) / (2 ** bit_depth - 1)

    return ranges


def legal_to_full(CV, bit_depth=10, in_int=False, out_int=False):
    """
    Converts given code value :math:`CV` or float equivalent of a code value at
    a given bit depth from legal range (studio swing) to full range
    (full swing).

    Parameters
    ----------
    CV : array_like
        Legal range code value :math:`CV` or float equivalent of a code value
        at a given bit depth.
    bit_depth : int, optional
        Bit depth used for conversion.
    in_int : bool, optional
        Whether to treat the input value as integer code value or float
        equivalent of a code value at a given bit depth.
    out_int : bool, optional
        Whether to return value as integer code value or float equivalent of a
        code value at a given bit depth.

    Returns
    -------
    ndarray
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

    CV = np.round(CV).astype(DEFAULT_INT_DTYPE) if in_int else CV * MV

    B, W = CV_range(bit_depth, True, True)

    CV = (CV - B) / (W - B)

    return np.round(CV * MV).astype(DEFAULT_INT_DTYPE) if out_int else CV


def full_to_legal(CV, bit_depth=10, in_int=False, out_int=False):
    """
    Converts given code value :math:`CV` or float equivalent of a code value at
    a given bit depth from full range (full swing) to legal range
    (studio swing).

    Parameters
    ----------
    CV : array_like
        Full range code value :math:`CV` or float equivalent of a code value at
        a given bit depth.
    bit_depth : int, optional
        Bit depth used for conversion.
    in_int : bool, optional
        Whether to treat the input value as integer code value or float
        equivalent of a code value at a given bit depth.
    out_int : bool, optional
        Whether to return value as integer code value or float equivalent of a
        code value at a given bit depth.

    Returns
    -------
    ndarray
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

    CV = np.round(CV / MV).astype(DEFAULT_INT_DTYPE) if in_int else CV

    B, W = CV_range(bit_depth, True, True)

    CV = (W - B) * CV + B

    return np.round(CV).astype(DEFAULT_INT_DTYPE) if out_int else CV / MV
