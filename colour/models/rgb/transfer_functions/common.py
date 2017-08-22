#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Common Transfer Functions Utilities
===================================

Defines various Transfer Functions common utilities.

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_
"""

from __future__ import division, unicode_literals

import numpy as np

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['CV_range', 'CV_to_IRE', 'IRE_to_CV']


def CV_range(bit_depth, is_legal, is_int):
    """"
    Returns the code value :math:`CV` range for given bit depth, range legality
    and representation.

    Parameters
    ----------
    bit_depth : int
        Bit depth of the code value :math:`CV` range.
    is_legal : bool
        Whether the code value :math:`CV` range is legal.
    is_int : bool
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
        ranges = ranges.astype(np.float_) / (2 ** bit_depth - 1)

    return ranges


def CV_to_IRE(CV, bit_depth, is_legal):
    """
    Converts from code values :math:`CV` to :math:`IRE`
    (Institute of Radio Engineers).

    Parameters
    ----------
    CV : array_like
        Code values :math:`CV`.
    bit_depth : int
        Bit depth used for conversion.
    is_legal : bool
        Whether the code value :math:`CV` range is legal.

    Returns
    -------
    ndarray
        :math:`IRE`.

    Examples
    --------
    >>> CV_to_IRE(390, 10, True)  # doctest: +ELLIPSIS
    37.2146118...
    >>> CV_to_IRE(390, 10, False)  # doctest: +ELLIPSIS
    38.1231671...
    """

    CV = np.asarray(CV)

    B, W = CV_range(bit_depth, is_legal, True)

    return (CV - B) / (W - B) * 100


def IRE_to_CV(IRE, bit_depth, is_legal):
    """
    Converts from :math:`IRE` (Institute of Radio Engineers) to code values
    :math:`CV`.

    Parameters
    ----------
    IRE : array_like
        :math:`IRE`.
    bit_depth : int
        Bit depth used for conversion.
    is_legal : bool
        Whether the code value :math:`CV` range is legal.

    Returns
    -------
    ndarray
        Code values :math:`CV`.

    Examples
    --------
    >>> IRE_to_CV(37.214611872146122, 10, True)  # doctest: +ELLIPSIS
    390...
    >>> IRE_to_CV(38.123167155425222, 10, False)  # doctest: +ELLIPSIS
    390...
    """

    IRE = np.asarray(IRE)

    B, W = CV_range(bit_depth, is_legal, True)

    return (W - B) * IRE / 100 + B
