# -*- coding: utf-8 -*-
"""
LUT Processing Common Utilities
===============================

Defines LUT Processing common utilities objects that don't fall in any specific
category.
"""

from __future__ import division, unicode_literals

import os
import re

from colour.constants import DEFAULT_FLOAT_DTYPE
from colour.utilities import as_array, is_string

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['parse_array', 'path_to_title']


def parse_array(a, separator=' ', dtype=DEFAULT_FLOAT_DTYPE):
    """
    Converts given string or array of strings to :class:`ndarray` class.

    Parameters
    ----------
    a : unicode or array_like
        String or array of strings to convert.
    separator : unicode
        Separator to split the string with.
    dtype : object
        Type to use for conversion.

    Returns
    -------
    ndarray
        Converted string or array of strings.

    Examples
    --------
    >>> parse_array('-0.25 0.5 0.75')
    array([-0.25,  0.5 ,  0.75])
    >>> parse_array(['-0.25', '0.5', '0.75'])
    array([-0.25,  0.5 ,  0.75])
    """

    if is_string(a):
        a = a.split(separator)

    return as_array([dtype(token) for token in a], dtype)


def path_to_title(path):
    """
    Converts given file path to title.

    Parameters
    ----------
    path : unicode
        File path to convert to title.

    Returns
    -------
    unicode
        File path converted to title.

    Examples
    --------
    >>> # Doctests skip for Python 2.x compatibility.
    >>> path_to_title(
    ...     'colour/io/luts/tests/resources/sony_spi3d/ColourCorrect.spi3d'
    ... )  # doctest: +SKIP
    u'ColourCorrect'
    """

    return re.sub('_|-|\\.', ' ', os.path.splitext(os.path.basename(path))[0])
