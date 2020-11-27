# -*- coding: utf-8 -*-
"""
Common Constants
================

Defines common constants objects that don't belong to any specific category.
"""

from __future__ import division, unicode_literals

import os
import numpy as np

from colour.utilities.documentation import (DocstringFloat,
                                            is_documentation_building)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'FLOATING_POINT_NUMBER_PATTERN', 'INTEGER_THRESHOLD', 'EPSILON',
    'DEFAULT_FLOAT_DTYPE', 'DEFAULT_INT_DTYPE'
]

FLOATING_POINT_NUMBER_PATTERN = '[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?'
"""
Floating point number regex matching pattern.

FLOATING_POINT_NUMBER_PATTERN : unicode
"""

INTEGER_THRESHOLD = 1e-3
if is_documentation_building():  # pragma: no cover
    INTEGER_THRESHOLD = DocstringFloat(INTEGER_THRESHOLD)
    INTEGER_THRESHOLD.__doc__ = """
Integer threshold value when checking if a floating point number is almost an
integer.

INTEGER_THRESHOLD : numeric
"""

EPSILON = np.finfo(np.float_).eps
"""
Default epsilon value for tolerance and singularities avoidance in various
computations.

EPSILON : numeric
"""

DEFAULT_FLOAT_DTYPE = np.sctypeDict.get(
    os.environ.get('COLOUR_SCIENCE__FLOAT_PRECISION', 'float64'), 'float64')
"""
Default floating point number dtype.

DEFAULT_FLOAT_DTYPE : type
"""

DEFAULT_INT_DTYPE = np.sctypeDict.get(
    os.environ.get('COLOUR_SCIENCE__INT_PRECISION', 'int64'), 'int64')
"""
Default integer number dtype.

DEFAULT_INT_DTYPE : type
"""
