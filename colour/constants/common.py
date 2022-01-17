# -*- coding: utf-8 -*-
"""
Common Constants
================

Defines the common constants objects that don't belong to any specific
category.
"""

import os
import numpy as np

from colour.utilities.documentation import (
    DocstringFloat,
    is_documentation_building,
)
from colour.hints import DTypeFloating, Union

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'FLOATING_POINT_NUMBER_PATTERN',
    'INTEGER_THRESHOLD',
    'EPSILON',
    'DEFAULT_FLOAT_DTYPE',
    'DEFAULT_INT_DTYPE',
]

FLOATING_POINT_NUMBER_PATTERN = '[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?'
"""
Floating point number regex matching pattern.

FLOATING_POINT_NUMBER_PATTERN : str
"""

INTEGER_THRESHOLD = 1e-3
if is_documentation_building():  # pragma: no cover
    INTEGER_THRESHOLD = DocstringFloat(INTEGER_THRESHOLD)
    INTEGER_THRESHOLD.__doc__ = """
Integer threshold value when checking if a floating point number is almost an
integer.

INTEGER_THRESHOLD : :class:`numpy.floating`
"""

EPSILON = np.finfo(np.float_).eps
"""
Default epsilon value for tolerance and singularities avoidance in various
computations.

EPSILON : :class:`numpy.floating`
"""

DEFAULT_FLOAT_DTYPE: DTypeFloating = (np.sctypeDict.get(
    os.environ.get('COLOUR_SCIENCE__FLOAT_PRECISION', 'float64'), np.float64))
"""
Default floating point number dtype.

DEFAULT_FLOAT_DTYPE : :class:`numpy.dtype`
"""

DEFAULT_INT_DTYPE: Union[np.int32, np.int64] = np.sctypeDict.get(
    os.environ.get('COLOUR_SCIENCE__INT_PRECISION', 'int64'), np.int64)
"""
Default integer number dtype.

DEFAULT_INT_DTYPE : :class:`numpy.dtype`
"""
