"""
Common Constants
================

Defines the common constants objects that don't belong to any specific
category.
"""

from __future__ import annotations

import os
import numpy as np

from colour.utilities.documentation import (
    DocstringFloat,
    is_documentation_building,
)
from colour.hints import DTypeFloating, Floating, Type, Union, cast

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "FLOATING_POINT_NUMBER_PATTERN",
    "INTEGER_THRESHOLD",
    "EPSILON",
    "DEFAULT_INT_DTYPE",
    "DEFAULT_FLOAT_DTYPE",
]

FLOATING_POINT_NUMBER_PATTERN: str = "[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?"
"""Floating point number regex matching pattern."""

INTEGER_THRESHOLD: float = 1e-3
if is_documentation_building():  # pragma: no cover
    INTEGER_THRESHOLD = DocstringFloat(INTEGER_THRESHOLD)
    INTEGER_THRESHOLD.__doc__ = """
Integer threshold value when checking if a floating point number is almost an
integer.
"""

EPSILON: Floating = cast(Floating, np.finfo(np.float_).eps)
"""
Default epsilon value for tolerance and singularities avoidance in various
computations.
"""

DEFAULT_INT_DTYPE: Type[Union[np.int32, np.int64]] = cast(
    Type[Union[np.int32, np.int64]],
    np.sctypeDict.get(
        os.environ.get("COLOUR_SCIENCE__DEFAULT_INT_DTYPE", "int64"), np.int64
    ),
)
"""Default integer number dtype."""


DEFAULT_FLOAT_DTYPE: Type[DTypeFloating] = cast(
    Type[DTypeFloating],
    np.sctypeDict.get(
        os.environ.get("COLOUR_SCIENCE__DEFAULT_FLOAT_DTYPE", "float64"),
        np.float64,
    ),
)
"""Default floating point number dtype."""
