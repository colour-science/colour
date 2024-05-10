"""
Common Constants
================

Defines the common constants objects that don't belong to any specific
category.
"""

from __future__ import annotations

import os

import numpy as np

from colour.hints import DTypeFloat, Type, Union, cast
from colour.utilities.documentation import (
    DocstringFloat,
    is_documentation_building,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "PATTERN_FLOATING_POINT_NUMBER",
    "THRESHOLD_INTEGER",
    "EPSILON",
    "DTYPE_INT_DEFAULT",
    "DTYPE_FLOAT_DEFAULT",
    "TOLERANCE_ABSOLUTE_DEFAULT",
    "TOLERANCE_RELATIVE_DEFAULT",
    "TOLERANCE_ABSOLUTE_TESTS",
    "TOLERANCE_RELATIVE_TESTS",
]

PATTERN_FLOATING_POINT_NUMBER: str = "[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?"
"""Floating point number regex matching pattern."""

THRESHOLD_INTEGER: float = 1e-3
if is_documentation_building():  # pragma: no cover
    THRESHOLD_INTEGER = DocstringFloat(THRESHOLD_INTEGER)
    THRESHOLD_INTEGER.__doc__ = """
Integer threshold value when checking if a float point number is almost an
int.
"""

EPSILON: float = cast(float, np.finfo(np.float_).eps)
"""
Default epsilon value for tolerance and singularities avoidance in various
computations.
"""

DTYPE_INT_DEFAULT: Type[np.int32 | np.int64] = cast(
    Type[Union[np.int32, np.int64]],
    np.sctypeDict.get(
        os.environ.get("COLOUR_SCIENCE__DEFAULT_INT_DTYPE", "int64"), np.int64
    ),
)
"""Default int number dtype."""


DTYPE_FLOAT_DEFAULT: Type[DTypeFloat] = cast(
    Type[DTypeFloat],
    np.sctypeDict.get(
        os.environ.get("COLOUR_SCIENCE__DEFAULT_FLOAT_DTYPE", "float64"),
        np.float64,
    ),
)
"""Default floating point number dtype."""

TOLERANCE_ABSOLUTE_DEFAULT: float = 1e-8
"""Default absolute tolerance for computations."""

TOLERANCE_RELATIVE_DEFAULT: float = 1e-8
"""Default relative tolerance for computations."""

TOLERANCE_ABSOLUTE_TESTS: float = 1e-7
"""Absolute tolerance for computations during unit tests."""

TOLERANCE_RELATIVE_TESTS: float = 1e-7
"""Relative tolerance for computations during unit tests."""
