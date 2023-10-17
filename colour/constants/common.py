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
from colour.hints import DTypeFloat, Type, Union, cast

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
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
"""float point number regex matching pattern."""

INTEGER_THRESHOLD: float = 1e-3
if is_documentation_building():  # pragma: no cover
    INTEGER_THRESHOLD = DocstringFloat(INTEGER_THRESHOLD)
    INTEGER_THRESHOLD.__doc__ = """
int threshold value when checking if a float point number is almost an
int.
"""

EPSILON: float = cast(float, np.finfo(np.float_).eps)
"""
Default epsilon value for tolerance and singularities avoidance in various
computations.
"""

DEFAULT_INT_DTYPE: Type[np.int32 | np.int64] = cast(
    Type[Union[np.int32, np.int64]],
    np.sctypeDict.get(
        os.environ.get("COLOUR_SCIENCE__DEFAULT_INT_DTYPE", "int64"), np.int64
    ),
)
"""Default int number dtype."""


DEFAULT_FLOAT_DTYPE: Type[DTypeFloat] = cast(
    Type[DTypeFloat],
    np.sctypeDict.get(
        os.environ.get("COLOUR_SCIENCE__DEFAULT_FLOAT_DTYPE", "float64"),
        np.float64,
    ),
)
"""Default floating point number dtype."""
