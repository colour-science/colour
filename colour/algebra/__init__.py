from __future__ import annotations

import sys
import typing

from colour.utilities.deprecation import ModuleAPI, build_API_changes
from colour.utilities.documentation import is_documentation_building

if typing.TYPE_CHECKING:
    from colour.hints import Any

from .common import (
    eigen_decomposition,
    euclidean_distance,
    get_sdiv_mode,
    is_identity,
    is_spow_enabled,
    lerp,
    linear_conversion,
    linstep_function,
    manhattan_distance,
    normalise_maximum,
    normalise_vector,
    sdiv,
    sdiv_mode,
    set_sdiv_mode,
    set_spow_enable,
    smooth,
    smoothstep_function,
    spow,
    spow_enable,
    vecmul,
)

# isort: split

from . import coordinates
from .coordinates import *  # noqa: F403
from .interpolation import (
    TABLE_INTERPOLATION_METHODS,
    CubicSplineInterpolator,
    KernelInterpolator,
    LinearInterpolator,
    NearestNeighbourInterpolator,
    NullInterpolator,
    PchipInterpolator,
    SpragueInterpolator,
    kernel_cardinal_spline,
    kernel_lanczos,
    kernel_linear,
    kernel_nearest_neighbour,
    kernel_sinc,
    lagrange_coefficients,
    table_interpolation,
    table_interpolation_tetrahedral,
    table_interpolation_trilinear,
)

# isort: split

from .extrapolation import Extrapolator
from .prng import random_triplet_generator
from .regression import least_square_mapping_MoorePenrose

__all__ = [
    "eigen_decomposition",
    "euclidean_distance",
    "get_sdiv_mode",
    "is_identity",
    "is_spow_enabled",
    "lerp",
    "linear_conversion",
    "linstep_function",
    "manhattan_distance",
    "normalise_maximum",
    "normalise_vector",
    "sdiv",
    "sdiv_mode",
    "set_sdiv_mode",
    "set_spow_enable",
    "smooth",
    "smoothstep_function",
    "spow",
    "spow_enable",
    "vecmul",
]
__all__ += coordinates.__all__
__all__ += [
    "TABLE_INTERPOLATION_METHODS",
    "CubicSplineInterpolator",
    "KernelInterpolator",
    "LinearInterpolator",
    "NearestNeighbourInterpolator",
    "NullInterpolator",
    "PchipInterpolator",
    "SpragueInterpolator",
    "kernel_cardinal_spline",
    "kernel_lanczos",
    "kernel_linear",
    "kernel_nearest_neighbour",
    "kernel_sinc",
    "lagrange_coefficients",
    "table_interpolation",
    "table_interpolation_tetrahedral",
    "table_interpolation_trilinear",
]
__all__ += [
    "Extrapolator",
]
__all__ += [
    "random_triplet_generator",
]
__all__ += [
    "least_square_mapping_MoorePenrose",
]


# ----------------------------------------------------------------------------#
# ---                API Changes and Deprecation Management                ---#
# ----------------------------------------------------------------------------#
class algebra(ModuleAPI):
    """Define a class acting like the *algebra* module."""

    def __getattr__(self, attribute: str) -> Any:
        """Return the value from the attribute with given name."""

        return super().__getattr__(attribute)


# v0.4.5
API_CHANGES: dict = {
    "ObjectRenamed": [
        [
            "colour.algebra.vector_dot",
            "colour.algebra.vecmul",
        ],
    ]
}
"""*colour.algebra* sub-package API changes."""


API_CHANGES["ObjectRemoved"] = [
    "colour.algebra.matrix_dot",
]

if not is_documentation_building():
    sys.modules["colour.algebra"] = algebra(  # pyright: ignore
        sys.modules["colour.algebra"], build_API_changes(API_CHANGES)
    )

    del ModuleAPI, is_documentation_building, build_API_changes, sys
