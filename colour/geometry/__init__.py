import sys

from colour.utilities.deprecation import ModuleAPI, build_API_changes
from colour.utilities.documentation import is_documentation_building

from colour.hints import Any

from .ellipse import (
    ellipse_coefficients_general_form,
    ellipse_coefficients_canonical_form,
    point_at_angle_on_ellipse,
    ellipse_fitting_Halir1998,
    ELLIPSE_FITTING_METHODS,
    ellipse_fitting,
)
from .intersection import (
    extend_line_segment,
    LineSegmentsIntersections_Specification,
    intersect_line_segments,
)
from .primitives import MAPPING_PLANE_TO_AXIS, primitive_grid, primitive_cube
from .primitives import PRIMITIVE_METHODS, primitive
from .section import hull_section
from .vertices import (
    primitive_vertices_quad_mpl,
    primitive_vertices_grid_mpl,
    primitive_vertices_cube_mpl,
    primitive_vertices_sphere,
)
from .vertices import PRIMITIVE_VERTICES_METHODS, primitive_vertices


__all__ = [
    "ellipse_coefficients_general_form",
    "ellipse_coefficients_canonical_form",
    "point_at_angle_on_ellipse",
    "ellipse_fitting_Halir1998",
    "ELLIPSE_FITTING_METHODS",
    "ellipse_fitting",
]
__all__ += [
    "extend_line_segment",
    "LineSegmentsIntersections_Specification",
    "intersect_line_segments",
]
__all__ += [
    "MAPPING_PLANE_TO_AXIS",
    "primitive_grid",
    "primitive_cube",
]
__all__ += [
    "hull_section",
]
__all__ += [
    "PRIMITIVE_METHODS",
    "primitive",
]
__all__ += [
    "primitive_vertices_quad_mpl",
    "primitive_vertices_grid_mpl",
    "primitive_vertices_cube_mpl",
    "primitive_vertices_sphere",
]
__all__ += [
    "PRIMITIVE_VERTICES_METHODS",
    "primitive_vertices",
]


# ----------------------------------------------------------------------------#
# ---                API Changes and Deprecation Management                ---#
# ----------------------------------------------------------------------------#
class geometry(ModuleAPI):
    """Define a class acting like the *geometry* module."""

    def __getattr__(self, attribute) -> Any:
        """Return the value from the attribute with given name."""

        return super().__getattr__(attribute)


# v0.4.0
API_CHANGES = {
    "ObjectRenamed": [
        [
            "colour.geometry.PLANE_TO_AXIS_MAPPING",
            "colour.geometry.MAPPING_PLANE_TO_AXIS",
        ],
    ]
}
"""Defines the *colour.geometry* sub-package API changes."""

if not is_documentation_building():
    sys.modules["colour.geometry"] = geometry(  # type:ignore[assignment]
        sys.modules["colour.geometry"], build_API_changes(API_CHANGES)
    )

    del ModuleAPI, is_documentation_building, build_API_changes, sys
