# -*- coding: utf-8 -*-

from .coordinates import *  # noqa
from . import coordinates
from .common import (
    is_spow_enabled,
    set_spow_enable,
    spow_enable,
    spow,
    normalise_maximum,
    vector_dot,
    matrix_dot,
    linear_conversion,
    linstep_function,
    lerp,
    smoothstep_function,
    smooth,
    is_identity,
)
from .geometry import (
    normalise_vector,
    euclidean_distance,
    manhattan_distance,
    extend_line_segment,
    LineSegmentsIntersections_Specification,
    intersect_line_segments,
    ellipse_coefficients_general_form,
    ellipse_coefficients_canonical_form,
    point_at_angle_on_ellipse,
    ellipse_fitting_Halir1998,
    ELLIPSE_FITTING_METHODS,
    ellipse_fitting,
)
from .interpolation import (
    kernel_nearest_neighbour,
    kernel_linear,
    kernel_sinc,
    kernel_lanczos,
    kernel_cardinal_spline,
    KernelInterpolator,
    NearestNeighbourInterpolator,
    LinearInterpolator,
    SpragueInterpolator,
    CubicSplineInterpolator,
    PchipInterpolator,
    NullInterpolator,
    lagrange_coefficients,
    table_interpolation_trilinear,
    table_interpolation_tetrahedral,
    TABLE_INTERPOLATION_METHODS,
    table_interpolation,
)
from .extrapolation import Extrapolator
from .random import random_triplet_generator
from .regression import least_square_mapping_MoorePenrose

__all__ = []
__all__ += coordinates.__all__
__all__ += [
    'is_spow_enabled',
    'set_spow_enable',
    'spow_enable',
    'spow',
    'normalise_maximum',
    'vector_dot',
    'matrix_dot',
    'linear_conversion',
    'linstep_function',
    'lerp',
    'smoothstep_function',
    'smooth',
    'is_identity',
]
__all__ += [
    'normalise_vector',
    'euclidean_distance',
    'manhattan_distance',
    'extend_line_segment',
    'LineSegmentsIntersections_Specification',
    'intersect_line_segments',
    'ellipse_coefficients_general_form',
    'ellipse_coefficients_canonical_form',
    'point_at_angle_on_ellipse',
    'ellipse_fitting_Halir1998',
    'ELLIPSE_FITTING_METHODS',
    'ellipse_fitting',
]
__all__ += [
    'kernel_nearest_neighbour',
    'kernel_linear',
    'kernel_sinc',
    'kernel_lanczos',
    'kernel_cardinal_spline',
    'KernelInterpolator',
    'NearestNeighbourInterpolator',
    'LinearInterpolator',
    'SpragueInterpolator',
    'CubicSplineInterpolator',
    'PchipInterpolator',
    'NullInterpolator',
    'lagrange_coefficients',
    'table_interpolation_trilinear',
    'table_interpolation_tetrahedral',
    'TABLE_INTERPOLATION_METHODS',
    'table_interpolation',
]
__all__ += [
    'Extrapolator',
]
__all__ += [
    'random_triplet_generator',
]
__all__ += [
    'least_square_mapping_MoorePenrose',
]
