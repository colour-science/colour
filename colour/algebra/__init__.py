#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

from .coordinates import *  # noqa
from . import coordinates
from .extrapolation import Extrapolator
from .geometry import (
    normalise_vector,
    euclidean_distance,
    extend_line_segment,
    LineSegmentsIntersections_Specification,
    intersect_line_segments)
from .interpolation import (
    LinearInterpolator,
    SpragueInterpolator,
    CubicSplineInterpolator,
    PchipInterpolator,
    lagrange_coefficients)
from .matrix import is_identity
from .random import random_triplet_generator

__all__ = []
__all__ += coordinates.__all__
__all__ += ['Extrapolator']
__all__ += ['normalise_vector',
            'euclidean_distance',
            'extend_line_segment',
            'LineSegmentsIntersections_Specification',
            'intersect_line_segments']
__all__ += ['LinearInterpolator',
            'SpragueInterpolator',
            'CubicSplineInterpolator',
            'PchipInterpolator',
            'lagrange_coefficients']
__all__ += ['is_identity']
__all__ += ['random_triplet_generator']
