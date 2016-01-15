#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

from .coordinates import *  # noqa
from . import coordinates
from .extrapolation import Extrapolator
from .geometry import (
    normalise_vector,
    LineSegmentsIntersections_Specification,
    line_segments_intersections)
from .interpolation import (LinearInterpolator,
                            SpragueInterpolator,
                            CubicSplineInterpolator,
                            PchipInterpolator)
from .matrix import is_identity
from .random import random_triplet_generator

__all__ = []
__all__ += coordinates.__all__
__all__ += ['Extrapolator']
__all__ += ['normalise_vector',
            'LineSegmentsIntersections_Specification',
            'line_segments_intersections']
__all__ += ['LinearInterpolator',
            'SpragueInterpolator',
            'CubicSplineInterpolator',
            'PchipInterpolator']
__all__ += ['is_identity']
__all__ += ['random_triplet_generator']
