#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

from .common import (
    steps,
    closest,
    as_array,
    is_uniform,
    is_iterable,
    is_numeric,
    is_integer,
    normalise)
from .coordinates import *  # noqa
from . import coordinates
from .extrapolation import Extrapolator1d
from .interpolation import (LinearInterpolator1d,
                            SplineInterpolator,
                            SpragueInterpolator)
from .matrix import is_identity
from .random import random_triplet_generator
from .regression import linear_regression

__all__ = ['steps',
           'closest',
           'as_array',
           'is_uniform',
           'is_iterable',
           'is_numeric',
           'is_integer',
           'normalise']
__all__ += coordinates.__all__
__all__ += ['Extrapolator1d']
__all__ += ['LinearInterpolator1d',
            'SplineInterpolator',
            'SpragueInterpolator']
__all__ += ['is_identity']
__all__ += ['random_triplet_generator']
__all__ += ['linear_regression']
