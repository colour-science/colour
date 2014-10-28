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
__all__ += ['Extrapolator1d',
            'LinearInterpolator1d',
            'SplineInterpolator',
            'SpragueInterpolator',
            'is_identity',
            'linear_regression']
