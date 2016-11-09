#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

from .common import (
    handle_numpy_errors,
    ignore_numpy_errors,
    raise_numpy_errors,
    print_numpy_errors,
    warn_numpy_errors,
    ignore_python_warnings,
    batch,
    is_openimageio_installed,
    is_iterable,
    is_string,
    is_numeric,
    is_integer,
    filter_kwargs)
from .array import (
    as_numeric,
    closest,
    normalise_maximum,
    interval,
    is_uniform,
    in_array,
    tstack,
    tsplit,
    row_as_diagonal,
    dot_vector,
    dot_matrix,
    orient,
    centroid,
    linear_conversion)
from .data_structures import (
    ArbitraryPrecisionMapping,
    Lookup,
    Structure,
    CaseInsensitiveMapping)
from .verbose import ColourWarning, message_box, warning, filter_warnings

__all__ = ['handle_numpy_errors',
           'ignore_numpy_errors',
           'raise_numpy_errors',
           'print_numpy_errors',
           'warn_numpy_errors',
           'ignore_python_warnings',
           'batch',
           'is_openimageio_installed',
           'is_iterable',
           'is_string',
           'is_numeric',
           'is_integer',
           'filter_kwargs']
__all__ += ['as_numeric',
            'closest',
            'normalise_maximum',
            'interval',
            'is_uniform',
            'in_array',
            'tstack',
            'tsplit',
            'row_as_diagonal',
            'dot_vector',
            'dot_matrix',
            'orient',
            'centroid',
            'linear_conversion']
__all__ += ['ArbitraryPrecisionMapping',
            'Lookup',
            'Structure',
            'CaseInsensitiveMapping']
__all__ += ['ColourWarning', 'message_box', 'warning', 'filter_warnings']
