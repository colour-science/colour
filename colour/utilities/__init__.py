# -*- coding: utf-8 -*-

from __future__ import absolute_import

from .common import (
    handle_numpy_errors, ignore_numpy_errors, raise_numpy_errors,
    print_numpy_errors, warn_numpy_errors, ignore_python_warnings, batch,
    is_openimageio_installed, is_pandas_installed, is_iterable, is_string,
    is_numeric, is_integer, filter_kwargs, first_item, get_domain_range_scale,
    set_domain_range_scale, domain_range_scale, to_domain_1, to_domain_10,
    to_domain_100, to_domain_degrees, to_domain_int, from_range_1,
    from_range_10, from_range_100, from_range_degrees, from_range_int)
from .array import (as_numeric, as_namedtuple, closest_indexes, closest,
                    normalise_maximum, interval, is_uniform, in_array, tstack,
                    tsplit, row_as_diagonal, dot_vector, dot_matrix, orient,
                    centroid, linear_conversion, fill_nan, ndarray_write)
from .data_structures import Lookup, Structure, CaseInsensitiveMapping
from .metrics import metric_mse, metric_psnr
from .verbose import (ColourWarning, message_box, show_warning, warning,
                      filter_warnings, suppress_warnings, numpy_print_options)

__all__ = [
    'handle_numpy_errors', 'ignore_numpy_errors', 'raise_numpy_errors',
    'print_numpy_errors', 'warn_numpy_errors', 'ignore_python_warnings',
    'batch', 'is_openimageio_installed', 'is_pandas_installed', 'is_iterable',
    'is_string', 'is_numeric', 'is_integer', 'filter_kwargs', 'first_item',
    'get_domain_range_scale', 'set_domain_range_scale', 'domain_range_scale',
    'to_domain_1', 'to_domain_10', 'to_domain_100', 'to_domain_degrees',
    'to_domain_int', 'from_range_1', 'from_range_10', 'from_range_100',
    'from_range_degrees', 'from_range_int'
]
__all__ += [
    'as_numeric', 'as_namedtuple', 'closest_indexes', 'closest',
    'normalise_maximum', 'interval', 'is_uniform', 'in_array', 'tstack',
    'tsplit', 'row_as_diagonal', 'dot_vector', 'dot_matrix', 'orient',
    'centroid', 'linear_conversion', 'fill_nan', 'ndarray_write'
]
__all__ += ['Lookup', 'Structure', 'CaseInsensitiveMapping']
__all__ += ['metric_mse', 'metric_psnr']
__all__ += [
    'ColourWarning', 'message_box', 'show_warning', 'warning',
    'filter_warnings', 'suppress_warnings', 'numpy_print_options'
]
