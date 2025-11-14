from __future__ import annotations

import sys
import typing

if typing.TYPE_CHECKING:
    from colour.hints import Any

from .verbose import (
    ANCILLARY_COLOUR_SCIENCE_PACKAGES,
    ANCILLARY_DEVELOPMENT_PACKAGES,
    ANCILLARY_EXTRAS_PACKAGES,
    ANCILLARY_RUNTIME_PACKAGES,
    ColourRuntimeWarning,
    ColourUsageWarning,
    ColourWarning,
    MixinLogging,
    as_bool,
    describe_environment,
    filter_warnings,
    message_box,
    multiline_repr,
    multiline_str,
    numpy_print_options,
    runtime_warning,
    show_warning,
    suppress_stdout,
    suppress_warnings,
    usage_warning,
    warning,
)

# isort: split

from .structures import (
    CanonicalMapping,
    LazyCanonicalMapping,
    Lookup,
    Structure,
)

# isort: split

from .requirements import (
    is_ctlrender_installed,
    is_imageio_installed,
    is_matplotlib_installed,
    is_networkx_installed,
    is_opencolorio_installed,
    is_openimageio_installed,
    is_pandas_installed,
    is_pydot_installed,
    is_scipy_installed,
    is_tqdm_installed,
    is_trimesh_installed,
    is_xxhash_installed,
    required,
)

# isort: split

from .callback import (
    Callback,
    MixinCallback,
)
from .common import (
    CACHE_REGISTRY,
    CacheRegistry,
    attest,
    batch,
    caching_enable,
    copy_definition,
    disable_multiprocessing,
    filter_kwargs,
    filter_mapping,
    first_item,
    handle_numpy_errors,
    ignore_numpy_errors,
    ignore_python_warnings,
    int_digest,
    is_caching_enabled,
    is_integer,
    is_iterable,
    is_numeric,
    is_sibling,
    multiprocessing_pool,
    optional,
    print_numpy_errors,
    raise_numpy_errors,
    set_caching_enable,
    slugify,
    validate_method,
    warn_numpy_errors,
)

# isort: split

from .array import (
    MixinDataclassArithmetic,
    MixinDataclassArray,
    MixinDataclassFields,
    MixinDataclassIterable,
    as_array,
    as_complex_array,
    as_float,
    as_float_array,
    as_float_scalar,
    as_int,
    as_int_array,
    as_int_scalar,
    centroid,
    closest,
    closest_indexes,
    domain_range_scale,
    fill_nan,
    format_array_as_row,
    from_range_1,
    from_range_10,
    from_range_100,
    from_range_degrees,
    from_range_int,
    full,
    get_domain_range_scale,
    get_domain_range_scale_metadata,
    has_only_nan,
    in_array,
    index_along_last_axis,
    interval,
    is_ndarray_copy_enabled,
    is_uniform,
    ndarray_copy,
    ndarray_copy_enable,
    ndarray_write,
    ones,
    orient,
    row_as_diagonal,
    set_default_float_dtype,
    set_default_int_dtype,
    set_domain_range_scale,
    set_ndarray_copy_enable,
    to_domain_1,
    to_domain_10,
    to_domain_100,
    to_domain_degrees,
    to_domain_int,
    tsplit,
    tstack,
    zeros,
)
from .metrics import metric_mse, metric_psnr
from .network import (
    ControlFlowNode,
    ExecutionNode,
    ExecutionPort,
    For,
    ParallelForMultiprocess,
    ParallelForThread,
    Port,
    PortGraph,
    PortNode,
    TreeNode,
)

# isort: split

from colour.utilities.deprecation import ModuleAPI, build_API_changes
from colour.utilities.documentation import is_documentation_building

__all__ = [
    "ANCILLARY_COLOUR_SCIENCE_PACKAGES",
    "ANCILLARY_DEVELOPMENT_PACKAGES",
    "ANCILLARY_EXTRAS_PACKAGES",
    "ANCILLARY_RUNTIME_PACKAGES",
    "ColourRuntimeWarning",
    "ColourUsageWarning",
    "ColourWarning",
    "MixinLogging",
    "as_bool",
    "describe_environment",
    "filter_warnings",
    "message_box",
    "multiline_repr",
    "multiline_str",
    "numpy_print_options",
    "runtime_warning",
    "show_warning",
    "suppress_stdout",
    "suppress_warnings",
    "usage_warning",
    "warning",
]
__all__ += [
    "CanonicalMapping",
    "LazyCanonicalMapping",
    "Lookup",
    "Structure",
]
__all__ += [
    "is_ctlrender_installed",
    "is_imageio_installed",
    "is_matplotlib_installed",
    "is_networkx_installed",
    "is_opencolorio_installed",
    "is_openimageio_installed",
    "is_pandas_installed",
    "is_pydot_installed",
    "is_scipy_installed",
    "is_tqdm_installed",
    "is_trimesh_installed",
    "is_xxhash_installed",
    "required",
]
__all__ += [
    "Callback",
    "MixinCallback",
]
__all__ += [
    "CACHE_REGISTRY",
    "CacheRegistry",
    "attest",
    "batch",
    "caching_enable",
    "copy_definition",
    "disable_multiprocessing",
    "filter_kwargs",
    "filter_mapping",
    "first_item",
    "handle_numpy_errors",
    "ignore_numpy_errors",
    "ignore_python_warnings",
    "int_digest",
    "is_caching_enabled",
    "is_integer",
    "is_iterable",
    "is_numeric",
    "is_sibling",
    "multiprocessing_pool",
    "optional",
    "print_numpy_errors",
    "raise_numpy_errors",
    "set_caching_enable",
    "slugify",
    "validate_method",
    "warn_numpy_errors",
]
__all__ += [
    "MixinDataclassArithmetic",
    "MixinDataclassArray",
    "MixinDataclassFields",
    "MixinDataclassIterable",
    "as_array",
    "as_complex_array",
    "as_float",
    "as_float_array",
    "as_float_scalar",
    "as_int",
    "as_int_array",
    "as_int_scalar",
    "centroid",
    "closest",
    "closest_indexes",
    "domain_range_scale",
    "fill_nan",
    "format_array_as_row",
    "from_range_1",
    "from_range_10",
    "from_range_100",
    "from_range_degrees",
    "from_range_int",
    "full",
    "get_domain_range_scale",
    "get_domain_range_scale_metadata",
    "has_only_nan",
    "in_array",
    "index_along_last_axis",
    "interval",
    "is_ndarray_copy_enabled",
    "is_uniform",
    "ndarray_copy",
    "ndarray_copy_enable",
    "ndarray_write",
    "ones",
    "orient",
    "row_as_diagonal",
    "set_default_float_dtype",
    "set_default_int_dtype",
    "set_domain_range_scale",
    "set_ndarray_copy_enable",
    "to_domain_1",
    "to_domain_10",
    "to_domain_100",
    "to_domain_degrees",
    "to_domain_int",
    "tsplit",
    "tstack",
    "zeros",
]
__all__ += [
    "metric_mse",
    "metric_psnr",
]
__all__ += [
    "ControlFlowNode",
    "ExecutionNode",
    "ExecutionPort",
    "For",
    "ParallelForMultiprocess",
    "ParallelForThread",
    "Port",
    "PortGraph",
    "PortNode",
    "TreeNode",
]


# ----------------------------------------------------------------------------#
# ---                API Changes and Deprecation Management                ---#
# ----------------------------------------------------------------------------#
class utilities(ModuleAPI):
    """Define a class acting like the *utilities* module."""

    def __getattr__(self, attribute: str) -> Any:
        """Return the value from the specified attribute."""

        return super().__getattr__(attribute)


# v0.4.5
API_CHANGES: dict = {
    "ObjectRemoved": [
        "colour.utilities.is_string",
    ]
}
"""
Define the *colour.utilities* sub-package API changes.

API_CHANGES
"""

if not is_documentation_building():
    sys.modules["colour.utilities"] = utilities(  # pyright: ignore
        sys.modules["colour.utilities"], build_API_changes(API_CHANGES)
    )

    del ModuleAPI, is_documentation_building, build_API_changes, sys
