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
    OrderedSet,
    Structure,
)

# isort: split

from .requirements import (
    is_array_api_compat_installed,
    is_array_api_extra_installed,
    is_ctlrender_installed,
    is_imageio_installed,
    is_matplotlib_installed,
    is_networkx_installed,
    is_onnxruntime_installed,
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
    optional,
    print_numpy_errors,
    raise_numpy_errors,
    set_caching_enabled,
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
    array_api_enable,
    array_namespace,
    as_array,
    as_complex_array,
    as_float,
    as_float_array,
    as_float_scalar,
    as_int,
    as_int_array,
    as_int_scalar,
    as_ndarray,
    cast_non_ndarray,
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
    is_array_api_enabled,
    is_ndarray_copy_enabled,
    is_non_ndarray,
    is_numpy_namespace,
    is_uniform,
    ndarray_copy,
    ndarray_copy_enable,
    ndarray_write,
    ones,
    orient,
    row_as_diagonal,
    set_array_api_enabled,
    set_default_complex_dtype,
    set_default_float_dtype,
    set_default_int_dtype,
    set_domain_range_scale,
    set_ndarray_copy_enabled,
    to_domain_1,
    to_domain_10,
    to_domain_100,
    to_domain_degrees,
    to_domain_int,
    trace_array_namespace,
    tsplit,
    tstack,
    xp_as_array,
    xp_as_float_array,
    xp_as_int_array,
    xp_ascontiguousarray,
    xp_assert_close,
    xp_assert_equal,
    xp_astype,
    xp_atleast_1d,
    xp_atleast_2d,
    xp_average,
    xp_broadcast_to,
    xp_create_diagonal,
    xp_degrees,
    xp_eig,
    xp_eigh,
    xp_gradient,
    xp_insert,
    xp_interp,
    xp_isclose,
    xp_isin,
    xp_linspace,
    xp_lstsq,
    xp_matrix_transpose,
    xp_median,
    xp_nan_to_num,
    xp_nanmean,
    xp_pad,
    xp_radians,
    xp_reshape,
    xp_resize,
    xp_round,
    xp_select,
    xp_setxor1d,
    xp_sinc,
    xp_squeeze,
    xp_trapezoid,
    xp_unique,
    zeros,
)
from .common import download_url, hash_sha256
from .delegate import Delegate
from .metrics import metric_mse, metric_psnr
from .network import (
    ControlFlowNode,
    ExecutionNode,
    ExecutionPort,
    For,
    NodeLog,
    NodePassthrough,
    NodeSetGraphOutputPort,
    NodeSleep,
    ParallelForMultiprocess,
    ParallelForThread,
    Port,
    PortGraph,
    PortNode,
    TreeNode,
    notify_process_state,
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
    "OrderedSet",
    "Structure",
]
__all__ += [
    "is_array_api_compat_installed",
    "is_array_api_extra_installed",
    "is_ctlrender_installed",
    "is_imageio_installed",
    "is_matplotlib_installed",
    "is_networkx_installed",
    "is_opencolorio_installed",
    "is_onnxruntime_installed",
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
    "optional",
    "print_numpy_errors",
    "raise_numpy_errors",
    "set_caching_enabled",
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
    "set_default_complex_dtype",
    "set_default_float_dtype",
    "set_default_int_dtype",
    "set_domain_range_scale",
    "set_ndarray_copy_enabled",
    "to_domain_1",
    "to_domain_10",
    "to_domain_100",
    "to_domain_degrees",
    "to_domain_int",
    "tsplit",
    "tstack",
    "zeros",
]
__all__ += ["Delegate"]
__all__ += [
    "array_api_enable",
    "array_namespace",
    "as_ndarray",
    "cast_non_ndarray",
    "is_array_api_enabled",
    "is_non_ndarray",
    "is_numpy_namespace",
    "set_array_api_enabled",
    "trace_array_namespace",
    "xp_as_array",
    "xp_as_float_array",
    "xp_as_int_array",
    "xp_ascontiguousarray",
    "xp_assert_close",
    "xp_assert_equal",
    "xp_astype",
    "xp_atleast_1d",
    "xp_atleast_2d",
    "xp_average",
    "xp_broadcast_to",
    "xp_create_diagonal",
    "xp_degrees",
    "xp_eig",
    "xp_eigh",
    "xp_gradient",
    "xp_insert",
    "xp_interp",
    "xp_isclose",
    "xp_isin",
    "xp_linspace",
    "xp_lstsq",
    "xp_matrix_transpose",
    "xp_median",
    "xp_nan_to_num",
    "xp_nanmean",
    "xp_pad",
    "xp_radians",
    "xp_reshape",
    "xp_resize",
    "xp_round",
    "xp_select",
    "xp_setxor1d",
    "xp_sinc",
    "xp_squeeze",
    "xp_trapezoid",
    "xp_unique",
]
__all__ += [
    "metric_mse",
    "metric_psnr",
]
__all__ += [
    "hash_sha256",
    "download_url",
]
__all__ += [
    "ControlFlowNode",
    "ExecutionNode",
    "ExecutionPort",
    "For",
    "NodeLog",
    "NodePassthrough",
    "NodeSetGraphOutputPort",
    "NodeSleep",
    "notify_process_state",
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
    "ObjectRenamed": [
        # v0.4.8
        [
            "colour.utilities.set_caching_enable",
            "colour.utilities.set_caching_enabled",
        ],
        [
            "colour.utilities.set_ndarray_copy_enable",
            "colour.utilities.set_ndarray_copy_enabled",
        ],
    ],
    "ObjectRemoved": [
        "colour.utilities.is_string",
        # v0.4.8
        "colour.utilities.disable_multiprocessing",
        "colour.utilities.multiprocessing_pool",
    ],
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
