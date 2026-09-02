Utilities
=========

Callback Management
-------------------

``colour``


``colour.utilities``

.. currentmodule:: colour.utilities

.. autosummary::
    :toctree: generated/
    :template: class.rst

    Callback
    MixinCallback

Common
------

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    domain_range_scale
    get_domain_range_scale
    get_domain_range_scale_metadata
    set_domain_range_scale


``colour.utilities``

.. currentmodule:: colour.utilities

.. autosummary::
    :toctree: generated/
    :template: class.rst

    CacheRegistry
    caching_enable

.. currentmodule:: colour.utilities

.. autosummary::
    :toctree: generated/

    attest
    batch
    CACHE_REGISTRY
    copy_definition
    filter_kwargs
    filter_mapping
    first_item
    handle_numpy_errors
    ignore_numpy_errors
    ignore_python_warnings
    int_digest
    is_caching_enabled
    is_integer
    is_iterable
    is_numeric
    is_sibling
    optional
    print_numpy_errors
    raise_numpy_errors
    set_caching_enabled
    slugify
    download_url
    hash_sha256
    validate_method
    warn_numpy_errors

Array
-----

``colour.utilities``

.. currentmodule:: colour.utilities

.. autosummary::
    :toctree: generated/
    :template: class.rst

    MixinDataclassArithmetic
    MixinDataclassArray
    MixinDataclassFields
    MixinDataclassIterable
    ndarray_copy_enable

.. autosummary::
    :toctree: generated/

    as_array
    as_complex_array
    as_float
    as_float_array
    as_float_scalar
    as_int
    as_int_array
    as_int_scalar
    centroid
    closest
    closest_indexes
    fill_nan
    format_array_as_row
    from_range_1
    from_range_10
    from_range_100
    from_range_degrees
    from_range_int
    full
    has_only_nan
    in_array
    index_along_last_axis
    interval
    is_ndarray_copy_enabled
    is_uniform
    ndarray_copy
    ndarray_write
    ones
    orient
    row_as_diagonal
    set_default_complex_dtype
    set_default_float_dtype
    set_default_int_dtype
    set_ndarray_copy_enabled
    to_domain_1
    to_domain_10
    to_domain_100
    to_domain_degrees
    to_domain_int
    tsplit
    tstack
    zeros

Array API
---------

``colour.utilities``

.. currentmodule:: colour.utilities

**Context Managers**

.. autosummary::
    :toctree: generated/
    :template: class.rst

    array_api_enable
    trace_array_namespace

**Namespace and Dispatch**

.. autosummary::
    :toctree: generated/

    array_namespace
    is_array_api_enabled
    is_non_ndarray
    is_numpy_namespace
    set_array_api_enabled

**Boundary Conversion**

.. autosummary::
    :toctree: generated/

    as_ndarray
    cast_non_ndarray
    xp_as_array
    xp_as_float_array
    xp_as_int_array

**Assertion Helpers**

.. autosummary::
    :toctree: generated/

    xp_assert_close
    xp_assert_equal

**Array Operations**

.. autosummary::
    :toctree: generated/

    xp_ascontiguousarray
    xp_astype
    xp_atleast_1d
    xp_atleast_2d
    xp_average
    xp_broadcast_to
    xp_create_diagonal
    xp_degrees
    xp_eig
    xp_eigh
    xp_gradient
    xp_insert
    xp_interp
    xp_isclose
    xp_isin
    xp_linspace
    xp_lstsq
    xp_matrix_transpose
    xp_median
    xp_nan_to_num
    xp_nanmean
    xp_pad
    xp_radians
    xp_reshape
    xp_resize
    xp_round
    xp_select
    xp_setxor1d
    xp_sinc
    xp_squeeze
    xp_trapezoid
    xp_unique

Data Structures
---------------

``colour.utilities``

.. currentmodule:: colour.utilities

.. autosummary::
    :toctree: generated/
    :template: class.rst

    CanonicalMapping
    LazyCanonicalMapping
    Lookup
    OrderedSet
    Structure


Delegate - Event Notifications
------------------------------

``colour.utilities``

.. currentmodule:: colour.utilities

.. autosummary::
    :toctree: generated/
    :template: class.rst

    Delegate

Network
-------

``colour.utilities``

.. currentmodule:: colour.utilities

.. autosummary::
    :toctree: generated/
    :template: class.rst

    Port
    PortGraph
    PortNode
    TreeNode
    ExecutionPort
    ExecutionNode
    ControlFlowNode
    For
    ThreadPoolExecutorManager
    ParallelForThread
    ProcessPoolExecutorManager
    ParallelForMultiprocess
    NodePassthrough
    NodeLog
    NodeSleep
    NodeSetGraphOutputPort

.. currentmodule:: colour.utilities

.. autosummary::
    :toctree: generated/

    notify_process_state

Metrics
-------

``colour.utilities``

.. currentmodule:: colour.utilities

.. autosummary::
    :toctree: generated/

    metric_mse
    metric_psnr

Requirements
------------

``colour.utilities``

.. currentmodule:: colour.utilities

.. autosummary::
    :toctree: generated/

    is_array_api_compat_installed
    is_array_api_extra_installed
    is_ctlrender_installed
    is_imageio_installed
    is_matplotlib_installed
    is_networkx_installed
    is_opencolorio_installed
    is_openimageio_installed
    is_pandas_installed
    is_pydot_installed
    is_tqdm_installed
    is_trimesh_installed
    is_xxhash_installed
    required

Verbose
-------

``colour.utilities``

.. currentmodule:: colour.utilities

.. autosummary::
    :toctree: generated/

    as_bool
    describe_environment
    filter_warnings
    message_box
    multiline_repr
    multiline_str
    numpy_print_options
    show_warning
    suppress_stdout
    suppress_warnings
    warning

**Ancillary Objects**

``colour.utilities``

.. currentmodule:: colour.utilities

.. autosummary::
    :toctree: generated/
    :template: class.rst

    ColourRuntimeWarning
    ColourUsageWarning
    ColourWarning
    MixinLogging
