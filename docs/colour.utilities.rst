Utilities
=========

.. contents:: :local:

Common
------

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    domain_range_scale
    get_domain_range_scale
    set_domain_range_scale


``colour.utilities``

.. currentmodule:: colour.utilities

.. autosummary::
    :toctree: generated/

    handle_numpy_errors
    ignore_numpy_errors
    raise_numpy_errors
    print_numpy_errors
    warn_numpy_errors
    ignore_python_warnings
    batch
    disable_multiprocessing
    multiprocessing_pool
    is_openimageio_installed
    is_pandas_installed
    is_iterable
    is_string
    is_numeric
    is_integer
    is_sibling
    filter_kwargs
    filter_mapping
    first_item
    to_domain_1
    to_domain_10
    to_domain_100
    to_domain_degrees
    to_domain_int
    from_range_1
    from_range_10
    from_range_100
    from_range_degrees
    from_range_int

Array
-----

``colour.utilities``

.. currentmodule:: colour.utilities

.. autosummary::
    :toctree: generated/

    as_array
    as_int_array
    as_float_array
    as_numeric
    as_int
    as_float
    as_namedtuple
    closest_indexes
    closest
    normalise_maximum
    interval
    is_uniform
    in_array
    tstack
    tsplit
    row_as_diagonal
    dot_vector
    dot_matrix
    orient
    centroid
    linear_conversion
    lerp
    fill_nan
    ndarray_write

Metrics
-------

``colour.utilities``

.. currentmodule:: colour.utilities

.. autosummary::
    :toctree: generated/

    metric_mse
    metric_psnr

Data Structures
---------------

``colour.utilities``

.. currentmodule:: colour.utilities

.. autosummary::
    :toctree: generated/

    CaseInsensitiveMapping
    Lookup
    Structure

Verbose
-------

``colour.utilities``

.. currentmodule:: colour.utilities

.. autosummary::
    :toctree: generated/

    message_box
    warning
    filter_warnings
    suppress_warnings
    numpy_print_options
    describe_environment

**Ancillary Objects**

``colour.utilities``

.. currentmodule:: colour.utilities

.. autosummary::
    :toctree: generated/

    ColourWarning
    ColourUsageWarning
    ColourRuntimeWarning