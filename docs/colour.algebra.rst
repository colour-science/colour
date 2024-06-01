Algebra
=======

Extrapolation
-------------

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/
    :template: class.rst

    Extrapolator

Interpolation
-------------

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/
    :template: class.rst

    KernelInterpolator
    NearestNeighbourInterpolator
    LinearInterpolator
    NullInterpolator
    PchipInterpolator
    SpragueInterpolator

.. autosummary::
    :toctree: generated/

    lagrange_coefficients
    TABLE_INTERPOLATION_METHODS
    table_interpolation

**Interpolation Kernels**

``colour``

.. autosummary::
    :toctree: generated/

    kernel_nearest_neighbour
    kernel_linear
    kernel_sinc
    kernel_lanczos
    kernel_cardinal_spline

**Ancillary Objects**

``colour.algebra``

.. currentmodule:: colour.algebra

.. autosummary::
    :toctree: generated/

    table_interpolation_trilinear
    table_interpolation_tetrahedral

Coordinates
-----------

``colour.algebra``

.. currentmodule:: colour.algebra

.. autosummary::
    :toctree: generated/

    cartesian_to_spherical
    spherical_to_cartesian
    cartesian_to_polar
    polar_to_cartesian
    cartesian_to_cylindrical
    cylindrical_to_cartesian

Random
------

``colour.algebra``

.. currentmodule:: colour.algebra

.. autosummary::
    :toctree: generated/

    random_triplet_generator

Regression
----------

``colour.algebra``

.. currentmodule:: colour.algebra

.. autosummary::
    :toctree: generated/

    least_square_mapping_MoorePenrose

Common
------

``colour.algebra``

.. currentmodule:: colour.algebra

.. autosummary::
    :toctree: generated/

    get_sdiv_mode
    set_sdiv_mode
    sdiv_mode
    sdiv
    is_spow_enabled
    set_spow_enable
    spow_enable
    spow
    normalise_vector
    normalise_maximum
    vecmul
    euclidean_distance
    manhattan_distance
    linear_conversion
    linstep_function
    lerp
    smoothstep_function
    smooth
    is_identity
    eigen_decomposition
