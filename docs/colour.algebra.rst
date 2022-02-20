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

Geometry
--------

``colour.algebra``

.. currentmodule:: colour.algebra

.. autosummary::
    :toctree: generated/

    normalise_vector
    euclidean_distance
    manhattan_distance
    extend_line_segment
    intersect_line_segments
    ellipse_coefficients_general_form
    ellipse_coefficients_canonical_form
    point_at_angle_on_ellipse
    ELLIPSE_FITTING_METHODS
    ellipse_fitting

**Ancillary Objects**

``colour.algebra``

.. autosummary::
    :toctree: generated/

    LineSegmentsIntersections_Specification
    ellipse_fitting_Halir1998

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

    is_spow_enabled
    set_spow_enable
    spow_enable
    spow
    normalise_maximum
    vector_dot
    matrix_dot
    linear_conversion
    linstep_function
    lerp
    smoothstep_function
    smooth
    is_identity
