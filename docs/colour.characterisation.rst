Colour Characterisation
=======================

ACES Spectral Conversion
~~~~~~~~~~~~~~~~~~~~~~~~

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    sd_to_ACES2065_1
    sd_to_aces_relative_exposure_values

**Ancillary Objects**

``colour.characterisation``

.. currentmodule:: colour.characterisation

.. autosummary::
    :toctree: generated/

    MSDS_ACES_RICD

ACES Input Transform Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    camera_RGB_to_ACES2065_1
    matrix_idt

**Ancillary Objects**

``colour.characterisation``

.. currentmodule:: colour.characterisation

.. autosummary::
    :toctree: generated/

    best_illuminant
    generate_illuminants_rawtoaces_v1
    normalise_illuminant
    optimisation_factory_Jzazbz
    optimisation_factory_Oklab_15
    optimisation_factory_rawtoaces_v1
    read_training_data_rawtoaces_v1
    training_data_sds_to_RGB
    training_data_sds_to_XYZ
    white_balance_multipliers
    whitepoint_preserving_matrix

Colour Fitting
--------------

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    apply_matrix_colour_correction
    APPLY_MATRIX_COLOUR_CORRECTION_METHODS
    colour_correction
    COLOUR_CORRECTION_METHODS
    matrix_colour_correction
    MATRIX_COLOUR_CORRECTION_METHODS
    polynomial_expansion
    POLYNOMIAL_EXPANSION_METHODS

**Ancillary Objects**

``colour.characterisation``

.. currentmodule:: colour.characterisation

.. autosummary::
    :toctree: generated/

    apply_matrix_colour_correction_Cheung2004
    apply_matrix_colour_correction_Finlayson2015
    apply_matrix_colour_correction_Vandermonde
    colour_correction_Cheung2004
    colour_correction_Finlayson2015
    colour_correction_Vandermonde
    matrix_augmented_Cheung2004
    matrix_colour_correction_Cheung2004
    matrix_colour_correction_Finlayson2015
    matrix_colour_correction_Vandermonde
    polynomial_expansion_Finlayson2015
    polynomial_expansion_Vandermonde

Colour Rendition Charts
-----------------------

**Dataset**

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    CCS_COLOURCHECKERS
    SDS_COLOURCHECKERS

**Ancillary Objects**

``colour.characterisation``

.. currentmodule:: colour.characterisation

.. autosummary::
    :toctree: generated/

    ColourChecker

Cameras
-------

``colour.characterisation``

.. currentmodule:: colour.characterisation

.. autosummary::
    :toctree: generated/
    :template: class.rst

    RGB_CameraSensitivities

**Dataset**

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    MSDS_CAMERA_SENSITIVITIES

Displays
--------

``colour.characterisation``

.. currentmodule:: colour.characterisation

.. autosummary::
    :toctree: generated/
    :template: class.rst

    RGB_DisplayPrimaries

**Dataset**

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    MSDS_DISPLAY_PRIMARIES

Filters
-------

**Dataset**

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    SDS_FILTERS

Lenses
------

**Dataset**

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    SDS_LENSES
