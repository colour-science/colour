Colour Characterisation
=======================

ACES Spectral Conversion
~~~~~~~~~~~~~~~~~~~~~~~~

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    sd_to_aces_relative_exposure_values
    sd_to_ACES2065_1

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

    matrix_idt
    camera_RGB_to_ACES2065_1

**Ancillary Objects**

``colour.characterisation``

.. currentmodule:: colour.characterisation

.. autosummary::
    :toctree: generated/

    read_training_data_rawtoaces_v1
    generate_illuminants_rawtoaces_v1
    white_balance_multipliers
    best_illuminant
    normalise_illuminant
    training_data_sds_to_RGB
    training_data_sds_to_XYZ
    whitepoint_preserving_matrix
    optimisation_factory_rawtoaces_v1
    optimisation_factory_Jzazbz
    optimisation_factory_Oklab_15

Colour Fitting
--------------

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    POLYNOMIAL_EXPANSION_METHODS
    polynomial_expansion
    MATRIX_COLOUR_CORRECTION_METHODS
    matrix_colour_correction
    APPLY_MATRIX_COLOUR_CORRECTION_METHODS
    apply_matrix_colour_correction
    COLOUR_CORRECTION_METHODS
    colour_correction

**Ancillary Objects**

``colour.characterisation``

.. currentmodule:: colour.characterisation

.. autosummary::
    :toctree: generated/

    matrix_augmented_Cheung2004
    polynomial_expansion_Finlayson2015
    polynomial_expansion_Vandermonde
    matrix_colour_correction_Cheung2004
    matrix_colour_correction_Finlayson2015
    matrix_colour_correction_Vandermonde
    apply_matrix_colour_correction_Cheung2004
    apply_matrix_colour_correction_Finlayson2015
    apply_matrix_colour_correction_Vandermonde
    colour_correction_Cheung2004
    colour_correction_Finlayson2015
    colour_correction_Vandermonde

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
