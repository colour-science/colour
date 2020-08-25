Colour Characterisation
=======================

.. contents:: :local:

ACES Spectral Conversion
~~~~~~~~~~~~~~~~~~~~~~~~

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    sd_to_aces_relative_exposure_values

**Ancillary Objects**

``colour.characterisation``

.. currentmodule:: colour.characterisation

.. autosummary::
    :toctree: generated/

    ACES_RICD

ACES Input Transform Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    idt_matrix

**Ancillary Objects**

``colour.characterisation``

.. currentmodule:: colour.characterisation

.. autosummary::
    :toctree: generated/

    read_training_data_rawtoaces_v1
    generate_illuminants_rawtoaces_v1
    white_balance_multipliers
    normalise_illuminant
    training_data_sds_to_RGB
    training_data_sds_to_XYZ
    best_illuminant
    optimisation_factory_rawtoaces_v1
    optimisation_factory_JzAzBz

Colour Fitting
--------------

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    POLYNOMIAL_EXPANSION_METHODS
    polynomial_expansion
    COLOUR_CORRECTION_MATRIX_METHODS
    colour_correction_matrix
    COLOUR_CORRECTION_METHODS
    colour_correction

**Ancillary Objects**

``colour.characterisation``

.. currentmodule:: colour.characterisation

.. autosummary::
    :toctree: generated/

    augmented_matrix_Cheung2004
    polynomial_expansion_Finlayson2015
    polynomial_expansion_Vandermonde
    colour_correction_matrix_Cheung2004
    colour_correction_matrix_Finlayson2015
    colour_correction_matrix_Vandermonde
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

    COLOURCHECKERS
    COLOURCHECKER_SDS

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

    RGB_SpectralSensitivities

**Dataset**

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    CAMERA_RGB_SPECTRAL_SENSITIVITIES

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

    DISPLAY_RGB_PRIMARIES

Filters
-------

**Dataset**

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    FILTER_SDS

Lenses
------

**Dataset**

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    LENS_SDS
