Colorimetry
===========

.. contents:: :local:

Spectral Data Structure
-----------------------

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    SpectralPowerDistribution
    MultiSpectralPowerDistribution
    SpectralShape
    DEFAULT_SPECTRAL_SHAPE
    ASTME30815_PRACTISE_SHAPE

Spectral Data Generation
------------------------

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    blackbody_spd
    CIE_standard_illuminant_A_function
    D_illuminant_relative_spd
    constant_spd
    ones_spd
    zeros_spd

``colour.colorimetry``

.. currentmodule:: colour.colorimetry

.. autosummary::
    :toctree: generated/

    blackbody_spectral_radiance
    planck_law

Conversion to Tristimulus Values
--------------------------------

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    spectral_to_XYZ
    SPECTRAL_TO_XYZ_METHODS
    wavelength_to_XYZ

ASTM E308-15
~~~~~~~~~~~~

``colour.colorimetry``

.. currentmodule:: colour.colorimetry

.. autosummary::
    :toctree: generated/

    spectral_to_XYZ_ASTME30815

**Ancillary Objects**

``colour.colorimetry``

.. currentmodule:: colour.colorimetry

.. autosummary::
    :toctree: generated/

    spectral_to_XYZ_tristimulus_weighting_factors_ASTME30815
    adjust_tristimulus_weighting_factors_ASTME30815
    lagrange_coefficients_ASTME202211
    tristimulus_weighting_factors_ASTME202211

Integration
~~~~~~~~~~~

``colour.colorimetry``

.. currentmodule:: colour.colorimetry

.. autosummary::
    :toctree: generated/

    spectral_to_XYZ_integration

Spectral Bandpass Dependence Correction
---------------------------------------

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    bandpass_correction
    BANDPASS_CORRECTION_METHODS

Stearns and Stearns (1988)
~~~~~~~~~~~~~~~~~~~~~~~~~~

``colour.colorimetry``

.. currentmodule:: colour.colorimetry

.. autosummary::
    :toctree: generated/

    bandpass_correction_Stearns1988

Colour Matching Functions
-------------------------

``colour.colorimetry``

.. currentmodule:: colour.colorimetry

.. autosummary::
    :toctree: generated/

    LMS_ConeFundamentals
    RGB_ColourMatchingFunctions
    XYZ_ColourMatchingFunctions

**Dataset**

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    CMFS
    LMS_CMFS
    RGB_CMFS
    STANDARD_OBSERVERS_CMFS

Colour Matching Functions Transformations
-----------------------------------------
**Ancillary Objects**

``colour.colorimetry``

.. currentmodule:: colour.colorimetry

.. autosummary::
    :toctree: generated/

    RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs
    RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs
    RGB_10_degree_cmfs_to_LMS_10_degree_cmfs
    LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs
    LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs

Illuminants and Light Sources
-----------------------------

**Dataset**

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    ILLUMINANTS
    ILLUMINANTS_RELATIVE_SPDS
    HUNTERLAB_ILLUMINANTS
    LIGHT_SOURCES
    LIGHT_SOURCES_RELATIVE_SPDS

Dominant Wavelength and Purity
------------------------------

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    dominant_wavelength
    complementary_wavelength
    excitation_purity
    colorimetric_purity

Luminous Efficiency Functions
-----------------------------

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    luminous_efficacy
    luminous_efficiency
    luminous_flux
    mesopic_luminous_efficiency_function

**Dataset**

``colour``

.. autosummary::
    :toctree: generated/

    LEFS
    PHOTOPIC_LEFS
    SCOTOPIC_LEFS

Lightness Computation
---------------------

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    lightness
    LIGHTNESS_METHODS

Glasser, Mckinney, Reilly and Schnelle (1958)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``colour.colorimetry``

.. currentmodule:: colour.colorimetry

.. autosummary::
    :toctree: generated/

    lightness_Glasser1958

Wyszecki (1963)
~~~~~~~~~~~~~~~

``colour.colorimetry``

.. currentmodule:: colour.colorimetry

.. autosummary::
    :toctree: generated/

    lightness_Wyszecki1963

CIE 1976
~~~~~~~~

``colour.colorimetry``

.. currentmodule:: colour.colorimetry

.. autosummary::
    :toctree: generated/

    lightness_CIE1976

Fairchild and Wyble (2010)
~~~~~~~~~~~~~~~~~~~~~~~~~~

``colour.colorimetry``

.. currentmodule:: colour.colorimetry

.. autosummary::
    :toctree: generated/

    lightness_Fairchild2010

Fairchild and Chen (2011)
~~~~~~~~~~~~~~~~~~~~~~~~~

``colour.colorimetry``

.. currentmodule:: colour.colorimetry

.. autosummary::
    :toctree: generated/

    lightness_Fairchild2011

Luminance Computation
---------------------

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    luminance
    LUMINANCE_METHODS

Newhall, Nickerson and Judd (1943)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``colour.colorimetry``

.. currentmodule:: colour.colorimetry

.. autosummary::
    :toctree: generated/

    luminance_Newhall1943

CIE 1976
~~~~~~~~

``colour.colorimetry``

.. currentmodule:: colour.colorimetry

.. autosummary::
    :toctree: generated/

    luminance_CIE1976

ASTM D1535-08e1
~~~~~~~~~~~~~~~

``colour.colorimetry``

.. currentmodule:: colour.colorimetry

.. autosummary::
    :toctree: generated/

    luminance_ASTMD153508

Fairchild and Wyble (2010)
~~~~~~~~~~~~~~~~~~~~~~~~~~

``colour.colorimetry``

.. currentmodule:: colour.colorimetry

.. autosummary::
    :toctree: generated/

    luminance_Fairchild2010

Fairchild and Chen (2011)
~~~~~~~~~~~~~~~~~~~~~~~~~

``colour.colorimetry``

.. currentmodule:: colour.colorimetry

.. autosummary::
    :toctree: generated/

    luminance_Fairchild2011

Whiteness Computation
---------------------

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    whiteness
    WHITENESS_METHODS

Berger (1959)
~~~~~~~~~~~~~

``colour.colorimetry``

.. currentmodule:: colour.colorimetry

.. autosummary::
    :toctree: generated/

    whiteness_Berger1959

Taube (1960)
~~~~~~~~~~~~

``colour.colorimetry``

.. currentmodule:: colour.colorimetry

.. autosummary::
    :toctree: generated/

    whiteness_Taube1960

Stensby (1968)
~~~~~~~~~~~~~~

``colour.colorimetry``

.. currentmodule:: colour.colorimetry

.. autosummary::
    :toctree: generated/

    whiteness_Stensby1968

ASTM E313
~~~~~~~~~

``colour.colorimetry``

.. currentmodule:: colour.colorimetry

.. autosummary::
    :toctree: generated/

    whiteness_ASTME313

Ganz and Griesser (1979)
~~~~~~~~~~~~~~~~~~~~~~~~

``colour.colorimetry``

.. currentmodule:: colour.colorimetry

.. autosummary::
    :toctree: generated/

    whiteness_Ganz1979

CIE 2004
~~~~~~~~

``colour.colorimetry``

.. currentmodule:: colour.colorimetry

.. autosummary::
    :toctree: generated/

    whiteness_CIE2004

Yellowness Computation
----------------------

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    yellowness
    YELLOWNESS_METHODS

ASTM D1925
~~~~~~~~~~

``colour.colorimetry``

.. currentmodule:: colour.colorimetry

.. autosummary::
    :toctree: generated/

    yellowness_ASTMD1925

ASTM E313
~~~~~~~~~

``colour.colorimetry``

.. currentmodule:: colour.colorimetry

.. autosummary::
    :toctree: generated/

    yellowness_ASTME313
