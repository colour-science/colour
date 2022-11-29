Colorimetry
===========

Spectral Data Structure
-----------------------

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/
    :template: class.rst

    SpectralShape
    SpectralDistribution
    MultiSpectralDistributions

.. autosummary::
    :toctree: generated/

    SPECTRAL_SHAPE_ASTME308
    SPECTRAL_SHAPE_DEFAULT

**Ancillary Objects**

``colour.colorimetry``

.. currentmodule:: colour.colorimetry

.. autosummary::
    :toctree: generated/

    reshape_sd
    reshape_msds
    sds_and_msds_to_sds
    sds_and_msds_to_msds

Spectral Data Generation
------------------------

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    sd_CIE_standard_illuminant_A
    sd_CIE_illuminant_D_series
    sd_blackbody
    sd_rayleigh_jeans
    sd_constant
    sd_ones
    sd_zeros
    msds_constant
    msds_ones
    msds_zeros
    SD_GAUSSIAN_METHODS
    sd_gaussian
    SD_SINGLE_LED_METHODS
    sd_single_led
    SD_MULTI_LEDS_METHODS
    sd_multi_leds

``colour.colorimetry``

.. currentmodule:: colour.colorimetry

.. autosummary::
    :toctree: generated/

    blackbody_spectral_radiance
    planck_law
    rayleigh_jeans_law
    daylight_locus_function
    sd_gaussian_normal
    sd_gaussian_fwhm
    sd_single_led_Ohno2005
    sd_multi_leds_Ohno2005


**Aliases**

``colour.colorimetry``

.. currentmodule:: colour.colorimetry

.. autosummary::
    :toctree: generated/

    planck_law

Conversion to Tristimulus Values
--------------------------------

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    sd_to_XYZ
    SD_TO_XYZ_METHODS
    msds_to_XYZ
    MSDS_TO_XYZ_METHODS
    wavelength_to_XYZ

**Ancillary Objects**

``colour.colorimetry``

.. currentmodule:: colour.colorimetry

.. autosummary::
    :toctree: generated/

    handle_spectral_arguments

ASTM E308-15
~~~~~~~~~~~~

``colour.colorimetry``

.. currentmodule:: colour.colorimetry

.. autosummary::
    :toctree: generated/

    sd_to_XYZ_ASTME308
    msds_to_XYZ_ASTME308

**Ancillary Objects**

``colour.colorimetry``

.. currentmodule:: colour.colorimetry

.. autosummary::
    :toctree: generated/

    sd_to_XYZ_tristimulus_weighting_factors_ASTME308
    adjust_tristimulus_weighting_factors_ASTME308
    lagrange_coefficients_ASTME2022
    tristimulus_weighting_factors_ASTME2022

Integration
~~~~~~~~~~~

``colour.colorimetry``

.. currentmodule:: colour.colorimetry

.. autosummary::
    :toctree: generated/

    sd_to_XYZ_integration
    msds_to_XYZ_integration

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
    :template: class.rst

    LMS_ConeFundamentals
    RGB_ColourMatchingFunctions
    XYZ_ColourMatchingFunctions

**Dataset**

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    MSDS_CMFS


**Ancillary Objects**

``colour.colorimetry``

.. currentmodule:: colour.colorimetry

.. autosummary::
    :toctree: generated/

    MSDS_CMFS_LMS
    MSDS_CMFS_RGB
    MSDS_CMFS_STANDARD_OBSERVER

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

    CCS_ILLUMINANTS
    SDS_ILLUMINANTS
    CCS_LIGHT_SOURCES
    SDS_LIGHT_SOURCES
    TVS_ILLUMINANTS
    TVS_ILLUMINANTS_HUNTERLAB

**Ancillary Objects**

``colour.colorimetry``

.. currentmodule:: colour.colorimetry

.. autosummary::
    :toctree: generated/

    SDS_BASIS_FUNCTIONS_CIE_ILLUMINANT_D_SERIES

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
    sd_mesopic_luminous_efficiency_function

**Dataset**

``colour``

.. autosummary::
    :toctree: generated/

    SDS_LEFS


**Ancillary Objects**

``colour.colorimetry``

.. currentmodule:: colour.colorimetry

.. autosummary::
    :toctree: generated/

    SDS_LEFS_PHOTOPIC
    SDS_LEFS_SCOTOPIC

Spectral Uniformity
-------------------

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    spectral_uniformity

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
    intermediate_lightness_function_CIE1976

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

Abebe, Pouli, Larabi and Reinhard (2017)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``colour.colorimetry``

.. currentmodule:: colour.colorimetry

.. autosummary::
    :toctree: generated/

    lightness_Abebe2017

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
    intermediate_luminance_function_CIE1976

ASTM D1535-08e1
~~~~~~~~~~~~~~~

``colour.colorimetry``

.. currentmodule:: colour.colorimetry

.. autosummary::
    :toctree: generated/

    luminance_ASTMD1535

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

    yellowness_ASTME313_alternative
    YELLOWNESS_COEFFICIENTS_ASTME313
    yellowness_ASTME313
