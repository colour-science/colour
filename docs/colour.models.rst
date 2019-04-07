Colour Models
=============

.. contents:: :local:

Tristimulus Values, CIE xyY Colourspace and Chromaticity Coordinates
--------------------------------------------------------------------

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    XYZ_to_xyY
    xyY_to_XYZ
    XYZ_to_xy
    xy_to_XYZ
    xyY_to_xy
    xy_to_xyY

CIE L*a*b* Colourspace
----------------------

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    XYZ_to_Lab
    Lab_to_XYZ
    Lab_to_LCHab
    LCHab_to_Lab

CIE L*u*v* Colourspace
----------------------

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    XYZ_to_Luv
    Luv_to_XYZ
    Luv_to_LCHuv
    LCHuv_to_Luv
    Luv_to_uv
    uv_to_Luv
    Luv_uv_to_xy
    xy_to_Luv_uv

CIE 1960 UCS Colourspace
------------------------

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    XYZ_to_UCS
    UCS_to_XYZ
    UCS_to_uv
    uv_to_UCS
    UCS_uv_to_xy
    xy_to_UCS_uv

CIE 1964 U*V*W* Colourspace
---------------------------

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    XYZ_to_UVW
    UVW_to_XYZ

Hunter L,a,b Colour Scale
-------------------------

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    XYZ_to_Hunter_Lab
    Hunter_Lab_to_XYZ
    XYZ_to_K_ab_HunterLab1966

Hunter Rd,a,b Colour Scale
--------------------------

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    XYZ_to_Hunter_Rdab
    Hunter_Rdab_to_XYZ

DIN99 Colourspace
-----------------

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    Lab_to_DIN99
    DIN99_to_Lab

CAM02-LCD, CAM02-SCD, and CAM02-UCS Colourspaces - Luo, Cui and Li (2006)
-------------------------------------------------------------------------

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    JMh_CIECAM02_to_CAM02LCD
    CAM02LCD_to_JMh_CIECAM02
    JMh_CIECAM02_to_CAM02SCD
    CAM02SCD_to_JMh_CIECAM02
    JMh_CIECAM02_to_CAM02UCS
    CAM02UCS_to_JMh_CIECAM02

CAM16-LCD, CAM16-SCD, and CAM16-UCS Colourspaces - Li et al. (2017)
-------------------------------------------------------------------

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    JMh_CAM16_to_CAM16LCD
    CAM16LCD_to_JMh_CAM16
    JMh_CAM16_to_CAM16SCD
    CAM16SCD_to_JMh_CAM16
    JMh_CAM16_to_CAM16UCS
    CAM16UCS_to_JMh_CAM16

IPT Colourspace
---------------

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    XYZ_to_IPT
    IPT_to_XYZ
    IPT_hue_angle

hdr-CIELAB Colourspace
----------------------

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    XYZ_to_hdr_CIELab
    hdr_CIELab_to_XYZ
    HDR_CIELAB_METHODS

hdr-IPT Colourspace
-------------------

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    XYZ_to_hdr_IPT
    hdr_IPT_to_XYZ
    HDR_IPT_METHODS

OSA UCS Colourspace
-------------------

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    XYZ_to_OSA_UCS
    OSA_UCS_to_XYZ

:math:`JzAzBz` Colourspace
--------------------------

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    XYZ_to_JzAzBz
    JzAzBz_to_XYZ

RGB Colourspace and Transformations
-----------------------------------

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    XYZ_to_RGB
    RGB_to_XYZ
    RGB_to_RGB
    RGB_to_RGB_matrix

**Ancillary Objects**

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    XYZ_to_sRGB
    sRGB_to_XYZ

RGB Colourspace Derivation
~~~~~~~~~~~~~~~~~~~~~~~~~~
``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    normalised_primary_matrix
    chromatically_adapted_primaries
    primaries_whitepoint
    RGB_luminance
    RGB_luminance_equation

RGB Colourspaces
~~~~~~~~~~~~~~~~

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    RGB_Colourspace
    RGB_COLOURSPACES

``colour.models``

.. currentmodule:: colour.models

.. autosummary::
    :toctree: generated/

    ACES_2065_1_COLOURSPACE
    ACES_CC_COLOURSPACE
    ACES_CCT_COLOURSPACE
    ACES_PROXY_COLOURSPACE
    ACES_CG_COLOURSPACE
    ADOBE_RGB_1998_COLOURSPACE
    ADOBE_WIDE_GAMUT_RGB_COLOURSPACE
    ALEXA_WIDE_GAMUT_COLOURSPACE
    APPLE_RGB_COLOURSPACE
    BEST_RGB_COLOURSPACE
    BETA_RGB_COLOURSPACE
    BT470_525_COLOURSPACE
    BT470_625_COLOURSPACE
    BT709_COLOURSPACE
    BT2020_COLOURSPACE
    CIE_RGB_COLOURSPACE
    CINEMA_GAMUT_COLOURSPACE
    COLOR_MATCH_RGB_COLOURSPACE
    DCDM_XYZ_COLOURSPACE
    DCI_P3_COLOURSPACE
    DCI_P3_P_COLOURSPACE
    DON_RGB_4_COLOURSPACE
    ECI_RGB_V2_COLOURSPACE
    EKTA_SPACE_PS_5_COLOURSPACE
    PROTUNE_NATIVE_COLOURSPACE
    MAX_RGB_COLOURSPACE
    NTSC_COLOURSPACE
    P3_D65_COLOURSPACE
    PAL_SECAM_COLOURSPACE
    RED_COLOR_COLOURSPACE
    RED_COLOR_2_COLOURSPACE
    RED_COLOR_3_COLOURSPACE
    RED_COLOR_4_COLOURSPACE
    RED_WIDE_GAMUT_RGB_COLOURSPACE
    DRAGON_COLOR_COLOURSPACE
    DRAGON_COLOR_2_COLOURSPACE
    ROMM_RGB_COLOURSPACE
    RIMM_RGB_COLOURSPACE
    ERIMM_RGB_COLOURSPACE
    PROPHOTO_RGB_COLOURSPACE
    RUSSELL_RGB_COLOURSPACE
    SMPTE_240M_COLOURSPACE
    S_GAMUT_COLOURSPACE
    S_GAMUT3_COLOURSPACE
    S_GAMUT3_CINE_COLOURSPACE
    sRGB_COLOURSPACE
    V_GAMUT_COLOURSPACE
    XTREME_RGB_COLOURSPACE

Colour Component Transfer Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    encoding_cctf
    ENCODING_CCTFS
    decoding_cctf
    DECODING_CCTFS

Opto-Electronic Transfer Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    oetf
    OETFS
    oetf_reverse
    OETFS_REVERSE

``colour.models``

.. currentmodule:: colour.models

.. autosummary::
    :toctree: generated/

    oetf_ARIBSTDB67
    oetf_reverse_ARIBSTDB67
    oetf_DICOMGSDF
    oetf_BT2020
    oetf_BT2100_HLG
    oetf_reverse_BT2100_HLG
    oetf_BT2100_PQ
    oetf_reverse_BT2100_PQ
    oetf_BT601
    oetf_reverse_BT601
    oetf_BT709
    oetf_reverse_BT709
    oetf_ProPhotoRGB
    oetf_RIMMRGB
    oetf_ROMMRGB
    oetf_SMPTE240M
    oetf_ST2084
    oetf_sRGB
    oetf_reverse_sRGB

**Ancillary Objects**

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    gamma_function
    linear_function

Electro-Optical Transfer Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    eotf
    EOTFS
    eotf_reverse
    EOTFS_REVERSE

``colour.models``

.. currentmodule:: colour.models

.. autosummary::
    :toctree: generated/

    eotf_DCDM
    eotf_reverse_DCDM
    eotf_DICOMGSDF
    eotf_BT1886
    eotf_reverse_BT1886
    eotf_BT2020
    eotf_BT2100_HLG
    eotf_reverse_BT2100_HLG
    eotf_BT2100_PQ
    eotf_reverse_BT2100_PQ
    eotf_ProPhotoRGB
    eotf_RIMMRGB
    eotf_ROMMRGB
    eotf_SMPTE240M
    eotf_ST2084

Opto-Optical Transfer Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    ootf
    OOTFS
    ootf_reverse
    OOTFS_REVERSE


``colour.models``

.. currentmodule:: colour.models

.. autosummary::
    :toctree: generated/

    ootf_BT2100_HLG
    ootf_reverse_BT2100_HLG
    ootf_BT2100_PQ
    ootf_reverse_BT2100_PQ

Log Encoding and Decoding Curves
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    log_encoding_curve
    LOG_ENCODING_CURVES
    log_decoding_curve
    LOG_DECODING_CURVES

``colour.models``

.. currentmodule:: colour.models

.. autosummary::
    :toctree: generated/

    log_encoding_ACEScc
    log_decoding_ACEScc
    log_encoding_ACEScct
    log_decoding_ACEScct
    log_encoding_ACESproxy
    log_decoding_ACESproxy
    log_encoding_ALEXALogC
    log_decoding_ALEXALogC
    log_encoding_CanonLog2
    log_decoding_CanonLog2
    log_encoding_CanonLog3
    log_decoding_CanonLog3
    log_encoding_CanonLog
    log_decoding_CanonLog
    log_encoding_Cineon
    log_decoding_Cineon
    log_encoding_ERIMMRGB
    log_decoding_ERIMMRGB
    log_encoding_Log3G10
    log_decoding_Log3G10
    log_encoding_Log3G12
    log_decoding_Log3G12
    log_encoding_Panalog
    log_decoding_Panalog
    log_encoding_PivotedLog
    log_decoding_PivotedLog
    log_encoding_Protune
    log_decoding_Protune
    log_encoding_REDLog
    log_decoding_REDLog
    log_encoding_REDLogFilm
    log_decoding_REDLogFilm
    log_encoding_SLog
    log_decoding_SLog
    log_encoding_SLog2
    log_decoding_SLog2
    log_encoding_SLog3
    log_decoding_SLog3
    log_encoding_VLog
    log_decoding_VLog
    log_encoding_ViperLog
    log_decoding_ViperLog

ACES Spectral Conversion
~~~~~~~~~~~~~~~~~~~~~~~~

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    sd_to_aces_relative_exposure_values

**Ancillary Objects**

``colour.models``

.. currentmodule:: colour.models

.. autosummary::
    :toctree: generated/

    ACES_RICD

Colour Encodings
~~~~~~~~~~~~~~~~

Y'CbCr Colour Encoding
^^^^^^^^^^^^^^^^^^^^^^

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    RGB_to_YCbCr
    YCbCr_to_RGB
    YCBCR_WEIGHTS
    RGB_to_YcCbcCrc
    YcCbcCrc_to_RGB

**Ancillary Objects**

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    full_to_legal
    legal_to_full
    CV_range

YCoCg Colour Encoding
^^^^^^^^^^^^^^^^^^^^^

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    RGB_to_YCoCg
    YCoCg_to_RGB

:math:`IC_TC_P` Colour Encoding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    RGB_to_ICTCP
    ICTCP_to_RGB

RGB Representations
~~~~~~~~~~~~~~~~~~~

Prismatic Colourspace
^^^^^^^^^^^^^^^^^^^^^

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    RGB_to_Prismatic
    Prismatic_to_RGB

HSV Colourspace
^^^^^^^^^^^^^^^

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    RGB_to_HSV
    HSV_to_RGB

HSL Colourspace
^^^^^^^^^^^^^^^

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    RGB_to_HSL
    HSL_to_RGB

CMY Colourspace
^^^^^^^^^^^^^^^

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    RGB_to_CMY
    CMY_to_RGB
    CMY_to_CMYK
    CMYK_to_CMY

Pointer's Gamut
---------------

``colour``

.. currentmodule:: colour

.. autosummary::
    :toctree: generated/

    POINTER_GAMUT_BOUNDARIES
    POINTER_GAMUT_DATA
    POINTER_GAMUT_ILLUMINANT
