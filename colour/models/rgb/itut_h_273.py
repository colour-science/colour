"""
Recommendation ITU-T H.273 Code points for Video Signal Type Identification
===========================================================================

Defines a set of standard video signal colour primaries, transfer functions and
matrix coefficients used in deriving luma and chroma signals along with
related definitions:

-   :attr:`colour.COLOUR_PRIMARIES_ITUTH273`
-   :attr:`colour.TRANSFER_CHARACTERISTICS_ITUTH273`
-   :attr:`colour.MATRIX_COEFFICIENTS_ITUTH273`
-   :attr:`colour.models.describe_video_signal_colour_primaries`
-   :attr:`colour.models.describe_video_signal_transfer_characteristics`
-   :attr:`colour.models.describe_video_signal_matrix_coefficients`

These values were historically defined in
:cite:`InternationalOrganizationforStandardization2013` then
superseded and duplicated by other standards such as
:cite:`InternationalOrganizationforStandardization2020`,
:cite:`InternationalOrganizationforStandardization2021` and
:cite:`InternationalTelecommunicationUnion2021`. They are widely in use to
define colour-related properties in video encoding and decoding software
libraries, including *FFmpeg*.

References
----------
-   :cite:`EuropeanBroadcastingUnion1975` : European Broadcasting Union.
    (1975). EBU Tech 3213 - EBU Standard for Chromaticity Tolerances for Studio
    Monitors. https://tech.ebu.ch/docs/tech/tech3213.pdf
-   :cite:`FFmpegDevelopers2022` : FFmpeg Developers. (2022).
    FFmpeg::AVColorPrimaries. https://github.com/FFmpeg/FFmpeg/\
blob/c469c3c3b18fbacd6ee0165573034d2a0408b83f/libavutil/pixfmt.h#L478
-   :cite:`FFmpegDevelopers2022a` : FFmpeg Developers. (2022).
    FFmpeg::AVColorTransferCharacteristic. https://github.com/FFmpeg/FFmpeg/\
blob/c469c3c3b18fbacd6ee0165573034d2a0408b83f/libavutil/pixfmt.h#L503
-   :cite:`FFmpegDevelopers2022b` : FFmpeg Developers. (2022).
    FFmpeg::AVColorSpace. https://github.com/FFmpeg/FFmpeg/\
blob/c469c3c3b18fbacd6ee0165573034d2a0408b83f/libavutil/pixfmt.h#L532
-   :cite:`InternationalOrganizationforStandardization2013` : International
    Organization for Standardization. (2013). INTERNATIONAL STANDARD ISO/IEC
    23001-8 - Information technology - MPEG systems technologies - Part 8:
    Coding-independent code points.
-   :cite:`InternationalOrganizationforStandardization2020` : International
    Organization for Standardization. (2020). INTERNATIONAL STANDARD ISO/IEC
    14496-10 - Information technology - Coding of audio-visual objects - Part
    10: Advanced video coding.
-   :cite:`InternationalOrganizationforStandardization2021` : International
    Organization for Standardization. (2021). INTERNATIONAL STANDARD ISO/IEC
    23091-2 - Information technology - Coding- independent code points -
    Part 2: Video.
-   :cite:`InternationalTelecommunicationUnion2021` : International
    Telecommunication Union. (2021). Recommendation ITU-T H.273 -
    Coding-independent code points for video signal type identification.
    https://www.itu.int/rec/T-REC-H.273-202107-I/en
"""

import functools
import numpy as np
from dataclasses import dataclass
from enum import IntEnum, auto

from colour.models.rgb.datasets.dcdm_xyz import (
    PRIMARIES_DCDM_XYZ,
    WHITEPOINT_NAME_DCDM_XYZ,
    CCS_WHITEPOINT_DCDM_XYZ,
    MATRIX_DCDM_XYZ_TO_XYZ,
    MATRIX_XYZ_TO_DCDM_XYZ,
)
from colour.models.rgb.datasets.dci_p3 import (
    PRIMARIES_DCI_P3,
    WHITEPOINT_NAME_DCI_P3,
    CCS_WHITEPOINT_DCI_P3,
    MATRIX_DCI_P3_TO_XYZ,
    MATRIX_XYZ_TO_DCI_P3,
)
from colour.models.rgb.datasets.itur_bt_2020 import (
    PRIMARIES_BT2020,
    WHITEPOINT_NAME_BT2020,
    CCS_WHITEPOINT_BT2020,
    MATRIX_BT2020_TO_XYZ,
    MATRIX_XYZ_TO_BT2020,
)
from colour.models.rgb.datasets.itur_bt_470 import (
    PRIMARIES_BT470_525,
    CCS_WHITEPOINT_BT470_525,
    WHITEPOINT_NAME_BT470_525,
    MATRIX_BT470_525_TO_XYZ,
    MATRIX_XYZ_TO_BT470_525,
    PRIMARIES_BT470_625,
    CCS_WHITEPOINT_BT470_625,
    WHITEPOINT_NAME_BT470_625,
    MATRIX_BT470_625_TO_XYZ,
    MATRIX_XYZ_TO_BT470_625,
)
from colour.models.rgb.datasets.itur_bt_709 import (
    PRIMARIES_BT709,
    CCS_WHITEPOINT_BT709,
    WHITEPOINT_NAME_BT709,
    MATRIX_BT709_TO_XYZ,
    MATRIX_XYZ_TO_BT709,
)
from colour.models.rgb.datasets.itut_h_273 import (
    PRIMARIES_H273_GENERIC_FILM,
    WHITEPOINT_NAME_H273_GENERIC_FILM,
    CCS_WHITEPOINT_H273_GENERIC_FILM,
    MATRIX_H273_GENERIC_FILM_RGB_TO_XYZ,
    MATRIX_XYZ_TO_H273_GENERIC_FILM_RGB,
    PRIMARIES_H273_22_UNSPECIFIED,
    WHITEPOINT_NAME_H273_22_UNSPECIFIED,
    CCS_WHITEPOINT_H273_22_UNSPECIFIED,
    MATRIX_H273_22_UNSPECIFIED_RGB_TO_XYZ,
    MATRIX_XYZ_TO_H273_22_UNSPECIFIED_RGB,
)
from colour.models.rgb.datasets.p3_d65 import (
    PRIMARIES_P3_D65,
    WHITEPOINT_NAME_P3_D65,
    CCS_WHITEPOINT_P3_D65,
    MATRIX_P3_D65_TO_XYZ,
    MATRIX_XYZ_TO_P3_D65,
)
from colour.models.rgb.datasets.smpte_240m import (
    PRIMARIES_SMPTE_240M,
    WHITEPOINT_NAME_SMPTE_240M,
    CCS_WHITEPOINT_SMPTE_240M,
    MATRIX_SMPTE_240M_TO_XYZ,
    MATRIX_XYZ_TO_SMPTE_240M,
)
from colour.models.rgb.transfer_functions import (
    eotf_inverse_H273_ST428_1,
    eotf_inverse_ST2084,
    gamma_function,
    linear_function,
    oetf_BT1361,
    oetf_BT2020,
    oetf_BT2100_HLG,
    oetf_BT601,
    oetf_BT709,
    oetf_H273_IEC61966_2,
    oetf_H273_Log,
    oetf_H273_LogSqrt,
    oetf_SMPTE240M,
)
from colour.hints import Any, Callable, Dict, NDArrayFloat, Union
from colour.utilities import message_box, multiline_str
from colour.utilities.documentation import (
    DocstringDict,
    is_documentation_building,
)

__all__ = [
    "COLOUR_PRIMARIES_ITUTH273",
    "FFmpegConstantsColourPrimaries_ITUTH273",
    "TRANSFER_CHARACTERISTICS_ITUTH273",
    "FFmpegConstantsTransferCharacteristics_ITUTH273",
    "MATRIX_COEFFICIENTS_ITUTH273",
    "FFmpegConstantsMatrixCoefficients_ITUTH273",
    "CCS_WHITEPOINTS_ITUTH273",
    "WHITEPOINT_NAMES_ITUTH273",
    "MATRICES_ITUTH273_RGB_TO_XYZ",
    "MATRICES_XYZ_TO_ITUTH273_RGB",
    "COLOUR_PRIMARIES_ISO23091_2",
    "TRANSFER_CHARACTERISTICS_ISO23091_2",
    "MATRIX_COEFFICIENTS_ISO23091_2",
    "CCS_WHITEPOINTS_ISO23091_2",
    "WHITEPOINT_NAMES_ISO23091_2",
    "MATRICES_ISO23091_2_RGB_TO_XYZ",
    "MATRICES_XYZ_TO_ISO23091_2_RGB",
    "COLOUR_PRIMARIES_23001_8",
    "TRANSFER_CHARACTERISTICS_23001_8",
    "MATRIX_COEFFICIENTS_23001_8",
    "CCS_WHITEPOINTS_23001_8",
    "WHITEPOINT_NAMES_23001_8",
    "MATRICES_23001_8_RGB_TO_XYZ",
    "MATRICES_XYZ_TO_23001_8_RGB",
    "COLOUR_PRIMARIES_ISO14496_10",
    "TRANSFER_CHARACTERISTICS_ISO14496_10",
    "MATRIX_COEFFICIENTS_ISO14496_10",
    "CCS_WHITEPOINTS_ISO14496_10",
    "WHITEPOINT_NAMES_ISO14496_10",
    "MATRICES_ISO14496_10_RGB_TO_XYZ",
    "MATRICES_XYZ_TO_ISO14496_10_RGB",
    "describe_video_signal_colour_primaries",
    "describe_video_signal_transfer_characteristics",
    "describe_video_signal_matrix_coefficients",
]


def _clipped_domain_function(
    function: Callable, domain: Union[list, tuple] = (0, 1)
) -> Callable:
    """
    Wrap given function and produce a new callable clipping the input value to
    given domain.

    Parameters
    ----------
    function
        Function to wrap.
    domain
        Domain to use for clipping.

    Examples
    --------
    >>> linear_clipped = _clipped_domain_function(linear_function, (0.1, 0.9))
    >>> linear_clipped(1)  # doctest: +ELLIPSIS
    0.9000000...
    """

    @functools.wraps(function)
    def wrapped(x, *args: Any, **kwargs: Any) -> Any:
        """Wrap given function."""

        return function(np.clip(x, *domain), *args, **kwargs)

    return wrapped


def _reserved(*args: Any):  # noqa: ARG001
    """
    Define a reserved function.

    Examples
    --------
    >>> try:
    ...     _reserved()
    ... except RuntimeError:
    ...     pass
    ...
    """

    raise RuntimeError("Reserved; For future use by ITU-T | ISO/IEC.")


def _unspecified(*args: Any):  # noqa: ARG001
    """
    Define an unspecified function.

    Examples
    --------
    >>> try:
    ...     _unspecified()
    ... except RuntimeError:
    ...     pass
    ...
    """

    raise RuntimeError(
        "Unspecified; Image characteristics are unknown or are determined by "
        "the application."
    )


COLOUR_PRIMARIES_ITUTH273: Dict[int, NDArrayFloat] = {
    0: np.array("Reserved"),
    # For future use by ITU-T | ISO/IEC.
    #
    1: PRIMARIES_BT709,
    # Rec. ITU-R BT.709-6 Rec. ITU-R BT.1361-0 conventional colour gamut
    # system and extended colour gamut system (historical) IEC 61966-2-1 sRGB
    # or sYCC IEC 61966-2-4 Society of Motion Picture and Television Engineers
    # (SMPTE) RP 177 (1993) Annex B.
    #
    2: np.array("Unspecified"),
    # Image characteristics are unknown or are determined by the
    # application.
    #
    3: np.array("Reserved"),
    # For future use by ITU-T | ISO/IEC.
    #
    4: PRIMARIES_BT470_525,
    # Rec. ITU-R BT.470-6 System M (historical) United States National
    # Television System Committee 1953 Recommendation for transmission
    # standards for color television United States Federal Communication
    # Commission (2003) Title 47 Code of Federal Regulations 73.682 (a) (20).
    #
    5: PRIMARIES_BT470_625,
    # 5: Rec. ITU-R BT.470-6 System B, G (historical) Rec. ITU-R BT.601-7 625
    # Rec. ITU-R BT.1358-0 625 (historical) Rec. ITU-R BT.1700-0 625 PAL and
    # 625 SECAM.
    #
    6: PRIMARIES_SMPTE_240M,
    # 6: Rec. ITU-R BT.601-7 525 Rec. ITU-R BT.1358-1 525 or 625 (historical)
    # Rec. ITU-R BT.1700-0 NTSC SMPTE ST 170 (2004) (functionally the same as
    # the value 7).
    #
    7: PRIMARIES_SMPTE_240M,
    # SMPTE ST 240 (1999) (functionally the same as the value 6).
    #
    8: PRIMARIES_H273_GENERIC_FILM,
    # Generic film (colour filters using Illuminant C), Red: Wratten 25,
    # Green: Wratten 58, Blue: Wratten 47.
    #
    9: PRIMARIES_BT2020,
    # Rec. ITU-R BT.2020-2 Rec. ITU-R BT.2100-2.
    #
    10: PRIMARIES_DCDM_XYZ,
    # SMPTE ST 428-1 (2019) (CIE 1931 XYZ as in ISO 11664-1).
    #
    11: PRIMARIES_DCI_P3,
    # SMPTE RP 431-2 (2011).
    #
    12: PRIMARIES_P3_D65,
    # SMPTE EG 432-1 (2010).
    #
    # 13-21:
    # For future use by ITU-T | ISO/IEC.
    #
    22: PRIMARIES_H273_22_UNSPECIFIED,
    # No corresponding industry specification identified.
    #
    23: np.array("Reserved"),
    # 23-255:
    # For future use by ITU-T | ISO/IEC.
    #
}
if is_documentation_building():  # pragma: no cover
    COLOUR_PRIMARIES_ITUTH273 = DocstringDict(COLOUR_PRIMARIES_ITUTH273)
    COLOUR_PRIMARIES_ITUTH273.__doc__ = """
*ColourPrimaries* indicates the chromaticity coordinates of the source colour
primaries as specified in Table 3 of
:cite:`InternationalOrganizationforStandardization2021` and
:cite:`InternationalTelecommunicationUnion2021` in terms of the CIE 1931
definition of x and y, which shall be interpreted as specified by
*ISO/ CIE 11664-1*.

References
----------
:cite:`InternationalOrganizationforStandardization2013`
:cite:`InternationalOrganizationforStandardization2020`
:cite:`InternationalOrganizationforStandardization2021`,
:cite:`InternationalTelecommunicationUnion2021`
"""


class FFmpegConstantsColourPrimaries_ITUTH273(IntEnum):
    """
    Define the constant names used by *FFmpeg* in the `AVColorPrimaries` enum.

    Notes
    -----
    -   *AVCOL_PRI_JEDEC_P22* is equal to `AVCOL_PRI_EBU3213` in *FFmpeg*
        but neither *Recommendation ITU-T H.273* (2021) nor *ISO/IEC 23091-2*
        (2021) define the same primaries as *EBU Tech 3213*, nor do they refer
        to it. *ColourPrimaries 22* in both standards specifies the
        informative remark *No corresponding industry specification identified*.
        However, *ISO/IEC 23001-8* (2013) and *ISO/IEC 14497-10* (2020)
        specify the *JEDEC P22 phosphors* and *EBU Tech. 3213-E (1975)*
        informative remarks respectively while defining the same primaries and
        whitepoint as the 2021 standards. This is likely an error in the
        earlier standards that was discovered and corrected.

    References
    ----------
    :cite:`FFmpegDevelopers2022`,
    :cite:`InternationalOrganizationforStandardization2013`,
    :cite:`InternationalOrganizationforStandardization2021`,
    :cite:`InternationalOrganizationforStandardization2020`,
    :cite:`InternationalTelecommunicationUnion2021`
    """

    AVCOL_PRI_RESERVED0 = 0
    AVCOL_PRI_BT709 = 1
    AVCOL_PRI_UNSPECIFIED = 2
    AVCOL_PRI_RESERVED = 3
    AVCOL_PRI_BT470M = 4
    AVCOL_PRI_BT470BG = 5
    AVCOL_PRI_SMPTE170M = 6
    AVCOL_PRI_SMPTE240M = 7
    AVCOL_PRI_FILM = 8
    AVCOL_PRI_BT2020 = 9
    AVCOL_PRI_SMPTE428 = 10
    AVCOL_PRI_SMPTEST428_1 = AVCOL_PRI_SMPTE428
    AVCOL_PRI_SMPTE431 = 11
    AVCOL_PRI_SMPTE432 = 12
    AVCOL_PRI_EBU3213 = 22
    AVCOL_PRI_JEDEC_P22 = AVCOL_PRI_EBU3213
    AVCOL_PRI_NB = auto()

    RESERVED0 = AVCOL_PRI_RESERVED0
    BT709 = AVCOL_PRI_BT709
    UNSPECIFIED = AVCOL_PRI_UNSPECIFIED
    RESERVED = AVCOL_PRI_RESERVED
    BT470M = AVCOL_PRI_BT470M
    BT470BG = AVCOL_PRI_BT470BG
    SMPTE170M = AVCOL_PRI_SMPTE170M
    SMPTE240M = AVCOL_PRI_SMPTE240M
    FILM = AVCOL_PRI_FILM
    BT2020 = AVCOL_PRI_BT2020
    SMPTE428 = AVCOL_PRI_SMPTE428
    SMPTEST428_1 = AVCOL_PRI_SMPTEST428_1
    SMPTE431 = AVCOL_PRI_SMPTE431
    SMPTE432 = AVCOL_PRI_SMPTE432
    EBU3213 = AVCOL_PRI_EBU3213
    JEDEC_P22 = AVCOL_PRI_JEDEC_P22
    NB = AVCOL_PRI_NB


TRANSFER_CHARACTERISTICS_ITUTH273: Dict[int, Callable] = {
    0: _reserved,
    # For future use by ITU-T | ISO/IEC.
    #
    1: _clipped_domain_function(oetf_BT709),
    # Rec. ITU-R BT.709-6 Rec. ITU-R BT.1361-0 conventional colour gamut
    # system (historical) (functionally the same as the values 6, 14 and 15).
    #
    2: _unspecified,
    # Image characteristics are unknown or are determined by the
    # application.
    #
    3: _reserved,
    # For future use by ITU-T | ISO/IEC.
    #
    4: _clipped_domain_function(
        functools.partial(gamma_function, exponent=1 / 2.2)
    ),
    # Assumed display gamma 2.2 Rec. ITU-R BT.470-6 System M (historical)
    # United States National Television System Committee 1953 Recommendation
    # for transmission standards for color television United States Federal
    # Communications Commission (2003) Title 47 Code of Federal Regulations
    # 73.682 (a) (20) Rec. ITU-R BT.1700-0 625 PAL and 625 SECAM.
    #
    5: _clipped_domain_function(
        functools.partial(gamma_function, exponent=1 / 2.8)
    ),
    # 5: Assumed display gamma 2.8 Rec. ITU-R BT.470-6 System B, G (historical).
    #
    6: _clipped_domain_function(oetf_BT601),
    # Rec. ITU-R BT.601-7 525 or 625 Rec. ITU-R BT.1358-1 525 or 625
    # (historical) Rec. ITU-R BT.1700-0 NTSC SMPTE ST 170 (2004) (functionally
    # the same as the values 1, 14 and 15).
    #
    7: _clipped_domain_function(oetf_SMPTE240M),
    # SMPTE ST 240 (1999).
    #
    8: _clipped_domain_function(linear_function),
    # Linear transfer characteristics.
    #
    9: _clipped_domain_function(oetf_H273_Log),
    # Logarithmic transfer characteristic (100:1 range).
    #
    10: _clipped_domain_function(oetf_H273_LogSqrt),
    # Logarithmic transfer characteristic (100 * Sqrt( 10 ) : 1 range).
    #
    11: oetf_H273_IEC61966_2,
    # IEC 61966-2-4.
    #
    12: _clipped_domain_function(oetf_BT1361, (-0.25, 1.33)),
    # Rec. ITU-R BT.1361-0 extended colour gamut system (historical).
    #
    13: oetf_H273_IEC61966_2,
    # IEC 61966-2-1 sRGB (with MatrixCoefficients equal to 0)
    # IEC 61966-2-1 sYCC (with MatrixCoefficients equal to 5).
    #
    14: _clipped_domain_function(
        functools.partial(oetf_BT2020, is_12_bits_system=False)
    ),
    # Rec. ITU-R BT.2020-2 (10-bit system) (functionally the same as the values
    # 1, 6 and 15).
    #
    15: _clipped_domain_function(
        functools.partial(oetf_BT2020, is_12_bits_system=True)
    ),
    # Rec. ITU-R BT.2020-2 (12-bit system) (functionally the same as the values
    # 1, 6 and 14).
    #
    16: eotf_inverse_ST2084,
    # SMPTE ST 2084 (2014) for 10-, 12-, 14- and 16-bit systems Rec.
    # ITU-R BT.2100-2 perceptual quantization (PQ) system.
    #
    17: eotf_inverse_H273_ST428_1,
    # SMPTE ST 428-1 (2019).
    #
    18: oetf_BT2100_HLG,
    # ARIB STD-B67 (2015) Rec. ITU-R BT.2100-2 hybrid log-gamma (HLG) system.
    #
    19: _reserved,
    # 19-255:
    # For future use by ITU-T | ISO/IEC.
    #
}
if is_documentation_building():  # pragma: no cover
    TRANSFER_CHARACTERISTICS_ITUTH273 = DocstringDict(
        TRANSFER_CHARACTERISTICS_ITUTH273
    )
    TRANSFER_CHARACTERISTICS_ITUTH273.__doc__ = """
*TransferCharacteristics*, as specified in Table 3 of
:cite:`InternationalOrganizationforStandardization2021` and
:cite:`InternationalTelecommunicationUnion2021`, either indicates the reference
opto-electronic transfer characteristic function of the source picture as a
function of a source input linear optical intensity input Lc with a nominal
real-valued range of 0 to 1 or indicates the inverse of the reference
electro-optical transfer characteristic function as a function of an output
linear optical intensity Lo with a nominal real-valued range of 0 to 1.

Notes
-----
-   For simplicity, no clipping is implemented for *TransferCharacteristics 13*
    as it is a function of whether the context is *sRGB* or *sYCC*.
-   For TransferCharacteristics equal to 18, the equations given in Table 3 are
    normalized for a source input linear optical intensity Lc with a nominal
    real-valued range of 0 to 1. An alternative scaling that is mathematically
    equivalent is used in ARIB STD-B67 (2015) with the source input linear
    optical intensity having a nominal real-valued range of 0 to 12.

References
----------
:cite:`InternationalOrganizationforStandardization2013`
:cite:`InternationalOrganizationforStandardization2020`
:cite:`InternationalOrganizationforStandardization2021`,
:cite:`InternationalTelecommunicationUnion2021`
"""


class FFmpegConstantsTransferCharacteristics_ITUTH273(IntEnum):
    """
    Define the constant names used by *FFmpeg* in the
    `AVColorTransferCharacteristic` enum.

    References
    ----------
    :cite:`FFmpegDevelopers2022a`,
    :cite:`InternationalOrganizationforStandardization2013`,
    :cite:`InternationalOrganizationforStandardization2021`,
    :cite:`InternationalOrganizationforStandardization2020`,
    :cite:`InternationalTelecommunicationUnion2021`
    """

    AVCOL_TRC_RESERVED0 = 0
    AVCOL_TRC_BT709 = 1
    AVCOL_TRC_UNSPECIFIED = 2
    AVCOL_TRC_RESERVED = 3
    AVCOL_TRC_GAMMA22 = 4
    AVCOL_TRC_GAMMA28 = 5
    AVCOL_TRC_SMPTE170M = 6
    AVCOL_TRC_SMPTE240M = 7
    AVCOL_TRC_LINEAR = 8
    AVCOL_TRC_LOG = 9
    AVCOL_TRC_LOG_SQRT = 10
    AVCOL_TRC_IEC61966_2_4 = 11
    AVCOL_TRC_BT1361_ECG = 12
    AVCOL_TRC_IEC61966_2_1 = 13
    AVCOL_TRC_BT2020_10 = 14
    AVCOL_TRC_BT2020_12 = 15
    AVCOL_TRC_SMPTE2084 = 16
    AVCOL_TRC_SMPTEST2084 = AVCOL_TRC_SMPTE2084
    AVCOL_TRC_SMPTE428 = 17
    AVCOL_TRC_SMPTEST428_1 = AVCOL_TRC_SMPTE428
    AVCOL_TRC_ARIB_STD_B67 = 18
    AVCOL_TRC_NB = auto()

    RESERVED0 = AVCOL_TRC_RESERVED0
    BT709 = AVCOL_TRC_BT709
    UNSPECIFIED = AVCOL_TRC_UNSPECIFIED
    RESERVED = AVCOL_TRC_RESERVED
    GAMMA22 = AVCOL_TRC_GAMMA22
    GAMMA28 = AVCOL_TRC_GAMMA28
    SMPTE170M = AVCOL_TRC_SMPTE170M
    SMPTE240M = AVCOL_TRC_SMPTE240M
    LINEAR = AVCOL_TRC_LINEAR
    LOG = AVCOL_TRC_LOG
    LOG_SQRT = AVCOL_TRC_LOG_SQRT
    IEC61966_2_4 = AVCOL_TRC_IEC61966_2_4
    BT1361_ECG = AVCOL_TRC_BT1361_ECG
    IEC61966_2_1 = AVCOL_TRC_IEC61966_2_1
    BT2020_10 = AVCOL_TRC_BT2020_10
    BT2020_12 = AVCOL_TRC_BT2020_12
    SMPTE2084 = AVCOL_TRC_SMPTE2084
    SMPTEST2084 = AVCOL_TRC_SMPTEST2084
    SMPTE428 = AVCOL_TRC_SMPTE428
    SMPTEST428_1 = AVCOL_TRC_SMPTEST428_1
    ARIB_STD_B67 = AVCOL_TRC_ARIB_STD_B67
    NB = AVCOL_TRC_NB


MATRIX_COEFFICIENTS_ITUTH273: Dict[int, NDArrayFloat] = {
    0: np.array("Identity"),
    # The identity matrix. Typically used for GBR (often referred to as RGB);
    # however, may also be used for YZX (often referred to as XYZ);
    # IEC 61966-2-1 sRGB SMPTE ST 428-1 (2019) See equations 41 to 43.
    #
    1: np.array([0.2126, 0.0722]),
    # Rec. ITU-R BT.709-6 Rec. ITU-R BT.1361-0 conventional colour gamut system
    # and extended colour gamut system (historical) IEC 61966-2-4 xvYCC709
    # SMPTE RP 177 (1993) Annex B See equations 38 to 40.
    #
    2: np.array("Unspecified"),
    # Image characteristics are unknown or are determined by the application.
    #
    3: np.array("Reserved"),
    # For future use by ITU-T | ISO/IEC.
    #
    4: np.array([0.30, 0.11]),
    # United States Federal Communications Commission (2003) Title 47 Code of
    # Federal Regulations 73.682 (a) (20) See equations 38 to 40.
    #
    5: np.array([0.299, 0.114]),
    # Rec. ITU-R BT.470-6 System B, G (historical) Rec. ITU-R BT.601-7 625
    # Rec. ITU-R BT.1358-0 625 (historical) Rec. ITU-R BT.1700-0 625 PAL and
    # 625 SECAM IEC 61966-2-1 sYCC IEC 61966-2-4 xvYCC601 (functionally the
    # same as the value 6) See equations 38 to 40.
    #
    6: np.array([0.299, 0.114]),
    # Rec. ITU-R BT.601-7 525 Rec. ITU-R BT.1358-1 525 or 625 (historical)
    # Rec. ITU-R BT.1700-0 NTSC SMPTE ST 170 (2004) (functionally the same as
    # the value 5) See equations 38 to 40.
    #
    7: np.array([0.212, 0.087]),
    # SMPTE ST 240 (1999) See equations 38 to 40.
    #
    8: np.array("YCgCo"),
    # See equations 44 to 58.
    #
    9: np.array([0.2627, 0.0593]),
    # Rec. ITU-R BT.2020-2 (non-constant luminance) Rec. ITU-R BT.2100-2
    # Y'CbCr See equations 38 to 40.
    #
    10: np.array([0.2627, 0.0593]),
    # Rec. ITU-R BT.2020-2 (constant luminance) See equations 59 to 68.
    #
    11: np.array("Y'D'ZD'X"),
    # SMPTE ST 2085 (2015) See equations 69 to 71.
    #
    12: np.array("See equations 32 to 37"),
    # Chromaticity-derived non-constant luminance system See equations 38 to 40.
    #
    13: np.array("See equations 32 to 37"),
    # Chromaticity-derived constant luminance system See equations 59 to 68.
    #
    14: np.array("ICTCP"),
    # Rec. ITU-R BT.2100-2 ICTCP See equations 72 to 74 for
    # TransferCharacteristics value 16 (PQ) See equations 75 to 77 for
    # TransferCharacteristics value 18 (HLG).
    #
    15: np.array("Reserved"),
    # 15-255:
    # For future use by ITU-T | ISO/IEC.
    #
}
if is_documentation_building():  # pragma: no cover
    MATRIX_COEFFICIENTS_ITUTH273 = DocstringDict(MATRIX_COEFFICIENTS_ITUTH273)
    MATRIX_COEFFICIENTS_ITUTH273.__doc__ = """
*MatrixCoefficients* describes the matrix coefficients used in deriving luma
and chroma signals from the green, blue and red or X, Y and Z primaries, as
specified in Table 4 and equations 11 to 77 of
:cite:`InternationalOrganizationforStandardization2021` and
:cite:`InternationalTelecommunicationUnion2021`.

Notes
-----
-   See :attr:`colour.WEIGHTS_YCBCR` attribute and the
    :func:`colour.matrix_YCbCr`, :func:`colour.offset_YCbCr`,
    :func:`colour.RGB_to_YCbCr`, :func:`colour.YCbCr_to_RGB`,
    :func:`colour.RGB_to_YcCbcCrc`, :func:`colour.YcCbcCrc_to_RGB` definitions
    for an implementation.

References
----------
:cite:`InternationalOrganizationforStandardization2013`
:cite:`InternationalOrganizationforStandardization2020`
:cite:`InternationalOrganizationforStandardization2021`,
:cite:`InternationalTelecommunicationUnion2021`
"""


class FFmpegConstantsMatrixCoefficients_ITUTH273(IntEnum):
    """
    Define the constant names used by *FFmpeg* in the `AVColorSpace` enum.

    References
    ----------
    :cite:`FFmpegDevelopers2022b`,
    :cite:`InternationalOrganizationforStandardization2013`,
    :cite:`InternationalOrganizationforStandardization2021`,
    :cite:`InternationalOrganizationforStandardization2020`,
    :cite:`InternationalTelecommunicationUnion2021`
    """

    AVCOL_SPC_RGB = 0
    AVCOL_SPC_BT709 = 1
    AVCOL_SPC_UNSPECIFIED = 2
    AVCOL_SPC_RESERVED = 3
    AVCOL_SPC_FCC = 4
    AVCOL_SPC_BT470BG = 5
    AVCOL_SPC_SMPTE170M = 6
    AVCOL_SPC_SMPTE240M = 7
    AVCOL_SPC_YCGCO = 8
    AVCOL_SPC_YCOCG = AVCOL_SPC_YCGCO
    AVCOL_SPC_BT2020_NCL = 9
    AVCOL_SPC_BT2020_CL = 10
    AVCOL_SPC_SMPTE2085 = 11
    AVCOL_SPC_CHROMA_DERIVED_NCL = 12
    AVCOL_SPC_CHROMA_DERIVED_CL = 13
    AVCOL_SPC_ICTCP = 14
    AVCOL_SPC_NB = auto()

    RGB = AVCOL_SPC_RGB
    BT709 = AVCOL_SPC_BT709
    UNSPECIFIED = AVCOL_SPC_UNSPECIFIED
    RESERVED = AVCOL_SPC_RESERVED
    FCC = AVCOL_SPC_FCC
    BT470BG = AVCOL_SPC_BT470BG
    SMPTE170M = AVCOL_SPC_SMPTE170M
    SMPTE240M = AVCOL_SPC_SMPTE240M
    YCGCO = AVCOL_SPC_YCGCO
    YCOCG = AVCOL_SPC_YCOCG
    BT2020_NCL = AVCOL_SPC_BT2020_NCL
    BT2020_CL = AVCOL_SPC_BT2020_CL
    SMPTE2085 = AVCOL_SPC_SMPTE2085
    CHROMA_DERIVED_NCL = AVCOL_SPC_CHROMA_DERIVED_NCL
    CHROMA_DERIVED_CL = AVCOL_SPC_CHROMA_DERIVED_CL
    ICTCP = AVCOL_SPC_ICTCP
    NB = AVCOL_SPC_NB


CCS_WHITEPOINTS_ITUTH273: Dict[int, NDArrayFloat] = {
    0: np.array("Reserved"),
    1: CCS_WHITEPOINT_BT709,
    2: np.array("Unspecified"),
    3: np.array("Reserved"),
    4: np.around(CCS_WHITEPOINT_BT470_525, 3),
    5: CCS_WHITEPOINT_BT470_625,
    6: CCS_WHITEPOINT_SMPTE_240M,
    7: CCS_WHITEPOINT_SMPTE_240M,
    8: CCS_WHITEPOINT_H273_GENERIC_FILM,
    9: CCS_WHITEPOINT_BT2020,
    10: CCS_WHITEPOINT_DCDM_XYZ,
    11: CCS_WHITEPOINT_DCI_P3,
    12: CCS_WHITEPOINT_P3_D65,
    22: CCS_WHITEPOINT_H273_22_UNSPECIFIED,
    23: np.array("Reserved"),
}
if is_documentation_building():  # pragma: no cover
    CCS_WHITEPOINTS_ITUTH273 = DocstringDict(CCS_WHITEPOINTS_ITUTH273)
    CCS_WHITEPOINTS_ITUTH273.__doc__ = """
Chromaticity coordinates of the whitepoints associated with the source colour
primaries as specified in Table 3 of
:cite:`InternationalOrganizationforStandardization2021` and
:cite:`InternationalTelecommunicationUnion2021` in terms of the CIE 1931
definition of x and y, which shall be interpreted as specified by
*ISO/ CIE 11664-1*.

References
----------
:cite:`InternationalOrganizationforStandardization2013`
:cite:`InternationalOrganizationforStandardization2020`
:cite:`InternationalOrganizationforStandardization2021`,
:cite:`InternationalTelecommunicationUnion2021`

Notes
-----
-   :cite:`InternationalTelecommunicationUnion2021` defines
    *CIE Illuminant C* as [0.310, 0.316], while *Colour* definition has a
    slightly higher precision.
"""


WHITEPOINT_NAMES_ITUTH273: Dict[int, str] = {
    0: "Reserved",
    1: WHITEPOINT_NAME_BT709,
    2: "Unspecified",
    3: "Reserved",
    4: WHITEPOINT_NAME_BT470_525,
    5: WHITEPOINT_NAME_BT470_625,
    6: WHITEPOINT_NAME_SMPTE_240M,
    7: WHITEPOINT_NAME_SMPTE_240M,
    8: WHITEPOINT_NAME_H273_GENERIC_FILM,
    9: WHITEPOINT_NAME_BT2020,
    10: WHITEPOINT_NAME_DCDM_XYZ,
    11: WHITEPOINT_NAME_DCI_P3,
    12: WHITEPOINT_NAME_P3_D65,
    22: WHITEPOINT_NAME_H273_22_UNSPECIFIED,
    23: "Reserved",
}
if is_documentation_building():  # pragma: no cover
    WHITEPOINT_NAMES_ITUTH273 = DocstringDict(WHITEPOINT_NAMES_ITUTH273)
    WHITEPOINT_NAMES_ITUTH273.__doc__ = """
Whitepoint names associated with the source colour primaries as specified in
Table 3 of :cite:`InternationalOrganizationforStandardization2021` and
:cite:`InternationalTelecommunicationUnion2021`.

References
----------
:cite:`InternationalOrganizationforStandardization2013`
:cite:`InternationalOrganizationforStandardization2020`
:cite:`InternationalOrganizationforStandardization2021`,
:cite:`InternationalTelecommunicationUnion2021`
"""


MATRICES_ITUTH273_RGB_TO_XYZ = {
    0: np.array("Reserved"),
    1: MATRIX_BT709_TO_XYZ,
    2: np.array("Unspecified"),
    3: np.array("Reserved"),
    4: MATRIX_BT470_525_TO_XYZ,
    5: MATRIX_BT470_625_TO_XYZ,
    6: MATRIX_SMPTE_240M_TO_XYZ,
    7: MATRIX_SMPTE_240M_TO_XYZ,
    8: MATRIX_H273_GENERIC_FILM_RGB_TO_XYZ,
    9: MATRIX_BT2020_TO_XYZ,
    10: MATRIX_DCDM_XYZ_TO_XYZ,
    11: MATRIX_DCI_P3_TO_XYZ,
    12: MATRIX_P3_D65_TO_XYZ,
    22: MATRIX_H273_22_UNSPECIFIED_RGB_TO_XYZ,
    23: np.array("Reserved"),
}
if is_documentation_building():  # pragma: no cover
    WHITEPOINT_NAMES_ITUTH273 = DocstringDict(WHITEPOINT_NAMES_ITUTH273)
    WHITEPOINT_NAMES_ITUTH273.__doc__ = """
*RGB* to *CIE XYZ* tristimulus values matrices associated with the source
colour primaries as specified in Table 3 of
:cite:`InternationalOrganizationforStandardization2021` and
:cite:`InternationalTelecommunicationUnion2021`.

References
----------
:cite:`InternationalOrganizationforStandardization2013`
:cite:`InternationalOrganizationforStandardization2020`
:cite:`InternationalOrganizationforStandardization2021`,
:cite:`InternationalTelecommunicationUnion2021`
"""

MATRICES_XYZ_TO_ITUTH273_RGB = {
    0: np.array("Reserved"),
    1: MATRIX_XYZ_TO_BT709,
    2: np.array("Unspecified"),
    3: np.array("Reserved"),
    4: MATRIX_XYZ_TO_BT470_525,
    5: MATRIX_XYZ_TO_BT470_625,
    6: MATRIX_XYZ_TO_SMPTE_240M,
    7: MATRIX_XYZ_TO_SMPTE_240M,
    8: MATRIX_XYZ_TO_H273_GENERIC_FILM_RGB,
    9: MATRIX_XYZ_TO_BT2020,
    10: MATRIX_XYZ_TO_DCDM_XYZ,
    11: MATRIX_XYZ_TO_DCI_P3,
    12: MATRIX_XYZ_TO_P3_D65,
    22: MATRIX_XYZ_TO_H273_22_UNSPECIFIED_RGB,
    23: np.array("Reserved"),
}
if is_documentation_building():  # pragma: no cover
    WHITEPOINT_NAMES_ITUTH273 = DocstringDict(WHITEPOINT_NAMES_ITUTH273)
    WHITEPOINT_NAMES_ITUTH273.__doc__ = """
*CIE XYZ* tristimulus values to *RGB* matrices associated with the source
colour primaries as specified in Table 3 of
:cite:`InternationalOrganizationforStandardization2021` and
:cite:`InternationalTelecommunicationUnion2021`.

References
----------
:cite:`InternationalOrganizationforStandardization2013`
:cite:`InternationalOrganizationforStandardization2020`
:cite:`InternationalOrganizationforStandardization2021`,
:cite:`InternationalTelecommunicationUnion2021`
"""


# Aliases for *ISO/IEC 23091-2* (2021).
#
# Verified to be functionally identical to *Recommendation ITU-T H.273* for
# the values defined here.
COLOUR_PRIMARIES_ISO23091_2 = COLOUR_PRIMARIES_ITUTH273
TRANSFER_CHARACTERISTICS_ISO23091_2 = TRANSFER_CHARACTERISTICS_ITUTH273
MATRIX_COEFFICIENTS_ISO23091_2 = MATRIX_COEFFICIENTS_ITUTH273
CCS_WHITEPOINTS_ISO23091_2 = CCS_WHITEPOINTS_ITUTH273
WHITEPOINT_NAMES_ISO23091_2 = WHITEPOINT_NAMES_ITUTH273
MATRICES_ISO23091_2_RGB_TO_XYZ = MATRICES_ITUTH273_RGB_TO_XYZ
MATRICES_XYZ_TO_ISO23091_2_RGB = MATRICES_XYZ_TO_ITUTH273_RGB

# Aliases for *ISO/IEC 23001-8* (2013).
#
# Verified to be functionally identical to *Recommendation ITU-T H.273* for
# the values defined here except for the note regarding *EBU Tech 3213* and
# *ColourPrimaries 22*.
COLOUR_PRIMARIES_23001_8 = COLOUR_PRIMARIES_ITUTH273
TRANSFER_CHARACTERISTICS_23001_8 = TRANSFER_CHARACTERISTICS_ITUTH273
MATRIX_COEFFICIENTS_23001_8 = MATRIX_COEFFICIENTS_ITUTH273
CCS_WHITEPOINTS_23001_8 = CCS_WHITEPOINTS_ITUTH273
WHITEPOINT_NAMES_23001_8 = WHITEPOINT_NAMES_ITUTH273
MATRICES_23001_8_RGB_TO_XYZ = MATRICES_ITUTH273_RGB_TO_XYZ
MATRICES_XYZ_TO_23001_8_RGB = MATRICES_XYZ_TO_ITUTH273_RGB

# Aliases for *ISO/IEC 14496-10* (2020).
#
# Verified to be functionally identical to *Recommendation ITU-T H.273* for
# the values defined here except for the note regarding *EBU Tech 3213* and
# *ColourPrimaries 22*.
COLOUR_PRIMARIES_ISO14496_10 = COLOUR_PRIMARIES_ITUTH273
TRANSFER_CHARACTERISTICS_ISO14496_10 = TRANSFER_CHARACTERISTICS_ITUTH273
MATRIX_COEFFICIENTS_ISO14496_10 = MATRIX_COEFFICIENTS_ITUTH273
CCS_WHITEPOINTS_ISO14496_10 = CCS_WHITEPOINTS_ITUTH273
WHITEPOINT_NAMES_ISO14496_10 = WHITEPOINT_NAMES_ITUTH273
MATRICES_ISO14496_10_RGB_TO_XYZ = MATRICES_ITUTH273_RGB_TO_XYZ
MATRICES_XYZ_TO_ISO14496_10_RGB = MATRICES_XYZ_TO_ITUTH273_RGB


def describe_video_signal_colour_primaries(
    code_point: int, print_description: bool = True, **kwargs
) -> str:
    """
    Describe given video signal colour primaries code point.

    Parameters
    ----------
    code_point
        Video signal colour primaries code point to describe from
        :attr:`colour.COLOUR_PRIMARIES_ITUTH273` attribute.
    print_description
        Whether to print the description.

    Other Parameters
    ----------------
    padding
        {:func:`colour.utilities.message_box`},
        Padding on each side of the message.
    print_callable
        {:func:`colour.utilities.message_box`},
        Callable used to print the message box.
    width
        {:func:`colour.utilities.message_box`},
        Message box width.

    Returns
    -------
    str
        Video signal colour primaries code point description.

    References
    ----------
    :cite:`FFmpegDevelopers2022`,
    :cite:`InternationalOrganizationforStandardization2013`,
    :cite:`InternationalOrganizationforStandardization2021`,
    :cite:`InternationalOrganizationforStandardization2020`,
    :cite:`InternationalTelecommunicationUnion2021`

    Examples
    --------
    >>> description = describe_video_signal_colour_primaries(1, width=75)
    ===========================================================================
    *                                                                         *
    *   Colour Primaries: 1                                                   *
    *   -------------------                                                   *
    *                                                                         *
    *   Primaries        : [[ 0.64  0.33]                                     *
    *                       [ 0.3   0.6 ]                                     *
    *                       [ 0.15  0.06]]                                    *
    *   Whitepoint       : [ 0.3127  0.329 ]                                  *
    *   Whitepoint Name  : D65                                                *
    *   NPM              : [[ 0.4123908   0.35758434  0.18048079]             *
    *                       [ 0.21263901  0.71516868  0.07219232]             *
    *                       [ 0.01933082  0.11919478  0.95053215]]            *
    *   NPM -1           : [[ 3.24096994 -1.53738318 -0.49861076]             *
    *                       [-0.96924364  1.8759675   0.04155506]             *
    *                       [ 0.05563008 -0.20397696  1.05697151]]            *
    *   FFmpeg Constants : ['AVCOL_PRI_BT709', 'BT709']                       *
    *                                                                         *
    ===========================================================================
    >>> description = describe_video_signal_colour_primaries(2, width=75)
    ===========================================================================
    *                                                                         *
    *   Colour Primaries: 2                                                   *
    *   -------------------                                                   *
    *                                                                         *
    *   Primaries        : Unspecified                                        *
    *   Whitepoint       : Unspecified                                        *
    *   Whitepoint Name  : Unspecified                                        *
    *   NPM              : Unspecified                                        *
    *   NPM -1           : Unspecified                                        *
    *   FFmpeg Constants : ['AVCOL_PRI_UNSPECIFIED', 'UNSPECIFIED']           *
    *                                                                         *
    ===========================================================================
    >>> description = describe_video_signal_colour_primaries(
    ...     FFmpegConstantsColourPrimaries_ITUTH273.JEDEC_P22, width=75
    ... )
    ===========================================================================
    *                                                                         *
    *   Colour Primaries: 22                                                  *
    *   --------------------                                                  *
    *                                                                         *
    *   Primaries        : [[ 0.63   0.34 ]                                   *
    *                       [ 0.295  0.605]                                   *
    *                       [ 0.155  0.077]]                                  *
    *   Whitepoint       : [ 0.3127  0.329 ]                                  *
    *   Whitepoint Name  : D65                                                *
    *   NPM              : [[ 0.42942013  0.3277917   0.1932441 ]             *
    *                       [ 0.23175055  0.67225077  0.09599868]             *
    *                       [ 0.02044858  0.11111583  0.95749334]]            *
    *   NPM -1           : [[ 3.13288278 -1.44707454 -0.48720324]             *
    *                       [-1.08850877  2.01538781  0.01762239]             *
    *                       [ 0.05941301 -0.20297883  1.05275352]]            *
    *   FFmpeg Constants : ['AVCOL_PRI_EBU3213', 'AVCOL_PRI_JEDEC_P22',       *
    *   'EBU3213', 'JEDEC_P22']                                               *
    *                                                                         *
    ===========================================================================
    """

    @dataclass
    class SpecificationColourPrimaries:
        """Specification for video signal colour primaries code point."""

        code_point: int
        primaries: NDArrayFloat
        whitepoint: NDArrayFloat
        whitepoint_name: str
        matrix_RGB_to_XYZ: NDArrayFloat
        matrix_XYZ_to_RGB: NDArrayFloat
        ffmpeg_constants: list

    members = FFmpegConstantsColourPrimaries_ITUTH273.__members__.items()
    ffmpeg_constants = [
        name
        for name, member in members
        if member.name.startswith("AVCOL_PRI_") and member.value == code_point
    ]

    description = SpecificationColourPrimaries(
        code_point,
        COLOUR_PRIMARIES_ITUTH273[code_point],
        CCS_WHITEPOINTS_ITUTH273[code_point],
        WHITEPOINT_NAMES_ITUTH273[code_point],
        MATRICES_ITUTH273_RGB_TO_XYZ[code_point],
        MATRICES_XYZ_TO_ITUTH273_RGB[code_point],
        ffmpeg_constants,
    )

    message = multiline_str(
        description,
        [
            {
                "name": "code_point",
                "section": True,
                "formatter": lambda x: f"Colour Primaries: {x}",
            },
            {"line_break": True},
            {"name": "primaries", "label": "Primaries"},
            {"name": "whitepoint", "label": "Whitepoint"},
            {"name": "whitepoint_name", "label": "Whitepoint Name"},
            {"name": "matrix_RGB_to_XYZ", "label": "NPM"},
            {"name": "matrix_XYZ_to_RGB", "label": "NPM -1"},
            {"name": "ffmpeg_constants", "label": "FFmpeg Constants"},
        ],
    )

    if print_description:
        message_box(
            message,
            **kwargs,
        )

    return message


def describe_video_signal_transfer_characteristics(
    code_point: int, print_description: bool = True, **kwargs
) -> str:
    """
    Describe given video signal transfer characteristics code point.

    Parameters
    ----------
    code_point
        Video signal transfer characteristics code point to describe from
        :attr:`colour.TRANSFER_CHARACTERISTICS_ITUTH273` attribute.
    print_description
        Whether to print the description.

    Other Parameters
    ----------------
    padding
        {:func:`colour.utilities.message_box`},
        Padding on each side of the message.
    print_callable
        {:func:`colour.utilities.message_box`},
        Callable used to print the message box.
    width
        {:func:`colour.utilities.message_box`},
        Message box width.

    Returns
    -------
    str
        Video signal colour primaries code point description.

    References
    ----------
    :cite:`FFmpegDevelopers2022a`,
    :cite:`InternationalOrganizationforStandardization2013`,
    :cite:`InternationalOrganizationforStandardization2021`,
    :cite:`InternationalOrganizationforStandardization2020`,
    :cite:`InternationalTelecommunicationUnion2021`

    Examples
    --------
    >>> description = describe_video_signal_transfer_characteristics(
    ...     1, width=75
    ... )
    ... # doctest: +ELLIPSIS
    ===========================================================================
    *                                                                         *
    *   Transfer Characteristics: 1                                           *
    *   ---------------------------                                           *
    *                                                                         *
    *   Function         : <function oetf_BT709 at 0x...>...*
    *   FFmpeg Constants : ['AVCOL_TRC_BT709', 'BT709']                       *
    *                                                                         *
    ===========================================================================
    >>> description = describe_video_signal_transfer_characteristics(
    ...     2, width=75
    ... )
    ... # doctest: +ELLIPSIS
    ===========================================================================
    *                                                                         *
    *   Transfer Characteristics: 2                                           *
    *   ---------------------------                                           *
    *                                                                         *
    *   Function         : <function _unspecified at 0x...>...*
    *   FFmpeg Constants : ['AVCOL_TRC_UNSPECIFIED', 'UNSPECIFIED']           *
    *                                                                         *
    ===========================================================================
    >>> description = describe_video_signal_transfer_characteristics(
    ...     FFmpegConstantsTransferCharacteristics_ITUTH273.SMPTE428, width=75
    ... )
    ... # doctest: +ELLIPSIS
    ===========================================================================
    *                                                                         *
    *   Transfer Characteristics: 17                                          *
    *   ----------------------------                                          *
    *                                                                         *
    *   Function         : <function eotf_inverse_H273_ST428_1 at             *
    *   0x...>...*
    *   FFmpeg Constants : ['AVCOL_TRC_SMPTE428', 'AVCOL_TRC_SMPTEST428_1',   *
    *   'SMPTE428', 'SMPTEST428_1']                                           *
    *                                                                         *
    ===========================================================================
    """

    @dataclass
    class SpecificationTransferCharacteristics:
        """Specification for video signal transfer characteristics code point."""

        code_point: int
        function: Callable
        ffmpeg_constants: list

    members = (
        FFmpegConstantsTransferCharacteristics_ITUTH273.__members__.items()
    )
    ffmpeg_constants = [
        name
        for name, member in members
        if member.name.startswith("AVCOL_TRC_") and member.value == code_point
    ]

    description = SpecificationTransferCharacteristics(
        code_point,
        TRANSFER_CHARACTERISTICS_ITUTH273[code_point],
        ffmpeg_constants,
    )

    message = multiline_str(
        description,
        [
            {
                "name": "code_point",
                "section": True,
                "formatter": lambda x: f"Transfer Characteristics: {x}",
            },
            {"line_break": True},
            {"name": "function", "label": "Function"},
            {"name": "ffmpeg_constants", "label": "FFmpeg Constants"},
        ],
    )

    if print_description:
        message_box(
            message,
            **kwargs,
        )

    return message


def describe_video_signal_matrix_coefficients(
    code_point: int, print_description: bool = True, **kwargs
) -> str:
    """
    Describe given video signal matrix coefficients code point.

    Parameters
    ----------
    code_point
        Video signal matrix coefficients code point to describe from
        :attr:`colour.MATRIX_COEFFICIENTS_ITUTH273` attribute.
    print_description
        Whether to print the description.

    Other Parameters
    ----------------
    padding
        {:func:`colour.utilities.message_box`},
        Padding on each side of the message.
    print_callable
        {:func:`colour.utilities.message_box`},
        Callable used to print the message box.
    width
        {:func:`colour.utilities.message_box`},
        Message box width.

    Returns
    -------
    str
        Video signal colour primaries code point description.

    References
    ----------
    :cite:`FFmpegDevelopers2022b`,
    :cite:`InternationalOrganizationforStandardization2013`,
    :cite:`InternationalOrganizationforStandardization2021`,
    :cite:`InternationalOrganizationforStandardization2020`,
    :cite:`InternationalTelecommunicationUnion2021`

    Examples
    --------
    >>> description = describe_video_signal_matrix_coefficients(1, width=75)
    ===========================================================================
    *                                                                         *
    *   Matrix Coefficients: 1                                                *
    *   ----------------------                                                *
    *                                                                         *
    *   Matrix Coefficients : [ 0.2126  0.0722]                               *
    *   FFmpeg Constants    : ['AVCOL_SPC_BT709', 'BT709']                    *
    *                                                                         *
    ===========================================================================
    >>> description = describe_video_signal_matrix_coefficients(2, width=75)
    ===========================================================================
    *                                                                         *
    *   Matrix Coefficients: 2                                                *
    *   ----------------------                                                *
    *                                                                         *
    *   Matrix Coefficients : Unspecified                                     *
    *   FFmpeg Constants    : ['AVCOL_SPC_UNSPECIFIED', 'UNSPECIFIED']        *
    *                                                                         *
    ===========================================================================
    >>> description = describe_video_signal_matrix_coefficients(
    ...     FFmpegConstantsMatrixCoefficients_ITUTH273.ICTCP, width=75
    ... )
    ... # doctest: +ELLIPSIS
    ===========================================================================
    *                                                                         *
    *   Matrix Coefficients: 14                                               *
    *   -----------------------                                               *
    *                                                                         *
    *   Matrix Coefficients : ICTCP                                           *
    *   FFmpeg Constants    : ['AVCOL_SPC_ICTCP', 'ICTCP']                    *
    *                                                                         *
    ===========================================================================
    """

    @dataclass
    class SpecificationMatrixCoefficients:
        """Specification for video signal matrix coefficients code point."""

        code_point: int
        matrix_coefficients: NDArrayFloat
        ffmpeg_constants: list

    members = FFmpegConstantsMatrixCoefficients_ITUTH273.__members__.items()
    ffmpeg_constants = [
        name
        for name, member in members
        if member.name.startswith("AVCOL_SPC_") and member.value == code_point
    ]

    description = SpecificationMatrixCoefficients(
        code_point,
        MATRIX_COEFFICIENTS_ITUTH273[code_point],
        ffmpeg_constants,
    )

    message = multiline_str(
        description,
        [
            {
                "name": "code_point",
                "section": True,
                "formatter": lambda x: f"Matrix Coefficients: {x}",
            },
            {"line_break": True},
            {
                "name": "matrix_coefficients",
                "label": "Matrix Coefficients",
            },
            {"name": "ffmpeg_constants", "label": "FFmpeg Constants"},
        ],
    )

    if print_description:
        message_box(
            message,
            **kwargs,
        )

    return message
