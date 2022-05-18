"""
ITU-T H.273 video colour metadata
=================================

Defines aliases to a set of standard colour primaries, transfer functions and
YUV to RGB matrix coefficients used in video metadata.

These metadata were historically defined in ISO/IEC 23001-8 (:cite:`ISO2013`) then
superseded and duplicated by other standards (among others: :cite:`ISO2020`,
:cite:`ISO2021`, :cite:`ITU2021`). They are widely in use to define color-related
properties in video encoding and decoding software libraries, including ffmpeg.

References
----------
-   :cite:`ITU2021` : International Telecommunication Union. (2021).
    Recommendation ITU-T H.273 - Coding-independent code points for video
    signal type identification.
    https://www.itu.int/rec/T-REC-H.273-202107-I/en
-   :cite:`ISO2013` : International Organization for Standardization. (2013).
    INTERNATIONAL STANDARD ISO/IEC 23001-8:2013 - Information technology - MPEG
    systems technologies - Part 8: Coding-independent code points, §7.1 "Colour
    primaries"
-   :cite:`ISO2021` : International Organization for Standardization. (2021).
    INTERNATIONAL STANDARD ISO/IEC 23091-2:2021 - Information technology -
    Coding-independent code points - Part 2: Video, §8.1 "Colour primaries"
-   :cite:`ISO2020` : International Organization for Standardization. (2020).
    INTERNATIONAL STANDARD ISO/IEC 14496-10:2020 - Information technology -
    Coding of audio-visual objects - Part 10: Advanced video coding"
"""

import functools
import enum

import numpy as np

import colour

import colour.models.rgb.datasets.itut_h_273
from colour.models.rgb.transfer_functions.gamma import gamma_function
from colour.models.rgb.transfer_functions.itur_bt_709 import oetf_BT709
from colour.models.rgb.transfer_functions.itur_bt_1886 import eotf_BT1886
from colour.models.rgb.transfer_functions.itur_bt_2020 import oetf_BT2020
from colour.models.rgb.transfer_functions.itur_bt_1361 import (
    oetf_BT1361_extended,
)
from colour.models.rgb.transfer_functions.itur_bt_2100 import oetf_BT2100_HLG
from colour.models.rgb.transfer_functions.st_428 import eotf_inverse_ST428_1
from colour.models.rgb.transfer_functions.st_2084 import eotf_inverse_ST2084
from colour.models.rgb.transfer_functions.iec_61966_2 import (
    oetf_iec_61966_2_unbounded,
)
from colour.models.rgb.transfer_functions.itur_bt_601 import oetf_BT601
from colour.models.rgb.transfer_functions.smpte_240m import oetf_SMPTE240M
from colour.models.rgb.transfer_functions.itut_h_273 import (
    oetf_linear,
    oetf_log,
    oetf_log_sqrt,
)

__all__ = [
    "CCS_WHITEPOINTS_ISO14496_10",
    "CCS_WHITEPOINTS_ISO23001_8",
    "CCS_WHITEPOINTS_ISO23091_2",
    "CCS_WHITEPOINTS_ITU_T_H273",
    "ColourPrimaries_ISO14496_10",
    "ColourPrimaries_ISO23001_8",
    "ColourPrimaries_ISO23091_2",
    "ColourPrimaries_ITU_T_H273",
    "MATRICES_ISO14496_10_RGB_TO_XYZ",
    "MATRICES_ISO23001_8_RGB_TO_XYZ",
    "MATRICES_ISO23091_2_RGB_TO_XYZ",
    "MATRICES_ITU_T_H273_RGB_TO_XYZ",
    "MATRICES_XYZ_TO_ISO14496_10_RGB",
    "MATRICES_XYZ_TO_ISO23001_8_RGB",
    "MATRICES_XYZ_TO_ISO23091_2_RGB",
    "MATRICES_XYZ_TO_ITU_T_H273_RGB",
    "MatrixCoefficients_ISO14496_10",
    "MatrixCoefficients_ISO23001_8",
    "MatrixCoefficients_ISO23091_2",
    "MatrixCoefficients_ITU_T_H273",
    "PRIMARIES_ISO14496_10",
    "PRIMARIES_ISO23001_8",
    "PRIMARIES_ISO23091_2",
    "PRIMARIES_ITU_T_H273",
    "TRANSFER_FUNCTIONS_ISO14496_10",
    "TRANSFER_FUNCTIONS_ISO23001_8",
    "TRANSFER_FUNCTIONS_ISO23091_2",
    "TRANSFER_FUNCTIONS_ITU_T_H273",
    "TransferCharacteristics_ISO14496_10",
    "TransferCharacteristics_ISO23001_8",
    "TransferCharacteristics_ISO23091_2",
    "TransferCharacteristics_ITU_T_H273",
    "WEIGHTS_YCBCR_ISO14496_10",
    "WEIGHTS_YCBCR_ISO23001_8",
    "WEIGHTS_YCBCR_ISO23091_2",
    "WEIGHTS_YCBCR_ITU_T_H273",
    "WHITEPOINTS_NAMES_ISO14496_10",
    "WHITEPOINTS_NAMES_ISO23001_8",
    "WHITEPOINTS_NAMES_ISO23091_2",
    "WHITEPOINTS_NAMES_ITU_T_H273",
    "eotf_BT1886",
    "eotf_inverse_ST2084",
    "eotf_inverse_ST428_1",
    "eotf_inverse_gamma22",
    "eotf_inverse_gamma28",
    "oetf_BT1361_extended",
    "oetf_BT2020_10bit",
    "oetf_BT2020_12bit",
    "oetf_BT2100_HLG",
    "oetf_BT601",
    "oetf_BT709",
    "oetf_SMPTE240M",
    "oetf_iec_61966_2_1",
    "oetf_iec_61966_2_4",
    "oetf_linear",
    "oetf_log",
    "oetf_log_sqrt",
]

oetf_BT2020_12bit = functools.partial(oetf_BT2020, is_12_bits_system=True)
oetf_BT2020_10bit = functools.partial(oetf_BT2020, is_12_bits_system=False)
eotf_inverse_gamma22 = functools.partial(gamma_function, exponent=1 / 2.2)
eotf_inverse_gamma28 = functools.partial(gamma_function, exponent=1 / 2.8)
oetf_iec_61966_2_4 = oetf_iec_61966_2_unbounded
oetf_iec_61966_2_1 = oetf_iec_61966_2_unbounded


@enum.unique
class ColourPrimaries_ITU_T_H273(enum.IntEnum):
    """ColourPrimaries tags as defined in ITU-T H.273 § 8.1.

    The enumeration members use  the same names as ffmpeg `AVCOL_PRI_*` constants.
    """

    BT709 = 1  # also ITU-R BT1361 / IEC 61966-2-4 / SMPTE RP177 Annex B
    BT470M = 4  # also FCC Title 47 Code of Federal Regulations 73.682 (a)(20)

    # also ITU-R BT601-6 625 / ITU-R BT1358 625 / ITU-R BT1700 625 PAL & SECAM
    BT470BG = 5

    SMPTE170M = (
        6  # also ITU-R BT601-6 525 / ITU-R BT1358 525 / ITU-R BT1700 NTSC
    )
    SMPTE240M = 7  # functionally identical to above
    FILM = 8  # colour filters using Illuminant C
    BT2020 = 9  # ITU-R BT2020
    SMPTE428 = 10  # SMPTE ST 428-1 (CIE 1931 XYZ)
    SMPTE431 = 11  # SMPTE ST 431-2 (2011) / DCI P3
    SMPTE432 = 12  # SMPTE ST 432-1 (2010) / P3 D65 / Display P3

    # Note: This corresponds to "AVCOL_PRI_EBU3213" in ffmpeg, but neither ITU-T
    # H.273:2021 nor ISO/IEC 23091-2:2021 contain the same values as EBU Tech. 3213, nor
    # do they refer to it directly (ColourPrimaries=22 contains the remark "No
    # corresponding industry specification identified" in both cases).
    #
    # Since both ISO/IEC 23001-8:2013 and 14497-10:2020 do refer to EBU Tech 3213-E
    # (1975) but contain the same primaries and whitepoint as the 2021 revisions, this
    # is likely a error in the initial standards that was later discovered and
    # corrected.
    UNKNOWN22 = 22


@enum.unique
class TransferCharacteristics_ITU_T_H273(enum.IntEnum):
    """TransferCharacteristics tags as defined in ITU-T H.273 § 8.2.

    The enumeration members use the same names as ffmpeg `AVCOL_TRC_*` constants.
    """

    BT709 = 1  # also ITU-R BT1361
    GAMMA22 = 4  # also ITU-R BT470M / ITU-R BT1700 625 PAL & SECAM
    GAMMA28 = 5  # also ITU-R BT470BG

    # also ITU-R BT601-6 525 or 625 / ITU-R BT1358 525 or 625 / ITU-R BT1700 NTSC
    SMPTE170M = 6

    SMPTE240M = 7
    LINEAR = 8  # "Linear transfer characteristics"
    LOG = 9  # "Logarithmic transfer characteristic (100:1 range)"

    # "Logarithmic transfer characteristic (100 * Sqrt(10) : 1 range)"
    LOG_SQRT = 10

    IEC61966_2_4 = 11  # IEC 61966-2-4
    BT1361_ECG = 12  # ITU-R BT1361 Extended Colour Gamut
    IEC61966_2_1 = 13  # IEC 61966-2-1 (sRGB or sYCC)
    BT2020_10 = 14  # ITU-R BT2020 for 10-bit system
    BT2020_12 = 15  # ITU-R BT2020 for 12-bit system
    SMPTE2084 = 16  # SMPTE ST 2084 for 10-, 12-, 14- and 16-bit systems
    SMPTE428 = 17  # SMPTE ST 428-1
    ARIB_STD_B67 = 18  # ARIB STD-B67, known as "Hybrid log-gamma"


@enum.unique
class MatrixCoefficients_ITU_T_H273(enum.IntEnum):
    """MatrixCoefficients tags as defined in ITU-T H.273 § 8.3

    The enumeration members use the same names as ffmpeg `AVCOL_SPC_*` constants.
    """

    RGB = 0  # order of coefficients is actually GBR, also IEC 61966-2-1 (sRGB)
    BT709 = (
        1  # also ITU-R BT1361 / IEC 61966-2-4 xvYCC709 / SMPTE RP177 Annex B
    )
    FCC = 4  # FCC Title 47 Code of Federal Regulations 73.682 (a)(20)

    # also ITU-R BT601-6 625 / ITU-R BT1358 625 / ITU-R BT1700 625 PAL & SECAM
    # / IEC 61966-2-4 xvYCC601
    BT470BG = 5

    SMPTE170M = (
        6  # also ITU-R BT601-6 525 / ITU-R BT1358 525 / ITU-R BT1700 NTSC
    )
    SMPTE240M = 7  # functionally identical to above
    YCGCO = 8  # Used by Dirac / VC-2 and H.264 FRext, see ITU-T SG16
    BT2020_NCL = 9  # ITU-R BT2020 non-constant luminance system
    BT2020_CL = 10  # ITU-R BT2020 constant luminance system
    SMPTE2085 = 11  # SMPTE 2085, Y'D'zD'x
    CHROMA_DERIVED_NCL = (
        12  # Chromaticity-derived non-constant luminance system
    )
    CHROMA_DERIVED_CL = 13  # Chromaticity-derived constant luminance system
    ICTCP = 14  # ITU-R BT.2100-0, ICtCp


PRIMARIES_ITU_T_H273 = {
    # 0: Reserved for future use
    # 1: ITU-R BT.709-5, IEC 61966-2-1 sRGB or sYCC, IEC 61966-2-4, SMPTE RP177 Annex B
    ColourPrimaries_ITU_T_H273.BT709: (
        colour.models.rgb.datasets.itur_bt_709.PRIMARIES_BT709
    ),
    # 2: Unspecified
    # 3: Reserved for future use
    # 4: ITU-R BT.470-6 System M
    ColourPrimaries_ITU_T_H273.BT470M: (
        colour.models.rgb.datasets.itur_bt_470.PRIMARIES_BT470_525
    ),
    # 5: ITU-R BT.470-6 Systems BG, ITU-R BT.601-6 625
    ColourPrimaries_ITU_T_H273.BT470BG: (
        colour.models.rgb.datasets.itur_bt_470.PRIMARIES_BT470_625
    ),
    # 6: SMPTE 170M, ITU-R BT.601-6 525 (same as 7)
    ColourPrimaries_ITU_T_H273.SMPTE170M: (
        colour.models.rgb.datasets.smpte_240m.PRIMARIES_SMPTE_240M
    ),
    # 7: SMPTE 240M (same as 6)
    ColourPrimaries_ITU_T_H273.SMPTE240M: (
        colour.models.rgb.datasets.smpte_240m.PRIMARIES_SMPTE_240M
    ),
    # 8: Generic film (colour filters using Illuminant C)
    ColourPrimaries_ITU_T_H273.FILM: (
        colour.models.rgb.datasets.itut_h_273.PRIMARIES_FILM_C
    ),
    # 9: ITU-R BT.2020
    ColourPrimaries_ITU_T_H273.BT2020: (
        colour.models.rgb.datasets.itur_bt_2020.PRIMARIES_BT2020
    ),
    # 10: SMPTE ST 428-1 (DCDM, CIE 1931 XYZ as in ISO 11664-1)
    ColourPrimaries_ITU_T_H273.SMPTE428: (
        colour.models.rgb.datasets.dcdm_xyz.PRIMARIES_DCDM_XYZ
    ),
    # 11: SMPTE RP 431-2 (2011)
    ColourPrimaries_ITU_T_H273.SMPTE431: (
        colour.models.rgb.datasets.dci_p3.PRIMARIES_DCI_P3
    ),
    # 12: SMPTE EG 432-1 (2010)
    ColourPrimaries_ITU_T_H273.SMPTE432: (
        colour.models.rgb.datasets.p3_d65.PRIMARIES_P3_D65
    ),
    # 13-21: Reserved for future use
    # 22: No corresponding industry specification
    ColourPrimaries_ITU_T_H273.UNKNOWN22: (
        colour.models.rgb.datasets.itut_h_273.PRIMARIES_ITUT_H273_22
    ),
    # 23-255: Reserved for future use
}
"""ColourPrimaries tag to colour primaries mapping defined in ITU-T H.273 §
8.1"""


CCS_WHITEPOINTS_ITU_T_H273 = {
    # 0: Reserved for future use
    # 1: ITU-R BT.709-5, IEC 61966-2-1 sRGB or sYCC, IEC 61966-2-4, SMPTE RP177 Annex B
    ColourPrimaries_ITU_T_H273.BT709: (
        colour.models.rgb.datasets.itur_bt_709.CCS_WHITEPOINT_BT709
    ),
    # 2: Unspecified
    # 3: Reserved for future use
    # 4: ITU-R BT.470-6 System M
    ColourPrimaries_ITU_T_H273.BT470M: (
        # Note: ITU-T H.273 defines white point C as [0.310, 0.316], while this
        # has a slightly higher precision.
        colour.models.rgb.datasets.itur_bt_470.CCS_WHITEPOINT_BT470_525
    ),
    # 5: ITU-R BT.470-6 Systems BG, ITU-R BT.601-6 625
    ColourPrimaries_ITU_T_H273.BT470BG: (
        colour.models.rgb.datasets.itur_bt_470.CCS_WHITEPOINT_BT470_625
    ),
    # 6: SMPTE 170M, ITU-R BT.601-6 525 (same as 7)
    ColourPrimaries_ITU_T_H273.SMPTE170M: (
        colour.models.rgb.datasets.smpte_240m.CCS_WHITEPOINT_SMPTE_240M
    ),
    # 7: SMPTE 240M (same as 6)
    ColourPrimaries_ITU_T_H273.SMPTE240M: (
        colour.models.rgb.datasets.smpte_240m.CCS_WHITEPOINT_SMPTE_240M
    ),
    # 8: Generic film (colour filters using Illuminant C)
    ColourPrimaries_ITU_T_H273.FILM: (
        colour.models.rgb.datasets.itut_h_273.CCS_WHITEPOINT_FILM_C
    ),
    # 9: ITU-R BT.2020
    ColourPrimaries_ITU_T_H273.BT2020: (
        colour.models.rgb.datasets.itur_bt_2020.CCS_WHITEPOINT_BT2020
    ),
    # 10: SMPTE ST 428-1 (DCDM, CIE 1931 XYZ as in ISO 11664-1)
    ColourPrimaries_ITU_T_H273.SMPTE428: (
        colour.models.rgb.datasets.dcdm_xyz.CCS_WHITEPOINT_DCDM_XYZ
    ),
    # 11: SMPTE RP 431-2 (2011)
    ColourPrimaries_ITU_T_H273.SMPTE431: (
        colour.models.rgb.datasets.dci_p3.CCS_WHITEPOINT_DCI_P3
    ),
    # 12: SMPTE EG 432-1 (2010)
    ColourPrimaries_ITU_T_H273.SMPTE432: (
        colour.models.rgb.datasets.p3_d65.CCS_WHITEPOINT_P3_D65
    ),
    # 13-21: Reserved for future use
    # 22: No corresponding industry specification
    ColourPrimaries_ITU_T_H273.UNKNOWN22: (
        colour.models.rgb.datasets.itut_h_273.CCS_WHITEPOINT_ITUT_H273_22
    ),
    # 23-255: Reserved for future use
}
"""ColourPrimaries tag to whitepoint chromaticity coordinates mapping defined
in ITU-T H.273 § 8.1"""


WHITEPOINTS_NAMES_ITU_T_H273 = {
    # 0: Reserved for future use
    # 1: ITU-R BT.709-5, IEC 61966-2-1 sRGB or sYCC, IEC 61966-2-4, SMPTE RP177 Annex B
    ColourPrimaries_ITU_T_H273.BT709: (
        colour.models.rgb.datasets.itur_bt_709.WHITEPOINT_NAME_BT709
    ),
    # 2: Unspecified
    # 3: Reserved for future use
    # 4: ITU-R BT.470-6 System M
    ColourPrimaries_ITU_T_H273.BT470M: (
        colour.models.rgb.datasets.itur_bt_470.WHITEPOINT_NAME_BT470_525
    ),
    # 5: ITU-R BT.470-6 Systems BG, ITU-R BT.601-6 625
    ColourPrimaries_ITU_T_H273.BT470BG: (
        colour.models.rgb.datasets.itur_bt_470.WHITEPOINT_NAME_BT470_625
    ),
    # 6: SMPTE 170M, ITU-R BT.601-6 525 (same as 7)
    ColourPrimaries_ITU_T_H273.SMPTE170M: (
        colour.models.rgb.datasets.smpte_240m.WHITEPOINT_NAME_SMPTE_240M
    ),
    # 7: SMPTE 240M (same as 6)
    ColourPrimaries_ITU_T_H273.SMPTE240M: (
        colour.models.rgb.datasets.smpte_240m.WHITEPOINT_NAME_SMPTE_240M
    ),
    # 8: Generic film (colour filters using Illuminant C)
    ColourPrimaries_ITU_T_H273.FILM: (
        colour.models.rgb.datasets.itut_h_273.WHITEPOINT_NAME_FILM_C
    ),
    # 9: ITU-R BT.2020
    ColourPrimaries_ITU_T_H273.BT2020: (
        colour.models.rgb.datasets.itur_bt_2020.WHITEPOINT_NAME_BT2020
    ),
    # 10: SMPTE ST 428-1 (DCDM, CIE 1931 XYZ as in ISO 11664-1)
    ColourPrimaries_ITU_T_H273.SMPTE428: (
        colour.models.rgb.datasets.dcdm_xyz.WHITEPOINT_NAME_DCDM_XYZ
    ),
    # 11: SMPTE RP 431-2 (2011)
    ColourPrimaries_ITU_T_H273.SMPTE431: (
        colour.models.rgb.datasets.dci_p3.WHITEPOINT_NAME_DCI_P3
    ),
    # 12: SMPTE EG 432-1 (2010)
    ColourPrimaries_ITU_T_H273.SMPTE432: (
        colour.models.rgb.datasets.p3_d65.WHITEPOINT_NAME_P3_D65
    ),
    # 13-21: Reserved for future use
    # 22: No corresponding industry specification
    ColourPrimaries_ITU_T_H273.UNKNOWN22: (
        colour.models.rgb.datasets.itut_h_273.WHITEPOINT_NAME_ITUT_H273_22
    ),
    # 23-255: Reserved for future use
}
"""ColourPrimaries tag to whitepoint names mapping defined in ITU-T H.273 §
8.1"""


MATRICES_ITU_T_H273_RGB_TO_XYZ = {
    # 0: Reserved for future use
    # 1: ITU-R BT.709-5, IEC 61966-2-1 sRGB or sYCC, IEC 61966-2-4, SMPTE RP177 Annex B
    ColourPrimaries_ITU_T_H273.BT709: (
        colour.models.rgb.datasets.itur_bt_709.MATRIX_BT709_TO_XYZ
    ),
    # 2: Unspecified
    # 3: Reserved for future use
    # 4: ITU-R BT.470-6 System M
    ColourPrimaries_ITU_T_H273.BT470M: (
        colour.models.rgb.datasets.itur_bt_470.MATRIX_BT470_525_TO_XYZ
    ),
    # 5: ITU-R BT.470-6 Systems BG, ITU-R BT.601-6 625
    ColourPrimaries_ITU_T_H273.BT470BG: (
        colour.models.rgb.datasets.itur_bt_470.MATRIX_BT470_625_TO_XYZ
    ),
    # 6: SMPTE 170M, ITU-R BT.601-6 525 (same as 7)
    ColourPrimaries_ITU_T_H273.SMPTE170M: (
        colour.models.rgb.datasets.smpte_240m.MATRIX_SMPTE_240M_TO_XYZ
    ),
    # 7: SMPTE 240M (same as 6)
    ColourPrimaries_ITU_T_H273.SMPTE240M: (
        colour.models.rgb.datasets.smpte_240m.MATRIX_SMPTE_240M_TO_XYZ
    ),
    # 8: Generic film (colour filters using Illuminant C)
    ColourPrimaries_ITU_T_H273.FILM: (
        colour.models.rgb.datasets.itut_h_273.MATRIX_FILM_C_RGB_TO_XYZ
    ),
    # 9: ITU-R BT.2020
    ColourPrimaries_ITU_T_H273.BT2020: (
        colour.models.rgb.datasets.itur_bt_2020.MATRIX_BT2020_TO_XYZ
    ),
    # 10: SMPTE ST 428-1 (DCDM, CIE 1931 XYZ as in ISO 11664-1)
    ColourPrimaries_ITU_T_H273.SMPTE428: (
        colour.models.rgb.datasets.dcdm_xyz.MATRIX_DCDM_XYZ_TO_XYZ
    ),
    # 11: SMPTE RP 431-2 (2011)
    ColourPrimaries_ITU_T_H273.SMPTE431: (
        colour.models.rgb.datasets.dci_p3.MATRIX_DCI_P3_TO_XYZ
    ),
    # 12: SMPTE EG 432-1 (2010)
    ColourPrimaries_ITU_T_H273.SMPTE432: (
        colour.models.rgb.datasets.p3_d65.MATRIX_P3_D65_TO_XYZ
    ),
    # 13-21: Reserved for future use
    # 22: No corresponding industry specification
    ColourPrimaries_ITU_T_H273.UNKNOWN22: (
        colour.models.rgb.datasets.itut_h_273.MATRIX_ITUT_H273_22_RGB_TO_XYZ
    ),
    # 23-255: Reserved for future use
}
"""ColourPrimaries tag to RGB to XYZ matrices determined from primaries and
whitepoints defined in ITU-T H.273 § 8.1"""


MATRICES_XYZ_TO_ITU_T_H273_RGB = {
    # 0: Reserved for future use
    # 1: ITU-R BT.709-5, IEC 61966-2-1 sRGB or sYCC, IEC 61966-2-4, SMPTE RP177 Annex B
    ColourPrimaries_ITU_T_H273.BT709: (
        colour.models.rgb.datasets.itur_bt_709.MATRIX_XYZ_TO_BT709
    ),
    # 2: Unspecified
    # 3: Reserved for future use
    # 4: ITU-R BT.470-6 System M
    ColourPrimaries_ITU_T_H273.BT470M: (
        colour.models.rgb.datasets.itur_bt_470.MATRIX_XYZ_TO_BT470_525
    ),
    # 5: ITU-R BT.470-6 Systems BG, ITU-R BT.601-6 625
    ColourPrimaries_ITU_T_H273.BT470BG: (
        colour.models.rgb.datasets.itur_bt_470.MATRIX_XYZ_TO_BT470_625
    ),
    # 6: SMPTE 170M, ITU-R BT.601-6 525 (same as 7)
    ColourPrimaries_ITU_T_H273.SMPTE170M: (
        colour.models.rgb.datasets.smpte_240m.MATRIX_XYZ_TO_SMPTE_240M
    ),
    # 7: SMPTE 240M (same as 6)
    ColourPrimaries_ITU_T_H273.SMPTE240M: (
        colour.models.rgb.datasets.smpte_240m.MATRIX_XYZ_TO_SMPTE_240M
    ),
    # 8: Generic film (colour filters using Illuminant C)
    ColourPrimaries_ITU_T_H273.FILM: (
        colour.models.rgb.datasets.itut_h_273.MATRIX_XYZ_TO_FILM_C_RGB
    ),
    # 9: ITU-R BT.2020
    ColourPrimaries_ITU_T_H273.BT2020: (
        colour.models.rgb.datasets.itur_bt_2020.MATRIX_XYZ_TO_BT2020
    ),
    # 10: SMPTE ST 428-1 (DCDM, CIE 1931 XYZ as in ISO 11664-1)
    ColourPrimaries_ITU_T_H273.SMPTE428: (
        colour.models.rgb.datasets.dcdm_xyz.MATRIX_XYZ_TO_DCDM_XYZ
    ),
    # 11: SMPTE RP 431-2 (2011)
    ColourPrimaries_ITU_T_H273.SMPTE431: (
        colour.models.rgb.datasets.dci_p3.MATRIX_XYZ_TO_DCI_P3
    ),
    # 12: SMPTE EG 432-1 (2010)
    ColourPrimaries_ITU_T_H273.SMPTE432: (
        colour.models.rgb.datasets.p3_d65.MATRIX_XYZ_TO_P3_D65
    ),
    # 13-21: Reserved for future use
    # 22: No corresponding industry specification
    ColourPrimaries_ITU_T_H273.UNKNOWN22: (
        colour.models.rgb.datasets.itut_h_273.MATRIX_XYZ_TO_ITUT_H273_22_RGB
    ),
    # 23-255: Reserved for future use
}
"""ColourPrimaries tag to XYZ to RGB matrices determined from primaries and
whitepoints defined in ITU-T H.273 § 8.1"""


TRANSFER_FUNCTIONS_ITU_T_H273 = {
    # 0: Reserved for future use
    # 1: ITU-R BT.709
    TransferCharacteristics_ITU_T_H273.BT709: oetf_BT709,
    # 2: Unspecified
    # 3: Reserved for future use
    # 4: Gamma 2.2 (also ITU-R BT470M / ITU-R BT1700 625 PAL & SECAM) (this is
    # an inverse-EOTF)
    TransferCharacteristics_ITU_T_H273.GAMMA22: eotf_inverse_gamma22,
    # 5: Gamma 2.8 (also ITU-R BT470BG) (this is an inverse-EOTF)
    TransferCharacteristics_ITU_T_H273.GAMMA28: eotf_inverse_gamma28,
    # 6: SMPTE 170M (also ITU-R BT601-6 525 or 625 / ITU-R BT1358 525 or 625 /
    # ITU-R BT1700 NTSC)
    TransferCharacteristics_ITU_T_H273.SMPTE170M: oetf_BT601,
    # 7: SMPTE 240M
    TransferCharacteristics_ITU_T_H273.SMPTE240M: oetf_SMPTE240M,
    # 8: Linear transfer characteristics (this is an OETF)
    TransferCharacteristics_ITU_T_H273.LINEAR: oetf_linear,
    # 9: Logarithmic transfer characteristic (100:1 range)
    TransferCharacteristics_ITU_T_H273.LOG: oetf_log,
    # 10: Logarithmic transfer characteristic (100 * Sqrt(10) : 1 range)
    TransferCharacteristics_ITU_T_H273.LOG_SQRT: oetf_log_sqrt,
    # 11: IEC 61966-2-4
    TransferCharacteristics_ITU_T_H273.IEC61966_2_4: oetf_iec_61966_2_4,
    # 12: ITU-R BT1361 Extended Colour Gamut
    TransferCharacteristics_ITU_T_H273.BT1361_ECG: oetf_BT1361_extended,
    # 13: IEC 61966-2-1 (sRGB or sYCC)
    TransferCharacteristics_ITU_T_H273.IEC61966_2_1: oetf_iec_61966_2_1,
    # 14: ITU-R BT2020 for 10-bit system
    TransferCharacteristics_ITU_T_H273.BT2020_10: oetf_BT2020_10bit,
    # 15: ITU-R BT2020 for 12-bit system
    TransferCharacteristics_ITU_T_H273.BT2020_12: oetf_BT2020_12bit,
    # 16: SMPTE ST 2084 (PQ, Perceptual Quantizer) for 10-, 12-, 14- and 16-bit systems
    TransferCharacteristics_ITU_T_H273.SMPTE2084: eotf_inverse_ST2084,
    # 17: SMPTE ST 428-1
    TransferCharacteristics_ITU_T_H273.SMPTE428: eotf_inverse_ST428_1,
    # 18: ARIB STD B67, also ITU-R BT.2100 HLG (Hybrid Log-Gamma), in [0-1] range
    TransferCharacteristics_ITU_T_H273.ARIB_STD_B67: oetf_BT2100_HLG,
}
"""Mapping from TransferCharacteristics tag to transfer functions defined in
ITU-T H.273 § 8.2

Note that the standard contains both OETFs or inverse-EOTFs."""


WEIGHTS_YCBCR_ITU_T_H273 = {}
"""K_R and K_B coefficients for YCbCr conversion defined in ITU-T H.273 §8.3

Notes
-----

Several values of MatrixCoefficients don't directly correspond to K_R and
K_B values but instead have separate equations, which are not included in
this mapping:

- MatrixCoefficients_ITU_T_H273.YCGCO
- MatrixCoefficients_ITU_T_H273.SMPTE2085
- MatrixCoefficients_ITU_T_H273.CHROMA_DERIVED_NCL
- MatrixCoefficients_ITU_T_H273.CHROMA_DERIVED_CL
- MatrixCoefficients_ITU_T_H273.ICTCP
"""

WEIGHTS_YCBCR_ITU_T_H273[MatrixCoefficients_ITU_T_H273.RGB] = np.array(
    [1.0, 1.0]
)
WEIGHTS_YCBCR_ITU_T_H273[
    MatrixCoefficients_ITU_T_H273.BT709
] = colour.models.rgb.ycbcr.WEIGHTS_YCBCR["ITU-R BT.709"]
WEIGHTS_YCBCR_ITU_T_H273[MatrixCoefficients_ITU_T_H273.FCC] = np.array(
    [0.30, 0.11]
)
WEIGHTS_YCBCR_ITU_T_H273[
    MatrixCoefficients_ITU_T_H273.BT470BG
] = colour.models.rgb.ycbcr.WEIGHTS_YCBCR["ITU-R BT.601"]
WEIGHTS_YCBCR_ITU_T_H273[
    MatrixCoefficients_ITU_T_H273.SMPTE170M
] = colour.models.rgb.ycbcr.WEIGHTS_YCBCR["ITU-R BT.601"]
# Note: The precision is different from WEIGHTS_YCBCR["ITU-R SMPTE-240M"]
WEIGHTS_YCBCR_ITU_T_H273[MatrixCoefficients_ITU_T_H273.SMPTE240M] = np.array(
    [0.212, 0.087]
)
WEIGHTS_YCBCR_ITU_T_H273[
    MatrixCoefficients_ITU_T_H273.BT2020_NCL
] = colour.models.rgb.ycbcr.WEIGHTS_YCBCR["ITU-R BT.2020"]
WEIGHTS_YCBCR_ITU_T_H273[
    MatrixCoefficients_ITU_T_H273.BT2020_CL
] = colour.models.rgb.ycbcr.WEIGHTS_YCBCR["ITU-R BT.2020"]


# Aliases for ISO/IEC 23091-2:2021. Verified to be functionally identical to ITU-T H.273
# for the values defined here.
ColourPrimaries_ISO23091_2 = ColourPrimaries_ITU_T_H273
TransferCharacteristics_ISO23091_2 = TransferCharacteristics_ITU_T_H273
MatrixCoefficients_ISO23091_2 = MatrixCoefficients_ITU_T_H273
PRIMARIES_ISO23091_2 = PRIMARIES_ITU_T_H273
WHITEPOINTS_NAMES_ISO23091_2 = WHITEPOINTS_NAMES_ITU_T_H273
CCS_WHITEPOINTS_ISO23091_2 = CCS_WHITEPOINTS_ITU_T_H273
MATRICES_ISO23091_2_RGB_TO_XYZ = MATRICES_ITU_T_H273_RGB_TO_XYZ
MATRICES_XYZ_TO_ISO23091_2_RGB = MATRICES_XYZ_TO_ITU_T_H273_RGB
TRANSFER_FUNCTIONS_ISO23091_2 = TRANSFER_FUNCTIONS_ITU_T_H273
WEIGHTS_YCBCR_ISO23091_2 = WEIGHTS_YCBCR_ITU_T_H273


@enum.unique
class ColourPrimaries_ISO23001_8(enum.IntEnum):
    """ColourPrimaries tags as defined in ISO/IEC 23001-8:2013 § 7.1.

    The enumeration members use  the same names as ffmpeg `AVCOL_PRI_*` constants.
    """

    BT709 = 1  # also ITU-R BT1361 / IEC 61966-2-4 / SMPTE RP177 Annex B
    BT470M = 4  # also FCC Title 47 Code of Federal Regulations 73.682 (a)(20)

    # also ITU-R BT601-6 625 / ITU-R BT1358 625 / ITU-R BT1700 625 PAL & SECAM
    BT470BG = 5

    SMPTE170M = (
        6  # also ITU-R BT601-6 525 / ITU-R BT1358 525 / ITU-R BT1700 NTSC
    )
    SMPTE240M = 7  # functionally identical to above
    FILM = 8  # colour filters using Illuminant C
    BT2020 = 9  # ITU-R BT2020
    SMPTE428 = 10  # SMPTE ST 428-1 (CIE 1931 XYZ)
    SMPTE431 = 11  # SMPTE ST 431-2 (2011) / DCI P3
    SMPTE432 = 12  # SMPTE ST 432-1 (2010) / P3 D65 / Display P3

    # Note: This corresponds to "AVCOL_PRI_EBU3213" in ffmpeg, but neither ITU-T
    # H.273:2021 nor ISO/IEC 23091-2:2021 contain the same values as EBU Tech. 3213, nor
    # do they refer to it directly (ColourPrimaries=22 contains the remark "No
    # corresponding industry specification identified" in both cases).
    #
    # Since both ISO/IEC 23001-8:2013 and 14497-10:2020 do refer to EBU Tech 3213-E
    # (1975) but contain the same primaries and whitepoint as the 2021 revisions, this
    # is likely a error in the initial standards that was later discovered and
    # corrected.
    EBU3213 = 22


# Aliases for ISO/IEC 23001-8:2013. Verified to be functionally identical to ITU-T H.273
# for the values defined here, except for the remark regarding EBU Tech. 3213-E.
TransferCharacteristics_ISO23001_8 = TransferCharacteristics_ITU_T_H273
MatrixCoefficients_ISO23001_8 = MatrixCoefficients_ITU_T_H273
PRIMARIES_ISO23001_8 = PRIMARIES_ITU_T_H273
WHITEPOINTS_NAMES_ISO23001_8 = WHITEPOINTS_NAMES_ITU_T_H273
CCS_WHITEPOINTS_ISO23001_8 = CCS_WHITEPOINTS_ITU_T_H273
MATRICES_ISO23001_8_RGB_TO_XYZ = MATRICES_ITU_T_H273_RGB_TO_XYZ
MATRICES_XYZ_TO_ISO23001_8_RGB = MATRICES_XYZ_TO_ITU_T_H273_RGB
TRANSFER_FUNCTIONS_ISO23001_8 = TRANSFER_FUNCTIONS_ITU_T_H273
WEIGHTS_YCBCR_ISO23001_8 = WEIGHTS_YCBCR_ITU_T_H273

# Aliases for ISO/IEC 14496-10:2020. Verified to be functionally identical to ITU-T
# H.273 for the values defined here, except for the remark regarding EBU Tech. 3213-E.
ColourPrimaries_ISO14496_10 = ColourPrimaries_ISO23001_8
TransferCharacteristics_ISO14496_10 = TransferCharacteristics_ITU_T_H273
MatrixCoefficients_ISO14496_10 = MatrixCoefficients_ITU_T_H273
PRIMARIES_ISO14496_10 = PRIMARIES_ITU_T_H273
WHITEPOINTS_NAMES_ISO14496_10 = WHITEPOINTS_NAMES_ITU_T_H273
CCS_WHITEPOINTS_ISO14496_10 = CCS_WHITEPOINTS_ITU_T_H273
MATRICES_ISO14496_10_RGB_TO_XYZ = MATRICES_ITU_T_H273_RGB_TO_XYZ
MATRICES_XYZ_TO_ISO14496_10_RGB = MATRICES_XYZ_TO_ITU_T_H273_RGB
TRANSFER_FUNCTIONS_ISO14496_10 = TRANSFER_FUNCTIONS_ITU_T_H273
WEIGHTS_YCBCR_ISO14496_10 = WEIGHTS_YCBCR_ITU_T_H273
