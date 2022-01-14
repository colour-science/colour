# -*- coding: utf-8 -*-
"""
Academy Color Encoding System
=============================

Defines the *Academy Color Encoding System* (ACES) related encodings:

-   :attr:`colour.models.RGB_COLOURSPACE_ACES2065_1`
-   :attr:`colour.models.RGB_COLOURSPACE_ACESCG`
-   :attr:`colour.models.RGB_COLOURSPACE_ACESCC`
-   :attr:`colour.models.RGB_COLOURSPACE_ACESCCT`
-   :attr:`colour.models.RGB_COLOURSPACE_ACESPROXY`

References
----------
-   :cite:`TheAcademyofMotionPictureArtsandSciences2014q` : The Academy of
    Motion Picture Arts and Sciences, Science and Technology Council, & Academy
    Color Encoding System (ACES) Project Subcommittee. (2014). Technical
    Bulletin TB-2014-004 - Informative Notes on SMPTE ST 2065-1 - Academy Color
    Encoding Specification (ACES) (pp. 1-40). Retrieved December 19, 2014, from
    http://j.mp/TB-2014-004
-   :cite:`TheAcademyofMotionPictureArtsandSciences2014r` : The Academy of
    Motion Picture Arts and Sciences, Science and Technology Council, & Academy
    Color Encoding System (ACES) Project Subcommittee. (2014). Technical
    Bulletin TB-2014-012 - Academy Color Encoding System Version 1.0 Component
    Names (pp. 1-8). Retrieved December 19, 2014, from http://j.mp/TB-2014-012
-   :cite:`TheAcademyofMotionPictureArtsandSciences2014s` : The Academy of
    Motion Picture Arts and Sciences, Science and Technology Council, & Academy
    Color Encoding System (ACES) Project Subcommittee. (2013). Specification
    S-2013-001 - ACESproxy, an Integer Log Encoding of ACES Image Data.
    Retrieved December 19, 2014, from http://j.mp/S-2013-001
-   :cite:`TheAcademyofMotionPictureArtsandSciences2014t` : The Academy of
    Motion Picture Arts and Sciences, Science and Technology Council, & Academy
    Color Encoding System (ACES) Project Subcommittee. (2014). Specification
    S-2014-003 - ACEScc, A Logarithmic Encoding of ACES Data for use within
    Color Grading Systems (pp. 1-12). Retrieved December 19, 2014, from
    http://j.mp/S-2014-003
-   :cite:`TheAcademyofMotionPictureArtsandSciences2015b` : The Academy of
    Motion Picture Arts and Sciences, Science and Technology Council, & Academy
    Color Encoding System (ACES) Project Subcommittee. (2015). Specification
    S-2014-004 - ACEScg - A Working Space for CGI Render and Compositing
    (pp. 1-9). Retrieved April 24, 2015, from http://j.mp/S-2014-004
-   :cite:`TheAcademyofMotionPictureArtsandSciences2016c` : The Academy of
    Motion Picture Arts and Sciences, Science and Technology Council, & Academy
    Color Encoding System (ACES) Project. (2016). Specification S-2016-001 -
    ACEScct, A Quasi-Logarithmic Encoding of ACES Data for use within Color
    Grading Systems. Retrieved October 10, 2016, from http://j.mp/S-2016-001
-   :cite:`TheAcademyofMotionPictureArtsandSciencese` : The Academy of Motion
    Picture Arts and Sciences, Science and Technology Council, & Academy Color
    Encoding System (ACES) Project Subcommittee. (n.d.). Academy Color Encoding
    System. Retrieved February 24, 2014, from
    http://www.oscars.org/science-technology/council/projects/aces.html
"""

from __future__ import annotations

import numpy as np

from colour.colorimetry import CCS_ILLUMINANTS
from colour.hints import NDArray
from colour.models.rgb import (
    RGB_Colourspace,
    linear_function,
    normalised_primary_matrix,
    log_encoding_ACEScc,
    log_decoding_ACEScc,
    log_encoding_ACEScct,
    log_decoding_ACEScct,
    log_encoding_ACESproxy,
    log_decoding_ACESproxy,
)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'AP0',
    'AP1',
    'WHITEPOINT_NAME_ACES',
    'CCS_WHITEPOINT_ACES',
    'MATRIX_AP0_TO_XYZ',
    'MATRIX_XYZ_TO_AP0',
    'MATRIX_AP1_TO_XYZ',
    'MATRIX_XYZ_TO_AP1',
    'RGB_COLOURSPACE_ACES2065_1',
    'RGB_COLOURSPACE_ACESCG',
    'RGB_COLOURSPACE_ACESCC',
    'RGB_COLOURSPACE_ACESCCT',
    'RGB_COLOURSPACE_ACESPROXY',
]

AP0: NDArray = np.array([
    [0.73470, 0.26530],
    [0.00000, 1.00000],
    [0.00010, -0.07700],
])
"""
*ACES Primaries 0* or *AP0* primaries.
"""

AP1: NDArray = np.array([
    [0.71300, 0.29300],
    [0.16500, 0.83000],
    [0.12800, 0.04400],
])
"""
*ACES Primaries 1* or *AP1* primaries (known as *ITU-R BT.2020+* primaries
prior to *ACES* 1.0 release).
"""

WHITEPOINT_NAME_ACES: str = 'ACES'
"""
*ACES2065-1* colourspace whitepoint name.
"""

CCS_WHITEPOINT_ACES: NDArray = (CCS_ILLUMINANTS[
    'CIE 1931 2 Degree Standard Observer'][WHITEPOINT_NAME_ACES])
"""
*ACES2065-1* colourspace whitepoint chromaticity coordinates.
"""

MATRIX_AP0_TO_XYZ: NDArray = np.array([
    [0.9525523959, 0.0000000000, 0.0000936786],
    [0.3439664498, 0.7281660966, -0.0721325464],
    [0.0000000000, 0.0000000000, 1.0088251844],
])
"""
*ACES Primaries 0* to *CIE XYZ* tristimulus values matrix defined as per [2].
"""

MATRIX_XYZ_TO_AP0: NDArray = np.array([
    [1.0498110175, 0.0000000000, -0.0000974845],
    [-0.4959030231, 1.3733130458, 0.0982400361],
    [0.0000000000, 0.0000000000, 0.9912520182],
])
"""
*CIE XYZ* tristimulus values to *ACES Primaries 0* matrix.
"""

MATRIX_AP1_TO_XYZ: NDArray = normalised_primary_matrix(AP1,
                                                       CCS_WHITEPOINT_ACES)
"""
*ACES Primaries 1* to *CIE XYZ* tristimulus values matrix.
"""

MATRIX_XYZ_TO_AP1: NDArray = np.linalg.inv(MATRIX_AP1_TO_XYZ)
"""
*CIE XYZ* tristimulus values to *ACES Primaries 1* matrix.
"""

RGB_COLOURSPACE_ACES2065_1: RGB_Colourspace = RGB_Colourspace(
    'ACES2065-1',
    AP0,
    CCS_WHITEPOINT_ACES,
    WHITEPOINT_NAME_ACES,
    MATRIX_AP0_TO_XYZ,
    MATRIX_XYZ_TO_AP0,
    linear_function,
    linear_function,
)
RGB_COLOURSPACE_ACES2065_1.__doc__ = """
*ACES2065-1* colourspace, base encoding, used for exchange of full fidelity
images and archiving.

References
----------
:cite:`TheAcademyofMotionPictureArtsandSciences2014q`,
:cite:`TheAcademyofMotionPictureArtsandSciences2014r`,
:cite:`TheAcademyofMotionPictureArtsandSciencese`
"""

RGB_COLOURSPACE_ACESCG: RGB_Colourspace = RGB_Colourspace(
    'ACEScg',
    AP1,
    CCS_WHITEPOINT_ACES,
    WHITEPOINT_NAME_ACES,
    MATRIX_AP1_TO_XYZ,
    MATRIX_XYZ_TO_AP1,
    linear_function,
    linear_function,
)
RGB_COLOURSPACE_ACESCG.__doc__ = """
*ACEScg* colourspace, a working space for paint/compositor applications that
don't support ACES2065-1 or ACEScc.

References
----------
:cite:`TheAcademyofMotionPictureArtsandSciences2014q`,
:cite:`TheAcademyofMotionPictureArtsandSciences2014r`,
:cite:`TheAcademyofMotionPictureArtsandSciences2015b`,
:cite:`TheAcademyofMotionPictureArtsandSciencese`
"""

RGB_COLOURSPACE_ACESCC: RGB_Colourspace = RGB_Colourspace(
    'ACEScc',
    AP1,
    CCS_WHITEPOINT_ACES,
    WHITEPOINT_NAME_ACES,
    MATRIX_AP1_TO_XYZ,
    MATRIX_XYZ_TO_AP1,
    log_encoding_ACEScc,
    log_decoding_ACEScc,
)
RGB_COLOURSPACE_ACESCC.__doc__ = """
*ACEScc* colourspace, a working space for color correctors, target for ASC-CDL
values created on-set.

References
----------
:cite:`TheAcademyofMotionPictureArtsandSciences2014q`,
:cite:`TheAcademyofMotionPictureArtsandSciences2014r`,
:cite:`TheAcademyofMotionPictureArtsandSciences2014t`,
:cite:`TheAcademyofMotionPictureArtsandSciencese`
"""

RGB_COLOURSPACE_ACESCCT: RGB_Colourspace = RGB_Colourspace(
    'ACEScct',
    AP1,
    CCS_WHITEPOINT_ACES,
    WHITEPOINT_NAME_ACES,
    MATRIX_AP1_TO_XYZ,
    MATRIX_XYZ_TO_AP1,
    log_encoding_ACEScct,
    log_decoding_ACEScct,
)
RGB_COLOURSPACE_ACESCCT.__doc__ = """
*ACEScct* colourspace, an alternative working space for colour correctors,
intended to be transient and internal to software or hardware systems,
and is specifically not intended for interchange or archiving.

References
----------
:cite:`TheAcademyofMotionPictureArtsandSciences2014q`,
:cite:`TheAcademyofMotionPictureArtsandSciences2014r`,
:cite:`TheAcademyofMotionPictureArtsandSciences2016c`,
:cite:`TheAcademyofMotionPictureArtsandSciencese`
"""

RGB_COLOURSPACE_ACESPROXY: RGB_Colourspace = RGB_Colourspace(
    'ACESproxy',
    AP1,
    CCS_WHITEPOINT_ACES,
    WHITEPOINT_NAME_ACES,
    MATRIX_AP1_TO_XYZ,
    MATRIX_XYZ_TO_AP1,
    log_encoding_ACESproxy,
    log_decoding_ACESproxy,
)
RGB_COLOURSPACE_ACESPROXY.__doc__ = """
*ACESproxy* colourspace, a lightweight encoding for transmission over HD-SDI
(or other production transmission schemes), onset look management. Not
intended to be stored or used in production imagery or for final colour
grading / mastering.

References
----------
:cite:`TheAcademyofMotionPictureArtsandSciences2014q`,
:cite:`TheAcademyofMotionPictureArtsandSciences2014r`,
:cite:`TheAcademyofMotionPictureArtsandSciences2014s`,
:cite:`TheAcademyofMotionPictureArtsandSciencese`
"""
