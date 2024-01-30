"""
Recommendation ITU-T H.273 Colour Primaries (and Colourspaces)
==============================================================

Defines the *Recommendation ITU-T H.273* colourspaces that do not belong in
another specification or standard, or have been modified for inclusion:

-   :attr:`colour.models.RGB_COLOURSPACE_H273_GENERIC_FILM`.
-   :attr:`colour.models.RGB_COLOURSPACE_H273_22_UNSPECIFIED`.

References
----------
-   :cite:`InternationalTelecommunicationUnion2021` : International
    Telecommunication Union. (2021). Recommendation ITU-T H.273 -
    Coding-independent code points for video signal type identification.
    https://www.itu.int/rec/T-REC-H.273-202107-I/en
"""

from __future__ import annotations

import numpy as np

from colour.hints import NDArrayFloat
from colour.models.rgb import (
    RGB_Colourspace,
    linear_function,
    normalised_primary_matrix,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "PRIMARIES_H273_GENERIC_FILM",
    "WHITEPOINT_NAME_H273_GENERIC_FILM",
    "CCS_WHITEPOINT_H273_GENERIC_FILM",
    "MATRIX_H273_GENERIC_FILM_RGB_TO_XYZ",
    "MATRIX_XYZ_TO_H273_GENERIC_FILM_RGB",
    "RGB_COLOURSPACE_H273_GENERIC_FILM",
    "PRIMARIES_H273_22_UNSPECIFIED",
    "WHITEPOINT_NAME_H273_22_UNSPECIFIED",
    "CCS_WHITEPOINT_H273_22_UNSPECIFIED",
    "MATRIX_H273_22_UNSPECIFIED_RGB_TO_XYZ",
    "MATRIX_XYZ_TO_H273_22_UNSPECIFIED_RGB",
    "RGB_COLOURSPACE_H273_22_UNSPECIFIED",
]

PRIMARIES_H273_GENERIC_FILM: NDArrayFloat = np.array(
    [
        [0.681, 0.319],  # Wratten 25
        [0.243, 0.692],  # Wratten 58
        [0.145, 0.049],  # Wratten 47
    ]
)
"""
Colourspace primaries for *Generic Film* (colour filters using Illuminant C).

References
----------
- :cite:`InternationalTelecommunicationUnion2021`
"""

WHITEPOINT_NAME_H273_GENERIC_FILM: str = "C"
"""
Whitepoint name for *Generic Film* (colour filters using Illuminant C).

References
----------
- :cite:`InternationalTelecommunicationUnion2021`
"""

CCS_WHITEPOINT_H273_GENERIC_FILM: NDArrayFloat = np.array([0.310, 0.316])
"""
Whitepoint chromaticity coordinates for *Generic Film* (colour filters using
Illuminant C).

Notes
-----

-   *Recommendation ITU-T H.273* defines whitepoint *C* as [0.310, 0.316],
    while *Colour* has a slightly higher precision.

References
----------
- :cite:`InternationalTelecommunicationUnion2021`
"""

MATRIX_H273_GENERIC_FILM_RGB_TO_XYZ: NDArrayFloat = normalised_primary_matrix(
    PRIMARIES_H273_GENERIC_FILM, CCS_WHITEPOINT_H273_GENERIC_FILM
)
"""
*Generic Film* (colour filters using Illuminant C) colourspace to *CIE XYZ*
tristimulus values matrix.

References
----------
- :cite:`InternationalTelecommunicationUnion2021`
"""

MATRIX_XYZ_TO_H273_GENERIC_FILM_RGB: NDArrayFloat = np.linalg.inv(
    MATRIX_H273_GENERIC_FILM_RGB_TO_XYZ
)
"""
*CIE XYZ* tristimulus values to *Generic Film* (colour filters using
Illuminant C) colourspace matrix.

References
----------
- :cite:`InternationalTelecommunicationUnion2021`
"""

RGB_COLOURSPACE_H273_GENERIC_FILM: RGB_Colourspace = RGB_Colourspace(
    "ITU-T H.273 - Generic Film",
    PRIMARIES_H273_GENERIC_FILM,
    CCS_WHITEPOINT_H273_GENERIC_FILM,
    WHITEPOINT_NAME_H273_GENERIC_FILM,
    MATRIX_H273_GENERIC_FILM_RGB_TO_XYZ,
    MATRIX_XYZ_TO_H273_GENERIC_FILM_RGB,
    linear_function,
    linear_function,
)
RGB_COLOURSPACE_H273_GENERIC_FILM.__doc__ = """
*Recommendation ITU-T H.273* *Generic Film* (colour filters using Illuminant C)
colourspace.

References
----------
:cite:`InternationalTelecommunicationUnion2021`
"""


PRIMARIES_H273_22_UNSPECIFIED: NDArrayFloat = np.array(
    [
        [0.630, 0.340],
        [0.295, 0.605],
        [0.155, 0.077],
    ]
)
"""
Colourspace primaries for row *22* as given in
*Table 2 - Interpretation of colour primaries (ColourPrimaries) value*.

References
----------
- :cite:`InternationalTelecommunicationUnion2021`
"""

WHITEPOINT_NAME_H273_22_UNSPECIFIED: str = "D65"
"""
Whitepoint name for row *22* as given in
*Table 2 - Interpretation of colour primaries (ColourPrimaries) value*.

References
----------
- :cite:`InternationalTelecommunicationUnion2021`
"""

CCS_WHITEPOINT_H273_22_UNSPECIFIED: NDArrayFloat = np.array([0.3127, 0.3290])
"""
Whitepoint chromaticity coordinates for row *22* as given in
*Table 2 - Interpretation of colour primaries (ColourPrimaries) value*.

References
----------
- :cite:`InternationalTelecommunicationUnion2021`
"""

MATRIX_H273_22_UNSPECIFIED_RGB_TO_XYZ: NDArrayFloat = normalised_primary_matrix(
    PRIMARIES_H273_22_UNSPECIFIED, CCS_WHITEPOINT_H273_22_UNSPECIFIED
)
"""
Row *22* colourspace as given in
*Table 2 - Interpretation of colour primaries (ColourPrimaries) value* to
*CIE XYZ* tristimulus values matrix.

References
----------
- :cite:`InternationalTelecommunicationUnion2021`
"""

MATRIX_XYZ_TO_H273_22_UNSPECIFIED_RGB: NDArrayFloat = np.linalg.inv(
    MATRIX_H273_22_UNSPECIFIED_RGB_TO_XYZ
)
"""
*CIE XYZ* tristimulus values to row *22* colourspace as given in
*Table 2 - Interpretation of colour primaries (ColourPrimaries) value* matrix.

References
----------
- :cite:`InternationalTelecommunicationUnion2021`
"""

RGB_COLOURSPACE_H273_22_UNSPECIFIED: RGB_Colourspace = RGB_Colourspace(
    "ITU-T H.273 - 22 Unspecified",
    PRIMARIES_H273_22_UNSPECIFIED,
    CCS_WHITEPOINT_H273_22_UNSPECIFIED,
    WHITEPOINT_NAME_H273_22_UNSPECIFIED,
    MATRIX_H273_22_UNSPECIFIED_RGB_TO_XYZ,
    MATRIX_XYZ_TO_H273_22_UNSPECIFIED_RGB,
    linear_function,
    linear_function,
)
RGB_COLOURSPACE_H273_22_UNSPECIFIED.__doc__ = """
*Recommendation ITU-T H.273* row *22* colourspace as given in
*Table 2 - Interpretation of colour primaries (ColourPrimaries) value*.

References
----------
:cite:`InternationalTelecommunicationUnion2021`
"""
