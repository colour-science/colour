"""
CIE XYZ Tristimulus Values of Illuminants
=========================================

Defines the *CIE XYZ* tristimulus values of the illuminants for the
*CIE 1931 2 Degree Standard Observer* and
*CIE 1964 10 Degree Standard Observer*.

The following *CIE* illuminants are available:

-   CIE Standard Illuminant A
-   CIE Illuminant C
-   CIE Illuminant D Series (D50, D55, D60, D65, D75)

Notes
-----
-   The intent of the data in this module is to provide a practical reference
    if it is required to use the exact *CIE XYZ* tristimulus values of the
    *CIE* illuminants as given in :cite:`Carter2018`. Indeed different rounding
    practises in the colorimetric conversions yield different values for those
    illuminants, as a related example, *CIE Standard Illuminant D Series D65*
    chromaticity coordinates are commonly given as (0.31270, 0.32900) but
    :cite:`Carter2018` defines them as (0.31271, 0.32903).

References
----------
-   :cite:`Carter2018` : Carter, E. C., Schanda, J. D., Hirschler, R., Jost,
    S., Luo, M. R., Melgosa, M., Ohno, Y., Pointer, M. R., Rich, D. C., Vienot,
    F., Whitehead, L., & Wold, J. H. (2018). CIE 015:2018 Colorimetry, 4th
    Edition. International Commission on Illumination. doi:10.25039/TR.015.2018
"""

from __future__ import annotations

import numpy as np

from colour.utilities import CaseInsensitiveMapping

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TVS_ILLUMINANTS_CIE_STANDARD_OBSERVER_2_DEGREE_CIE1931",
    "TVS_ILLUMINANTS_CIE_STANDARD_OBSERVER_10_DEGREE_CIE1964",
    "TVS_ILLUMINANTS",
]

TVS_ILLUMINANTS_CIE_STANDARD_OBSERVER_2_DEGREE_CIE1931: (
    CaseInsensitiveMapping
) = CaseInsensitiveMapping(
    {
        "A": np.array([109.85, 100.00, 35.58]),
        "C": np.array([98.07, 100.00, 118.22]),
        "D50": np.array([96.42, 100.00, 82.51]),
        "D55": np.array([95.68, 100.00, 92.14]),
        "D65": np.array([95.04, 100.00, 108.88]),
        "D75": np.array([94.97, 100.00, 122.61]),
    }
)
"""
*CIE XYZ* tristimulus values of the *CIE* illuminants for the
*CIE 1931 2 Degree Standard Observer*.

References
----------
:cite:`Carter2018`
"""

TVS_ILLUMINANTS_CIE_STANDARD_OBSERVER_10_DEGREE_CIE1964: (
    CaseInsensitiveMapping
) = CaseInsensitiveMapping(
    {
        "A": np.array([111.14, 100.00, 35.20]),
        "C": np.array([97.29, 100.00, 116.14]),
        "D50": np.array([96.72, 100.00, 81.43]),
        "D55": np.array([95.80, 100.00, 90.93]),
        "D65": np.array([94.81, 100.00, 107.32]),
        "D75": np.array([94.42, 100.00, 120.64]),
    }
)
"""
*CIE XYZ* tristimulus values of the *CIE* illuminants for the
*CIE 1964 10 Degree Standard Observer*.

References
----------
:cite:`Carter2018`

TVS_ILLUMINANTS_CIE_STANDARD_OBSERVER_10_DEGREE_CIE1964 : \
CaseInsensitiveMapping
"""

TVS_ILLUMINANTS: CaseInsensitiveMapping = CaseInsensitiveMapping(
    {
        "CIE 1931 2 Degree Standard Observer": CaseInsensitiveMapping(
            TVS_ILLUMINANTS_CIE_STANDARD_OBSERVER_2_DEGREE_CIE1931
        ),
        "CIE 1964 10 Degree Standard Observer": CaseInsensitiveMapping(
            TVS_ILLUMINANTS_CIE_STANDARD_OBSERVER_10_DEGREE_CIE1964
        ),
    }
)
TVS_ILLUMINANTS.__doc__ = """
*CIE XYZ* tristimulus values of the illuminants.

References
----------
:cite:`Carter2018`

Aliases:

-   'cie_2_1931': 'CIE 1931 2 Degree Standard Observer'
-   'cie_10_1964': 'CIE 1964 10 Degree Standard Observer'
"""
TVS_ILLUMINANTS["cie_2_1931"] = TVS_ILLUMINANTS[
    "CIE 1931 2 Degree Standard Observer"
]
TVS_ILLUMINANTS["cie_10_1964"] = TVS_ILLUMINANTS[
    "CIE 1964 10 Degree Standard Observer"
]
