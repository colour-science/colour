"""
CIE XYZ Tristimulus Values of the Hunter L,a,b Illuminants
==========================================================

Defines the *CIE XYZ* tristimulus values of the *Hunter L,a,b* illuminants
dataset for the *CIE 1931 2 Degree Standard Observer* and
*CIE 1964 10 Degree Standard Observer*.

The currently implemented data has been extracted from :cite:`HunterLab2008b`,
however you may want to use different data according to the tables given in
:cite:`HunterLab2008c`.

References
----------
-   :cite:`HunterLab2008b` : HunterLab. (2008). Hunter L,a,b Color Scale.
    http://www.hunterlab.se/wp-content/uploads/2012/11/Hunter-L-a-b.pdf
-   :cite:`HunterLab2008c` : HunterLab. (2008). Illuminant Factors in
    Universal Software and EasyMatch Coatings.
    https://support.hunterlab.com/hc/en-us/article_attachments/201437785/\
an02_02.pdf
"""

from __future__ import annotations

import numpy as np
from collections import namedtuple

from colour.hints import Tuple
from colour.utilities import CaseInsensitiveMapping

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "Illuminant_Specification_HunterLab",
    "DATA_ILLUMINANTS_HUNTERLAB_STANDARD_OBSERVER_2_DEGREE_CIE1931",
    "TVS_ILLUMINANTS_HUNTERLAB_STANDARD_OBSERVER_2_DEGREE_CIE1931",
    "DATA_ILLUMINANTS_HUNTERLAB_STANDARD_OBSERVER_10_DEGREE_CIE1964",
    "TVS_ILLUMINANTS_HUNTERLAB_STANDARD_OBSERVER_10_DEGREE_CIE1964",
    "TVS_ILLUMINANTS_HUNTERLAB",
]

Illuminant_Specification_HunterLab = namedtuple(
    "Illuminant_Specification_HunterLab", ("name", "XYZ_n", "K_ab")
)

DATA_ILLUMINANTS_HUNTERLAB_STANDARD_OBSERVER_2_DEGREE_CIE1931: Tuple = (
    ("A", np.array([109.83, 100.00, 35.55]), np.array([185.20, 38.40])),
    ("C", np.array([98.04, 100.00, 118.11]), np.array([175.00, 70.00])),
    ("D50", np.array([96.38, 100.00, 82.45]), np.array([173.51, 58.48])),
    ("D60", np.array([95.23, 100.00, 100.86]), np.array([172.47, 64.72])),
    ("D65", np.array([95.02, 100.00, 108.82]), np.array([172.30, 67.20])),
    ("D75", np.array([94.96, 100.00, 122.53]), np.array([172.22, 71.30])),
    ("FL2", np.array([98.09, 100.00, 67.53]), np.array([175.00, 52.90])),
    ("TL 4", np.array([101.40, 100.00, 65.90]), np.array([178.00, 52.30])),
    ("UL 3000", np.array([107.99, 100.00, 33.91]), np.array([183.70, 37.50])),
)

TVS_ILLUMINANTS_HUNTERLAB_STANDARD_OBSERVER_2_DEGREE_CIE1931: (
    CaseInsensitiveMapping
) = CaseInsensitiveMapping(
    {
        x[0]: Illuminant_Specification_HunterLab(*x)
        for x in DATA_ILLUMINANTS_HUNTERLAB_STANDARD_OBSERVER_2_DEGREE_CIE1931
    }
)
"""
*CIE XYZ* tristimulus values of the *Hunter L,a,b* illuminants for the
*CIE 1931 2 Degree Standard Observer*.

References
----------
:cite:`HunterLab2008b`, :cite:`HunterLab2008c`
"""

DATA_ILLUMINANTS_HUNTERLAB_STANDARD_OBSERVER_10_DEGREE_CIE1964: Tuple = (
    ("A", np.array([111.16, 100.00, 35.19]), np.array([186.30, 38.20])),
    ("C", np.array([97.30, 100.00, 116.14]), np.array([174.30, 69.40])),
    ("D50", np.array([96.72, 100.00, 81.45]), np.array([173.82, 58.13])),
    ("D60", np.array([95.21, 100.00, 99.60]), np.array([172.45, 64.28])),
    ("D65", np.array([94.83, 100.00, 107.38]), np.array([172.10, 66.70])),
    ("D75", np.array([94.45, 100.00, 120.70]), np.array([171.76, 70.76])),
    ("FL2", np.array([102.13, 100.00, 69.37]), np.array([178.60, 53.60])),
    ("TL 4", np.array([103.82, 100.00, 66.90]), np.array([180.10, 52.70])),
    ("UL 3000", np.array([111.12, 100.00, 35.21]), np.array([186.30, 38.20])),
)

TVS_ILLUMINANTS_HUNTERLAB_STANDARD_OBSERVER_10_DEGREE_CIE1964: (
    CaseInsensitiveMapping
) = CaseInsensitiveMapping(
    {
        x[0]: Illuminant_Specification_HunterLab(*x)
        for x in DATA_ILLUMINANTS_HUNTERLAB_STANDARD_OBSERVER_10_DEGREE_CIE1964
    }
)
"""
*CIE XYZ* tristimulus values of the *Hunter L,a,b* illuminants for the
*CIE 1964 10 Degree Standard Observer*.

References
----------
:cite:`HunterLab2008b`, :cite:`HunterLab2008c`
"""

TVS_ILLUMINANTS_HUNTERLAB: CaseInsensitiveMapping = CaseInsensitiveMapping(
    {
        "CIE 1931 2 Degree Standard Observer": (
            TVS_ILLUMINANTS_HUNTERLAB_STANDARD_OBSERVER_2_DEGREE_CIE1931
        ),
        "CIE 1964 10 Degree Standard Observer": (
            TVS_ILLUMINANTS_HUNTERLAB_STANDARD_OBSERVER_10_DEGREE_CIE1964
        ),
    }
)
TVS_ILLUMINANTS_HUNTERLAB.__doc__ = """
*CIE XYZ* tristimulus values of the *HunterLab* illuminants.

References
----------
:cite:`HunterLab2008b`, :cite:`HunterLab2008c`

Aliases:

-   'cie_2_1931': 'CIE 1931 2 Degree Standard Observer'
-   'cie_10_1964': 'CIE 1964 10 Degree Standard Observer'
"""
TVS_ILLUMINANTS_HUNTERLAB["cie_2_1931"] = TVS_ILLUMINANTS_HUNTERLAB[
    "CIE 1931 2 Degree Standard Observer"
]
TVS_ILLUMINANTS_HUNTERLAB["cie_10_1964"] = TVS_ILLUMINANTS_HUNTERLAB[
    "CIE 1964 10 Degree Standard Observer"
]
