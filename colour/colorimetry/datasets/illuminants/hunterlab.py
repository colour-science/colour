# -*- coding: utf-8 -*-
"""
HunterLab Dataset
=================

Defines the *HunterLab* illuminants datasets for the
*CIE 1931 2 Degree Standard Observer* and
*CIE 1964 10 Degree Standard Observer*.

The currently implemented data has been extracted from :cite:`HunterLab2008b`,
however you may want to use different data according to the tables given in
:cite:`HunterLab2008c`.

See Also
--------
`Illuminants Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/colorimetry/illuminants.ipynb>`_

References
----------
-   :cite:`HunterLab2008b` : HunterLab. (2008). Hunter L,a,b Color Scale.
    Retrieved from http://www.hunterlab.se/wp-content/uploads/2012/11/\
Hunter-L-a-b.pdf
-   :cite:`HunterLab2008c` : HunterLab. (2008). Illuminant Factors in Universal
    Software and EasyMatch Coatings. Retrieved from
    https://support.hunterlab.com/hc/en-us/article_attachments/201437785/\
an02_02.pdf
"""

from __future__ import division, unicode_literals

import numpy as np
from collections import namedtuple

from colour.utilities import CaseInsensitiveMapping

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'HunterLab_Illuminant_Specification',
    'HUNTERLAB_ILLUMINANTS_CIE_1931_2_DEGREE_STANDARD_OBSERVER_DATA',
    'HUNTERLAB_ILLUMINANTS_CIE_1931_2_DEGREE_STANDARD_OBSERVER',
    'HUNTERLAB_ILLUMINANTS_CIE_1964_10_DEGREE_STANDARD_OBSERVER_DATA',
    'HUNTERLAB_ILLUMINANTS_CIE_1964_10_DEGREE_STANDARD_OBSERVER',
    'HUNTERLAB_ILLUMINANTS'
]

HunterLab_Illuminant_Specification = namedtuple(
    'HunterLab_Illuminant_Specification', ('name', 'XYZ_n', 'K_ab'))

# yapf: disable
HUNTERLAB_ILLUMINANTS_CIE_1931_2_DEGREE_STANDARD_OBSERVER_DATA = (
    ('A', np.array([109.83, 100.00, 35.55]), np.array([185.20, 38.40])),
    ('C', np.array([98.04, 100.00, 118.11]), np.array([175.00, 70.00])),
    ('D50', np.array([96.38, 100.00, 82.45]), np.array([173.51, 58.48])),
    ('D60', np.array([95.23, 100.00, 100.86]), np.array([172.47, 64.72])),
    ('D65', np.array([95.02, 100.00, 108.82]), np.array([172.30, 67.20])),
    ('D75', np.array([94.96, 100.00, 122.53]), np.array([172.22, 71.30])),
    ('FL2', np.array([98.09, 100.00, 67.53]), np.array([175.00, 52.90])),
    ('TL 4', np.array([101.40, 100.00, 65.90]), np.array([178.00, 52.30])),
    ('UL 3000', np.array([107.99, 100.00, 33.91]), np.array([183.70, 37.50])))
# yapf: enable

HUNTERLAB_ILLUMINANTS_CIE_1931_2_DEGREE_STANDARD_OBSERVER = (
    CaseInsensitiveMapping({
        x[0]: HunterLab_Illuminant_Specification(*x)
        for x in HUNTERLAB_ILLUMINANTS_CIE_1931_2_DEGREE_STANDARD_OBSERVER_DATA
    }))
"""
*Hunter L,a,b* illuminant datasets for *CIE 1931 2 Degree Standard Observer*.

References
----------
:cite:`HunterLab2008b`, :cite:`HunterLab2008c`

HUNTERLAB_ILLUMINANTS_CIE_1931_2_DEGREE_STANDARD_OBSERVER :
    CaseInsensitiveMapping
"""

# yapf: disable
HUNTERLAB_ILLUMINANTS_CIE_1964_10_DEGREE_STANDARD_OBSERVER_DATA = (
    ('A', np.array([111.16, 100.00, 35.19]), np.array([186.30, 38.20])),
    ('C', np.array([97.30, 100.00, 116.14]), np.array([174.30, 69.40])),
    ('D50', np.array([96.72, 100.00, 81.45]), np.array([173.82, 58.13])),
    ('D60', np.array([95.21, 100.00, 99.60]), np.array([172.45, 64.28])),
    ('D65', np.array([94.83, 100.00, 107.38]), np.array([172.10, 66.70])),
    ('D75', np.array([94.45, 100.00, 120.70]), np.array([171.76, 70.76])),
    ('FL2', np.array([102.13, 100.00, 69.37]), np.array([178.60, 53.60])),
    ('TL 4', np.array([103.82, 100.00, 66.90]), np.array([180.10, 52.70])),
    ('UL 3000', np.array([111.12, 100.00, 35.21]), np.array([186.30, 38.20])))
# yapf: enable

HUNTERLAB_ILLUMINANTS_CIE_1964_10_DEGREE_STANDARD_OBSERVER = (
    CaseInsensitiveMapping({
        x[0]: HunterLab_Illuminant_Specification(*x)
        for x in
        HUNTERLAB_ILLUMINANTS_CIE_1964_10_DEGREE_STANDARD_OBSERVER_DATA
    }))
"""
*Hunter L,a,b* illuminant datasets for *CIE 1964 10 Degree Standard Observer*.

References
----------
:cite:`HunterLab2008b`, :cite:`HunterLab2008c`

HUNTERLAB_ILLUMINANTS_CIE_1964_10_DEGREE_STANDARD_OBSERVER :
    CaseInsensitiveMapping
"""

HUNTERLAB_ILLUMINANTS = CaseInsensitiveMapping({
    'CIE 1931 2 Degree Standard Observer':
        HUNTERLAB_ILLUMINANTS_CIE_1931_2_DEGREE_STANDARD_OBSERVER,
    'CIE 1964 10 Degree Standard Observer':
        HUNTERLAB_ILLUMINANTS_CIE_1964_10_DEGREE_STANDARD_OBSERVER
})
HUNTERLAB_ILLUMINANTS.__doc__ = """
Aggregated *Hunter L,a,b* illuminant datasets.

References
----------
:cite:`HunterLab2008b`, :cite:`HunterLab2008c`

HUNTERLAB_ILLUMINANTS : CaseInsensitiveMapping
    **{'CIE 1931 2 Degree Standard Observer',
    'CIE 1964 10 Degree Standard Observer'}**

Aliases:

-   'cie_2_1931': 'CIE 1931 2 Degree Standard Observer'
-   'cie_10_1964': 'CIE 1964 10 Degree Standard Observer'
"""
HUNTERLAB_ILLUMINANTS['cie_2_1931'] = (
    HUNTERLAB_ILLUMINANTS['CIE 1931 2 Degree Standard Observer'])
HUNTERLAB_ILLUMINANTS['cie_10_1964'] = (
    HUNTERLAB_ILLUMINANTS['CIE 1964 10 Degree Standard Observer'])
