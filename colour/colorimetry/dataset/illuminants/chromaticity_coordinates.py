# -*- coding: utf-8 -*-
"""
Illuminants Chromaticity Coordinates
====================================

Defines *CIE* illuminants chromaticity coordinates for the
*CIE 1931 2 Degree Standard Observer* and
*CIE 1964 10 Degree Standard Observer*.

The following *CIE* illuminants are available:

-   CIE Standard Illuminant A
-   CIE Illuminant B
-   CIE Illuminant C
-   CIE Illuminant D Series (D50, D55, D60, D65, D75)
-   CIE Illuminant E
-   Illuminants F Series (FL1, FL2, FL3, FL4, FL5, FL6, FL7, FL8, FL9, FL10,
    FL11, FL12, FL3.1, FL3.10, FL3.11, FL3.12, FL3.13, FL3.14, FL3.15, FL3.2,
    FL3.3, FL3.4, FL3.5, FL3.6, FL3.7, FL3.8, FL3.9)
-   High Pressure Discharge Lamps (HP1, HP2, HP3, HP4, HP5)

The following other illuminants are available:

- DCI-P3 (*CIE 1931 2 Degree Standard Observer* only)
- ACES (*CIE 1931 2 Degree Standard Observer* only)

See Also
--------
`Illuminants Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/colorimetry/illuminants.ipynb>`_

Warning
-------
DCI-P3 illuminant has no associated spectral distribution. DCI has no
official reference spectral measurement for this whitepoint. The closest
matching spectral distribution is Kinoton 75P projector.

Notes
-----
-   *CIE* illuminants with chromaticity coordinates not defined in the
    reference :cite:`Wikipedia2006a` have been calculated using their
    correlated colour temperature and
    :func:`colour.temperature.CCT_to_xy_CIE_D`
    :func:`colour.sd_CIE_illuminant_D_series` and / or
    :func:`colour.sd_to_XYZ` definitions.

References
----------
-   :cite:`CIETC1-482004h` : CIE TC 1-48. (2004). CIE 015:2004 Colorimetry,
    3rd Edition. CIE 015:2004 Colorimetry, 3rd Edition. Commission
    internationale de l'éclairage. ISBN:978-3-901-90633-6
-   :cite:`DigitalCinemaInitiatives2007b` : Digital Cinema Initiatives. (2007).
    Digital Cinema System Specification - Version 1.1. Retrieved from
    http://www.dcimovies.com/archives/spec_v1_1/\
DCI_DCinema_System_Spec_v1_1.pdf
-   :cite:`TheAcademyofMotionPictureArtsandSciences2014q` : The Academy of
    Motion Picture Arts and Sciences, Science and Technology Council, & Academy
    Color Encoding System (ACES) Project Subcommittee. (2014). Technical
    Bulletin TB-2014-004 - Informative Notes on SMPTE ST 2065-1 - Academy Color
    Encoding Specification (ACES). Retrieved from
    https://github.com/ampas/aces-dev/tree/master/documents
-   :cite:`Wikipedia2006a` : Wikipedia. (2006). White points of standard
    illuminants. Retrieved February 24, 2014, from http://en.wikipedia.org/\
wiki/Standard_illuminant#White_points_of_standard_illuminants
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.utilities import CaseInsensitiveMapping

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'ILLUMINANTS_CIE_1931_2_DEGREE_STANDARD_OBSERVER',
    'ILLUMINANTS_CIE_1964_10_DEGREE_STANDARD_OBSERVER', 'ILLUMINANTS'
]

ILLUMINANTS_CIE_1931_2_DEGREE_STANDARD_OBSERVER = CaseInsensitiveMapping({
    'A': np.array([0.44758, 0.40745]),
    'B': np.array([0.34842, 0.35161]),
    'C': np.array([0.31006, 0.31616]),
    'D50': np.array([0.34570, 0.35850]),
    'D55': np.array([0.33243, 0.34744]),
    'D60': np.array([0.321626242047397, 0.337736995955436]),
    'D65': np.array([0.31270, 0.32900]),
    'D75': np.array([0.29903, 0.31488]),
    'E': np.array([1 / 3, 1 / 3]),
    'FL1': np.array([0.31310, 0.33710]),
    'FL2': np.array([0.37210, 0.37510]),
    'FL3': np.array([0.40910, 0.39410]),
    'FL4': np.array([0.44020, 0.40310]),
    'FL5': np.array([0.31380, 0.34520]),
    'FL6': np.array([0.37790, 0.38820]),
    'FL7': np.array([0.31290, 0.32920]),
    'FL8': np.array([0.34580, 0.35860]),
    'FL9': np.array([0.37410, 0.37270]),
    'FL10': np.array([0.34580, 0.35880]),
    'FL11': np.array([0.38050, 0.37690]),
    'FL12': np.array([0.43700, 0.40420])
})
"""
*CIE* illuminant chromaticity coordinates for
*CIE 1931 2 Degree Standard Observer*.

ILLUMINANTS_CIE_1931_2_DEGREE_STANDARD_OBSERVER : CaseInsensitiveMapping
"""

ILLUMINANTS_CIE_1931_2_DEGREE_STANDARD_OBSERVER.update({
    'DCI-P3': np.array([0.31400, 0.35100]),
    'ACES': np.array([0.32168, 0.33767])
})

ILLUMINANTS_CIE_1931_2_DEGREE_STANDARD_OBSERVER.update({
    'FL3.1': np.array([0.44070, 0.40330]),
    'FL3.2': np.array([0.38080, 0.37340]),
    'FL3.3': np.array([0.31530, 0.34390]),
    'FL3.4': np.array([0.44290, 0.40430]),
    'FL3.5': np.array([0.37490, 0.36720]),
    'FL3.6': np.array([0.34880, 0.36000]),
    'FL3.7': np.array([0.43840, 0.40450]),
    'FL3.8': np.array([0.38200, 0.38320]),
    'FL3.9': np.array([0.34990, 0.35910]),
    'FL3.10': np.array([0.34550, 0.35600]),
    'FL3.11': np.array([0.32450, 0.34340]),
    'FL3.12': np.array([0.43770, 0.40370]),
    'FL3.13': np.array([0.38300, 0.37240]),
    'FL3.14': np.array([0.34470, 0.36090]),
    'FL3.15': np.array([0.31270, 0.32880]),
    'HP1': np.array([0.53300, 0.4150]),
    'HP2': np.array([0.47780, 0.41580]),
    'HP3': np.array([0.43020, 0.40750]),
    'HP4': np.array([0.38120, 0.37970]),
    'HP5': np.array([0.37760, 0.37130])
})

ILLUMINANTS_CIE_1964_10_DEGREE_STANDARD_OBSERVER = CaseInsensitiveMapping({
    'A': np.array([0.45117, 0.40594]),
    'B': np.array([0.34980, 0.35270]),
    'C': np.array([0.31039, 0.31905]),
    'D50': np.array([0.34773, 0.35952]),
    'D55': np.array([0.33412, 0.34877]),
    'D60': np.array([0.322986926715820, 0.339275732345997]),
    'D65': np.array([0.31382, 0.33100]),
    'D75': np.array([0.29968, 0.31740]),
    'E': np.array([1 / 3, 1 / 3]),
    'FL1': np.array([0.31811, 0.33559]),
    'FL2': np.array([0.37925, 0.36733]),
    'FL3': np.array([0.41761, 0.38324]),
    'FL4': np.array([0.44920, 0.39074]),
    'FL5': np.array([0.31975, 0.34246]),
    'FL6': np.array([0.38660, 0.37847]),
    'FL7': np.array([0.31569, 0.32960]),
    'FL8': np.array([0.34902, 0.35939]),
    'FL9': np.array([0.37829, 0.37045]),
    'FL10': np.array([0.35090, 0.35444]),
    'FL11': np.array([0.38541, 0.37123]),
    'FL12': np.array([0.44256, 0.39717])
})
"""
*CIE* illuminant chromaticity coordinates for
*CIE 1964 10 Degree Standard Observer*.

ILLUMINANTS_CIE_1964_10_DEGREE_STANDARD_OBSERVER : CaseInsensitiveMapping
"""

ILLUMINANTS_CIE_1964_10_DEGREE_STANDARD_OBSERVER.update({
    'FL3.1': np.array([0.449830684010003, 0.390231404321266]),
    'FL3.2': np.array([0.386924116672933, 0.365756034732821]),
    'FL3.3': np.array([0.321176986855865, 0.340501092654981]),
    'FL3.4': np.array([0.448121275113995, 0.397077112142482]),
    'FL3.5': np.array([0.377814166608895, 0.366625766963060]),
    'FL3.6': np.array([0.351976478983504, 0.361094432889677]),
    'FL3.7': np.array([0.444309208810922, 0.396791387314871]),
    'FL3.8': np.array([0.387588931999771, 0.376305569410173]),
    'FL3.9': np.array([0.354688990710449, 0.353445033593383]),
    'FL3.10': np.array([0.349344792334400, 0.354984421140869]),
    'FL3.11': np.array([0.329267975695120, 0.338865386643537]),
    'FL3.12': np.array([0.442252080438001, 0.401220551071252]),
    'FL3.13': np.array([0.386275268780817, 0.374283190950586]),
    'FL3.14': np.array([0.347255078638291, 0.366808242504180]),
    'FL3.15': np.array([0.314613997909246, 0.333377149377113]),
    'HP1': np.array([0.543334600247307, 0.405289298480431]),
    'HP2': np.array([0.482647330648721, 0.410815644179685]),
    'HP3': np.array([0.435560034503954, 0.398801084399711]),
    'HP4': np.array([0.385193641123543, 0.368275479241015]),
    'HP5': np.array([0.380316415606638, 0.366617114797851])
})

ILLUMINANTS = CaseInsensitiveMapping({
    'CIE 1931 2 Degree Standard Observer':
        ILLUMINANTS_CIE_1931_2_DEGREE_STANDARD_OBSERVER,
    'CIE 1964 10 Degree Standard Observer':
        ILLUMINANTS_CIE_1964_10_DEGREE_STANDARD_OBSERVER
})
ILLUMINANTS.__doc__ = """
Aggregated *CIE* illuminants chromaticity coordinates.

Notes
-----

*CIE Illuminant D Series D60* illuminant chromaticity coordinates were
computed as follows::

    CCT = 6000 * 1.4388 / 1.438
    xy = colour.temperature.CCT_to_xy_CIE_D(CCT)

    sd = colour.sd_CIE_illuminant_D_series(xy)
    sd.interpolator = colour.LinearInterpolator
    colour.XYZ_to_xy(
        colour.sd_to_XYZ(
            sd, colour.CMFS['CIE 1964 10 Degree Standard Observer']) / 100.0)

*CIE Illuminant D Series D50* illuminant and
*CIE Standard Illuminant D Series D65* chromaticity coordinates are rounded to
4 decimals as given in the typical RGB colourspaces litterature. Their
chromaticity coordinates as given in :cite:`CIETC1-482004h` are
(0.34567, 0.35851) and (0.31272, 0.32903) respectively.

References
----------
:cite:`CIETC1-482004h`, :cite:`DigitalCinemaInitiatives2007b`,
:cite:`TheAcademyofMotionPictureArtsandSciences2014q`, :cite:`Wikipedia2006a`

ILLUMINANTS : CaseInsensitiveMapping
    **{'CIE 1931 2 Degree Standard Observer',
    'CIE 1964 10 Degree Standard Observer'}**

Aliases:

-   'cie_2_1931': 'CIE 1931 2 Degree Standard Observer'
-   'cie_10_1964': 'CIE 1964 10 Degree Standard Observer'
"""
ILLUMINANTS['cie_2_1931'] = (
    ILLUMINANTS['CIE 1931 2 Degree Standard Observer'])
ILLUMINANTS['cie_10_1964'] = (
    ILLUMINANTS['CIE 1964 10 Degree Standard Observer'])
