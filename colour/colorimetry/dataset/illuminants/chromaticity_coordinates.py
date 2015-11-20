#!/usr/bin/env python
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
-   Illuminants F Series (F1, F10, F11, F12, F2, F3, F4, F5, F6, F7, F8, F9,
    FL3.1, FL3.10, FL3.11, FL3.12, FL3.13, FL3.14, FL3.15, FL3.2, FL3.3, FL3.4,
    FL3.5, FL3.6, FL3.7, FL3.8, FL3.9)
-   High Pressure Discharge Lamps (HP1, HP2, HP3, HP4, HP5)

The following other illuminants are available:

- DCI-P3 (*CIE 1931 2 Degree Standard Observer* only) [2]_

See Also
--------
`Illuminants IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/\
blob/master/notebooks/colorimetry/illuminants.ipynb>`_

Warning
-------
DCI-P3 illuminant has no associated spectral power distribution. DCI has no
official reference spectral measurement for this whitepoint. The closest
matching spectral power distribution is Kinoton 75P projector.

Notes
-----
-   *CIE* illuminants with chromaticity coordinates not defined in the
    reference [1]_ have been calculated using their relative spectral power
    distributions and the
    :func:`colour.colorimetry.tristimulus.spectral_to_XYZ` definition.

References
----------
.. [1]  Wikipedia. (n.d.). White points of standard illuminants. Retrieved
        February 24, 2014, from http://en.wikipedia.org/wiki/\
Standard_illuminant#White_points_of_standard_illuminants
.. [2]  Digital Cinema Initiatives. (2007). Digital Cinema System
        Specification - Version 1.1. Retrieved from
        http://www.dcimovies.com/archives/spec_v1_1/\
DCI_DCinema_System_Spec_v1_1.pdf
"""

from __future__ import division, unicode_literals

from colour.utilities import CaseInsensitiveMapping

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'ILLUMINANTS_CIE_1931_2_DEGREE_STANDARD_OBSERVER',
    'ILLUMINANTS_CIE_1964_10_DEGREE_STANDARD_OBSERVER',
    'ILLUMINANTS']

ILLUMINANTS_CIE_1931_2_DEGREE_STANDARD_OBSERVER = (
    CaseInsensitiveMapping(
        {'A': (0.44757, 0.40745),
         'B': (0.34842, 0.35161),
         'C': (0.31006, 0.31616),
         'D50': (0.34567, 0.35850),
         'D55': (0.33242, 0.34743),
         'D60': (0.32168, 0.33767),
         'D65': (0.31271, 0.32902),
         'D75': (0.29902, 0.31485),
         'E': (1 / 3, 1 / 3),
         'F1': (0.31310, 0.33727),
         'F2': (0.37208, 0.37529),
         'F3': (0.40910, 0.39430),
         'F4': (0.44018, 0.40329),
         'F5': (0.31379, 0.34531),
         'F6': (0.37790, 0.38835),
         'F7': (0.31292, 0.32933),
         'F8': (0.34588, 0.35875),
         'F9': (0.37417, 0.37281),
         'F10': (0.34609, 0.35986),
         'F11': (0.38052, 0.37713),
         'F12': (0.43695, 0.40441)}))
"""
*CIE* illuminant chromaticity coordinates for
*CIE 1931 2 Degree Standard Observer*.

ILLUMINANTS_CIE_1931_2_DEGREE_STANDARD_OBSERVER : CaseInsensitiveMapping
"""

ILLUMINANTS_CIE_1964_10_DEGREE_STANDARD_OBSERVER = (
    CaseInsensitiveMapping(
        {'A': (0.45117, 0.40594),
         'B': (0.34980, 0.35270),
         'C': (0.31039, 0.31905),
         'D50': (0.34773, 0.35952),
         'D55': (0.33411, 0.34877),
         'D60': (0.32299152277736748, 0.33912831290965012),
         'D65': (0.31382, 0.33100),
         'D75': (0.29968, 0.31740),
         'E': (1. / 3., 1. / 3.),
         'F1': (0.31811, 0.33559),
         'F2': (0.37925, 0.36733),
         'F3': (0.41761, 0.38324),
         'F4': (0.44920, 0.39074),
         'F5': (0.31975, 0.34246),
         'F6': (0.38660, 0.37847),
         'F7': (0.31569, 0.32960),
         'F8': (0.34902, 0.35939),
         'F9': (0.37829, 0.37045),
         'F10': (0.35090, 0.35444),
         'F11': (0.38541, 0.37123),
         'F12': (0.44256, 0.39717)}))
"""
*CIE* illuminant chromaticity coordinates for
*CIE 1964 10 Degree Standard Observer*.

ILLUMINANTS_CIE_1964_10_DEGREE_STANDARD_OBSERVER : CaseInsensitiveMapping
"""

ILLUMINANTS = CaseInsensitiveMapping(
    {'CIE 1931 2 Degree Standard Observer':
        ILLUMINANTS_CIE_1931_2_DEGREE_STANDARD_OBSERVER,
     'CIE 1964 10 Degree Standard Observer':
        ILLUMINANTS_CIE_1964_10_DEGREE_STANDARD_OBSERVER})
"""
Aggregated *CIE* illuminants chromaticity coordinates.

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

ILLUMINANTS_CIE_1931_2_DEGREE_STANDARD_OBSERVER.update(
    {'FL3.1': (0.44067505367083887, 0.40329824005212678),
     'FL3.2': (0.38077509554542083, 0.37335593850329946),
     'FL3.3': (0.31528299995992398, 0.34386075294770529),
     'FL3.4': (0.44290921051578830, 0.40432363641263036),
     'FL3.5': (0.37489860509162237, 0.36715721144008995),
     'FL3.6': (0.34880430112004895, 0.36000066745930004),
     'FL3.7': (0.43842546777142649, 0.40453105392942212),
     'FL3.8': (0.38197740008964315, 0.38317934336436121),
     'FL3.9': (0.34985695559881980, 0.35908801298815513),
     'FL3.10': (0.34550427834531300, 0.35595124320540111),
     'FL3.11': (0.32450996922163172, 0.34336978810428798),
     'FL3.12': (0.43767464805198042, 0.40366659242713887),
     'FL3.13': (0.38305207018919446, 0.37244321884144194),
     'FL3.14': (0.34472222710251049, 0.36093702097281111),
     'FL3.15': (0.31267061727694151, 0.32875456607137038),
     'HP1': (0.53300082279238248, 0.41495323768693643),
     'HP2': (0.47779207498613968, 0.41584045880847942),
     'HP3': (0.43023140032725588, 0.40751701140483843),
     'HP4': (0.38117189471910373, 0.37972657452791470),
     'HP5': (0.37758320909195225, 0.37134797280226028)})

ILLUMINANTS_CIE_1964_10_DEGREE_STANDARD_OBSERVER.update(
    {'FL3.1': (0.44983072060097606, 0.39023145487491850),
     'FL3.2': (0.38692417905929311, 0.36575613990679151),
     'FL3.3': (0.32117704635900118, 0.34050122948531319),
     'FL3.4': (0.44812132825245227, 0.39707718739452286),
     'FL3.5': (0.37781421452801472, 0.36662585171298506),
     'FL3.6': (0.35197652659197876, 0.36109452832152122),
     'FL3.7': (0.44430926608426524, 0.39679146953766947),
     'FL3.8': (0.38758897850585272, 0.37630564994236454),
     'FL3.9': (0.35468902355093479, 0.35344509693791348),
     'FL3.10': (0.34934480052537570, 0.35498443751900227),
     'FL3.11': (0.32926800293079544, 0.33886544553697306),
     'FL3.12': (0.44225210471167209, 0.40122058660685506),
     'FL3.13': (0.38627529279993350, 0.37428323257289459),
     'FL3.14': (0.34725510277403043, 0.36680829309760771),
     'FL3.15': (0.31461426562662165, 0.33337778105773069),
     'HP1': (0.54333464101412698, 0.40528934193733374),
     'HP2': (0.48264737821938364, 0.41081570598949713),
     'HP3': (0.43556008048844208, 0.39880115305546204),
     'HP4': (0.38519374626557140, 0.36827565930394440),
     'HP5': (0.38031641910835223, 0.36661712091281040)})

ILLUMINANTS_CIE_1931_2_DEGREE_STANDARD_OBSERVER.update({
    'DCI-P3': (0.31400, 0.35100)})
