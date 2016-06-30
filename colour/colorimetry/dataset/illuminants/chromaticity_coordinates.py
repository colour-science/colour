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
         'D50': (0.34570, 0.35850),
         'D55': (0.33242, 0.34743),
         'D60': (0.32168, 0.33767),
         'D65': (0.31270, 0.32900),
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
         'D60': (0.32303762352150689, 0.33921806639373481),
         'D65': (0.31382, 0.33100),
         'D75': (0.29968, 0.31740),
         'E': (1 / 3, 1 / 3),
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
    {'FL3.1': (0.44067353289223504, 0.40329596318733379),
     'FL3.2': (0.38077244876069538, 0.37335120824041923),
     'FL3.3': (0.31528043713433851, 0.34385457574132050),
     'FL3.4': (0.44290698484208529, 0.40432032345370644),
     'FL3.5': (0.37489654731472360, 0.36715348958268679),
     'FL3.6': (0.34880224546494737, 0.35999647646308836),
     'FL3.7': (0.43842306932684200, 0.40452742119369783),
     'FL3.8': (0.38197543093288233, 0.38317575120044350),
     'FL3.9': (0.34985554925959556, 0.35908517036460258),
     'FL3.10': (0.34550392600048208, 0.35595051932918587),
     'FL3.11': (0.32450879474655048, 0.34336713555796194),
     'FL3.12': (0.43767362795770576, 0.40366504631208772),
     'FL3.13': (0.38305104379106764, 0.37244140904299711),
     'FL3.14': (0.34472118254736284, 0.36093483446077979),
     'FL3.15': (0.31265880422387227, 0.32872684767045252),
     'HP1': (0.53299914700394979, 0.41495132086862591),
     'HP2': (0.47779010118558457, 0.41583778397924470),
     'HP3': (0.43022945458819589, 0.40751394706915489),
     'HP4': (0.38116733745857934, 0.37971830490253644),
     'HP5': (0.37758305715775720, 0.37134769850018201)})

ILLUMINANTS_CIE_1964_10_DEGREE_STANDARD_OBSERVER.update(
    {'FL3.1': (0.44983068401000342, 0.39023140432126552),
     'FL3.2': (0.38692411667293330, 0.36575603473282087),
     'FL3.3': (0.32117698685586532, 0.34050109265498119),
     'FL3.4': (0.44812127511399519, 0.39707711214248231),
     'FL3.5': (0.37781416660889494, 0.36662576696306032),
     'FL3.6': (0.35197647898350387, 0.36109443288967663),
     'FL3.7': (0.44430920881092179, 0.39679138731487085),
     'FL3.8': (0.38758893199977079, 0.37630556941017346),
     'FL3.9': (0.35468899071044874, 0.35344503359338325),
     'FL3.10': (0.34934479233440002, 0.35498442114086876),
     'FL3.11': (0.32926797569512017, 0.33886538664353716),
     'FL3.12': (0.44225208043800057, 0.40122055107125215),
     'FL3.13': (0.38627526878081714, 0.37428319095058565),
     'FL3.14': (0.34725507863829103, 0.36680824250417976),
     'FL3.15': (0.31461399790924588, 0.33337714937711332),
     'HP1': (0.54333460024730662, 0.40528929848043083),
     'HP2': (0.48264733064872084, 0.41081564417968452),
     'HP3': (0.43556003450395431, 0.39880108439971107),
     'HP4': (0.38519364112354304, 0.36827547924101506),
     'HP5': (0.38031641560663798, 0.36661711479785053)})

ILLUMINANTS_CIE_1931_2_DEGREE_STANDARD_OBSERVER.update({
    'DCI-P3': (0.31400, 0.35100)})
