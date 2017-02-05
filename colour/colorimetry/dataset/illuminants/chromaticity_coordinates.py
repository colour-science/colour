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
`Illuminants Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
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

import numpy as np

from colour.utilities import CaseInsensitiveMapping

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'ILLUMINANTS_CIE_1931_2_DEGREE_STANDARD_OBSERVER',
    'ILLUMINANTS_CIE_1964_10_DEGREE_STANDARD_OBSERVER',
    'ILLUMINANTS']

ILLUMINANTS_CIE_1931_2_DEGREE_STANDARD_OBSERVER = (
    CaseInsensitiveMapping({
        'A': np.array([0.44757, 0.40745]),
        'B': np.array([0.34842, 0.35161]),
        'C': np.array([0.31006, 0.31616]),
        'D50': np.array([0.34570, 0.35850]),
        'D55': np.array([0.33242, 0.34743]),
        'D60': np.array([0.32168, 0.33767]),
        'D65': np.array([0.31270, 0.32900]),
        'D75': np.array([0.29902, 0.31485]),
        'E': np.array([1 / 3, 1 / 3]),
        'F1': np.array([0.31310, 0.33727]),
        'F2': np.array([0.37208, 0.37529]),
        'F3': np.array([0.40910, 0.39430]),
        'F4': np.array([0.44018, 0.40329]),
        'F5': np.array([0.31379, 0.34531]),
        'F6': np.array([0.37790, 0.38835]),
        'F7': np.array([0.31292, 0.32933]),
        'F8': np.array([0.34588, 0.35875]),
        'F9': np.array([0.37417, 0.37281]),
        'F10': np.array([0.34609, 0.35986]),
        'F11': np.array([0.38052, 0.37713]),
        'F12': np.array([0.43695, 0.40441])}))
"""
*CIE* illuminant chromaticity coordinates for
*CIE 1931 2 Degree Standard Observer*.

ILLUMINANTS_CIE_1931_2_DEGREE_STANDARD_OBSERVER : CaseInsensitiveMapping
"""

ILLUMINANTS_CIE_1964_10_DEGREE_STANDARD_OBSERVER = (
    CaseInsensitiveMapping({
        'A': np.array([0.45117, 0.40594]),
        'B': np.array([0.34980, 0.35270]),
        'C': np.array([0.31039, 0.31905]),
        'D50': np.array([0.34773, 0.35952]),
        'D55': np.array([0.33411, 0.34877]),
        'D60': np.array([0.322957407931312, 0.339135835524579]),
        'D65': np.array([0.31382, 0.33100]),
        'D75': np.array([0.29968, 0.31740]),
        'E': np.array([1 / 3, 1 / 3]),
        'F1': np.array([0.31811, 0.33559]),
        'F2': np.array([0.37925, 0.36733]),
        'F3': np.array([0.41761, 0.38324]),
        'F4': np.array([0.44920, 0.39074]),
        'F5': np.array([0.31975, 0.34246]),
        'F6': np.array([0.38660, 0.37847]),
        'F7': np.array([0.31569, 0.32960]),
        'F8': np.array([0.34902, 0.35939]),
        'F9': np.array([0.37829, 0.37045]),
        'F10': np.array([0.35090, 0.35444]),
        'F11': np.array([0.38541, 0.37123]),
        'F12': np.array([0.44256, 0.39717])}))
"""
*CIE* illuminant chromaticity coordinates for
*CIE 1964 10 Degree Standard Observer*.

ILLUMINANTS_CIE_1964_10_DEGREE_STANDARD_OBSERVER : CaseInsensitiveMapping
"""

ILLUMINANTS = CaseInsensitiveMapping({
    'CIE 1931 2 Degree Standard Observer':
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

ILLUMINANTS_CIE_1931_2_DEGREE_STANDARD_OBSERVER.update({
    'FL3.1': np.array([0.440673532892235, 0.403295963187334]),
    'FL3.2': np.array([0.380772448760695, 0.373351208240419]),
    'FL3.3': np.array([0.315280437134339, 0.343854575741321]),
    'FL3.4': np.array([0.442906984842085, 0.404320323453706]),
    'FL3.5': np.array([0.374896547314724, 0.367153489582687]),
    'FL3.6': np.array([0.348802245464947, 0.359996476463088]),
    'FL3.7': np.array([0.438423069326842, 0.404527421193698]),
    'FL3.8': np.array([0.381975430932882, 0.383175751200444]),
    'FL3.9': np.array([0.349855549259596, 0.359085170364603]),
    'FL3.10': np.array([0.345503926000482, 0.355950519329186]),
    'FL3.11': np.array([0.324508794746550, 0.343367135557962]),
    'FL3.12': np.array([0.437673627957706, 0.403665046312088]),
    'FL3.13': np.array([0.383051043791068, 0.372441409042997]),
    'FL3.14': np.array([0.344721182547363, 0.360934834460780]),
    'FL3.15': np.array([0.312658804223872, 0.328726847670453]),
    'HP1': np.array([0.532999147003950, 0.414951320868626]),
    'HP2': np.array([0.477790101185585, 0.415837783979245]),
    'HP3': np.array([0.430229454588196, 0.407513947069155]),
    'HP4': np.array([0.381167337458579, 0.379718304902536]),
    'HP5': np.array([0.377583057157757, 0.371347698500182])})

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
    'HP5': np.array([0.380316415606638, 0.366617114797851])})

ILLUMINANTS_CIE_1931_2_DEGREE_STANDARD_OBSERVER.update({
    'DCI-P3': np.array([0.31400, 0.35100])})
