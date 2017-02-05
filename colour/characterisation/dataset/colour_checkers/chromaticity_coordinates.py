#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ColourCheckers Chromaticity Coordinates
=======================================

Defines *ColourCheckers* chromaticity coordinates in *CIE xyY* colourspace.

Each *ColourChecker* data is in the form of a list of 24 *namedtuples* as
follows::

    [('index', 'name', 'xyY'), ..., ('index', 'name', 'xyY')]

The following *ColourCheckers* data is available:

-   :attr:`BABELCOLOR_AVERAGE`: Average data derived from measurements of 30
    *ColourChecker* charts.
-   :attr:`COLORCHECKER_2005`: Reference data from *GretagMacbeth* published in
    2005.
-   :attr:`COLORCHECKER_1976`: *ColourChecker* developed by *McCamy et al.* at
    Macbeth, a Division of Kollmorgen.

See Also
--------
`Colour Fitting Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/characterisation/fitting.ipynb>`_

References
----------
.. [1]  BabelColor. (2012). ColorChecker RGB and spectra. Retrieved from
        http://www.babelcolor.com/download/ColorChecker_RGB_and_spectra.xls
.. [2]  BabelColor. (2012). The ColorChecker (since 1976!). Retrieved
        September 26, 2014, from
        http://www.babelcolor.com/main_level/ColorChecker.htm
"""

from __future__ import division, unicode_literals

import numpy as np
from collections import namedtuple

from colour.colorimetry import ILLUMINANTS
from colour.utilities import CaseInsensitiveMapping

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['ColourChecker_Specification',
           'ColourChecker',
           'BABELCOLOR_AVERAGE_DATA',
           'BABELCOLOR_AVERAGE_ILLUMINANT',
           'BABELCOLOR_AVERAGE',
           'COLORCHECKER_2005_DATA',
           'COLORCHECKER_2005_ILLUMINANT',
           'COLORCHECKER_2005',
           'COLORCHECKER_1976_DATA',
           'COLORCHECKER_1976_ILLUMINANT',
           'COLORCHECKER_1976',
           'COLOURCHECKERS']

ColourChecker_Specification = namedtuple(
    'ColourChecker_Specification', ('index', 'name', 'xyY'))

ColourChecker = namedtuple('ColourChecker', ('name', 'data', 'illuminant'))

BABELCOLOR_AVERAGE_DATA = (
    (1, 'dark skin', np.array([0.4325, 0.3788, 0.1034])),
    (2, 'light skin', np.array([0.4191, 0.3748, 0.3525])),
    (3, 'blue sky', np.array([0.2761, 0.3004, 0.1847])),
    (4, 'foliage', np.array([0.3700, 0.4501, 0.1335])),
    (5, 'blue flower', np.array([0.3020, 0.2877, 0.2324])),
    (6, 'bluish green', np.array([0.2856, 0.3910, 0.4174])),
    (7, 'orange', np.array([0.5291, 0.4075, 0.3117])),
    (8, 'purplish blue', np.array([0.2339, 0.2155, 0.1140])),
    (9, 'moderate red', np.array([0.5008, 0.3293, 0.1979])),
    (10, 'purple', np.array([0.3326, 0.2556, 0.0644])),
    (11, 'yellow green', np.array([0.3989, 0.4998, 0.4435])),
    (12, 'orange yellow', np.array([0.4962, 0.4428, 0.4358])),
    (13, 'blue', np.array([0.2040, 0.1696, 0.0579])),
    (14, 'green', np.array([0.3270, 0.5033, 0.2307])),
    (15, 'red', np.array([0.5709, 0.3298, 0.1268])),
    (16, 'yellow', np.array([0.4694, 0.4732, 0.6081])),
    (17, 'magenta', np.array([0.4177, 0.2704, 0.2007])),
    (18, 'cyan', np.array([0.2151, 0.3037, 0.1903])),
    (19, 'white 9.5 (.05 D)', np.array([0.3488, 0.3628, 0.9129])),
    (20, 'neutral 8 (.23 D)', np.array([0.3451, 0.3596, 0.5885])),
    (21, 'neutral 6.5 (.44 D)', np.array([0.3446, 0.3590, 0.3595])),
    (22, 'neutral 5 (.70 D)', np.array([0.3438, 0.3589, 0.1912])),
    (23, 'neutral 3.5 (1.05 D)', np.array([0.3423, 0.3576, 0.0893])),
    (24, 'black 2 (1.5 D)', np.array([0.3439, 0.3565, 0.0320])))

BABELCOLOR_AVERAGE_ILLUMINANT = (
    ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D50'])
"""
*BabelColor Average* illuminant.

BABELCOLOR_AVERAGE_ILLUMINANT : ndarray
"""

BABELCOLOR_AVERAGE = [ColourChecker_Specification(*x) for x in
                      BABELCOLOR_AVERAGE_DATA]
"""
Average data derived from measurements of 30 *ColourChecker* charts.

BABELCOLOR_AVERAGE : list
"""

COLORCHECKER_2005_DATA = (
    (1, 'dark skin', np.array([0.4316, 0.3777, 0.1008])),
    (2, 'light skin', np.array([0.4197, 0.3744, 0.3495])),
    (3, 'blue sky', np.array([0.2760, 0.3016, 0.1836])),
    (4, 'foliage', np.array([0.3703, 0.4499, 0.1325])),
    (5, 'blue flower', np.array([0.2999, 0.2856, 0.2304])),
    (6, 'bluish green', np.array([0.2848, 0.3911, 0.4178])),
    (7, 'orange', np.array([0.5295, 0.4055, 0.3118])),
    (8, 'purplish blue', np.array([0.2305, 0.2106, 0.1126])),
    (9, 'moderate red', np.array([0.5012, 0.3273, 0.1938])),
    (10, 'purple', np.array([0.3319, 0.2482, 0.0637])),
    (11, 'yellow green', np.array([0.3984, 0.5008, 0.4446])),
    (12, 'orange yellow', np.array([0.4957, 0.4427, 0.4357])),
    (13, 'blue', np.array([0.2018, 0.1692, 0.0575])),
    (14, 'green', np.array([0.3253, 0.5032, 0.2318])),
    (15, 'red', np.array([0.5686, 0.3303, 0.1257])),
    (16, 'yellow', np.array([0.4697, 0.4734, 0.5981])),
    (17, 'magenta', np.array([0.4159, 0.2688, 0.2009])),
    (18, 'cyan', np.array([0.2131, 0.3023, 0.1930])),
    (19, 'white 9.5 (.05 D)', np.array([0.3469, 0.3608, 0.9131])),
    (20, 'neutral 8 (.23 D)', np.array([0.3440, 0.3584, 0.5894])),
    (21, 'neutral 6.5 (.44 D)', np.array([0.3432, 0.3581, 0.3632])),
    (22, 'neutral 5 (.70 D)', np.array([0.3446, 0.3579, 0.1915])),
    (23, 'neutral 3.5 (1.05 D)', np.array([0.3401, 0.3548, 0.0883])),
    (24, 'black 2 (1.5 D)', np.array([0.3406, 0.3537, 0.0311])))

COLORCHECKER_2005_ILLUMINANT = (
    ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D50'])
"""
*ColorChecker 2005* illuminant.

COLORCHECKER_2005_ILLUMINANT : ndarray
"""

COLORCHECKER_2005 = [ColourChecker_Specification(*x) for x in
                     COLORCHECKER_2005_DATA]
"""
Reference data from *GretagMacbeth (2005)*.

COLORCHECKER_2005 : list
"""

COLORCHECKER_1976_DATA = (
    (1, 'dark skin', np.array([0.4002, 0.3504, 0.1005])),
    (2, 'light skin', np.array([0.3773, 0.3446, 0.3582])),
    (3, 'blue sky', np.array([0.2470, 0.2514, 0.1933])),
    (4, 'foliage', np.array([0.3372, 0.4220, 0.1329])),
    (5, 'blue flower', np.array([0.2651, 0.2400, 0.2427])),
    (6, 'bluish green', np.array([0.2608, 0.3430, 0.4306])),
    (7, 'orange', np.array([0.5060, 0.4070, 0.3005])),
    (8, 'purplish blue', np.array([0.2110, 0.1750, 0.1200])),
    (9, 'moderate red', np.array([0.4533, 0.3058, 0.1977])),
    (10, 'purple', np.array([0.2845, 0.2020, 0.0656])),
    (11, 'yellow green', np.array([0.3800, 0.4887, 0.4429])),
    (12, 'orange yellow', np.array([0.4729, 0.4375, 0.4306])),
    (13, 'blue', np.array([0.1866, 0.1285, 0.0611])),
    (14, 'green', np.array([0.3046, 0.4782, 0.2339])),
    (15, 'red', np.array([0.5385, 0.3129, 0.1200])),
    (16, 'yellow', np.array([0.4480, 0.4703, 0.5910])),
    (17, 'magenta', np.array([0.3635, 0.2325, 0.1977])),
    (18, 'cyan', np.array([0.1958, 0.2519, 0.1977])),
    (19, 'white 9.5 (.05 D)', np.array([0.3101, 0.3163, 0.9001])),
    (20, 'neutral 8 (.23 D)', np.array([0.3101, 0.3163, 0.5910])),
    (21, 'neutral 6.5 (.44 D)', np.array([0.3101, 0.3163, 0.3620])),
    (22, 'neutral 5 (.70 D)', np.array([0.3101, 0.3163, 0.1977])),
    (23, 'neutral 3.5 (1.05 D)', np.array([0.3101, 0.3163, 0.0900])),
    (24, 'black 2 (1.5 D)', np.array([0.3101, 0.3163, 0.0313])))

COLORCHECKER_1976_ILLUMINANT = (
    ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['C'])
"""
*ColorChecker 1976* illuminant.

COLORCHECKER_1976_ILLUMINANT : ndarray
"""

COLORCHECKER_1976 = [ColourChecker_Specification(*x) for x in
                     COLORCHECKER_1976_DATA]
"""
*ColourChecker* developed by *McCamy et al.* at Macbeth, a Division of
Kollmorgen.

COLORCHECKER_1976 : list
"""

COLOURCHECKERS = CaseInsensitiveMapping({
    'BabelColor Average': ColourChecker(
        'BabelColor Average',
        BABELCOLOR_AVERAGE,
        BABELCOLOR_AVERAGE_ILLUMINANT),
    'ColorChecker 2005': ColourChecker(
        'ColorChecker 2005',
        COLORCHECKER_2005,
        COLORCHECKER_2005_ILLUMINANT),
    'ColorChecker 1976': ColourChecker(
        'ColorChecker 1976',
        COLORCHECKER_1976,
        COLORCHECKER_1976_ILLUMINANT)})
"""
Aggregated *ColourCheckers* chromaticity coordinates.

COLOURCHECKERS : CaseInsensitiveMapping
    **{'BabelColor Average', 'ColorChecker 2005', 'ColorChecker 1976'}**

Aliases:

-   'babel_average': 'BabelColor Average'
-   'cc2005': 'ColorChecker 2005'
"""
COLOURCHECKERS['babel_average'] = COLOURCHECKERS['BabelColor Average']
COLOURCHECKERS['cc2005'] = COLOURCHECKERS['ColorChecker 2005']
