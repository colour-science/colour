# -*- coding: utf-8 -*-
"""
Colour Checkers Chromaticity Coordinates
========================================

Defines *Colour Checkers* chromaticity coordinates in *CIE xyY* colourspace.

Each *Colour Checker* data is in the form of a list of an :class:`OrderedDict`
class instance of 24 samples as follows::

    {'name': 'xyY', ..., 'name': 'xyY'}

The following *Colour Checkers* data is available:

-   :attr:`colour.characterisation.datasets.colour_checkers.\
chromaticity_coordinates.COLORCHECKER_1976`: *Colour Checker* developed by
    *McCamy et al.* at Macbeth, a Division of Kollmorgen.
-   :attr:`colour.characterisation.datasets.colour_checkers.\
chromaticity_coordinates.COLORCHECKER_2005`: Reference data from
    *GretagMacbeth* published in 2005.
-   :attr:`colour.characterisation.datasets.colour_checkers.\
chromaticity_coordinates.BABELCOLOR_AVERAGE`: Average data derived from
    measurements of 30 *Colour Checker* charts.
-   :attr:`colour.characterisation.datasets.colour_checkers.\
chromaticity_coordinates.COLORCHECKER24_BEFORE_NOV2014`: Reference data from
    *X-Rite* published in 2015 and matching the data from *GretagMacbeth*
    published in 2005.
-   :attr:`colour.characterisation.datasets.colour_checkers.\
chromaticity_coordinates.COLORCHECKER24_AFTER_NOV2014`: Reference data from
    *X-Rite* published in 2015 and matching the *Colour Checker* edition after
    November 2014.

See Also
--------
`Colour Fitting Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/characterisation/fitting.ipynb>`_

References
----------
-   :cite:`BabelColor2012b` : BabelColor. (2012). The ColorChecker
    (since 1976!). Retrieved September 26, 2014, from
    http://www.babelcolor.com/main_level/ColorChecker.htm
-   :cite:`BabelColor2012c` : BabelColor. (2012). ColorChecker RGB and spectra.
    Retrieved from http://www.babelcolor.com/download/\
ColorChecker_RGB_and_spectra.xls
-   :cite:`X-Rite2015` : X-Rite. (2015). New color specifications for
    ColorChecker SG and Classic Charts. Retrieved October 29, 2018,
    from http://xritephoto.com/ph_product_overview.aspx?\
ID=938&Action=Support&SupportID=5884#
"""

from __future__ import division, unicode_literals

import numpy as np
from collections import OrderedDict, namedtuple

from colour.colorimetry import ILLUMINANTS
from colour.models import Lab_to_XYZ, XYZ_to_xyY
from colour.utilities import CaseInsensitiveMapping

__author__ = 'Colour Developers, Danny Pascale '
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__copyright__ += ', '
__copyright__ += (
    'BabelColor ColorChecker data: Copyright (C) 2004-2012 Danny Pascale '
    '(www.babelcolor.com); used by permission.')
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'ColourChecker', 'COLORCHECKER_CLASSIC_SAMPLE_LABELS',
    'COLORCHECKER_1976_DATA', 'COLORCHECKER_1976_ILLUMINANT',
    'COLORCHECKER_1976', 'COLORCHECKER_2005_DATA',
    'COLORCHECKER_2005_ILLUMINANT', 'COLORCHECKER_2005',
    'BABELCOLOR_AVERAGE_DATA', 'BABELCOLOR_AVERAGE_ILLUMINANT',
    'BABELCOLOR_AVERAGE', 'COLORCHECKER24_BEFORE_NOV2014_LAB_DATA',
    'COLORCHECKER24_BEFORE_NOV2014_DATA',
    'COLORCHECKER24_BEFORE_NOV2014_ILLUMINANT',
    'COLORCHECKER24_BEFORE_NOV2014', 'COLORCHECKER24_AFTER_NOV2014_LAB_DATA',
    'COLORCHECKER24_AFTER_NOV2014_DATA',
    'COLORCHECKER24_AFTER_NOV2014_ILLUMINANT', 'COLORCHECKER24_AFTER_NOV2014',
    'COLOURCHECKERS'
]


class ColourChecker(
        namedtuple('ColourChecker', ('name', 'data', 'illuminant'))):
    """
    *Colour Checker* data.

    Parameters
    ----------
    name : unicode
        *Colour Checker* name.
    data : OrderedDict
        chromaticity coordinates in *CIE xyY* colourspace.
    illuminant : array_like
        *Colour Checker* illuminant chromaticity coordinates.
    """


COLORCHECKER_CLASSIC_SAMPLE_LABELS = (
    'dark skin',
    'light skin',
    'blue sky',
    'foliage',
    'blue flower',
    'bluish green',
    'orange',
    'purplish blue',
    'moderate red',
    'purple',
    'yellow green',
    'orange yellow',
    'blue',
    'green',
    'red',
    'yellow',
    'magenta',
    'cyan',
    'white 9.5 (.05 D)',
    'neutral 8 (.23 D)',
    'neutral 6.5 (.44 D)',
    'neutral 5 (.70 D)',
    'neutral 3.5 (1.05 D)',
    'black 2 (1.5 D)',
)
"""
*ColorChecker Classic* illuminant.

COLORCHECKER_CLASSIC_SAMPLE_LABELS : tuple
"""

COLORCHECKER_1976_DATA = OrderedDict(
    zip(COLORCHECKER_CLASSIC_SAMPLE_LABELS, [
        np.array([0.4002, 0.3504, 0.1005]),
        np.array([0.3773, 0.3446, 0.3582]),
        np.array([0.2470, 0.2514, 0.1933]),
        np.array([0.3372, 0.4220, 0.1329]),
        np.array([0.2651, 0.2400, 0.2427]),
        np.array([0.2608, 0.3430, 0.4306]),
        np.array([0.5060, 0.4070, 0.3005]),
        np.array([0.2110, 0.1750, 0.1200]),
        np.array([0.4533, 0.3058, 0.1977]),
        np.array([0.2845, 0.2020, 0.0656]),
        np.array([0.3800, 0.4887, 0.4429]),
        np.array([0.4729, 0.4375, 0.4306]),
        np.array([0.1866, 0.1285, 0.0611]),
        np.array([0.3046, 0.4782, 0.2339]),
        np.array([0.5385, 0.3129, 0.1200]),
        np.array([0.4480, 0.4703, 0.5910]),
        np.array([0.3635, 0.2325, 0.1977]),
        np.array([0.1958, 0.2519, 0.1977]),
        np.array([0.3101, 0.3163, 0.9001]),
        np.array([0.3101, 0.3163, 0.5910]),
        np.array([0.3101, 0.3163, 0.3620]),
        np.array([0.3101, 0.3163, 0.1977]),
        np.array([0.3101, 0.3163, 0.0900]),
        np.array([0.3101, 0.3163, 0.0313]),
    ]))

COLORCHECKER_1976_ILLUMINANT = (
    ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['C'])
"""
*ColorChecker 1976* illuminant.

COLORCHECKER_1976_ILLUMINANT : ndarray
"""

COLORCHECKER_1976 = ColourChecker('ColorChecker 1976', COLORCHECKER_1976_DATA,
                                  COLORCHECKER_1976_ILLUMINANT)
"""
*Colour Checker* developed by *McCamy et al.* at Macbeth, a Division of
Kollmorgen.

COLORCHECKER_1976 : ColourChecker
"""

COLORCHECKER_2005_DATA = OrderedDict(
    zip(COLORCHECKER_CLASSIC_SAMPLE_LABELS, [
        np.array([0.4316, 0.3777, 0.1008]),
        np.array([0.4197, 0.3744, 0.3495]),
        np.array([0.2760, 0.3016, 0.1836]),
        np.array([0.3703, 0.4499, 0.1325]),
        np.array([0.2999, 0.2856, 0.2304]),
        np.array([0.2848, 0.3911, 0.4178]),
        np.array([0.5295, 0.4055, 0.3118]),
        np.array([0.2305, 0.2106, 0.1126]),
        np.array([0.5012, 0.3273, 0.1938]),
        np.array([0.3319, 0.2482, 0.0637]),
        np.array([0.3984, 0.5008, 0.4446]),
        np.array([0.4957, 0.4427, 0.4357]),
        np.array([0.2018, 0.1692, 0.0575]),
        np.array([0.3253, 0.5032, 0.2318]),
        np.array([0.5686, 0.3303, 0.1257]),
        np.array([0.4697, 0.4734, 0.5981]),
        np.array([0.4159, 0.2688, 0.2009]),
        np.array([0.2131, 0.3023, 0.1930]),
        np.array([0.3469, 0.3608, 0.9131]),
        np.array([0.3440, 0.3584, 0.5894]),
        np.array([0.3432, 0.3581, 0.3632]),
        np.array([0.3446, 0.3579, 0.1915]),
        np.array([0.3401, 0.3548, 0.0883]),
        np.array([0.3406, 0.3537, 0.0311]),
    ]))

COLORCHECKER_2005_ILLUMINANT = (
    ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['ICC D50'])
"""
*ColorChecker 2005* illuminant.

COLORCHECKER_2005_ILLUMINANT : ndarray
"""

COLORCHECKER_2005 = ColourChecker('ColorChecker 2005', COLORCHECKER_2005_DATA,
                                  COLORCHECKER_2005_ILLUMINANT)
"""
Reference data from *GretagMacbeth (2005)*.

COLORCHECKER_2005 : ColourChecker
"""
BABELCOLOR_AVERAGE_DATA = OrderedDict(
    zip(COLORCHECKER_CLASSIC_SAMPLE_LABELS, [
        np.array([0.4325, 0.3788, 0.1034]),
        np.array([0.4191, 0.3748, 0.3525]),
        np.array([0.2761, 0.3004, 0.1847]),
        np.array([0.3700, 0.4501, 0.1335]),
        np.array([0.3020, 0.2877, 0.2324]),
        np.array([0.2856, 0.3910, 0.4174]),
        np.array([0.5291, 0.4075, 0.3117]),
        np.array([0.2339, 0.2155, 0.1140]),
        np.array([0.5008, 0.3293, 0.1979]),
        np.array([0.3326, 0.2556, 0.0644]),
        np.array([0.3989, 0.4998, 0.4435]),
        np.array([0.4962, 0.4428, 0.4358]),
        np.array([0.2040, 0.1696, 0.0579]),
        np.array([0.3270, 0.5033, 0.2307]),
        np.array([0.5709, 0.3298, 0.1268]),
        np.array([0.4694, 0.4732, 0.6081]),
        np.array([0.4177, 0.2704, 0.2007]),
        np.array([0.2151, 0.3037, 0.1903]),
        np.array([0.3488, 0.3628, 0.9129]),
        np.array([0.3451, 0.3596, 0.5885]),
        np.array([0.3446, 0.3590, 0.3595]),
        np.array([0.3438, 0.3589, 0.1912]),
        np.array([0.3423, 0.3576, 0.0893]),
        np.array([0.3439, 0.3565, 0.0320]),
    ]))

BABELCOLOR_AVERAGE_ILLUMINANT = (
    ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['ICC D50'])
"""
*BabelColor Average* illuminant.

BABELCOLOR_AVERAGE_ILLUMINANT : ndarray
"""

BABELCOLOR_AVERAGE = ColourChecker('BabelColor Average',
                                   BABELCOLOR_AVERAGE_DATA,
                                   BABELCOLOR_AVERAGE_ILLUMINANT)
"""
Average data derived from measurements of 30 *Colour Checker* charts.

BABELCOLOR_AVERAGE : ColourChecker
"""

COLORCHECKER24_BEFORE_NOV2014_LAB_DATA = OrderedDict(
    zip(COLORCHECKER_CLASSIC_SAMPLE_LABELS, [
        np.array([37.986, 13.555, 14.059]),
        np.array([65.711, 18.13, 17.81]),
        np.array([49.927, -4.88, -21.905]),
        np.array([43.139, -13.095, 21.905]),
        np.array([55.112, 8.844, -25.399]),
        np.array([70.719, -33.397, -0.199]),
        np.array([62.661, 36.067, 57.096]),
        np.array([40.02, 10.41, -45.964]),
        np.array([51.124, 48.239, 16.248]),
        np.array([30.325, 22.976, -21.587]),
        np.array([72.532, -23.709, 57.255]),
        np.array([71.941, 19.363, 67.857]),
        np.array([28.778, 14.179, -50.297]),
        np.array([55.261, -38.342, 31.37]),
        np.array([42.101, 53.378, 28.19]),
        np.array([81.733, 4.039, 79.819]),
        np.array([51.935, 49.986, -14.574]),
        np.array([51.038, -28.631, -28.638]),
        np.array([96.539, -0.425, 1.186]),
        np.array([81.257, -0.638, -0.335]),
        np.array([66.766, -0.734, -0.504]),
        np.array([50.867, -0.153, -0.27]),
        np.array([35.656, -0.421, -1.231]),
        np.array([20.461, -0.079, -0.973]),
    ]))
"""
*ColorChecker24 - Before November 2014* illuminant.

Notes
-----
-   *X-Rite* data is given as *CIE L\\*a\\*b\\** colourspace values under
    *CIE Illuminant D Series D50* for the
    *CIE 1931 2 Degree Standard Observer*.

COLORCHECKER24_BEFORE_NOV2014_LAB_DATA : ndarray
"""

COLORCHECKER24_BEFORE_NOV2014_DATA = OrderedDict(
    zip(
        COLORCHECKER_CLASSIC_SAMPLE_LABELS,
        XYZ_to_xyY(
            Lab_to_XYZ(
                list(COLORCHECKER24_BEFORE_NOV2014_LAB_DATA.values()),
                ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][
                    'ICC D50']))))

COLORCHECKER24_BEFORE_NOV2014_ILLUMINANT = (
    ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['ICC D50'])
"""
*ColorChecker24 - Before November 2014* illuminant.

COLORCHECKER24_BEFORE_NOV2014_ILLUMINANT : ndarray
"""

COLORCHECKER24_BEFORE_NOV2014 = ColourChecker(
    'ColorChecker24 - Before November 2014',
    COLORCHECKER24_BEFORE_NOV2014_DATA,
    COLORCHECKER24_BEFORE_NOV2014_ILLUMINANT)
"""
Reference *Colour Checker* data from *X-Rite (2015)*.

Notes
-----
-   The rounded *ColorChecker24 - Before November 2014* values should match the
    *ColorChecker 2005* values. They are given for reference of the original
    *CIE L\\*a\\*b\\** colourspace values.

COLORCHECKER24_BEFORE_NOV2014 : ColourChecker
"""

COLORCHECKER24_AFTER_NOV2014_LAB_DATA = OrderedDict((
    ('dark skin', np.array([37.54, 14.37, 14.92])),
    ('light skin', np.array([64.66, 19.27, 17.5])),
    ('blue sky', np.array([49.32, -3.82, -22.54])),
    ('foliage', np.array([43.46, -12.74, 22.72])),
    ('blue flower', np.array([54.94, 9.61, -24.79])),
    ('bluish green', np.array([70.48, -32.26, -0.37])),
    ('orange', np.array([62.73, 35.83, 56.5])),
    ('purplish blue', np.array([39.43, 10.75, -45.17])),
    ('moderate red', np.array([50.57, 48.64, 16.67])),
    ('purple', np.array([30.1, 22.54, -20.87])),
    ('yellow green', np.array([71.77, -24.13, 58.19])),
    ('orange yellow', np.array([71.51, 18.24, 67.37])),
    ('blue', np.array([28.37, 15.42, -49.8])),
    ('green', np.array([54.38, -39.72, 32.27])),
    ('red', np.array([42.43, 51.05, 28.62])),
    ('yellow', np.array([81.8, 2.67, 80.41])),
    ('magenta', np.array([50.63, 51.28, -14.12])),
    ('cyan', np.array([49.57, -29.71, -28.32])),
    ('white 9.5 (.05 D)', np.array([95.19, -1.03, 2.93])),
    ('neutral 8 (.23 D)', np.array([81.29, -0.57, 0.44])),
    ('neutral 6.5 (.44 D)', np.array([66.89, -0.75, -0.06])),
    ('neutral 5 (.70 D)', np.array([50.76, -0.13, 0.14])),
    ('neutral 3.5 (1.05 D)', np.array([35.63, -0.46, -0.48])),
    ('black 2 (1.5 D)', np.array([20.64, 0.07, -0.46])),
))
"""
*ColorChecker24 - After November 2014* illuminant.

Notes
-----
-   *X-Rite* data is given as *CIE L\\*a\\*b\\** colourspace values under
    *CIE Illuminant D Series D50* for the
    *CIE 1931 2 Degree Standard Observer*.

COLORCHECKER24_AFTER_NOV2014_LAB_DATA : ndarray
"""

COLORCHECKER24_AFTER_NOV2014_DATA = OrderedDict(
    zip(
        COLORCHECKER24_AFTER_NOV2014_LAB_DATA.keys(),
        XYZ_to_xyY(
            Lab_to_XYZ(
                list(COLORCHECKER24_AFTER_NOV2014_LAB_DATA.values()),
                ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][
                    'ICC D50']))))

COLORCHECKER24_AFTER_NOV2014_ILLUMINANT = (
    ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['ICC D50'])
"""
*ColorChecker24 - After November 2014* illuminant.

COLORCHECKER24_AFTER_NOV2014_ILLUMINANT : ndarray
"""

COLORCHECKER24_AFTER_NOV2014 = ColourChecker(
    'ColorChecker24 - After November 2014', COLORCHECKER24_AFTER_NOV2014_DATA,
    COLORCHECKER24_AFTER_NOV2014_ILLUMINANT)
"""
Reference *Colour Checker* data from *X-Rite (2015)* and matching the
*Colour Checker* edition after November 2014.

COLORCHECKER24_AFTER_NOV2014 : ColourChecker
"""

COLOURCHECKERS = CaseInsensitiveMapping({
    'ColorChecker 1976': COLORCHECKER_1976,
    'ColorChecker 2005': COLORCHECKER_2005,
    'BabelColor Average': BABELCOLOR_AVERAGE,
    'ColorChecker24 - Before November 2014': COLORCHECKER24_BEFORE_NOV2014,
    'ColorChecker24 - After November 2014': COLORCHECKER24_AFTER_NOV2014,
})
COLOURCHECKERS.__doc__ = """
Aggregated *Colour Checkers* chromaticity coordinates.

References
----------
:cite:`BabelColor2012b`, :cite:`BabelColor2012c`, :cite:`X-Rite2015`

COLOURCHECKERS : CaseInsensitiveMapping
    **{'ColorChecker 1976', 'ColorChecker 2005', 'BabelColor Average',
    'ColorChecker24 - Before November 2014',
    'ColorChecker24 - After November 2014'}**

Aliases:

-   'babel_average': 'BabelColor Average'
-   'cc2005': 'ColorChecker 2005'
-   'ccb2014': 'ColorChecker24 - Before November 2014'
-   'cca2014': 'ColorChecker24 - After November 2014'
"""
COLOURCHECKERS['babel_average'] = COLOURCHECKERS['BabelColor Average']
COLOURCHECKERS['cc2005'] = COLOURCHECKERS['ColorChecker 2005']
COLOURCHECKERS['ccb2014'] = COLOURCHECKERS[
    'ColorChecker24 - Before November 2014']
COLOURCHECKERS['cca2014'] = COLOURCHECKERS[
    'ColorChecker24 - After November 2014']
