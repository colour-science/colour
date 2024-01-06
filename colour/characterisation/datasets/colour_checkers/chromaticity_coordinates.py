"""
Chromaticity Coordinates of the Colour Checkers
===============================================

Defines the chromaticity coordinates of the colour checkers.

Each colour checker data is in the form of an :class:`dict` class instance of
24 samples as follows::

    {'name': 'xyY', ..., 'name': 'xyY'}

The following colour checkers are available:

-   :attr:`colour.characterisation.datasets.colour_checkers.\
chromaticity_coordinates.CCS_COLORCHECKER1976`: *ColorChecker Classic*
    developed by *McCamy et al. (1976)* at Macbeth, a Division of Kollmorgen.
-   :attr:`colour.characterisation.datasets.colour_checkers.\
chromaticity_coordinates.CCS_COLORCHECKER2005`: *ColorChecker Classic*
    reference data from *GretagMacbeth* published in 2005.
-   :attr:`colour.characterisation.datasets.colour_checkers.\
chromaticity_coordinates.CCS_BABELCOLOR_AVERAGE`: Average data derived from
    measurements of 30 *ColorChecker Classic* charts.
-   :attr:`colour.characterisation.datasets.colour_checkers.\
chromaticity_coordinates.CCS_COLORCHECKER24_BEFORE_NOV2014`:
    *ColorChecker Classic* reference data from *X-Rite* published in 2015 and
    matching the data from *GretagMacbeth* published in 2005.
-   :attr:`colour.characterisation.datasets.colour_checkers.\
chromaticity_coordinates.CCS_COLORCHECKER24_AFTER_NOV2014`:
    *ColorChecker Classic* reference data from *X-Rite* published in 2015 and
    matching the *ColorChecker Classic* edition after November 2014.
-   :attr:`colour.characterisation.datasets.colour_checkers.\
chromaticity_coordinates. CCS_TE226_V2`: Reference data from *TE226 V2*.

References
----------
-   :cite:`BabelColor2012b` : BabelColor. (2012). The ColorChecker (since
    1976!). Retrieved September 26, 2014, from
    http://www.babelcolor.com/main_level/ColorChecker.htm
-   :cite:`BabelColor2012c` : BabelColor. (2012). ColorChecker RGB and
    spectra.
    http://www.babelcolor.com/download/ColorChecker_RGB_and_spectra.xls
-   :cite:`ImageEngineering2017` : Image Engineering. (2017). TE226 V2 data
    sheet, from https://www.image-engineering.de/content/products/charts/\
te226/downloads/TE226_D_data_sheet.pdf
-   :cite:`X-Rite2016` : X-Rite. (2016). New color specifications for
    ColorChecker SG and Classic Charts. Retrieved October 29, 2018, from
    http://xritephoto.com/ph_product_overview.aspx?ID=938&Action=Support&\
SupportID=5884#
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np

from colour.colorimetry import CCS_ILLUMINANTS
from colour.hints import NDArrayFloat
from colour.models import Lab_to_XYZ, XYZ_to_xyY
from colour.utilities import CanonicalMapping

__author__ = "Colour Developers, Danny Pascale "
__copyright__ = "Copyright 2013 Colour Developers"
__copyright__ += ", "
__copyright__ += (
    "BabelColor ColorChecker data: Copyright (C) 2004-2012 Danny Pascale "
    "(www.babelcolor.com); used by permission."
)
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "ColourChecker",
    "SAMPLE_LABELS_COLORCHECKER_CLASSIC",
    "DATA_COLORCHECKER1976",
    "CCS_ILLUMINANT_COLORCHECKER1976",
    "CCS_COLORCHECKER1976",
    "DATA_COLORCHECKER2005",
    "CCS_ILLUMINANT_COLORCHECKER2005",
    "CCS_COLORCHECKER2005",
    "DATA_BABELCOLOR_AVERAGE",
    "CCS_ILLUMINANT_BABELCOLOR_AVERAGE",
    "CCS_BABELCOLOR_AVERAGE",
    "DATA_COLORCHECKER24_BEFORE_NOV2014_CIE_LAB",
    "DATA_COLORCHECKER24_BEFORE_NOV2014",
    "CCS_ILLUMINANT_COLORCHECKER24_BEFORE_NOV2014",
    "CCS_COLORCHECKER24_BEFORE_NOV2014",
    "DATA_COLORCHECKER24_AFTER_NOV2014",
    "DATA_COLORCHECKER24_AFTER_NOV2014",
    "CCS_ILLUMINANT_COLORCHECKER24_AFTER_NOV2014",
    "CCS_COLORCHECKER24_AFTER_NOV2014",
    "DATA_TE226_V2",
    "CCS_TE226_V2",
    "CCS_COLOURCHECKERS",
]


class ColourChecker(
    namedtuple(
        "ColourChecker", ("name", "data", "illuminant", "rows", "columns")
    )
):
    """
    *Colour Checker* data.

    Parameters
    ----------
    name
        *Colour Checker* name.
    data
        Chromaticity coordinates in *CIE xyY* colourspace.
    illuminant
        *Colour Checker* illuminant chromaticity coordinates.
    rows
        *Colour Checker* row count.
    columns
        *Colour Checker* column count.
    """


SAMPLE_LABELS_COLORCHECKER_CLASSIC: tuple = (
    "dark skin",
    "light skin",
    "blue sky",
    "foliage",
    "blue flower",
    "bluish green",
    "orange",
    "purplish blue",
    "moderate red",
    "purple",
    "yellow green",
    "orange yellow",
    "blue",
    "green",
    "red",
    "yellow",
    "magenta",
    "cyan",
    "white 9.5 (.05 D)",
    "neutral 8 (.23 D)",
    "neutral 6.5 (.44 D)",
    "neutral 5 (.70 D)",
    "neutral 3.5 (1.05 D)",
    "black 2 (1.5 D)",
)
"""*ColorChecker Classic* sample labels."""

DATA_COLORCHECKER1976: dict = dict(
    zip(
        SAMPLE_LABELS_COLORCHECKER_CLASSIC,
        [
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
        ],
    )
)

CCS_ILLUMINANT_COLORCHECKER1976: NDArrayFloat = CCS_ILLUMINANTS[
    "CIE 1931 2 Degree Standard Observer"
]["C"]
"""*ColorChecker Classic 1976* illuminant."""

CCS_COLORCHECKER1976: ColourChecker = ColourChecker(
    "ColorChecker 1976",
    DATA_COLORCHECKER1976,
    CCS_ILLUMINANT_COLORCHECKER1976,
    4,
    6,
)
"""
*ColorChecker Classic* developed by *McCamy et al.* (1976) at Macbeth, a
Division of Kollmorgen.
"""

DATA_COLORCHECKER2005: dict = dict(
    zip(
        SAMPLE_LABELS_COLORCHECKER_CLASSIC,
        [
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
        ],
    )
)

CCS_ILLUMINANT_COLORCHECKER2005: NDArrayFloat = CCS_ILLUMINANTS[
    "CIE 1931 2 Degree Standard Observer"
]["ICC D50"]
"""*ColorChecker Classic 2005* illuminant."""

CCS_COLORCHECKER2005: ColourChecker = ColourChecker(
    "ColorChecker 2005",
    DATA_COLORCHECKER2005,
    CCS_ILLUMINANT_COLORCHECKER2005,
    4,
    6,
)
"""*ColorChecker Classic* data from *GretagMacbeth (2005)*."""
DATA_BABELCOLOR_AVERAGE: dict = dict(
    zip(
        SAMPLE_LABELS_COLORCHECKER_CLASSIC,
        [
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
        ],
    )
)

CCS_ILLUMINANT_BABELCOLOR_AVERAGE: NDArrayFloat = CCS_ILLUMINANTS[
    "CIE 1931 2 Degree Standard Observer"
]["ICC D50"]
"""*BabelColor Average* illuminant."""

CCS_BABELCOLOR_AVERAGE: ColourChecker = ColourChecker(
    "BabelColor Average",
    DATA_BABELCOLOR_AVERAGE,
    CCS_ILLUMINANT_BABELCOLOR_AVERAGE,
    4,
    6,
)
"""Average data derived from measurements of 30 *ColorChecker Classic* charts."""

DATA_COLORCHECKER24_BEFORE_NOV2014_CIE_LAB: dict = dict(
    zip(
        SAMPLE_LABELS_COLORCHECKER_CLASSIC,
        [
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
        ],
    )
)

DATA_COLORCHECKER24_BEFORE_NOV2014: dict = dict(
    zip(
        SAMPLE_LABELS_COLORCHECKER_CLASSIC,
        XYZ_to_xyY(
            Lab_to_XYZ(
                list(DATA_COLORCHECKER24_BEFORE_NOV2014_CIE_LAB.values()),
                CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"][
                    "ICC D50"
                ],
            )
        ),
    )
)

CCS_ILLUMINANT_COLORCHECKER24_BEFORE_NOV2014: NDArrayFloat = CCS_ILLUMINANTS[
    "CIE 1931 2 Degree Standard Observer"
]["ICC D50"]
"""*ColorChecker24 - Before November 2014* illuminant."""

CCS_COLORCHECKER24_BEFORE_NOV2014: ColourChecker = ColourChecker(
    "ColorChecker24 - Before November 2014",
    DATA_COLORCHECKER24_BEFORE_NOV2014,
    CCS_ILLUMINANT_COLORCHECKER24_BEFORE_NOV2014,
    4,
    6,
)
"""
Reference *ColorChecker Classic* data from *X-Rite (2015)*.

Notes
-----
-   The rounded *ColorChecker24 - Before November 2014* values should match the
    *ColorChecker Classic 2005* values. They are given for reference of the
    original *CIE L\\*a\\*b\\** colourspace values.
"""

DATA_COLORCHECKER24_AFTER_NOV2014_CIE_LAB: dict = dict(
    zip(
        SAMPLE_LABELS_COLORCHECKER_CLASSIC,
        [
            np.array([37.54, 14.37, 14.92]),
            np.array([64.66, 19.27, 17.5]),
            np.array([49.32, -3.82, -22.54]),
            np.array([43.46, -12.74, 22.72]),
            np.array([54.94, 9.61, -24.79]),
            np.array([70.48, -32.26, -0.37]),
            np.array([62.73, 35.83, 56.5]),
            np.array([39.43, 10.75, -45.17]),
            np.array([50.57, 48.64, 16.67]),
            np.array([30.1, 22.54, -20.87]),
            np.array([71.77, -24.13, 58.19]),
            np.array([71.51, 18.24, 67.37]),
            np.array([28.37, 15.42, -49.8]),
            np.array([54.38, -39.72, 32.27]),
            np.array([42.43, 51.05, 28.62]),
            np.array([81.8, 2.67, 80.41]),
            np.array([50.63, 51.28, -14.12]),
            np.array([49.57, -29.71, -28.32]),
            np.array([95.19, -1.03, 2.93]),
            np.array([81.29, -0.57, 0.44]),
            np.array([66.89, -0.75, -0.06]),
            np.array([50.76, -0.13, 0.14]),
            np.array([35.63, -0.46, -0.48]),
            np.array([20.64, 0.07, -0.46]),
        ],
    )
)

DATA_COLORCHECKER24_AFTER_NOV2014: dict = dict(
    zip(
        SAMPLE_LABELS_COLORCHECKER_CLASSIC,
        XYZ_to_xyY(
            Lab_to_XYZ(
                list(DATA_COLORCHECKER24_AFTER_NOV2014_CIE_LAB.values()),
                CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"][
                    "ICC D50"
                ],
            )
        ),
    )
)

CCS_ILLUMINANT_COLORCHECKER24_AFTER_NOV2014: NDArrayFloat = CCS_ILLUMINANTS[
    "CIE 1931 2 Degree Standard Observer"
]["ICC D50"]
"""*ColorChecker24 - After November 2014* illuminant."""

CCS_COLORCHECKER24_AFTER_NOV2014: ColourChecker = ColourChecker(
    "ColorChecker24 - After November 2014",
    DATA_COLORCHECKER24_AFTER_NOV2014,
    CCS_ILLUMINANT_COLORCHECKER24_AFTER_NOV2014,
    4,
    6,
)
"""
Reference *ColorChecker Classic* data from *X-Rite (2015)* and matching the
*ColorChecker Classic* edition after November 2014.
"""

DATA_TE226_V2_CIE_XYZ: dict = {
    "dark skin": np.array([0.1278, 0.1074, 0.0726]),
    "light skin": np.array([0.4945, 0.4484, 0.3586]),
    "blue sky": np.array([0.1459, 0.1690, 0.2925]),
    "foliage": np.array([0.0714, 0.1243, 0.0254]),
    "blue flower": np.array([0.4470, 0.4039, 0.7304]),
    "bluish green": np.array([0.3921, 0.5420, 0.6113]),
    "orange": np.array([0.4574, 0.3628, 0.0624]),
    "purplish blue": np.array([0.2979, 0.3180, 0.8481]),
    "moderate red": np.array([0.3884, 0.2794, 0.1886]),
    "purple": np.array([0.1324, 0.0796, 0.3824]),
    "yellow green": np.array([0.3399, 0.5786, 0.1360]),
    "orange yellow": np.array([0.5417, 0.4677, 0.0644]),
    "blue": np.array([0.0859, 0.0361, 0.4728]),
    "green": np.array([0.1000, 0.2297, 0.0530]),
    "red": np.array([0.3594, 0.1796, 0.0197]),
    "yellow": np.array([0.5236, 0.5972, 0.0368]),
    "magenta": np.array([0.4253, 0.2050, 0.5369]),
    "cyan": np.array([0.4942, 0.6119, 1.0304]),
    "patch 19": np.array([0.2646, 0.2542, 0.1631]),
    "patch 20": np.array([0.7921, 0.7560, 0.5988]),
    "patch 21": np.array([0.4409, 0.4004, 0.3366]),
    "patch 22": np.array([0.1546, 0.3395, 0.1016]),
    "patch 23": np.array([0.3182, 0.3950, 0.5857]),
    "patch 24": np.array([0.5920, 0.5751, 0.9892]),
    "patch 25": np.array([0.4287, 0.2583, 0.0444]),
    "patch 26": np.array([0.4282, 0.5757, 0.4770]),
    "patch 27": np.array([0.1697, 0.1294, 0.7026]),
    "patch 28": np.array([0.2143, 0.1564, 0.1908]),
    "patch 29": np.array([0.1659, 0.3876, 0.3945]),
    "patch 30": np.array([0.1869, 0.1093, 0.7069]),
    "patch 31": np.array([0.3316, 0.1596, 0.1714]),
    "patch 32": np.array([0.8298, 0.8910, 0.5199]),
    "patch 33": np.array([0.1412, 0.1758, 0.4643]),
    "patch 34": np.array([0.0153, 0.0668, 0.0694]),
    "patch 35": np.array([0.6053, 0.5088, 0.1593]),
    "patch 36": np.array([0.4217, 0.4459, 0.3173]),
    "white": np.array([0.9505, 1.0000, 1.0888]),
    "neutral 87": np.array([0.8331, 0.8801, 0.9576]),
    "neutral 63": np.array([0.6050, 0.6401, 0.6958]),
    "neutral 44": np.array([0.4119, 0.4358, 0.4724]),
    "neutral 28": np.array([0.2638, 0.2798, 0.3018]),
    "neutral 15": np.array([0.1405, 0.1489, 0.1598]),
    "neutral 7": np.array([0.0628, 0.0665, 0.0701]),
    "neutral 2": np.array([0.0190, 0.0202, 0.0202]),
    "neutral < 0.1": np.array([0.0000, 0.0001, 0.0000]),
}

DATA_TE226_V2: dict = dict(
    zip(
        tuple(DATA_TE226_V2_CIE_XYZ.keys()),
        XYZ_to_xyY(list(DATA_TE226_V2_CIE_XYZ.values())),
    )
)

CCS_ILLUMINANT_TE226_V2: NDArrayFloat = CCS_ILLUMINANTS[
    "CIE 1931 2 Degree Standard Observer"
]["D65"]
"""*TE226 V2* illuminant."""

CCS_TE226_V2: ColourChecker = ColourChecker(
    "TE226 V2", DATA_TE226_V2, CCS_ILLUMINANT_TE226_V2, 5, 9
)
"""
Reference data from *TE226 V2*. Transparent color rendition test chart
for HDTV cameras, in addition to known colors from "ColorChecker", the test
chart contains colors which are critical in reproduction.
"""

CCS_COLOURCHECKERS: CanonicalMapping = CanonicalMapping(
    {
        "ColorChecker 1976": CCS_COLORCHECKER1976,
        "ColorChecker 2005": CCS_COLORCHECKER2005,
        "BabelColor Average": CCS_BABELCOLOR_AVERAGE,
        "ColorChecker24 - Before November 2014": CCS_COLORCHECKER24_BEFORE_NOV2014,
        "ColorChecker24 - After November 2014": CCS_COLORCHECKER24_AFTER_NOV2014,
        "TE226 V2": CCS_TE226_V2,
    }
)
CCS_COLOURCHECKERS.__doc__ = """
Chromaticity coordinates of the colour checkers.

References
----------
:cite:`BabelColor2012b`, :cite:`BabelColor2012c`,
:cite:`ImageEngineering2017`, :cite:`X-Rite2016`

Aliases:

-   'babel_average': 'BabelColor Average'
-   'cc2005': 'ColorChecker 2005'
-   'ccb2014': 'ColorChecker24 - Before November 2014'
-   'cca2014': 'ColorChecker24 - After November 2014'
"""
CCS_COLOURCHECKERS["babel_average"] = CCS_COLOURCHECKERS["BabelColor Average"]
CCS_COLOURCHECKERS["cc2005"] = CCS_COLOURCHECKERS["ColorChecker 2005"]
CCS_COLOURCHECKERS["ccb2014"] = CCS_COLOURCHECKERS[
    "ColorChecker24 - Before November 2014"
]
CCS_COLOURCHECKERS["cca2014"] = CCS_COLOURCHECKERS[
    "ColorChecker24 - After November 2014"
]
