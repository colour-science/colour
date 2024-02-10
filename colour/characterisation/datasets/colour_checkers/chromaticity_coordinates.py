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
    *ColorChecker Classic* reference data from *X-Rite* published in 2016 and
    matching the data from *GretagMacbeth* published in 2005.
-   :attr:`colour.characterisation.datasets.colour_checkers.\
chromaticity_coordinates.CCS_COLORCHECKER24_AFTER_NOV2014`:
    *ColorChecker Classic* reference data from *X-Rite* published in 2016 and
    matching the *ColorChecker Classic* edition after November 2014.
-   :attr:`colour.characterisation.datasets.colour_checkers.\
chromaticity_coordinates.CCS_COLORCHECKERSG_BEFORE_NOV2014`:
    *ColorChecker SG* reference data from *X-Rite* published in 2016
-   :attr:`colour.characterisation.datasets.colour_checkers.\
chromaticity_coordinates.CCS_COLORCHECKERSG_AFTER_NOV2014`:
    *ColorChecker SG* reference data from *X-Rite* published in 2016
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
    "DATA_COLORCHECKER24_AFTER_NOV2014_CIE_LAB",
    "DATA_COLORCHECKER24_AFTER_NOV2014",
    "CCS_ILLUMINANT_COLORCHECKER24_AFTER_NOV2014",
    "CCS_COLORCHECKER24_AFTER_NOV2014",
    "SAMPLE_LABELS_COLORCHECKER_SG",
    "DATA_COLORCHECKERSG_BEFORE_NOV2014_CIE_LAB",
    "DATA_COLORCHECKERSG_BEFORE_NOV2014",
    "CCS_ILLUMINANT_COLORCHECKERSG_BEFORE_NOV2014",
    "CCS_COLORCHECKERSG_BEFORE_NOV2014",
    "DATA_COLORCHECKERSG_AFTER_NOV2014_CIE_LAB",
    "DATA_COLORCHECKERSG_AFTER_NOV2014",
    "CCS_ILLUMINANT_COLORCHECKERSG_AFTER_NOV2014",
    "CCS_COLORCHECKERSG_AFTER_NOV2014",
    "DATA_TE226_V2_CIE_XYZ",
    "DATA_TE226_V2",
    "CCS_ILLUMINANT_TE226_V2",
    "CCS_TE226_V2",
    "CCS_COLOURCHECKERS",
]


class ColourChecker(
    namedtuple("ColourChecker", ("name", "data", "illuminant", "rows", "columns"))
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
                CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["ICC D50"],
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
Reference *ColorChecker Classic* data from *X-Rite (2016)*.

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
                CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["ICC D50"],
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
Reference *ColorChecker Classic* data from *X-Rite (2016)* and matching the
*ColorChecker Classic* edition after November 2014.
"""

SAMPLE_LABELS_COLORCHECKER_SG: tuple = (
    "A1",
    "A2",
    "A3",
    "A4",
    "A5",
    "A6",
    "A7",
    "A8",
    "A9",
    "A10",
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B9",
    "B10",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "C7",
    "C8",
    "C9",
    "C10",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "D6",
    "D7",
    "D8",
    "D9",
    "D10",
    "E1",
    "E2",
    "E3",
    "E4",
    "E5",
    "E6",
    "E7",
    "E8",
    "E9",
    "E10",
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
    "F6",
    "F7",
    "F8",
    "F9",
    "F10",
    "G1",
    "G2",
    "G3",
    "G4",
    "G5",
    "G6",
    "G7",
    "G8",
    "G9",
    "G10",
    "H1",
    "H2",
    "H3",
    "H4",
    "H5",
    "H6",
    "H7",
    "H8",
    "H9",
    "H10",
    "I1",
    "I2",
    "I3",
    "I4",
    "I5",
    "I6",
    "I7",
    "I8",
    "I9",
    "I10",
    "J1",
    "J2",
    "J3",
    "J4",
    "J5",
    "J6",
    "J7",
    "J8",
    "J9",
    "J10",
    "K1",
    "K2",
    "K3",
    "K4",
    "K5",
    "K6",
    "K7",
    "K8",
    "K9",
    "K10",
    "L1",
    "L2",
    "L3",
    "L4",
    "L5",
    "L6",
    "L7",
    "L8",
    "L9",
    "L10",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",
    "M10",
    "N1",
    "N2",
    "N3",
    "N4",
    "N5",
    "N6",
    "N7",
    "N8",
    "N9",
    "N10",
)
"""*ColorChecker SG* sample labels."""

DATA_COLORCHECKERSG_BEFORE_NOV2014_CIE_LAB: dict = dict(
    zip(
        SAMPLE_LABELS_COLORCHECKER_SG,
        [
            np.array([96.55, -0.91, 0.57]),
            np.array([6.43, -0.06, -0.41]),
            np.array([49.7, -0.18, 0.03]),
            np.array([96.5, -0.89, 0.59]),
            np.array([6.5, -0.06, -0.44]),
            np.array([49.66, -0.2, 0.01]),
            np.array([96.52, -0.91, 0.58]),
            np.array([6.49, -0.02, -0.28]),
            np.array([49.72, -0.2, 0.04]),
            np.array([96.43, -0.91, 0.67]),
            np.array([49.72, -0.19, 0.02]),
            np.array([32.6, 51.58, -10.85]),
            np.array([60.75, 26.22, -18.69]),
            np.array([28.69, 48.28, -39]),
            np.array([49.38, -15.43, -48.48]),
            np.array([60.63, -30.77, -26.23]),
            np.array([19.29, -26.37, -6.15]),
            np.array([60.15, -41.77, -12.6]),
            np.array([21.42, 1.67, 8.79]),
            np.array([49.69, -0.2, 0.01]),
            np.array([6.5, -0.03, -0.67]),
            np.array([21.82, 17.33, -18.35]),
            np.array([41.53, 18.48, -37.26]),
            np.array([19.99, -0.16, -36.29]),
            np.array([60.16, -18.45, -31.42]),
            np.array([19.94, -17.92, -20.96]),
            np.array([60.68, -6.05, -32.81]),
            np.array([50.81, -49.8, -9.63]),
            np.array([60.65, -39.77, 20.76]),
            np.array([6.53, -0.03, -0.43]),
            np.array([96.56, -0.91, 0.59]),
            np.array([84.19, -1.95, -8.23]),
            np.array([84.75, 14.55, 0.23]),
            np.array([84.87, -19.07, -0.82]),
            np.array([85.15, 13.48, 6.82]),
            np.array([84.17, -10.45, 26.78]),
            np.array([61.74, 31.06, 36.42]),
            np.array([64.37, 20.82, 18.92]),
            np.array([50.4, -53.22, 14.62]),
            np.array([96.51, -0.89, 0.65]),
            np.array([49.74, -0.19, 0.03]),
            np.array([31.91, 18.62, 21.99]),
            np.array([60.74, 38.66, 70.97]),
            np.array([19.35, 22.23, -58.86]),
            np.array([96.52, -0.91, 0.62]),
            np.array([6.66, 0, -0.3]),
            np.array([76.51, 20.81, 22.72]),
            np.array([72.79, 29.15, 24.18]),
            np.array([22.33, -20.7, 5.75]),
            np.array([49.7, -0.19, 0.01]),
            np.array([6.53, -0.05, -0.61]),
            np.array([63.42, 20.19, 19.22]),
            np.array([34.94, 11.64, -50.7]),
            np.array([52.03, -44.15, 39.04]),
            np.array([79.43, 0.29, -0.17]),
            np.array([30.67, -0.14, -0.53]),
            np.array([63.6, 14.44, 26.07]),
            np.array([64.37, 14.5, 17.05]),
            np.array([60.01, -44.33, 8.49]),
            np.array([6.63, -0.01, -0.47]),
            np.array([96.56, -0.93, 0.59]),
            np.array([46.37, -5.09, -24.46]),
            np.array([47.08, 52.97, 20.49]),
            np.array([36.04, 64.92, 38.51]),
            np.array([65.05, 0, -0.32]),
            np.array([40.14, -0.19, -0.38]),
            np.array([43.77, 16.46, 27.12]),
            np.array([64.39, 17, 16.59]),
            np.array([60.79, -29.74, 41.5]),
            np.array([96.48, -0.89, 0.64]),
            np.array([49.75, -0.21, 0.01]),
            np.array([38.18, -16.99, 30.87]),
            np.array([21.31, 29.14, -27.51]),
            np.array([80.57, 3.85, 89.61]),
            np.array([49.71, -0.2, 0.03]),
            np.array([60.27, 0.08, -0.41]),
            np.array([67.34, 14.45, 16.9]),
            np.array([64.69, 16.95, 18.57]),
            np.array([51.12, -49.31, 44.41]),
            np.array([49.7, -0.2, 0.02]),
            np.array([6.67, -0.05, -0.64]),
            np.array([51.56, 9.16, -26.88]),
            np.array([70.83, -24.26, 64.77]),
            np.array([48.06, 55.33, -15.61]),
            np.array([35.26, -0.09, -0.24]),
            np.array([75.16, 0.25, -0.2]),
            np.array([44.54, 26.27, 38.93]),
            np.array([35.91, 16.59, 26.46]),
            np.array([61.49, -52.73, 47.3]),
            np.array([6.59, -0.05, -0.5]),
            np.array([96.58, -0.9, 0.61]),
            np.array([68.93, -34.58, -0.34]),
            np.array([69.65, 20.09, 78.57]),
            np.array([47.79, -33.18, -30.21]),
            np.array([15.94, -0.42, -1.2]),
            np.array([89.02, -0.36, -0.48]),
            np.array([63.43, 25.44, 26.25]),
            np.array([65.75, 22.06, 27.82]),
            np.array([61.47, 17.1, 50.72]),
            np.array([96.53, -0.89, 0.66]),
            np.array([49.79, -0.2, 0.03]),
            np.array([85.17, 10.89, 17.26]),
            np.array([89.74, -16.52, 6.19]),
            np.array([84.55, 5.07, -6.12]),
            np.array([84.02, -13.87, -8.72]),
            np.array([70.76, 0.07, -0.35]),
            np.array([45.59, -0.05, 0.23]),
            np.array([20.3, 0.07, -0.32]),
            np.array([61.79, -13.41, 55.42]),
            np.array([49.72, -0.19, 0.02]),
            np.array([6.77, -0.05, -0.44]),
            np.array([21.85, 34.37, 7.83]),
            np.array([42.66, 67.43, 48.42]),
            np.array([60.33, 36.56, 3.56]),
            np.array([61.22, 36.61, 17.32]),
            np.array([62.07, 52.8, 77.14]),
            np.array([72.42, -9.82, 89.66]),
            np.array([62.03, 3.53, 57.01]),
            np.array([71.95, -27.34, 73.69]),
            np.array([6.59, -0.04, -0.45]),
            np.array([49.77, -0.19, 0.04]),
            np.array([41.84, 62.05, 10.01]),
            np.array([19.78, 29.16, -7.85]),
            np.array([39.56, 65.98, 33.71]),
            np.array([52.39, 68.33, 47.84]),
            np.array([81.23, 24.12, 87.51]),
            np.array([81.8, 6.78, 95.75]),
            np.array([71.72, -16.23, 76.28]),
            np.array([20.31, 14.45, 16.74]),
            np.array([49.68, -0.19, 0.05]),
            np.array([96.48, -0.88, 0.68]),
            np.array([49.69, -0.18, 0.03]),
            np.array([6.39, -0.04, -0.33]),
            np.array([96.54, -0.9, 0.67]),
            np.array([49.72, -0.18, 0.05]),
            np.array([6.49, -0.03, -0.41]),
            np.array([96.51, -0.9, 0.69]),
            np.array([49.7, -0.19, 0.07]),
            np.array([6.47, 0, -0.38]),
            np.array([96.46, -0.89, 0.71]),
        ],
    )
)

_DATA_COLORCHECKERSG_BEFORE_NOV2014 = np.reshape(
    np.transpose(
        np.reshape(
            np.array(
                list(
                    zip(
                        SAMPLE_LABELS_COLORCHECKER_SG,
                        DATA_COLORCHECKERSG_BEFORE_NOV2014_CIE_LAB.values(),
                    )
                ),
                dtype=object,
            ),
            (14, 10, 2),
        ),
        [1, 0, 2],
    ),
    (-1, 2),
)

DATA_COLORCHECKERSG_BEFORE_NOV2014: dict = dict(
    zip(
        _DATA_COLORCHECKERSG_BEFORE_NOV2014[..., 0],
        XYZ_to_xyY(
            Lab_to_XYZ(
                list(_DATA_COLORCHECKERSG_BEFORE_NOV2014[..., 1]),
                CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["ICC D50"],
            )
        ),
    )
)

del _DATA_COLORCHECKERSG_BEFORE_NOV2014

CCS_ILLUMINANT_COLORCHECKERSG_BEFORE_NOV2014: NDArrayFloat = CCS_ILLUMINANTS[
    "CIE 1931 2 Degree Standard Observer"
]["ICC D50"]
"""*ColorCheckerSG - Before November 2014* illuminant."""

CCS_COLORCHECKERSG_BEFORE_NOV2014: ColourChecker = ColourChecker(
    "ColorCheckerSG - Before November 2014",
    DATA_COLORCHECKERSG_BEFORE_NOV2014,
    CCS_ILLUMINANT_COLORCHECKERSG_BEFORE_NOV2014,
    10,
    14,
)
"""
Reference *ColorChecker SG* data from *X-Rite (2016)*.
"""

DATA_COLORCHECKERSG_AFTER_NOV2014_CIE_LAB: dict = dict(
    zip(
        SAMPLE_LABELS_COLORCHECKER_SG,
        [
            np.array([96.71, -0.62, 2.06]),
            np.array([8.05, 0.17, -0.69]),
            np.array([49.76, 0.11, 0.72]),
            np.array([96.72, -0.63, 2.06]),
            np.array([8.17, 0.15, -0.65]),
            np.array([49.68, 0.14, 0.74]),
            np.array([96.60, -0.62, 2.11]),
            np.array([7.99, 0.21, -0.75]),
            np.array([49.67, 0.15, 0.73]),
            np.array([96.51, -0.63, 2.11]),
            np.array([49.70, 0.12, 0.68]),
            np.array([33.02, 52.00, -10.30]),
            np.array([61.40, 27.14, -18.42]),
            np.array([30.54, 50.39, -41.79]),
            np.array([49.56, -13.90, -49.65]),
            np.array([60.62, -29.91, -27.54]),
            np.array([20.13, -24.81, -7.50]),
            np.array([60.32, -40.29, -13.25]),
            np.array([19.62, 1.77, 11.99]),
            np.array([49.68, 0.15, 0.78]),
            np.array([8.13, 0.15, -0.76]),
            np.array([19.65, 20.42, -18.82]),
            np.array([41.70, 18.90, -37.42]),
            np.array([20.25, 0.26, -36.44]),
            np.array([60.13, -17.88, -32.08]),
            np.array([19.75, -17.79, -22.37]),
            np.array([60.43, -5.12, -32.79]),
            np.array([50.46, -47.90, -11.56]),
            np.array([60.53, -40.75, 19.37]),
            np.array([8.09, 0.19, -0.69]),
            np.array([96.79, -0.66, 1.99]),
            np.array([84.00, -1.70, -8.37]),
            np.array([85.48, 15.15, 0.79]),
            np.array([84.56, -19.74, -1.13]),
            np.array([85.26, 13.37, 7.95]),
            np.array([84.38, -11.97, 27.16]),
            np.array([62.35, 29.94, 36.89]),
            np.array([64.17, 21.34, 19.36]),
            np.array([50.48, -53.21, 12.65]),
            np.array([96.57, -0.64, 2.00]),
            np.array([49.79, 0.13, 0.66]),
            np.array([32.77, 19.91, 22.33]),
            np.array([62.28, 37.56, 68.87]),
            np.array([19.92, 25.07, -61.05]),
            np.array([96.78, -0.66, 2.01]),
            np.array([8.07, 0.12, -0.93]),
            np.array([77.37, 20.28, 24.27]),
            np.array([74.01, 29.00, 25.80]),
            np.array([20.33, -23.98, 7.20]),
            np.array([49.72, 0.14, 0.71]),
            np.array([8.09, 0.19, -0.69]),
            np.array([63.88, 20.34, 19.93]),
            np.array([35.28, 12.93, -51.17]),
            np.array([52.75, -44.12, 38.68]),
            np.array([79.65, -0.08, 0.62]),
            np.array([30.32, -0.10, 0.22]),
            np.array([63.46, 13.53, 26.37]),
            np.array([64.44, 14.31, 17.63]),
            np.array([60.05, -44.00, 7.27]),
            np.array([8.08, 0.18, -0.78]),
            np.array([96.70, -0.66, 1.97]),
            np.array([45.84, -3.74, -25.32]),
            np.array([47.60, 53.66, 22.15]),
            np.array([36.88, 65.72, 41.63]),
            np.array([65.22, -0.27, 0.16]),
            np.array([39.55, -0.37, -0.09]),
            np.array([44.49, 16.06, 26.79]),
            np.array([64.97, 15.89, 16.79]),
            np.array([60.77, -30.19, 40.76]),
            np.array([96.71, -0.64, 2.01]),
            np.array([49.74, 0.14, 0.68]),
            np.array([38.29, -17.44, 30.22]),
            np.array([20.76, 31.66, -28.04]),
            np.array([81.43, 2.41, 88.98]),
            np.array([49.71, 0.12, 0.69]),
            np.array([60.04, 0.09, 0.05]),
            np.array([67.60, 14.47, 17.12]),
            np.array([64.75, 17.30, 18.88]),
            np.array([51.26, -50.65, 43.80]),
            np.array([49.76, 0.14, 0.71]),
            np.array([8.10, 0.19, -0.93]),
            np.array([51.36, 9.52, -26.98]),
            np.array([71.62, -24.77, 64.10]),
            np.array([48.75, 57.24, -14.45]),
            np.array([34.85, -0.21, 0.73]),
            np.array([75.36, 0.35, 0.26]),
            np.array([45.14, 26.38, 41.24]),
            np.array([36.20, 16.70, 27.06]),
            np.array([61.65, -54.33, 46.18]),
            np.array([7.97, 0.14, -0.80]),
            np.array([96.69, -0.67, 1.95]),
            np.array([68.71, -35.41, -1.11]),
            np.array([70.39, 19.37, 79.73]),
            np.array([47.42, -30.91, -32.27]),
            np.array([15.43, -0.24, -0.25]),
            np.array([88.85, -0.59, 0.25]),
            np.array([64.00, 25.09, 27.14]),
            np.array([66.65, 22.21, 28.81]),
            np.array([62.05, 16.45, 51.74]),
            np.array([96.71, -0.64, 2.02]),
            np.array([49.72, 0.12, 0.64]),
            np.array([85.68, 10.75, 18.39]),
            np.array([89.35, -16.38, 6.41]),
            np.array([84.59, 5.21, -5.87]),
            np.array([83.63, -12.47, -8.89]),
            np.array([70.60, -0.24, 0.07]),
            np.array([45.14, -0.04, 0.86]),
            np.array([20.33, 0.40, -0.21]),
            np.array([62.33, -14.54, 54.58]),
            np.array([49.74, 0.14, 0.69]),
            np.array([8.08, 0.13, -0.81]),
            np.array([23.03, 33.95, 8.88]),
            np.array([44.35, 67.94, 50.62]),
            np.array([60.91, 36.55, 4.15]),
            np.array([62.20, 37.45, 18.18]),
            np.array([63.33, 51.30, 81.88]),
            np.array([73.74, -11.45, 85.07]),
            np.array([62.35, 1.96, 57.52]),
            np.array([72.77, -29.09, 71.26]),
            np.array([8.13, 0.15, -0.86]),
            np.array([49.71, 0.12, 0.62]),
            np.array([42.52, 63.55, 11.43]),
            np.array([18.09, 32.61, -5.90]),
            np.array([40.66, 65.54, 31.98]),
            np.array([53.13, 68.44, 49.57]),
            np.array([82.08, 23.39, 87.24]),
            np.array([82.50, 5.29, 96.68]),
            np.array([71.90, -17.32, 77.72]),
            np.array([21.95, 13.41, 16.36]),
            np.array([49.74, 0.12, 0.69]),
            np.array([96.79, -0.67, 1.97]),
            np.array([49.78, 0.12, 0.65]),
            np.array([8.23, 0.18, -0.82]),
            np.array([96.73, -0.67, 1.99]),
            np.array([49.80, 0.11, 0.67]),
            np.array([8.18, 0.15, -0.84]),
            np.array([96.73, -0.65, 2.01]),
            np.array([49.75, 0.13, 0.67]),
            np.array([8.11, 0.15, -0.90]),
            np.array([96.55, -0.64, 2.02]),
        ],
    )
)

_DATA_COLORCHECKERSG_AFTER_NOV2014 = np.reshape(
    np.transpose(
        np.reshape(
            np.array(
                list(
                    zip(
                        SAMPLE_LABELS_COLORCHECKER_SG,
                        DATA_COLORCHECKERSG_AFTER_NOV2014_CIE_LAB.values(),
                    )
                ),
                dtype=object,
            ),
            (14, 10, 2),
        ),
        [1, 0, 2],
    ),
    (-1, 2),
)

DATA_COLORCHECKERSG_AFTER_NOV2014: dict = dict(
    zip(
        _DATA_COLORCHECKERSG_AFTER_NOV2014[..., 0],
        XYZ_to_xyY(
            Lab_to_XYZ(
                list(_DATA_COLORCHECKERSG_AFTER_NOV2014[..., 1]),
                CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["ICC D50"],
            )
        ),
    )
)

del _DATA_COLORCHECKERSG_AFTER_NOV2014

CCS_ILLUMINANT_COLORCHECKERSG_AFTER_NOV2014: NDArrayFloat = CCS_ILLUMINANTS[
    "CIE 1931 2 Degree Standard Observer"
]["ICC D50"]
"""*ColorCheckerSG - After November 2014* illuminant."""

CCS_COLORCHECKERSG_AFTER_NOV2014: ColourChecker = ColourChecker(
    "ColorCheckerSG - After November 2014",
    DATA_COLORCHECKERSG_AFTER_NOV2014,
    CCS_ILLUMINANT_COLORCHECKERSG_AFTER_NOV2014,
    10,
    14,
)
"""
Reference *ColorChecker SG* data from *X-Rite (2016)* and matching the
*ColorChecker SG* edition after November 2014.
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
        "ColorCheckerSG - Before November 2014": CCS_COLORCHECKERSG_BEFORE_NOV2014,
        "ColorCheckerSG - After November 2014": CCS_COLORCHECKERSG_AFTER_NOV2014,
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
