from .all import MUNSELL_COLOURS_ALL
from .experimental import MUNSELL_COLOURS_1929
from .real import MUNSELL_COLOURS_REAL
from colour.utilities import CaseInsensitiveMapping

__all__ = [
    "MUNSELL_COLOURS_ALL",
]
__all__ += [
    "MUNSELL_COLOURS_1929",
]
__all__ += [
    "MUNSELL_COLOURS_REAL",
]

MUNSELL_COLOURS = CaseInsensitiveMapping(
    {
        "Munsell Colours All": MUNSELL_COLOURS_ALL,
        "Munsell Colours 1929": MUNSELL_COLOURS_1929,
        "Munsell Colours Real": MUNSELL_COLOURS_REAL,
    }
)
MUNSELL_COLOURS.__doc__ = """
Defines the *Munsell Renotation System* datasets.

-   ``Munsell Colours All``: *all* published *Munsell* colours, including the
    extrapolated colors.
-   ``Munsell Colours 1929``: the colours appearing in the 1929
    *Munsell Book of Color*. These data has been used in the scaling
    experiments leading to the 1943 renotation.
-   ``Munsell Colours Real``: *real*, within MacAdam limits *Munsell* colours
    only. They are the colours listed in the original 1943 renotation article
    *(Newhall, Nickerson, & Judd, 1943)*.

Notes
-----
-   The Munsell Renotation data commonly available within the *all.dat*,
    *experimental.dat* and *real.dat* files features *CIE xyY* colourspace
    values that are scaled by a :math:`1 / 0.975 \\simeq 1.02568` factor. If
    you are performing conversions using *Munsell* *Colorlab* specification,
    e.g. *2.5R 9/2*, according to *ASTM D1535-08e1* method, you should not
    scale the output :math:`Y` Luminance. However, if you use directly the
    *CIE xyY* colourspace values from the Munsell Renotation data data, you
    should scale the :math:`Y` Luminance before conversions by a :math:`0.975`
    factor.

    *ASTM D1535-08e1* states that::

        The coefficients of this equation are obtained from the 1943 equation
        by multiplying each coefficient by 0.975, the reflectance factor of
        magnesium oxide with respect to the perfect reflecting diffuser, and
        rounding to ve digits of precision.

-   Chromaticities assume *CIE Illuminant C*, approximately 6700K, as neutral
    origin for both the hue and chroma loci.

References
----------
-   :cite:`MunsellColorSciencec` : Munsell Color Science. (n.d.). Munsell
    Colours Data. Retrieved August 20, 2014, from
    http://www.cis.rit.edu/research/mcsl2/online/munsell.php

Aliases:

-   'all': 'Munsell Colours All'
-   '1929': 'Munsell Colours 1929'
-   'real': 'Munsell Colours Real'
"""
MUNSELL_COLOURS["all"] = MUNSELL_COLOURS["Munsell Colours All"]
MUNSELL_COLOURS["1929"] = MUNSELL_COLOURS["Munsell Colours 1929"]
MUNSELL_COLOURS["real"] = MUNSELL_COLOURS["Munsell Colours Real"]

__all__ += [
    "MUNSELL_COLOURS",
]
