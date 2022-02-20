"""
Spectral Distributions of Lenses
================================

Defines the spectral distributions of lenses.

Each lens data is in the form of :class:`dict` class instance of
:class:`colour.SpectralDistribution` classes as follows::

    {'name': SpectralDistribution, ..., 'name': SpectralDistribution}

The following *lenses* are available:

-   ISO Standard Lens

References
----------
-   :cite:`InternationalOrganizationforStandardization2002` : International
    Organization for Standardization. (2002). INTERNATIONAL STANDARD ISO
    7589-2002 - Photography - Illuminants for sensitometry - Specifications for
    daylight, incandescent tungsten and printer.
"""

from __future__ import annotations

from functools import partial

from colour.colorimetry import SpectralDistribution
from colour.hints import Dict
from colour.utilities import LazyCaseInsensitiveMapping

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "DATA_LENSES_ISO",
    "SDS_LENSES_ISO",
    "SDS_LENSES",
]

DATA_LENSES_ISO: Dict = {
    "ISO Standard Lens": {
        350: 0.00,
        360: 0.07,
        370: 0.23,
        380: 0.42,
        390: 0.60,
        400: 0.74,
        410: 0.83,
        420: 0.88,
        430: 0.91,
        440: 0.94,
        450: 0.95,
        460: 0.97,
        470: 0.98,
        480: 0.98,
        490: 0.99,
        500: 0.99,
        510: 1.00,
        520: 1.00,
        530: 1.00,
        540: 1.00,
        550: 1.00,
        560: 1.00,
        570: 1.00,
        580: 1.00,
        590: 0.99,
        600: 0.99,
        610: 0.99,
        620: 0.98,
        630: 0.98,
        640: 0.97,
        650: 0.97,
        660: 0.96,
        670: 0.95,
        680: 0.94,
        690: 0.94,
    }
}

SDS_LENSES_ISO: LazyCaseInsensitiveMapping = LazyCaseInsensitiveMapping(
    {
        "ISO Standard Lens": partial(
            SpectralDistribution,
            DATA_LENSES_ISO["ISO Standard Lens"],
            name="ISO Standard Lens",
        ),
    }
)
SDS_LENSES_ISO.__doc__ = """
Spectral distributions of *ISO* lenses.

References
----------
:cite:`InternationalOrganizationforStandardization2002`
"""

SDS_LENSES: LazyCaseInsensitiveMapping = LazyCaseInsensitiveMapping(
    SDS_LENSES_ISO
)
SDS_LENSES.__doc__ = """
Spectral distributions of lenses.

References
----------
:cite:`InternationalOrganizationforStandardization2002`
"""
