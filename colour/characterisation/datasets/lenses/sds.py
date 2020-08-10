# -*- coding: utf-8 -*-
"""
Spectral Distributions of Lenses
================================

Defines spectral distributions of lenses.

Each lens data is in the form of :class:`OrderedDict` class instance of
:class:`colour.SpectralDistribution` classes as follows::

    {'name': SpectralDistribution, ..., 'name': SpectralDistribution}

The following *lenses* are available:

-   ISO Standard Lens

References
----------
-   :cite:`ISO2002` : ISO. (2002). INTERNATIONAL STANDARD 7589-2002 -
    Photography - Illuminants for sensitometry - Specifications for daylight,
    incandescent tungsten and printer.
"""

from __future__ import division, unicode_literals

from colour.colorimetry import SpectralDistribution
from colour.utilities import CaseInsensitiveMapping

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['DATA_LENSES_ISO', 'SDS_LENSES_ISO', 'SDS_LENSES']

DATA_LENSES_ISO = {
    'ISO Standard Lens': {
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

SDS_LENSES_ISO = CaseInsensitiveMapping({
    'ISO Standard Lens':
        SpectralDistribution(
            DATA_LENSES_ISO['ISO Standard Lens'], name='ISO Standard Lens'),
})
SDS_LENSES_ISO.__doc__ = """
Spectral distributions of *ISO* lenses.

References
----------
:cite:`ISO2002`

SDS_LENSES_ISO : CaseInsensitiveMapping
"""

SDS_LENSES = CaseInsensitiveMapping(SDS_LENSES_ISO)
SDS_LENSES.__doc__ = """
Spectral distributions of lenses.

References
----------
:cite:`ISO2002`

SDS_LENSES : CaseInsensitiveMapping
"""
