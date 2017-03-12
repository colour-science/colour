#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CIE Standard Illuminant D Series :math:`S_n(\lambda)` Distributions
===================================================================

Defines the *CIE Standard Illuminant D Series* :math:`S_n(\lambda)`
distributions involved in the computation of
*CIE Standard Illuminant D Series* relative spectral power distributions.

See Also
--------
`Illuminants Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/colorimetry/illuminants.ipynb>`_

References
----------
.. [1]  Wyszecki, G., & Stiles, W. S. (2000). CIE Method of Calculating
        D-Illuminants. In Color Science: Concepts and Methods,
        Quantitative Data and Formulae (pp. 145–146). Wiley.
        ISBN:978-0471399186
.. [2]  Lindbloom, B. (2007). Spectral Power Distribution of a
        CIE D-Illuminant. Retrieved April 05, 2014, from
        http://www.brucelindbloom.com/Eqn_DIlluminant.html
"""

from __future__ import division, unicode_literals

from colour.colorimetry import SpectralPowerDistribution
from colour.utilities import CaseInsensitiveMapping

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['D_ILLUMINANTS_S_SPDS_DATA',
           'D_ILLUMINANTS_S_SPDS']

D_ILLUMINANTS_S_SPDS_DATA = {
    'S0': {
        300: 0.04,
        310: 6.00,
        320: 29.60,
        330: 55.30,
        340: 57.30,
        350: 61.80,
        360: 61.50,
        370: 68.80,
        380: 63.40,
        390: 65.80,
        400: 94.80,
        410: 104.80,
        420: 105.90,
        430: 96.80,
        440: 113.90,
        450: 125.60,
        460: 125.50,
        470: 121.30,
        480: 121.30,
        490: 113.50,
        500: 113.10,
        510: 110.80,
        520: 106.50,
        530: 108.80,
        540: 105.30,
        550: 104.40,
        560: 100.00,
        570: 96.00,
        580: 95.10,
        590: 89.10,
        600: 90.50,
        610: 90.30,
        620: 88.40,
        630: 84.00,
        640: 85.10,
        650: 81.90,
        660: 82.60,
        670: 84.90,
        680: 81.30,
        690: 71.90,
        700: 74.30,
        710: 76.40,
        720: 63.30,
        730: 71.70,
        740: 77.00,
        750: 65.20,
        760: 47.70,
        770: 68.60,
        780: 65.00,
        790: 66.00,
        800: 61.00,
        810: 53.30,
        820: 58.90,
        830: 61.90
    },
    'S1': {
        300: 0.02,
        310: 4.50,
        320: 22.40,
        330: 42.00,
        340: 40.60,
        350: 41.60,
        360: 38.00,
        370: 43.40,
        380: 38.50,
        390: 35.00,
        400: 43.40,
        410: 46.30,
        420: 43.90,
        430: 37.10,
        440: 36.70,
        450: 35.90,
        460: 32.60,
        470: 27.90,
        480: 24.30,
        490: 20.10,
        500: 16.20,
        510: 13.20,
        520: 8.60,
        530: 6.10,
        540: 4.20,
        550: 1.90,
        560: 0.00,
        570: -1.60,
        580: -3.50,
        590: -3.50,
        600: -5.80,
        610: -7.20,
        620: -8.60,
        630: -9.50,
        640: -10.90,
        650: -10.70,
        660: -12.00,
        670: -14.00,
        680: -13.60,
        690: -12.00,
        700: -13.30,
        710: -12.90,
        720: -10.60,
        730: -11.60,
        740: -12.20,
        750: -10.20,
        760: -7.80,
        770: -11.20,
        780: -10.40,
        790: -10.60,
        800: -9.70,
        810: -8.30,
        820: -9.30,
        830: -9.80
    },
    'S2': {
        300: 0.0,
        310: 2.0,
        320: 4.0,
        330: 8.5,
        340: 7.8,
        350: 6.7,
        360: 5.3,
        370: 6.1,
        380: 3.0,
        390: 1.2,
        400: -1.1,
        410: -0.5,
        420: -0.7,
        430: -1.2,
        440: -2.6,
        450: -2.9,
        460: -2.8,
        470: -2.6,
        480: -2.6,
        490: -1.8,
        500: -1.5,
        510: -1.3,
        520: -1.2,
        530: -1.0,
        540: -0.5,
        550: -0.3,
        560: 0.0,
        570: 0.2,
        580: 0.5,
        590: 2.1,
        600: 3.2,
        610: 4.1,
        620: 4.7,
        630: 5.1,
        640: 6.7,
        650: 7.3,
        660: 8.6,
        670: 9.8,
        680: 10.2,
        690: 8.3,
        700: 9.6,
        710: 8.5,
        720: 7.0,
        730: 7.6,
        740: 8.0,
        750: 6.7,
        760: 5.2,
        770: 7.4,
        780: 6.8,
        790: 7.0,
        800: 6.4,
        810: 5.5,
        820: 6.1,
        830: 6.5}}

D_ILLUMINANTS_S_SPDS = CaseInsensitiveMapping(
    {'S0': SpectralPowerDistribution(
        'S0', D_ILLUMINANTS_S_SPDS_DATA['S0']),
     'S1': SpectralPowerDistribution(
         'S1', D_ILLUMINANTS_S_SPDS_DATA['S1']),
     'S2': SpectralPowerDistribution(
         'S2', D_ILLUMINANTS_S_SPDS_DATA['S2'])})
"""
*CIE Standard Illuminant D Series* :math:`S_n(\lambda)` spectral power
distributions

D_ILLUMINANTS_S_SPDS : CaseInsensitiveMapping
   **{'S0', 'S1', 'S1'}**
"""
