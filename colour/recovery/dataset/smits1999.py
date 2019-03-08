# -*- coding: utf-8 -*-
"""
Smits (1999) - Reflectance Recovery Dataset
===========================================

Defines the dataset for reflectance recovery using *Smits (1999)* method.

References
----------
-   :cite:`Smits1999a` : Smits, B. (1999). An RGB-to-Spectrum Conversion for
    Reflectances. Journal of Graphics Tools, 4(4), 11-22.
    doi:10.1080/10867651.1999.10487511
"""

from __future__ import division, unicode_literals

from colour.algebra import LinearInterpolator
from colour.colorimetry.spectrum import SpectralDistribution
from colour.utilities import CaseInsensitiveMapping

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['SMITS_1999_SDS_DATA', 'SMITS_1999_SDS']

SMITS_1999_SDS_DATA = {
    'white': {
        380.0000: 1.0000,
        417.7778: 1.0000,
        455.5556: 0.9999,
        493.3333: 0.9993,
        531.1111: 0.9992,
        568.8889: 0.9998,
        606.6667: 1.0000,
        644.4444: 1.0000,
        682.2222: 1.0000,
        720.0000: 1.0000
    },
    'cyan': {
        380.0000: 0.9710,
        417.7778: 0.9426,
        455.5556: 1.0007,
        493.3333: 1.0007,
        531.1111: 1.0007,
        568.8889: 1.0007,
        606.6667: 0.1564,
        644.4444: 0.0000,
        682.2222: 0.0000,
        720.0000: 0.0000
    },
    'magenta': {
        380.0000: 1.0000,
        417.7778: 1.0000,
        455.5556: 0.9685,
        493.3333: 0.2229,
        531.1111: 0.0000,
        568.8889: 0.0458,
        606.6667: 0.8369,
        644.4444: 1.0000,
        682.2222: 1.0000,
        720.0000: 0.9959
    },
    'yellow': {
        380.0000: 0.0001,
        417.7778: 0.0000,
        455.5556: 0.1088,
        493.3333: 0.6651,
        531.1111: 1.0000,
        568.8889: 1.0000,
        606.6667: 0.9996,
        644.4444: 0.9586,
        682.2222: 0.9685,
        720.0000: 0.9840
    },
    'red': {
        380.0000: 0.1012,
        417.7778: 0.0515,
        455.5556: 0.0000,
        493.3333: 0.0000,
        531.1111: 0.0000,
        568.8889: 0.0000,
        606.6667: 0.8325,
        644.4444: 1.0149,
        682.2222: 1.0149,
        720.0000: 1.0149
    },
    'green': {
        380.0000: 0.0000,
        417.7778: 0.0000,
        455.5556: 0.0273,
        493.3333: 0.7937,
        531.1111: 1.0000,
        568.8889: 0.9418,
        606.6667: 0.1719,
        644.4444: 0.0000,
        682.2222: 0.0000,
        720.0000: 0.0025
    },
    'blue': {
        380.0000: 1.0000,
        417.7778: 1.0000,
        455.5556: 0.8916,
        493.3333: 0.3323,
        531.1111: 0.0000,
        568.8889: 0.0000,
        606.6667: 0.0003,
        644.4444: 0.0369,
        682.2222: 0.0483,
        720.0000: 0.0496
    }
}

SMITS_1999_SDS = CaseInsensitiveMapping({
    'white':
        SpectralDistribution(
            SMITS_1999_SDS_DATA['white'],
            name='white'),
    'cyan':
        SpectralDistribution(
            SMITS_1999_SDS_DATA['cyan'],
            name='cyan'),
    'magenta':
        SpectralDistribution(
            SMITS_1999_SDS_DATA['magenta'],
            name='magenta'),
    'yellow':
        SpectralDistribution(
            SMITS_1999_SDS_DATA['yellow'],
            name='yellow'),
    'red':
        SpectralDistribution(
            SMITS_1999_SDS_DATA['red'],
            name='red'),
    'green':
        SpectralDistribution(
            SMITS_1999_SDS_DATA['green'],
            name='green'),
    'blue':
        SpectralDistribution(
            SMITS_1999_SDS_DATA['blue'],
            name='blue')
})  # yapf: disable
SMITS_1999_SDS.__doc__ = """
*Smits (1999)* spectral distributions.

References
----------
:cite:`Smits1999a`

SMITS_1999_SDS : CaseInsensitiveMapping
"""

# Using linear interpolation to preserve the shape of the basis spectral
# distributions once combined and interpolated.
for _sd in SMITS_1999_SDS.values():
    _sd.interpolator = LinearInterpolator
