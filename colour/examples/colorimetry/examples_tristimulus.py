#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases *CIE XYZ* tristimulus values computations.
"""

import colour
from colour.utilities.verbose import message_box

message_box('"CIE XYZ" Tristimulus Values Computations')

sample_spd_data = {
    380: 0.048,
    385: 0.051,
    390: 0.055,
    395: 0.060,
    400: 0.065,
    405: 0.068,
    410: 0.068,
    415: 0.067,
    420: 0.064,
    425: 0.062,
    430: 0.059,
    435: 0.057,
    440: 0.055,
    445: 0.054,
    450: 0.053,
    455: 0.053,
    460: 0.052,
    465: 0.052,
    470: 0.052,
    475: 0.053,
    480: 0.054,
    485: 0.055,
    490: 0.057,
    495: 0.059,
    500: 0.061,
    505: 0.062,
    510: 0.065,
    515: 0.067,
    520: 0.070,
    525: 0.072,
    530: 0.074,
    535: 0.075,
    540: 0.076,
    545: 0.078,
    550: 0.079,
    555: 0.082,
    560: 0.087,
    565: 0.092,
    570: 0.100,
    575: 0.107,
    580: 0.115,
    585: 0.122,
    590: 0.129,
    595: 0.134,
    600: 0.138,
    605: 0.142,
    610: 0.146,
    615: 0.150,
    620: 0.154,
    625: 0.158,
    630: 0.163,
    635: 0.167,
    640: 0.173,
    645: 0.180,
    650: 0.188,
    655: 0.196,
    660: 0.204,
    665: 0.213,
    670: 0.222,
    675: 0.231,
    680: 0.242,
    685: 0.251,
    690: 0.261,
    695: 0.271,
    700: 0.282,
    705: 0.294,
    710: 0.305,
    715: 0.318,
    720: 0.334,
    725: 0.354,
    730: 0.372,
    735: 0.392,
    740: 0.409,
    745: 0.420,
    750: 0.436,
    755: 0.450,
    760: 0.462,
    765: 0.465,
    770: 0.448,
    775: 0.432,
    780: 0.421}

spd = colour.SpectralPowerDistribution('Sample', sample_spd_data)

cmfs = colour.CMFS['CIE 1931 2 Degree Standard Observer']
illuminant = colour.ILLUMINANTS_RELATIVE_SPDS['A']

message_box(('Computing *CIE XYZ* tristimulus values for sample spectral '
             'power distribution and "CIE Standard Illuminant A".'))
print(colour.spectral_to_XYZ(spd, cmfs, illuminant))

print('\n')

message_box(('Computing "CIE Standard Illuminant A" chromaticity coordinates '
             'from its relative spectral power distribution.'))
print(colour.XYZ_to_xy(colour.spectral_to_XYZ(illuminant, cmfs)))

print('\n')

message_box(('Computing *CIE XYZ* tristimulus values for a single given '
             'wavelength in nm.'))
print(colour.wavelength_to_XYZ(
    546.1,
    colour.CMFS['CIE 1931 2 Degree Standard Observer']))
