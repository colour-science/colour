#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases colour spectrum computations.
"""

import numpy as np

import colour
from colour.utilities.verbose import message_box

message_box('Spectrum Computations')

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

message_box('Sample spectral power distribution shape.')
print(spd.shape)

print('\n')

message_box('Sample spectral power distribution uniformity.')
print(spd.is_uniform())

print('\n')

message_box(('Sample spectral power distribution cloning:\n'
             '\n\t("Original Id", "Clone Id")\n'
             '\nCloning is a convenient way to get a copy of the spectral '
             'power distribution, this an important feature because some '
             'operations happen in place.'))
clone_spd = spd.clone()
print(id(spd), id(clone_spd))

print('\n')

message_box('Sample spectral power distribution arithmetical operations.')
message_box('Regular arithmetical operation: adding a constant.')
clone_spd_alternate = clone_spd + 10
print(clone_spd[380], clone_spd_alternate[380])

print('\n')

message_box('Regular arithmetical operation: adding an array.')
print((clone_spd + np.linspace(0, 1, len(clone_spd))).values)

print('\n')

message_box('Regular arithmetical operation: adding a spectral power '
            'distribution.')
print((clone_spd + clone_spd).values)

print('\n')

message_box('In-place arithmetical operation: adding a constant.')
clone_spd += 10
print(clone_spd[380])

print('\n')

message_box('In-place arithmetical operation: adding an array.')
clone_spd += np.linspace(0, 1, len(clone_spd))
print(clone_spd.values)

print('\n')

message_box('In-place arithmetical operation: adding a spectral power '
            'distribution.')
clone_spd += clone_spd
print(clone_spd.values)

print('\n')

message_box('Sample spectral power distribution interpolation.')
clone_spd.interpolate(colour.SpectralShape(360, 780, 1))
print(clone_spd[666])

print('\n')

message_box('Sample spectral power distribution extrapolation.')
clone_spd.extrapolate(colour.SpectralShape(340, 830))
print(clone_spd[340], clone_spd[360])

print('\n')

message_box('Sample spectral power distribution align.')
clone_spd.align(colour.SpectralShape(400, 700, 5))
print(clone_spd[400], clone_spd[700])

print('\n')

message_box('Constant value filled spectral power distribution.')
print(colour.constant_spd(3.1415)[400])

print('\n')

message_box('Zeros filled spectral power distribution.')
print(colour.zeros_spd()[400])

print('\n')

message_box('Ones filled spectral power distribution.')
print(colour.ones_spd()[400])
