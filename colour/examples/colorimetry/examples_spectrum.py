# -*- coding: utf-8 -*-
"""
Showcases colour spectrum computations.
"""

import numpy as np

import colour
from colour.utilities import message_box

message_box('Spectrum Computations')

data_sample = {
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
    780: 0.421
}

sd_sample = colour.SpectralDistribution(data_sample, name='Sample')

message_box('Sample spectral distribution shape.')
print(sd_sample.shape)

print('\n')

message_box('Sample spectral distribution uniformity.')
print(sd_sample.is_uniform())

print('\n')

message_box(('Sample spectral distribution cloning:\n'
             '\n\t("Original Id", "Clone Id")\n'
             '\nCloning is a convenient way to get a copy of the spectral '
             'distribution, this an important feature because some '
             'operations happen in place.'))
sd_clone = sd_sample.copy()
print(id(sd_sample), id(sd_clone))

print('\n')

message_box('Sample spectral distribution arithmetical operations.')
message_box('Regular arithmetical operation: adding a constant.')
sd_clone_alternate = sd_clone + 10
print(sd_clone[380], sd_clone_alternate[380])

print('\n')

message_box('Regular arithmetical operation: adding an array.')
print((sd_clone + np.linspace(0, 1, len(sd_clone.wavelengths))).values)

print('\n')

message_box('Regular arithmetical operation: adding a spectral '
            'distribution.')
print((sd_clone + sd_clone).values)

print('\n')

message_box('In-place arithmetical operation: adding a constant.')
sd_clone += 10
print(sd_clone[380])

print('\n')

message_box('In-place arithmetical operation: adding an array.')
sd_clone += np.linspace(0, 1, len(sd_clone.wavelengths))
print(sd_clone.values)

print('\n')

message_box('In-place arithmetical operation: adding a spectral '
            'distribution.')
sd_clone += sd_clone
print(sd_clone.values)

print('\n')

message_box('Sample spectral distribution interpolation.')
sd_clone.interpolate(colour.SpectralShape(360, 780, 1))
print(sd_clone[666])

print('\n')

message_box('Sample spectral distribution extrapolation.')
sd_clone.extrapolate(colour.SpectralShape(340, 830, 1))
print(sd_clone[340], sd_clone[360])

print('\n')

message_box('Sample spectral distribution align.')
sd_clone.align(colour.SpectralShape(400, 700, 5))
print(sd_clone[400], sd_clone[700])

print('\n')

message_box('Constant value filled spectral distribution.')
print(colour.sd_constant(3.1415)[400])

print('\n')

message_box('Zeros filled spectral distribution.')
print(colour.sd_zeros()[400])

print('\n')

message_box('Ones filled spectral distribution.')
print(colour.sd_ones()[400])
