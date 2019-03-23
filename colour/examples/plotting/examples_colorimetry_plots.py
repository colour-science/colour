# -*- coding: utf-8 -*-
"""
Showcases colorimetry plotting examples.
"""

from pprint import pprint

import colour
from colour.plotting import (
    ASTM_G_173_ETR, plot_blackbody_colours, plot_blackbody_spectral_radiance,
    colour_style, plot_multi_cmfs, plot_multi_illuminant_sds,
    plot_multi_lightness_functions, plot_multi_sds, plot_single_cmfs,
    plot_single_illuminant_sd, plot_single_lightness_function, plot_single_sd,
    plot_visible_spectrum)
from colour.utilities import message_box

message_box('Colorimetry Plots')

colour_style()

message_box('Plotting a single illuminant spectral ' 'distribution.')
plot_single_illuminant_sd('FL1')

print('\n')

message_box(('Plotting multiple illuminants spectral ' 'distributions.'))
pprint(sorted(colour.ILLUMINANTS_SDS.keys()))
plot_multi_illuminant_sds(
    ['A', 'B', 'C', 'D50', 'D55', 'D60', 'D65', 'D75', 'FL1'])

print('\n')

message_box(('Plotting "CIE Standard Illuminant "A", "B", and "C" with their '
             'normalised colours.'))
plot_multi_illuminant_sds(
    ['A', 'B', 'C'], use_sds_colours=True, normalise_sds_colours=True)

print('\n')

message_box(('Plotting "CIE Standard Illuminant D Series" "S" spectral '
             'distributions.'))
plot_multi_sds(
    [
        value for key, value in sorted(
            colour.colorimetry.D_ILLUMINANTS_S_SDS.items())
    ],
    title='CIE Standard Illuminant D Series - S Distributions')

print('\n')

sample_sd_data = {
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

# http://speclib.jpl.nasa.gov/speclibdata/
# jhu.becknic.manmade.roofing.metal.solid.0525uuua.spectrum.txt  # noqa
galvanized_steel_metal_sd_data = {
    360: 2.24,
    362: 2.25,
    364: 2.26,
    366: 2.28,
    368: 2.29,
    370: 2.30,
    372: 2.31,
    374: 2.32,
    376: 2.32,
    378: 2.33,
    380: 2.33,
    382: 2.34,
    384: 2.34,
    386: 2.34,
    388: 2.34,
    390: 2.34,
    392: 2.34,
    394: 2.34,
    396: 2.34,
    398: 2.34,
    400: 2.35,
    402: 2.35,
    404: 2.35,
    406: 2.35,
    408: 2.35,
    410: 2.35,
    412: 2.35,
    414: 2.36,
    416: 2.36,
    418: 2.37,
    420: 2.37,
    422: 2.37,
    424: 2.38,
    426: 2.38,
    428: 2.39,
    430: 2.39,
    432: 2.40,
    434: 2.40,
    436: 2.41,
    438: 2.41,
    440: 2.41,
    442: 2.41,
    444: 2.41,
    446: 2.41,
    448: 2.41,
    450: 2.41,
    452: 2.41,
    454: 2.41,
    456: 2.41,
    458: 2.41,
    460: 2.40,
    462: 2.40,
    464: 2.40,
    466: 2.39,
    468: 2.39,
    470: 2.39,
    472: 2.38,
    474: 2.38,
    476: 2.38,
    478: 2.38,
    480: 2.38,
    482: 2.38,
    484: 2.39,
    486: 2.40,
    488: 2.40,
    490: 2.42,
    492: 2.43,
    494: 2.44,
    496: 2.46,
    498: 2.47,
    500: 2.49,
    502: 2.51,
    504: 2.52,
    506: 2.54,
    508: 2.56,
    510: 2.58,
    512: 2.60,
    514: 2.62,
    516: 2.65,
    518: 2.67,
    520: 2.70,
    522: 2.74,
    524: 2.77,
    526: 2.81,
    528: 2.86,
    530: 2.90,
    532: 2.95,
    534: 3.01,
    536: 3.07,
    538: 3.14,
    540: 3.21,
    542: 3.29,
    544: 3.37,
    546: 3.47,
    548: 3.57,
    550: 3.67,
    552: 3.79,
    554: 3.91,
    556: 4.04,
    558: 4.18,
    560: 4.32,
    562: 4.48,
    564: 4.64,
    566: 4.80,
    568: 4.97,
    570: 5.14,
    572: 5.32,
    574: 5.49,
    576: 5.67,
    578: 5.85,
    580: 6.02,
    582: 6.19,
    584: 6.36,
    586: 6.53,
    588: 6.68,
    590: 6.83,
    592: 6.97,
    594: 7.11,
    596: 7.24,
    598: 7.35,
    600: 7.45,
    602: 7.55,
    604: 7.64,
    606: 7.72,
    608: 7.80,
    610: 7.86,
    612: 7.92,
    614: 7.97,
    616: 8.01,
    618: 8.05,
    620: 8.08,
    622: 8.11,
    624: 8.13,
    626: 8.15,
    628: 8.16,
    630: 8.17,
    632: 8.18,
    634: 8.19,
    636: 8.19,
    638: 8.20,
    640: 8.20,
    642: 8.20,
    644: 8.20,
    646: 8.20,
    648: 8.20,
    650: 8.20,
    652: 8.20,
    654: 8.19,
    656: 8.20,
    658: 8.20,
    660: 8.20,
    662: 8.20,
    664: 8.21,
    666: 8.22,
    668: 8.23,
    670: 8.24,
    672: 8.25,
    674: 8.26,
    676: 8.28,
    678: 8.29,
    680: 8.31,
    682: 8.33,
    684: 8.35,
    686: 8.38,
    688: 8.40,
    690: 8.43,
    692: 8.46,
    694: 8.49,
    696: 8.52,
    698: 8.56,
    700: 8.59,
    702: 8.63,
    704: 8.67,
    706: 8.70,
    708: 8.74,
    710: 8.78,
    712: 8.83,
    714: 8.87,
    716: 8.91,
    718: 8.95,
    720: 8.99,
    722: 9.03,
    724: 9.08,
    726: 9.12,
    728: 9.16,
    730: 9.20,
    732: 9.24,
    734: 9.28,
    736: 9.32,
    738: 9.36,
    740: 9.39,
    742: 9.43,
    744: 9.46,
    746: 9.49,
    748: 9.52,
    750: 9.54,
    752: 9.57,
    754: 9.59,
    756: 9.61,
    758: 9.62,
    760: 9.64,
    762: 9.65,
    764: 9.65,
    766: 9.66,
    768: 9.66,
    770: 9.65,
    772: 9.65,
    774: 9.63,
    776: 9.62,
    778: 9.61,
    780: 9.59,
    782: 9.56,
    784: 9.54,
    786: 9.50,
    788: 9.47,
    790: 9.43,
    792: 9.39,
    794: 9.35,
    796: 9.31,
    798: 9.26,
    800: 9.21,
    820: 8.59
}

# http://speclib.jpl.nasa.gov/speclibdata/
# jhu.becknic.manmade.construction.marble.solid.0722uuu.spectrum.txt
white_marble_sd_data = {
    360: 40.93,
    362: 41.58,
    364: 42.25,
    366: 42.97,
    368: 43.71,
    370: 44.49,
    372: 45.30,
    374: 46.15,
    376: 47.03,
    378: 48.07,
    380: 49.15,
    382: 50.07,
    384: 50.98,
    386: 51.89,
    388: 52.77,
    390: 53.62,
    392: 54.44,
    394: 55.21,
    396: 55.96,
    398: 56.67,
    400: 57.34,
    402: 57.97,
    404: 58.58,
    406: 59.16,
    408: 59.71,
    410: 60.22,
    412: 60.72,
    414: 61.20,
    416: 61.68,
    418: 62.15,
    420: 62.62,
    422: 63.09,
    424: 63.55,
    426: 64.03,
    428: 64.51,
    430: 64.99,
    432: 65.47,
    434: 65.95,
    436: 66.42,
    438: 66.88,
    440: 67.31,
    442: 67.72,
    444: 68.11,
    446: 68.46,
    448: 68.78,
    450: 69.07,
    452: 69.35,
    454: 69.59,
    456: 69.81,
    458: 70.00,
    460: 70.19,
    462: 70.37,
    464: 70.53,
    466: 70.71,
    468: 70.87,
    470: 71.04,
    472: 71.20,
    474: 71.38,
    476: 71.56,
    478: 71.74,
    480: 71.94,
    482: 72.16,
    484: 72.37,
    486: 72.57,
    488: 72.80,
    490: 73.04,
    492: 73.28,
    494: 73.52,
    496: 73.77,
    498: 74.01,
    500: 74.26,
    502: 74.50,
    504: 74.75,
    506: 75.00,
    508: 75.26,
    510: 75.50,
    512: 75.72,
    514: 75.96,
    516: 76.18,
    518: 76.40,
    520: 76.62,
    522: 76.83,
    524: 77.04,
    526: 77.24,
    528: 77.43,
    530: 77.61,
    532: 77.80,
    534: 77.97,
    536: 78.14,
    538: 78.31,
    540: 78.47,
    542: 78.63,
    544: 78.78,
    546: 78.92,
    548: 79.07,
    550: 79.21,
    552: 79.33,
    554: 79.45,
    556: 79.56,
    558: 79.69,
    560: 79.81,
    562: 79.90,
    564: 79.99,
    566: 80.08,
    568: 80.16,
    570: 80.24,
    572: 80.33,
    574: 80.41,
    576: 80.48,
    578: 80.54,
    580: 80.61,
    582: 80.67,
    584: 80.73,
    586: 80.78,
    588: 80.83,
    590: 80.88,
    592: 80.92,
    594: 80.95,
    596: 81.00,
    598: 81.06,
    600: 81.09,
    602: 81.13,
    604: 81.17,
    606: 81.20,
    608: 81.24,
    610: 81.27,
    612: 81.3,
    614: 81.34,
    616: 81.37,
    618: 81.40,
    620: 81.43,
    622: 81.47,
    624: 81.51,
    626: 81.54,
    628: 81.58,
    630: 81.61,
    632: 81.66,
    634: 81.72,
    636: 81.75,
    638: 81.79,
    640: 81.83,
    642: 81.86,
    644: 81.88,
    646: 81.93,
    648: 81.98,
    650: 82.03,
    652: 82.07,
    654: 82.11,
    656: 82.16,
    658: 82.21,
    660: 82.25,
    662: 82.28,
    664: 82.34,
    666: 82.40,
    668: 82.44,
    670: 82.48,
    672: 82.53,
    674: 82.60,
    676: 82.64,
    678: 82.67,
    680: 82.70,
    682: 82.72,
    684: 82.74,
    686: 82.78,
    688: 82.80,
    690: 82.85,
    692: 82.89,
    694: 82.93,
    696: 82.96,
    698: 83.02,
    700: 83.05,
    702: 83.10,
    704: 83.14,
    706: 83.15,
    708: 83.20,
    710: 83.25,
    712: 83.29,
    714: 83.31,
    716: 83.35,
    718: 83.38,
    720: 83.40,
    722: 83.44,
    724: 83.48,
    726: 83.51,
    728: 83.53,
    730: 83.56,
    732: 83.60,
    734: 83.63,
    736: 83.66,
    738: 83.68,
    740: 83.70,
    742: 83.73,
    744: 83.77,
    746: 83.80,
    748: 83.81,
    750: 83.83,
    752: 83.87,
    754: 83.88,
    756: 83.92,
    758: 83.96,
    760: 83.96,
    762: 83.97,
    764: 83.99,
    766: 84.02,
    768: 84.04,
    770: 84.07,
    772: 84.10,
    774: 84.11,
    776: 84.16,
    778: 84.18,
    780: 84.19,
    782: 84.19,
    784: 84.21,
    786: 84.22,
    788: 84.25,
    790: 84.29,
    792: 84.29,
    794: 84.31,
    796: 84.31,
    798: 84.34,
    800: 84.34,
    820: 84.47
}

message_box('Plotting various single spectral distributions.')
plot_single_sd(colour.SpectralDistribution(sample_sd_data, name='Custom'))
plot_single_sd(
    colour.SpectralDistribution(
        galvanized_steel_metal_sd_data, name='Galvanized Steel Metal'))

print('\n')

message_box('Plotting multiple spectral distributions.')
plot_multi_sds((colour.SpectralDistribution(
    galvanized_steel_metal_sd_data, name='Galvanized Steel Metal'),
                colour.SpectralDistribution(
                    white_marble_sd_data, name='White Marble')))

print('\n')

message_box('Plotting spectral bandpass dependence correction.')
street_light_sd_data = {
    380: 8.9770000e-003,
    382: 5.8380000e-003,
    384: 8.3290000e-003,
    386: 8.6940000e-003,
    388: 1.0450000e-002,
    390: 1.0940000e-002,
    392: 8.4260000e-003,
    394: 1.1720000e-002,
    396: 1.2260000e-002,
    398: 7.4550000e-003,
    400: 9.8730000e-003,
    402: 1.2970000e-002,
    404: 1.4000000e-002,
    406: 1.1000000e-002,
    408: 1.1330000e-002,
    410: 1.2100000e-002,
    412: 1.4070000e-002,
    414: 1.5150000e-002,
    416: 1.4800000e-002,
    418: 1.6800000e-002,
    420: 1.6850000e-002,
    422: 1.7070000e-002,
    424: 1.7220000e-002,
    426: 1.8250000e-002,
    428: 1.9930000e-002,
    430: 2.2640000e-002,
    432: 2.4630000e-002,
    434: 2.5250000e-002,
    436: 2.6690000e-002,
    438: 2.8320000e-002,
    440: 2.5500000e-002,
    442: 1.8450000e-002,
    444: 1.6470000e-002,
    446: 2.2470000e-002,
    448: 3.6250000e-002,
    450: 4.3970000e-002,
    452: 2.7090000e-002,
    454: 2.2400000e-002,
    456: 1.4380000e-002,
    458: 1.3210000e-002,
    460: 1.8250000e-002,
    462: 2.6440000e-002,
    464: 4.5690000e-002,
    466: 9.2240000e-002,
    468: 6.0570000e-002,
    470: 2.6740000e-002,
    472: 2.2430000e-002,
    474: 3.4190000e-002,
    476: 2.8160000e-002,
    478: 1.9570000e-002,
    480: 1.8430000e-002,
    482: 1.9800000e-002,
    484: 2.1840000e-002,
    486: 2.2840000e-002,
    488: 2.5760000e-002,
    490: 2.9800000e-002,
    492: 3.6620000e-002,
    494: 6.2500000e-002,
    496: 1.7130000e-001,
    498: 2.3920000e-001,
    500: 1.0620000e-001,
    502: 4.1250000e-002,
    504: 3.3340000e-002,
    506: 3.0820000e-002,
    508: 3.0750000e-002,
    510: 3.2500000e-002,
    512: 4.5570000e-002,
    514: 7.5490000e-002,
    516: 6.6560000e-002,
    518: 3.9350000e-002,
    520: 3.3880000e-002,
    522: 3.4610000e-002,
    524: 3.6270000e-002,
    526: 3.6580000e-002,
    528: 3.7990000e-002,
    530: 4.0010000e-002,
    532: 4.0540000e-002,
    534: 4.2380000e-002,
    536: 4.4190000e-002,
    538: 4.6760000e-002,
    540: 5.1490000e-002,
    542: 5.7320000e-002,
    544: 7.0770000e-002,
    546: 1.0230000e-001,
    548: 1.6330000e-001,
    550: 2.3550000e-001,
    552: 2.7540000e-001,
    554: 2.9590000e-001,
    556: 3.2950000e-001,
    558: 3.7630000e-001,
    560: 4.1420000e-001,
    562: 4.4850000e-001,
    564: 5.3330000e-001,
    566: 7.3490000e-001,
    568: 8.6530000e-001,
    570: 7.8120000e-001,
    572: 6.8580000e-001,
    574: 6.6740000e-001,
    576: 6.9300000e-001,
    578: 6.9540000e-001,
    580: 6.3260000e-001,
    582: 4.6240000e-001,
    584: 2.3550000e-001,
    586: 8.4450000e-002,
    588: 3.5550000e-002,
    590: 4.0580000e-002,
    592: 1.3370000e-001,
    594: 3.4150000e-001,
    596: 5.8250000e-001,
    598: 7.2080000e-001,
    600: 7.6530000e-001,
    602: 7.5290000e-001,
    604: 7.1080000e-001,
    606: 6.5840000e-001,
    608: 6.0140000e-001,
    610: 5.5270000e-001,
    612: 5.4450000e-001,
    614: 5.9260000e-001,
    616: 5.4520000e-001,
    618: 4.4690000e-001,
    620: 3.9040000e-001,
    622: 3.5880000e-001,
    624: 3.3400000e-001,
    626: 3.1480000e-001,
    628: 2.9800000e-001,
    630: 2.8090000e-001,
    632: 2.6370000e-001,
    634: 2.5010000e-001,
    636: 2.3610000e-001,
    638: 2.2550000e-001,
    640: 2.1680000e-001,
    642: 2.0720000e-001,
    644: 1.9920000e-001,
    646: 1.9070000e-001,
    648: 1.8520000e-001,
    650: 1.7970000e-001,
    652: 1.7410000e-001,
    654: 1.7070000e-001,
    656: 1.6500000e-001,
    658: 1.6080000e-001,
    660: 1.5660000e-001,
    662: 1.5330000e-001,
    664: 1.4860000e-001,
    666: 1.4540000e-001,
    668: 1.4260000e-001,
    670: 1.3840000e-001,
    672: 1.3500000e-001,
    674: 1.3180000e-001,
    676: 1.2730000e-001,
    678: 1.2390000e-001,
    680: 1.2210000e-001,
    682: 1.1840000e-001,
    684: 1.1530000e-001,
    686: 1.1210000e-001,
    688: 1.1060000e-001,
    690: 1.0950000e-001,
    692: 1.0840000e-001,
    694: 1.0740000e-001,
    696: 1.0630000e-001,
    698: 1.0550000e-001,
    700: 1.0380000e-001,
    702: 1.0250000e-001,
    704: 1.0380000e-001,
    706: 1.0250000e-001,
    708: 1.0130000e-001,
    710: 1.0020000e-001,
    712: 9.8310000e-002,
    714: 9.8630000e-002,
    716: 9.8140000e-002,
    718: 9.6680000e-002,
    720: 9.4430000e-002,
    722: 9.4050000e-002,
    724: 9.2510000e-002,
    726: 9.1880000e-002,
    728: 9.1120000e-002,
    730: 8.9860000e-002,
    732: 8.9460000e-002,
    734: 8.8610000e-002,
    736: 8.9640000e-002,
    738: 8.9910000e-002,
    740: 8.7700000e-002,
    742: 8.7540000e-002,
    744: 8.5880000e-002,
    746: 8.1340000e-002,
    748: 8.8200000e-002,
    750: 8.9410000e-002,
    752: 8.9360000e-002,
    754: 8.4970000e-002,
    756: 8.9030000e-002,
    758: 8.7810000e-002,
    760: 8.5330000e-002,
    762: 8.5880000e-002,
    764: 1.1310000e-001,
    766: 1.6180000e-001,
    768: 1.6770000e-001,
    770: 1.5340000e-001,
    772: 1.1740000e-001,
    774: 9.2280000e-002,
    776: 9.0480000e-002,
    778: 9.0020000e-002,
    780: 8.8190000e-002
}

street_light_sd = colour.SpectralDistribution(
    street_light_sd_data, name='Street Light')

bandpass_corrected_street_light_sd = street_light_sd.copy()
bandpass_corrected_street_light_sd.name = 'Street Light (Bandpass Corrected)'
bandpass_corrected_street_light_sd = colour.bandpass_correction(
    bandpass_corrected_street_light_sd, method='Stearns 1988')

plot_multi_sds(
    (street_light_sd, bandpass_corrected_street_light_sd),
    title='Stearns Bandpass Correction')

print('\n')

message_box('Plotting a single "cone fundamentals" colour matching functions.')
plot_single_cmfs(
    'Stockman & Sharpe 2 Degree Cone Fundamentals',
    y_label='Sensitivity',
    bounding_box=(390, 870, 0, 1.1))

print('\n')

message_box('Plotting multiple "cone fundamentals" colour matching functions.')
plot_multi_cmfs(
    [
        'Stockman & Sharpe 2 Degree Cone Fundamentals',
        'Stockman & Sharpe 10 Degree Cone Fundamentals'
    ],
    y_label='Sensitivity',
    bounding_box=(390, 870, 0, 1.1))

print('\n')

message_box('Plotting various single colour matching functions.')
pprint(sorted(colour.CMFS.keys()))
plot_single_cmfs('CIE 1931 2 Degree Standard Observer')
plot_single_cmfs('CIE 1964 10 Degree Standard Observer')
plot_single_cmfs(
    'Stiles & Burch 1955 2 Degree RGB CMFs',
    bounding_box=(390, 830, -0.5, 3.5))
plot_single_cmfs(
    'Stiles & Burch 1959 10 Degree RGB CMFs',
    bounding_box=(390, 830, -0.5, 3.5))

print('\n')

message_box('Comparing various colour matching functions.')
plot_multi_cmfs([
    'CIE 1931 2 Degree Standard Observer',
    'CIE 1964 10 Degree Standard Observer'
])
plot_multi_cmfs([
    'CIE 2012 10 Degree Standard Observer',
    'CIE 1964 10 Degree Standard Observer'
])
plot_multi_cmfs([
    'Wright & Guild 1931 2 Degree RGB CMFs',
    'Stiles & Burch 1955 2 Degree RGB CMFs'
])

print('\n')

message_box('Plotting visible colours under given standard observer.')
plot_visible_spectrum('CIE 1931 2 Degree Standard Observer')
plot_visible_spectrum('CIE 2012 2 Degree Standard Observer')

print('\n')

message_box('Plotting photopic luminous efficiency functions.')
plot_multi_sds(
    colour.PHOTOPIC_LEFS.values(),
    title='Luminous Efficiency Functions',
    y_label='Luminous Efficiency')

print('\n')

message_box('Comparing photopic and scotopic luminous efficiency functions.')
plot_multi_sds(
    (colour.PHOTOPIC_LEFS['CIE 2008 2 Degree Physiologically Relevant LEF'],
     colour.SCOTOPIC_LEFS['CIE 1951 Scotopic Standard Observer']),
    title='Photopic & Scotopic Luminous Efficiency Functions',
    y_label='Luminous Efficiency')

print('\n')

message_box(('Plotting a mesopic luminous efficiency function with given '
             'photopic luminance value:\n'
             '\n\t0.2'))
sd_mesopic_luminous_efficiency_function = (
    colour.sd_mesopic_luminous_efficiency_function(0.2))

plot_multi_sds(
    (sd_mesopic_luminous_efficiency_function,
     colour.PHOTOPIC_LEFS['CIE 1924 Photopic Standard Observer'],
     colour.SCOTOPIC_LEFS['CIE 1951 Scotopic Standard Observer']),
    y_label='Luminous Efficiency')

print('\n')

message_box('Plotting a single "Lightness" function.')
plot_single_lightness_function('CIE 1976')

print('\n')

message_box('Plotting multiple "Lightness" functions.')
plot_multi_lightness_functions(['CIE 1976', 'Glasser 1958'])

print('\n')

message_box('Plotting various blackbody spectral radiance.')
plot_blackbody_spectral_radiance(
    temperature=3500, blackbody='VY Canis Majoris')
plot_blackbody_spectral_radiance(temperature=5778, blackbody='The Sun')
plot_blackbody_spectral_radiance(temperature=12130, blackbody='Rigel')

print('\n')

message_box('Comparing theoretical and measured "Sun" spectral distributions.')
# Arbitrary ASTM_G_173_ETR scaling factor calculated with
# :func:`colour.sd_to_XYZ` definition.
ASTM_G_173_sd = ASTM_G_173_ETR.copy() * 1.37905559e+13

ASTM_G_173_sd.interpolate(
    colour.SpectralShape(interval=5), interpolator=colour.LinearInterpolator)

blackbody_sd = colour.sd_blackbody(5778, ASTM_G_173_sd.shape)
blackbody_sd.name = 'The Sun - 5778K'

plot_multi_sds((ASTM_G_173_sd, blackbody_sd), y_label='W / (sr m$^2$) / m')

print('\n')

message_box('Plotting various "blackbody" spectral distributions.')
blackbody_sds = [
    colour.sd_blackbody(i, colour.SpectralShape(0, 10000, 10))
    for i in range(1000, 15000, 1000)
]

plot_multi_sds(
    blackbody_sds,
    y_label='W / (sr m$^2$) / m',
    use_sds_colours=True,
    normalise_sds_colours=True,
    bounding_box=(0, 1250, 0, 2.5e15))

print('\n')

message_box('Plotting "blackbody" colours.')
plot_blackbody_colours()
