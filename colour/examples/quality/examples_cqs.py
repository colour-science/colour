# -*- coding: utf-8 -*-
"""
Showcases *Colour Quality Scale* (CQS) computations.
"""

from pprint import pprint

import colour
from colour.utilities import message_box

message_box('"Colour Quality Scale (CQS)" Computations')

message_box('Computing "F2" illuminant "Colour Quality Scale (CQS)".')
print(colour.colour_quality_scale(colour.ILLUMINANTS_SDS['FL2']))

print('\n')

message_box(('Computing "H38HT-100" mercury lamp "Colour Quality Scale (CQS)" '
             'with detailed output data.'))
pprint(
    colour.colour_quality_scale(
        colour.LIGHT_SOURCES_SDS['H38HT-100 (Mercury)'], additional_data=True))

print('\n')

message_box('Computing "SDW-T 100W/LV Super HPS" lamp '
            '"Colour Quality Scale (CQS)".')
print(
    colour.colour_quality_scale(
        colour.LIGHT_SOURCES_SDS['SDW-T 100W/LV (Super HPS)']))

print('\n')

message_box('Computing sample light "Colour Quality Scale (CQS)".')
SAMPLE_SD_DATA = {
    380: 0.00588346,
    385: 0.00315377,
    390: 0.00242868,
    395: 0.00508709,
    400: 0.00323282,
    405: 0.00348764,
    410: 0.00369248,
    415: 0.00520924,
    420: 0.00747913,
    425: 0.01309795,
    430: 0.02397167,
    435: 0.04330206,
    440: 0.08272117,
    445: 0.14123187,
    450: 0.23400416,
    455: 0.34205230,
    460: 0.43912850,
    465: 0.44869766,
    470: 0.37549764,
    475: 0.27829316,
    480: 0.19453198,
    485: 0.14168353,
    490: 0.11233585,
    495: 0.10301871,
    500: 0.11438976,
    505: 0.14553810,
    510: 0.18971677,
    515: 0.25189581,
    520: 0.31072378,
    525: 0.35998103,
    530: 0.38208860,
    535: 0.37610602,
    540: 0.34653432,
    545: 0.30803672,
    550: 0.26015946,
    555: 0.21622002,
    560: 0.17448497,
    565: 0.13561398,
    570: 0.10873008,
    575: 0.08599236,
    580: 0.06863164,
    585: 0.05875286,
    590: 0.05276579,
    595: 0.05548599,
    600: 0.07291154,
    605: 0.15319944,
    610: 0.38753740,
    615: 0.81754322,
    620: 1.00000000,
    625: 0.64794360,
    630: 0.21375526,
    635: 0.03710525,
    640: 0.01761510,
    645: 0.01465312,
    650: 0.01384908,
    655: 0.01465716,
    660: 0.01347059,
    665: 0.01424768,
    670: 0.01215791,
    675: 0.01209338,
    680: 0.01155313,
    685: 0.01061995,
    690: 0.01014779,
    695: 0.00864212,
    700: 0.00951386,
    705: 0.00786982,
    710: 0.00841476,
    715: 0.00741868,
    720: 0.00637711,
    725: 0.00556483,
    730: 0.00590016,
    735: 0.00416819,
    740: 0.00422222,
    745: 0.00345776,
    750: 0.00336879,
    755: 0.00298999,
    760: 0.00367047,
    765: 0.00340568,
    770: 0.00261153,
    775: 0.00258850,
    780: 0.00293663
}

print(
    colour.colour_quality_scale(
        colour.SpectralDistribution(SAMPLE_SD_DATA, name='Sample')))
