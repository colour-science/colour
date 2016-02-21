#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Light Sources Chromaticity Coordinates
======================================

Defines various light sources chromaticity coordinates.

The following light sources are available:

-   *RIT* *PointerData.xls* spreadsheet light sources: Natural,
    Philips TL-84, T8 Luxline Plus White, SA, SC, T8 Polylux 3000,
    T8 Polylux 4000, Thorn Kolor-rite
-   *NIST* *NIST CQS simulation 7.4.xls* spreadsheet traditional light sources:
    Cool White FL, Daylight FL, HPS, Incandescent, LPS, Mercury,
    Metal Halide, Neodimium Incandescent, Super HPS, Triphosphor FL
-   *NIST* *NIST CQS simulation 7.4.xls* spreadsheet LED light sources:
    3-LED-1 (457/540/605), 3-LED-2 (473/545/616), 3-LED-2 Yellow,
    3-LED-3 (465/546/614), 3-LED-4 (455/547/623), 4-LED No Yellow,
    4-LED Yellow, 4-LED-1 (461/526/576/624), 4-LED-2 (447/512/573/627),
    Luxeon WW 2880, PHOS-1, PHOS-2, PHOS-3, PHOS-4,
    Phosphor LED YAG
-   *NIST* *NIST CQS simulation 7.4.xls* spreadsheet Philips light sources:
    60 A/W (Soft White), C100S54 (HPS), C100S54C (HPS),
    F32T8/TL830 (Triphosphor), F32T8/TL835 (Triphosphor),
    F32T8/TL841 (Triphosphor), F32T8/TL850 (Triphosphor),
    F32T8/TL865 /PLUS (Triphosphor), F34/CW/RS/EW (Cool White FL),
    F34T12/LW/RS /EW, F34T12WW/RS /EW (Warm White FL),
    F40/C50 (Broadband FL), F40/C75 (Broadband FL),
    F40/CWX (Broadband FL), F40/DX (Broadband FL), F40/DXTP (Delux FL),
    F40/N (Natural FL), H38HT-100 (Mercury), H38JA-100/DX (Mercury DX),
    MHC100/U/MP /3K, MHC100/U/MP /4K, SDW-T 100W/LV (Super HPS)
-   Projectors and Xenon Arc Lamps:
    Kinoton 75P

See Also
--------
`Light Sources IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/\
blob/master/notebooks/colorimetry/light_sources.ipynb>`_
"""

from __future__ import division, unicode_literals

from colour.utilities import CaseInsensitiveMapping

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'LIGHT_SOURCES_RIT_CIE_1931_2_DEGREE_STANDARD_OBSERVER',
    'LIGHT_SOURCES_RIT_CIE_1964_10_DEGREE_STANDARD_OBSERVER',
    'LIGHT_SOURCES_NIST_TRADITIONAL_CIE_1931_2_DEGREE_STANDARD_OBSERVER',
    'LIGHT_SOURCES_NIST_TRADITIONAL_CIE_1964_10_DEGREE_STANDARD_OBSERVER',
    'LIGHT_SOURCES_NIST_LED_CIE_1931_2_DEGREE_STANDARD_OBSERVER',
    'LIGHT_SOURCES_NIST_LED_CIE_1964_10_DEGREE_STANDARD_OBSERVER',
    'LIGHT_SOURCES_NIST_PHILIPS_CIE_1931_2_DEGREE_STANDARD_OBSERVER',
    'LIGHT_SOURCES_NIST_PHILIPS_CIE_1964_10_DEGREE_STANDARD_OBSERVER',
    'LIGHT_SOURCES_PROJECTORS_CIE_1931_2_DEGREE_STANDARD_OBSERVER',
    'LIGHT_SOURCES_PROJECTORS_CIE_1964_10_DEGREE_STANDARD_OBSERVER',
    'LIGHT_SOURCES']

LIGHT_SOURCES_RIT_CIE_1931_2_DEGREE_STANDARD_OBSERVER = (
    CaseInsensitiveMapping(
        {'Natural': (0.38158573064778706, 0.35922413827406652),
         'Philips TL-84': (0.37841359997098778, 0.37929025454409004),
         'SA': (0.44757303073415383, 0.40743813715646698),
         'SC': (0.31005673430392766, 0.31614570478920412),
         'T8 Luxline Plus White': (0.41049220408625048, 0.38893252967684039),
         'T8 Polylux 3000': (0.43170608220718482, 0.41387773607264738),
         'T8 Polylux 4000': (0.37921947313979382, 0.38446908557763143),
         'Thorn Kolor-rite': (0.38191912428280589, 0.37430926164125050)}))
"""
Light sources chromaticity coordinates from *RIT* *PointerData.xls* spreadsheet
for *CIE 1931 2 Degree Standard Observer*.

Warning
-------
The chromaticity coordinates have been calculated from *PointerData.xls*
spreadsheet that doesn't mention the data source thus the light source names
cannot be accurately verified.

References
----------
.. [1]  Pointer, M. R. (1980). Pointer’s Gamut Data. Retrieved from
        http://www.cis.rit.edu/research/mcsl2/online/PointerData.xls

LIGHT_SOURCES_RIT_CIE_1931_2_DEGREE_STANDARD_OBSERVER : CaseInsensitiveMapping
    **{'Natural', 'Philips TL-84', 'T8 Luxline Plus White', 'SA', 'SC',
    'T8 Polylux 3000', 'T8 Polylux 4000', 'Thorn Kolor-rite'}**
"""

LIGHT_SOURCES_RIT_CIE_1964_10_DEGREE_STANDARD_OBSERVER = (
    CaseInsensitiveMapping(
        {'Natural': (0.38487099118303530, 0.35386922336654492),
         'Philips TL-84': (0.38359200289294981, 0.37392274181576163),
         'SA': (0.45117680359407014, 0.40593604678159123),
         'SC': (0.31038863741564865, 0.31905065122098569),
         'T8 Luxline Plus White': (0.41694697883120341, 0.38099142646275569),
         'T8 Polylux 3000': (0.43903892628866975, 0.40455433012471542),
         'T8 Polylux 4000': (0.38511516187287526, 0.37780092839576879),
         'Thorn Kolor-rite': (0.38553392928246683, 0.37084049209094833)}))
"""
Light sources chromaticity coordinates from *RIT* *PointerData.xls* spreadsheet
for *CIE 1964 10 Degree Standard Observer*. [1]_

LIGHT_SOURCES_RIT_CIE_1964_10_DEGREE_STANDARD_OBSERVER : CaseInsensitiveMapping
    **{'Natural', 'Philips TL-84', 'T8 Luxline Plus White', 'SA', 'SC',
    'T8 Polylux 3000', 'T8 Polylux 4000', 'Thorn Kolor-rite'}**
"""

LIGHT_SOURCES_NIST_TRADITIONAL_CIE_1931_2_DEGREE_STANDARD_OBSERVER = (
    CaseInsensitiveMapping(
        {'Cool White FL': (0.36925631897128064, 0.37254987817663071),
         'Daylight FL': (0.31266299396365116, 0.33198568879300933),
         'HPS': (0.52167769606281578, 0.41797117711723858),
         'Incandescent': (0.45073021751967995, 0.40804612894500503),
         'LPS': (0.57515131136516462, 0.42423223492490486),
         'Mercury': (0.39201845763711246, 0.38377707198445288),
         'Metal Halide': (0.37254455897279315, 0.38560392592758758),
         'Neodimium Incandescent': (0.44739869705209967, 0.39500860124826787),
         'Super HPS': (0.47006165927184557, 0.40611658424874059),
         'Triphosphor FL': (0.41316326825727467, 0.39642205375867962)}))
"""
Traditional light sources chromaticity coordinates from *NIST*
*NIST CQS simulation 7.4.xls* spreadsheet.

References
----------
.. [2]  Ohno, Y., & Davis, W. (2008). NIST CQS simulation 7.4. Retrieved from
        http://cie2.nist.gov/TC1-69/NIST CQS simulation 7.4.xls

LIGHT_SOURCES_NIST_TRADITIONAL_CIE_1931_2_DEGREE_STANDARD_OBSERVER :
    CaseInsensitiveMapping
    **{'Cool White FL', 'Daylight FL', 'HPS', 'Incandescent', 'LPS', 'Mercury',
    'Metal Halide', 'Neodimium Incandescent', 'Super HPS', 'Triphosphor FL'}**
"""

LIGHT_SOURCES_NIST_TRADITIONAL_CIE_1964_10_DEGREE_STANDARD_OBSERVER = (
    CaseInsensitiveMapping(
        {'Cool White FL': (0.37671504751845475, 0.36457680211867294),
         'Daylight FL': (0.31739587873896502, 0.33078081913667573),
         'HPS': (0.53176449517751267, 0.40875271528464524),
         'Incandescent': (0.45436560497357181, 0.40657368421677365),
         'LPS': (0.58996004588789108, 0.41003995411210897),
         'Mercury': (0.40126641287375525, 0.36473253822118268),
         'Metal Halide': (0.37878616775122587, 0.37749692850466127),
         'Neodimium Incandescent': (0.44751671715669383, 0.39673415136849710),
         'Super HPS': (0.47385956714613520, 0.40138182530919686),
         'Triphosphor FL': (0.41859196393173603, 0.38894771333219225)}))
"""
Traditional light sources chromaticity coordinates from *NIST*
*NIST CQS simulation 7.4.xls* spreadsheet. [2]_

LIGHT_SOURCES_NIST_TRADITIONAL_CIE_1964_10_DEGREE_STANDARD_OBSERVER :
    CaseInsensitiveMapping
    **{'Cool White FL', 'Daylight FL', 'HPS', 'Incandescent', 'LPS', 'Mercury',
    'Metal Halide', 'Neodimium Incandescent', 'Super HPS', 'Triphosphor FL'}**
"""

LIGHT_SOURCES_NIST_LED_CIE_1931_2_DEGREE_STANDARD_OBSERVER = (
    CaseInsensitiveMapping(
        {'3-LED-1 (457/540/605)': (
            0.41705768694917045, 0.39626245798660154),
         '3-LED-2 (473/545/616)': (
            0.41706047556600634, 0.39626812052341781),
         '3-LED-2 Yellow': (
            0.43656307918404658, 0.44364961929867602),
         '3-LED-3 (465/546/614)': (
            0.38046050218448185, 0.37677200148192225),
         '3-LED-4 (455/547/623)': (
            0.41706794369104505, 0.39627628007175708),
         '4-LED No Yellow': (
            0.41706058930133155, 0.39626815371234958),
         '4-LED Yellow': (
            0.41706963794046309, 0.39627676601485945),
         '4-LED-1 (461/526/576/624)': (
            0.41706761544055604, 0.39627505677958696),
         '4-LED-2 (447/512/573/627)': (
            0.41707157056005439, 0.39627874513037303),
         'Luxeon WW 2880': (
            0.45908852792091304, 0.43291648060790278),
         'PHOS-1': (
            0.43644316780116360, 0.40461603354991749),
         'PHOS-2': (
            0.45270446219857147, 0.43758454305271072),
         'PHOS-3': (
            0.43689987075135950, 0.40403737213446272),
         'PHOS-4': (
            0.43693602390642705, 0.40411355827862949),
         'Phosphor LED YAG': (
            0.30776181731431046, 0.32526893923994132)}))
"""
LED light sources chromaticity coordinates from *NIST*
*NIST CQS simulation 7.4.xls* spreadsheet. [2]_

LIGHT_SOURCES_NIST_LED_CIE_1931_2_DEGREE_STANDARD_OBSERVER :
    **{'3-LED-1 (457/540/605)', '3-LED-2 (473/545/616)', '3-LED-2 Yellow',
    '3-LED-3 (465/546/614)', '3-LED-4 (455/547/623)', '4-LED No Yellow',
    '4-LED Yellow', '4-LED-1 (461/526/576/624)', '4-LED-2 (447/512/573/627)',
    'Luxeon WW 2880', 'PHOS-1', 'PHOS-2', 'PHOS-3', 'PHOS-4',
    'Phosphor LED YAG'}**
"""

LIGHT_SOURCES_NIST_LED_CIE_1964_10_DEGREE_STANDARD_OBSERVER = (
    CaseInsensitiveMapping(
        {'3-LED-1 (457/540/605)': (
            0.42509998892654827, 0.38945134991107527),
         '3-LED-2 (473/545/616)': (
            0.42222211877421717, 0.40129849559422559),
         '3-LED-2 Yellow': (
            0.44622221613912466, 0.44164646427608734),
         '3-LED-3 (465/546/614)': (
            0.38747046580193578, 0.37640471601566622),
         '3-LED-4 (455/547/623)': (
            0.42286546410704112, 0.38877224017163703),
         '4-LED No Yellow': (
            0.41980753295243900, 0.39946529493037725),
         '4-LED Yellow': (
            0.42272060175005299, 0.39028466347347851),
         '4-LED-1 (461/526/576/624)': (
            0.42389978332303729, 0.39417088622697133),
         '4-LED-2 (447/512/573/627)': (
            0.42157104205386658, 0.39408974192860141),
         'Luxeon WW 2880': (
            0.46663929962326273, 0.43081741721805056),
         'PHOS-1': (
            0.44012000128114004, 0.40313578339341560),
         'PHOS-2': (
            0.46148739887055840, 0.43615029466702421),
         'PHOS-3': (
            0.44089265530217153, 0.40866226440229930),
         'PHOS-4': (
            0.44176044395147457, 0.40726747826887888),
         'Phosphor LED YAG': (
            0.31280783477269591, 0.33418093786403530)}))
"""
LED light sources chromaticity coordinates from *NIST*
*NIST CQS simulation 7.4.xls* spreadsheet. [2]_

LIGHT_SOURCES_NIST_LED_CIE_1964_10_DEGREE_STANDARD_OBSERVER :
    CaseInsensitiveMapping
    **{'3-LED-1 (457/540/605)', '3-LED-2 (473/545/616)', '3-LED-2 Yellow',
    '3-LED-3 (465/546/614)', '3-LED-4 (455/547/623)', '4-LED No Yellow',
    '4-LED Yellow', '4-LED-1 (461/526/576/624)', '4-LED-2 (447/512/573/627)',
    'Luxeon WW 2880', 'PHOS-1', 'PHOS-2', 'PHOS-3', 'PHOS-4',
    'Phosphor LED YAG'}**
"""

LIGHT_SOURCES_NIST_PHILIPS_CIE_1931_2_DEGREE_STANDARD_OBSERVER = (
    CaseInsensitiveMapping(
        {'60 A/W (Soft White)': (
            0.45073021751967995, 0.40804612894500503),
         'C100S54 (HPS)': (
            0.52923151540765723, 0.41137016498842699),
         'C100S54C (HPS)': (
            0.50238041437483938, 0.41587729990547495),
         'F32T8/TL830 (Triphosphor)': (
            0.44325076447575318, 0.40952370029692808),
         'F32T8/TL835 (Triphosphor)': (
            0.40715027456993313, 0.39317274348257092),
         'F32T8/TL841 (Triphosphor)': (
            0.38537668668160530, 0.39037076210280575),
         'F32T8/TL850 (Triphosphor)': (
            0.34376891039228652, 0.35844743610410795),
         'F32T8/TL865 /PLUS (Triphosphor)': (
            0.31636887961520122, 0.34532079014301720),
         'F34/CW/RS/EW (Cool White FL)': (
            0.37725093136437809, 0.39308765863605977),
         'F34T12/LW/RS /EW': (
            0.37886364299377639, 0.39496062997982029),
         'F34T12WW/RS /EW (Warm White FL)': (
            0.43846696765678927, 0.40863544156570619),
         'F40/C50 (Broadband FL)': (
            0.34583657497302128, 0.36172445038943024),
         'F40/C75 (Broadband FL)': (
            0.29996666338521955, 0.31658216580482373),
         'F40/CWX (Broadband FL)': (
            0.37503704575421437, 0.36054395212946222),
         'F40/DX (Broadband FL)': (
            0.31192231074653659, 0.34280210341732908),
         'F40/DXTP (Delux FL)': (
            0.31306654382695792, 0.34222571448441214),
         'F40/N (Natural FL)': (
            0.37687869736511453, 0.35415345830287792),
         'H38HT-100 (Mercury)': (
            0.31120059019364149, 0.38294424585701847),
         'H38JA-100/DX (Mercury DX)': (
            0.38979163036035869, 0.37339468893176725),
         'MHC100/U/MP /3K': (
            0.42858176867022246, 0.38816891567832956),
         'MHC100/U/MP /4K': (
            0.37314525348276206, 0.37136699021671654),
         'SDW-T 100W/LV (Super HPS)': (
            0.47233915793867193, 0.40710633088031550)}))
"""
Philips light sources chromaticity coordinates from *NIST*
*NIST CQS simulation 7.4.xls* spreadsheet. [2]_

LIGHT_SOURCES_NIST_PHILIPS_CIE_1931_2_DEGREE_STANDARD_OBSERVER :
    CaseInsensitiveMapping
    **{'60 A/W (Soft White)', 'C100S54 (HPS)', 'C100S54C (HPS)',
    'F32T8/TL830 (Triphosphor)', 'F32T8/TL835 (Triphosphor)',
    'F32T8/TL841 (Triphosphor)', 'F32T8/TL850 (Triphosphor)',
    'F32T8/TL865 /PLUS (Triphosphor)', 'F34/CW/RS/EW (Cool White FL)',
    'F34T12/LW/RS /EW', 'F34T12WW/RS /EW (Warm White FL)',
    'F40/C50 (Broadband FL)', 'F40/C75 (Broadband FL)',
    'F40/CWX (Broadband FL)', 'F40/DX (Broadband FL)', 'F40/DXTP (Delux FL)',
    'F40/N (Natural FL)', 'H38HT-100 (Mercury)', 'H38JA-100/DX (Mercury DX)',
    'MHC100/U/MP /3K', 'MHC100/U/MP /4K', 'SDW-T 100W/LV (Super HPS)'}**
"""

LIGHT_SOURCES_NIST_PHILIPS_CIE_1964_10_DEGREE_STANDARD_OBSERVER = (
    CaseInsensitiveMapping(
        {'60 A/W (Soft White)': (
            0.45436560497357181, 0.40657368421677365),
         'C100S54 (HPS)': (
            0.53855460506301012, 0.40257582797296232),
         'C100S54C (HPS)': (
            0.50966305997089156, 0.40906450820919277),
         'F32T8/TL830 (Triphosphor)': (
            0.44879521930181066, 0.40357463609167810),
         'F32T8/TL835 (Triphosphor)': (
            0.41208253429065178, 0.38800107112759191),
         'F32T8/TL841 (Triphosphor)': (
            0.39090861921952658, 0.38529055999270534),
         'F32T8/TL850 (Triphosphor)': (
            0.34788243125745216, 0.35584574221055110),
         'F32T8/TL865 /PLUS (Triphosphor)': (
            0.32069819959376811, 0.34387144104385381),
         'F34/CW/RS/EW (Cool White FL)': (
            0.38651485354533738, 0.38284332609781435),
         'F34T12/LW/RS /EW': (
            0.38962890915939891, 0.38207472188990355),
         'F34T12WW/RS /EW (Warm White FL)': (
            0.44839537761696036, 0.39566664333529578),
         'F40/C50 (Broadband FL)': (
            0.34988082719688351, 0.36066131649143884),
         'F40/C75 (Broadband FL)': (
            0.30198853387276064, 0.31847902587581839),
         'F40/CWX (Broadband FL)': (
            0.37850230991029593, 0.35637189016893728),
         'F40/DX (Broadband FL)': (
            0.31678303755915344, 0.34174926908507686),
         'F40/DXTP (Delux FL)': (
            0.31877474506579057, 0.33979882560548835),
         'F40/N (Natural FL)': (
            0.37883315774175097, 0.35072440265864635),
         'H38HT-100 (Mercury)': (
            0.32626062708248438, 0.36000109589520463),
         'H38JA-100/DX (Mercury DX)': (
            0.39705859751753286, 0.35653243180697386),
         'MHC100/U/MP /3K': (
            0.43142298659189793, 0.38064221388753933),
         'MHC100/U/MP /4K': (
            0.37570710594811468, 0.36615646577977945),
         'SDW-T 100W/LV (Super HPS)': (
            0.47646190819266110, 0.40228801240357470)}))
"""
Philips light sources chromaticity coordinates from *NIST*
*NIST CQS simulation 7.4.xls* spreadsheet. [2]_

LIGHT_SOURCES_NIST_PHILIPS_CIE_1964_10_DEGREE_STANDARD_OBSERVER :
    CaseInsensitiveMapping
    **{'60 A/W (Soft White)', 'C100S54 (HPS)', 'C100S54C (HPS)',
    'F32T8/TL830 (Triphosphor)', 'F32T8/TL835 (Triphosphor)',
    'F32T8/TL841 (Triphosphor)', 'F32T8/TL850 (Triphosphor)',
    'F32T8/TL865 /PLUS (Triphosphor)', 'F34/CW/RS/EW (Cool White FL)',
    'F34T12/LW/RS /EW', 'F34T12WW/RS /EW (Warm White FL)',
    'F40/C50 (Broadband FL)', 'F40/C75 (Broadband FL)',
    'F40/CWX (Broadband FL)', 'F40/DX (Broadband FL)', 'F40/DXTP (Delux FL)',
    'F40/N (Natural FL)', 'H38HT-100 (Mercury)', 'H38JA-100/DX (Mercury DX)',
    'MHC100/U/MP /3K', 'MHC100/U/MP /4K', 'SDW-T 100W/LV (Super HPS)'}**
"""

LIGHT_SOURCES_PROJECTORS_CIE_1931_2_DEGREE_STANDARD_OBSERVER = (
    CaseInsensitiveMapping(
        {'Kinoton 75P': (
            0.31525377007844130, 0.33287715705757265)}))
"""
Projectors and Xenon Arc Lamps.

References
----------
.. [3]  Houston, J. (2015). Private Discussion with Mansencal, T.

LIGHT_SOURCES_PROJECTORS_CIE_1931_2_DEGREE_STANDARD_OBSERVER :
    CaseInsensitiveMapping
    **{'Kinoton 75P', }**
"""

LIGHT_SOURCES_PROJECTORS_CIE_1964_10_DEGREE_STANDARD_OBSERVER = (
    CaseInsensitiveMapping(
        {'Kinoton 75P': (
            0.31708866083487858, 0.33622914391777436)}))
"""
Projectors and Xenon Arc Lamps. [3_]

LIGHT_SOURCES_PROJECTORS_CIE_1964_10_DEGREE_STANDARD_OBSERVER :
    CaseInsensitiveMapping
    **{'Kinoton 75P', }**
"""

LIGHT_SOURCES = CaseInsensitiveMapping(
    {'CIE 1931 2 Degree Standard Observer': CaseInsensitiveMapping(
        LIGHT_SOURCES_RIT_CIE_1931_2_DEGREE_STANDARD_OBSERVER),
     'CIE 1964 10 Degree Standard Observer': CaseInsensitiveMapping(
         LIGHT_SOURCES_RIT_CIE_1964_10_DEGREE_STANDARD_OBSERVER)})
"""
Aggregated light sources chromaticity coordinates.

LIGHT_SOURCES : CaseInsensitiveMapping
    **{'CIE 1931 2 Degree Standard Observer',
    'CIE 1964 10 Degree Standard Observer'}**

Aliases:

-   'cie_2_1931': 'CIE 1931 2 Degree Standard Observer'
-   'cie_10_1964': 'CIE 1964 10 Degree Standard Observer'
"""
LIGHT_SOURCES['cie_2_1931'] = (
    LIGHT_SOURCES['CIE 1931 2 Degree Standard Observer'])
LIGHT_SOURCES['cie_10_1964'] = (
    LIGHT_SOURCES['CIE 1964 10 Degree Standard Observer'])

LIGHT_SOURCES['CIE 1931 2 Degree Standard Observer'].update(
    LIGHT_SOURCES_NIST_TRADITIONAL_CIE_1931_2_DEGREE_STANDARD_OBSERVER)
LIGHT_SOURCES['CIE 1964 10 Degree Standard Observer'].update(
    LIGHT_SOURCES_NIST_TRADITIONAL_CIE_1964_10_DEGREE_STANDARD_OBSERVER)

LIGHT_SOURCES['CIE 1931 2 Degree Standard Observer'].update(
    LIGHT_SOURCES_NIST_LED_CIE_1931_2_DEGREE_STANDARD_OBSERVER)
LIGHT_SOURCES['CIE 1964 10 Degree Standard Observer'].update(
    LIGHT_SOURCES_NIST_LED_CIE_1964_10_DEGREE_STANDARD_OBSERVER)

LIGHT_SOURCES['CIE 1931 2 Degree Standard Observer'].update(
    LIGHT_SOURCES_NIST_PHILIPS_CIE_1931_2_DEGREE_STANDARD_OBSERVER)
LIGHT_SOURCES['CIE 1964 10 Degree Standard Observer'].update(
    LIGHT_SOURCES_NIST_PHILIPS_CIE_1964_10_DEGREE_STANDARD_OBSERVER)

LIGHT_SOURCES['CIE 1931 2 Degree Standard Observer'].update(
    LIGHT_SOURCES_PROJECTORS_CIE_1931_2_DEGREE_STANDARD_OBSERVER)
LIGHT_SOURCES['CIE 1964 10 Degree Standard Observer'].update(
    LIGHT_SOURCES_PROJECTORS_CIE_1964_10_DEGREE_STANDARD_OBSERVER)
