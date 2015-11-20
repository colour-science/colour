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
        {'Natural': (0.38158985649350702, 0.35923120103270789),
         'Philips TL-84': (0.37841538915063888, 0.37929354134172549),
         'SA': (0.44757741609747725, 0.40744460332183946),
         'SC': (0.31006249258289620, 0.31615894048704551),
         'T8 Luxline Plus White': (0.41049446037884174, 0.38893620278233082),
         'T8 Polylux 3000': (0.43170881206177303, 0.41388207821561945),
         'T8 Polylux 4000': (0.37922140508761693, 0.38447266959681337),
         'Thorn Kolor-rite': (0.38192166897848884, 0.37431379604708248)}))
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
        {'Natural': (0.38487108708598944, 0.35386938106691807),
         'Philips TL-84': (0.38359204507923228, 0.37392281581600095),
         'SA': (0.45117690796326249, 0.40593619633937900),
         'SC': (0.31038876517325836, 0.31905094833988284),
         'T8 Luxline Plus White': (0.41694703237204389, 0.38099150867187254),
         'T8 Polylux 3000': (0.43903899178116679, 0.40455442804863512),
         'T8 Polylux 4000': (0.38511520756441064, 0.37780100881568229),
         'Thorn Kolor-rite': (0.38553398880839984, 0.37084059461212321)}))
"""
Light sources chromaticity coordinates from *RIT* *PointerData.xls* spreadsheet
for *CIE 1964 10 Degree Standard Observer*. [1]_

LIGHT_SOURCES_RIT_CIE_1964_10_DEGREE_STANDARD_OBSERVER : CaseInsensitiveMapping
    **{'Natural', 'Philips TL-84', 'T8 Luxline Plus White', 'SA', 'SC',
    'T8 Polylux 3000', 'T8 Polylux 4000', 'Thorn Kolor-rite'}**
"""

LIGHT_SOURCES_NIST_TRADITIONAL_CIE_1931_2_DEGREE_STANDARD_OBSERVER = (
    CaseInsensitiveMapping(
        {'Cool White FL': (
            0.36925870807864725, 0.37255439092346182),
         'Daylight FL': (
             0.31266552605571413, 0.33199168981463328),
         'HPS': (
             0.52167961205678837, 0.41797345689504539),
         'Incandescent': (
             0.45073259760741963, 0.40804960344754043),
         'LPS': (
             0.57515131136516484, 0.42423223492490475),
         'Mercury': (
             0.39202154627365182, 0.38378245438705483),
         'Metal Halide': (
             0.37255427739944014, 0.38562261918652763),
         'Neodimium Incandescent': (
             0.44740318836426740, 0.39501502309332737),
         'Super HPS': (
             0.47006165927184551, 0.40611658424874064),
         'Triphosphor FL': (
             0.41316326825727473, 0.39642205375867939)}))
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
        {'Cool White FL': (
            0.37671510408872622, 0.36457690213857558),
         'Daylight FL': (
             0.31739593703352309, 0.33078095274723718),
         'HPS': (
             0.53176454178730148, 0.40875276707350805),
         'Incandescent': (
             0.45436566167222903, 0.40657376464652656),
         'LPS': (
             0.58996004588789108, 0.41003995411210886),
         'Mercury': (
             0.40126648675178034, 0.36473265428767382),
         'Metal Halide': (
             0.37878639417698795, 0.37749733947763814),
         'Neodimium Incandescent': (
             0.44751682295667389, 0.39673430140144011),
         'Super HPS': (
             0.47385956714613520, 0.40138182530919675),
         'Triphosphor FL': (
             0.41859196393173587, 0.38894771333219230)}))
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
            0.41705768695552681, 0.39626245799686116),
         '3-LED-2 (473/545/616)': (
             0.41706047556601339, 0.39626812052342902),
         '3-LED-2 Yellow': (
             0.43656307918405129, 0.44364961929868374),
         '3-LED-3 (465/546/614)': (
             0.38046050218478095, 0.37677200148246243),
         '3-LED-4 (455/547/623)': (
             0.41706794370458505, 0.39627628009361210),
         '4-LED No Yellow': (
             0.41706058956634484, 0.39626815414010280),
         '4-LED Yellow': (
             0.41706963799377295, 0.39627676610090434),
         '4-LED-1 (461/526/576/624)': (
             0.41706761544198467, 0.39627505678189306),
         '4-LED-2 (447/512/573/627)': (
             0.41707157075929724, 0.39627874545196168),
         'Luxeon WW 2880': (
             0.45908852792091304, 0.43291648060790272),
         'PHOS-1': (
             0.43644419153116243, 0.40461759619272675),
         'PHOS-2': (
             0.45270446285822602, 0.43758454407895764),
         'PHOS-3': (
             0.43689987306504829, 0.40403737565486431),
         'PHOS-4': (
             0.43693602634706474, 0.40411356199238802),
         'Phosphor LED YAG': (
             0.30776185306080994, 0.32526902527554236)}))
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
            0.42509998892670109, 0.38945134991130748),
         '3-LED-2 (473/545/616)': (
             0.42222211877421723, 0.40129849559422581),
         '3-LED-2 Yellow': (
             0.44622221613912494, 0.44164646427608739),
         '3-LED-3 (465/546/614)': (
             0.38747046580194294, 0.37640471601567871),
         '3-LED-4 (455/547/623)': (
             0.42286546410736542, 0.38877224017213308),
         '4-LED No Yellow': (
             0.41980753295868123, 0.39946529494033128),
         '4-LED Yellow': (
             0.42272060175132126, 0.39028466347542795),
         '4-LED-1 (461/526/576/624)': (
             0.42389978332307165, 0.39417088622702445),
         '4-LED-2 (447/512/573/627)': (
             0.42157104205857177, 0.39408974193594359),
         'Luxeon WW 2880': (
             0.46663929962326245, 0.43081741721805084),
         'PHOS-1': (
             0.44012002557571367, 0.40313581943314591),
         'PHOS-2': (
             0.46148739888659074, 0.43615029469088301),
         'PHOS-3': (
             0.44089265535724609, 0.40866226448493076),
         'PHOS-4': (
             0.44176044400976400, 0.40726747835572896),
         'Phosphor LED YAG': (
             0.31280783561056424, 0.33418093987336706)}))
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
            0.45073259760741963, 0.40804960344754043),
         'C100S54 (HPS)': (
             0.52923376397609678, 0.41137274159083304),
         'C100S54C (HPS)': (
             0.50238041437483938, 0.41587729990547478),
         'F32T8/TL830 (Triphosphor)': (
             0.44325174374980375, 0.40952517507904407),
         'F32T8/TL835 (Triphosphor)': (
             0.40715116396946999, 0.39317422842369792),
         'F32T8/TL841 (Triphosphor)': (
             0.38537743893676257, 0.39037213791517428),
         'F32T8/TL850 (Triphosphor)': (
             0.34376937302886135, 0.35844840316417925),
         'F32T8/TL865 /PLUS (Triphosphor)': (
             0.31636935243196435, 0.34532192591376282),
         'F34/CW/RS/EW (Cool White FL)': (
             0.37725318688336951, 0.39309197961767905),
         'F34T12/LW/RS /EW': (
             0.37886468516122646, 0.39496262028620821),
         'F34T12WW/RS /EW (Warm White FL)': (
             0.43846866903462800, 0.40863804458652164),
         'F40/C50 (Broadband FL)': (
             0.34583865254299739, 0.36172878048941465),
         'F40/C75 (Broadband FL)': (
             0.29997001994648365, 0.31659051460116211),
         'F40/CWX (Broadband FL)': (
             0.37503976277492962, 0.36054877322178092),
         'F40/DX (Broadband FL)': (
             0.31192452342224031, 0.34280755026772286),
         'F40/DXTP (Delux FL)': (
             0.31306918458210292, 0.34223215032277970),
         'F40/N (Natural FL)': (
             0.37688154063294127, 0.35415836752602542),
         'H38HT-100 (Mercury)': (
             0.31120059019364160, 0.38294424585701864),
         'H38JA-100/DX (Mercury DX)': (
             0.38979163036035858, 0.37339468893176730),
         'MHC100/U/MP /3K': (
             0.42858523665462062, 0.38817414851241644),
         'MHC100/U/MP /4K': (
             0.37314904074932925, 0.37137398108931774),
         'SDW-T 100W/LV (Super HPS)': (
             0.47233915793867187, 0.40710633088031556)}))
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
            0.45436566167222903, 0.40657376464652656),
         'C100S54 (HPS)': (
             0.53855465963806315, 0.40257588651356163),
         'C100S54C (HPS)': (
             0.50966305997089167, 0.40906450820919271),
         'F32T8/TL830 (Triphosphor)': (
             0.44879524270399968, 0.40357466971840461),
         'F32T8/TL835 (Triphosphor)': (
             0.41208255534422666, 0.38800110477749572),
         'F32T8/TL841 (Triphosphor)': (
             0.39090863700792133, 0.38529059107116664),
         'F32T8/TL850 (Triphosphor)': (
             0.34788244200594859, 0.35584576394739581),
         'F32T8/TL865 /PLUS (Triphosphor)': (
             0.32069821051031028, 0.34387146649818662),
         'F34/CW/RS/EW (Cool White FL)': (
             0.38651490749625111, 0.38284342173411157),
         'F34T12/LW/RS /EW': (
             0.38962893428416084, 0.38207476566744725),
         'F34T12WW/RS /EW (Warm White FL)': (
             0.44839541868796040, 0.39566670122099346),
         'F40/C50 (Broadband FL)': (
             0.34988087545007562, 0.36066141429674070),
         'F40/C75 (Broadband FL)': (
             0.30198860919197257, 0.31847921291797271),
         'F40/CWX (Broadband FL)': (
             0.37850237300051637, 0.35637199807284170),
         'F40/DX (Broadband FL)': (
             0.31678308857261384, 0.34174939068075244),
         'F40/DXTP (Delux FL)': (
             0.31877480629298705, 0.33979896855444874),
         'F40/N (Natural FL)': (
             0.37883322344059039, 0.35072451295316365),
         'H38HT-100 (Mercury)': (
             0.32626062708248443, 0.36000109589520496),
         'H38JA-100/DX (Mercury DX)': (
             0.39705859751753292, 0.35653243180697375),
         'MHC100/U/MP /3K': (
             0.43142306761570764, 0.38064233096970834),
         'MHC100/U/MP /4K': (
             0.37570719315261863, 0.36615662147422956),
         'SDW-T 100W/LV (Super HPS)': (
             0.47646190819266115, 0.40228801240357442)}))
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
            0.31525568387423564, 0.33287848220034716)}))
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
            0.31708720400118184, 0.33622317634402810)}))
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
