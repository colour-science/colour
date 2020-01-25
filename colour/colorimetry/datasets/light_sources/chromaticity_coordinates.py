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
`Light Sources Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/colorimetry/light_sources.ipynb>`_

References
----------
-   :cite:`Houston2015a` : Houston, J. (2015). Private Discussion with
    Mansencal, T.
-   :cite:`Ohno2008a` : Ohno, Y., & Davis, W. (2008). NIST CQS simulation 7.4.
    Retrieved from https://drive.google.com/file/d/\
1PsuU6QjUJjCX6tQyCud6ul2Tbs8rYWW9/view?usp=sharing
-   :cite:`Pointer1980a` : Pointer, M. R. (1980). Pointer's Gamut Data.
    Retrieved from http://www.cis.rit.edu/research/mcsl2/online/PointerData.xls
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.utilities import CaseInsensitiveMapping

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
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
    'LIGHT_SOURCES'
]

LIGHT_SOURCES_RIT_CIE_1931_2_DEGREE_STANDARD_OBSERVER = (
    CaseInsensitiveMapping({
        'Natural':
            np.array([0.381585730647787, 0.359224138274067]),
        'Philips TL-84':
            np.array([0.378413599970988, 0.379290254544090]),
        'SA':
            np.array([0.447573030734154, 0.407438137156467]),
        'SC':
            np.array([0.310056734303928, 0.316145704789204]),
        'T8 Luxline Plus White':
            np.array([0.410492204086250, 0.388932529676840]),
        'T8 Polylux 3000':
            np.array([0.431706082207185, 0.413877736072647]),
        'T8 Polylux 4000':
            np.array([0.379219473139794, 0.384469085577631]),
        'Thorn Kolor-rite':
            np.array([0.381919124282806, 0.374309261641251])
    }))
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
:cite:`Pointer1980a`

LIGHT_SOURCES_RIT_CIE_1931_2_DEGREE_STANDARD_OBSERVER : CaseInsensitiveMapping
    **{'Natural', 'Philips TL-84', 'T8 Luxline Plus White', 'SA', 'SC',
    'T8 Polylux 3000', 'T8 Polylux 4000', 'Thorn Kolor-rite'}**
"""

LIGHT_SOURCES_RIT_CIE_1964_10_DEGREE_STANDARD_OBSERVER = (
    CaseInsensitiveMapping({
        'Natural':
            np.array([0.384870991183035, 0.353869223366545]),
        'Philips TL-84':
            np.array([0.383592002892950, 0.373922741815762]),
        'SA':
            np.array([0.451176803594070, 0.405936046781591]),
        'SC':
            np.array([0.310388637415649, 0.319050651220986]),
        'T8 Luxline Plus White':
            np.array([0.416946978831203, 0.380991426462756]),
        'T8 Polylux 3000':
            np.array([0.439038926288670, 0.404554330124715]),
        'T8 Polylux 4000':
            np.array([0.385115161872875, 0.377800928395769]),
        'Thorn Kolor-rite':
            np.array([0.385533929282467, 0.370840492090948])
    }))
"""
Light sources chromaticity coordinates from *RIT* *PointerData.xls* spreadsheet
for *CIE 1964 10 Degree Standard Observer*. [1]_

LIGHT_SOURCES_RIT_CIE_1964_10_DEGREE_STANDARD_OBSERVER : CaseInsensitiveMapping
    **{'Natural', 'Philips TL-84', 'T8 Luxline Plus White', 'SA', 'SC',
    'T8 Polylux 3000', 'T8 Polylux 4000', 'Thorn Kolor-rite'}**
"""

LIGHT_SOURCES_NIST_TRADITIONAL_CIE_1931_2_DEGREE_STANDARD_OBSERVER = (
    CaseInsensitiveMapping({
        'Cool White FL':
            np.array([0.369256318971281, 0.372549878176631]),
        'Daylight FL':
            np.array([0.312662993963651, 0.331985688793009]),
        'HPS':
            np.array([0.521677696062816, 0.417971177117239]),
        'Incandescent':
            np.array([0.450730217519680, 0.408046128945005]),
        'LPS':
            np.array([0.575151311365165, 0.424232234924905]),
        'Mercury':
            np.array([0.392018457637112, 0.383777071984453]),
        'Metal Halide':
            np.array([0.372544558972793, 0.385603925927588]),
        'Neodimium Incandescent':
            np.array([0.447398697052100, 0.395008601248268]),
        'Super HPS':
            np.array([0.470061659271846, 0.406116584248741]),
        'Triphosphor FL':
            np.array([0.413163268257275, 0.396422053758680])
    }))
"""
Traditional light sources chromaticity coordinates from *NIST*
*NIST CQS simulation 7.4.xls* spreadsheet.

References
----------
:cite:`Ohno2008a`

LIGHT_SOURCES_NIST_TRADITIONAL_CIE_1931_2_DEGREE_STANDARD_OBSERVER :
    CaseInsensitiveMapping
    **{'Cool White FL', 'Daylight FL', 'HPS', 'Incandescent', 'LPS', 'Mercury',
    'Metal Halide', 'Neodimium Incandescent', 'Super HPS', 'Triphosphor FL'}**
"""

LIGHT_SOURCES_NIST_TRADITIONAL_CIE_1964_10_DEGREE_STANDARD_OBSERVER = (
    CaseInsensitiveMapping({
        'Cool White FL':
            np.array([0.376715047518455, 0.364576802118673]),
        'Daylight FL':
            np.array([0.317395878738965, 0.330780819136676]),
        'HPS':
            np.array([0.531764495177513, 0.408752715284645]),
        'Incandescent':
            np.array([0.454365604973572, 0.406573684216774]),
        'LPS':
            np.array([0.589960045887891, 0.410039954112109]),
        'Mercury':
            np.array([0.401266412873755, 0.364732538221183]),
        'Metal Halide':
            np.array([0.378786167751226, 0.377496928504661]),
        'Neodimium Incandescent':
            np.array([0.447516717156694, 0.396734151368497]),
        'Super HPS':
            np.array([0.473859567146135, 0.401381825309197]),
        'Triphosphor FL':
            np.array([0.418591963931736, 0.388947713332192])
    }))
"""
Traditional light sources chromaticity coordinates from *NIST*
*NIST CQS simulation 7.4.xls* spreadsheet. [2]_

LIGHT_SOURCES_NIST_TRADITIONAL_CIE_1964_10_DEGREE_STANDARD_OBSERVER :
    CaseInsensitiveMapping
    **{'Cool White FL', 'Daylight FL', 'HPS', 'Incandescent', 'LPS', 'Mercury',
    'Metal Halide', 'Neodimium Incandescent', 'Super HPS', 'Triphosphor FL'}**
"""

LIGHT_SOURCES_NIST_LED_CIE_1931_2_DEGREE_STANDARD_OBSERVER = (
    CaseInsensitiveMapping({
        '3-LED-1 (457/540/605)':
            np.array([0.417057686949170, 0.396262457986602]),
        '3-LED-2 (473/545/616)':
            np.array([0.417060475566006, 0.396268120523418]),
        '3-LED-2 Yellow':
            np.array([0.436563079184047, 0.443649619298676]),
        '3-LED-3 (465/546/614)':
            np.array([0.380460502184482, 0.376772001481922]),
        '3-LED-4 (455/547/623)':
            np.array([0.417067943691045, 0.396276280071757]),
        '4-LED No Yellow':
            np.array([0.417060589301332, 0.396268153712350]),
        '4-LED Yellow':
            np.array([0.417069637940463, 0.396276766014859]),
        '4-LED-1 (461/526/576/624)':
            np.array([0.417067615440556, 0.396275056779587]),
        '4-LED-2 (447/512/573/627)':
            np.array([0.417071570560054, 0.396278745130373]),
        'Luxeon WW 2880':
            np.array([0.459088527920913, 0.432916480607903]),
        'PHOS-1':
            np.array([0.436443167801164, 0.404616033549917]),
        'PHOS-2':
            np.array([0.452704462198571, 0.437584543052711]),
        'PHOS-3':
            np.array([0.436899870751359, 0.404037372134463]),
        'PHOS-4':
            np.array([0.436936023906427, 0.404113558278629]),
        'Phosphor LED YAG':
            np.array([0.307761817314310, 0.325268939239941])
    }))
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
    CaseInsensitiveMapping({
        '3-LED-1 (457/540/605)':
            np.array([0.425099988926548, 0.389451349911075]),
        '3-LED-2 (473/545/616)':
            np.array([0.422222118774217, 0.401298495594226]),
        '3-LED-2 Yellow':
            np.array([0.446222216139125, 0.441646464276087]),
        '3-LED-3 (465/546/614)':
            np.array([0.387470465801936, 0.376404716015666]),
        '3-LED-4 (455/547/623)':
            np.array([0.422865464107041, 0.388772240171637]),
        '4-LED No Yellow':
            np.array([0.419807532952439, 0.399465294930377]),
        '4-LED Yellow':
            np.array([0.422720601750053, 0.390284663473479]),
        '4-LED-1 (461/526/576/624)':
            np.array([0.423899783323037, 0.394170886226971]),
        '4-LED-2 (447/512/573/627)':
            np.array([0.421571042053867, 0.394089741928601]),
        'Luxeon WW 2880':
            np.array([0.466639299623263, 0.430817417218051]),
        'PHOS-1':
            np.array([0.440120001281140, 0.403135783393416]),
        'PHOS-2':
            np.array([0.461487398870558, 0.436150294667024]),
        'PHOS-3':
            np.array([0.440892655302172, 0.408662264402299]),
        'PHOS-4':
            np.array([0.441760443951475, 0.407267478268879]),
        'Phosphor LED YAG':
            np.array([0.312807834772696, 0.334180937864035])
    }))
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
    CaseInsensitiveMapping({
        '60 A/W (Soft White)':
            np.array([0.450730217519680, 0.408046128945005]),
        'C100S54 (HPS)':
            np.array([0.529231515407657, 0.411370164988427]),
        'C100S54C (HPS)':
            np.array([0.502380414374839, 0.415877299905475]),
        'F32T8/TL830 (Triphosphor)':
            np.array([0.443250764475753, 0.409523700296928]),
        'F32T8/TL835 (Triphosphor)':
            np.array([0.407150274569933, 0.393172743482571]),
        'F32T8/TL841 (Triphosphor)':
            np.array([0.385376686681605, 0.390370762102806]),
        'F32T8/TL850 (Triphosphor)':
            np.array([0.343768910392287, 0.358447436104108]),
        'F32T8/TL865 /PLUS (Triphosphor)':
            np.array([0.316368879615201, 0.345320790143017]),
        'F34/CW/RS/EW (Cool White FL)':
            np.array([0.377250931364378, 0.393087658636060]),
        'F34T12/LW/RS /EW':
            np.array([0.378863642993776, 0.394960629979820]),
        'F34T12WW/RS /EW (Warm White FL)':
            np.array([0.438466967656789, 0.408635441565706]),
        'F40/C50 (Broadband FL)':
            np.array([0.345836574973021, 0.361724450389430]),
        'F40/C75 (Broadband FL)':
            np.array([0.299966663385220, 0.316582165804824]),
        'F40/CWX (Broadband FL)':
            np.array([0.375037045754214, 0.360543952129462]),
        'F40/DX (Broadband FL)':
            np.array([0.311922310746537, 0.342802103417329]),
        'F40/DXTP (Delux FL)':
            np.array([0.313066543826958, 0.342225714484412]),
        'F40/N (Natural FL)':
            np.array([0.376878697365115, 0.354153458302878]),
        'H38HT-100 (Mercury)':
            np.array([0.311200590193641, 0.382944245857018]),
        'H38JA-100/DX (Mercury DX)':
            np.array([0.389791630360359, 0.373394688931767]),
        'MHC100/U/MP /3K':
            np.array([0.428581768670222, 0.388168915678330]),
        'MHC100/U/MP /4K':
            np.array([0.373145253482762, 0.371366990216717]),
        'SDW-T 100W/LV (Super HPS)':
            np.array([0.472339157938672, 0.407106330880316])
    }))
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
    CaseInsensitiveMapping({
        '60 A/W (Soft White)':
            np.array([0.454365604973572, 0.406573684216774]),
        'C100S54 (HPS)':
            np.array([0.538554605063010, 0.402575827972962]),
        'C100S54C (HPS)':
            np.array([0.509663059970892, 0.409064508209193]),
        'F32T8/TL830 (Triphosphor)':
            np.array([0.448795219301811, 0.403574636091678]),
        'F32T8/TL835 (Triphosphor)':
            np.array([0.412082534290652, 0.388001071127592]),
        'F32T8/TL841 (Triphosphor)':
            np.array([0.390908619219527, 0.385290559992705]),
        'F32T8/TL850 (Triphosphor)':
            np.array([0.347882431257452, 0.355845742210551]),
        'F32T8/TL865 /PLUS (Triphosphor)':
            np.array([0.320698199593768, 0.343871441043854]),
        'F34/CW/RS/EW (Cool White FL)':
            np.array([0.386514853545337, 0.382843326097814]),
        'F34T12/LW/RS /EW':
            np.array([0.389628909159399, 0.382074721889904]),
        'F34T12WW/RS /EW (Warm White FL)':
            np.array([0.448395377616960, 0.395666643335296]),
        'F40/C50 (Broadband FL)':
            np.array([0.349880827196884, 0.360661316491439]),
        'F40/C75 (Broadband FL)':
            np.array([0.301988533872761, 0.318479025875818]),
        'F40/CWX (Broadband FL)':
            np.array([0.378502309910296, 0.356371890168937]),
        'F40/DX (Broadband FL)':
            np.array([0.316783037559153, 0.341749269085077]),
        'F40/DXTP (Delux FL)':
            np.array([0.318774745065791, 0.339798825605488]),
        'F40/N (Natural FL)':
            np.array([0.378833157741751, 0.350724402658646]),
        'H38HT-100 (Mercury)':
            np.array([0.326260627082484, 0.360001095895205]),
        'H38JA-100/DX (Mercury DX)':
            np.array([0.397058597517533, 0.356532431806974]),
        'MHC100/U/MP /3K':
            np.array([0.431422986591898, 0.380642213887539]),
        'MHC100/U/MP /4K':
            np.array([0.375707105948115, 0.366156465779779]),
        'SDW-T 100W/LV (Super HPS)':
            np.array([0.476461908192661, 0.402288012403575])
    }))
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
    CaseInsensitiveMapping({
        'Kinoton 75P': np.array([0.315252413629716, 0.332870794805328])
    }))
"""
Projectors and Xenon Arc Lamps.

References
----------
:cite:`Houston2015a`

LIGHT_SOURCES_PROJECTORS_CIE_1931_2_DEGREE_STANDARD_OBSERVER :
    CaseInsensitiveMapping
    **{'Kinoton 75P', }**
"""

LIGHT_SOURCES_PROJECTORS_CIE_1964_10_DEGREE_STANDARD_OBSERVER = (
    CaseInsensitiveMapping({
        'Kinoton 75P': np.array([0.317086642148234, 0.336222428041514])
    }))
"""
Projectors and Xenon Arc Lamps. [3_]

LIGHT_SOURCES_PROJECTORS_CIE_1964_10_DEGREE_STANDARD_OBSERVER :
    CaseInsensitiveMapping
    **{'Kinoton 75P', }**
"""

LIGHT_SOURCES = CaseInsensitiveMapping({
    'CIE 1931 2 Degree Standard Observer':
        CaseInsensitiveMapping(
            LIGHT_SOURCES_RIT_CIE_1931_2_DEGREE_STANDARD_OBSERVER),
    'CIE 1964 10 Degree Standard Observer':
        CaseInsensitiveMapping(
            LIGHT_SOURCES_RIT_CIE_1964_10_DEGREE_STANDARD_OBSERVER)
})
LIGHT_SOURCES.__doc__ = """
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
