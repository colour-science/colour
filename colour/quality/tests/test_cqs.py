# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.quality.cqs` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.quality import CQS_Specification, colour_quality_scale
from colour.colorimetry import ILLUMINANTS_SDS, LIGHT_SOURCES_SDS
from colour.quality.cqs import VS_ColorimetryData, VS_ColourQualityScaleData

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['TestColourQualityScale']


class TestColourQualityScale(unittest.TestCase):
    """
    Defines :func:`colour.quality.cqs.colour_quality_scale` definition unit
    tests methods.
    """

    def test_colour_quality_scale(self):
        """
        Tests :func:`colour.quality.cqs.colour_quality_scale` definition.
        """

        self.assertAlmostEqual(
            colour_quality_scale(ILLUMINANTS_SDS['FL1']),
            74.933405395713180,
            places=7)

        self.assertAlmostEqual(
            colour_quality_scale(
                ILLUMINANTS_SDS['FL1'], method='NIST CQS 7.4'),
            75.332008182589348,
            places=7)

        self.assertAlmostEqual(
            colour_quality_scale(ILLUMINANTS_SDS['FL2']),
            64.017283509280588,
            places=7)

        self.assertAlmostEqual(
            colour_quality_scale(
                ILLUMINANTS_SDS['FL2'], method='NIST CQS 7.4'),
            64.686339173112856,
            places=7)

        self.assertAlmostEqual(
            colour_quality_scale(LIGHT_SOURCES_SDS['Neodimium Incandescent']),
            89.693921013642381,
            places=7)

        self.assertAlmostEqual(
            colour_quality_scale(
                LIGHT_SOURCES_SDS['Neodimium Incandescent'],
                method='NIST CQS 7.4'),
            87.655035241231985,
            places=7)

        self.assertAlmostEqual(
            colour_quality_scale(
                LIGHT_SOURCES_SDS['F32T8/TL841 (Triphosphor)']),
            84.878441814420910,
            places=7)

        self.assertAlmostEqual(
            colour_quality_scale(
                LIGHT_SOURCES_SDS['F32T8/TL841 (Triphosphor)'],
                method='NIST CQS 7.4'),
            83.179881092827671,
            places=7)

        self.assertAlmostEqual(
            colour_quality_scale(LIGHT_SOURCES_SDS['H38HT-100 (Mercury)']),
            19.836071708638958,
            places=7)

        self.assertAlmostEqual(
            colour_quality_scale(
                LIGHT_SOURCES_SDS['H38HT-100 (Mercury)'],
                method='NIST CQS 7.4'),
            22.860610106043985,
            places=7)

        self.assertAlmostEqual(
            colour_quality_scale(LIGHT_SOURCES_SDS['Luxeon WW 2880']),
            86.491761709787994,
            places=7)

        self.assertAlmostEqual(
            colour_quality_scale(
                LIGHT_SOURCES_SDS['Luxeon WW 2880'], method='NIST CQS 7.4'),
            84.879524259605077,
            places=7)

        specification_r = CQS_Specification(
            name='FL1',
            Q_a=75.332008182589348,
            Q_f=76.339439008799872,
            Q_p=74.235391056498486,
            Q_g=84.222748330112054,
            Q_d=84.126957016581841,
            Q_as={
                1:
                    VS_ColourQualityScaleData(
                        name='VS1',
                        Q_a=77.680854801718439,
                        D_C_ab=-1.411454116288503,
                        D_E_ab=7.191809380490093,
                        D_Ep_ab=7.191809380490093),
                2:
                    VS_ColourQualityScaleData(
                        name='VS2',
                        Q_a=98.449964521835454,
                        D_C_ab=2.416949235948934,
                        D_E_ab=2.468031951275495,
                        D_Ep_ab=0.499537889816771),
                3:
                    VS_ColourQualityScaleData(
                        name='VS3',
                        Q_a=85.128488974188656,
                        D_C_ab=2.051673516490851,
                        D_E_ab=5.212485950404579,
                        D_Ep_ab=4.791726783206154),
                4:
                    VS_ColourQualityScaleData(
                        name='VS4',
                        Q_a=75.929101405967231,
                        D_C_ab=-1.544244803829976,
                        D_E_ab=7.756423964663330,
                        D_Ep_ab=7.756423964663330),
                5:
                    VS_ColourQualityScaleData(
                        name='VS5',
                        Q_a=76.560036127768228,
                        D_C_ab=-5.962959343458696,
                        D_E_ab=7.553059580932679,
                        D_Ep_ab=7.553059580932679),
                6:
                    VS_ColourQualityScaleData(
                        name='VS6',
                        Q_a=75.674875949078682,
                        D_C_ab=-7.785240663276603,
                        D_E_ab=7.838368326299190,
                        D_Ep_ab=7.838368326299190),
                7:
                    VS_ColourQualityScaleData(
                        name='VS7',
                        Q_a=74.856012003857145,
                        D_C_ab=-6.773721961614541,
                        D_E_ab=8.102319790027652,
                        D_Ep_ab=8.102319790027652),
                8:
                    VS_ColourQualityScaleData(
                        name='VS8',
                        Q_a=85.007243427918965,
                        D_C_ab=-0.684332522268967,
                        D_E_ab=4.830795743950247,
                        D_Ep_ab=4.830795743950247),
                9:
                    VS_ColourQualityScaleData(
                        name='VS9',
                        Q_a=95.600642739549912,
                        D_C_ab=3.245002214218772,
                        D_E_ab=3.541112179773836,
                        D_Ep_ab=1.417545801537952),
                10:
                    VS_ColourQualityScaleData(
                        name='VS10',
                        Q_a=84.897394198279201,
                        D_C_ab=3.2779983437733051,
                        D_E_ab=5.867290950207070,
                        D_Ep_ab=4.866192551944616),
                11:
                    VS_ColourQualityScaleData(
                        name='VS11',
                        Q_a=76.652664070814652,
                        D_C_ab=0.889135877229947,
                        D_E_ab=7.575563455510737,
                        D_Ep_ab=7.523204042181251),
                12:
                    VS_ColourQualityScaleData(
                        name='VS12',
                        Q_a=74.392341352838727,
                        D_C_ab=-1.950461685183292,
                        D_E_ab=8.251784061375552,
                        D_Ep_ab=8.251784061375552),
                13:
                    VS_ColourQualityScaleData(
                        name='VS13',
                        Q_a=71.733157403665416,
                        D_C_ab=-5.748914457790455,
                        D_E_ab=9.109057119322612,
                        D_Ep_ab=9.109057119322612),
                14:
                    VS_ColourQualityScaleData(
                        name='VS14',
                        Q_a=53.322108325530110,
                        D_C_ab=-14.642967562714134,
                        D_E_ab=15.053589749938459,
                        D_Ep_ab=15.053589749938459),
                15:
                    VS_ColourQualityScaleData(
                        name='VS15',
                        Q_a=62.611167146423085,
                        D_C_ab=-10.071676496831763,
                        D_E_ab=12.051527779341935,
                        D_Ep_ab=12.051527779341935)
            },
            colorimetry_data=([
                VS_ColorimetryData(
                    name='VS1',
                    XYZ=np.array([0.13184562, 0.09885158, 0.22533524]),
                    Lab=np.array([37.63552369, 27.62126274, -26.23567072]),
                    C=38.095204070389400),
                VS_ColorimetryData(
                    name='VS2',
                    XYZ=np.array([0.13066767, 0.10335651, 0.30768635]),
                    Lab=np.array([38.43822294, 23.38828386, -37.84013645]),
                    C=44.484691170961092),
                VS_ColorimetryData(
                    name='VS3',
                    XYZ=np.array([0.10087152, 0.09197794, 0.32255009]),
                    Lab=np.array([36.36236009, 11.00844158, -43.50665998]),
                    C=44.877781233091532),
                VS_ColorimetryData(
                    name='VS4',
                    XYZ=np.array([0.13339874, 0.15690626, 0.38428052]),
                    Lab=np.array([46.56595565, -9.85969969, -33.955993]),
                    C=35.358494572170223),
                VS_ColorimetryData(
                    name='VS5',
                    XYZ=np.array([0.18662805, 0.24704107, 0.40046308]),
                    Lab=np.array([56.78597632, -23.14117043, -18.29881674]),
                    C=29.501872194883592),
                VS_ColorimetryData(
                    name='VS6',
                    XYZ=np.array([0.15841746, 0.24152639, 0.26931454]),
                    Lab=np.array([56.24029681, -36.23772023, -1.42733778]),
                    C=36.265819453020534),
                VS_ColorimetryData(
                    name='VS7',
                    XYZ=np.array([0.14989922, 0.24926212, 0.138238]),
                    Lab=np.array([57.003456, -44.54988402, 24.99942371]),
                    C=51.084864212342254),
                VS_ColorimetryData(
                    name='VS8',
                    XYZ=np.array([0.26139991, 0.36817885, 0.11429053]),
                    Lab=np.array([67.14017484, -33.23131783, 48.67413212]),
                    C=58.936335329004450),
                VS_ColorimetryData(
                    name='VS9',
                    XYZ=np.array([0.42407924, 0.5285824, 0.11437796]),
                    Lab=np.array([77.791228, -22.23481066, 67.01388225]),
                    C=70.606283139833707),
                VS_ColorimetryData(
                    name='VS10',
                    XYZ=np.array([0.553588, 0.62024037, 0.09664934]),
                    Lab=np.array([82.9262011, -8.87753163, 81.03696819]),
                    C=81.521781025372348),
                VS_ColorimetryData(
                    name='VS11',
                    XYZ=np.array([0.39748223, 0.39523993, 0.0573943]),
                    Lab=np.array([69.1291429, 6.94195834, 71.52497069]),
                    C=71.861061901342111),
                VS_ColorimetryData(
                    name='VS12',
                    XYZ=np.array([0.43749683, 0.38968504, 0.08635639]),
                    Lab=np.array([68.72844451, 20.81629234, 59.85617607]),
                    C=63.372548004889417),
                VS_ColorimetryData(
                    name='VS13',
                    XYZ=np.array([0.34652079, 0.27550838, 0.08903238]),
                    Lab=np.array([59.48075943, 31.8153513, 43.0302754]),
                    C=53.514681906739007),
                VS_ColorimetryData(
                    name='VS14',
                    XYZ=np.array([0.14262899, 0.09104011, 0.04951868]),
                    Lab=np.array([36.18378646, 40.75048676, 18.33518283]),
                    C=44.685356670534020),
                VS_ColorimetryData(
                    name='VS15',
                    XYZ=np.array([0.13589052, 0.092129, 0.11599991]),
                    Lab=np.array([36.39100967, 35.60560293, -4.81131533]),
                    C=35.929204214045100)
            ], [
                VS_ColorimetryData(
                    name='VS1',
                    XYZ=np.array([0.15204948, 0.10842571, 0.21637094]),
                    Lab=np.array([39.31404365, 32.98305053, -21.74613572]),
                    C=39.506658186677903),
                VS_ColorimetryData(
                    name='VS2',
                    XYZ=np.array([0.13188419, 0.10619651, 0.29958251]),
                    Lab=np.array([38.93233648, 22.05678627, -35.82168464]),
                    C=42.067741935012158),
                VS_ColorimetryData(
                    name='VS3',
                    XYZ=np.array([0.10125948, 0.09854418, 0.32974284]),
                    Lab=np.array([37.57986983, 6.06363078, -42.39466811]),
                    C=42.826107716600681),
                VS_ColorimetryData(
                    name='VS4',
                    XYZ=np.array([0.13147324, 0.16804591, 0.39334981]),
                    Lab=np.array([48.01287105, -17.35257125, -32.5683964]),
                    C=36.902739376000198),
                VS_ColorimetryData(
                    name='VS5',
                    XYZ=np.array([0.18148174, 0.25847203, 0.41337803]),
                    Lab=np.array([57.89173056, -30.60333248, -17.92178331]),
                    C=35.464831538342288),
                VS_ColorimetryData(
                    name='VS6',
                    XYZ=np.array([0.15185555, 0.25077644, 0.28171886]),
                    Lab=np.array([57.15099451, -44.01579207, -1.76236936]),
                    C=44.051060116297137),
                VS_ColorimetryData(
                    name='VS7',
                    XYZ=np.array([0.1395656, 0.25329693, 0.1447564]),
                    Lab=np.array([57.39525164, -52.59411951, 24.11378416]),
                    C=57.858586173956795),
                VS_ColorimetryData(
                    name='VS8',
                    XYZ=np.array([0.24671064, 0.36211041, 0.11980564]),
                    Lab=np.array([66.68086148, -37.45951248, 46.3832832]),
                    C=59.620667851273417),
                VS_ColorimetryData(
                    name='VS9',
                    XYZ=np.array([0.4081725, 0.508617, 0.11898838]),
                    Lab=np.array([76.59515888, -21.91694258, 63.69607363]),
                    C=67.361280925614935),
                VS_ColorimetryData(
                    name='VS10',
                    XYZ=np.array([0.56031473, 0.60567807, 0.10172838]),
                    Lab=np.array([82.14584971, -3.82960002, 78.15000763]),
                    C=78.243782681599043),
                VS_ColorimetryData(
                    name='VS11',
                    XYZ=np.array([0.40536278, 0.38001713, 0.05756101]),
                    Lab=np.array([68.02187691, 14.16935995, 69.5431055]),
                    C=70.971926024112165),
                VS_ColorimetryData(
                    name='VS12',
                    XYZ=np.array([0.45402541, 0.37917641, 0.08624766]),
                    Lab=np.array([67.95987006, 28.93047228, 58.56725509]),
                    C=65.323009690072709),
                VS_ColorimetryData(
                    name='VS13',
                    XYZ=np.array([0.37202687, 0.27411307, 0.08885905]),
                    Lab=np.array([59.35311972, 40.9220464, 42.86677002]),
                    C=59.263596364529462),
                VS_ColorimetryData(
                    name='VS14',
                    XYZ=np.array([0.19305452, 0.11049115, 0.04885402]),
                    Lab=np.array([39.66307016, 53.96236134, 24.65590426]),
                    C=59.328324233248154),
                VS_ColorimetryData(
                    name='VS15',
                    XYZ=np.array([0.17304703, 0.10699393, 0.11285076]),
                    Lab=np.array([39.0694885, 45.99591907, 0.67561493]),
                    C=46.000880710876864)
            ]))

        specification_t = colour_quality_scale(
            ILLUMINANTS_SDS['FL1'],
            additional_data=True,
            method='NIST CQS 7.4')

        np.testing.assert_almost_equal(
            [
                data.Q_a
                for _index, data in sorted(specification_r.Q_as.items())
            ],
            [
                data.Q_a
                for _index, data in sorted(specification_t.Q_as.items())
            ],
            decimal=7,
        )

        specification_r = CQS_Specification(
            name='FL1',
            Q_a=74.933405395713152,
            Q_f=75.895019131814692,
            Q_p=None,
            Q_g=83.859802969126989,
            Q_d=None,
            Q_as={
                1:
                    VS_ColourQualityScaleData(
                        name='VS1',
                        Q_a=51.884484734774539,
                        D_C_ab=-14.642964823527400,
                        D_E_ab=15.053586922553365,
                        D_Ep_ab=15.053586922553365),
                2:
                    VS_ColourQualityScaleData(
                        name='VS2',
                        Q_a=70.859386580835093,
                        D_C_ab=-5.748912901584454,
                        D_E_ab=9.109057753686301,
                        D_Ep_ab=9.109057753686301),
                3:
                    VS_ColourQualityScaleData(
                        name='VS3',
                        Q_a=74.615079552044733,
                        D_C_ab=-0.199055128982607,
                        D_E_ab=7.934584371287040,
                        D_Ep_ab=7.934584371287040),
                4:
                    VS_ColourQualityScaleData(
                        name='VS4',
                        Q_a=85.470454126502858,
                        D_C_ab=2.620284036804321,
                        D_E_ab=5.242841277796065,
                        D_Ep_ab=4.541089762450278),
                5:
                    VS_ColourQualityScaleData(
                        name='VS5',
                        Q_a=95.464568465261436,
                        D_C_ab=3.244999121708176,
                        D_E_ab=3.541109287718627,
                        D_Ep_ab=1.417545656294811),
                6:
                    VS_ColourQualityScaleData(
                        name='VS6',
                        Q_a=84.543583902150289,
                        D_C_ab=-0.684333146377760,
                        D_E_ab=4.830795633277699,
                        D_Ep_ab=4.830795633277699),
                7:
                    VS_ColourQualityScaleData(
                        name='VS7',
                        Q_a=74.078633444044030,
                        D_C_ab=-6.773727982269591,
                        D_E_ab=8.102322827150674,
                        D_Ep_ab=8.102322827150674),
                8:
                    VS_ColourQualityScaleData(
                        name='VS8',
                        Q_a=74.922760174879087,
                        D_C_ab=-7.785252182726744,
                        D_E_ab=7.838379721335357,
                        D_Ep_ab=7.838379721335357),
                9:
                    VS_ColourQualityScaleData(
                        name='VS9',
                        Q_a=75.835254560528938,
                        D_C_ab=-5.962964833116001,
                        D_E_ab=7.553073243258186,
                        D_Ep_ab=7.553073243258186),
                10:
                    VS_ColourQualityScaleData(
                        name='VS10',
                        Q_a=75.184803574188905,
                        D_C_ab=-1.544240934167505,
                        D_E_ab=7.756446084611037,
                        D_Ep_ab=7.756446084611037),
                11:
                    VS_ColourQualityScaleData(
                        name='VS11',
                        Q_a=78.089782546668729,
                        D_C_ab=1.745770231264103,
                        D_E_ab=7.067228869124241,
                        D_Ep_ab=6.848212189193262),
                12:
                    VS_ColourQualityScaleData(
                        name='VS12',
                        Q_a=91.955688469006077,
                        D_C_ab=2.855869216256835,
                        D_E_ab=3.804866911398577,
                        D_Ep_ab=2.514164519893699),
                13:
                    VS_ColourQualityScaleData(
                        name='VS13',
                        Q_a=81.468184220490429,
                        D_C_ab=0.030402986778803,
                        D_E_ab=5.792177528304779,
                        D_Ep_ab=5.792097735518090),
                14:
                    VS_ColourQualityScaleData(
                        name='VS14',
                        Q_a=76.482628265932789,
                        D_C_ab=-5.080750395110528,
                        D_E_ab=7.350669246766205,
                        D_Ep_ab=7.350669246766205),
                15:
                    VS_ColourQualityScaleData(
                        name='VS15',
                        Q_a=65.370026142948518,
                        D_C_ab=-9.509238816066890,
                        D_E_ab=10.826397690468854,
                        D_Ep_ab=10.826397690468854)
            },
            colorimetry_data=([
                VS_ColorimetryData(
                    name='VS1',
                    XYZ=np.array([0.14262899, 0.09104012, 0.04951869]),
                    Lab=np.array([36.18378647, 40.75048793, 18.33517831]),
                    C=44.685355882758152),
                VS_ColorimetryData(
                    name='VS2',
                    XYZ=np.array([0.3465208, 0.27550838, 0.08903238]),
                    Lab=np.array([59.48075944, 31.81535155, 43.03027426]),
                    C=53.514681133664730),
                VS_ColorimetryData(
                    name='VS3',
                    XYZ=np.array([0.41042219, 0.38688299, 0.06618968]),
                    Lab=np.array([68.5248742, 13.56049316, 66.82043633]),
                    C=68.182532115437340),
                VS_ColorimetryData(
                    name='VS4',
                    XYZ=np.array([0.45420723, 0.50405653, 0.08194422]),
                    Lab=np.array([76.31757819, -7.04196646, 74.43516148]),
                    C=74.767523404153394),
                VS_ColorimetryData(
                    name='VS5',
                    XYZ=np.array([0.42407924, 0.5285824, 0.11437796]),
                    Lab=np.array([77.79122799, -22.23481106, 67.01388407]),
                    C=70.606284995201392),
                VS_ColorimetryData(
                    name='VS6',
                    XYZ=np.array([0.26139991, 0.36817885, 0.11429053]),
                    Lab=np.array([67.14017484, -33.23131787, 48.67413226]),
                    C=58.936335472298275),
                VS_ColorimetryData(
                    name='VS7',
                    XYZ=np.array([0.14989922, 0.24926212, 0.13823798]),
                    Lab=np.array([57.00345599, -44.54988535, 24.99942632]),
                    C=51.084866649348065),
                VS_ColorimetryData(
                    name='VS8',
                    XYZ=np.array([0.15841745, 0.24152639, 0.2693145]),
                    Lab=np.array([56.24029678, -36.23772472, -1.42733192]),
                    C=36.265823711788478),
                VS_ColorimetryData(
                    name='VS9',
                    XYZ=np.array([0.18662804, 0.24704107, 0.40046303]),
                    Lab=np.array([56.78597628, -23.14117544, -18.29881115]),
                    C=29.501872653031871),
                VS_ColorimetryData(
                    name='VS10',
                    XYZ=np.array([0.13339873, 0.15690626, 0.38428046]),
                    Lab=np.array([46.56595559, -9.85970727, -33.95598605]),
                    C=35.358490015776226),
                VS_ColorimetryData(
                    name='VS11',
                    XYZ=np.array([0.09902437, 0.09951361, 0.32046526]),
                    Lab=np.array([37.75499466, 3.55194161, -40.81669619]),
                    C=40.970952848102279),
                VS_ColorimetryData(
                    name='VS12',
                    XYZ=np.array([0.11579654, 0.09612863, 0.31943786]),
                    Lab=np.array([37.1384526, 18.80505207, -41.7368833]),
                    C=45.777695561949436),
                VS_ColorimetryData(
                    name='VS13',
                    XYZ=np.array([0.20977187, 0.16844709, 0.37299873]),
                    Lab=np.array([48.06376988, 26.00174066, -29.97180426]),
                    C=39.678704214193381),
                VS_ColorimetryData(
                    name='VS14',
                    XYZ=np.array([0.32294822, 0.2416386, 0.3623672]),
                    Lab=np.array([56.25148226, 37.44488177, -14.50799142]),
                    C=40.157203412534642),
                VS_ColorimetryData(
                    name='VS15',
                    XYZ=np.array([0.22031646, 0.15367587, 0.17564225]),
                    Lab=np.array([46.13360422, 39.30141636, -2.12353664]),
                    C=39.358744081490819)
            ], [
                VS_ColorimetryData(
                    name='VS1',
                    XYZ=np.array([0.19305453, 0.11049115, 0.04885406]),
                    Lab=np.array([39.6630702, 53.96236472, 24.65588837]),
                    C=59.328320706285552),
                VS_ColorimetryData(
                    name='VS2',
                    XYZ=np.array([0.37202688, 0.27411307, 0.08885906]),
                    Lab=np.array([59.35311972, 40.92204723, 42.86676601]),
                    C=59.263594035249184),
                VS_ColorimetryData(
                    name='VS3',
                    XYZ=np.array([0.42075427, 0.37269638, 0.06621283]),
                    Lab=np.array([67.47883546, 21.21408713, 65.00772248]),
                    C=68.381587244419947),
                VS_ColorimetryData(
                    name='VS4',
                    XYZ=np.array([0.4619684, 0.49480415, 0.08591535]),
                    Lab=np.array([75.74923125, -2.37840511, 72.10802547]),
                    C=72.147239367349073),
                VS_ColorimetryData(
                    name='VS5',
                    XYZ=np.array([0.40817249, 0.508617, 0.11898836]),
                    Lab=np.array([76.59515887, -21.91694373, 63.69607847]),
                    C=67.361285873493216),
                VS_ColorimetryData(
                    name='VS6',
                    XYZ=np.array([0.24671064, 0.36211041, 0.11980563]),
                    Lab=np.array([66.68086148, -37.45951274, 46.38328397]),
                    C=59.620668618676035),
                VS_ColorimetryData(
                    name='VS7',
                    XYZ=np.array([0.1395656, 0.25329693, 0.14475636]),
                    Lab=np.array([57.39525162, -52.59412462, 24.11379331]),
                    C=57.858594631617656),
                VS_ColorimetryData(
                    name='VS8',
                    XYZ=np.array([0.15185552, 0.25077644, 0.28171872]),
                    Lab=np.array([57.15099443, -44.01580867, -1.76234921]),
                    C=44.051075894515222),
                VS_ColorimetryData(
                    name='VS9',
                    XYZ=np.array([0.1814817, 0.25847202, 0.41337786]),
                    Lab=np.array([57.89173046, -30.60335063, -17.92176409]),
                    C=35.464837486147871),
                VS_ColorimetryData(
                    name='VS10',
                    XYZ=np.array([0.1314732, 0.16804591, 0.39334961]),
                    Lab=np.array([48.0128709, -17.35259857, -32.5683723]),
                    C=36.902730949943731),
                VS_ColorimetryData(
                    name='VS11',
                    XYZ=np.array([0.0972772, 0.10656633, 0.32348741]),
                    Lab=np.array([38.99602774, -3.18915863, -39.09532221]),
                    C=39.225182616838175),
                VS_ColorimetryData(
                    name='VS12',
                    XYZ=np.array([0.11499913, 0.09966411, 0.31524145]),
                    Lab=np.array([37.78207842, 15.46056944, -40.04065396]),
                    C=42.921826345692601),
                VS_ColorimetryData(
                    name='VS13',
                    XYZ=np.array([0.23126112, 0.17972713, 0.36052054]),
                    Lab=np.array([49.46299499, 29.95421157, -25.97562318]),
                    C=39.648301227414578),
                VS_ColorimetryData(
                    name='VS14',
                    XYZ=np.array([0.35886253, 0.2560872, 0.35532873]),
                    Lab=np.array([57.66377054, 43.8380947, -11.16664308]),
                    C=45.237953807645169),
                VS_ColorimetryData(
                    name='VS15',
                    XYZ=np.array([0.26550827, 0.17191964, 0.17308227]),
                    Lab=np.array([48.50100521, 48.80442473, 2.49156155]),
                    C=48.86798289755771)
            ]))

        specification_t = colour_quality_scale(
            ILLUMINANTS_SDS['FL1'],
            additional_data=True,
            method='NIST CQS 9.0')

        np.testing.assert_almost_equal(
            [
                data.Q_a
                for _index, data in sorted(specification_r.Q_as.items())
            ],
            [
                data.Q_a
                for _index, data in sorted(specification_t.Q_as.items())
            ],
            decimal=7,
        )


if __name__ == '__main__':
    unittest.main()
