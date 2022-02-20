"""Defines the unit tests for the :mod:`colour.quality.cqs` module."""

import numpy as np
import unittest

from colour.quality import (
    ColourRendering_Specification_CQS,
    colour_quality_scale,
)
from colour.colorimetry import SDS_ILLUMINANTS, SDS_LIGHT_SOURCES
from colour.quality.cqs import VS_ColorimetryData, VS_ColourQualityScaleData

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestColourQualityScale",
]


class TestColourQualityScale(unittest.TestCase):
    """
    Define :func:`colour.quality.cqs.colour_quality_scale` definition unit
    tests methods.
    """

    def test_colour_quality_scale(self):
        """Test :func:`colour.quality.cqs.colour_quality_scale` definition."""

        self.assertAlmostEqual(
            colour_quality_scale(SDS_ILLUMINANTS["FL1"]),
            74.982585798279871,
            places=7,
        )

        self.assertAlmostEqual(
            colour_quality_scale(
                SDS_ILLUMINANTS["FL1"], method="NIST CQS 7.4"
            ),
            75.377089740493290,
            places=7,
        )

        self.assertAlmostEqual(
            colour_quality_scale(SDS_ILLUMINANTS["FL2"]),
            64.111703163816699,
            places=7,
        )

        self.assertAlmostEqual(
            colour_quality_scale(
                SDS_ILLUMINANTS["FL2"], method="NIST CQS 7.4"
            ),
            64.774490832419872,
            places=7,
        )

        self.assertAlmostEqual(
            colour_quality_scale(SDS_LIGHT_SOURCES["Neodimium Incandescent"]),
            89.737441458687044,
            places=7,
        )

        self.assertAlmostEqual(
            colour_quality_scale(
                SDS_LIGHT_SOURCES["Neodimium Incandescent"],
                method="NIST CQS 7.4",
            ),
            87.700319996664561,
            places=7,
        )

        self.assertAlmostEqual(
            colour_quality_scale(
                SDS_LIGHT_SOURCES["F32T8/TL841 (Triphosphor)"]
            ),
            84.934929181986888,
            places=7,
        )

        self.assertAlmostEqual(
            colour_quality_scale(
                SDS_LIGHT_SOURCES["F32T8/TL841 (Triphosphor)"],
                method="NIST CQS 7.4",
            ),
            83.255458192000233,
            places=7,
        )

        self.assertAlmostEqual(
            colour_quality_scale(SDS_LIGHT_SOURCES["H38HT-100 (Mercury)"]),
            20.019979778489535,
            places=7,
        )

        self.assertAlmostEqual(
            colour_quality_scale(
                SDS_LIGHT_SOURCES["H38HT-100 (Mercury)"], method="NIST CQS 7.4"
            ),
            23.011011107054145,
            places=7,
        )

        self.assertAlmostEqual(
            colour_quality_scale(SDS_LIGHT_SOURCES["Luxeon WW 2880"]),
            86.497986329513722,
            places=7,
        )

        self.assertAlmostEqual(
            colour_quality_scale(
                SDS_LIGHT_SOURCES["Luxeon WW 2880"], method="NIST CQS 7.4"
            ),
            84.887918431764191,
            places=7,
        )

        specification_r = ColourRendering_Specification_CQS(
            name="FL1",
            Q_a=75.37708974049329,
            Q_f=76.387614864939863,
            Q_p=74.266319651754912,
            Q_g=84.236404852169287,
            Q_d=84.137299859998564,
            Q_as={
                1: VS_ColourQualityScaleData(
                    name="VS1",
                    Q_a=77.696843895396597,
                    D_C_ab=-1.4488382159802313,
                    D_E_ab=7.1866560767284948,
                    D_Ep_ab=7.1866560767284948,
                ),
                2: VS_ColourQualityScaleData(
                    name="VS2",
                    Q_a=98.458786425718415,
                    D_C_ab=2.3828597530721822,
                    D_E_ab=2.4340762421637514,
                    D_Ep_ab=0.49669563100030184,
                ),
                3: VS_ColourQualityScaleData(
                    name="VS3",
                    Q_a=85.114145145039103,
                    D_C_ab=2.0800948785243278,
                    D_E_ab=5.2279782355370283,
                    D_Ep_ab=4.7963487912771443,
                ),
                4: VS_ColourQualityScaleData(
                    name="VS4",
                    Q_a=75.996470841598779,
                    D_C_ab=-1.5405082308077453,
                    D_E_ab=7.7347089860761855,
                    D_Ep_ab=7.7347089860761855,
                ),
                5: VS_ColourQualityScaleData(
                    name="VS5",
                    Q_a=76.592736502399703,
                    D_C_ab=-5.9501334881332397,
                    D_E_ab=7.5425196871488174,
                    D_Ep_ab=7.5425196871488174,
                ),
                6: VS_ColourQualityScaleData(
                    name="VS6",
                    Q_a=75.700247357264502,
                    D_C_ab=-7.777714756522542,
                    D_E_ab=7.8301903253053968,
                    D_Ep_ab=7.8301903253053968,
                ),
                7: VS_ColourQualityScaleData(
                    name="VS7",
                    Q_a=74.888666225318985,
                    D_C_ab=-6.7678158335932963,
                    D_E_ab=8.0917938479197353,
                    D_Ep_ab=8.0917938479197353,
                ),
                8: VS_ColourQualityScaleData(
                    name="VS8",
                    Q_a=85.018563761368782,
                    D_C_ab=-0.69329167385927803,
                    D_E_ab=4.8271479880620198,
                    D_Ep_ab=4.8271479880620198,
                ),
                9: VS_ColourQualityScaleData(
                    name="VS9",
                    Q_a=95.598698383415993,
                    D_C_ab=3.2171709711080183,
                    D_E_ab=3.5158784941231844,
                    D_Ep_ab=1.4181722490931066,
                ),
                10: VS_ColourQualityScaleData(
                    name="VS10",
                    Q_a=84.964267430208068,
                    D_C_ab=3.2356461456271859,
                    D_E_ab=5.8258030130410683,
                    D_Ep_ab=4.8446439257231608,
                ),
                11: VS_ColourQualityScaleData(
                    name="VS11",
                    Q_a=76.736800445076625,
                    D_C_ab=0.87588476209290889,
                    D_E_ab=7.5470837614115442,
                    D_Ep_ab=7.496085590846417,
                ),
                12: VS_ColourQualityScaleData(
                    name="VS12",
                    Q_a=74.434858693757448,
                    D_C_ab=-1.9423783441654692,
                    D_E_ab=8.238078426246771,
                    D_Ep_ab=8.238078426246771,
                ),
                13: VS_ColourQualityScaleData(
                    name="VS13",
                    Q_a=71.82010326219509,
                    D_C_ab=-5.7426330500006273,
                    D_E_ab=9.081024809959402,
                    D_Ep_ab=9.081024809959402,
                ),
                14: VS_ColourQualityScaleData(
                    name="VS14",
                    Q_a=53.401625187360011,
                    D_C_ab=-14.617738406753311,
                    D_E_ab=15.027848279487852,
                    D_Ep_ab=15.027848279487852,
                ),
                15: VS_ColourQualityScaleData(
                    name="VS15",
                    Q_a=62.672288506573636,
                    D_C_ab=-10.059255730250122,
                    D_E_ab=12.031799070041069,
                    D_Ep_ab=12.031799070041069,
                ),
            },
            colorimetry_data=(
                [
                    VS_ColorimetryData(
                        name="VS1",
                        XYZ=np.array([0.13183826, 0.09887241, 0.22510560]),
                        Lab=np.array([37.63929023, 27.59987425, -26.20530751]),
                        C=38.058786112947161,
                    ),
                    VS_ColorimetryData(
                        name="VS2",
                        XYZ=np.array([0.13061818, 0.10336027, 0.30741529]),
                        Lab=np.array([38.43888224, 23.35252921, -37.81858787]),
                        C=44.447566961689617,
                    ),
                    VS_ColorimetryData(
                        name="VS3",
                        XYZ=np.array([0.10089287, 0.09200351, 0.32265558]),
                        Lab=np.array([36.36721207, 11.00392324, -43.53144829]),
                        C=44.900705083686788,
                    ),
                    VS_ColorimetryData(
                        name="VS4",
                        XYZ=np.array([0.13339390, 0.15696027, 0.38421714]),
                        Lab=np.array([46.57313273, -9.89411460, -33.95550516]),
                        C=35.367638235231176,
                    ),
                    VS_ColorimetryData(
                        name="VS5",
                        XYZ=np.array([0.18662999, 0.24708620, 0.40043676]),
                        Lab=np.array(
                            [56.79040832, -23.15964295, -18.30798276]
                        ),
                        C=29.522047597875542,
                    ),
                    VS_ColorimetryData(
                        name="VS6",
                        XYZ=np.array([0.15843362, 0.24157338, 0.26933196]),
                        Lab=np.array([56.24498076, -36.24891195, -1.43946286]),
                        C=36.277481597875926,
                    ),
                    VS_ColorimetryData(
                        name="VS7",
                        XYZ=np.array([0.14991085, 0.24929718, 0.13823961]),
                        Lab=np.array([57.00687838, -44.55799945, 24.99093151]),
                        C=51.087786926248803,
                    ),
                    VS_ColorimetryData(
                        name="VS8",
                        XYZ=np.array([0.26141761, 0.36817692, 0.11429088]),
                        Lab=np.array([67.14003019, -33.22377274, 48.66064659]),
                        C=58.920943658800866,
                    ),
                    VS_ColorimetryData(
                        name="VS9",
                        XYZ=np.array([0.42410903, 0.52851922, 0.11439812]),
                        Lab=np.array([77.78749106, -22.21024952, 66.98873308]),
                        C=70.5746806093467,
                    ),
                    VS_ColorimetryData(
                        name="VS10",
                        XYZ=np.array([0.55367933, 0.62018757, 0.09672217]),
                        Lab=np.array([82.92339395, -8.84301088, 80.99721844]),
                        C=81.478513954529475,
                    ),
                    VS_ColorimetryData(
                        name="VS11",
                        XYZ=np.array([0.39755898, 0.39521027, 0.05739407]),
                        Lab=np.array([69.12701310, 6.97471851, 71.51095397]),
                        C=71.850283480415115,
                    ),
                    VS_ColorimetryData(
                        name="VS12",
                        XYZ=np.array([0.43757530, 0.38969458, 0.08630191]),
                        Lab=np.array([68.72913590, 20.83589132, 59.86354051]),
                        C=63.385943630694335,
                    ),
                    VS_ColorimetryData(
                        name="VS13",
                        XYZ=np.array([0.34657727, 0.27547744, 0.08900676]),
                        Lab=np.array([59.47793328, 31.84647528, 43.02166812]),
                        C=53.526273138196274,
                    ),
                    VS_ColorimetryData(
                        name="VS14",
                        XYZ=np.array([0.14271714, 0.09107438, 0.04949461]),
                        Lab=np.array([36.19033157, 40.77665898, 18.34813575]),
                        C=44.714539058373845,
                    ),
                    VS_ColorimetryData(
                        name="VS15",
                        XYZ=np.array([0.13593948, 0.09214669, 0.11591665]),
                        Lab=np.array([36.39436371, 35.62220213, -4.79596673]),
                        C=35.94360278722214,
                    ),
                ],
                [
                    VS_ColorimetryData(
                        name="VS1",
                        XYZ=np.array([0.15205130, 0.10842697, 0.21629425]),
                        Lab=np.array([39.31425803, 32.98285941, -21.74818073]),
                        C=39.507624328927392,
                    ),
                    VS_ColorimetryData(
                        name="VS2",
                        XYZ=np.array([0.13187179, 0.10619377, 0.29945481]),
                        Lab=np.array([38.93186373, 22.05038057, -35.82206456]),
                        C=42.064707208617435,
                    ),
                    VS_ColorimetryData(
                        name="VS3",
                        XYZ=np.array([0.10123263, 0.09853741, 0.32956604]),
                        Lab=np.array([37.57864252, 6.04766746, -42.39139508]),
                        C=42.82061020516246,
                    ),
                    VS_ColorimetryData(
                        name="VS4",
                        XYZ=np.array([0.13144454, 0.16803553, 0.39315864]),
                        Lab=np.array(
                            [48.01155296, -17.36604069, -32.56734417]
                        ),
                        C=36.908146466038922,
                    ),
                    VS_ColorimetryData(
                        name="VS5",
                        XYZ=np.array([0.18145723, 0.25845953, 0.41319313]),
                        Lab=np.array(
                            [57.89053983, -30.61152779, -17.92233237]
                        ),
                        C=35.472181086008781,
                    ),
                    VS_ColorimetryData(
                        name="VS6",
                        XYZ=np.array([0.15184114, 0.25076481, 0.28160235]),
                        Lab=np.array([57.14986362, -44.01984887, -1.76443512]),
                        C=44.055196354398468,
                    ),
                    VS_ColorimetryData(
                        name="VS7",
                        XYZ=np.array([0.13956282, 0.25328776, 0.14470413]),
                        Lab=np.array([57.39436644, -52.59240053, 24.11037488]),
                        C=57.8556027598421,
                    ),
                    VS_ColorimetryData(
                        name="VS8",
                        XYZ=np.array([0.24672357, 0.36210726, 0.11976641]),
                        Lab=np.array([66.68062145, -37.45331629, 46.38001890]),
                        C=59.614235332660144,
                    ),
                    VS_ColorimetryData(
                        name="VS9",
                        XYZ=np.array([0.40820163, 0.50861708, 0.11894288]),
                        Lab=np.array([76.59516365, -21.90847578, 63.69499819]),
                        C=67.357509638238682,
                    ),
                    VS_ColorimetryData(
                        name="VS10",
                        XYZ=np.array([0.56036726, 0.60569219, 0.10169199]),
                        Lab=np.array([82.14661230, -3.82032735, 78.14954550]),
                        C=78.24286780890229,
                    ),
                    VS_ColorimetryData(
                        name="VS11",
                        XYZ=np.array([0.40540651, 0.38003446, 0.05753983]),
                        Lab=np.array([68.02315451, 14.17690044, 69.54409225]),
                        C=70.974398718322206,
                    ),
                    VS_ColorimetryData(
                        name="VS12",
                        XYZ=np.array([0.45407809, 0.37920609, 0.08621297]),
                        Lab=np.array([67.96206127, 28.93563884, 58.57062794]),
                        C=65.328321974859804,
                    ),
                    VS_ColorimetryData(
                        name="VS13",
                        XYZ=np.array([0.37207030, 0.27413935, 0.08882217]),
                        Lab=np.array([59.35552778, 40.92542311, 42.87088737]),
                        C=59.268906188196901,
                    ),
                    VS_ColorimetryData(
                        name="VS14",
                        XYZ=np.array([0.19307398, 0.11049957, 0.04883445]),
                        Lab=np.array([39.66448332, 53.96576475, 24.65796798]),
                        C=59.332277465127156,
                    ),
                    VS_ColorimetryData(
                        name="VS15",
                        XYZ=np.array([0.17306027, 0.10700056, 0.11280793]),
                        Lab=np.array([39.07062485, 45.99788526, 0.67641984]),
                        C=46.002858517472262,
                    ),
                ],
            ),
        )

        specification_t = colour_quality_scale(
            SDS_ILLUMINANTS["FL1"], additional_data=True, method="NIST CQS 7.4"
        )

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

        specification_r = ColourRendering_Specification_CQS(
            name="FL1",
            Q_a=74.982585798279871,
            Q_f=75.945236961962692,
            Q_p=None,
            Q_g=83.880315530074398,
            Q_d=None,
            Q_as={
                1: VS_ColourQualityScaleData(
                    name="VS1",
                    Q_a=51.966399639139652,
                    D_C_ab=-14.617735662894113,
                    D_E_ab=15.027845447317082,
                    D_Ep_ab=15.027845447317082,
                ),
                2: VS_ColourQualityScaleData(
                    name="VS2",
                    Q_a=70.949015257799516,
                    D_C_ab=-5.742631494382664,
                    D_E_ab=9.0810254398122403,
                    D_Ep_ab=9.0810254398122403,
                ),
                3: VS_ColourQualityScaleData(
                    name="VS3",
                    Q_a=74.662762947552991,
                    D_C_ab=-0.18830304572394141,
                    D_E_ab=7.9196747607137059,
                    D_Ep_ab=7.9196747607137059,
                ),
                4: VS_ColourQualityScaleData(
                    name="VS4",
                    Q_a=85.532024779429392,
                    D_C_ab=2.6056579409845568,
                    D_E_ab=5.2188636120497893,
                    D_Ep_ab=4.5218452091774983,
                ),
                5: VS_ColourQualityScaleData(
                    name="VS5",
                    Q_a=95.462563991193164,
                    D_C_ab=3.2171678090164875,
                    D_E_ab=3.5158755402235844,
                    D_Ep_ab=1.4181720992074747,
                ),
                6: VS_ColourQualityScaleData(
                    name="VS6",
                    Q_a=84.55525422632698,
                    D_C_ab=-0.69329228720708613,
                    D_E_ab=4.8271478805630519,
                    D_Ep_ab=4.8271478805630519,
                ),
                7: VS_ColourQualityScaleData(
                    name="VS7",
                    Q_a=74.112296060906786,
                    D_C_ab=-6.7678218514458237,
                    D_E_ab=8.0917968864976295,
                    D_Ep_ab=8.0917968864976295,
                ),
                8: VS_ColourQualityScaleData(
                    name="VS8",
                    Q_a=74.948915115579581,
                    D_C_ab=-7.7777262876453435,
                    D_E_ab=7.8302017501424785,
                    D_Ep_ab=7.8302017501424785,
                ),
                9: VS_ColourQualityScaleData(
                    name="VS9",
                    Q_a=75.868964954677168,
                    D_C_ab=-5.9501389861388283,
                    D_E_ab=7.5425333918134534,
                    D_Ep_ab=7.5425333918134534,
                ),
                10: VS_ColourQualityScaleData(
                    name="VS10",
                    Q_a=75.254253773456256,
                    D_C_ab=-1.540504353515054,
                    D_E_ab=7.7347311479094465,
                    D_Ep_ab=7.7347311479094465,
                ),
                11: VS_ColourQualityScaleData(
                    name="VS11",
                    Q_a=78.159233353416667,
                    D_C_ab=1.7422491866669034,
                    D_E_ab=7.0453200659888839,
                    D_Ep_ab=6.8265000259125559,
                ),
                12: VS_ColourQualityScaleData(
                    name="VS12",
                    Q_a=92.007574903997778,
                    D_C_ab=2.8431539673587451,
                    D_E_ab=3.7846096930430031,
                    D_Ep_ab=2.4979483674742524,
                ),
                13: VS_ColourQualityScaleData(
                    name="VS13",
                    Q_a=81.483789597909151,
                    D_C_ab=-0.0084301917293103656,
                    D_E_ab=5.7872196432106016,
                    D_Ep_ab=5.7872196432106016,
                ),
                14: VS_ColourQualityScaleData(
                    name="VS14",
                    Q_a=76.536988871102693,
                    D_C_ab=-5.0697802563888956,
                    D_E_ab=7.3336734748351624,
                    D_Ep_ab=7.3336734748351624,
                ),
                15: VS_ColourQualityScaleData(
                    name="VS15",
                    Q_a=65.429713405731931,
                    D_C_ab=-9.4962363392015661,
                    D_E_ab=10.807718438492838,
                    D_Ep_ab=10.807718438492838,
                ),
            },
            colorimetry_data=(
                [
                    VS_ColorimetryData(
                        name="VS1",
                        XYZ=np.array([0.14271715, 0.09107438, 0.04949462]),
                        Lab=np.array([36.19033159, 40.77666015, 18.34813122]),
                        C=44.714538268056295,
                    ),
                    VS_ColorimetryData(
                        name="VS2",
                        XYZ=np.array([0.34657727, 0.27547744, 0.08900676]),
                        Lab=np.array([59.47793328, 31.84647553, 43.02166698]),
                        C=53.526272364542265,
                    ),
                    VS_ColorimetryData(
                        name="VS3",
                        XYZ=np.array([0.41050103, 0.38688447, 0.06613576]),
                        Lab=np.array([68.52498199, 13.58373774, 66.83110597]),
                        C=68.197614738277153,
                    ),
                    VS_ColorimetryData(
                        name="VS4",
                        XYZ=np.array([0.45427132, 0.50401485, 0.08193287]),
                        Lab=np.array([76.31503360, -7.01311937, 74.42292542]),
                        C=74.752629859451574,
                    ),
                    VS_ColorimetryData(
                        name="VS5",
                        XYZ=np.array([0.42410903, 0.52851922, 0.11439811]),
                        Lab=np.array([77.78749106, -22.21024992, 66.98873490]),
                        C=70.574682470722237,
                    ),
                    VS_ColorimetryData(
                        name="VS6",
                        XYZ=np.array([0.26141761, 0.36817692, 0.11429088]),
                        Lab=np.array([67.14003019, -33.22377278, 48.66064673]),
                        C=58.920943801328043,
                    ),
                    VS_ColorimetryData(
                        name="VS7",
                        XYZ=np.array([0.14991085, 0.24929718, 0.13823960]),
                        Lab=np.array([57.00687837, -44.55800078, 24.99093412]),
                        C=51.087789364109447,
                    ),
                    VS_ColorimetryData(
                        name="VS8",
                        XYZ=np.array([0.15843361, 0.24157338, 0.26933192]),
                        Lab=np.array([56.24498073, -36.24891645, -1.43945699]),
                        C=36.277485856650493,
                    ),
                    VS_ColorimetryData(
                        name="VS9",
                        XYZ=np.array([0.18662998, 0.24708620, 0.40043672]),
                        Lab=np.array(
                            [56.79040828, -23.15964795, -18.30797717]
                        ),
                        C=29.522048055967336,
                    ),
                    VS_ColorimetryData(
                        name="VS10",
                        XYZ=np.array([0.13339389, 0.15696027, 0.38421709]),
                        Lab=np.array([46.57313267, -9.89412218, -33.95549821]),
                        C=35.367633681665495,
                    ),
                    VS_ColorimetryData(
                        name="VS11",
                        XYZ=np.array([0.09900743, 0.09954465, 0.32039098]),
                        Lab=np.array([37.76058147, 3.51413565, -40.81527590]),
                        C=40.966277550944625,
                    ),
                    VS_ColorimetryData(
                        name="VS12",
                        XYZ=np.array([0.11576390, 0.09613722, 0.31928926]),
                        Lab=np.array([37.14003664, 18.77460935, -41.73197608]),
                        C=45.760723157938472,
                    ),
                    VS_ColorimetryData(
                        name="VS13",
                        XYZ=np.array([0.20975356, 0.16847879, 0.37267453]),
                        Lab=np.array([48.06778877, 25.97523691, -29.94366223]),
                        C=39.640078711661452,
                    ),
                    VS_ColorimetryData(
                        name="VS14",
                        XYZ=np.array([0.32298108, 0.24163045, 0.36212750]),
                        Lab=np.array([56.25066973, 37.45976513, -14.49801776]),
                        C=40.167480906459893,
                    ),
                    VS_ColorimetryData(
                        name="VS15",
                        XYZ=np.array([0.22039693, 0.15371392, 0.17553541]),
                        Lab=np.array([46.13873255, 39.31630210, -2.10769974]),
                        C=39.372757193029329,
                    ),
                ],
                [
                    VS_ColorimetryData(
                        name="VS1",
                        XYZ=np.array([0.19307399, 0.11049957, 0.04883449]),
                        Lab=np.array([39.66448335, 53.96576813, 24.65795205]),
                        C=59.332273930950407,
                    ),
                    VS_ColorimetryData(
                        name="VS2",
                        XYZ=np.array([0.37207030, 0.27413935, 0.08882218]),
                        Lab=np.array([59.35552779, 40.92542394, 42.87088336]),
                        C=59.268903858924929,
                    ),
                    VS_ColorimetryData(
                        name="VS3",
                        XYZ=np.array([0.42080177, 0.37272049, 0.06618662]),
                        Lab=np.array([67.48063529, 21.22017785, 65.01028998]),
                        C=68.385917784001094,
                    ),
                    VS_ColorimetryData(
                        name="VS4",
                        XYZ=np.array([0.46201298, 0.49481812, 0.08588402]),
                        Lab=np.array([75.75009449, -2.36998858, 72.10803500]),
                        C=72.146971918467017,
                    ),
                    VS_ColorimetryData(
                        name="VS5",
                        XYZ=np.array([0.40820163, 0.50861708, 0.11894286]),
                        Lab=np.array([76.59516364, -21.90847694, 63.69500310]),
                        C=67.35751466170575,
                    ),
                    VS_ColorimetryData(
                        name="VS6",
                        XYZ=np.array([0.24672357, 0.36210726, 0.11976641]),
                        Lab=np.array([66.68062144, -37.45331655, 46.38001966]),
                        C=59.614236088535129,
                    ),
                    VS_ColorimetryData(
                        name="VS7",
                        XYZ=np.array([0.13956281, 0.25328776, 0.14470409]),
                        Lab=np.array([57.39436642, -52.59240564, 24.11038403]),
                        C=57.855611215555271,
                    ),
                    VS_ColorimetryData(
                        name="VS8",
                        XYZ=np.array([0.15184111, 0.25076481, 0.28160222]),
                        Lab=np.array([57.14986354, -44.01986548, -1.76441495]),
                        C=44.055212144295837,
                    ),
                    VS_ColorimetryData(
                        name="VS9",
                        XYZ=np.array([0.18145720, 0.25845953, 0.41319296]),
                        Lab=np.array(
                            [57.89053974, -30.61154597, -17.92231311]
                        ),
                        C=35.472187042106164,
                    ),
                    VS_ColorimetryData(
                        name="VS10",
                        XYZ=np.array([0.13144449, 0.16803553, 0.39315843]),
                        Lab=np.array(
                            [48.01155280, -17.36606803, -32.56732004]
                        ),
                        C=36.908138035180549,
                    ),
                    VS_ColorimetryData(
                        name="VS11",
                        XYZ=np.array([0.09725029, 0.10655822, 0.32331756]),
                        Lab=np.array([38.99463289, -3.20501320, -39.09286753]),
                        C=39.224028364277721,
                    ),
                    VS_ColorimetryData(
                        name="VS12",
                        XYZ=np.array([0.11497971, 0.09965866, 0.31509326]),
                        Lab=np.array([37.78109906, 15.45054732, -40.03995920]),
                        C=42.917569190579727,
                    ),
                    VS_ColorimetryData(
                        name="VS13",
                        XYZ=np.array([0.23125767, 0.17972670, 0.36038776]),
                        Lab=np.array([49.46294201, 29.95248104, -25.97793559]),
                        C=39.648508903390763,
                    ),
                    VS_ColorimetryData(
                        name="VS14",
                        XYZ=np.array([0.35887695, 0.25609884, 0.35518732]),
                        Lab=np.array([57.66488716, 43.83765559, -11.16556087]),
                        C=45.237261162848789,
                    ),
                    VS_ColorimetryData(
                        name="VS15",
                        XYZ=np.array([0.26552457, 0.17192965, 0.17300682]),
                        Lab=np.array([48.50225789, 48.80528996, 2.49443403]),
                        C=48.868993532230895,
                    ),
                ],
            ),
        )

        specification_t = colour_quality_scale(
            SDS_ILLUMINANTS["FL1"], additional_data=True, method="NIST CQS 9.0"
        )

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


if __name__ == "__main__":
    unittest.main()
