"""Defines the unit tests for the :mod:`colour.quality.cri` module."""

from __future__ import annotations

import numpy as np
import unittest

from colour.colorimetry import SDS_ILLUMINANTS, SpectralDistribution
from colour.hints import Dict
from colour.quality import (
    ColourRendering_Specification_CRI,
    colour_rendering_index,
)
from colour.quality.cri import TCS_ColorimetryData, TCS_ColourQualityScaleData

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestColourRenderingIndex",
]

DATA_SAMPLE: Dict = {
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
    780: 0.00293663,
}


class TestColourRenderingIndex(unittest.TestCase):
    """
    Define :func:`colour.quality.cri.colour_rendering_index`
    definition unit tests methods.
    """

    def test_colour_rendering_index(self):
        """Test :func:`colour.quality.cri.colour_rendering_index` definition."""

        self.assertAlmostEqual(
            colour_rendering_index(SDS_ILLUMINANTS["FL1"]),
            75.852827992149358,
            places=7,
        )

        self.assertAlmostEqual(
            colour_rendering_index(SDS_ILLUMINANTS["FL2"]),
            64.233724121664778,
            places=7,
        )

        self.assertAlmostEqual(
            colour_rendering_index(SDS_ILLUMINANTS["A"]),
            99.996230290506887,
            places=7,
        )

        self.assertAlmostEqual(
            colour_rendering_index(SpectralDistribution(DATA_SAMPLE)),
            70.815265381660197,
            places=7,
        )

        specification_r = ColourRendering_Specification_CRI(
            name="FL1",
            Q_a=75.852827992149358,
            Q_as={
                1: TCS_ColourQualityScaleData(
                    name="TCS01", Q_a=69.196992839933557
                ),
                2: TCS_ColourQualityScaleData(
                    name="TCS02", Q_a=83.650754065677816
                ),
                3: TCS_ColourQualityScaleData(
                    name="TCS03", Q_a=92.136090764490675
                ),
                4: TCS_ColourQualityScaleData(
                    name="TCS04", Q_a=72.665649420972784
                ),
                5: TCS_ColourQualityScaleData(
                    name="TCS05", Q_a=73.890734513517486
                ),
                6: TCS_ColourQualityScaleData(
                    name="TCS06", Q_a=79.619188860052745
                ),
                7: TCS_ColourQualityScaleData(
                    name="TCS07", Q_a=82.272569853644669
                ),
                8: TCS_ColourQualityScaleData(
                    name="TCS08", Q_a=53.390643618905109
                ),
                9: TCS_ColourQualityScaleData(
                    name="TCS09", Q_a=-47.284477750225903
                ),
                10: TCS_ColourQualityScaleData(
                    name="TCS10", Q_a=61.568336431199967
                ),
                11: TCS_ColourQualityScaleData(
                    name="TCS11", Q_a=67.522241168172485
                ),
                12: TCS_ColourQualityScaleData(
                    name="TCS12", Q_a=74.890093866757994
                ),
                13: TCS_ColourQualityScaleData(
                    name="TCS13", Q_a=72.771930354944615
                ),
                14: TCS_ColourQualityScaleData(
                    name="TCS14", Q_a=94.884867470552663
                ),
            },
            colorimetry_data=(
                [
                    TCS_ColorimetryData(
                        name="TCS01",
                        XYZ=np.array([31.19561134, 29.74560797, 23.44190201]),
                        uv=np.array([0.22782766, 0.32585700]),
                        UVW=np.array([25.43090825, 8.72673714, 60.46061819]),
                    ),
                    TCS_ColorimetryData(
                        name="TCS02",
                        XYZ=np.array([26.70829694, 29.25797165, 13.98804447]),
                        uv=np.array([0.21049132, 0.34587843]),
                        UVW=np.array([11.90704877, 24.68727835, 60.03499882]),
                    ),
                    TCS_ColorimetryData(
                        name="TCS03",
                        XYZ=np.array([24.20150315, 31.45341032, 9.19170689]),
                        uv=np.array([0.18489328, 0.36044399]),
                        UVW=np.array([-8.11343024, 37.47469142, 61.91555021]),
                    ),
                    TCS_ColorimetryData(
                        name="TCS04",
                        XYZ=np.array([20.74577359, 29.44046997, 19.40749007]),
                        uv=np.array([0.15940652, 0.33932233]),
                        UVW=np.array([-27.55686536, 19.30727375, 60.19483706]),
                    ),
                    TCS_ColorimetryData(
                        name="TCS05",
                        XYZ=np.array([25.09405566, 30.60912259, 37.93634878]),
                        uv=np.array([0.16784200, 0.30709443]),
                        UVW=np.array([-21.30541564, -6.63859760, 61.20303996]),
                    ),
                    TCS_ColorimetryData(
                        name="TCS06",
                        XYZ=np.array([27.43598853, 28.84467787, 55.15209308]),
                        uv=np.array([0.17543246, 0.27665994]),
                        UVW=np.array(
                            [-14.91500194, -30.51166823, 59.67054901]
                        ),
                    ),
                    TCS_ColorimetryData(
                        name="TCS07",
                        XYZ=np.array([31.30354023, 28.49931283, 51.55875721]),
                        uv=np.array([0.20410821, 0.27873574]),
                        UVW=np.array([6.88826195, -28.65430811, 59.36332055]),
                    ),
                    TCS_ColorimetryData(
                        name="TCS08",
                        XYZ=np.array([33.81156122, 29.92921717, 44.07459562]),
                        uv=np.array([0.21992203, 0.29200489]),
                        UVW=np.array([19.29699368, -18.57115711, 60.61967045]),
                    ),
                    TCS_ColorimetryData(
                        name="TCS09",
                        XYZ=np.array([14.75210654, 9.07663825, 4.24056478]),
                        uv=np.array([0.36063567, 0.33283649]),
                        UVW=np.array([74.85972274, 8.59043120, 35.14928413]),
                    ),
                    TCS_ColorimetryData(
                        name="TCS10",
                        XYZ=np.array([53.37923227, 60.57123196, 10.90035400]),
                        uv=np.array([0.21466565, 0.36538264]),
                        UVW=np.array([20.48612095, 54.66177264, 81.18130740]),
                    ),
                    TCS_ColorimetryData(
                        name="TCS11",
                        XYZ=np.array([12.25313424, 19.77564573, 14.04738059]),
                        uv=np.array([0.13962494, 0.33801637]),
                        UVW=np.array([-36.00690822, 15.29571595, 50.60573931]),
                    ),
                    TCS_ColorimetryData(
                        name="TCS12",
                        XYZ=np.array([5.77964943, 5.69096075, 24.73530409]),
                        uv=np.array([0.13981616, 0.20650602]),
                        UVW=np.array(
                            [-19.30689974, -39.58762581, 27.63428055]
                        ),
                    ),
                    TCS_ColorimetryData(
                        name="TCS13",
                        XYZ=np.array([56.62340157, 57.50304691, 39.13768236]),
                        uv=np.array([0.21850039, 0.33284220]),
                        UVW=np.array([23.93135946, 18.85365757, 79.49473722]),
                    ),
                    TCS_ColorimetryData(
                        name="TCS14",
                        XYZ=np.array([9.42270977, 12.06929274, 5.06066928]),
                        uv=np.array([0.18328188, 0.35214117]),
                        UVW=np.array([-6.11563143, 19.91896684, 40.34566797]),
                    ),
                ],
                [
                    TCS_ColorimetryData(
                        name="TCS01",
                        XYZ=np.array([33.04774537, 29.80902109, 24.23929188]),
                        uv=np.array([0.23908620, 0.32348313]),
                        UVW=np.array([32.11891091, 8.39794012, 60.51562388]),
                    ),
                    TCS_ColorimetryData(
                        name="TCS02",
                        XYZ=np.array([27.53677515, 28.90913487, 14.75211494]),
                        uv=np.array([0.21792745, 0.34318256]),
                        UVW=np.array([15.27177183, 23.58438306, 59.72761646]),
                    ),
                    TCS_ColorimetryData(
                        name="TCS03",
                        XYZ=np.array([23.95706051, 30.44569936, 9.79977770]),
                        uv=np.array([0.18788308, 0.35815528]),
                        UVW=np.array([-8.23665275, 35.99767796, 61.06361523]),
                    ),
                    TCS_ColorimetryData(
                        name="TCS04",
                        XYZ=np.array([20.43647757, 29.46748627, 21.03631859]),
                        uv=np.array([0.15554126, 0.33641389]),
                        UVW=np.array([-33.44111710, 18.47940853, 60.21844268]),
                    ),
                    TCS_ColorimetryData(
                        name="TCS05",
                        XYZ=np.array([24.95511680, 30.81953457, 39.91407614]),
                        uv=np.array([0.16445149, 0.30464603]),
                        UVW=np.array([-26.97713986, -6.51317420, 61.38182431]),
                    ),
                    TCS_ColorimetryData(
                        name="TCS06",
                        XYZ=np.array([28.14601546, 29.75632189, 57.16886797]),
                        uv=np.array([0.17427942, 0.27637560]),
                        UVW=np.array(
                            [-18.85053083, -28.64005452, 60.46991712]
                        ),
                    ),
                    TCS_ColorimetryData(
                        name="TCS07",
                        XYZ=np.array([33.29720901, 29.36938555, 52.53803430]),
                        uv=np.array([0.21092469, 0.27906521]),
                        UVW=np.array([9.90110830, -26.37778265, 60.13265766]),
                    ),
                    TCS_ColorimetryData(
                        name="TCS08",
                        XYZ=np.array([37.64974363, 31.35401774, 44.84930911]),
                        uv=np.array([0.23439240, 0.29279655]),
                        UVW=np.array([29.04479043, -16.08583648, 61.83233828]),
                    ),
                    TCS_ColorimetryData(
                        name="TCS09",
                        XYZ=np.array([20.69443384, 11.28822434, 4.28723010]),
                        uv=np.array([0.40801431, 0.33384028]),
                        UVW=np.array([106.56664825, 10.68535426, 39.08093160]),
                    ),
                    TCS_ColorimetryData(
                        name="TCS10",
                        XYZ=np.array([55.04099876, 59.04719161, 11.86354410]),
                        uv=np.array([0.22549942, 0.36286881]),
                        UVW=np.array([28.45432459, 52.29127793, 80.35085218]),
                    ),
                    TCS_ColorimetryData(
                        name="TCS11",
                        XYZ=np.array([12.13069359, 20.35272336, 15.17132466]),
                        uv=np.array([0.13369530, 0.33646842]),
                        UVW=np.array([-43.02145539, 15.76573781, 51.25705062]),
                    ),
                    TCS_ColorimetryData(
                        name="TCS12",
                        XYZ=np.array([6.18799870, 6.41132549, 27.28158667]),
                        uv=np.array([0.13437372, 0.20883497]),
                        UVW=np.array(
                            [-24.45285903, -39.79705961, 29.44325151]
                        ),
                    ),
                    TCS_ColorimetryData(
                        name="TCS13",
                        XYZ=np.array([58.97935503, 57.14777525, 40.83450223]),
                        uv=np.array([0.22712769, 0.33011150]),
                        UVW=np.array([29.75912462, 17.83690656, 79.29560174]),
                    ),
                    TCS_ColorimetryData(
                        name="TCS14",
                        XYZ=np.array([9.34469915, 11.70922060, 5.33442353]),
                        uv=np.array([0.18597686, 0.34955284]),
                        UVW=np.array([-6.34991066, 18.99712303, 39.76962229]),
                    ),
                ],
            ),
        )

        specification_t = colour_rendering_index(
            SDS_ILLUMINANTS["FL1"], additional_data=True
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
