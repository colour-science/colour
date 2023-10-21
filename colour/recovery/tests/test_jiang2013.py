# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.recovery.jiang2013` module."""

import unittest

import numpy as np

from colour.characterisation import (
    MSDS_CAMERA_SENSITIVITIES,
    SDS_COLOURCHECKERS,
)
from colour.colorimetry import (
    SDS_ILLUMINANTS,
    SpectralDistribution,
    SpectralShape,
    msds_to_XYZ,
    reshape_msds,
    reshape_sd,
    sds_and_msds_to_msds,
)
from colour.hints import cast
from colour.recovery import (
    BASIS_FUNCTIONS_DYER2017,
    SPECTRAL_SHAPE_BASIS_FUNCTIONS_DYER2017,
    PCA_Jiang2013,
    RGB_to_msds_camera_sensitivities_Jiang2013,
    RGB_to_sd_camera_sensitivity_Jiang2013,
)
from colour.utilities import tsplit

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestPCA_Jiang2013",
    "TestMixinJiang2013",
    "TestRGB_to_sd_camera_sensitivity_Jiang2013",
    "TestRGB_to_msds_camera_sensitivities_Jiang2013",
]


class TestPCA_Jiang2013(unittest.TestCase):
    """
    Define :func:`colour.recovery.jiang2013.PCA_Jiang2013` definition unit
    tests methods.
    """

    def test_PCA_Jiang2013(self):
        """Test :func:`colour.recovery.jiang2013.PCA_Jiang2013` definition."""

        shape = SpectralShape(400, 700, 10)
        camera_sensitivities = {
            camera: msds.copy().align(shape)
            for camera, msds in MSDS_CAMERA_SENSITIVITIES.items()
        }
        w, v = PCA_Jiang2013(camera_sensitivities, 3, True)

        # TODO: Last eigen value seems to be very sensitive and produce
        # differences on ARM.
        np.testing.assert_array_almost_equal(
            np.array(w)[..., 0:2],
            np.array(
                [
                    [
                        [-0.00137594, -0.00399416, -0.00060424],
                        [-0.00214835, -0.00184422, 0.32158985],
                        [-0.02757181, -0.00553587, 0.05403898],
                        [-0.02510621, 0.04216468, -0.02702351],
                        [-0.02011623, 0.03371162, -0.00568937],
                        [-0.01392282, 0.03297985, -0.01952216],
                        [-0.00944513, 0.03300938, -0.03045553],
                        [-0.02019958, 0.01289400, 0.02225622],
                        [-0.02394423, 0.00980934, -0.01011474],
                        [-0.04196326, -0.04987050, -0.00857578],
                        [-0.04988732, -0.06936603, 0.20547221],
                        [-0.06527141, -0.09378614, 0.03734956],
                        [-0.09412575, -0.12244081, 0.14001439],
                        [-0.10915913, -0.13119983, 0.08831632],
                        [-0.12314840, -0.24280936, 0.20130990],
                        [-0.11673941, -0.27700737, 0.18667656],
                        [-0.12534133, -0.29994127, 0.21783786],
                        [-0.14599255, -0.25586532, 0.22713199],
                        [-0.25249090, 0.11499750, -0.22053343],
                        [-0.35163407, 0.45286818, -0.13965255],
                        [-0.35805737, 0.40724252, 0.25882968],
                        [-0.36927899, 0.18120838, 0.51996857],
                        [-0.35374885, 0.03010008, -0.25565598],
                        [-0.35340909, -0.16847527, -0.28410654],
                        [-0.32696116, -0.29068981, -0.16331414],
                        [-0.29067354, -0.32862702, -0.24973353],
                        [-0.08964758, 0.09682656, -0.06974404],
                        [-0.01891664, 0.08221113, -0.00242073],
                        [-0.00521149, 0.01578907, -0.00004335],
                        [-0.00232366, 0.00137751, 0.00040639],
                        [-0.00153787, -0.00254398, -0.00038028],
                    ],
                    [
                        [-0.00119598, -0.00267792, -0.00047163],
                        [-0.00200327, -0.00322983, 0.58674915],
                        [-0.01247816, 0.03313976, -0.03562970],
                        [-0.03207685, 0.05703294, -0.00969283],
                        [-0.04715050, 0.05296451, 0.07669022],
                        [-0.05794010, 0.05455737, -0.00031457],
                        [-0.10745571, -0.00158911, -0.07053271],
                        [-0.14178525, 0.03362764, -0.10570131],
                        [-0.16811402, 0.05569833, -0.12315365],
                        [-0.18463716, 0.04615404, 0.20069739],
                        [-0.21531623, 0.09745078, 0.18037692],
                        [-0.25442570, 0.18330481, -0.13661865],
                        [-0.28168018, 0.25193267, -0.07739509],
                        [-0.29237178, 0.28545428, 0.01460388],
                        [-0.29693117, 0.23909467, -0.24182353],
                        [-0.28631319, 0.19476441, -0.05441354],
                        [-0.27195968, 0.12087420, 0.40655125],
                        [-0.25988140, 0.01581316, 0.32470450],
                        [-0.24222660, -0.07912972, -0.27085507],
                        [-0.23069698, -0.18583667, -0.02532450],
                        [-0.20831983, -0.26745561, -0.23243075],
                        [-0.19437168, -0.32425009, 0.10237571],
                        [-0.18470894, -0.34768079, -0.01854361],
                        [-0.18056180, -0.35983221, -0.06301260],
                        [-0.17141337, -0.35067306, 0.19769116],
                        [-0.14712541, -0.30423172, -0.07278775],
                        [-0.02897026, -0.04573993, -0.01107455],
                        [-0.00190228, 0.00461591, 0.00033462],
                        [-0.00069122, 0.00118817, -0.00067360],
                        [-0.00045559, -0.00015286, -0.00003903],
                        [-0.00039509, -0.00049719, -0.00031217],
                    ],
                    [
                        [-0.03283371, -0.04707162, 0.99800447],
                        [-0.05932690, -0.07529740, -0.02052607],
                        [-0.11947381, 0.07977219, 0.00630795],
                        [-0.18492233, 0.26127374, 0.00520247],
                        [-0.22091564, 0.29279976, 0.01193361],
                        [-0.25377875, 0.30677709, -0.00125936],
                        [-0.29969822, 0.26541777, 0.00701600],
                        [-0.30232755, 0.25378622, 0.00603327],
                        [-0.30031732, 0.19751184, -0.00689292],
                        [-0.28072276, 0.11804285, -0.00059888],
                        [-0.26005747, 0.01836333, -0.01245795],
                        [-0.23839367, -0.07182421, -0.01335875],
                        [-0.21721831, -0.14245410, -0.01256502],
                        [-0.19828405, -0.17684950, -0.02162238],
                        [-0.19018451, -0.20137781, -0.01456925],
                        [-0.18196762, -0.22086321, -0.00980506],
                        [-0.17168644, -0.22771873, -0.01723250],
                        [-0.16977073, -0.23504018, -0.02204447],
                        [-0.16277670, -0.22897797, -0.01554531],
                        [-0.15880423, -0.22583675, -0.02017303],
                        [-0.14966812, -0.21494312, -0.00986008],
                        [-0.13480155, -0.19511162, -0.01155616],
                        [-0.12541764, -0.18113238, -0.00597802],
                        [-0.12355731, -0.17835150, -0.01026176],
                        [-0.11175064, -0.15997651, -0.00814162],
                        [-0.09440304, -0.13423453, -0.00911544],
                        [-0.01670581, -0.02019670, -0.00109904],
                        [-0.00045002, 0.00147362, 0.00006449],
                        [-0.00102919, -0.00095904, -0.00008201],
                        [-0.00097397, -0.00123434, -0.00006730],
                        [-0.00097116, -0.00124835, -0.00008008],
                    ],
                ]
            )[..., 0:2],
            decimal=7,
        )
        np.testing.assert_array_almost_equal(
            np.array(v)[..., 0:2],
            np.array(
                [
                    [1.05516066e01, 7.28373797e-01, 3.84530151e-16],
                    [2.00917798e01, 1.57662524e00, 1.65371431e-15],
                    [1.90414282e01, 2.60426480e00, 5.11235833e-15],
                ]
            )[..., 0:2],
            decimal=7,
        )


class TestMixinJiang2013:
    """A mixin for testing the :mod:`colour.recovery.jiang2013` module."""

    def __init__(self) -> None:
        """Initialise common tests attributes for the mixin."""

        self._sensitivities = reshape_msds(
            MSDS_CAMERA_SENSITIVITIES["Nikon 5100 (NPL)"],
            SPECTRAL_SHAPE_BASIS_FUNCTIONS_DYER2017,
        )
        self._sd_D65 = reshape_sd(
            SDS_ILLUMINANTS["D65"], SPECTRAL_SHAPE_BASIS_FUNCTIONS_DYER2017
        )

        reflectances = list(SDS_COLOURCHECKERS["BabelColor Average"].values())
        self._reflectances = sds_and_msds_to_msds(reflectances)
        self._RGB = msds_to_XYZ(
            cast(SpectralDistribution, self._reflectances.copy()).align(
                SPECTRAL_SHAPE_BASIS_FUNCTIONS_DYER2017
            ),
            method="Integration",
            cmfs=self._sensitivities,
            illuminant=self._sd_D65,
            k=1,
            shape=SPECTRAL_SHAPE_BASIS_FUNCTIONS_DYER2017,
        )


class TestRGB_to_sd_camera_sensitivity_Jiang2013(
    unittest.TestCase, TestMixinJiang2013
):
    """
    Define :func:`colour.recovery.jiang2013.RGB_to_sd_camera_sensitivity_Jiang2013`
    definition unit tests methods.
    """

    def setUp(self):
        """Initialise the common tests attributes."""

        TestMixinJiang2013.__init__(self)

    def test_RGB_to_sd_camera_sensitivity_Jiang2013(self):
        """
        Test :func:`colour.recovery.jiang2013.\
RGB_to_sd_camera_sensitivity_Jiang2013` definition.
        """

        R_w, _G_w, _B_w = tsplit(np.moveaxis(BASIS_FUNCTIONS_DYER2017, 0, 1))

        np.testing.assert_array_almost_equal(
            RGB_to_sd_camera_sensitivity_Jiang2013(
                self._RGB[..., 0],
                self._sd_D65,
                self._reflectances,
                R_w,
                SPECTRAL_SHAPE_BASIS_FUNCTIONS_DYER2017,
            ).values,
            np.array(
                [
                    0.00072067,
                    -0.00089699,
                    0.0046872,
                    0.0077695,
                    0.00693355,
                    0.00531349,
                    0.004482,
                    0.00463938,
                    0.00518667,
                    0.00438283,
                    0.00420012,
                    0.00540655,
                    0.00964451,
                    0.01427711,
                    0.00799507,
                    0.00464298,
                    0.00534238,
                    0.01051938,
                    0.05288944,
                    0.09785117,
                    0.09960038,
                    0.08384089,
                    0.06918086,
                    0.05696785,
                    0.04293031,
                    0.03024127,
                    0.02323005,
                    0.01372194,
                    0.00409449,
                    -0.00044223,
                    -0.00061428,
                ]
            ),
            decimal=7,
        )


class TestRGB_to_msds_camera_sensitivities_Jiang2013(
    unittest.TestCase, TestMixinJiang2013
):
    """
    Define :func:`colour.recovery.jiang2013.\
RGB_to_msds_camera_sensitivities_Jiang2013` definition unit tests methods.
    """

    def setUp(self):
        """Initialise the common tests attributes."""

        TestMixinJiang2013.__init__(self)

    def test_RGB_to_msds_camera_sensitivities_Jiang2013(self):
        """
        Test :func:`colour.recovery.jiang2013.\
RGB_to_msds_camera_sensitivities_Jiang2013` definition.
        """

        np.testing.assert_array_almost_equal(
            RGB_to_msds_camera_sensitivities_Jiang2013(
                self._RGB,
                self._sd_D65,
                self._reflectances,
                BASIS_FUNCTIONS_DYER2017,
                SPECTRAL_SHAPE_BASIS_FUNCTIONS_DYER2017,
            ).values,
            np.array(
                [
                    [7.04378461e-03, 9.21260449e-03, -7.64080878e-03],
                    [-8.76715607e-03, 1.12726694e-02, 6.37434190e-03],
                    [4.58126856e-02, 7.18000418e-02, 4.00001696e-01],
                    [7.59391152e-02, 1.15620933e-01, 7.11521550e-01],
                    [6.77685732e-02, 1.53406449e-01, 8.52668310e-01],
                    [5.19341313e-02, 1.88575472e-01, 9.38957846e-01],
                    [4.38070562e-02, 2.61086603e-01, 9.72130729e-01],
                    [4.53453213e-02, 3.75440392e-01, 9.61450686e-01],
                    [5.06945146e-02, 4.47658155e-01, 8.86481146e-01],
                    [4.28378252e-02, 4.50713447e-01, 7.51770770e-01],
                    [4.10520309e-02, 6.16577286e-01, 5.52730730e-01],
                    [5.28436974e-02, 7.80199548e-01, 3.82269175e-01],
                    [9.42655432e-02, 9.17674257e-01, 2.40354614e-01],
                    [1.39544593e-01, 1.00000000e00, 1.55374812e-01],
                    [7.81438836e-02, 9.27720273e-01, 1.04409358e-01],
                    [4.53805297e-02, 8.56701565e-01, 6.51222854e-02],
                    [5.22164960e-02, 7.52322921e-01, 3.42954473e-02],
                    [1.02816526e-01, 6.25809730e-01, 2.09495104e-02],
                    [5.16941760e-01, 4.92746166e-01, 1.48524616e-02],
                    [9.56397935e-01, 3.43364817e-01, 1.08983186e-02],
                    [9.73494777e-01, 2.08587708e-01, 7.00494396e-03],
                    [8.19461415e-01, 1.11784838e-01, 4.47180002e-03],
                    [6.76174158e-01, 6.59071962e-02, 4.10135388e-03],
                    [5.56804177e-01, 4.46268353e-02, 4.18528982e-03],
                    [4.19601114e-01, 3.33671033e-02, 4.49165886e-03],
                    [2.95578342e-01, 2.39487762e-02, 4.45932739e-03],
                    [2.27050628e-01, 1.87787770e-02, 4.31697313e-03],
                    [1.34118359e-01, 1.06954985e-02, 3.41192651e-03],
                    [4.00195568e-02, 5.55512389e-03, 1.36794925e-03],
                    [-4.32240535e-03, 2.49731193e-03, 3.80303275e-04],
                    [-6.00395414e-03, 1.54678227e-03, 5.40394352e-04],
                ]
            ),
            decimal=7,
        )


if __name__ == "__main__":
    unittest.main()
