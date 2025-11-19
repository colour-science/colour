"""Define the unit tests for the :mod:`colour.recovery.jiang2013` module."""

from __future__ import annotations

import platform
import warnings

import numpy as np
import pytest

from colour.characterisation import MSDS_CAMERA_SENSITIVITIES, SDS_COLOURCHECKERS
from colour.colorimetry import (
    SDS_ILLUMINANTS,
    SpectralDistribution,
    SpectralShape,
    msds_to_XYZ,
    reshape_msds,
    reshape_sd,
    sds_and_msds_to_msds,
)
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
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
    "FixtureJiang2013",
    "TestRGB_to_sd_camera_sensitivity_Jiang2013",
    "TestRGB_to_msds_camera_sensitivities_Jiang2013",
]


class TestPCA_Jiang2013:
    """
    Define :func:`colour.recovery.jiang2013.PCA_Jiang2013` definition unit
    tests methods.
    """

    @pytest.mark.skipif(
        platform.system() in ("Windows", "Microsoft", "Linux"),
        reason="PCA tests only run on macOS",
    )
    def test_PCA_Jiang2013(self) -> None:
        """Test :func:`colour.recovery.jiang2013.PCA_Jiang2013` definition."""

        shape = SpectralShape(400, 700, 10)
        camera_sensitivities = {
            camera: msds.copy().align(shape)
            for camera, msds in MSDS_CAMERA_SENSITIVITIES.items()
        }
        w, v = PCA_Jiang2013(
            camera_sensitivities,
            3,
            additional_data=True,
        )

        np.testing.assert_allclose(
            np.abs(np.array(w)),
            np.array(
                [
                    [
                        [0.00137594, 0.00399416, 0.00074515],
                        [0.00214835, 0.00184422, 0.85753040],
                        [0.02757181, 0.00553587, 0.02033235],
                        [0.02510621, 0.04216468, 0.01860012],
                        [0.02011623, 0.03371162, 0.01474896],
                        [0.01392282, 0.03297985, 0.00645105],
                        [0.00944513, 0.03300938, 0.00649418],
                        [0.02019958, 0.01289400, 0.01138365],
                        [0.02394423, 0.00980934, 0.00348705],
                        [0.04196326, 0.04987050, 0.00462977],
                        [0.04988732, 0.06936603, 0.01299320],
                        [0.06527141, 0.09378614, 0.00320186],
                        [0.09412575, 0.12244081, 0.02936062],
                        [0.10915913, 0.13119983, 0.02403866],
                        [0.12314840, 0.24280936, 0.03433110],
                        [0.11673941, 0.27700737, 0.11678028],
                        [0.12534133, 0.29994127, 0.11683063],
                        [0.14599255, 0.25586532, 0.04332511],
                        [0.25249090, 0.11499750, 0.04112949],
                        [0.35163407, 0.45286818, 0.04259130],
                        [0.35805737, 0.40724252, 0.24276366],
                        [0.36927899, 0.18120838, 0.28042101],
                        [0.35374885, 0.03010008, 0.05772918],
                        [0.35340909, 0.16847527, 0.23388908],
                        [0.32696116, 0.29068981, 0.09224334],
                        [0.29067354, 0.32862702, 0.14708450],
                        [0.08964758, 0.09682656, 0.04541265],
                        [0.01891664, 0.08221113, 0.01157780],
                        [0.00521149, 0.01578907, 0.00133064],
                        [0.00232366, 0.00137751, 0.00139661],
                        [0.00153787, 0.00254398, 0.00042512],
                    ],
                    [
                        [0.00119598, 0.00267792, 0.00026101],
                        [0.00200327, 0.00322983, 0.62788798],
                        [0.01247816, 0.03313976, 0.01072884],
                        [0.03207685, 0.05703294, 0.03730884],
                        [0.04715050, 0.05296451, 0.02043038],
                        [0.05794010, 0.05455737, 0.01279190],
                        [0.10745571, 0.00158911, 0.03200481],
                        [0.14178525, 0.03362764, 0.15020663],
                        [0.16811402, 0.05569833, 0.00844788],
                        [0.18463716, 0.04615404, 0.03103250],
                        [0.21531623, 0.09745078, 0.39693604],
                        [0.25442570, 0.18330481, 0.14940077],
                        [0.28168018, 0.25193267, 0.00347081],
                        [0.29237178, 0.28545428, 0.03254141],
                        [0.29693117, 0.23909467, 0.00779634],
                        [0.28631319, 0.19476441, 0.31698680],
                        [0.27195968, 0.12087420, 0.28305850],
                        [0.25988140, 0.01581316, 0.04130875],
                        [0.24222660, 0.07912972, 0.18481425],
                        [0.23069698, 0.18583667, 0.18113417],
                        [0.20831983, 0.26745561, 0.13793240],
                        [0.19437168, 0.32425009, 0.20908259],
                        [0.18470894, 0.34768079, 0.15013614],
                        [0.18056180, 0.35983221, 0.24060984],
                        [0.17141337, 0.35067306, 0.04478566],
                        [0.14712541, 0.30423172, 0.05583266],
                        [0.02897026, 0.04573993, 0.02137366],
                        [0.00190228, 0.00461591, 0.00240276],
                        [0.00069122, 0.00118817, 0.00011696],
                        [0.00045559, 0.00015286, 0.00075939],
                        [0.00039509, 0.00049719, 0.00104581],
                    ],
                    [
                        [0.03283371, 0.04707162, 0.99591944],
                        [0.05932690, 0.07529740, 0.06152683],
                        [0.11947381, 0.07977219, 0.00156116],
                        [0.18492233, 0.26127374, 0.00717981],
                        [0.22091564, 0.29279976, 0.00487132],
                        [0.25377875, 0.30677709, 0.00614140],
                        [0.29969822, 0.26541777, 0.00429149],
                        [0.30232755, 0.25378622, 0.00354243],
                        [0.30031732, 0.19751184, 0.00199307],
                        [0.28072276, 0.11804285, 0.00591452],
                        [0.26005747, 0.01836333, 0.00698676],
                        [0.23839367, 0.07182421, 0.01904751],
                        [0.21721831, 0.14245410, 0.00452400],
                        [0.19828405, 0.17684950, 0.01371456],
                        [0.19018451, 0.20137781, 0.01184653],
                        [0.18196762, 0.22086321, 0.01434790],
                        [0.17168644, 0.22771873, 0.02205056],
                        [0.16977073, 0.23504018, 0.01730589],
                        [0.16277670, 0.22897797, 0.02229014],
                        [0.15880423, 0.22583675, 0.02123217],
                        [0.14966812, 0.21494312, 0.01417066],
                        [0.13480155, 0.19511162, 0.01901915],
                        [0.12541764, 0.18113238, 0.01413883],
                        [0.12355731, 0.17835150, 0.01809536],
                        [0.11175064, 0.15997651, 0.01436804],
                        [0.09440304, 0.13423453, 0.01316519],
                        [0.01670581, 0.02019670, 0.00202853],
                        [0.00045002, 0.00147362, 0.00007713],
                        [0.00102919, 0.00095904, 0.00008866],
                        [0.00097397, 0.00123434, 0.00011166],
                        [0.00097116, 0.00124835, 0.00014463],
                    ],
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        np.testing.assert_allclose(
            np.array(v),
            np.array(
                [
                    [10.55160659, 0.72837380, 0.00000000],
                    [20.09177982, 1.57662524, 0.00000000],
                    [19.04142816, 2.60426480, 0.00000000],
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # Test with additional_data=False (default)
        R_w, G_w, B_w = PCA_Jiang2013(camera_sensitivities, 3)  # type: ignore[misc]

        np.testing.assert_allclose(
            np.abs(R_w),
            np.abs(w[0]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        np.testing.assert_allclose(
            np.abs(G_w),
            np.abs(w[1]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        np.testing.assert_allclose(
            np.abs(B_w),
            np.abs(w[2]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class FixtureJiang2013:
    """A fixture for testing the :mod:`colour.recovery.jiang2013` module."""

    @pytest.fixture(autouse=True)
    def setup_fixture_jiang_2013(self) -> None:
        """Configure the class instance."""

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
            cast("SpectralDistribution", self._reflectances.copy()).align(
                SPECTRAL_SHAPE_BASIS_FUNCTIONS_DYER2017
            ),
            method="Integration",
            cmfs=self._sensitivities,
            illuminant=self._sd_D65,
            k=1,
            shape=SPECTRAL_SHAPE_BASIS_FUNCTIONS_DYER2017,
        )


class TestRGB_to_sd_camera_sensitivity_Jiang2013(FixtureJiang2013):
    """
    Define :func:`colour.recovery.jiang2013.RGB_to_sd_camera_sensitivity_Jiang2013`
    definition unit tests methods.
    """

    def setup_method(self) -> None:
        """Initialise the common tests attributes."""

        FixtureJiang2013.__init__(self)

    def test_RGB_to_sd_camera_sensitivity_Jiang2013(self) -> None:
        """
        Test :func:`colour.recovery.jiang2013.\
RGB_to_sd_camera_sensitivity_Jiang2013` definition.
        """

        R_w, _G_w, _B_w = tsplit(np.moveaxis(BASIS_FUNCTIONS_DYER2017, 0, 1))

        np.testing.assert_allclose(
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
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # Test with mismatched illuminant shape to trigger warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Use original D65 illuminant without reshaping
            sd = RGB_to_sd_camera_sensitivity_Jiang2013(
                self._RGB[..., 0],
                SDS_ILLUMINANTS["D65"],  # Not reshaped
                self._reflectances,
                R_w,
                SPECTRAL_SHAPE_BASIS_FUNCTIONS_DYER2017,
            )
            # Should have triggered a warning about aligning illuminant shape
            assert len(w) >= 1
            assert "Aligning" in str(w[0].message)

        # Result should still be valid
        np.testing.assert_allclose(
            sd.values,
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
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestRGB_to_msds_camera_sensitivities_Jiang2013(FixtureJiang2013):
    """
    Define :func:`colour.recovery.jiang2013.\
RGB_to_msds_camera_sensitivities_Jiang2013` definition unit tests methods.
    """

    def setup_method(self) -> None:
        """Initialise the common tests attributes."""

        FixtureJiang2013.__init__(self)

    def test_RGB_to_msds_camera_sensitivities_Jiang2013(self) -> None:
        """
        Test :func:`colour.recovery.jiang2013.\
RGB_to_msds_camera_sensitivities_Jiang2013` definition.
        """

        np.testing.assert_allclose(
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
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # Test with mismatched illuminant shape to trigger warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Use original D65 illuminant without reshaping
            msds = RGB_to_msds_camera_sensitivities_Jiang2013(
                self._RGB,
                SDS_ILLUMINANTS["D65"],  # Not reshaped
                self._reflectances,
                BASIS_FUNCTIONS_DYER2017,
                SPECTRAL_SHAPE_BASIS_FUNCTIONS_DYER2017,
            )
            # Should have triggered a warning about aligning illuminant shape
            assert len(w) >= 1
            assert "Aligning" in str(w[0].message)

        # Result should still be valid
        np.testing.assert_allclose(
            msds.values,
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
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
