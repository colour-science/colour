# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.recovery.jiang2013` module."""

import numpy as np
import unittest

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
from colour.characterisation import (
    MSDS_CAMERA_SENSITIVITIES,
    SDS_COLOURCHECKERS,
)
from colour.recovery import (
    BASIS_FUNCTIONS_DYER2017,
    PCA_Jiang2013,
    RGB_to_sd_camera_sensitivity_Jiang2013,
    RGB_to_msds_camera_sensitivities_Jiang2013,
    SPECTRAL_SHAPE_BASIS_FUNCTIONS_DYER2017,
)
from colour.utilities import tsplit

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
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
                        [-1.37594347e-03, 3.99415732e-03, -6.04239828e-04],
                        [-2.14834542e-03, 1.84422104e-03, 3.21589852e-01],
                        [-2.75718149e-02, 5.53587216e-03, 5.40389791e-02],
                        [-2.51062139e-02, -4.21646800e-02, -2.70235131e-02],
                        [-2.01162330e-02, -3.37116245e-02, -5.68936622e-03],
                        [-1.39228153e-02, -3.29798532e-02, -1.95221649e-02],
                        [-9.44512929e-03, -3.30093792e-02, -3.04555338e-02],
                        [-2.01995779e-02, -1.28939958e-02, 2.22562167e-02],
                        [-2.39442292e-02, -9.80934362e-03, -1.01147393e-02],
                        [-4.19632625e-02, 4.98705001e-02, -8.57577942e-03],
                        [-4.98873166e-02, 6.93660333e-02, 2.05472209e-01],
                        [-6.52714132e-02, 9.37861421e-02, 3.73495566e-02],
                        [-9.41257461e-02, 1.22440811e-01, 1.40014392e-01],
                        [-1.09159133e-01, 1.31199835e-01, 8.83163250e-02],
                        [-1.23148401e-01, 2.42809357e-01, 2.01309902e-01],
                        [-1.16739413e-01, 2.77007375e-01, 1.86676559e-01],
                        [-1.25341332e-01, 2.99941272e-01, 2.17837858e-01],
                        [-1.45992549e-01, 2.55865322e-01, 2.27131989e-01],
                        [-2.52490904e-01, -1.14997503e-01, -2.20533434e-01],
                        [-3.51634066e-01, -4.52868183e-01, -1.39652545e-01],
                        [-3.58057369e-01, -4.07242523e-01, 2.58829683e-01],
                        [-3.69278993e-01, -1.81208375e-01, 5.19968568e-01],
                        [-3.53748849e-01, -3.01000786e-02, -2.55655981e-01],
                        [-3.53409087e-01, 1.68475269e-01, -2.84106537e-01],
                        [-3.26961156e-01, 2.90689809e-01, -1.63314143e-01],
                        [-2.90673539e-01, 3.28627021e-01, -2.49733529e-01],
                        [-8.96475804e-02, -9.68265584e-02, -6.97440356e-02],
                        [-1.89166432e-02, -8.22111312e-02, -2.42073036e-03],
                        [-5.21149231e-03, -1.57890680e-02, -4.33465348e-05],
                        [-2.32366424e-03, -1.37750678e-03, 4.06389245e-04],
                        [-1.53786508e-03, 2.54398457e-03, -3.80279581e-04],
                    ],
                    [
                        [-1.19597520e-03, 2.67792288e-03, -3.86643802e-04],
                        [-2.00326785e-03, 3.22982608e-03, 5.86749223e-01],
                        [-1.24781553e-02, -3.31397595e-02, -3.56296307e-02],
                        [-3.20768524e-02, -5.70329419e-02, -9.69280990e-03],
                        [-4.71505049e-02, -5.29645114e-02, 7.66902127e-02],
                        [-5.79400988e-02, -5.45573700e-02, -3.14568063e-04],
                        [-1.07455711e-01, 1.58910582e-03, -7.05327185e-02],
                        [-1.41785255e-01, -3.36276409e-02, -1.05701318e-01],
                        [-1.68114016e-01, -5.56983321e-02, -1.23153686e-01],
                        [-1.84637156e-01, -4.61540356e-02, 2.00697387e-01],
                        [-2.15316233e-01, -9.74507835e-02, 1.80376932e-01],
                        [-2.54425702e-01, -1.83304806e-01, -1.36618663e-01],
                        [-2.81680176e-01, -2.51932665e-01, -7.73950380e-02],
                        [-2.92371785e-01, -2.85454281e-01, 1.46039739e-02],
                        [-2.96931166e-01, -2.39094667e-01, -2.41823482e-01],
                        [-2.86313189e-01, -1.94764406e-01, -5.44135282e-02],
                        [-2.71959682e-01, -1.20874204e-01, 4.06551254e-01],
                        [-2.59881399e-01, -1.58131608e-02, 3.24704412e-01],
                        [-2.42226602e-01, 7.91297201e-02, -2.70855184e-01],
                        [-2.30696979e-01, 1.85836669e-01, -2.53245992e-02],
                        [-2.08319826e-01, 2.67455613e-01, -2.32430787e-01],
                        [-1.94371680e-01, 3.24250095e-01, 1.02375615e-01],
                        [-1.84708937e-01, 3.47680795e-01, -1.85437047e-02],
                        [-1.80561798e-01, 3.59832206e-01, -6.30127108e-02],
                        [-1.71413369e-01, 3.50673059e-01, 1.97691095e-01],
                        [-1.47125410e-01, 3.04231720e-01, -7.27877894e-02],
                        [-2.89702613e-02, 4.57399297e-02, -1.10745662e-02],
                        [-1.90227628e-03, -4.61591110e-03, 3.34622914e-04],
                        [-6.91221876e-04, -1.18816974e-03, -6.73601845e-04],
                        [-4.55594621e-04, 1.52863579e-04, -3.90316310e-05],
                        [-3.95094896e-04, 4.97187702e-04, -3.12174899e-04],
                    ],
                    [
                        [-3.28337112e-02, 4.70716173e-02, -9.98004473e-01],
                        [-5.93269036e-02, 7.52974019e-02, 2.05260665e-02],
                        [-1.19473811e-01, -7.97721941e-02, -6.30794639e-03],
                        [-1.84922334e-01, -2.61273738e-01, -5.20246519e-03],
                        [-2.20915638e-01, -2.92799758e-01, -1.19336143e-02],
                        [-2.53778754e-01, -3.06777085e-01, 1.25936314e-03],
                        [-2.99698219e-01, -2.65417765e-01, -7.01599755e-03],
                        [-3.02327547e-01, -2.53786221e-01, -6.03326686e-03],
                        [-3.00317318e-01, -1.97511844e-01, 6.89292013e-03],
                        [-2.80722764e-01, -1.18042847e-01, 5.98880768e-04],
                        [-2.60057468e-01, -1.83633282e-02, 1.24579462e-02],
                        [-2.38393671e-01, 7.18242068e-02, 1.33587494e-02],
                        [-2.17218308e-01, 1.42454103e-01, 1.25650206e-02],
                        [-1.98284048e-01, 1.76849496e-01, 2.16223803e-02],
                        [-1.90184510e-01, 2.01377808e-01, 1.45692504e-02],
                        [-1.81967620e-01, 2.20863209e-01, 9.80505521e-03],
                        [-1.71686442e-01, 2.27718732e-01, 1.72325013e-02],
                        [-1.69770729e-01, 2.35040183e-01, 2.20444682e-02],
                        [-1.62776705e-01, 2.28977971e-01, 1.55453131e-02],
                        [-1.58804226e-01, 2.25836749e-01, 2.01730266e-02],
                        [-1.49668124e-01, 2.14943120e-01, 9.86008333e-03],
                        [-1.34801552e-01, 1.95111622e-01, 1.15561577e-02],
                        [-1.25417642e-01, 1.81132377e-01, 5.97801693e-03],
                        [-1.23557313e-01, 1.78351495e-01, 1.02617561e-02],
                        [-1.11750645e-01, 1.59976514e-01, 8.14162092e-03],
                        [-9.44030423e-02, 1.34234535e-01, 9.11543681e-03],
                        [-1.67058128e-02, 2.01966959e-02, 1.09903718e-03],
                        [-4.50023573e-04, -1.47361849e-03, -6.44936616e-05],
                        [-1.02919085e-03, 9.59038318e-04, 8.20098491e-05],
                        [-9.73971054e-04, 1.23434432e-03, 6.73016617e-05],
                        [-9.71160886e-04, 1.24835480e-03, 8.00827657e-05],
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

        # pylint: disable=E1102
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
