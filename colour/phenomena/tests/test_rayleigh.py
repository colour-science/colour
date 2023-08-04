# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.phenomena.rayleigh` module."""

from __future__ import annotations

import numpy as np
import unittest

from colour.phenomena.rayleigh import (
    air_refraction_index_Penndorf1957,
    air_refraction_index_Edlen1966,
    air_refraction_index_Peck1972,
    air_refraction_index_Bodhaine1999,
    N2_depolarisation,
    O2_depolarisation,
    F_air_Penndorf1957,
    F_air_Young1981,
    F_air_Bates1984,
    F_air_Bodhaine1999,
    molecular_density,
    mean_molecular_weights,
    gravity_List1968,
)
from colour.phenomena import (
    scattering_cross_section,
    rayleigh_optical_depth,
    sd_rayleigh_scattering,
)
from colour.utilities import ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "DATA_SD_RAYLEIGH_SCATTERING",
    "TestAirRefractionIndexPenndorf1957",
    "TestAirRefractionIndexEdlen1966",
    "TestAirRefractionIndexPeck1972",
    "TestAirRefractionIndexBodhaine1999",
    "TestN2Depolarisation",
    "TestO2Depolarisation",
    "TestF_airPenndorf1957",
    "TestF_airYoung1981",
    "TestF_airBates1984",
    "TestF_airBodhaine1999",
    "TestMolecularDensity",
    "TestMeanMolecularWeights",
    "TestGravityList1968",
    "TestScatteringCrossSection",
    "TestRayleighOpticalDepth",
    "TestSdRayleighScattering",
]

DATA_SD_RAYLEIGH_SCATTERING: tuple = (
    0.56024658,
    0.55374814,
    0.54734469,
    0.54103456,
    0.53481611,
    0.52868771,
    0.52264779,
    0.51669481,
    0.51082726,
    0.50504364,
    0.49934250,
    0.49372243,
    0.48818203,
    0.48271993,
    0.47733480,
    0.47202532,
    0.46679021,
    0.46162820,
    0.45653807,
    0.45151861,
    0.44656862,
    0.44168694,
    0.43687245,
    0.43212401,
    0.42744053,
    0.42282094,
    0.41826419,
    0.41376924,
    0.40933507,
    0.40496071,
    0.40064516,
    0.39638749,
    0.39218673,
    0.38804199,
    0.38395236,
    0.37991695,
    0.37593489,
    0.37200533,
    0.36812743,
    0.36430038,
    0.36052337,
    0.35679561,
    0.35311633,
    0.34948475,
    0.34590015,
    0.34236177,
    0.33886891,
    0.33542085,
    0.33201690,
    0.32865638,
    0.32533863,
    0.32206298,
    0.31882879,
    0.31563542,
    0.31248226,
    0.30936870,
    0.30629412,
    0.30325795,
    0.30025961,
    0.29729852,
    0.29437413,
    0.29148588,
    0.28863325,
    0.28581570,
    0.28303270,
    0.28028376,
    0.27756836,
    0.27488601,
    0.27223623,
    0.26961854,
    0.26703247,
    0.26447756,
    0.26195335,
    0.25945941,
    0.25699529,
    0.25456057,
    0.25215482,
    0.24977762,
    0.24742857,
    0.24510727,
    0.24281331,
    0.24054632,
    0.23830590,
    0.23609169,
    0.23390332,
    0.23174041,
    0.22960262,
    0.22748959,
    0.22540097,
    0.22333643,
    0.22129563,
    0.21927825,
    0.21728395,
    0.21531242,
    0.21336335,
    0.21143642,
    0.20953135,
    0.20764781,
    0.20578553,
    0.20394422,
    0.20212358,
    0.20032334,
    0.19854323,
    0.19678297,
    0.19504230,
    0.19332095,
    0.19161866,
    0.18993519,
    0.18827027,
    0.18662367,
    0.18499514,
    0.18338444,
    0.18179134,
    0.18021561,
    0.17865701,
    0.17711532,
    0.17559033,
    0.17408181,
    0.17258954,
    0.17111333,
    0.16965296,
    0.16820822,
    0.16677892,
    0.16536485,
    0.16396583,
    0.16258165,
    0.16121212,
    0.15985707,
    0.15851631,
    0.15718965,
    0.15587691,
    0.15457793,
    0.15329252,
    0.15202052,
    0.15076176,
    0.14951607,
    0.14828329,
    0.14706325,
    0.14585580,
    0.14466079,
    0.14347805,
    0.14230744,
    0.14114881,
    0.14000200,
    0.13886688,
    0.13774330,
    0.13663112,
    0.13553019,
    0.13444039,
    0.13336158,
    0.13229362,
    0.13123639,
    0.13018974,
    0.12915357,
    0.12812773,
    0.12711210,
    0.12610657,
    0.12511101,
    0.12412531,
    0.12314934,
    0.12218298,
    0.12122614,
    0.12027869,
    0.11934052,
    0.11841152,
    0.11749159,
    0.11658061,
    0.11567849,
    0.11478512,
    0.11390040,
    0.11302422,
    0.11215649,
    0.11129711,
    0.11044599,
    0.10960302,
    0.10876811,
    0.10794118,
    0.10712212,
    0.10631086,
    0.10550729,
    0.10471133,
    0.10392290,
    0.10314191,
    0.10236828,
    0.10160191,
    0.10084274,
    0.10009067,
    0.09934563,
    0.09860754,
    0.09787633,
    0.09715190,
    0.09643420,
    0.09572314,
    0.09501864,
    0.09432065,
    0.09362907,
    0.09294386,
    0.09226492,
    0.09159220,
    0.09092562,
    0.09026512,
    0.08961063,
    0.08896209,
    0.08831943,
    0.08768258,
    0.08705149,
    0.08642609,
    0.08580631,
    0.08519210,
    0.08458341,
    0.08398016,
    0.08338229,
    0.08278976,
    0.08220251,
    0.08162047,
    0.08104360,
    0.08047183,
    0.07990512,
    0.07934340,
    0.07878663,
    0.07823475,
    0.07768772,
    0.07714548,
    0.07660798,
    0.07607516,
    0.07554699,
    0.07502342,
    0.07450438,
    0.07398985,
    0.07347976,
    0.07297408,
    0.07247276,
    0.07197575,
    0.07148301,
    0.07099449,
    0.07051016,
    0.07002996,
    0.06955386,
    0.06908181,
    0.06861378,
    0.06814971,
    0.06768958,
    0.06723333,
    0.06678094,
    0.06633236,
    0.06588755,
    0.06544648,
    0.06500910,
    0.06457539,
    0.06414530,
    0.06371879,
    0.06329584,
    0.06287640,
    0.06246044,
    0.06204793,
    0.06163883,
    0.06123310,
    0.06083072,
    0.06043165,
    0.06003586,
    0.05964332,
    0.05925399,
    0.05886783,
    0.05848484,
    0.05810496,
    0.05772817,
    0.05735443,
    0.05698373,
    0.05661603,
    0.05625130,
    0.05588951,
    0.05553063,
    0.05517464,
    0.05482150,
    0.05447120,
    0.05412370,
    0.05377897,
    0.05343699,
    0.05309773,
    0.05276117,
    0.05242728,
    0.05209604,
    0.05176742,
    0.05144139,
    0.05111793,
    0.05079701,
    0.05047862,
    0.05016273,
    0.04984931,
    0.04953835,
    0.04922981,
    0.04892367,
    0.04861992,
    0.04831853,
    0.04801948,
    0.04772275,
    0.04742831,
    0.04713614,
    0.04684623,
    0.04655855,
    0.04627308,
    0.04598980,
    0.04570870,
    0.04542974,
    0.04515291,
    0.04487820,
    0.04460557,
    0.04433502,
    0.04406653,
    0.04380007,
    0.04353562,
    0.04327318,
    0.04301271,
    0.04275421,
    0.04249765,
    0.04224301,
    0.04199029,
    0.04173946,
    0.04149050,
    0.04124341,
    0.04099815,
    0.04075472,
    0.04051310,
    0.04027327,
    0.04003522,
    0.03979893,
    0.03956438,
    0.03933157,
    0.03910047,
    0.03887106,
    0.03864335,
    0.03841730,
    0.03819290,
    0.03797015,
    0.03774902,
    0.03752951,
    0.03731159,
    0.03709525,
    0.03688049,
    0.03666728,
    0.03645561,
    0.03624547,
    0.03603685,
    0.03582973,
    0.03562410,
    0.03541995,
    0.03521726,
    0.03501602,
    0.03481622,
    0.03461785,
    0.03442089,
    0.03422534,
    0.03403117,
    0.03383838,
    0.03364696,
    0.03345690,
    0.03326817,
    0.03308078,
    0.03289471,
    0.03270995,
    0.03252649,
    0.03234432,
    0.03216342,
    0.03198379,
    0.03180541,
    0.03162828,
    0.03145238,
    0.03127771,
    0.03110425,
    0.03093199,
    0.03076093,
    0.03059105,
    0.03042234,
    0.03025480,
    0.03008842,
    0.02992318,
    0.02975907,
    0.02959609,
    0.02943422,
    0.02927347,
    0.02911381,
    0.02895524,
    0.02879776,
    0.02864134,
    0.02848599,
    0.02833169,
    0.02817843,
    0.02802622,
    0.02787503,
    0.02772486,
    0.02757571,
    0.02742756,
    0.02728040,
    0.02713423,
    0.02698905,
    0.02684483,
    0.02670158,
    0.02655928,
    0.02641794,
    0.02627753,
    0.02613806,
    0.02599951,
    0.02586188,
    0.02572517,
    0.02558936,
    0.02545444,
    0.02532041,
    0.02518727,
    0.02505501,
    0.02492361,
    0.02479307,
    0.02466339,
    0.02453456,
    0.02440657,
    0.02427941,
    0.02415309,
    0.02402758,
    0.02390290,
    0.02377902,
    0.02365594,
    0.02353366,
)


class TestAirRefractionIndexPenndorf1957(unittest.TestCase):
    """
    Define :func:`colour.phenomena.rayleigh.\
air_refraction_index_Penndorf1957` definition unit tests methods.
    """

    def test_air_refraction_index_Penndorf1957(self):
        """
        Test :func:`colour.phenomena.rayleigh.\
air_refraction_index_Penndorf1957` definition.
        """

        self.assertAlmostEqual(
            air_refraction_index_Penndorf1957(0.360),
            1.000285316795146,
            places=10,
        )

        self.assertAlmostEqual(
            air_refraction_index_Penndorf1957(0.555),
            1.000277729533864,
            places=10,
        )

        self.assertAlmostEqual(
            air_refraction_index_Penndorf1957(0.830),
            1.000274856640486,
            places=10,
        )

    def test_n_dimensional_air_refraction_index_Penndorf1957(self):
        """
        Test :func:`colour.phenomena.rayleigh.\
air_refraction_index_Penndorf1957` definition n-dimensional arrays support.
        """

        wl = 0.360
        n = air_refraction_index_Penndorf1957(wl)

        wl = np.tile(wl, 6)
        n = np.tile(n, 6)
        np.testing.assert_array_almost_equal(
            air_refraction_index_Penndorf1957(wl), n, decimal=7
        )

        wl = np.reshape(wl, (2, 3))
        n = np.reshape(n, (2, 3))
        np.testing.assert_array_almost_equal(
            air_refraction_index_Penndorf1957(wl), n, decimal=7
        )

        wl = np.reshape(wl, (2, 3, 1))
        n = np.reshape(n, (2, 3, 1))
        np.testing.assert_array_almost_equal(
            air_refraction_index_Penndorf1957(wl), n, decimal=7
        )

    @ignore_numpy_errors
    def test_nan_air_refraction_index_Penndorf1957(self):
        """
        Test :func:`colour.phenomena.rayleigh.\
air_refraction_index_Penndorf1957` definition nan support.
        """

        air_refraction_index_Penndorf1957(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestAirRefractionIndexEdlen1966(unittest.TestCase):
    """
    Define :func:`colour.phenomena.rayleigh.air_refraction_index_Edlen1966`
    definition unit tests methods.
    """

    def test_air_refraction_index_Edlen1966(self):
        """
        Test :func:`colour.phenomena.\
rayleigh.air_refraction_index_Edlen1966` definition.
        """

        self.assertAlmostEqual(
            air_refraction_index_Edlen1966(0.360), 1.000285308809879, places=10
        )

        self.assertAlmostEqual(
            air_refraction_index_Edlen1966(0.555), 1.000277727690364, places=10
        )

        self.assertAlmostEqual(
            air_refraction_index_Edlen1966(0.830), 1.000274862218835, places=10
        )

    def test_n_dimensional_air_refraction_index_Edlen1966(self):
        """
        Test :func:`colour.phenomena.rayleigh.\
air_refraction_index_Edlen1966` definition n-dimensional arrays support.
        """

        wl = 0.360
        n = air_refraction_index_Edlen1966(wl)

        wl = np.tile(wl, 6)
        n = np.tile(n, 6)
        np.testing.assert_array_almost_equal(
            air_refraction_index_Edlen1966(wl), n, decimal=7
        )

        wl = np.reshape(wl, (2, 3))
        n = np.reshape(n, (2, 3))
        np.testing.assert_array_almost_equal(
            air_refraction_index_Edlen1966(wl), n, decimal=7
        )

        wl = np.reshape(wl, (2, 3, 1))
        n = np.reshape(n, (2, 3, 1))
        np.testing.assert_array_almost_equal(
            air_refraction_index_Edlen1966(wl), n, decimal=7
        )

    @ignore_numpy_errors
    def test_nan_air_refraction_index_Edlen1966(self):
        """
        Test :func:`colour.phenomena.rayleigh.\
air_refraction_index_Edlen1966` definition nan support.
        """

        air_refraction_index_Edlen1966(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestAirRefractionIndexPeck1972(unittest.TestCase):
    """
    Define :func:`colour.phenomena.rayleigh.air_refraction_index_Peck1972`
    definition unit tests methods.
    """

    def test_air_refraction_index_Peck1972(self):
        """
        Test :func:`colour.phenomena.rayleigh.air_refraction_index_Peck1972`
        definition.
        """

        self.assertAlmostEqual(
            air_refraction_index_Peck1972(0.360), 1.000285310285056, places=10
        )

        self.assertAlmostEqual(
            air_refraction_index_Peck1972(0.555), 1.000277726541484, places=10
        )

        self.assertAlmostEqual(
            air_refraction_index_Peck1972(0.830), 1.000274859144804, places=10
        )

    def test_n_dimensional_air_refraction_index_Peck1972(self):
        """
        Test :func:`colour.phenomena.rayleigh.air_refraction_index_Peck1972`
        definition n-dimensional arrays support.
        """

        wl = 0.360
        n = air_refraction_index_Peck1972(wl)

        wl = np.tile(wl, 6)
        n = np.tile(n, 6)
        np.testing.assert_array_almost_equal(
            air_refraction_index_Peck1972(wl), n, decimal=7
        )

        wl = np.reshape(wl, (2, 3))
        n = np.reshape(n, (2, 3))
        np.testing.assert_array_almost_equal(
            air_refraction_index_Peck1972(wl), n, decimal=7
        )

        wl = np.reshape(wl, (2, 3, 1))
        n = np.reshape(n, (2, 3, 1))
        np.testing.assert_array_almost_equal(
            air_refraction_index_Peck1972(wl), n, decimal=7
        )

    @ignore_numpy_errors
    def test_nan_air_refraction_index_Peck1972(self):
        """
        Test :func:`colour.phenomena.rayleigh.air_refraction_index_Peck1972`
        definition nan support.
        """

        air_refraction_index_Peck1972(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestAirRefractionIndexBodhaine1999(unittest.TestCase):
    """
    Define :func:`colour.phenomena.rayleigh.\
air_refraction_index_Bodhaine1999` definition unit tests methods.
    """

    def test_air_refraction_index_Bodhaine1999(self):
        """
        Test :func:`colour.phenomena.rayleigh.\
air_refraction_index_Bodhaine1999` definition.
        """

        self.assertAlmostEqual(
            air_refraction_index_Bodhaine1999(0.360),
            1.000285310285056,
            places=10,
        )

        self.assertAlmostEqual(
            air_refraction_index_Bodhaine1999(0.555),
            1.000277726541484,
            places=10,
        )

        self.assertAlmostEqual(
            air_refraction_index_Bodhaine1999(0.830),
            1.000274859144804,
            places=10,
        )

        self.assertAlmostEqual(
            air_refraction_index_Bodhaine1999(0.360, 0),
            1.000285264064789,
            places=10,
        )

        self.assertAlmostEqual(
            air_refraction_index_Bodhaine1999(0.555, 360),
            1.000277735539824,
            places=10,
        )

        self.assertAlmostEqual(
            air_refraction_index_Bodhaine1999(0.830, 620),
            1.000274906640464,
            places=10,
        )

    def test_n_dimensional_air_refraction_index_Bodhaine1999(self):
        """
        Test :func:`colour.phenomena.rayleigh.\
air_refraction_index_Bodhaine1999` definition n-dimensional arrays support.
        """

        wl = 0.360
        n = air_refraction_index_Bodhaine1999(wl)

        wl = np.tile(wl, 6)
        n = np.tile(n, 6)
        np.testing.assert_array_almost_equal(
            air_refraction_index_Bodhaine1999(wl), n, decimal=7
        )

        wl = np.reshape(wl, (2, 3))
        n = np.reshape(n, (2, 3))
        np.testing.assert_array_almost_equal(
            air_refraction_index_Bodhaine1999(wl), n, decimal=7
        )

        wl = np.reshape(wl, (2, 3, 1))
        n = np.reshape(n, (2, 3, 1))
        np.testing.assert_array_almost_equal(
            air_refraction_index_Bodhaine1999(wl), n, decimal=7
        )

    @ignore_numpy_errors
    def test_nan_air_refraction_index_Bodhaine1999(self):
        """
        Test :func:`colour.phenomena.rayleigh.\
air_refraction_index_Bodhaine1999` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        air_refraction_index_Bodhaine1999(cases, cases)


class TestN2Depolarisation(unittest.TestCase):
    """
    Define :func:`colour.phenomena.rayleigh.N2_depolarisation` definition
    unit tests methods.
    """

    def test_N2_depolarisation(self):
        """Test :func:`colour.phenomena.rayleigh.N2_depolarisation` definition."""

        self.assertAlmostEqual(
            N2_depolarisation(0.360), 1.036445987654321, places=7
        )

        self.assertAlmostEqual(
            N2_depolarisation(0.555), 1.035029137245354, places=7
        )

        self.assertAlmostEqual(
            N2_depolarisation(0.830), 1.034460153868486, places=7
        )

    def test_n_dimensional_N2_depolarisation(self):
        """
        Test :func:`colour.phenomena.rayleigh.N2_depolarisation`
        definition n-dimensional arrays support.
        """

        wl = 0.360
        n = N2_depolarisation(wl)

        wl = np.tile(wl, 6)
        n = np.tile(n, 6)
        np.testing.assert_array_almost_equal(
            N2_depolarisation(wl), n, decimal=7
        )

        wl = np.reshape(wl, (2, 3))
        n = np.reshape(n, (2, 3))
        np.testing.assert_array_almost_equal(
            N2_depolarisation(wl), n, decimal=7
        )

        wl = np.reshape(wl, (2, 3, 1))
        n = np.reshape(n, (2, 3, 1))
        np.testing.assert_array_almost_equal(
            N2_depolarisation(wl), n, decimal=7
        )

    @ignore_numpy_errors
    def test_nan_N2_depolarisation(self):
        """
        Test :func:`colour.phenomena.rayleigh.N2_depolarisation` definition
        nan support.
        """

        N2_depolarisation(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestO2Depolarisation(unittest.TestCase):
    """
    Define :func:`colour.phenomena.rayleigh.O2_depolarisation` definition
    unit tests methods.
    """

    def test_O2_depolarisation(self):
        """Test :func:`colour.phenomena.rayleigh.O2_depolarisation` definition."""

        self.assertAlmostEqual(
            O2_depolarisation(0.360), 1.115307746532541, places=7
        )

        self.assertAlmostEqual(
            O2_depolarisation(0.555), 1.102022536201071, places=7
        )

        self.assertAlmostEqual(
            O2_depolarisation(0.830), 1.098315561269013, places=7
        )

    def test_n_dimensional_O2_depolarisation(self):
        """
        Test :func:`colour.phenomena.rayleigh.O2_depolarisation` definition
        n-dimensional arrays support.
        """

        wl = 0.360
        n = O2_depolarisation(wl)

        wl = np.tile(wl, 6)
        n = np.tile(n, 6)
        np.testing.assert_array_almost_equal(
            O2_depolarisation(wl), n, decimal=7
        )

        wl = np.reshape(wl, (2, 3))
        n = np.reshape(n, (2, 3))
        np.testing.assert_array_almost_equal(
            O2_depolarisation(wl), n, decimal=7
        )

        wl = np.reshape(wl, (2, 3, 1))
        n = np.reshape(n, (2, 3, 1))
        np.testing.assert_array_almost_equal(
            O2_depolarisation(wl), n, decimal=7
        )

    @ignore_numpy_errors
    def test_nan_O2_depolarisation(self):
        """
        Test :func:`colour.phenomena.rayleigh.O2_depolarisation` definition
        nan support.
        """

        O2_depolarisation(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestF_airPenndorf1957(unittest.TestCase):
    """
    Define :func:`colour.phenomena.rayleigh.F_air_Penndorf1957` definition
    unit tests methods.
    """

    def test_F_air_Penndorf1957(self):
        """
        Test :func:`colour.phenomena.rayleigh.F_air_Penndorf1957`
        definition.
        """

        self.assertEqual(F_air_Penndorf1957(0.360), 1.0608)

    def test_n_dimensional_F_air_Penndorf1957(self):
        """
        Test :func:`colour.phenomena.rayleigh.F_air_Penndorf1957` definition
        n-dimensional arrays support.
        """

        wl = 0.360
        n = F_air_Penndorf1957(wl)

        wl = np.tile(wl, 6)
        n = np.tile(n, 6)
        np.testing.assert_array_almost_equal(
            F_air_Penndorf1957(wl), n, decimal=7
        )

        wl = np.reshape(wl, (2, 3))
        n = np.reshape(n, (2, 3))
        np.testing.assert_array_almost_equal(
            F_air_Penndorf1957(wl), n, decimal=7
        )

        wl = np.reshape(wl, (2, 3, 1))
        n = np.reshape(n, (2, 3, 1))
        np.testing.assert_array_almost_equal(
            F_air_Penndorf1957(wl), n, decimal=7
        )

    @ignore_numpy_errors
    def test_nan_F_air_Penndorf1957(self):
        """
        Test :func:`colour.phenomena.rayleigh.F_air_Penndorf1957` definition
        nan support.
        """

        F_air_Penndorf1957(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestF_airYoung1981(unittest.TestCase):
    """
    Define :func:`colour.phenomena.rayleigh.F_air_Young1981` definition
    unit tests methods.
    """

    def test_F_air_Young1981(self):
        """Test :func:`colour.phenomena.rayleigh.F_air_Young1981` definition."""

        self.assertEqual(F_air_Young1981(0.360), 1.0480)

    def test_n_dimensional_F_air_Young1981(self):
        """
        Test :func:`colour.phenomena.rayleigh.F_air_Young1981` definition
        n-dimensional arrays support.
        """

        wl = 0.360
        n = F_air_Young1981(wl)

        wl = np.tile(wl, 6)
        n = np.tile(n, 6)
        np.testing.assert_array_almost_equal(F_air_Young1981(wl), n, decimal=7)

        wl = np.reshape(wl, (2, 3))
        n = np.reshape(n, (2, 3))
        np.testing.assert_array_almost_equal(F_air_Young1981(wl), n, decimal=7)

        wl = np.reshape(wl, (2, 3, 1))
        n = np.reshape(n, (2, 3, 1))
        np.testing.assert_array_almost_equal(F_air_Young1981(wl), n, decimal=7)

    @ignore_numpy_errors
    def test_nan_F_air_Young1981(self):
        """
        Test :func:`colour.phenomena.rayleigh.F_air_Young1981` definition
        nan support.
        """

        F_air_Young1981(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestF_airBates1984(unittest.TestCase):
    """
    Define :func:`colour.phenomena.rayleigh.F_air_Bates1984` definition unit
    tests methods.
    """

    def test_F_air_Bates1984(self):
        """Test :func:`colour.phenomena.rayleigh.F_air_Bates1984` definition."""

        self.assertAlmostEqual(
            F_air_Bates1984(0.360), 1.051997277711708, places=7
        )

        self.assertAlmostEqual(
            F_air_Bates1984(0.555), 1.048153579718658, places=7
        )

        self.assertAlmostEqual(
            F_air_Bates1984(0.830), 1.046947068600589, places=7
        )

    def test_n_dimensional_F_air_Bates1984(self):
        """
        Test :func:`colour.phenomena.rayleigh.F_air_Bates1984` definition
        n-dimensional arrays support.
        """

        wl = 0.360
        n = F_air_Bates1984(wl)

        wl = np.tile(wl, 6)
        n = np.tile(n, 6)
        np.testing.assert_array_almost_equal(F_air_Bates1984(wl), n, decimal=7)

        wl = np.reshape(wl, (2, 3))
        n = np.reshape(n, (2, 3))
        np.testing.assert_array_almost_equal(F_air_Bates1984(wl), n, decimal=7)

        wl = np.reshape(wl, (2, 3, 1))
        n = np.reshape(n, (2, 3, 1))
        np.testing.assert_array_almost_equal(F_air_Bates1984(wl), n, decimal=7)

    @ignore_numpy_errors
    def test_nan_F_air_Bates1984(self):
        """
        Test :func:`colour.phenomena.rayleigh.F_air_Bates1984` definition
        nan support.
        """

        F_air_Bates1984(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestF_airBodhaine1999(unittest.TestCase):
    """
    Define :func:`colour.phenomena.rayleigh.F_air_Bodhaine1999` definition
    unit tests methods.
    """

    def test_F_air_Bodhaine1999(self):
        """
        Test :func:`colour.phenomena.rayleigh.F_air_Bodhaine1999`
        definition.
        """

        self.assertAlmostEqual(
            F_air_Bodhaine1999(0.360), 1.052659005129014, places=7
        )

        self.assertAlmostEqual(
            F_air_Bodhaine1999(0.555), 1.048769718142427, places=7
        )

        self.assertAlmostEqual(
            F_air_Bodhaine1999(0.830), 1.047548896943893, places=7
        )

        self.assertAlmostEqual(
            F_air_Bodhaine1999(0.360, 0), 1.052629792313939, places=7
        )

        self.assertAlmostEqual(
            F_air_Bodhaine1999(0.555, 360), 1.048775791959338, places=7
        )

        self.assertAlmostEqual(
            F_air_Bodhaine1999(0.830, 620), 1.047581672775155, places=7
        )

    def test_n_dimensional_F_air_Bodhaine1999(self):
        """
        Test :func:`colour.phenomena.rayleigh.F_air_Bodhaine1999` definition
        n-dimensional arrays support.
        """

        wl = 0.360
        n = F_air_Bodhaine1999(wl)

        wl = np.tile(wl, 6)
        n = np.tile(n, 6)
        np.testing.assert_array_almost_equal(
            F_air_Bodhaine1999(wl), n, decimal=7
        )

        wl = np.reshape(wl, (2, 3))
        n = np.reshape(n, (2, 3))
        np.testing.assert_array_almost_equal(
            F_air_Bodhaine1999(wl), n, decimal=7
        )

        wl = np.reshape(wl, (2, 3, 1))
        n = np.reshape(n, (2, 3, 1))
        np.testing.assert_array_almost_equal(
            F_air_Bodhaine1999(wl), n, decimal=7
        )

    @ignore_numpy_errors
    def test_nan_F_air_Bodhaine1999(self):
        """
        Test :func:`colour.phenomena.rayleigh.F_air_Bodhaine1999` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        F_air_Bodhaine1999(cases, cases)


class TestMolecularDensity(unittest.TestCase):
    """
    Define :func:`colour.phenomena.rayleigh.molecular_density` definition
    unit tests methods.
    """

    def test_molecular_density(self):
        """Test :func:`colour.phenomena.rayleigh.molecular_density` definition."""

        self.assertAlmostEqual(
            molecular_density(200), 3.669449208173649e19, delta=10000
        )

        self.assertAlmostEqual(
            molecular_density(300), 2.4462994721157665e19, delta=10000
        )

        self.assertAlmostEqual(
            molecular_density(400), 1.834724604086825e19, delta=10000
        )

    def test_n_dimensional_molecular_density(self):
        """
        Test :func:`colour.phenomena.rayleigh.molecular_density` definition
        n-dimensional arrays support.
        """

        temperature = 200
        N_s = molecular_density(temperature)

        temperature = np.tile(temperature, 6)
        N_s = np.tile(N_s, 6)
        np.testing.assert_array_almost_equal(
            molecular_density(temperature), N_s, decimal=7
        )

        temperature = np.reshape(temperature, (2, 3))
        N_s = np.reshape(N_s, (2, 3))
        np.testing.assert_array_almost_equal(
            molecular_density(temperature), N_s, decimal=7
        )

        temperature = np.reshape(temperature, (2, 3, 1))
        N_s = np.reshape(N_s, (2, 3, 1))
        np.testing.assert_array_almost_equal(
            molecular_density(temperature), N_s, decimal=7
        )

    @ignore_numpy_errors
    def test_nan_molecular_density(self):
        """
        Test :func:`colour.phenomena.rayleigh.molecular_density` definition
        nan support.
        """

        molecular_density(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestMeanMolecularWeights(unittest.TestCase):
    """
    Define :func:`colour.phenomena.rayleigh.mean_molecular_weights`
    definition unit tests methods.
    """

    def test_mean_molecular_weights(self):
        """
        Test :func:`colour.phenomena.rayleigh.mean_molecular_weights`
        definition.
        """

        self.assertAlmostEqual(mean_molecular_weights(0), 28.9595, places=7)

        self.assertAlmostEqual(
            mean_molecular_weights(360), 28.964920015999997, places=7
        )

        self.assertAlmostEqual(
            mean_molecular_weights(620), 28.968834471999998, places=7
        )

    def test_n_dimensional_mean_molecular_weights(self):
        """
        Test :func:`colour.phenomena.rayleigh.mean_molecular_weights`
        definition n-dimensional arrays support.
        """

        CO2_c = 300
        m_a = mean_molecular_weights(CO2_c)

        CO2_c = np.tile(CO2_c, 6)
        m_a = np.tile(m_a, 6)
        np.testing.assert_array_almost_equal(
            mean_molecular_weights(CO2_c), m_a, decimal=7
        )

        CO2_c = np.reshape(CO2_c, (2, 3))
        m_a = np.reshape(m_a, (2, 3))
        np.testing.assert_array_almost_equal(
            mean_molecular_weights(CO2_c), m_a, decimal=7
        )

        CO2_c = np.reshape(CO2_c, (2, 3, 1))
        m_a = np.reshape(m_a, (2, 3, 1))
        np.testing.assert_array_almost_equal(
            mean_molecular_weights(CO2_c), m_a, decimal=7
        )

    @ignore_numpy_errors
    def test_nan_mean_molecular_weights(self):
        """
        Test :func:`colour.phenomena.rayleigh.mean_molecular_weights`
        definition nan support.
        """

        mean_molecular_weights(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestGravityList1968(unittest.TestCase):
    """
    Define :func:`colour.phenomena.rayleigh.gravity_List1968` definition
    unit tests methods.
    """

    def test_gravity_List1968(self):
        """Test :func:`colour.phenomena.rayleigh.gravity_List1968` definition."""

        self.assertAlmostEqual(
            gravity_List1968(0.0, 0.0), 978.03560706, places=7
        )

        self.assertAlmostEqual(
            gravity_List1968(45.0, 1500.0), 980.15334386, places=7
        )

        self.assertAlmostEqual(
            gravity_List1968(48.8567, 35.0), 980.95241784, places=7
        )

    def test_n_dimensional_gravity_List1968(self):
        """
        Test :func:`colour.phenomena.rayleigh.gravity_List1968`
        definition n-dimensional arrays support.
        """

        g = 978.03560706
        np.testing.assert_array_almost_equal(gravity_List1968(), g, decimal=7)

        g = np.tile(g, 6)
        np.testing.assert_array_almost_equal(gravity_List1968(), g, decimal=7)

        g = np.reshape(g, (2, 3))
        np.testing.assert_array_almost_equal(gravity_List1968(), g, decimal=7)

        g = np.reshape(g, (2, 3, 1))
        np.testing.assert_array_almost_equal(gravity_List1968(), g, decimal=7)

    @ignore_numpy_errors
    def test_nan_gravity_List1968(self):
        """
        Test :func:`colour.phenomena.rayleigh.gravity_List1968` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        gravity_List1968(cases, cases)


class TestScatteringCrossSection(unittest.TestCase):
    """
    Define :func:`colour.phenomena.rayleigh.scattering_cross_section`
    definition unit tests methods.
    """

    def test_scattering_cross_section(self):
        """
        Test :func:`colour.phenomena.rayleigh.scattering_cross_section`
        definition.
        """

        self.assertAlmostEqual(
            scattering_cross_section(360 * 10e-8),
            2.600908533851937e-26,
            places=32,
        )

        self.assertAlmostEqual(
            scattering_cross_section(555 * 10e-8),
            4.346669248087624e-27,
            places=32,
        )

        self.assertAlmostEqual(
            scattering_cross_section(830 * 10e-8),
            8.501515434751428e-28,
            places=32,
        )

        self.assertAlmostEqual(
            scattering_cross_section(555 * 10e-8, 0),
            4.346543336839102e-27,
            places=32,
        )

        self.assertAlmostEqual(
            scattering_cross_section(555 * 10e-8, 360),
            4.346694421271718e-27,
            places=32,
        )

        self.assertAlmostEqual(
            scattering_cross_section(555 * 10e-8, 620),
            4.346803470171720e-27,
            places=32,
        )

        self.assertAlmostEqual(
            scattering_cross_section(555 * 10e-8, temperature=200),
            2.094012829135068e-27,
            places=32,
        )

        self.assertAlmostEqual(
            scattering_cross_section(555 * 10e-8, temperature=300),
            4.711528865553901e-27,
            places=32,
        )

        self.assertAlmostEqual(
            scattering_cross_section(555 * 10e-8, temperature=400),
            8.376051316540270e-27,
            places=32,
        )

    def test_n_dimensional_scattering_cross_section(self):
        """
        Test :func:`colour.phenomena.rayleigh.scattering_cross_section`
        definition n-dimensional arrays support.
        """

        wl = 360 * 10e-8
        sigma = scattering_cross_section(wl)

        sigma = np.tile(sigma, 6)
        np.testing.assert_array_almost_equal(
            scattering_cross_section(wl), sigma, decimal=32
        )

        sigma = np.reshape(sigma, (2, 3))
        np.testing.assert_array_almost_equal(
            scattering_cross_section(wl), sigma, decimal=32
        )

        sigma = np.reshape(sigma, (2, 3, 1))
        np.testing.assert_array_almost_equal(
            scattering_cross_section(wl), sigma, decimal=32
        )

    @ignore_numpy_errors
    def test_nan_scattering_cross_section(self):
        """
        Test :func:`colour.phenomena.rayleigh.scattering_cross_section`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        scattering_cross_section(cases, cases, cases)


class TestRayleighOpticalDepth(unittest.TestCase):
    """
    Define :func:`colour.phenomena.rayleigh.rayleigh_optical_depth`
    definition unit tests methods.
    """

    def test_rayleigh_optical_depth(self):
        """
        Test :func:`colour.phenomena.rayleigh.rayleigh_optical_depth`
        definition.
        """

        self.assertAlmostEqual(
            rayleigh_optical_depth(360 * 10e-8), 0.560246579231107, places=7
        )

        self.assertAlmostEqual(
            rayleigh_optical_depth(555 * 10e-8), 0.093629074056042, places=7
        )

        self.assertAlmostEqual(
            rayleigh_optical_depth(830 * 10e-8), 0.018312619911882, places=7
        )

        self.assertAlmostEqual(
            rayleigh_optical_depth(555 * 10e-8, 0), 0.093640964348049, places=7
        )

        self.assertAlmostEqual(
            rayleigh_optical_depth(555 * 10e-8, 360),
            0.093626696247360,
            places=7,
        )

        self.assertAlmostEqual(
            rayleigh_optical_depth(555 * 10e-8, 620),
            0.093616393371777,
            places=7,
        )

        self.assertAlmostEqual(
            rayleigh_optical_depth(555 * 10e-8, temperature=200),
            0.045105912380991,
            places=7,
        )

        self.assertAlmostEqual(
            rayleigh_optical_depth(555 * 10e-8, temperature=300),
            0.101488302857230,
            places=7,
        )

        self.assertAlmostEqual(
            rayleigh_optical_depth(555 * 10e-8, temperature=400),
            0.180423649523964,
            places=7,
        )

        self.assertAlmostEqual(
            rayleigh_optical_depth(555 * 10e-8, pressure=101325),
            0.093629074056042,
            places=7,
        )

        self.assertAlmostEqual(
            rayleigh_optical_depth(555 * 10e-8, pressure=100325),
            0.092705026939772,
            places=7,
        )

        self.assertAlmostEqual(
            rayleigh_optical_depth(555 * 10e-8, pressure=99325),
            0.091780979823502,
            places=7,
        )

        self.assertAlmostEqual(
            rayleigh_optical_depth(555 * 10e-8, latitude=0, altitude=0),
            0.093629074056041,
            places=10,
        )

        self.assertAlmostEqual(
            rayleigh_optical_depth(555 * 10e-8, latitude=45, altitude=1500),
            0.093426777407767,
            places=10,
        )

        self.assertAlmostEqual(
            rayleigh_optical_depth(555 * 10e-8, latitude=48.8567, altitude=35),
            0.093350672894038,
            places=10,
        )

    def test_n_dimensional_rayleigh_optical_depth(self):
        """
        Test :func:`colour.phenomena.rayleigh.rayleigh_optical_depth`
        definition n-dimensional arrays support.
        """

        wl = 360 * 10e-8
        T_R = rayleigh_optical_depth(wl)

        T_R = np.tile(T_R, 6)
        np.testing.assert_array_almost_equal(
            rayleigh_optical_depth(wl), T_R, decimal=7
        )

        T_R = np.reshape(T_R, (2, 3))
        np.testing.assert_array_almost_equal(
            rayleigh_optical_depth(wl), T_R, decimal=7
        )

        T_R = np.reshape(T_R, (2, 3, 1))
        np.testing.assert_array_almost_equal(
            rayleigh_optical_depth(wl), T_R, decimal=7
        )

    @ignore_numpy_errors
    def test_nan_rayleigh_optical_depth(self):
        """
        Test :func:`colour.phenomena.rayleigh.rayleigh_optical_depth`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        rayleigh_optical_depth(cases, cases, cases, cases, cases)


class TestSdRayleighScattering(unittest.TestCase):
    """
    Define :func:`colour.phenomena.rayleigh.sd_rayleigh_scattering`
    definition unit tests methods.
    """

    def test_sd_rayleigh_scattering(self):
        """
        Test :func:`colour.phenomena.rayleigh.sd_rayleigh_scattering`
        definition.
        """

        np.testing.assert_array_almost_equal(
            sd_rayleigh_scattering().values,
            DATA_SD_RAYLEIGH_SCATTERING,
            decimal=7,
        )


if __name__ == "__main__":
    unittest.main()
