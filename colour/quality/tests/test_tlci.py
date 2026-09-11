"""Define the unit tests for the :mod:`colour.quality.tlci` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

import numpy as np
import pytest

from colour.algebra import linstep_function
from colour.characterisation import RGB_CameraSensitivities
from colour.colorimetry import (
    SDS_ILLUMINANTS,
    SDS_LIGHT_SOURCES,
    SpectralDistribution,
    SpectralShape,
    reshape_sd,
    sd_CIE_illuminant_D_series,
)
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models import xy_to_UCS_uv
from colour.quality.datasets import (
    DATA_DAYLIGHT_BASIS_TLCI2012,
    DATA_DAYLIGHT_LOCUS_TLCI2012,
    DATA_PLANCKIAN_LOCUS_TLCI2012,
    DATA_TCS_TLCI2012,
    MSDS_CAMERA_SENSITIVITIES_TLCI2012,
    SDS_TCS_TLCI2012,
)
from colour.quality.tlci import (
    ColourQuality_Specification_TLCI2012,
    ColourQuality_Specification_TLMF2013,
    colour_differences_TLCI2012,
    quality_index_TLCI2012,
    sd_daylight_TLCI2012,
    sd_planckian_TLCI2012,
    sd_reference_illuminant_TLCI2012,
    television_lighting_consistency_index,
    television_luminaire_matching_factor,
    uv_to_CCT_TLCI2012,
)
from colour.temperature import CCT_to_xy_CIE_D
from colour.utilities import xp_as_array, xp_assert_close

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "SPECTRAL_SHAPE_TLCI_TLMF_TESTS",
    "gaussian",
    "sd_normalised",
    "sd_planckian",
    "sd_daylight",
    "sd_mixed_reference",
    "sd_phosphor_led_warm",
    "sd_phosphor_led_cool",
    "sd_rgb_led_balanced",
    "TestDatasetsTLCI2012",
    "TestSdPlanckianTLCI2012",
    "TestSdDaylightTLCI2012",
    "TestUvToCCTTLCI2012",
    "TestSdReferenceIlluminantTLCI2012",
    "TestColourDifferencesTLCI2012",
    "TestQualityIndexTLCI2012",
    "TestTelevisionLightingConsistencyIndex",
    "TestTelevisionLuminaireMatchingFactor",
]

SPECTRAL_SHAPE_TLCI_TLMF_TESTS: SpectralShape = SpectralShape(380, 760, 5)


def gaussian(wavelengths: np.ndarray, centre: float, width: float) -> np.ndarray:
    """Return a unit Gaussian spectral lobe."""

    return np.exp(-0.5 * ((wavelengths - centre) / width) ** 2)


def sd_normalised(values: np.ndarray, name: str) -> SpectralDistribution:
    """Return a peak-normalised spectral distribution."""

    values = np.clip(values, 0, None)
    peak = np.max(values)

    if peak != 0:
        values = values / peak

    return SpectralDistribution(values, SPECTRAL_SHAPE_TLCI_TLMF_TESTS, name=name)


def sd_planckian(CCT: float) -> SpectralDistribution:
    """Return a generated Planckian spectrum."""

    wavelengths = SPECTRAL_SHAPE_TLCI_TLMF_TESTS.wavelengths
    values = (560 / wavelengths) ** 5 * (
        np.expm1(1.435e7 / (560 * CCT)) / np.expm1(1.435e7 / (wavelengths * CCT))
    )

    return sd_normalised(values, f"planckian-{CCT:.0f}k")


def sd_daylight(CCT: float) -> SpectralDistribution:
    """Return a generated daylight spectrum."""

    sd = sd_CIE_illuminant_D_series(
        CCT_to_xy_CIE_D(CCT), shape=SPECTRAL_SHAPE_TLCI_TLMF_TESTS
    )

    return sd_normalised(sd.values, f"daylight-{CCT:.0f}k")


def sd_mixed_reference(CCT: float) -> SpectralDistribution:
    """Return a generated mixed-reference spectrum."""

    planckian_3400 = sd_planckian(3400)
    daylight_5000 = sd_daylight(5000)
    planckian_3400_values = planckian_3400.values / planckian_3400[560]
    daylight_5000_values = daylight_5000.values / daylight_5000[560]
    weight = (CCT - 3400) / (5000 - 3400)

    return sd_normalised(
        linstep_function(weight, planckian_3400_values, daylight_5000_values),
        f"mixed-reference-{CCT:.0f}k",
    )


def sd_phosphor_led_warm() -> SpectralDistribution:
    """Return a generated warm phosphor LED spectrum."""

    wavelengths = SPECTRAL_SHAPE_TLCI_TLMF_TESTS.wavelengths

    return sd_normalised(
        0.85 * gaussian(wavelengths, 455, 12)
        + 1.00 * gaussian(wavelengths, 600, 80)
        + 0.18 * gaussian(wavelengths, 630, 25),
        "phosphor-led-warm",
    )


def sd_phosphor_led_cool() -> SpectralDistribution:
    """Return a generated cool phosphor LED spectrum."""

    wavelengths = SPECTRAL_SHAPE_TLCI_TLMF_TESTS.wavelengths

    return sd_normalised(
        1.00 * gaussian(wavelengths, 450, 10)
        + 0.82 * gaussian(wavelengths, 545, 75)
        + 0.22 * gaussian(wavelengths, 610, 35),
        "phosphor-led-cool",
    )


def sd_rgb_led_balanced() -> SpectralDistribution:
    """Return a generated RGB LED spectrum."""

    wavelengths = SPECTRAL_SHAPE_TLCI_TLMF_TESTS.wavelengths

    return sd_normalised(
        0.72 * gaussian(wavelengths, 455, 14)
        + 1.00 * gaussian(wavelengths, 535, 18)
        + 0.86 * gaussian(wavelengths, 625, 18),
        "rgb-led-balanced",
    )


class TestDatasetsTLCI2012:
    """
    Define :mod:`colour.quality.datasets.tlci2012` datasets unit tests
    methods.
    """

    def test_datasets(self) -> None:
        """Test the *TLCI-2012* datasets."""

        assert DATA_PLANCKIAN_LOCUS_TLCI2012.shape == (152, 3)
        assert DATA_DAYLIGHT_LOCUS_TLCI2012.shape == (104, 3)
        assert DATA_DAYLIGHT_BASIS_TLCI2012.shape == (77, 3)
        assert len(DATA_TCS_TLCI2012) == 24
        assert len(SDS_TCS_TLCI2012) == 24

        # Representative values and published sums from *EBU Tech 3355*
        # Appendix 2, covering the Planckian and daylight radiator tables.
        for table, index, values in (
            (DATA_PLANCKIAN_LOCUS_TLCI2012, 0, [1000, 0.652355, 0.344814]),
            (DATA_PLANCKIAN_LOCUS_TLCI2012, 150, [4999, 0.344774, 0.351363]),
            (DATA_PLANCKIAN_LOCUS_TLCI2012, 151, [5000, 0.344746, 0.351341]),
            (DATA_DAYLIGHT_LOCUS_TLCI2012, 0, [5000, 0.345747, 0.358680]),
            (DATA_DAYLIGHT_LOCUS_TLCI2012, 1, [5001, 0.345718, 0.358657]),
            (DATA_DAYLIGHT_LOCUS_TLCI2012, 103, [25000, 0.249866, 0.254845]),
        ):
            np.testing.assert_allclose(table[index], values)

        np.testing.assert_allclose(
            np.sum(DATA_PLANCKIAN_LOCUS_TLCI2012[:, 1:], axis=0),
            [74.357659, 60.180968],
        )
        np.testing.assert_allclose(
            np.sum(DATA_DAYLIGHT_LOCUS_TLCI2012[:, 1:], axis=0),
            [31.602241, 33.132636],
        )

        # Representative values from *EBU Tech 3355* Appendix 3, covering the
        # daylight radiation vector table and its published sums.
        for index, values in (
            (0, [63.4, 38.5, 3.0]),
            (1, [62.45, 35.98125, 2.05]),
            (5, [101.54375, 45.525, -0.93125]),
            (76, [47.7, -7.8, 5.2]),
        ):
            np.testing.assert_allclose(DATA_DAYLIGHT_BASIS_TLCI2012[index], values)

        np.testing.assert_allclose(
            np.sum(DATA_DAYLIGHT_BASIS_TLCI2012, axis=0),
            [7140.45, 545.05, 201.55],
        )

        # Representative values from *EBU Tech 3355* Appendix 4, covering the
        # coloured and greyscale test colour sample tables.
        for sample, wavelength, value in (
            ("dark skin", 380, 0.054),
            ("dark skin", 760, 0.490),
            ("white 9.5 (.05 D)", 380, 0.126),
            ("white 9.5 (.05 D)", 760, 0.898),
        ):
            np.testing.assert_allclose(SDS_TCS_TLCI2012[sample][wavelength], value)

        # Values mandated by *EBU Tech 3355* Appendix 4 that differ from the
        # cited *BBC R&D Report 1988/2* table over the overlapping spectral
        # range.
        for sample, wavelength, value, bbc_value in (
            ("foliage", 725, 0.445, 0.455),
            ("bluish green", 555, 0.445, 0.449),
            ("green", 430, 0.054, 0.064),
        ):
            np.testing.assert_allclose(SDS_TCS_TLCI2012[sample][wavelength], value)
            assert not np.isclose(SDS_TCS_TLCI2012[sample][wavelength], bbc_value)

        camera = MSDS_CAMERA_SENSITIVITIES_TLCI2012["EBU Standard Camera"]
        assert isinstance(camera, RGB_CameraSensitivities)
        assert camera.wavelengths[0] == 380


class TestSdPlanckianTLCI2012:
    """
    Define :func:`colour.quality.tlci.sd_planckian_TLCI2012` definition unit
    tests methods.
    """

    def test_sd_planckian_TLCI2012(self, xp: ModuleType) -> None:
        """Test :func:`colour.quality.tlci.sd_planckian_TLCI2012` definition."""

        sd = sd_planckian_TLCI2012(
            xp_as_array(3400, xp=xp), SPECTRAL_SHAPE_TLCI_TLMF_TESTS
        )

        xp_assert_close(sd[560], 100)

        for wavelength in (380, 560, 760):
            # EBU Tech 3355 section 1.1.2.1, equation [9].
            value = (
                100
                * (560 / wavelength) ** 5
                * (
                    np.expm1(1.435e7 / (560 * 3400))
                    / np.expm1(1.435e7 / (wavelength * 3400))
                )
            )
            xp_assert_close(sd[wavelength], value)


class TestSdDaylightTLCI2012:
    """
    Define :func:`colour.quality.tlci.sd_daylight_TLCI2012` definition unit
    tests methods.
    """

    def test_sd_daylight_TLCI2012(self, xp: ModuleType) -> None:
        """Test :func:`colour.quality.tlci.sd_daylight_TLCI2012` definition."""

        sd = sd_daylight_TLCI2012(
            xp_as_array(6500, xp=xp), SPECTRAL_SHAPE_TLCI_TLMF_TESTS
        )

        # EBU Tech 3355 section 1.1.2.2 anchors the daylight components at
        # 560 nm, so the reconstruction is 100 there for any CCT.
        xp_assert_close(sd[560], 100)
        xp_assert_close(sd[400], 82.423248, atol=TOLERANCE_ABSOLUTE_TESTS)


class TestUvToCCTTLCI2012:
    """
    Define :func:`colour.quality.tlci.uv_to_CCT_TLCI2012` definition unit
    tests methods.
    """

    @pytest.mark.mps_tolerance_absolute(3e-2)
    def test_uv_to_CCT_TLCI2012(self, xp: ModuleType) -> None:
        """Test :func:`colour.quality.tlci.uv_to_CCT_TLCI2012` definition."""

        # EBU Tech 3355 section 1.1.1 selects the Planckian locus below the
        # daylight range and the daylight locus above it.
        CCT, _uv_locus, is_daylight = uv_to_CCT_TLCI2012(
            xp_as_array([0.26, 0.35], xp=xp)
        )
        xp_assert_close(CCT, 2755.437250, atol=TOLERANCE_ABSOLUTE_TESTS)
        assert not is_daylight

        CCT, _uv_locus, is_daylight = uv_to_CCT_TLCI2012(
            xp_as_array([0.19, 0.31], xp=xp)
        )
        xp_assert_close(CCT, 7255.262985, atol=TOLERANCE_ABSOLUTE_TESTS)
        assert is_daylight

    def test_uv_to_CCT_TLCI2012_nearest_locus_sample(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.quality.tlci.uv_to_CCT_TLCI2012` definition nearest
        locus sample fallback.
        """

        # A test colour outside the tabulated normal-intersection range falls
        # back to the nearest Appendix 2 locus sample, here the 25000 K
        # daylight endpoint.
        uv = xp_as_array([0.18, 0.20], xp=xp)
        CCT, uv_locus, is_daylight = uv_to_CCT_TLCI2012(uv)
        xp_assert_close(CCT, 25000, atol=TOLERANCE_ABSOLUTE_TESTS)
        assert is_daylight

        xp_assert_close(
            uv_locus,
            xy_to_UCS_uv(DATA_DAYLIGHT_LOCUS_TLCI2012[-1, 1:]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_uv_to_CCT_TLCI2012_daylight_endpoint(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.quality.tlci.uv_to_CCT_TLCI2012` definition at the
        first daylight-locus sample.
        """

        uv = xp_as_array(xy_to_UCS_uv(DATA_DAYLIGHT_LOCUS_TLCI2012[0, 1:]), xp=xp)
        CCT, uv_locus, is_daylight = uv_to_CCT_TLCI2012(uv)

        xp_assert_close(CCT, 5000, atol=TOLERANCE_ABSOLUTE_TESTS)
        xp_assert_close(uv_locus, uv, atol=TOLERANCE_ABSOLUTE_TESTS)
        assert is_daylight


class TestSdReferenceIlluminantTLCI2012:
    """
    Define :func:`colour.quality.tlci.sd_reference_illuminant_TLCI2012`
    definition unit tests methods.
    """

    @pytest.mark.mps_tolerance_absolute(2e-3)
    def test_sd_reference_illuminant_TLCI2012(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.quality.tlci.sd_reference_illuminant_TLCI2012`
        definition.
        """

        # EBU Tech 3355 section 1.1.2 uses a Planckian reference below 3400 K
        # and a daylight reference above 5000 K.
        sd_reference, CCT, D_uv = sd_reference_illuminant_TLCI2012(
            SDS_ILLUMINANTS["A"].copy(xp=xp)
        )
        xp_assert_close(sd_reference[560], 100)
        xp_assert_close(CCT, 2848.132209, atol=TOLERANCE_ABSOLUTE_TESTS)
        xp_assert_close(D_uv, 0, atol=1.5e-2)

        sd_reference, CCT, D_uv = sd_reference_illuminant_TLCI2012(
            SDS_ILLUMINANTS["D65"].copy(xp=xp)
        )
        xp_assert_close(sd_reference[560], 100)
        xp_assert_close(CCT, 6505.096585, atol=TOLERANCE_ABSOLUTE_TESTS)
        xp_assert_close(D_uv, 0, atol=1.5e-2)

    def test_sd_reference_illuminant_TLCI2012_D_uv_sign(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.quality.tlci.sd_reference_illuminant_TLCI2012`
        definition ``D_uv`` sign convention.
        """

        sd_D65 = reshape_sd(
            SDS_ILLUMINANTS["D65"], SPECTRAL_SHAPE_TLCI_TLMF_TESTS, "Align", copy=False
        )
        wavelengths = SPECTRAL_SHAPE_TLCI_TLMF_TESTS.wavelengths
        peak = np.max(sd_D65.values)
        sd_green = sd_normalised(
            sd_D65.values + 0.2 * peak * gaussian(wavelengths, 545, 20),
            "green-offset",
        )
        sd_magenta = sd_normalised(
            sd_D65.values
            + 0.1 * peak * gaussian(wavelengths, 450, 15)
            + 0.1 * peak * gaussian(wavelengths, 650, 20),
            "magenta-offset",
        )

        # EBU Tech 3355 section 1.1.1 reverses the sign for green-side offsets
        # (uT < uL); section 1.1.2.3 equations [16]-[17] label positive d as
        # magenta and negative d as green.
        assert sd_reference_illuminant_TLCI2012(sd_green.copy(xp=xp))[2] < -0.5
        assert sd_reference_illuminant_TLCI2012(sd_magenta.copy(xp=xp))[2] > 0.5


class TestColourDifferencesTLCI2012:
    """
    Define :func:`colour.quality.tlci.colour_differences_TLCI2012` definition
    unit tests methods.
    """

    def test_colour_differences_TLCI2012(self, xp: ModuleType) -> None:
        """Test :func:`colour.quality.tlci.colour_differences_TLCI2012`."""

        msds_camera = MSDS_CAMERA_SENSITIVITIES_TLCI2012["EBU Standard Camera"].copy(
            xp=xp
        )
        delta_E_s, invalid = colour_differences_TLCI2012(
            SDS_ILLUMINANTS["D65"].copy(xp=xp),
            SDS_ILLUMINANTS["D65"].copy(xp=xp),
            msds_camera,
        )
        xp_assert_close(delta_E_s, 0, atol=TOLERANCE_ABSOLUTE_TESTS)
        assert invalid.shape == (24,)

        # EBU Tech 3355 section 1.3.1 applies the reference balance to both
        # TLMF sources before normalising the test-source camera luma.
        delta_E_s, invalid = colour_differences_TLCI2012(
            SDS_ILLUMINANTS["FL2"].copy(xp=xp),
            SDS_ILLUMINANTS["D65"].copy(xp=xp),
            msds_camera,
            normalise_test_luma_only=True,
        )
        Q_a, _delta_E_a = quality_index_TLCI2012(delta_E_s[~invalid])
        xp_assert_close(Q_a, 5.376079603398939)


class TestQualityIndexTLCI2012:
    """
    Define :func:`colour.quality.tlci.quality_index_TLCI2012` definition unit
    tests methods.
    """

    def test_quality_index_TLCI2012(self, xp: ModuleType) -> None:
        """Test :func:`colour.quality.tlci.quality_index_TLCI2012`."""

        # EBU Tech 3355 section 1.5.1 defines 3.16 as the watershed aggregate
        # colour difference that must produce a quality index of 50.
        # Public numerical definitions accept array-like inputs.
        delta_E_s = xp_as_array([3.16], xp=xp)
        Q_a, delta_E_a = quality_index_TLCI2012(delta_E_s)
        xp_assert_close(delta_E_a, 3.16)
        xp_assert_close(Q_a, 50)

    def test_raise_exception_quality_index_TLCI2012(self, xp: ModuleType) -> None:
        """Test :func:`colour.quality.tlci.quality_index_TLCI2012` exception."""

        with pytest.raises(ValueError, match="All TLCI/TLMF samples were excluded"):
            quality_index_TLCI2012(xp_as_array([], xp=xp))


class TestTelevisionLightingConsistencyIndex:
    """
    Define :func:`colour.quality.tlci.\
television_lighting_consistency_index` definition unit tests methods.
    """

    def test_television_lighting_consistency_index(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.quality.tlci.\
television_lighting_consistency_index` definition.
        """

        sd_fl2_xp = SDS_ILLUMINANTS["FL2"].copy(xp=xp)

        xp_assert_close(
            television_lighting_consistency_index(sd_fl2_xp),
            29.492541753138433,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # An explicit camera name selects the same *EBU Standard Camera*
        # sensitivities as the default and therefore yields the same score.
        xp_assert_close(
            television_lighting_consistency_index(
                sd_fl2_xp, camera="EBU Standard Camera"
            ),
            television_lighting_consistency_index(sd_fl2_xp),
        )

    def test_television_lighting_consistency_index_additional_data(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.quality.tlci.\
television_lighting_consistency_index` definition with additional data.
        """

        specification = television_lighting_consistency_index(
            SDS_ILLUMINANTS["FL2"].copy(xp=xp), additional_data=True
        )
        assert isinstance(specification, ColourQuality_Specification_TLCI2012)
        assert 0.0 <= specification.Q_a <= 100.0
        assert specification.delta_E_a >= 0.0
        assert specification.delta_E_s.shape == (18,)

        # EBU reference and near-blackbody illuminants score close to 100 with
        # a near-zero reference-locus distance.
        for name in ("A", "D65"):
            specification = television_lighting_consistency_index(
                SDS_ILLUMINANTS[name].copy(xp=xp), additional_data=True
            )
            xp_assert_close(specification.Q_a, 100, atol=5e-4)
            xp_assert_close(specification.D_uv, 0, atol=1.5e-2)

    def test_television_lighting_consistency_index_autodiff(
        self, xp: ModuleType, autodiff: typing.Callable
    ) -> None:
        """Test local numerical gradients through spectral alignment."""

        sd = SDS_ILLUMINANTS["FL2"]
        _Q_a, (gradient,), _inputs = autodiff(
            lambda values: television_lighting_consistency_index(
                SpectralDistribution(values, sd.wavelengths)
            ),
            sd.values,
        )

        assert xp.isfinite(gradient).all()
        assert xp.any(gradient != 0)

    def test_television_lighting_consistency_index_mixed_reference(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.quality.tlci.\
television_lighting_consistency_index` definition mixed-reference region.
        """

        specification = television_lighting_consistency_index(
            sd_planckian(4000).copy(xp=xp), additional_data=True
        )
        xp_assert_close(specification.Q_a, 100, atol=0.5)
        xp_assert_close(specification.D_uv, 0, atol=2e-2)

        specification = television_lighting_consistency_index(
            sd_mixed_reference(4500).copy(xp=xp), additional_data=True
        )
        assert 4000 < specification.CCT < 5000
        xp_assert_close(specification.Q_a, 100, atol=0.5)

    def test_television_lighting_consistency_index_regression_spectra(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.quality.tlci.\
television_lighting_consistency_index` definition against generated and
        in-tree regression spectra.
        """

        # Generated, redistributable regression spectra covering the main
        # specified algorithm paths.
        for sd, reference in (
            (sd_planckian(3000), 100),
            (sd_planckian(4000), 100),
            (sd_daylight(5600), 100),
            (sd_mixed_reference(4500), 100),
            (sd_phosphor_led_warm(), 88),
            (sd_phosphor_led_cool(), 77),
            (sd_rgb_led_balanced(), 58),
        ):
            xp_assert_close(
                television_lighting_consistency_index(sd.copy(xp=xp)),
                reference,
                atol=0.5,
            )

        # Public in-tree CIE/NIST LED spectral distributions as regression
        # coverage.
        for name, source, reference in (
            ("LED-B3", SDS_ILLUMINANTS, 73),
            ("LED-RGB1", SDS_ILLUMINANTS, 30),
            ("Phosphor LED YAG", SDS_LIGHT_SOURCES, 60),
            ("4-LED-1 (461/526/576/624)", SDS_LIGHT_SOURCES, 79),
        ):
            xp_assert_close(
                television_lighting_consistency_index(source[name].copy(xp=xp)),
                reference,
                atol=0.5,
            )


class TestTelevisionLuminaireMatchingFactor:
    """
    Define :func:`colour.quality.tlci.\
television_luminaire_matching_factor` definition unit tests methods.
    """

    def test_television_luminaire_matching_factor(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.quality.tlci.\
television_luminaire_matching_factor` definition.
        """

        # Identical test and reference sources are a perfect match.
        xp_assert_close(
            television_luminaire_matching_factor(
                SDS_ILLUMINANTS["D65"].copy(xp=xp),
                SDS_ILLUMINANTS["D65"].copy(xp=xp),
            ),
            100.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            television_luminaire_matching_factor(
                SDS_ILLUMINANTS["FL2"].copy(xp=xp),
                SDS_ILLUMINANTS["D65"].copy(xp=xp),
            ),
            5.376079603398939,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_television_luminaire_matching_factor_additional_data(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.quality.tlci.\
television_luminaire_matching_factor` definition with additional data.
        """

        specification = television_luminaire_matching_factor(
            SDS_ILLUMINANTS["FL2"].copy(xp=xp),
            SDS_ILLUMINANTS["D65"].copy(xp=xp),
            additional_data=True,
        )
        assert isinstance(specification, ColourQuality_Specification_TLMF2013)
        assert 0.0 <= specification.Q_a <= 100.0
        assert specification.delta_E_a >= 0.0
        assert specification.delta_E_s.shape == (24,)

    def test_television_luminaire_matching_factor_autodiff(
        self, xp: ModuleType, autodiff: typing.Callable
    ) -> None:
        """Test local numerical gradients through spectral alignment."""

        sd = SDS_ILLUMINANTS["FL2"]
        sd_reference = SDS_ILLUMINANTS["D65"]
        _Q_a, (gradient,), _inputs = autodiff(
            lambda values: television_luminaire_matching_factor(
                SpectralDistribution(values, sd.wavelengths), sd_reference
            ),
            sd.values,
        )

        assert xp.isfinite(gradient).all()
        assert xp.any(gradient != 0)

    def test_television_luminaire_matching_factor_regression_spectra(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.quality.tlci.\
television_luminaire_matching_factor` definition against generated regression
        spectra.
        """

        for sd_test, sd_reference, reference in (
            (sd_planckian(3000), sd_planckian(3000), 100),
            (sd_daylight(5600), sd_daylight(5600), 100),
            (sd_phosphor_led_warm(), sd_planckian(3000), 3.5),
            (sd_phosphor_led_cool(), sd_daylight(5600), 61),
            (sd_rgb_led_balanced(), sd_daylight(5600), 4),
            (sd_mixed_reference(4000), sd_planckian(3400), 18),
            (sd_mixed_reference(4990), sd_daylight(5000), 100),
        ):
            xp_assert_close(
                television_luminaire_matching_factor(
                    sd_test.copy(xp=xp), sd_reference.copy(xp=xp)
                ),
                reference,
                atol=0.5,
            )
