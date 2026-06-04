"""Define the unit tests for the :mod:`colour.quality.tlci` module."""

from __future__ import annotations

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
)
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
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
    _nearest_locus_sample_TLCI2012,
    _Q_from_delta_E,
    sd_daylight_TLCI2012,
    sd_planckian_TLCI2012,
    sd_reference_illuminant_TLCI2012,
    television_lighting_consistency_index,
    television_luminaire_matching_factor,
    uv_to_CCT_TLCI2012,
)

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

    sd = sd_planckian_TLCI2012(CCT, SPECTRAL_SHAPE_TLCI_TLMF_TESTS)

    return sd_normalised(sd.values, f"planckian-{CCT:.0f}k")


def sd_daylight(CCT: float) -> SpectralDistribution:
    """Return a generated daylight spectrum."""

    sd = sd_daylight_TLCI2012(CCT, SPECTRAL_SHAPE_TLCI_TLMF_TESTS)

    return sd_normalised(sd.values, f"daylight-{CCT:.0f}k")


def sd_mixed_reference(CCT: float) -> SpectralDistribution:
    """Return a generated mixed-reference spectrum."""

    planckian_3400 = sd_planckian_TLCI2012(3400, SPECTRAL_SHAPE_TLCI_TLMF_TESTS)
    daylight_5000 = sd_daylight_TLCI2012(5000, SPECTRAL_SHAPE_TLCI_TLMF_TESTS)
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

    def test_sd_planckian_TLCI2012(self) -> None:
        """Test :func:`colour.quality.tlci.sd_planckian_TLCI2012` definition."""

        sd = sd_planckian_TLCI2012(3400, SPECTRAL_SHAPE_TLCI_TLMF_TESTS)

        assert sd[560] == pytest.approx(1)

        for wavelength in (380, 560, 760):
            # EBU Tech 3355 section 1.1.2.1, equation [9], normalised to unity
            # at 560 nm instead of the published value of 100.
            value = (560 / wavelength) ** 5 * (
                np.expm1(1.435e7 / (560 * 3400))
                / np.expm1(1.435e7 / (wavelength * 3400))
            )
            assert sd[wavelength] == pytest.approx(value)


class TestSdDaylightTLCI2012:
    """
    Define :func:`colour.quality.tlci.sd_daylight_TLCI2012` definition unit
    tests methods.
    """

    def test_sd_daylight_TLCI2012(self) -> None:
        """Test :func:`colour.quality.tlci.sd_daylight_TLCI2012` definition."""

        sd = sd_daylight_TLCI2012(6500, SPECTRAL_SHAPE_TLCI_TLMF_TESTS)

        # EBU Tech 3355 section 1.1.2.2 anchors the daylight components at
        # 560 nm, so the reconstruction is 100 there for any CCT.
        assert sd[560] == pytest.approx(100)
        np.testing.assert_allclose(sd[400], 82.423248, atol=TOLERANCE_ABSOLUTE_TESTS)


class TestUvToCCTTLCI2012:
    """
    Define :func:`colour.quality.tlci.uv_to_CCT_TLCI2012` definition unit
    tests methods.
    """

    def test_uv_to_CCT_TLCI2012(self) -> None:
        """Test :func:`colour.quality.tlci.uv_to_CCT_TLCI2012` definition."""

        # EBU Tech 3355 section 1.1.1 selects the Planckian locus below the
        # daylight range and the daylight locus above it.
        CCT, _uv_locus, is_daylight = uv_to_CCT_TLCI2012(np.array([0.26, 0.35]))
        np.testing.assert_allclose(CCT, 2755.437250, atol=TOLERANCE_ABSOLUTE_TESTS)
        assert not is_daylight

        CCT, _uv_locus, is_daylight = uv_to_CCT_TLCI2012(np.array([0.19, 0.31]))
        np.testing.assert_allclose(CCT, 7255.262985, atol=TOLERANCE_ABSOLUTE_TESTS)
        assert is_daylight

    def test_uv_to_CCT_TLCI2012_nearest_locus_sample(self) -> None:
        """
        Test :func:`colour.quality.tlci.uv_to_CCT_TLCI2012` definition nearest
        locus sample fallback.
        """

        # A test colour outside the tabulated normal-intersection range falls
        # back to the nearest Appendix 2 locus sample, here the 25000 K
        # daylight endpoint.
        uv = np.array([0.18, 0.20])
        CCT, uv_locus, is_daylight = uv_to_CCT_TLCI2012(uv)
        np.testing.assert_allclose(CCT, 25000, atol=TOLERANCE_ABSOLUTE_TESTS)
        assert is_daylight

        CCT_nearest, uv_locus_nearest, is_daylight_nearest = (
            _nearest_locus_sample_TLCI2012(uv)
        )
        np.testing.assert_array_equal(CCT, CCT_nearest)
        assert is_daylight == is_daylight_nearest
        np.testing.assert_array_equal(uv_locus, uv_locus_nearest)


class TestSdReferenceIlluminantTLCI2012:
    """
    Define :func:`colour.quality.tlci.sd_reference_illuminant_TLCI2012`
    definition unit tests methods.
    """

    def test_sd_reference_illuminant_TLCI2012(self) -> None:
        """
        Test :func:`colour.quality.tlci.sd_reference_illuminant_TLCI2012`
        definition.
        """

        # EBU Tech 3355 section 1.1.2 uses a Planckian reference below 3400 K
        # and a daylight reference above 5000 K.
        _sd_reference, CCT, D_uv = sd_reference_illuminant_TLCI2012(
            SDS_ILLUMINANTS["A"]
        )
        np.testing.assert_allclose(CCT, 2848.132209, atol=TOLERANCE_ABSOLUTE_TESTS)
        assert D_uv == pytest.approx(0, abs=1.5e-2)

        _sd_reference, CCT, D_uv = sd_reference_illuminant_TLCI2012(
            SDS_ILLUMINANTS["D65"]
        )
        np.testing.assert_allclose(CCT, 6505.096585, atol=TOLERANCE_ABSOLUTE_TESTS)
        assert D_uv == pytest.approx(0, abs=1.5e-2)

    def test_sd_reference_illuminant_TLCI2012_D_uv_sign(self) -> None:
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
        assert sd_reference_illuminant_TLCI2012(sd_green)[2] < -0.5
        assert sd_reference_illuminant_TLCI2012(sd_magenta)[2] > 0.5


class TestTelevisionLightingConsistencyIndex:
    """
    Define :func:`colour.quality.tlci.\
television_lighting_consistency_index` definition unit tests methods.
    """

    def test_television_lighting_consistency_index(self) -> None:
        """
        Test :func:`colour.quality.tlci.\
television_lighting_consistency_index` definition.
        """

        np.testing.assert_allclose(
            television_lighting_consistency_index(SDS_ILLUMINANTS["FL2"]),
            29.492541753138433,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # A non-default camera name selects the same *EBU Standard Camera*
        # sensitivities and therefore yields the same score.
        assert television_lighting_consistency_index(
            SDS_ILLUMINANTS["FL2"], camera="EBU Standard Camera"
        ) == television_lighting_consistency_index(SDS_ILLUMINANTS["FL2"])

    def test_television_lighting_consistency_index_additional_data(self) -> None:
        """
        Test :func:`colour.quality.tlci.\
television_lighting_consistency_index` definition with additional data.
        """

        specification = television_lighting_consistency_index(
            SDS_ILLUMINANTS["FL2"], additional_data=True
        )
        assert isinstance(specification, ColourQuality_Specification_TLCI2012)
        assert 0.0 <= specification.Q_a <= 100.0
        assert specification.delta_E_a >= 0.0
        assert specification.delta_E_s.shape == (18,)

        # EBU reference and near-blackbody illuminants score close to 100 with
        # a near-zero reference-locus distance.
        for name in ("A", "D65"):
            specification = television_lighting_consistency_index(
                SDS_ILLUMINANTS[name], additional_data=True
            )
            assert specification.Q_a == pytest.approx(100, abs=5e-4)
            assert specification.D_uv == pytest.approx(0, abs=1.5e-2)

    def test_television_lighting_consistency_index_mixed_reference(self) -> None:
        """
        Test :func:`colour.quality.tlci.\
television_lighting_consistency_index` definition mixed-reference region.
        """

        specification = television_lighting_consistency_index(
            sd_planckian(4000), additional_data=True
        )
        assert specification.Q_a == pytest.approx(100, abs=0.5)
        assert specification.D_uv == pytest.approx(0, abs=2e-2)

        specification = television_lighting_consistency_index(
            sd_mixed_reference(4500), additional_data=True
        )
        assert 4000 < specification.CCT < 5000
        assert specification.Q_a == pytest.approx(100, abs=0.5)

    def test_television_lighting_consistency_index_validation_vectors(self) -> None:
        """
        Test :func:`colour.quality.tlci.\
television_lighting_consistency_index` definition against generated and
        in-tree validation spectra.
        """

        # Generated, redistributable validation spectra covering the main
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
            assert television_lighting_consistency_index(sd) == pytest.approx(
                reference, abs=0.5
            )

        # Public in-tree CIE/NIST LED spectral distributions as regression
        # coverage.
        for name, source, reference in (
            ("LED-B3", SDS_ILLUMINANTS, 73),
            ("LED-RGB1", SDS_ILLUMINANTS, 30),
            ("Phosphor LED YAG", SDS_LIGHT_SOURCES, 60),
            ("4-LED-1 (461/526/576/624)", SDS_LIGHT_SOURCES, 79),
        ):
            assert television_lighting_consistency_index(source[name]) == pytest.approx(
                reference, abs=0.5
            )

    def test_raise_exception_television_lighting_consistency_index(self) -> None:
        """
        Test :func:`colour.quality.tlci.\
television_lighting_consistency_index` definition raised exception when all
        samples are excluded.
        """

        with pytest.raises(ValueError, match="All TLCI/TLMF samples were excluded"):
            _Q_from_delta_E(np.array([]))


class TestTelevisionLuminaireMatchingFactor:
    """
    Define :func:`colour.quality.tlci.\
television_luminaire_matching_factor` definition unit tests methods.
    """

    def test_television_luminaire_matching_factor(self) -> None:
        """
        Test :func:`colour.quality.tlci.\
television_luminaire_matching_factor` definition.
        """

        # Identical test and reference sources are a perfect match.
        assert television_luminaire_matching_factor(
            SDS_ILLUMINANTS["D65"], SDS_ILLUMINANTS["D65"]
        ) == pytest.approx(100.0, abs=1e-10)

        np.testing.assert_allclose(
            television_luminaire_matching_factor(
                SDS_ILLUMINANTS["FL2"], SDS_ILLUMINANTS["D65"]
            ),
            5.393109771266282,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_television_luminaire_matching_factor_additional_data(self) -> None:
        """
        Test :func:`colour.quality.tlci.\
television_luminaire_matching_factor` definition with additional data.
        """

        specification = television_luminaire_matching_factor(
            SDS_ILLUMINANTS["FL2"], SDS_ILLUMINANTS["D65"], additional_data=True
        )
        assert isinstance(specification, ColourQuality_Specification_TLMF2013)
        assert 0.0 <= specification.Q_a <= 100.0
        assert specification.delta_E_a >= 0.0
        assert specification.delta_E_s.shape == (24,)

    def test_television_luminaire_matching_factor_validation_vectors(self) -> None:
        """
        Test :func:`colour.quality.tlci.\
television_luminaire_matching_factor` definition against generated
        validation spectra.
        """

        for sd_test, sd_reference, reference in (
            (sd_planckian(3000), sd_planckian(3000), 100),
            (sd_daylight(5600), sd_daylight(5600), 100),
            (sd_phosphor_led_warm(), sd_planckian(3000), 3),
            (sd_phosphor_led_cool(), sd_daylight(5600), 61),
            (sd_rgb_led_balanced(), sd_daylight(5600), 4),
            (sd_mixed_reference(4000), sd_planckian(3400), 18),
            (sd_mixed_reference(4990), sd_daylight(5000), 100),
        ):
            assert television_luminaire_matching_factor(
                sd_test, sd_reference
            ) == pytest.approx(reference, abs=0.5)
