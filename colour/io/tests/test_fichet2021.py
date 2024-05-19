# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.io.fichet2021` module."""

from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np

from colour.characterisation import SDS_COLOURCHECKERS
from colour.colorimetry import (
    MSDS_CMFS,
    SDS_ILLUMINANTS,
    SpectralShape,
    sds_and_msds_to_msds,
)
from colour.constants import CONSTANT_LIGHT_SPEED, TOLERANCE_ABSOLUTE_TESTS
from colour.io import (
    Specification_Fichet2021,
    read_spectral_image_Fichet2021,
    sd_to_spectrum_attribute_Fichet2021,
    spectrum_attribute_to_sd_Fichet2021,
    write_spectral_image_Fichet2021,
)
from colour.io.fichet2021 import (
    components_to_sRGB_Fichet2021,
    match_groups_to_nm,
    sds_and_msds_to_components_Fichet2021,
)
from colour.utilities import is_openimageio_installed

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "ROOT_RESOURCES",
    "TestMatchGroupsToNm",
    "TestSdToSpectrumAttributeFichet2021",
    "TestSpectrumAttributeToSdFichet2021",
    "TestSdsAndMsdsToComponentsFichet2021",
    "TestComponentsToSRGBFichet2021",
    "TestReadSpectralImageFichet2021",
    "TestWriteSpectralImageFichet2021",
]

ROOT_RESOURCES: str = os.path.join(os.path.dirname(__file__), "resources")


class TestMatchGroupsToNm:
    """
    Define :func:`colour.io.fichet2021.match_groups_to_nm` definition unit
    tests methods.
    """

    def test_match_groups_to_nm(self):
        """Test :func:`colour.io.fichet2021.match_groups_to_nm` definition."""

        np.testing.assert_allclose(
            match_groups_to_nm("555.5", "n", "m"),
            555.5,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            match_groups_to_nm("555.5", "", "m"),
            555500000000.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            match_groups_to_nm(str(CONSTANT_LIGHT_SPEED / (555 * 1e-9)), "", "Hz"),
            555.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestSdToSpectrumAttributeFichet2021:
    """
    Define :func:`colour.io.fichet2021.sd_to_spectrum_attribute_Fichet2021`
    definition unit tests methods.
    """

    def test_sd_to_spectrum_attribute_Fichet2021(self):
        """
        Test :func:`colour.io.fichet2021.\
sd_to_spectrum_attribute_Fichet2021` definition.
        """

        assert (
            sd_to_spectrum_attribute_Fichet2021(SDS_ILLUMINANTS["D65"], 2)[:56]
            == "300.00nm:0.03;305.00nm:1.66;310.00nm:3.29;315.00nm:11.77"
        )


class TestSpectrumAttributeToSdFichet2021:
    """
    Define :func:`colour.io.fichet2021.spectrum_attribute_to_sd_Fichet2021`
    definition unit tests methods.
    """

    def test_spectrum_attribute_to_sd_Fichet2021(self):
        """
        Test :func:`colour.io.fichet2021.\
spectrum_attribute_to_sd_Fichet2021` definition.
        """

        sd = spectrum_attribute_to_sd_Fichet2021(
            "300.00nm:0.03;305.00nm:1.66;310.00nm:3.29;315.00nm:11.77"
        )

        np.testing.assert_allclose(
            sd.wavelengths,
            np.array([300.0, 305.0, 310.0, 315.0]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            sd.values,
            np.array([0.03, 1.66, 3.29, 11.77]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestSdsAndMsdsToComponentsFichet2021:
    """
    Define :func:`colour.io.fichet2021.sds_and_msds_to_components_Fichet2021`
    definition unit tests methods.
    """

    def test_sds_and_msds_to_components_Fichet2021(self):
        """
        Test :func:`colour.io.fichet2021.\
sds_and_msds_to_components_Fichet2021` definition.
        """

        components = sds_and_msds_to_components_Fichet2021(SDS_ILLUMINANTS["D65"])

        assert "T" in components

        components = sds_and_msds_to_components_Fichet2021(
            SDS_ILLUMINANTS["D65"], Specification_Fichet2021(is_emissive=True)
        )

        assert "S0" in components

        np.testing.assert_allclose(
            components["S0"][0],
            SDS_ILLUMINANTS["D65"].wavelengths,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            components["S0"][1],
            np.reshape(SDS_ILLUMINANTS["D65"].values, (1, 1, -1)),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        components = sds_and_msds_to_components_Fichet2021(
            list(SDS_COLOURCHECKERS["ColorChecker N Ohta"].values())
        )

        assert components["T"][1].shape == (1, 24, 81)


class TestComponentsToSRGBFichet2021:
    """
    Define :func:`colour.io.fichet2021.components_to_sRGB_Fichet2021`
    definition unit tests methods.
    """

    def test_components_to_sRGB_Fichet2021(self):
        """
        Test :func:`colour.io.fichet2021.components_to_sRGB_Fichet2021`
        definition.
        """

        if not is_openimageio_installed():
            return

        specification = Specification_Fichet2021(is_emissive=True)
        components = sds_and_msds_to_components_Fichet2021(
            SDS_ILLUMINANTS["D65"], specification
        )
        RGB, attributes = components_to_sRGB_Fichet2021(components, specification)

        np.testing.assert_allclose(
            RGB,
            np.array([[[0.17998291, 0.18000802, 0.18000908]]]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        assert [attribute.name for attribute in attributes] == [
            "X",
            "Y",
            "Z",
            "illuminant",
            "chromaticities",
            "EV",
        ]

        for attribute in attributes:
            if attribute.name == "X":
                sd_X = spectrum_attribute_to_sd_Fichet2021(attribute.value)
                np.testing.assert_allclose(
                    sd_X.values,
                    MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
                    .signals["x_bar"]
                    .values,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )
            elif attribute.name == "illuminant":
                sd_illuminant = spectrum_attribute_to_sd_Fichet2021(attribute.value)
                np.testing.assert_allclose(
                    sd_illuminant.values,
                    SDS_ILLUMINANTS["E"].values,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )
            elif attribute.name == "chromaticities":
                assert attribute.value == [
                    0.64,
                    0.33,
                    0.3,
                    0.6,
                    0.15,
                    0.06,
                    0.3127,
                    0.329,
                ]

        specification = Specification_Fichet2021(is_emissive=False)
        components = sds_and_msds_to_components_Fichet2021(
            list(SDS_COLOURCHECKERS["ColorChecker N Ohta"].values()), specification
        )
        RGB, attributes = components_to_sRGB_Fichet2021(components, specification)

        np.testing.assert_allclose(
            RGB,
            np.array(
                [
                    [
                        [0.17617566, 0.07822266, 0.05031637],
                        [0.55943028, 0.30875974, 0.22283237],
                        [0.11315875, 0.19922170, 0.33614049],
                        [0.09458646, 0.14840988, 0.04988729],
                        [0.23628263, 0.22587419, 0.44382286],
                        [0.13383963, 0.51702099, 0.40286142],
                        [0.70140973, 0.19925074, 0.02292392],
                        [0.06838428, 0.10600215, 0.37710859],
                        [0.55811797, 0.09062764, 0.12199424],
                        [0.10779019, 0.04434715, 0.14682113],
                        [0.34888054, 0.50195490, 0.04773998],
                        [0.79166868, 0.36502900, 0.02678776],
                        [0.02722027, 0.04781536, 0.30913913],
                        [0.06013188, 0.30558427, 0.06062012],
                        [0.44611192, 0.02849786, 0.04207225],
                        [0.85188200, 0.57960585, 0.01053590],
                        [0.50608734, 0.08898812, 0.29720873],
                        [-0.03338628, 0.24880620, 0.38541145],
                        [0.88687341, 0.88867240, 0.87460352],
                        [0.58637305, 0.58330907, 0.58216473],
                        [0.35827233, 0.35810703, 0.35873042],
                        [0.20316001, 0.20298624, 0.20353015],
                        [0.09106388, 0.09288101, 0.09424415],
                        [0.03266569, 0.03364008, 0.03526672],
                    ]
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        assert [attribute.name for attribute in attributes] == [
            "X",
            "Y",
            "Z",
            "illuminant",
            "chromaticities",
        ]

        for attribute in attributes:
            if attribute.name == "illuminant":
                sd_illuminant = spectrum_attribute_to_sd_Fichet2021(attribute.value)
                np.testing.assert_allclose(
                    sd_illuminant.values,
                    SDS_ILLUMINANTS["D65"].values,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )


def _test_spectral_image_D65(path):
    """Test the *D65* spectral image."""

    components = read_spectral_image_Fichet2021(path)

    assert "S0" in components

    np.testing.assert_allclose(
        components["S0"][0],
        SDS_ILLUMINANTS["D65"].wavelengths,
        atol=TOLERANCE_ABSOLUTE_TESTS,
    )

    np.testing.assert_allclose(
        components["S0"][1],
        np.reshape(SDS_ILLUMINANTS["D65"].values, (1, 1, -1)),
        atol=0.05,
    )

    components, specification = read_spectral_image_Fichet2021(
        os.path.join(ROOT_RESOURCES, "D65.exr"), additional_data=True
    )

    assert specification.is_emissive is True
    assert specification.is_polarised is False
    assert specification.is_bispectral is False

    attribute_names = [attribute.name for attribute in specification.attributes]

    for attribute_name in [
        "EV",
        "X",
        "Y",
        "Z",
        "chromaticities",
        "emissiveUnits",
        "illuminant",
        "polarisationHandedness",
        "spectralLayoutVersion",
    ]:
        assert attribute_name in attribute_names

    for attribute in specification.attributes:
        if attribute.name == "spectralLayoutVersion":
            assert attribute.value == "1.0"
        elif attribute.name == "polarisationHandedness":
            assert attribute.value == "right"
        elif attribute.name == "emissiveUnits":
            assert attribute.value == "W.m^-2.sr^-1"
        elif attribute.name == "illuminant":
            sd_illuminant = spectrum_attribute_to_sd_Fichet2021(attribute.value)
            np.testing.assert_allclose(
                sd_illuminant.values,
                SDS_ILLUMINANTS["D65"].values,
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )


def _test_spectral_image_Ohta1997(path):
    """Test the *Ohta (1997)* spectral image."""

    components, specification = read_spectral_image_Fichet2021(
        path, additional_data=True
    )

    assert "T" in components

    msds = sds_and_msds_to_msds(
        [
            sd.copy().align(SpectralShape(400, 700, 20))
            for sd in SDS_COLOURCHECKERS["ColorChecker N Ohta"].values()
        ]
    )

    np.testing.assert_allclose(
        components["T"][0],
        msds.wavelengths,
        atol=TOLERANCE_ABSOLUTE_TESTS,
    )

    np.testing.assert_allclose(
        components["T"][1],
        np.reshape(np.transpose(msds.values), (4, 6, -1)),
        atol=0.0005,
    )

    assert specification.is_emissive is False
    assert specification.is_polarised is False
    assert specification.is_bispectral is False


def _test_spectral_image_Polarised(path):
    """Test the *Polarised* spectral image."""

    components, specification = read_spectral_image_Fichet2021(
        path, additional_data=True
    )

    assert list(components.keys()) == ["S0", "S1", "S2", "S3"]

    assert specification.is_emissive is True
    assert specification.is_polarised is True
    assert specification.is_bispectral is False


def _test_spectral_image_BiSpectral(path):
    """Test the *Bi-Spectral* image."""

    components, specification = read_spectral_image_Fichet2021(
        path, additional_data=True
    )

    assert list(components.keys()) == [
        "T",
        380.0,
        390.0,
        400.0,
        410.0,
        420.0,
        430.0,
        440.0,
        450.0,
        460.0,
        470.0,
        480.0,
        490.0,
        500.0,
        510.0,
        520.0,
        530.0,
        540.0,
        550.0,
        560.0,
        570.0,
        580.0,
        590.0,
        600.0,
        610.0,
        620.0,
        630.0,
        640.0,
        650.0,
        660.0,
        670.0,
        680.0,
        690.0,
        700.0,
        710.0,
        720.0,
        730.0,
        740.0,
        750.0,
        760.0,
        770.0,
    ]

    assert specification.is_emissive is False
    assert specification.is_polarised is False
    assert specification.is_bispectral is True


class TestReadSpectralImageFichet2021:
    """
    Define :func:`colour.io.fichet2021.read_spectral_image_Fichet2021`
    definition unit tests methods.
    """

    def test_read_spectral_image_Fichet2021(self):
        """
        Test :func:`colour.io.fichet2021.read_spectral_image_Fichet2021`
        definition.
        """

        if not is_openimageio_installed():
            return

        _test_spectral_image_D65(os.path.join(ROOT_RESOURCES, "D65.exr"))

        _test_spectral_image_Ohta1997(os.path.join(ROOT_RESOURCES, "Ohta1997.exr"))

        _test_spectral_image_Polarised(os.path.join(ROOT_RESOURCES, "Polarised.exr"))

        _test_spectral_image_BiSpectral(os.path.join(ROOT_RESOURCES, "BiSpectral.exr"))


class TestWriteSpectralImageFichet2021:
    """
    Define :func:`colour.io.fichet2021.write_spectral_image_Fichet2021`
    definition unit tests methods.
    """

    def setup_method(self):
        """Initialise the common tests attributes."""

        self._temporary_directory = tempfile.mkdtemp()

    def teardown_method(self):
        """After tests actions."""

        shutil.rmtree(self._temporary_directory)

    def test_write_spectral_image_Fichet2021(self):
        """
        Test :func:`colour.io.fichet2021.write_spectral_image_Fichet2021`
        definition.
        """

        if not is_openimageio_installed():
            return

        path = os.path.join(self._temporary_directory, "D65.exr")
        specification = Specification_Fichet2021(is_emissive=True)
        write_spectral_image_Fichet2021(
            SDS_ILLUMINANTS["D65"], path, "float16", specification
        )
        _test_spectral_image_D65(path)

        path = os.path.join(self._temporary_directory, "D65.exr")
        msds = [
            sd.copy().align(SpectralShape(400, 700, 20))
            for sd in SDS_COLOURCHECKERS["ColorChecker N Ohta"].values()
        ]
        specification = Specification_Fichet2021(is_emissive=False)
        write_spectral_image_Fichet2021(
            msds, path, "float16", specification, shape=(4, 6, 16)
        )
        _test_spectral_image_Ohta1997(path)

        for basename, test_callable in [
            ("Polarised.exr", _test_spectral_image_Polarised),
            ("BiSpectral.exr", _test_spectral_image_BiSpectral),
        ]:
            components, specification = read_spectral_image_Fichet2021(
                os.path.join(ROOT_RESOURCES, basename), additional_data=True
            )
            path = os.path.join(self._temporary_directory, basename)
            write_spectral_image_Fichet2021(components, path, "float16", specification)
            test_callable(path)
