"""Define the unit tests for the :mod:`colour.phenomena.sky.wilkie2021` module."""

from __future__ import annotations

import os
from dataclasses import fields

import numpy as np
import pytest

from colour.phenomena.sky.wilkie2021 import (
    PATH_PRAGUE_SKY_MODEL_DATASET_GROUND,
    SkyDataset_Wilkie2021,
    SkyParameters_Wilkie2021,
    compute_sky_parameters_Wilkie2021,
    sky_radiance_Wilkie2021,
    sky_transmittance_Wilkie2021,
    sun_radiance_Wilkie2021,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestSkyDatasetWilkie2021",
    "TestSkyParametersWilkie2021",
    "TestComputeSkyParametersWilkie2021",
    "TestSkyRadianceWilkie2021",
    "TestSunRadianceWilkie2021",
    "TestSkyTransmittanceWilkie2021",
]

DATASET_PATH = PATH_PRAGUE_SKY_MODEL_DATASET_GROUND
DATASET_AVAILABLE = os.path.isfile(DATASET_PATH)

WAVELENGTHS = np.array([320.0, 380.0, 460.0, 540.0, 620.0, 700.0])

TEST_SKY_CONDITIONS = {
    "sun30_zenith45_opposite": {
        "view_point": [0, 0, 0],
        "view_direction": [
            np.sin(0.7854) * np.cos(3.1416),
            np.sin(0.7854) * np.sin(3.1416),
            np.cos(0.7854),
        ],
        "sun_elevation": 0.5236,
        "sun_azimuth": 0.0,
        "visibility": 50.0,
        "albedo": 0.5,
        "parameters": {
            "theta": 0.7854000000,
            "gamma": 1.8325963268,
            "shadow": 0.2590079347,
            "zero": 0.7826079346,
            "elevation": 0.5236000000,
            "altitude": 50.0,
        },
        "sky_radiance": [
            5.3107873172e-02,
            5.6294252337e-02,
            5.8964049089e-02,
            3.2599713676e-02,
            1.9265830095e-02,
            1.2149839803e-02,
        ],
        "transmittance": [
            2.6225886464e-01,
            4.0168201340e-01,
            5.9302223884e-01,
            6.8508113960e-01,
            7.4315036866e-01,
            8.2037396166e-01,
        ],
    },
    "sun30_zenith": {
        "view_point": [0, 0, 0],
        "view_direction": [0, 0, 1],
        "sun_elevation": 0.5236,
        "sun_azimuth": 0.0,
        "visibility": 50.0,
        "albedo": 0.5,
        "parameters": {
            "theta": 0.0000000000,
            "gamma": 1.0471963268,
            "shadow": 0.5236000000,
            "zero": 0.0000000000,
            "elevation": 0.5236000000,
            "altitude": 50.0,
        },
        "sky_radiance": [
            4.8186861721e-02,
            5.0425723522e-02,
            5.4639955885e-02,
            3.2298205668e-02,
            2.0553615445e-02,
            1.3661244550e-02,
        ],
        "transmittance": [
            3.8806037351e-01,
            5.2700578388e-01,
            6.9319275102e-01,
            7.6748312617e-01,
            8.1297748038e-01,
            8.6946032117e-01,
        ],
    },
    "sun30_horizon_toward": {
        "view_point": [0, 0, 0],
        "view_direction": [
            np.sin(1.5708) * np.cos(0.0),
            np.sin(1.5708) * np.sin(0.0),
            np.cos(1.5708),
        ],
        "sun_elevation": 0.5236,
        "sun_azimuth": 0.0,
        "visibility": 50.0,
        "albedo": 0.5,
        "parameters": {
            "theta": 1.5708000000,
            "gamma": 0.5236036732,
            "shadow": 2.0904403854,
            "zero": 1.5668403854,
            "elevation": 0.5236000000,
            "altitude": 50.0,
        },
        "sky_radiance": [
            8.1679884518e-02,
            1.4069905245e-01,
            3.3032219097e-01,
            3.3906134031e-01,
            3.1930071062e-01,
            2.9045508795e-01,
        ],
        "transmittance": [
            8.7494797010e-08,
            1.5371123578e-10,
            1.8661696957e-07,
            3.3047864522e-06,
            1.1604377129e-05,
            1.8534990891e-04,
        ],
    },
    "sunset_zenith45": {
        "view_point": [0, 0, 0],
        "view_direction": [
            np.sin(0.7854) * np.cos(0.0),
            np.sin(0.7854) * np.sin(0.0),
            np.cos(0.7854),
        ],
        "sun_elevation": -0.0500,
        "sun_azimuth": 0.0,
        "visibility": 50.0,
        "albedo": 0.5,
        "parameters": {
            "theta": 0.7854000000,
            "gamma": 0.8353963268,
            "shadow": 0.7326079346,
            "zero": 0.7826079346,
            "elevation": -0.0500000000,
            "altitude": 50.0,
        },
        "sky_radiance": [
            3.7268466788e-04,
            6.3494761833e-04,
            1.0555130013e-03,
            5.1896359629e-04,
            3.5488616557e-04,
            5.3453233851e-04,
        ],
        "transmittance": [
            2.6225886464e-01,
            4.0168201340e-01,
            5.9302223884e-01,
            6.8508113960e-01,
            7.4315036866e-01,
            8.2037396166e-01,
        ],
    },
    "altitude100_vis100_alb02": {
        "view_point": [0, 0, 100],
        "view_direction": [
            np.sin(0.7854) * np.cos(3.1416),
            np.sin(0.7854) * np.sin(3.1416),
            np.cos(0.7854),
        ],
        "sun_elevation": 0.5236,
        "sun_azimuth": 0.0,
        "visibility": 100.0,
        "albedo": 0.2,
        "parameters": {
            "theta": 0.7854000000,
            "gamma": 1.8325963268,
            "shadow": 0.2569739464,
            "zero": 0.7805739464,
            "elevation": 0.5236000000,
            "altitude": 150.0,
        },
        "sky_radiance": [
            4.4284411690e-02,
            4.5185516640e-02,
            4.3990858779e-02,
            2.2686369986e-02,
            1.2535600923e-02,
            7.4289074652e-03,
        ],
        "transmittance": [
            3.2105589852e-01,
            4.8018432995e-01,
            6.8234902643e-01,
            7.6644676630e-01,
            8.1459429167e-01,
            8.8573679226e-01,
        ],
    },
}


class TestSkyDatasetWilkie2021:
    """
    Define :class:`colour.phenomena.sky.wilkie2021.SkyDataset_Wilkie2021`
    class unit tests methods.
    """

    def test_required_attributes(self) -> None:
        """Test the presence of required attributes."""

        required_attributes = (
            "channels",
            "channel_start",
            "channel_width",
            "visibilities_radiance",
            "albedos_radiance",
            "altitudes_radiance",
            "elevations_radiance",
            "metadata_radiance",
            "data_radiance",
            "metadata_polarisation",
            "data_polarisation",
            "altitude_dimension",
            "distance_dimension",
            "rank_transmittance",
            "altitudes_transmittance",
            "visibilities_transmittance",
            "data_transmittance_u",
            "data_transmittance_v",
        )

        field_names = {f.name for f in fields(SkyDataset_Wilkie2021)}
        for attribute in required_attributes:
            assert attribute in field_names

    @pytest.mark.skipif(
        not DATASET_AVAILABLE,
        reason=f"Prague Sky Model dataset not found at {DATASET_PATH}",
    )
    def test_read(self) -> None:
        """
        Test :meth:`colour.phenomena.sky.wilkie2021.\
SkyDataset_Wilkie2021.read` method.
        """

        dataset = SkyDataset_Wilkie2021(DATASET_PATH)

        assert isinstance(dataset, SkyDataset_Wilkie2021)
        assert dataset.channels == 11
        np.testing.assert_allclose(dataset.channel_start, 320.0)
        np.testing.assert_allclose(dataset.channel_width, 40.0)
        assert len(dataset.visibilities_radiance) >= 1
        assert len(dataset.albedos_radiance) >= 1
        assert len(dataset.altitudes_radiance) >= 1
        assert len(dataset.elevations_radiance) >= 1
        assert len(dataset.data_radiance) > 0
        assert len(dataset.data_transmittance_u) > 0
        assert len(dataset.data_transmittance_v) > 0

    def test_read_file_not_found(self) -> None:
        """
        Test :meth:`colour.phenomena.sky.wilkie2021.\
SkyDataset_Wilkie2021.read` method with a missing file.
        """

        with pytest.raises(FileNotFoundError):
            SkyDataset_Wilkie2021("/nonexistent/path.dat")


class TestSkyParametersWilkie2021:
    """
    Define :class:`colour.phenomena.sky.wilkie2021.SkyParameters_Wilkie2021`
    class unit tests methods.
    """

    def test_required_attributes(self) -> None:
        """Test the presence of required attributes."""

        required_attributes = (
            "theta",
            "gamma",
            "shadow",
            "zero",
            "elevation",
            "altitude",
            "visibility",
            "albedo",
        )

        field_names = {f.name for f in fields(SkyParameters_Wilkie2021)}
        for attribute in required_attributes:
            assert attribute in field_names


@pytest.mark.skipif(
    not DATASET_AVAILABLE,
    reason=f"Prague Sky Model dataset not found at {DATASET_PATH}",
)
class TestComputeSkyParametersWilkie2021:
    """
    Define :func:`colour.phenomena.sky.wilkie2021.\
compute_sky_parameters_Wilkie2021` definition unit tests methods.
    """

    def test_compute_sky_parameters_Wilkie2021(self) -> None:
        """
        Test :func:`colour.phenomena.sky.wilkie2021.\
compute_sky_parameters_Wilkie2021` definition.
        """

        for name, condition in TEST_SKY_CONDITIONS.items():
            parameters = compute_sky_parameters_Wilkie2021(
                np.array(condition["view_point"], dtype=float),
                np.array(condition["view_direction"], dtype=float),
                condition["sun_elevation"],
                condition["sun_azimuth"],
                condition["visibility"],
                condition["albedo"],
            )

            reference = condition["parameters"]
            np.testing.assert_allclose(
                parameters.theta,
                reference["theta"],
                atol=1e-6,
                err_msg=f"{name}: theta",
            )
            np.testing.assert_allclose(
                parameters.gamma,
                reference["gamma"],
                atol=1e-6,
                err_msg=f"{name}: gamma",
            )
            np.testing.assert_allclose(
                parameters.shadow,
                reference["shadow"],
                atol=1e-6,
                err_msg=f"{name}: shadow",
            )
            np.testing.assert_allclose(
                parameters.zero,
                reference["zero"],
                atol=1e-6,
                err_msg=f"{name}: zero",
            )
            np.testing.assert_allclose(
                parameters.elevation,
                reference["elevation"],
                atol=1e-6,
                err_msg=f"{name}: elevation",
            )
            np.testing.assert_allclose(
                parameters.altitude,
                reference["altitude"],
                atol=1e-3,
                err_msg=f"{name}: altitude",
            )


@pytest.mark.skipif(
    not DATASET_AVAILABLE,
    reason=f"Prague Sky Model dataset not found at {DATASET_PATH}",
)
class TestSkyRadianceWilkie2021:
    """
    Define :func:`colour.phenomena.sky.wilkie2021.sky_radiance_Wilkie2021`
    definition unit tests methods.
    """

    def test_sky_radiance_Wilkie2021(self) -> None:
        """
        Test :func:`colour.phenomena.sky.wilkie2021.\
sky_radiance_Wilkie2021` definition.
        """

        dataset = SkyDataset_Wilkie2021(DATASET_PATH)

        for name, condition in TEST_SKY_CONDITIONS.items():
            parameters = compute_sky_parameters_Wilkie2021(
                np.array(condition["view_point"], dtype=float),
                np.array(condition["view_direction"], dtype=float),
                condition["sun_elevation"],
                condition["sun_azimuth"],
                condition["visibility"],
                condition["albedo"],
            )

            result = sky_radiance_Wilkie2021(dataset, parameters, WAVELENGTHS)
            reference = np.array(condition["sky_radiance"])

            np.testing.assert_allclose(
                result,
                reference,
                rtol=1e-5,
                err_msg=f"{name}: sky_radiance",
            )

    def test_sky_radiance_Wilkie2021_out_of_range(self) -> None:
        """
        Test :func:`colour.phenomena.sky.wilkie2021.\
sky_radiance_Wilkie2021` definition with out-of-range wavelengths.
        """

        dataset = SkyDataset_Wilkie2021(DATASET_PATH)

        parameters = compute_sky_parameters_Wilkie2021(
            np.array([0, 0, 0.0]),
            np.array([0, 0, 1.0]),
            0.5236,
            0.0,
            50.0,
            0.5,
        )
        result = sky_radiance_Wilkie2021(dataset, parameters, np.array([100.0, 5000.0]))
        np.testing.assert_array_equal(result, 0.0)

    def test_n_dimensional_sky_radiance_Wilkie2021(self) -> None:
        """
        Test :func:`colour.phenomena.sky.wilkie2021.\
sky_radiance_Wilkie2021` definition n-dimensional support.
        """

        dataset = SkyDataset_Wilkie2021(DATASET_PATH)

        parameters = compute_sky_parameters_Wilkie2021(
            np.array([0, 0, 0.0]),
            np.array([0, 0, 1.0]),
            0.5236,
            0.0,
            50.0,
            0.5,
        )
        result = sky_radiance_Wilkie2021(dataset, parameters, WAVELENGTHS)
        assert result.shape == (6,)

        directions = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
        parameters = compute_sky_parameters_Wilkie2021(
            np.array([0, 0, 0.0]),
            directions,
            0.5236,
            0.0,
            50.0,
            0.5,
        )
        result = sky_radiance_Wilkie2021(dataset, parameters, WAVELENGTHS)
        assert result.shape == (3, 6)


@pytest.mark.skipif(
    not DATASET_AVAILABLE,
    reason=f"Prague Sky Model dataset not found at {DATASET_PATH}",
)
class TestSunRadianceWilkie2021:
    """
    Define :func:`colour.phenomena.sky.wilkie2021.sun_radiance_Wilkie2021`
    definition unit tests methods.
    """

    def test_sun_radiance_Wilkie2021(self) -> None:
        """
        Test :func:`colour.phenomena.sky.wilkie2021.\
sun_radiance_Wilkie2021` definition.
        """

        dataset = SkyDataset_Wilkie2021(DATASET_PATH)

        for name, condition in TEST_SKY_CONDITIONS.items():
            parameters = compute_sky_parameters_Wilkie2021(
                np.array(condition["view_point"], dtype=float),
                np.array(condition["view_direction"], dtype=float),
                condition["sun_elevation"],
                condition["sun_azimuth"],
                condition["visibility"],
                condition["albedo"],
            )

            result = sun_radiance_Wilkie2021(dataset, parameters, WAVELENGTHS)
            np.testing.assert_array_equal(result, 0.0, err_msg=f"{name}: sun_radiance")

    def test_sun_radiance_Wilkie2021_toward_sun(self) -> None:
        """
        Test :func:`colour.phenomena.sky.wilkie2021.\
sun_radiance_Wilkie2021` definition when looking at the sun.
        """

        dataset = SkyDataset_Wilkie2021(DATASET_PATH)

        sun_direction = np.array(
            [
                np.cos(0.0) * np.cos(0.5236),
                np.sin(0.0) * np.cos(0.5236),
                np.sin(0.5236),
            ]
        )
        parameters = compute_sky_parameters_Wilkie2021(
            np.array([0, 0, 0.0]),
            sun_direction,
            0.5236,
            0.0,
            50.0,
            0.5,
        )
        result = sun_radiance_Wilkie2021(dataset, parameters, WAVELENGTHS)
        assert np.all(result[:6] > 0)

    def test_n_dimensional_sun_radiance_Wilkie2021(self) -> None:
        """
        Test :func:`colour.phenomena.sky.wilkie2021.\
sun_radiance_Wilkie2021` definition n-dimensional support.
        """

        dataset = SkyDataset_Wilkie2021(DATASET_PATH)

        directions = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
        parameters = compute_sky_parameters_Wilkie2021(
            np.array([0, 0, 0.0]),
            directions,
            0.5236,
            0.0,
            50.0,
            0.5,
        )
        result = sun_radiance_Wilkie2021(dataset, parameters, WAVELENGTHS)
        assert result.shape == (3, 6)
        np.testing.assert_array_equal(result, 0.0)


@pytest.mark.skipif(
    not DATASET_AVAILABLE,
    reason=f"Prague Sky Model dataset not found at {DATASET_PATH}",
)
class TestSkyTransmittanceWilkie2021:
    """
    Define :func:`colour.phenomena.sky.wilkie2021.\
sky_transmittance_Wilkie2021` definition unit tests methods.
    """

    def test_sky_transmittance_Wilkie2021(self) -> None:
        """
        Test :func:`colour.phenomena.sky.wilkie2021.\
sky_transmittance_Wilkie2021` definition.
        """

        dataset = SkyDataset_Wilkie2021(DATASET_PATH)

        for name, condition in TEST_SKY_CONDITIONS.items():
            parameters = compute_sky_parameters_Wilkie2021(
                np.array(condition["view_point"], dtype=float),
                np.array(condition["view_direction"], dtype=float),
                condition["sun_elevation"],
                condition["sun_azimuth"],
                condition["visibility"],
                condition["albedo"],
            )

            result = sky_transmittance_Wilkie2021(
                dataset, parameters, WAVELENGTHS, np.inf
            )
            reference = np.array(condition["transmittance"])

            np.testing.assert_allclose(
                result,
                reference,
                rtol=1e-5,
                err_msg=f"{name}: transmittance",
            )

    def test_sky_transmittance_Wilkie2021_bounded(self) -> None:
        """
        Test :func:`colour.phenomena.sky.wilkie2021.\
sky_transmittance_Wilkie2021` definition values are in [0, 1].
        """

        dataset = SkyDataset_Wilkie2021(DATASET_PATH)

        parameters = compute_sky_parameters_Wilkie2021(
            np.array([0, 0, 0.0]),
            np.array([0, 0, 1.0]),
            0.5236,
            0.0,
            50.0,
            0.5,
        )
        result = sky_transmittance_Wilkie2021(dataset, parameters, WAVELENGTHS, np.inf)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_sky_transmittance_Wilkie2021_out_of_range(self) -> None:
        """
        Test :func:`colour.phenomena.sky.wilkie2021.\
sky_transmittance_Wilkie2021` definition with out-of-range wavelengths.
        """

        dataset = SkyDataset_Wilkie2021(DATASET_PATH)

        parameters = compute_sky_parameters_Wilkie2021(
            np.array([0, 0, 0.0]),
            np.array([0, 0, 1.0]),
            0.5236,
            0.0,
            50.0,
            0.5,
        )
        result = sky_transmittance_Wilkie2021(
            dataset, parameters, np.array([100.0, 5000.0]), np.inf
        )
        np.testing.assert_array_equal(result, 0.0)

    def test_n_dimensional_sky_transmittance_Wilkie2021(self) -> None:
        """
        Test :func:`colour.phenomena.sky.wilkie2021.\
sky_transmittance_Wilkie2021` definition n-dimensional support.
        """

        dataset = SkyDataset_Wilkie2021(DATASET_PATH)

        directions = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
        parameters = compute_sky_parameters_Wilkie2021(
            np.array([0, 0, 0.0]),
            directions,
            0.5236,
            0.0,
            50.0,
            0.5,
        )
        result = sky_transmittance_Wilkie2021(dataset, parameters, WAVELENGTHS, np.inf)
        assert result.shape == (3, 6)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)
