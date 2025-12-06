"""Define the unit tests for the :mod:`colour.plotting.phenomena` module."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from colour.plotting import (
    plot_multi_layer_stack,
    plot_multi_layer_thin_film,
    plot_ray,
    plot_single_layer_thin_film,
    plot_single_sd_rayleigh_scattering,
    plot_the_blue_sky,
    plot_thin_film_comparison,
    plot_thin_film_iridescence,
    plot_thin_film_reflectance_map,
    plot_thin_film_spectrum,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestPlotSingleSdRayleighScattering",
    "TestPlotTheBlueSky",
    "TestPlotSingleLayerThinFilm",
    "TestPlotMultiLayerThinFilm",
    "TestPlotThinFilmComparison",
    "TestPlotThinFilmSpectrum",
    "TestPlotThinFilmIridescence",
    "TestPlotThinFilmReflectanceMap",
    "TestPlotMultiLayerStack",
    "TestPlotRay",
]


class TestPlotSingleSdRayleighScattering:
    """
    Define :func:`colour.plotting.phenomena.\
plot_single_sd_rayleigh_scattering` definition unit tests methods.
    """

    def test_plot_single_sd_rayleigh_scattering(self) -> None:
        """
        Test :func:`colour.plotting.phenomena.\
plot_single_sd_rayleigh_scattering` definition.
        """

        figure, axes = plot_single_sd_rayleigh_scattering()

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)


class TestPlotTheBlueSky:
    """
    Define :func:`colour.plotting.phenomena.plot_the_blue_sky` definition unit
    tests methods.
    """

    def test_plot_the_blue_sky(self) -> None:
        """Test :func:`colour.plotting.phenomena.plot_the_blue_sky` definition."""

        figure, axes = plot_the_blue_sky()

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)


class TestPlotSingleLayerThinFilm:
    """
    Define :func:`colour.plotting.phenomena.plot_single_layer_thin_film`
    definition unit tests methods.
    """

    def test_plot_single_layer_thin_film(self) -> None:
        """
        Test :func:`colour.plotting.phenomena.plot_single_layer_thin_film`
        definition.
        """

        figure, axes = plot_single_layer_thin_film([1.0, 1.46, 1.5], 100)

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)


class TestPlotMultiLayerThinFilm:
    """
    Define :func:`colour.plotting.phenomena.plot_multi_layer_thin_film`
    definition unit tests methods.
    """

    def test_plot_multi_layer_thin_film(self) -> None:
        """
        Test :func:`colour.plotting.phenomena.plot_multi_layer_thin_film`
        definition.
        """

        figure, axes = plot_multi_layer_thin_film([1.0, 1.46, 2.4, 1.5], [100, 50])

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)

        figure, axes = plot_multi_layer_thin_film(
            [1.0, 1.46, 2.4, 1.5], [100, 50], method="Transmittance"
        )
        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)

        figure, axes = plot_multi_layer_thin_film(
            [1.0, 1.46, 2.4, 1.5], [100, 50], method="Both"
        )
        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)

        figure, axes = plot_multi_layer_thin_film(
            [1.0, 1.46, 2.4, 1.5], [100, 50], polarisation="S"
        )
        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)

        figure, axes = plot_multi_layer_thin_film(
            [1.0, 1.46, 2.4, 1.5], [100, 50], polarisation="P"
        )
        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)


class TestPlotThinFilmComparison:
    """
    Define :func:`colour.plotting.phenomena.plot_thin_film_comparison`
    definition unit tests methods.
    """

    def test_plot_thin_film_comparison(self) -> None:
        """
        Test :func:`colour.plotting.phenomena.plot_thin_film_comparison`
        definition.
        """

        configurations = [
            {
                "type": "single",
                "n_film": 1.46,
                "t": 100,
                "n_substrate": 1.5,
                "label": "MgF2 100nm",
            },
            {
                "type": "single",
                "n_film": 2.4,
                "t": 25,
                "n_substrate": 1.5,
                "label": "TiO2 25nm",
            },
        ]

        figure, axes = plot_thin_film_comparison(configurations)

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)

        configurations_multi = [
            {
                "type": "multilayer",
                "refractive_indices": [1.46, 2.4],
                "t": [100, 50],
                "n_substrate": 1.5,
                "label": "Multilayer",
            },
        ]

        figure, axes = plot_thin_film_comparison(configurations_multi)

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)

        # Test with invalid configuration type (should skip silently)
        configurations_invalid = [
            {
                "type": "single",
                "n_film": 1.46,
                "t": 100,
                "label": "Valid",
            },
            {
                "type": "invalid_type",
                "n_film": 1.5,
                "t": 50,
                "label": "Invalid (skipped)",
            },
        ]

        figure, axes = plot_thin_film_comparison(configurations_invalid)

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)


class TestPlotThinFilmSpectrum:
    """
    Define :func:`colour.plotting.phenomena.plot_thin_film_spectrum`
    definition unit tests methods.
    """

    def test_plot_thin_film_spectrum(self) -> None:
        """
        Test :func:`colour.plotting.phenomena.plot_thin_film_spectrum`
        definition.
        """

        figure, axes = plot_thin_film_spectrum([1.0, 1.33, 1.0], 200)

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)


class TestPlotThinFilmIridescence:
    """
    Define :func:`colour.plotting.phenomena.plot_thin_film_iridescence`
    definition unit tests methods.
    """

    def test_plot_thin_film_iridescence(self) -> None:
        """
        Test :func:`colour.plotting.phenomena.plot_thin_film_iridescence`
        definition.
        """

        figure, axes = plot_thin_film_iridescence([1.0, 1.33, 1.0])

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)


class TestPlotThinFilmReflectanceMap:
    """
    Define :func:`colour.plotting.phenomena.plot_thin_film_reflectance_map`
    definition unit tests methods.
    """

    def test_plot_thin_film_reflectance_map(self) -> None:
        """
        Test :func:`colour.plotting.phenomena.plot_thin_film_reflectance_map`
        definition.
        """

        figure, axes = plot_thin_film_reflectance_map(
            [1.0, 1.33, 1.0], method="Thickness"
        )

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)

        figure, axes = plot_thin_film_reflectance_map(
            [1.0, 1.33, 1.0], method="Thickness", polarisation="S"
        )

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)

        figure, axes = plot_thin_film_reflectance_map(
            [1.0, 1.33, 1.0], method="Thickness", polarisation="P"
        )

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)

        figure, axes = plot_thin_film_reflectance_map(
            [1.0, 1.33, 1.0], method="Angle", t=300, theta=np.linspace(0, 80, 20)
        )

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)

        figure, axes = plot_thin_film_reflectance_map(
            [1.0, 1.33, 1.0],
            method="Angle",
            t=300,
            theta=np.linspace(0, 80, 20),
            polarisation="S",
        )

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)

        figure, axes = plot_thin_film_reflectance_map(
            [1.0, 1.33, 1.0],
            method="Angle",
            t=300,
            theta=np.linspace(0, 80, 20),
            polarisation="P",
        )

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)

        # Test error: missing t in angle method
        with pytest.raises(ValueError, match="thickness 't' must be specified"):
            plot_thin_film_reflectance_map(
                [1.0, 1.33, 1.0],
                method="Angle",
                theta=np.linspace(0, 80, 20),
            )

        # Test error: missing theta in angle method
        with pytest.raises(ValueError, match="'theta' must be specified"):
            plot_thin_film_reflectance_map(
                [1.0, 1.33, 1.0],
                method="Angle",
                t=300,
            )

        # Test error: single angle in angle method
        with pytest.raises(ValueError, match="must be an array with multiple angles"):
            plot_thin_film_reflectance_map(
                [1.0, 1.33, 1.0],
                method="Angle",
                t=300,
                theta=45,
            )

        # Test multilayer with multiple thicknesses
        figure, axes = plot_thin_film_reflectance_map(
            [1.0, 1.33, 2.4, 1.0],
            method="Angle",
            t=[100, 50],
            theta=np.linspace(0, 80, 20),
        )

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)

        figure, axes = plot_thin_film_reflectance_map(
            [1.0, 1.46, 2.4, 1.5], method="Thickness", t=np.linspace(50, 500, 100)
        )

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)


class TestPlotMultiLayerStack:
    """
    Define :func:`colour.plotting.phenomena.plot_multi_layer_stack`
    definition unit tests methods.
    """

    def test_plot_multi_layer_stack(self) -> None:
        """
        Test :func:`colour.plotting.phenomena.plot_multi_layer_stack`
        definition.
        """

        configurations = [
            {"t": 100, "n": 1.46},
            {"t": 200, "n": 2.4},
            {"t": 80, "n": 1.46},
            {"t": 150, "n": 2.4},
        ]

        figure, axes = plot_multi_layer_stack(configurations, theta=45)

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)

        # Test error: empty configurations
        with pytest.raises(
            ValueError, match="At least one layer configuration is required"
        ):
            plot_multi_layer_stack([])

        # Test without theta
        figure, axes = plot_multi_layer_stack(configurations)

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)


class TestPlotRay:
    """
    Define :func:`colour.plotting.common.plot_ray` definition unit tests methods.
    """

    def test_plot_ray(self) -> None:
        """Test :func:`colour.plotting.common.plot_ray` definition."""

        figure, axes = plt.subplots()
        x = np.array([0, 1, 2])
        y = np.array([0, 1, 0])

        plot_ray(
            axes, x, y, style="solid", label="Ray", show_arrow=True, show_dots=True
        )

        plt.close(figure)

        # plot_ray returns None, so we just check it didn't raise an exception
        assert True
