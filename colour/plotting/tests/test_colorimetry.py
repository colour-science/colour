"""Define the unit tests for the :mod:`colour.plotting.colorimetry` module."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from colour.colorimetry import SpectralDistribution
from colour.plotting import (
    plot_blackbody_colours,
    plot_blackbody_spectral_radiance,
    plot_multi_cmfs,
    plot_multi_illuminant_sds,
    plot_multi_lightness_functions,
    plot_multi_luminance_functions,
    plot_multi_sds,
    plot_single_cmfs,
    plot_single_illuminant_sd,
    plot_single_lightness_function,
    plot_single_luminance_function,
    plot_single_sd,
    plot_visible_spectrum,
    plot_visible_spectrum_colours,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestPlotSingleSd",
    "TestPlotMultiSds",
    "TestPlotSingleCmfs",
    "TestPlotMultiCmfs",
    "TestPlotSingleIlluminantSd",
    "TestPlotMultiIlluminantSds",
    "TestPlotVisibleSpectrumColours",
    "TestPlotVisibleSpectrum",
    "TestPlotSingleLightnessFunction",
    "TestPlotMultiLightnessFunctions",
    "TestPlotSingleLuminanceFunction",
    "TestPlotMultiLuminanceFunctions",
    "TestPlotBlackbodySpectralRadiance",
    "TestPlotBlackbodyColours",
]


def _visible_spectrum_axes(figure: Figure, axes: Axes) -> Axes:
    return next(axis for axis in figure.axes if axis is not axes)


def _visible_spectrum_patches(
    axes: Axes, hatched: bool | None = None
) -> list[Rectangle]:
    strip_patches = [patch for patch in axes.patches if isinstance(patch, Rectangle)]

    if hatched is None:
        return strip_patches

    return [
        patch
        for patch in strip_patches
        if (patch.get_hatch() not in (None, "")) is hatched
    ]


class TestPlotSingleSd:
    """
    Define :func:`colour.plotting.colorimetry.plot_single_sd` definition unit
    tests methods.
    """

    def test_plot_single_sd(self) -> None:
        """Test :func:`colour.plotting.colorimetry.plot_single_sd` definition."""

        sd = SpectralDistribution(
            {
                500: 0.004900,
                510: 0.009300,
                520: 0.063270,
                530: 0.165500,
                540: 0.290400,
                550: 0.433450,
                560: 0.594500,
            },
            name="Custom 1",
        )

        figure, axes = plot_single_sd(
            sd,
            out_of_gamut_clipping=False,
            modulate_colours_with_sd_amplitude=True,
            equalize_sd_amplitude=True,
            show_visible_spectrum=True,
        )

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)
        assert len(figure.axes) == 2


class TestPlotMultiSds:
    """
    Define :func:`colour.plotting.colorimetry.plot_multi_sds` definition unit
    tests methods.
    """

    def test_plot_multi_sds(self) -> None:
        """Test :func:`colour.plotting.colorimetry.plot_multi_sds` definition."""

        sd_1 = SpectralDistribution(
            {
                500: 0.004900,
                510: 0.009300,
                520: 0.063270,
                530: 0.165500,
                540: 0.290400,
                550: 0.433450,
                560: 0.594500,
            },
            name="Custom 1",
        )
        sd_2 = SpectralDistribution(
            {
                500: 0.323000,
                510: 0.503000,
                520: 0.710000,
                530: 0.862000,
                540: 0.954000,
                550: 0.994950,
                560: 0.995000,
            },
            name="Custom 2",
        )

        figure, axes = plot_multi_sds(
            [sd_1, sd_2],
            plot_kwargs={
                "use_sd_colours": True,
                "normalise_sd_colours": True,
            },
            show_visible_spectrum=True,
        )

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)
        assert len(figure.axes) == 2

        figure, axes = plot_multi_sds(
            [sd_1, sd_2],
            plot_kwargs=[{"use_sd_colours": True, "normalise_sd_colours": True}] * 2,
        )

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)

    def test_plot_multi_sds_extended_domain_visible_spectrum(self) -> None:
        """
        Test :func:`colour.plotting.colorimetry.plot_multi_sds` definition
        with spectra extending outside the visible domain and wavelength
        colours enabled.
        """

        sd_1 = SpectralDistribution(
            {wavelength: 0.5 for wavelength in range(300, 1001, 10)},
            name="Extended",
        )
        sd_2 = SpectralDistribution(
            {wavelength: 0.25 for wavelength in range(400, 701, 10)},
            name="Visible",
        )

        figure, axes = plot_multi_sds(
            [sd_1, sd_2],
            show_visible_spectrum=True,
        )

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)

        visible_spectrum_axes = _visible_spectrum_axes(figure, axes)
        visible_strip_patches = _visible_spectrum_patches(
            visible_spectrum_axes, hatched=False
        )
        non_visible_strip_patches = _visible_spectrum_patches(
            visible_spectrum_axes, hatched=True
        )

        np.testing.assert_allclose(
            (
                min(patch.get_x() for patch in visible_strip_patches),
                max(
                    patch.get_x() + patch.get_width() for patch in visible_strip_patches
                ),
            ),
            (360, 830),
        )
        assert len(non_visible_strip_patches) == 2
        np.testing.assert_allclose(
            [
                (
                    patch.get_x(),
                    patch.get_x() + patch.get_width(),
                )
                for patch in non_visible_strip_patches
            ],
            [(300, 360), (830, 1000)],
        )


class TestPlotSingleCmfs:
    """
    Define :func:`colour.plotting.colorimetry.plot_single_cmfs` definition
    unit tests methods.
    """

    def test_plot_single_cmfs(self) -> None:
        """Test :func:`colour.plotting.colorimetry.plot_single_cmfs` definition."""

        figure, axes = plot_single_cmfs()

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)


class TestPlotMultiCmfs:
    """
    Define :func:`colour.plotting.colorimetry.plot_multi_cmfs` definition unit
    tests methods.
    """

    def test_plot_multi_cmfs(self) -> None:
        """Test :func:`colour.plotting.colorimetry.plot_multi_cmfs` definition."""

        figure, axes = plot_multi_cmfs(
            [
                "CIE 1931 2 Degree Standard Observer",
                "CIE 1964 10 Degree Standard Observer",
            ],
            show_visible_spectrum=True,
        )

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)
        assert len(figure.axes) == 2


class TestPlotSingleIlluminantSd:
    """
    Define :func:`colour.plotting.colorimetry.plot_single_illuminant_sd`
    definition unit tests methods.
    """

    def test_plot_single_illuminant_sd(self) -> None:
        """
        Test :func:`colour.plotting.colorimetry.plot_single_illuminant_sd`
        definition.
        """

        figure, axes = plot_single_illuminant_sd("A")

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)


class TestPlotMultiIlluminantSds:
    """
    Define :func:`colour.plotting.colorimetry.plot_multi_illuminant_sds`
    definition unit tests methods.
    """

    def test_plot_multi_illuminant_sds(self) -> None:
        """
        Test :func:`colour.plotting.colorimetry.plot_multi_illuminant_sds`
        definition.
        """

        figure, axes = plot_multi_illuminant_sds(["A", "B", "C"])

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)

        figure, axes = plot_multi_illuminant_sds(
            ["A", "B", "C"],
            plot_kwargs=[{"use_sd_colours": True, "normalise_sd_colours": True}] * 3,
        )

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)

        figure, axes = plot_multi_illuminant_sds(
            ["A", "B", "C"],
            show_visible_spectrum=True,
        )

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)
        assert len(figure.axes) == 2


class TestPlotVisibleSpectrumColours:
    """
    Define :func:`colour.plotting.colorimetry.plot_visible_spectrum_colours`
    definition unit tests methods.
    """

    def test_plot_visible_spectrum_colours(self) -> None:
        """
        Test :func:`colour.plotting.colorimetry.plot_visible_spectrum_colours`
        definition.
        """

        figure, axes = plot_visible_spectrum_colours(bounding_box=(300, 1000, 0, 1))

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)

        visible_strip_patches = _visible_spectrum_patches(axes, hatched=False)
        non_visible_strip_patches = _visible_spectrum_patches(axes, hatched=True)

        np.testing.assert_allclose(
            (
                min(patch.get_x() for patch in visible_strip_patches),
                max(
                    patch.get_x() + patch.get_width() for patch in visible_strip_patches
                ),
            ),
            (360, 830),
        )
        assert len(non_visible_strip_patches) == 2


class TestPlotVisibleSpectrum:
    """
    Define :func:`colour.plotting.colorimetry.plot_visible_spectrum`
    definition unit tests methods.
    """

    def test_plot_visible_spectrum(self) -> None:
        """
        Test :func:`colour.plotting.colorimetry.plot_visible_spectrum`
        definition.
        """

        figure, axes = plot_visible_spectrum()

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)


class TestPlotSingleLightnessFunction:
    """
    Define :func:`colour.plotting.colorimetry.plot_single_lightness_function`
    definition unit tests methods.
    """

    def test_plot_single_lightness_function(self) -> None:
        """
        Test :func:`colour.plotting.colorimetry.\
plot_single_lightness_function` definition.
        """

        figure, axes = plot_single_lightness_function("CIE 1976")

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)


class TestPlotMultiLightnessFunctions:
    """
    Define :func:`colour.plotting.colorimetry.plot_multi_lightness_functions`
    definition unit tests methods.
    """

    def test_plot_multi_lightness_functions(self) -> None:
        """
        Test :func:`colour.plotting.colorimetry.\
plot_multi_lightness_functions` definition.
        """

        figure, axes = plot_multi_lightness_functions(["CIE 1976", "Wyszecki 1963"])

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)


class TestPlotSingleLuminanceFunction:
    """
    Define :func:`colour.plotting.colorimetry.plot_single_luminance_function`
    definition unit tests methods.
    """

    def test_plot_single_luminance_function(self) -> None:
        """
        Test :func:`colour.plotting.colorimetry.\
plot_single_luminance_function` definition.
        """

        figure, axes = plot_single_luminance_function("CIE 1976")

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)


class TestPlotMultiLuminanceFunctions:
    """
    Define :func:`colour.plotting.colorimetry.plot_multi_luminance_functions`
    definition unit tests methods.
    """

    def test_plot_multi_luminance_functions(self) -> None:
        """
        Test :func:`colour.plotting.colorimetry.\
plot_multi_luminance_functions` definition.
        """

        figure, axes = plot_multi_luminance_functions(["CIE 1976", "Newhall 1943"])

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)


class TestPlotBlackbodySpectralRadiance:
    """
    Define :func:`colour.plotting.colorimetry.\
plot_blackbody_spectral_radiance` definition unit tests methods.
    """

    def test_plot_blackbody_spectral_radiance(self) -> None:
        """
        Test :func:`colour.plotting.colorimetry.\
plot_blackbody_spectral_radiance` definition.
        """

        figure, axes = plot_blackbody_spectral_radiance()

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)


class TestPlotBlackbodyColours:
    """
    Define :func:`colour.plotting.colorimetry.plot_blackbody_colours`
    definition unit tests methods.
    """

    def test_plot_blackbody_colours(self) -> None:
        """
        Test :func:`colour.plotting.colorimetry.plot_blackbody_colours`
        definition.
        """

        figure, axes = plot_blackbody_colours()

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)
