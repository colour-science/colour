"""
Optical Phenomenon Plotting
===========================

Defines the optical phenomena plotting objects:

-   :func:`colour.plotting.plot_single_sd_rayleigh_scattering`
-   :func:`colour.plotting.plot_the_blue_sky`
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from colour.algebra import normalise_maximum
from colour.colorimetry import (
    MultiSpectralDistributions,
    SpectralDistribution,
    sd_to_XYZ,
)
from colour.hints import (
    Any,
    Dict,
    FloatingOrArrayLike,
    Sequence,
    Tuple,
    Union,
    cast,
)
from colour.phenomena import sd_rayleigh_scattering
from colour.phenomena.rayleigh import (
    CONSTANT_AVERAGE_PRESSURE_MEAN_SEA_LEVEL,
    CONSTANT_DEFAULT_ALTITUDE,
    CONSTANT_DEFAULT_LATITUDE,
    CONSTANT_STANDARD_AIR_TEMPERATURE,
    CONSTANT_STANDARD_CO2_CONCENTRATION,
)
from colour.plotting import (
    SD_ASTMG173_ETR,
    CONSTANTS_COLOUR_STYLE,
    ColourSwatch,
    XYZ_to_plotting_colourspace,
    filter_cmfs,
    override_style,
    render,
    plot_single_colour_swatch,
    plot_single_sd,
)
from colour.utilities import first_item

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "plot_single_sd_rayleigh_scattering",
    "plot_the_blue_sky",
]


@override_style()
def plot_single_sd_rayleigh_scattering(
    CO2_concentration: FloatingOrArrayLike = CONSTANT_STANDARD_CO2_CONCENTRATION,
    temperature: FloatingOrArrayLike = CONSTANT_STANDARD_AIR_TEMPERATURE,
    pressure: FloatingOrArrayLike = CONSTANT_AVERAGE_PRESSURE_MEAN_SEA_LEVEL,
    latitude: FloatingOrArrayLike = CONSTANT_DEFAULT_LATITUDE,
    altitude: FloatingOrArrayLike = CONSTANT_DEFAULT_ALTITUDE,
    cmfs: Union[
        MultiSpectralDistributions,
        str,
        Sequence[Union[MultiSpectralDistributions, str]],
    ] = "CIE 1931 2 Degree Standard Observer",
    **kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a single *Rayleigh* scattering spectral distribution.

    Parameters
    ----------
    CO2_concentration
        :math:`CO_2` concentration in parts per million (ppm).
    temperature
        Air temperature :math:`T[K]` in kelvin degrees.
    pressure
        Surface pressure :math:`P` of the measurement site.
    latitude
        Latitude of the site in degrees.
    altitude
        Altitude of the site in meters.
    cmfs
        Standard observer colour matching functions used for computing the
        spectrum domain and colours. ``cmfs`` can be of any type or form
        supported by the :func:`colour.plotting.filter_cmfs` definition.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.plot_single_sd`,
        :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> plot_single_sd_rayleigh_scattering()  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_Single_SD_Rayleigh_Scattering.png
        :align: center
        :alt: plot_single_sd_rayleigh_scattering
    """

    title = "Rayleigh Scattering"

    cmfs = cast(
        MultiSpectralDistributions, first_item(filter_cmfs(cmfs).values())
    )

    settings: Dict[str, Any] = {"title": title, "y_label": "Optical Depth"}
    settings.update(kwargs)

    sd = sd_rayleigh_scattering(
        cmfs.shape,
        CO2_concentration,
        temperature,
        pressure,
        latitude,
        altitude,
    )

    return plot_single_sd(sd, **settings)


@override_style()
def plot_the_blue_sky(
    cmfs: Union[
        MultiSpectralDistributions,
        str,
        Sequence[Union[MultiSpectralDistributions, str]],
    ] = "CIE 1931 2 Degree Standard Observer",
    **kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the blue sky.

    Parameters
    ----------
    cmfs
        Standard observer colour matching functions used for computing the
        spectrum domain and colours. ``cmfs`` can be of any type or form
        supported by the :func:`colour.plotting.filter_cmfs` definition.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.plot_single_sd`,
        :func:`colour.plotting.plot_multi_colour_swatches`,
        :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> plot_the_blue_sky()  # doctest: +ELLIPSIS
    (<Figure size ... with 2 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_The_Blue_Sky.png
        :align: center
        :alt: plot_the_blue_sky
    """

    figure = plt.figure()

    figure.subplots_adjust(hspace=CONSTANTS_COLOUR_STYLE.geometry.short / 2)

    cmfs = cast(
        MultiSpectralDistributions, first_item(filter_cmfs(cmfs).values())
    )

    ASTMG173_sd = cast(SpectralDistribution, SD_ASTMG173_ETR.copy())
    rayleigh_sd = sd_rayleigh_scattering()
    ASTMG173_sd.align(rayleigh_sd.shape)

    sd = rayleigh_sd * ASTMG173_sd

    axes = figure.add_subplot(211)

    settings: Dict[str, Any] = {
        "axes": axes,
        "title": "The Blue Sky - Synthetic Spectral Distribution",
        "y_label": "W / m-2 / nm-1",
    }
    settings.update(kwargs)
    settings["standalone"] = False

    plot_single_sd(sd, cmfs, **settings)

    axes = figure.add_subplot(212)

    x_label = (
        "The sky is blue because molecules in the atmosphere "
        "scatter shorter wavelengths more than longer ones.\n"
        "The synthetic spectral distribution is computed as "
        "follows: "
        "(ASTM G-173 ETR * Standard Air Rayleigh Scattering)."
    )

    settings = {
        "axes": axes,
        "aspect": None,
        "title": "The Blue Sky - Colour",
        "x_label": x_label,
        "y_label": "",
        "x_ticker": False,
        "y_ticker": False,
    }
    settings.update(kwargs)
    settings["standalone"] = False

    blue_sky_color = XYZ_to_plotting_colourspace(
        sd_to_XYZ(cast(SpectralDistribution, sd))
    )

    figure, axes = plot_single_colour_swatch(
        ColourSwatch(normalise_maximum(blue_sky_color)), **settings
    )

    settings = {"axes": axes, "standalone": True}
    settings.update(kwargs)

    return render(**settings)
