"""
Optical Phenomenon Plotting
===========================

Define the optical phenomena plotting objects.

-   :func:`colour.plotting.plot_single_sd_rayleigh_scattering`
-   :func:`colour.plotting.plot_the_blue_sky`
"""

from __future__ import annotations

import typing

import matplotlib.pyplot as plt
import numpy as np

if typing.TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes

from colour.algebra import normalise_maximum
from colour.colorimetry import (
    SPECTRAL_SHAPE_DEFAULT,
    MultiSpectralDistributions,
    SpectralDistribution,
    SpectralShape,
    msds_to_XYZ,
    sd_to_XYZ,
)

if typing.TYPE_CHECKING:
    from colour.hints import Any, ArrayLike, Dict, Literal, Sequence, Tuple

from colour.hints import cast
from colour.phenomena import sd_rayleigh_scattering
from colour.phenomena.interference import (
    multilayer_tmm,
    thin_film_tmm,
)
from colour.phenomena.rayleigh import (
    CONSTANT_AVERAGE_PRESSURE_MEAN_SEA_LEVEL,
    CONSTANT_DEFAULT_ALTITUDE,
    CONSTANT_DEFAULT_LATITUDE,
    CONSTANT_STANDARD_AIR_TEMPERATURE,
    CONSTANT_STANDARD_CO2_CONCENTRATION,
)
from colour.phenomena.tmm import matrix_transfer_tmm
from colour.plotting import (
    CONSTANTS_COLOUR_STYLE,
    SD_ASTMG173_ETR,
    ColourSwatch,
    XYZ_to_plotting_colourspace,
    artist,
    colour_cycle,
    filter_cmfs,
    filter_illuminants,
    override_style,
    plot_ray,
    plot_single_colour_swatch,
    plot_single_sd,
    render,
)
from colour.utilities import (
    as_complex_array,
    as_float_array,
    as_float_scalar,
    first_item,
    optional,
    validate_method,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "plot_single_sd_rayleigh_scattering",
    "plot_the_blue_sky",
    "plot_single_layer_thin_film",
    "plot_multi_layer_thin_film",
    "plot_thin_film_comparison",
    "plot_thin_film_spectrum",
    "plot_thin_film_iridescence",
    "plot_thin_film_reflectance_map",
    "plot_multi_layer_stack",
]


@override_style()
def plot_single_sd_rayleigh_scattering(
    CO2_concentration: ArrayLike = CONSTANT_STANDARD_CO2_CONCENTRATION,
    temperature: ArrayLike = CONSTANT_STANDARD_AIR_TEMPERATURE,
    pressure: ArrayLike = CONSTANT_AVERAGE_PRESSURE_MEAN_SEA_LEVEL,
    latitude: ArrayLike = CONSTANT_DEFAULT_LATITUDE,
    altitude: ArrayLike = CONSTANT_DEFAULT_ALTITUDE,
    cmfs: (
        MultiSpectralDistributions | str | Sequence[MultiSpectralDistributions | str]
    ) = "CIE 1931 2 Degree Standard Observer",
    **kwargs: Any,
) -> Tuple[Figure, Axes]:
    """
    Plot a single *Rayleigh* scattering spectral distribution.

    Parameters
    ----------
    CO2_concentration
        :math:`CO_2` concentration in parts per million (ppm).
    temperature
        Air temperature :math:`T[K]` in kelvin degrees.
    pressure
        Surface pressure :math:`P` at the measurement site.
    latitude
        Latitude of the site in degrees.
    altitude
        Altitude of the site in meters.
    cmfs
        Standard observer colour matching functions used for computing
        the spectrum domain and colours. ``cmfs`` can be of any type or
        form supported by the :func:`colour.plotting.common.filter_cmfs`
        definition.

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
    (<Figure size ... with 1 Axes>, <...Axes...>)

    .. image:: ../_static/Plotting_Plot_Single_SD_Rayleigh_Scattering.png
        :align: center
        :alt: plot_single_sd_rayleigh_scattering
    """

    title = "Rayleigh Scattering"

    cmfs = cast("MultiSpectralDistributions", first_item(filter_cmfs(cmfs).values()))

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
    cmfs: (
        MultiSpectralDistributions | str | Sequence[MultiSpectralDistributions | str]
    ) = "CIE 1931 2 Degree Standard Observer",
    **kwargs: Any,
) -> Tuple[Figure, Axes]:
    """
    Plot the blue sky spectral radiance distribution.

    Parameters
    ----------
    cmfs
        Standard observer colour matching functions used for computing the
        spectrum domain and colours. ``cmfs`` can be of any type or form
        supported by the :func:`colour.plotting.common.filter_cmfs`
        definition.

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
    (<Figure size ... with 2 Axes>, <...Axes...>)

    .. image:: ../_static/Plotting_Plot_The_Blue_Sky.png
        :align: center
        :alt: plot_the_blue_sky
    """

    figure = plt.figure()

    figure.subplots_adjust(hspace=CONSTANTS_COLOUR_STYLE.geometry.short / 2)

    cmfs = cast("MultiSpectralDistributions", first_item(filter_cmfs(cmfs).values()))

    ASTMG173_sd = SD_ASTMG173_ETR.copy()
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
    settings["show"] = False

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
    settings["show"] = False

    blue_sky_color = XYZ_to_plotting_colourspace(sd_to_XYZ(sd))

    figure, axes = plot_single_colour_swatch(
        ColourSwatch(normalise_maximum(blue_sky_color)), **settings
    )

    settings = {"axes": axes, "show": True}
    settings.update(kwargs)

    return render(**settings)


@override_style()
def plot_single_layer_thin_film(
    n: ArrayLike,
    t: ArrayLike,
    theta: ArrayLike = 0,
    shape: SpectralShape = SPECTRAL_SHAPE_DEFAULT,
    polarisation: Literal["S", "P", "Both"] | str = "Both",
    method: Literal["Reflectance", "Transmittance", "Both"] | str = "Reflectance",
    **kwargs: Any,
) -> Tuple[Figure, Axes]:
    """
    Plot reflectance and/or transmittance of a single-layer thin film.

    Parameters
    ----------
    n
        Complete refractive index stack :math:`n_j` for single-layer film.
        Shape: (3,) or (3, wavelengths_count). The array should contain
        [n_incident, n_film, n_substrate].
    t
        Thickness :math:`t` of the film in nanometers.
    theta
        Incident angle :math:`\\theta` in degrees. Default is 0 (normal incidence).
    shape
        Spectral shape for wavelength sampling.
    polarisation
        Polarisation to plot: 'S', 'P', or 'Both' (case-insensitive).
    method
        Optical property to plot: 'Reflectance', 'Transmittance', or 'Both'
        (case-insensitive). Default is 'Reflectance'.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.plot_multi_layer_thin_film`,
        :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> plot_single_layer_thin_film([1.0, 1.46, 1.5], 100)  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...Axes...>)

    .. image:: ../_static/Plotting_Plot_Single_Layer_Thin_Film.png
        :align: center
        :alt: plot_single_layer_thin_film
    """

    n = as_complex_array(n)
    t_array = as_float_array(t)
    n_layer = n[1] if n.ndim == 1 else n[1, 0]
    t_scalar = t_array if t_array.ndim == 0 else t_array[0]
    title = (
        f"Single Layer Thin Film (n={np.real(n_layer):.2f}, "
        f"d={t_scalar:.0f} nm, θ={theta}°)"
    )

    settings: Dict[str, Any] = {"title": title}
    settings.update(kwargs)

    return plot_multi_layer_thin_film(
        n, np.atleast_1d(t_array), theta, shape, polarisation, method, **settings
    )


@override_style()
def plot_multi_layer_thin_film(
    n: ArrayLike,
    t: ArrayLike,
    theta: ArrayLike = 0,
    shape: SpectralShape = SPECTRAL_SHAPE_DEFAULT,
    polarisation: Literal["S", "P", "Both"] | str = "Both",
    method: Literal["Reflectance", "Transmittance", "Both"] | str = "Reflectance",
    **kwargs: Any,
) -> Tuple[Figure, Axes]:
    """
    Plot reflectance and/or transmittance of a multi-layer thin film stack.

    Parameters
    ----------
    n
        Complete refractive index stack :math:`n_j` including incident medium,
        layers, and substrate. Shape: (media_count,) or
        (media_count, wavelengths_count). The array should contain
        [n_incident, n_layer_1, ..., n_layer_n, n_substrate].
    t
        Thicknesses :math:`t_j` of the layers in nanometers (excluding incident
        and substrate). Shape: (layers_count,).
    theta
        Incident angle :math:`\\theta` in degrees. Default is 0 (normal incidence).
    shape
        Spectral shape for wavelength sampling.
    polarisation
        Polarisation to plot: 'S', 'P', or 'Both' (case-insensitive).
    method
        Optical property to plot: 'Reflectance', 'Transmittance', or 'Both'
        (case-insensitive). Default is 'Reflectance'.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> plot_multi_layer_thin_film(
    ...     [1.0, 1.46, 2.4, 1.5], [100, 50]
    ... )  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...Axes...>)

    .. image:: ../_static/Plotting_Plot_Multi_Layer_Thin_Film.png
        :align: center
        :alt: plot_multi_layer_thin_film
    """

    n = as_complex_array(n)
    t = as_float_array(t)

    _figure, axes = artist(**kwargs)

    wavelengths = shape.wavelengths

    polarisation = validate_method(polarisation, ("S", "P", "Both"))
    method = validate_method(method, ("Reflectance", "Transmittance", "Both"))

    R, T = multilayer_tmm(n, t, wavelengths, theta)

    if method in ["reflectance", "both"]:
        if polarisation in ["s", "both"]:
            axes.plot(wavelengths, R[:, 0, 0, 0], "b-", label="R (s-pol)", linewidth=2)

        if polarisation in ["p", "both"]:
            axes.plot(wavelengths, R[:, 0, 0, 1], "r--", label="R (p-pol)", linewidth=2)

    if method in ["transmittance", "both"]:
        if polarisation in ["s", "both"]:
            axes.plot(
                wavelengths,
                T[:, 0, 0, 0],
                "b:",
                label="T (s-pol)",
                linewidth=2,
                alpha=0.7,
            )
        if polarisation in ["p", "both"]:
            axes.plot(
                wavelengths,
                T[:, 0, 0, 1],
                "r:",
                label="T (p-pol)",
                linewidth=2,
                alpha=0.7,
            )

    # Extract layer indices (exclude incident and substrate)
    n_layers = n[1:-1] if n.ndim == 1 else n[1:-1, 0]
    layer_description = ", ".join(
        [
            f"n={np.real(n_val):.2f} d={d:.0f}nm"
            for n_val, d in zip(n_layers, t, strict=False)
        ]
    )

    if method == "reflectance":
        y_label = "Reflectance"
    elif method == "transmittance":
        y_label = "Transmittance"
    else:  # both
        y_label = "Reflectance / Transmittance"

    settings: Dict[str, Any] = {
        "axes": axes,
        "bounding_box": (np.min(wavelengths), np.max(wavelengths), 0, 1),
        "title": f"Multilayer Thin Film Stack ({layer_description}, θ={theta}°)",
        "x_label": "Wavelength (nm)",
        "y_label": y_label,
        "legend": True,
        "show": True,
    }
    settings.update(kwargs)

    return render(**settings)


@override_style()
def plot_thin_film_comparison(
    configurations: Sequence[Dict[str, Any]],
    shape: SpectralShape = SPECTRAL_SHAPE_DEFAULT,
    polarisation: Literal["S", "P", "Both"] | str = "Both",
    **kwargs: Any,
) -> Tuple[Figure, Axes]:
    """
    Plot comparison of multiple thin film configurations.

    Parameters
    ----------
    configurations
        List of dictionaries, each containing parameters for a thin film configuration.

        -   Single layer: ``{'type': 'single', 'n_film': float, 't': float,
            'n_substrate': float, 'label': str}``
        -   Multilayer: ``{'type': 'multilayer', 'refractive_indices': array,
            't': array, 'n_substrate': float, 'label': str}``
    shape
        Spectral shape for wavelength sampling.
    polarisation
        Polarisation to plot: 'S', 'P', or 'Both' (case-insensitive).

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> configurations = [
    ...     {
    ...         "type": "single",
    ...         "n_film": 1.46,
    ...         "t": 100,
    ...         "n_substrate": 1.5,
    ...         "label": "MgF2 100nm",
    ...     },
    ...     {
    ...         "type": "single",
    ...         "n_film": 2.4,
    ...         "t": 25,
    ...         "n_substrate": 1.5,
    ...         "label": "TiO2 25nm",
    ...     },
    ... ]
    >>> plot_thin_film_comparison(configurations)  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...Axes...>)

    .. image:: ../_static/Plotting_Plot_Thin_Film_Comparison.png
        :align: center
        :alt: plot_thin_film_comparison
    """

    wavelengths = shape.wavelengths

    polarisation = validate_method(polarisation, ("S", "P", "Both"))

    _figure, axes = artist(**kwargs)

    cycle = colour_cycle(**kwargs)

    for i, configuration in enumerate(configurations):
        theta = configuration.get("theta", 0)
        label = configuration.get("label", f"Config {i + 1}")
        color = next(cycle)[:3]  # Get RGB values from colour cycle

        if configuration["type"] == "single":
            # Build unified n array: [incident, film, substrate]
            n = [1.0, configuration["n_film"], configuration.get("n_substrate", 1.5)]
            R, _ = thin_film_tmm(n, configuration["t"], wavelengths, theta)
        elif configuration["type"] == "multilayer":
            # Build unified n array: [incident, layers..., substrate]
            n = np.concatenate(
                [
                    [1.0],
                    configuration["refractive_indices"],
                    [configuration.get("n_substrate", 1.5)],
                ]
            )
            R, _ = multilayer_tmm(n, configuration["t"], wavelengths, theta)
        else:
            continue

        if polarisation in ["s", "both"]:
            axes.plot(
                wavelengths,
                R[:, 0, 0, 0],
                color=color,
                linestyle="-",
                label=f"{label} (s-pol)",
                linewidth=2,
            )

        if polarisation in ["p", "both"]:
            axes.plot(
                wavelengths,
                R[:, 0, 0, 1],
                color=color,
                linestyle="--",
                label=f"{label} (p-pol)",
                linewidth=2,
            )

    settings: Dict[str, Any] = {
        "axes": axes,
        "bounding_box": (np.min(wavelengths), np.max(wavelengths), 0, 1),
        "title": "Thin Film Comparison",
        "x_label": "Wavelength (nm)",
        "y_label": "Reflectance",
        "legend": True,
        "show": True,
    }
    settings.update(kwargs)

    return render(**settings)


@override_style()
def plot_thin_film_spectrum(
    n: ArrayLike,
    t: ArrayLike,
    theta: ArrayLike = 0,
    shape: SpectralShape = SPECTRAL_SHAPE_DEFAULT,
    **kwargs: Any,
) -> Tuple[Figure, Axes]:
    """
    Plot reflectance spectrum of thin film using *Transfer Matrix Method*.

    Shows the characteristic oscillating reflectance spectra seen in soap films
    and other thin film interference phenomena.

    Parameters
    ----------
    n
        Complete refractive index stack :math:`n_j` for single-layer film.
        Shape: (3,) or (3, wavelengths_count). The array should contain
        [n_incident, n_film, n_substrate].
    t
        Film thickness :math:`t` in nanometers.
    theta
        Incident angle :math:`\\theta` in degrees. Default is 0 (normal incidence).
    shape
        Spectral shape for wavelength sampling.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> plot_thin_film_spectrum([1.0, 1.33, 1.0], 200)  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...Axes...>)

    .. image:: ../_static/Plotting_Plot_Thin_Film_Spectrum.png
        :align: center
        :alt: plot_thin_film_spectrum
    """

    n = as_complex_array(n)

    _figure, axes = artist(**kwargs)

    wavelengths = shape.wavelengths

    # Calculate reflectance using *Transfer Matrix Method*
    # R has shape (W, A, T, 2) for (wavelength, angle, thickness, polarisation)
    R, _ = thin_film_tmm(n, t, wavelengths, theta)
    # Average s and p polarisations for unpolarized light: R[:, 0, 0, :] -> (W,)
    reflectance = np.mean(R[:, 0, 0, :], axis=1)

    axes.plot(wavelengths, reflectance, "b-", linewidth=2)

    n_layer = n[1] if n.ndim == 1 else n[1, 0]
    title = (
        f"Thin Film Interference (n={np.real(n_layer):.2f}, d={t:.0f}nm, θ={theta}°)"
    )

    settings: Dict[str, Any] = {
        "axes": axes,
        "bounding_box": (np.min(wavelengths), np.max(wavelengths), 0, 1),
        "title": title,
        "x_label": "Wavelength (nm)",
        "y_label": "Reflectance",
        "show": True,
    }
    settings.update(kwargs)

    return render(**settings)


@override_style()
def plot_thin_film_iridescence(
    n: ArrayLike,
    t: ArrayLike | None = None,
    theta: ArrayLike = 0,
    shape: SpectralShape = SPECTRAL_SHAPE_DEFAULT,
    illuminant: SpectralDistribution | str = "D65",
    **kwargs: Any,
) -> Tuple[Figure, Axes]:
    """
    Plot thin film iridescence colours.

    Creates a colour strip showing how thin film interference produces
    iridescent colours, similar to soap films, oil slicks, or soap bubbles.

    Parameters
    ----------
    n
        Complete refractive index stack :math:`n_j` for single-layer film.
        Shape: (3,) or (3, wavelengths_count). The array should contain
        [n_incident, n_film, n_substrate]. Supports wavelength-dependent
        refractive index for dispersion.
    t
        Array of thicknesses :math:`t` in nanometers. If None, uses 0-1000 nm.
    theta
        Incident angle :math:`\\theta` in degrees. Default is 0 (normal incidence).
    shape
        Spectral shape for wavelength sampling.
    illuminant
        Illuminant used for color calculation. Can be either a string (e.g., "D65")
        or a :class:`colour.SpectralDistribution` class instance.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> plot_thin_film_iridescence([1.0, 1.33, 1.0])  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...Axes...>)

    .. image:: ../_static/Plotting_Plot_Thin_Film_Iridescence.png
        :align: center
        :alt: plot_thin_film_iridescence
    """

    n = as_complex_array(n)
    t = as_float_array(optional(t, np.arange(0, 1000, 1)))

    _figure, axes = artist(**kwargs)

    wavelengths = shape.wavelengths

    sd_illuminant = cast(
        "SpectralDistribution",
        first_item(filter_illuminants(illuminant).values()),
    )
    sd_illuminant = sd_illuminant.copy().align(shape)

    # R has shape (W, A, T, 2) for (wavelength, angle, thickness, polarisation)
    R, _ = thin_film_tmm(n, t, wavelengths, theta)
    # Extract single angle and average over polarisations: R[:, 0, :, :] -> (W, T, 2)
    # Average over polarisations (axis=-1): (W, T, 2) -> (W, T)
    msds = MultiSpectralDistributions(np.mean(R[:, 0, :, :], axis=-1), shape)
    XYZ = msds_to_XYZ(msds, illuminant=sd_illuminant, method="Integration") / 100
    RGB = XYZ_to_plotting_colourspace(XYZ)
    RGB = np.clip(normalise_maximum(RGB), 0, 1)

    axes.bar(
        x=t,
        height=1,
        width=np.min(np.diff(t)) if len(t) > 1 else 1,
        color=RGB,
        align="edge",
        zorder=CONSTANTS_COLOUR_STYLE.zorder.background_polygon,
    )

    x_min, x_max = t[0], t[-1]

    n_layer = n[1] if n.ndim == 1 else n[1, 0]
    title = f"Thin Film Iridescence (n={np.real(n_layer):.2f}, θ={theta}°)"

    settings: Dict[str, Any] = {
        "axes": axes,
        "bounding_box": (x_min, x_max, 0, 1),
        "title": title,
        "x_label": "Thickness (nm)",
        "y_label": "",
        "show": True,
    }
    settings.update(kwargs)

    return render(**settings)


@override_style()
def plot_thin_film_reflectance_map(
    n: ArrayLike,
    t: ArrayLike | None = None,
    theta: ArrayLike | None = None,
    shape: SpectralShape = SPECTRAL_SHAPE_DEFAULT,
    polarisation: Literal["Average", "S", "P"] | str = "Average",
    method: Literal["Angle", "Thickness"] | str = "Thickness",
    **kwargs: Any,
) -> Tuple[Figure, Axes]:
    """
    Plot thin film reflectance as a 2D pseudocolor map.

    Creates a 2D visualization showing reflectance as a function of wavelength
    (x-axis) and either film thickness or incident angle (y-axis).

    Parameters
    ----------
    n
        Complete refractive index stack :math:`n_j`. Shape: (media_count,) or
        (media_count, wavelengths_count). The array should contain:

        - **Single layer**: [n_incident, n_film, n_substrate] (length 3)
        - **Multi-layer**: [n_incident, n_layer_1, ..., n_layer_n, n_substrate]
          (length > 3)

        Supports wavelength-dependent refractive index for dispersion.
    t
        Thickness :math:`t` in nanometers. Behavior depends on the method:

        - **Thickness mode (single layer)**: Array of thicknesses or None
          (default: ``np.linspace(0, 1000, 250)``). Sweeps film thickness
          across the range.
        - **Thickness mode (multi-layer)**: Array of thicknesses or None.
          Sweeps all layers simultaneously with the same thickness value.
          For example, ``np.linspace(50, 500, 250)`` sweeps all layers from
          50nm to 500nm together.
        - **Angle mode (single layer)**: Scalar thickness (e.g., ``300``).
          Fixed thickness while varying angle.
        - **Angle mode (multi-layer)**: Array of layer thicknesses
          (e.g., ``[100, 50]`` for 2 layers). All layers kept at fixed
          thickness while varying angle.
    theta
        Incident angle :math:`\\theta` in degrees. Behavior depends on the method:

        - **Thickness mode**: Scalar angle or None (default: 0°).
          Fixed angle while varying thickness.
        - **Angle mode**: Array of angles (e.g., ``np.linspace(0, 90, 250)``).
          Sweeps angle across the range.
    shape
        Spectral shape for wavelength sampling.
    polarisation
        Polarisation to plot: 'S', 'P', or 'Average' (case-insensitive).
        Default is 'Average' (mean of s and p polarisations for unpolarized light).
    method
        Plotting method, one of (case-insensitive):

        - 'Thickness': Plot reflectance vs wavelength and thickness (y-axis)
        - 'Angle': Plot reflectance vs wavelength and angle (y-axis)

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> plot_thin_film_reflectance_map(
    ...     [1.0, 1.33, 1.0], method="Thickness"
    ... )  # doctest: +ELLIPSIS
    (<Figure size ... with 2 Axes>, <...Axes...>)

    .. image:: ../_static/Plotting_Plot_Thin_Film_Reflectance_Map.png
        :align: center
        :alt: plot_thin_film_reflectance_map
    """

    n = np.asarray(n)

    _figure, axes = artist(**kwargs)

    wavelengths = shape.wavelengths

    method = validate_method(method, ("Angle", "Thickness"))
    polarisation = validate_method(polarisation, ("Average", "S", "P"))

    if method == "angle":
        if t is None:
            error = "In angle method, thickness 't' must be specified."

            raise ValueError(error)
        if theta is None:
            error = "In angle method, 'theta' must be specified as an array of angles."

            raise ValueError(error)

        theta = np.atleast_1d(np.asarray(theta))

        if len(theta) == 1:
            error = (
                "In angle method, 'theta' must be an array with multiple angles. "
                "For a single angle, use method='thickness'."
            )

            raise ValueError(error)

        t_array = np.atleast_1d(np.asarray(t))
        R, _ = multilayer_tmm(n, t_array, wavelengths, theta)

        if len(t_array) == 1:
            title_thickness_info = f"d={t_array[0]:.0f} nm"
        else:
            layer_thicknesses = ", ".join([f"{d:.0f}" for d in t_array])
            title_thickness_info = f"d=[{layer_thicknesses}] nm"

        if polarisation == "average":
            R_data = np.transpose(np.mean(R[:, :, 0, :], axis=-1))
            pol_label = "Unpolarized"
        elif polarisation == "s":
            R_data = np.transpose(R[:, :, 0, 0])
            pol_label = "s-pol"
        elif polarisation == "p":
            R_data = np.transpose(R[:, :, 0, 1])
            pol_label = "p-pol"

        W, Y = np.meshgrid(wavelengths, theta)
        y_data = theta
        y_label = "Angle (deg)"
        title_suffix = title_thickness_info

    elif method == "thickness":
        t = as_float_array(optional(t, np.arange(0, 1000, 1)))

        t_array = np.atleast_1d(np.asarray(t))
        theta_scalar = as_float_scalar(theta if theta is not None else 0)

        # Determine number of layers (excluding incident and substrate)
        n_media = len(n) if n.ndim == 1 else n.shape[0]
        n_layers = n_media - 2

        # Create 2D array: (thickness_count, layers_count) where all layers
        # are swept simultaneously with the same thickness value
        t_layers_2d = np.tile(t_array[:, None], (1, n_layers))  # (T, L)

        # Calculate reflectance for all thicknesses at once
        # R has shape (W, 1, T, 2) for (wavelength, angle, thickness, polarisation)
        R, _ = multilayer_tmm(n, t_layers_2d, wavelengths, theta_scalar)

        if polarisation == "average":
            R_data = np.transpose(np.mean(R[:, 0, :, :], axis=-1))
            pol_label = "Unpolarized"
        elif polarisation == "s":
            R_data = np.transpose(R[:, 0, :, 0])
            pol_label = "s-pol"
        elif polarisation == "p":
            R_data = np.transpose(R[:, 0, :, 1])
            pol_label = "p-pol"

        W, Y = np.meshgrid(wavelengths, t_array)
        y_data = t_array
        y_label = "Thickness (nm)"
        title_suffix = f"θ={theta_scalar:.0f}°"

    n_media = len(n) if n.ndim == 1 else n.shape[0]
    if n_media == 3:
        n_layer = n[1] if n.ndim == 1 else n[1, 0]
        title_prefix = f"n={np.real(n_layer):.2f}"
    else:
        n_layers = n_media - 2  # Exclude incident and substrate
        title_prefix = f"{n_layers} layers"

    pcolormesh = axes.pcolormesh(
        W,
        Y,
        R_data,
        shading="auto",
        cmap=CONSTANTS_COLOUR_STYLE.colour.cmap,
        vmin=0,
        vmax=float(np.max(R_data)),
    )

    plt.colorbar(pcolormesh, ax=axes, label="Reflectance")

    title = f"Thin Film Reflectance ({title_prefix}, {title_suffix}, {pol_label})"

    settings: Dict[str, Any] = {
        "axes": axes,
        "bounding_box": (
            np.min(wavelengths),
            np.max(wavelengths),
            np.min(y_data),
            np.max(y_data),
        ),
        "title": title,
        "x_label": "Wavelength (nm)",
        "y_label": y_label,
        "show": True,
    }
    settings.update(kwargs)

    return render(**settings)


@override_style()
def plot_multi_layer_stack(
    configurations: Sequence[Dict[str, Any]],
    theta: ArrayLike | None = None,
    wavelength: ArrayLike = 555,
    **kwargs: Any,
) -> Tuple[Figure, Axes]:
    """
    Plot a multilayer stack as a stacked horizontal bar chart with optional ray paths.

    Creates a visualization showing the layer structure of a multilayer thin film
    or any other multilayer system. Each layer is represented as a horizontal bar
    with height proportional to its thickness, stacked vertically. If an incident
    angle is provided, the function also draws ray paths showing refraction through
    each layer using Snell's law.

    Parameters
    ----------
    configurations
        Sequence of dictionaries, each containing layer configuration:
        {'t': float, 'n': float, 'color': str, 'label': str}

        - 't': Layer thickness in nanometers or any other unit (required)
        - 'n': Refractive index (required)
        - 'color': Layer color (optional, automatically assigned from the
          default colour cycle if not provided)
        - 'label': Layer label (optional, defaults to "Layer N (n=value)")
    theta
        Incident angle :math:`\\theta` in degrees. If provided, ray paths will be
        drawn showing refraction through each layer using Snell's law. Default is
        None (no ray paths).
    wavelength
        Wavelength in nanometers used for transfer matrix calculations when theta
        is provided. Default is 555 nm.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> configurations = [
    ...     {"t": 100, "n": 1.46},
    ...     {"t": 200, "n": 2.4},
    ...     {"t": 80, "n": 1.46},
    ...     {"t": 150, "n": 2.4},
    ... ]
    >>> plot_multi_layer_stack(configurations, theta=45)  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...Axes...>)

    .. image:: ../_static/Plotting_Plot_Multi_Layer_Stack.png
        :align: center
        :alt: plot_multi_layer_stack
    """

    if not configurations:
        error = "At least one layer configuration is required"
        raise ValueError(error)

    _figure, axes = artist(**kwargs)

    cycle = colour_cycle(**kwargs)

    t_a = [configuration["t"] for configuration in configurations]
    t_total = np.sum(t_a)

    # Add space for ray entry and exit - 20% of total thickness if theta provided
    ray_space = (
        t_total * CONSTANTS_COLOUR_STYLE.geometry.x_short / 2.5
        if theta is not None
        else 0
    )
    height = t_total + 2 * ray_space
    width = height

    if theta is not None:
        # Calculate refraction angles using TMM and refractive index array:
        # [n_incident=1.0, n_layers..., n_substrate=1.0]
        result = matrix_transfer_tmm(
            n=[1.0] + [config["n"] for config in configurations] + [1.0],
            t=t_a,
            theta=theta,
            wavelength=wavelength,
        )

        # Get angles for each interface
        # result.theta has shape (angles_count, media_count)
        theta_interface = result.theta[0, :]  # Take first (and only) angle
        theta_entry = theta_interface[0]
        x_center = width / 2

        # Build transmitted ray path coordinates
        # Incident origin
        transmitted_x = [x_center - (ray_space * np.tan(np.radians(theta_entry)))]
        transmitted_y = [height]
        transmitted_x.append(x_center)  # Entry point
        transmitted_y.append(height - ray_space)

        # Traverse layers from top to bottom and build coordinates
        x_position = x_center
        y_position = height - ray_space

        for i in range(len(t_a) - 1, -1, -1):
            # Get angle in this layer
            # angles_at_interfaces: [incident, layer_0, ..., layer_n-1, substrate]
            angle = theta_interface[i + 1]  # +1 to skip incident angle
            thickness = t_a[i]

            # Travel through this layer to reach bottom interface
            y_position -= thickness
            x_position += thickness * np.tan(np.radians(angle))

            transmitted_x.append(x_position)
            transmitted_y.append(y_position)

        # Exit ray from bottom of stack
        x_position += ray_space * np.tan(np.radians(theta_interface[-1]))
        transmitted_x.append(x_position)
        transmitted_y.append(0)

    # Start from top and work downward
    # Layers are indexed 0 to n-1 from bottom to top physically,
    # but we draw them from top to bottom on screen
    t_cumulative = height - ray_space  # Start at top of stack

    # Iterate through configurations in REVERSE order (top layer first)
    for i in range(len(configurations) - 1, -1, -1):
        configuration = configurations[i]
        t = configuration["t"]
        n = configuration["n"]

        # Build label with refractive index
        # Layer numbering: bottom layer is 1, top layer is n
        label = configuration.get("label", f"Layer {i + 1} (n={n:.3f})")

        axes.barh(
            t_cumulative - t / 2,  # Center of the bar (going downward)
            width,
            height=t,
            color=configuration.get("color", next(cycle)[:3]),
            edgecolor="black",
            linewidth=CONSTANTS_COLOUR_STYLE.geometry.x_short,
            label=label,
            zorder=CONSTANTS_COLOUR_STYLE.zorder.background_polygon,
        )
        t_cumulative -= t  # Move down for next layer

    # Draw ray paths if theta was provided
    if theta is not None:
        # Plot incident ray
        plot_ray(
            axes,
            transmitted_x[:2],
            transmitted_y[:2],
            style="solid",
            label=f"Incident (θ={theta}°)",
            show_arrow=True,
            show_dots=False,
        )

        # Plot transmitted rays through stack
        # (black solid line with arrows and dots at interfaces)
        plot_ray(
            axes,
            transmitted_x[1:],
            transmitted_y[1:],
            style="solid",
            label="Transmitted",
            show_arrow=True,
            show_dots=True,
        )

        # Plot reflected rays at each interface (black dashed)
        # Build list of all reflections: [(start_x, start_y, end_x, end_y), ...]
        reflections = []

        # Reflection at entry point (index 1 in transmitted arrays)
        x_incident_origin = transmitted_x[0]
        reflections.append(
            (
                x_center,
                height - ray_space,
                x_center + (x_center - x_incident_origin),  # Mirror incident ray
                height,
            )
        )

        # Internal reflections (indices 2 to -2 in transmitted arrays)
        for idx in range(2, len(transmitted_x) - 1):
            x_refl_start = transmitted_x[idx]
            y_refl_start = transmitted_y[idx]
            y_refl_end = transmitted_y[idx - 1]  # Next interface above

            # Get angle in layer above this interface
            layer_idx = len(t_a) - (idx - 1)
            angle_in_layer = theta_interface[layer_idx + 1]

            # Calculate reflected ray endpoint
            distance = y_refl_start - y_refl_end
            x_refl_end = x_refl_start - distance * np.tan(np.radians(angle_in_layer))

            reflections.append((x_refl_start, y_refl_start, x_refl_end, y_refl_end))

        # Draw all reflected rays using plot_ray
        for i, (x1, y1, x2, y2) in enumerate(reflections):
            plot_ray(
                axes,
                [x1, x2],
                [y1, y2],
                style="dashed",
                label="Reflected" if i == 0 else None,
                show_arrow=True,
                show_dots=False,
            )

    axes.legend(
        loc="center left",
        bbox_to_anchor=(
            CONSTANTS_COLOUR_STYLE.geometry.short * 1.05,
            CONSTANTS_COLOUR_STYLE.geometry.x_short,
        ),
        frameon=True,
        fontsize=CONSTANTS_COLOUR_STYLE.font.size
        * CONSTANTS_COLOUR_STYLE.font.scaling.small,
    )

    settings: Dict[str, Any] = {
        "axes": axes,
        "aspect": "equal",
        "bounding_box": (0, height, 0, height),
        "title": "Multi-layer Stack",
        "x_label": "",
        "y_label": "Thickness [nm]",
        "x_ticker": theta is not None,
    }
    settings.update(kwargs)

    return render(**settings)
