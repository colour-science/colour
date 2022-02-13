"""
Colorimetry Plotting
====================

Defines the colorimetry plotting objects:

-   :func:`colour.plotting.plot_single_sd`
-   :func:`colour.plotting.plot_multi_sds`
-   :func:`colour.plotting.plot_single_cmfs`
-   :func:`colour.plotting.plot_multi_cmfs`
-   :func:`colour.plotting.plot_single_illuminant_sd`
-   :func:`colour.plotting.plot_multi_illuminant_sds`
-   :func:`colour.plotting.plot_visible_spectrum`
-   :func:`colour.plotting.plot_single_lightness_function`
-   :func:`colour.plotting.plot_multi_lightness_functions`
-   :func:`colour.plotting.plot_single_luminance_function`
-   :func:`colour.plotting.plot_multi_luminance_functions`
-   :func:`colour.plotting.plot_blackbody_spectral_radiance`
-   :func:`colour.plotting.plot_blackbody_colours`

References
----------
-   :cite:`Spiker2015a` : Borer, T. (2017). Private Discussion with Mansencal,
    T. and Shaw, N.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
from matplotlib.patches import Polygon

from colour.algebra import LinearInterpolator, normalise_maximum
from colour.colorimetry import (
    CCS_ILLUMINANTS,
    SDS_ILLUMINANTS,
    LIGHTNESS_METHODS,
    LUMINANCE_METHODS,
    MultiSpectralDistributions,
    SpectralDistribution,
    SpectralShape,
    sd_blackbody,
    sd_ones,
    sd_to_XYZ,
    sds_and_msds_to_sds,
    wavelength_to_XYZ,
)
from colour.hints import (
    Any,
    Boolean,
    Callable,
    Dict,
    Floating,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)
from colour.plotting import (
    CONSTANTS_COLOUR_STYLE,
    XYZ_to_plotting_colourspace,
    artist,
    filter_passthrough,
    filter_cmfs,
    filter_illuminants,
    override_style,
    render,
    plot_single_colour_swatch,
    plot_multi_functions,
    update_settings_collection,
)
from colour.utilities import (
    as_float_array,
    domain_range_scale,
    first_item,
    ones,
    tstack,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "plot_single_sd",
    "plot_multi_sds",
    "plot_single_cmfs",
    "plot_multi_cmfs",
    "plot_single_illuminant_sd",
    "plot_multi_illuminant_sds",
    "plot_visible_spectrum",
    "plot_single_lightness_function",
    "plot_multi_lightness_functions",
    "plot_single_luminance_function",
    "plot_multi_luminance_functions",
    "plot_blackbody_spectral_radiance",
    "plot_blackbody_colours",
]


@override_style()
def plot_single_sd(
    sd: SpectralDistribution,
    cmfs: Union[
        MultiSpectralDistributions,
        str,
        Sequence[Union[MultiSpectralDistributions, str]],
    ] = "CIE 1931 2 Degree Standard Observer",
    out_of_gamut_clipping: Boolean = True,
    modulate_colours_with_sd_amplitude: Boolean = False,
    equalize_sd_amplitude: Boolean = False,
    **kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot given spectral distribution.

    Parameters
    ----------
    sd
        Spectral distribution to plot.
    cmfs
        Standard observer colour matching functions used for computing the
        spectrum domain and colours. ``cmfs`` can be of any type or form
        supported by the :func:`colour.plotting.filter_cmfs` definition.
    out_of_gamut_clipping
        Whether to clip out of gamut colours otherwise, the colours will be
        offset by the absolute minimal colour leading to a rendering on
        gray background, less saturated and smoother.
    modulate_colours_with_sd_amplitude
        Whether to modulate the colours with the spectral distribution
        amplitude.
    equalize_sd_amplitude
        Whether to equalize the spectral distribution amplitude.
        Equalization occurs after the colours modulation thus setting both
        arguments to *True* will generate a spectrum strip where each
        wavelength colour is modulated by the spectral distribution amplitude.
        The usual 5% margin above the spectral distribution is also omitted.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`, :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    References
    ----------
    :cite:`Spiker2015a`

    Examples
    --------
    >>> from colour import SpectralDistribution
    >>> data = {
    ...     500: 0.0651,
    ...     520: 0.0705,
    ...     540: 0.0772,
    ...     560: 0.0870,
    ...     580: 0.1128,
    ...     600: 0.1360
    ... }
    >>> sd = SpectralDistribution(data, name='Custom')
    >>> plot_single_sd(sd)  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_Single_SD.png
        :align: center
        :alt: plot_single_sd
    """

    _figure, axes = artist(**kwargs)

    cmfs = cast(
        MultiSpectralDistributions, first_item(filter_cmfs(cmfs).values())
    )

    sd = cast(SpectralDistribution, sd.copy())
    sd.interpolator = LinearInterpolator
    wavelengths = cmfs.wavelengths[
        np.logical_and(
            cmfs.wavelengths
            >= max(min(cmfs.wavelengths), min(sd.wavelengths)),
            cmfs.wavelengths
            <= min(max(cmfs.wavelengths), max(sd.wavelengths)),
        )
    ]
    values = as_float_array(sd[wavelengths])

    RGB = XYZ_to_plotting_colourspace(
        wavelength_to_XYZ(wavelengths, cmfs),
        CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["E"],
        apply_cctf_encoding=False,
    )

    if not out_of_gamut_clipping:
        RGB += np.abs(np.min(RGB))

    RGB = normalise_maximum(RGB)

    if modulate_colours_with_sd_amplitude:
        RGB *= (values / np.max(values))[..., np.newaxis]

    RGB = CONSTANTS_COLOUR_STYLE.colour.colourspace.cctf_encoding(RGB)

    if equalize_sd_amplitude:
        values = ones(values.shape)

    margin = 0 if equalize_sd_amplitude else 0.05

    x_min, x_max = min(wavelengths), max(wavelengths)
    y_min, y_max = 0, max(values) + max(values) * margin

    polygon = Polygon(
        np.vstack(
            [
                (x_min, 0),
                tstack([wavelengths, values]),
                (x_max, 0),
            ]
        ),
        facecolor="none",
        edgecolor="none",
        zorder=CONSTANTS_COLOUR_STYLE.zorder.background_polygon,
    )
    axes.add_patch(polygon)

    padding = 0.1
    axes.bar(
        x=wavelengths - padding,
        height=max(values),
        width=1 + padding,
        color=RGB,
        align="edge",
        clip_path=polygon,
        zorder=CONSTANTS_COLOUR_STYLE.zorder.background_polygon,
    )

    axes.plot(
        wavelengths,
        values,
        color=CONSTANTS_COLOUR_STYLE.colour.dark,
        zorder=CONSTANTS_COLOUR_STYLE.zorder.midground_line,
    )

    settings: Dict[str, Any] = {
        "axes": axes,
        "bounding_box": (x_min, x_max, y_min, y_max),
        "title": f"{sd.strict_name} - {cmfs.strict_name}",
        "x_label": "Wavelength $\\lambda$ (nm)",
        "y_label": "Spectral Distribution",
    }
    settings.update(kwargs)

    return render(**settings)


@override_style()
def plot_multi_sds(
    sds: Union[
        Sequence[Union[SpectralDistribution, MultiSpectralDistributions]],
        MultiSpectralDistributions,
    ],
    plot_kwargs: Optional[Union[Dict, List[Dict]]] = None,
    **kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot given spectral distributions.

    Parameters
    ----------
    sds
        Spectral distributions or multi-spectral distributions to
        plot. `sds` can be a single
        :class:`colour.MultiSpectralDistributions` class instance, a list
        of :class:`colour.MultiSpectralDistributions` class instances or a
        list of :class:`colour.SpectralDistribution` class instances.
    plot_kwargs
        Keyword arguments for the :func:`matplotlib.pyplot.plot` definition,
        used to control the style of the plotted spectral distributions.
        `plot_kwargs`` can be either a single dictionary applied to all the
        plotted spectral distributions with the same settings or a sequence of
        dictionaries with different settings for each plotted spectral
        distributions. The following special keyword arguments can also be
        used:

        -   ``illuminant`` : The illuminant used to compute the spectral
            distributions colours. The default is the illuminant associated
            with the whitepoint of the default plotting colourspace.
            ``illuminant`` can be of any type or form supported by the
            :func:`colour.plotting.filter_cmfs` definition.
        -   ``cmfs`` : The standard observer colour matching functions used for
            computing the spectral distributions colours. ``cmfs`` can be of
            any type or form supported by the
            :func:`colour.plotting.filter_cmfs` definition.
        -   ``normalise_sd_colours`` : Whether to normalise the computed
            spectral distributions colours. The default is *True*.
        -   ``use_sd_colours`` : Whether to use the computed spectral
            distributions colours under the plotting colourspace illuminant.
            Alternatively, it is possible to use the
            :func:`matplotlib.pyplot.plot` definition ``color`` argument with
            pre-computed values. The default is *True*.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`, :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> from colour import SpectralDistribution
    >>> data_1 = {
    ...     500: 0.004900,
    ...     510: 0.009300,
    ...     520: 0.063270,
    ...     530: 0.165500,
    ...     540: 0.290400,
    ...     550: 0.433450,
    ...     560: 0.594500
    ... }
    >>> data_2 = {
    ...     500: 0.323000,
    ...     510: 0.503000,
    ...     520: 0.710000,
    ...     530: 0.862000,
    ...     540: 0.954000,
    ...     550: 0.994950,
    ...     560: 0.995000
    ... }
    >>> sd_1 = SpectralDistribution(data_1, name='Custom 1')
    >>> sd_2 = SpectralDistribution(data_2, name='Custom 2')
    >>> plot_kwargs = [
    ...     {'use_sd_colours': True},
    ...     {'use_sd_colours': True, 'linestyle': 'dashed'},
    ... ]
    >>> plot_multi_sds([sd_1, sd_2], plot_kwargs=plot_kwargs)
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_Multi_SDS.png
        :align: center
        :alt: plot_multi_sds
    """

    _figure, axes = artist(**kwargs)

    sds_converted = sds_and_msds_to_sds(sds)

    plot_settings_collection = [
        {
            "label": f"{sd.strict_name}",
            "zorder": CONSTANTS_COLOUR_STYLE.zorder.midground_line,
            "cmfs": "CIE 1931 2 Degree Standard Observer",
            "illuminant": SDS_ILLUMINANTS[
                CONSTANTS_COLOUR_STYLE.colour.colourspace.whitepoint_name
            ],
            "use_sd_colours": False,
            "normalise_sd_colours": False,
        }
        for sd in sds_converted
    ]

    if plot_kwargs is not None:
        update_settings_collection(
            plot_settings_collection, plot_kwargs, len(sds_converted)
        )

    x_limit_min, x_limit_max, y_limit_min, y_limit_max = [], [], [], []
    for i, sd in enumerate(sds_converted):
        plot_settings = plot_settings_collection[i]

        cmfs = cast(
            MultiSpectralDistributions,
            first_item(filter_cmfs(plot_settings.pop("cmfs")).values()),
        )
        illuminant = cast(
            SpectralDistribution,
            first_item(
                filter_illuminants(plot_settings.pop("illuminant")).values()
            ),
        )
        normalise_sd_colours = plot_settings.pop("normalise_sd_colours")
        use_sd_colours = plot_settings.pop("use_sd_colours")

        wavelengths, values = sd.wavelengths, sd.values

        shape = sd.shape
        x_limit_min.append(shape.start)
        x_limit_max.append(shape.end)
        y_limit_min.append(min(values))
        y_limit_max.append(max(values))

        if use_sd_colours:
            with domain_range_scale("1"):
                XYZ = sd_to_XYZ(sd, cmfs, illuminant)

            if normalise_sd_colours:
                XYZ /= XYZ[..., 1]

            plot_settings["color"] = np.clip(
                XYZ_to_plotting_colourspace(XYZ), 0, 1
            )

        axes.plot(wavelengths, values, **plot_settings)

    bounding_box = (
        min(x_limit_min),
        max(x_limit_max),
        min(y_limit_min),
        max(y_limit_max) + np.max(y_limit_max) * 0.05,
    )
    settings: Dict[str, Any] = {
        "axes": axes,
        "bounding_box": bounding_box,
        "legend": True,
        "x_label": "Wavelength $\\lambda$ (nm)",
        "y_label": "Spectral Distribution",
    }
    settings.update(kwargs)

    return render(**settings)


@override_style()
def plot_single_cmfs(
    cmfs: Union[
        MultiSpectralDistributions,
        str,
        Sequence[Union[MultiSpectralDistributions, str]],
    ] = "CIE 1931 2 Degree Standard Observer",
    **kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot given colour matching functions.

    Parameters
    ----------
    cmfs
        Colour matching functions to plot. ``cmfs`` can be of any type or form
        supported by the :func:`colour.plotting.filter_cmfs` definition.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.plot_multi_cmfs`,
        :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> plot_single_cmfs('CIE 1931 2 Degree Standard Observer')
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_Single_CMFS.png
        :align: center
        :alt: plot_single_cmfs
    """

    cmfs = cast(
        MultiSpectralDistributions, first_item(filter_cmfs(cmfs).values())
    )

    settings: Dict[str, Any] = {
        "title": f"{cmfs.strict_name} - Colour Matching Functions"
    }
    settings.update(kwargs)

    return plot_multi_cmfs((cmfs,), **settings)


@override_style()
def plot_multi_cmfs(
    cmfs: Union[
        MultiSpectralDistributions,
        str,
        Sequence[Union[MultiSpectralDistributions, str]],
    ],
    **kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot given colour matching functions.

    Parameters
    ----------
    cmfs
        Colour matching functions to plot. ``cmfs`` elements can be of any
        type or form supported by the :func:`colour.plotting.filter_cmfs`
        definition.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`, :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> cmfs = [
    ...     'CIE 1931 2 Degree Standard Observer',
    ...     'CIE 1964 10 Degree Standard Observer',
    ... ]
    >>> plot_multi_cmfs(cmfs)  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_Multi_CMFS.png
        :align: center
        :alt: plot_multi_cmfs
    """

    cmfs = cast(
        List[MultiSpectralDistributions], list(filter_cmfs(cmfs).values())
    )

    _figure, axes = artist(**kwargs)

    axes.axhline(
        color=CONSTANTS_COLOUR_STYLE.colour.dark,
        linestyle="--",
        zorder=CONSTANTS_COLOUR_STYLE.zorder.foreground_line,
    )

    x_limit_min, x_limit_max, y_limit_min, y_limit_max = [], [], [], []
    for i, cmfs_i in enumerate(cmfs):
        for j, RGB in enumerate(
            as_float_array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        ):
            RGB = [reduce(lambda y, _: y * 0.5, range(i), x) for x in RGB]
            values = cmfs_i.values[:, j]

            shape = cmfs_i.shape
            x_limit_min.append(shape.start)
            x_limit_max.append(shape.end)
            y_limit_min.append(np.min(values))
            y_limit_max.append(np.max(values))

            axes.plot(
                cmfs_i.wavelengths,
                values,
                color=RGB,
                label=f"{cmfs_i.strict_labels[j]} - {cmfs_i.strict_name}",
                zorder=CONSTANTS_COLOUR_STYLE.zorder.midground_line,
            )

    bounding_box = (
        min(x_limit_min),
        max(x_limit_max),
        min(y_limit_min) - np.abs(np.min(y_limit_min)) * 0.05,
        max(y_limit_max) + np.abs(np.max(y_limit_max)) * 0.05,
    )
    cmfs_strict_names = ", ".join([cmfs_i.strict_name for cmfs_i in cmfs])
    title = f"{cmfs_strict_names} - Colour Matching Functions"

    settings: Dict[str, Any] = {
        "axes": axes,
        "bounding_box": bounding_box,
        "legend": True,
        "title": title,
        "x_label": "Wavelength $\\lambda$ (nm)",
        "y_label": "Tristimulus Values",
    }
    settings.update(kwargs)

    return render(**settings)


@override_style()
def plot_single_illuminant_sd(
    illuminant: Union[SpectralDistribution, str],
    cmfs: Union[
        MultiSpectralDistributions,
        str,
        Sequence[Union[MultiSpectralDistributions, str]],
    ] = "CIE 1931 2 Degree Standard Observer",
    **kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot given single illuminant spectral distribution.

    Parameters
    ----------
    illuminant
        Illuminant to plot. ``illuminant`` can be of any type or form supported
        by the :func:`colour.plotting.filter_illuminants` definition.
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

    References
    ----------
    :cite:`Spiker2015a`

    Examples
    --------
    >>> plot_single_illuminant_sd('A')  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_Single_Illuminant_SD.png
        :align: center
        :alt: plot_single_illuminant_sd
    """

    cmfs = cast(
        MultiSpectralDistributions, first_item(filter_cmfs(cmfs).values())
    )

    title = f"Illuminant {illuminant} - {cmfs.strict_name}"

    illuminant = first_item(filter_illuminants(illuminant).values())

    settings: Dict[str, Any] = {"title": title, "y_label": "Relative Power"}
    settings.update(kwargs)

    return plot_single_sd(illuminant, **settings)


@override_style()
def plot_multi_illuminant_sds(
    illuminants: Union[
        SpectralDistribution, str, Sequence[Union[SpectralDistribution, str]]
    ],
    **kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot given illuminants spectral distributions.

    Parameters
    ----------
    illuminants
        Illuminants to plot. ``illuminants`` elements can be of any type or
        form supported by the :func:`colour.plotting.filter_illuminants`
        definition.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.plot_multi_sds`,
        :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> plot_multi_illuminant_sds(['A', 'B', 'C'])  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_Multi_Illuminant_SDS.png
        :align: center
        :alt: plot_multi_illuminant_sds
    """

    if "plot_kwargs" not in kwargs:
        kwargs["plot_kwargs"] = {}

    SD_E = SDS_ILLUMINANTS["E"]
    if isinstance(kwargs["plot_kwargs"], dict):
        kwargs["plot_kwargs"]["illuminant"] = SD_E
    else:
        for i in range(len(kwargs["plot_kwargs"])):
            kwargs["plot_kwargs"][i]["illuminant"] = SD_E

    illuminants = cast(
        List[SpectralDistribution],
        list(filter_illuminants(illuminants).values()),
    )

    illuminant_strict_names = ", ".join(
        [illuminant.strict_name for illuminant in illuminants]
    )
    title = f"{illuminant_strict_names} - Illuminants Spectral Distributions"

    settings: Dict[str, Any] = {"title": title, "y_label": "Relative Power"}
    settings.update(kwargs)

    return plot_multi_sds(illuminants, **settings)


@override_style(
    **{
        "ytick.left": False,
        "ytick.labelleft": False,
    }
)
def plot_visible_spectrum(
    cmfs: Union[
        MultiSpectralDistributions,
        str,
        Sequence[Union[MultiSpectralDistributions, str]],
    ] = "CIE 1931 2 Degree Standard Observer",
    out_of_gamut_clipping: Boolean = True,
    **kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the visible colours spectrum using given standard observer *CIE XYZ*
    colour matching functions.

    Parameters
    ----------
    cmfs
        Standard observer colour matching functions used for computing the
        spectrum domain and colours. ``cmfs`` can be of any type or form
        supported by the :func:`colour.plotting.filter_cmfs` definition.
    out_of_gamut_clipping
        Whether to clip out of gamut colours otherwise, the colours will be
        offset by the absolute minimal colour leading to a rendering on
        gray background, less saturated and smoother.

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

    References
    ----------
    :cite:`Spiker2015a`

    Examples
    --------
    >>> plot_visible_spectrum()  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_Visible_Spectrum.png
        :align: center
        :alt: plot_visible_spectrum
    """

    cmfs = cast(
        MultiSpectralDistributions, first_item(filter_cmfs(cmfs).values())
    )

    bounding_box = (min(cmfs.wavelengths), max(cmfs.wavelengths), 0, 1)

    settings: Dict[str, Any] = {"bounding_box": bounding_box, "y_label": None}
    settings.update(kwargs)
    settings["standalone"] = False

    _figure, axes = plot_single_sd(
        sd_ones(cmfs.shape),
        cmfs=cmfs,
        out_of_gamut_clipping=out_of_gamut_clipping,
        **settings,
    )

    # Removing wavelength line as it doubles with the axes spine.
    axes.lines.pop(0)

    settings = {
        "axes": axes,
        "standalone": True,
        "title": f"The Visible Spectrum - {cmfs.strict_name}",
        "x_label": "Wavelength $\\lambda$ (nm)",
    }
    settings.update(kwargs)

    return render(**settings)


@override_style()
def plot_single_lightness_function(
    function: Union[Callable, str], **kwargs: Any
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot given *Lightness* function.

    Parameters
    ----------
    function
        *Lightness* function to plot. ``function`` can be of any type or form
        supported by the :func:`colour.plotting.filter_passthrough` definition.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.plot_multi_functions`,
        :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> plot_single_lightness_function('CIE 1976')  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_Single_Lightness_Function.png
        :align: center
        :alt: plot_single_lightness_function
    """

    settings: Dict[str, Any] = {"title": f"{function} - Lightness Function"}
    settings.update(kwargs)

    return plot_multi_lightness_functions((function,), **settings)


@override_style()
def plot_multi_lightness_functions(
    functions: Union[Callable, str, Sequence[Union[Callable, str]]],
    **kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot given *Lightness* functions.

    Parameters
    ----------
    functions
        *Lightness* functions to plot. ``functions`` elements can be of any
        type or form supported by the
        :func:`colour.plotting.filter_passthrough` definition.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.plot_multi_functions`,
        :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> plot_multi_lightness_functions(['CIE 1976', 'Wyszecki 1963'])
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_Multi_Lightness_Functions.png
        :align: center
        :alt: plot_multi_lightness_functions
    """

    functions_filtered = filter_passthrough(LIGHTNESS_METHODS, functions)

    settings: Dict[str, Any] = {
        "bounding_box": (0, 1, 0, 1),
        "legend": True,
        "title": f"{', '.join(functions_filtered)} - Lightness Functions",
        "x_label": "Normalised Relative Luminance Y",
        "y_label": "Normalised Lightness",
    }
    settings.update(kwargs)

    with domain_range_scale("1"):
        return plot_multi_functions(functions_filtered, **settings)


@override_style()
def plot_single_luminance_function(
    function: Union[Callable, str], **kwargs: Any
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot given *Luminance* function.

    Parameters
    ----------
    function
        *Luminance* function to plot.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.plot_multi_functions`,
        :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> plot_single_luminance_function('CIE 1976')  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_Single_Luminance_Function.png
        :align: center
        :alt: plot_single_luminance_function
    """

    settings: Dict[str, Any] = {"title": f"{function} - Luminance Function"}
    settings.update(kwargs)

    return plot_multi_luminance_functions((function,), **settings)


@override_style()
def plot_multi_luminance_functions(
    functions: Union[Callable, str, Sequence[Union[Callable, str]]],
    **kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot given *Luminance* functions.

    Parameters
    ----------
    functions
        *Luminance* functions to plot. ``functions`` elements can be of any
        type or form supported by the
        :func:`colour.plotting.filter_passthrough` definition.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.plot_multi_functions`,
        :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> plot_multi_luminance_functions(['CIE 1976', 'Newhall 1943'])
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_Multi_Luminance_Functions.png
        :align: center
        :alt: plot_multi_luminance_functions
    """

    functions_filtered = filter_passthrough(LUMINANCE_METHODS, functions)

    settings: Dict[str, Any] = {
        "bounding_box": (0, 1, 0, 1),
        "legend": True,
        "title": f"{', '.join(functions_filtered)} - Luminance Functions",
        "x_label": "Normalised Munsell Value / Lightness",
        "y_label": "Normalised Relative Luminance Y",
    }
    settings.update(kwargs)

    with domain_range_scale("1"):
        return plot_multi_functions(functions_filtered, **settings)


@override_style()
def plot_blackbody_spectral_radiance(
    temperature: Floating = 3500,
    cmfs: Union[
        MultiSpectralDistributions,
        str,
        Sequence[Union[MultiSpectralDistributions, str]],
    ] = "CIE 1931 2 Degree Standard Observer",
    blackbody: str = "VY Canis Major",
    **kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot given blackbody spectral radiance.

    Parameters
    ----------
    temperature
        Blackbody temperature.
    cmfs
        Standard observer colour matching functions used for computing the
        spectrum domain and colours. ``cmfs`` can be of any type or form
        supported by the :func:`colour.plotting.filter_cmfs` definition.
    blackbody
        Blackbody name.

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
    >>> plot_blackbody_spectral_radiance(3500, blackbody='VY Canis Major')
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 2 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_Blackbody_Spectral_Radiance.png
        :align: center
        :alt: plot_blackbody_spectral_radiance
    """

    figure = plt.figure()

    figure.subplots_adjust(hspace=CONSTANTS_COLOUR_STYLE.geometry.short / 2)

    cmfs = cast(
        MultiSpectralDistributions, first_item(filter_cmfs(cmfs).values())
    )

    sd = sd_blackbody(temperature, cmfs.shape)

    axes = figure.add_subplot(211)
    settings: Dict[str, Any] = {
        "axes": axes,
        "title": f"{blackbody} - Spectral Radiance",
        "y_label": "W / (sr m$^2$) / m",
    }
    settings.update(kwargs)
    settings["standalone"] = False

    plot_single_sd(sd, cmfs.name, **settings)

    axes = figure.add_subplot(212)

    with domain_range_scale("1"):
        XYZ = sd_to_XYZ(sd, cmfs)

    RGB = normalise_maximum(XYZ_to_plotting_colourspace(XYZ))

    settings = {
        "axes": axes,
        "aspect": None,
        "title": f"{blackbody} - Colour",
        "x_label": f"{temperature}K",
        "y_label": "",
        "x_ticker": False,
        "y_ticker": False,
    }
    settings.update(kwargs)
    settings["standalone"] = False

    figure, axes = plot_single_colour_swatch(RGB, **settings)

    settings = {"axes": axes, "standalone": True}
    settings.update(kwargs)

    return render(**settings)


@override_style(
    **{
        "ytick.left": False,
        "ytick.labelleft": False,
    }
)
def plot_blackbody_colours(
    shape: SpectralShape = SpectralShape(150, 12500, 50),
    cmfs: Union[
        MultiSpectralDistributions,
        str,
        Sequence[Union[MultiSpectralDistributions, str]],
    ] = "CIE 1931 2 Degree Standard Observer",
    **kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot blackbody colours.

    Parameters
    ----------
    shape
        Spectral shape to use as plot boundaries.
    cmfs
        Standard observer colour matching functions used for computing the
        blackbody colours. ``cmfs`` can be of any type or form supported by the
        :func:`colour.plotting.filter_cmfs` definition.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`, :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> plot_blackbody_colours(SpectralShape(150, 12500, 50))
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_Blackbody_Colours.png
        :align: center
        :alt: plot_blackbody_colours
    """

    _figure, axes = artist(**kwargs)

    cmfs = cast(
        MultiSpectralDistributions, first_item(filter_cmfs(cmfs).values())
    )

    RGB = []
    temperatures = []

    for temperature in shape:
        sd = sd_blackbody(temperature, cmfs.shape)

        with domain_range_scale("1"):
            XYZ = sd_to_XYZ(sd, cmfs)

        RGB.append(normalise_maximum(XYZ_to_plotting_colourspace(XYZ)))
        temperatures.append(temperature)

    x_min, x_max = min(temperatures), max(temperatures)
    y_min, y_max = 0, 1

    padding = 0.1
    axes.bar(
        x=np.array(temperatures) - padding,
        height=1,
        width=shape.interval + (padding * shape.interval),
        color=RGB,
        align="edge",
        zorder=CONSTANTS_COLOUR_STYLE.zorder.background_polygon,
    )

    settings: Dict[str, Any] = {
        "axes": axes,
        "bounding_box": (x_min, x_max, y_min, y_max),
        "title": "Blackbody Colours",
        "x_label": "Temperature K",
        "y_label": None,
    }
    settings.update(kwargs)

    return render(**settings)
