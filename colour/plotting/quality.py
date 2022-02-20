"""
Colour Quality Plotting
=======================

Defines the colour quality plotting objects:

-   :func:`colour.plotting.plot_single_sd_colour_rendering_index_bars`
-   :func:`colour.plotting.plot_multi_sds_colour_rendering_indexes_bars`
-   :func:`colour.plotting.plot_single_sd_colour_quality_scale_bars`
-   :func:`colour.plotting.plot_multi_sds_colour_quality_scales_bars`
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle

from colour.constants import DEFAULT_FLOAT_DTYPE
from colour.colorimetry import (
    MultiSpectralDistributions,
    SpectralDistribution,
    sds_and_msds_to_sds,
)
from colour.hints import (
    Any,
    Boolean,
    Dict,
    Integer,
    List,
    Literal,
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
    label_rectangles,
    override_style,
    render,
)
from colour.quality import (
    COLOUR_QUALITY_SCALE_METHODS,
    ColourRendering_Specification_CQS,
    ColourRendering_Specification_CRI,
    colour_quality_scale,
    colour_rendering_index,
)
from colour.utilities import as_float_array, validate_method

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "plot_colour_quality_bars",
    "plot_single_sd_colour_rendering_index_bars",
    "plot_multi_sds_colour_rendering_indexes_bars",
    "plot_single_sd_colour_quality_scale_bars",
    "plot_multi_sds_colour_quality_scales_bars",
]


@override_style()
def plot_colour_quality_bars(
    specifications: Sequence[
        Union[
            ColourRendering_Specification_CQS,
            ColourRendering_Specification_CRI,
        ]
    ],
    labels: Boolean = True,
    hatching: Optional[Boolean] = None,
    hatching_repeat: Integer = 2,
    **kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the colour quality data of given illuminants or light sources colour
    quality specifications.

    Parameters
    ----------
    specifications
        Array of illuminants or light sources colour quality specifications.
    labels
        Add labels above bars.
    hatching
        Use hatching for the bars.
    hatching_repeat
        Hatching pattern repeat.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.quality.plot_colour_quality_bars`,
        :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> from colour import (SDS_ILLUMINANTS,
    ...                     SDS_LIGHT_SOURCES, SpectralShape)
    >>> illuminant = SDS_ILLUMINANTS['FL2']
    >>> light_source = SDS_LIGHT_SOURCES['Kinoton 75P']
    >>> light_source = light_source.copy().align(SpectralShape(360, 830, 1))
    >>> cqs_i = colour_quality_scale(illuminant, additional_data=True)
    >>> cqs_l = colour_quality_scale(light_source, additional_data=True)
    >>> plot_colour_quality_bars([cqs_i, cqs_l])  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_Colour_Quality_Bars.png
        :align: center
        :alt: plot_colour_quality_bars
    """

    settings: Dict[str, Any] = {"uniform": True}
    settings.update(kwargs)

    _figure, axes = artist(**settings)

    bar_width = 0.5
    y_ticks_interval = 10
    count_s, count_Q_as = len(specifications), 0
    patterns = cycle(CONSTANTS_COLOUR_STYLE.hatch.patterns)
    if hatching is None:
        hatching = False if count_s == 1 else True

    for i, specification in enumerate(specifications):
        Q_a, Q_as, colorimetry_data = (
            specification.Q_a,
            specification.Q_as,
            specification.colorimetry_data,
        )

        count_Q_as = len(Q_as)
        RGB = [[1] * 3] + [
            np.clip(XYZ_to_plotting_colourspace(x.XYZ), 0, 1)
            for x in colorimetry_data[0]
        ]

        x = (
            as_float_array(
                i
                + np.arange(
                    0,
                    (count_Q_as + 1) * (count_s + 1),
                    (count_s + 1),
                    dtype=DEFAULT_FLOAT_DTYPE,
                )
            )
            * bar_width
        )
        y = as_float_array(
            [Q_a]
            + [
                s[1].Q_a  # type: ignore[attr-defined]
                for s in sorted(Q_as.items(), key=lambda s: s[0])
            ]
        )

        bars = axes.bar(
            x,
            np.abs(y),
            color=RGB,
            width=bar_width,
            edgecolor=CONSTANTS_COLOUR_STYLE.colour.dark,
            label=specification.name,
            zorder=CONSTANTS_COLOUR_STYLE.zorder.background_polygon,
        )

        hatches = (
            [next(patterns) * hatching_repeat] * (count_Q_as + 1)
            if hatching
            else list(
                np.where(y < 0, next(patterns), None)  # type: ignore[call-overload]
            )
        )

        for j, bar in enumerate(bars.patches):
            bar.set_hatch(hatches[j])

        if labels:
            label_rectangles(
                [f"{y_v:.1f}" for y_v in y],
                bars,
                rotation="horizontal" if count_s == 1 else "vertical",
                offset=(
                    0 if count_s == 1 else 3 / 100 * count_s + 65 / 1000,
                    0.025,
                ),
                text_size=-5 / 7 * count_s + 12.5,
                axes=axes,
            )

    axes.axhline(
        y=100,
        color=CONSTANTS_COLOUR_STYLE.colour.dark,
        linestyle="--",
        zorder=CONSTANTS_COLOUR_STYLE.zorder.midground_line,
    )

    axes.set_xticks(
        (
            np.arange(
                0,
                (count_Q_as + 1) * (count_s + 1),
                (count_s + 1),
                dtype=DEFAULT_FLOAT_DTYPE,
            )
            - bar_width
        )
        * bar_width
        + (count_s * bar_width / 2)
    )
    axes.set_xticklabels(
        ["Qa"] + [f"Q{index + 1}" for index in range(0, count_Q_as, 1)]
    )
    axes.set_yticks(range(0, 100 + y_ticks_interval, y_ticks_interval))

    aspect = 1 / (120 / (bar_width + len(Q_as) + bar_width * 2))
    bounding_box = (
        -bar_width,
        ((count_Q_as + 1) * (count_s + 1)) / 2 - bar_width,
        0,
        120,
    )

    settings = {
        "axes": axes,
        "aspect": aspect,
        "bounding_box": bounding_box,
        "legend": hatching,
        "title": "Colour Quality",
    }
    settings.update(kwargs)

    return render(**settings)


@override_style()
def plot_single_sd_colour_rendering_index_bars(
    sd: SpectralDistribution, **kwargs: Any
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the *Colour Rendering Index* (CRI) of given illuminant or light
    source spectral distribution.

    Parameters
    ----------
    sd
        Illuminant or light source spectral distribution to plot the
        *Colour Rendering Index* (CRI).

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.quality.plot_colour_quality_bars`,
        :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> from colour import SDS_ILLUMINANTS
    >>> illuminant = SDS_ILLUMINANTS['FL2']
    >>> plot_single_sd_colour_rendering_index_bars(illuminant)
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_\
Plot_Single_SD_Colour_Rendering_Index_Bars.png
        :align: center
        :alt: plot_single_sd_colour_rendering_index_bars
    """

    return plot_multi_sds_colour_rendering_indexes_bars([sd], **kwargs)


@override_style()
def plot_multi_sds_colour_rendering_indexes_bars(
    sds: Union[
        Sequence[Union[SpectralDistribution, MultiSpectralDistributions]],
        MultiSpectralDistributions,
    ],
    **kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the *Colour Rendering Index* (CRI) of given illuminants or light
    sources spectral distributions.

    Parameters
    ----------
    sds
        Spectral distributions or multi-spectral distributions to
        plot. `sds` can be a single
        :class:`colour.MultiSpectralDistributions` class instance, a list
        of :class:`colour.MultiSpectralDistributions` class instances or a
        list of :class:`colour.SpectralDistribution` class instances.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.quality.plot_colour_quality_bars`,
        :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> from colour import (SDS_ILLUMINANTS,
    ...                     SDS_LIGHT_SOURCES)
    >>> illuminant = SDS_ILLUMINANTS['FL2']
    >>> light_source = SDS_LIGHT_SOURCES['Kinoton 75P']
    >>> plot_multi_sds_colour_rendering_indexes_bars(
    ...     [illuminant, light_source])  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_\
Plot_Multi_SDS_Colour_Rendering_Indexes_Bars.png
        :align: center
        :alt: plot_multi_sds_colour_rendering_indexes_bars
    """

    sds_converted = sds_and_msds_to_sds(sds)

    settings: Dict[str, Any] = dict(kwargs)
    settings.update({"standalone": False})

    specifications = cast(
        List[ColourRendering_Specification_CRI],
        [
            colour_rendering_index(sd, additional_data=True)
            for sd in sds_converted
        ],
    )

    # *colour rendering index* colorimetry data tristimulus values are
    # computed in [0, 100] domain however `plot_colour_quality_bars` expects
    # [0, 1] domain. As we want to keep `plot_colour_quality_bars` definition
    # agnostic from the colour quality data, we update the test sd
    # colorimetry data tristimulus values domain.
    for specification in specifications:
        colorimetry_data = specification.colorimetry_data
        for i in range(len(colorimetry_data[0])):
            colorimetry_data[0][i].XYZ /= 100

    _figure, axes = plot_colour_quality_bars(specifications, **settings)

    title = (
        f"Colour Rendering Index - "
        f"{', '.join([sd.strict_name for sd in sds_converted])}"
    )

    settings = {"axes": axes, "title": title}
    settings.update(kwargs)

    return render(**settings)


@override_style()
def plot_single_sd_colour_quality_scale_bars(
    sd: SpectralDistribution,
    method: Union[
        Literal["NIST CQS 7.4", "NIST CQS 9.0"], str
    ] = "NIST CQS 9.0",
    **kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the *Colour Quality Scale* (CQS) of given illuminant or light source
    spectral distribution.

    Parameters
    ----------
    sd
        Illuminant or light source spectral distribution to plot the
        *Colour Quality Scale* (CQS).
    method
        *Colour Quality Scale* (CQS) computation method.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.quality.plot_colour_quality_bars`,
        :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> from colour import SDS_ILLUMINANTS
    >>> illuminant = SDS_ILLUMINANTS['FL2']
    >>> plot_single_sd_colour_quality_scale_bars(illuminant)
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_\
Plot_Single_SD_Colour_Quality_Scale_Bars.png
        :align: center
        :alt: plot_single_sd_colour_quality_scale_bars
    """

    method = validate_method(method, COLOUR_QUALITY_SCALE_METHODS)

    return plot_multi_sds_colour_quality_scales_bars([sd], method, **kwargs)


@override_style()
def plot_multi_sds_colour_quality_scales_bars(
    sds: Union[
        Sequence[Union[SpectralDistribution, MultiSpectralDistributions]],
        MultiSpectralDistributions,
    ],
    method: Union[
        Literal["NIST CQS 7.4", "NIST CQS 9.0"], str
    ] = "NIST CQS 9.0",
    **kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the *Colour Quality Scale* (CQS) of given illuminants or light
    sources spectral distributions.

    Parameters
    ----------
    sds
        Spectral distributions or multi-spectral distributions to
        plot. `sds` can be a single
        :class:`colour.MultiSpectralDistributions` class instance, a list
        of :class:`colour.MultiSpectralDistributions` class instances or a
        list of :class:`colour.SpectralDistribution` class instances.
    method
        *Colour Quality Scale* (CQS) computation method.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.quality.plot_colour_quality_bars`,
        :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> from colour import (SDS_ILLUMINANTS,
    ...                     SDS_LIGHT_SOURCES)
    >>> illuminant = SDS_ILLUMINANTS['FL2']
    >>> light_source = SDS_LIGHT_SOURCES['Kinoton 75P']
    >>> plot_multi_sds_colour_quality_scales_bars([illuminant, light_source])
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_\
Plot_Multi_SDS_Colour_Quality_Scales_Bars.png
        :align: center
        :alt: plot_multi_sds_colour_quality_scales_bars
    """

    method = validate_method(method, COLOUR_QUALITY_SCALE_METHODS)

    sds_converted = sds_and_msds_to_sds(sds)

    settings: Dict[str, Any] = dict(kwargs)
    settings.update({"standalone": False})

    specifications = cast(
        List[ColourRendering_Specification_CQS],
        [colour_quality_scale(sd, True, method) for sd in sds_converted],
    )

    _figure, axes = plot_colour_quality_bars(specifications, **settings)

    title = (
        f"Colour Quality Scale - "
        f"{', '.join([sd.strict_name for sd in sds_converted])}"
    )

    settings = {"axes": axes, "title": title}
    settings.update(kwargs)

    return render(**settings)
