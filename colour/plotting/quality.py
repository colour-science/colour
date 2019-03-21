# -*- coding: utf-8 -*-
"""
Colour Quality Plotting
=======================

Defines the colour quality plotting objects:

-   :func:`colour.plotting.plot_single_sd_colour_rendering_index_bars`
-   :func:`colour.plotting.plot_multi_sds_colour_rendering_indexes_bars`
-   :func:`colour.plotting.plot_single_sd_colour_quality_scale_bars`
-   :func:`colour.plotting.plot_multi_sds_colour_quality_scales_bars`
"""

from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle

from colour.constants import DEFAULT_FLOAT_DTYPE
from colour.plotting import (COLOUR_STYLE_CONSTANTS,
                             XYZ_to_plotting_colourspace, artist,
                             label_rectangles, override_style, render)
from colour.quality import (colour_quality_scale, colour_rendering_index)
from colour.quality.cri import TCS_ColorimetryData

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'plot_colour_quality_bars', 'plot_single_sd_colour_rendering_index_bars',
    'plot_multi_sds_colour_rendering_indexes_bars',
    'plot_single_sd_colour_quality_scale_bars',
    'plot_multi_sds_colour_quality_scales_bars'
]


@override_style()
def plot_colour_quality_bars(specifications,
                             labels=True,
                             hatching=None,
                             hatching_repeat=2,
                             **kwargs):
    """
    Plots the colour quality data of given illuminants or light sources colour
    quality specifications.

    Parameters
    ----------
    specifications : array_like
        Array of illuminants or light sources colour quality specifications.
    labels : bool, optional
        Add labels above bars.
    hatching : bool or None, optional
        Use hatching for the bars.
    hatching_repeat : int, optional
        Hatching pattern repeat.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.quality.plot_colour_quality_bars`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> from colour import (ILLUMINANTS_SDS,
    ...                     LIGHT_SOURCES_SDS, SpectralShape)
    >>> illuminant = ILLUMINANTS_SDS['FL2']
    >>> light_source = LIGHT_SOURCES_SDS['Kinoton 75P']
    >>> light_source = light_source.copy().align(SpectralShape(360, 830, 1))
    >>> cqs_i = colour_quality_scale(illuminant, additional_data=True)
    >>> cqs_l = colour_quality_scale(light_source, additional_data=True)
    >>> plot_colour_quality_bars([cqs_i, cqs_l])  # doctest: +SKIP

    .. image:: ../_static/Plotting_Plot_Colour_Quality_Bars.png
        :align: center
        :alt: plot_colour_quality_bars
    """

    settings = {'uniform': True}
    settings.update(kwargs)

    _figure, axes = artist(**settings)

    bar_width = 0.5
    y_ticks_interval = 10
    count_s, count_Q_as = len(specifications), 0
    patterns = cycle(COLOUR_STYLE_CONSTANTS.hatch.patterns)
    if hatching is None:
        hatching = False if count_s == 1 else True
    for i, specification in enumerate(specifications):
        Q_a, Q_as, colorimetry_data = (specification.Q_a, specification.Q_as,
                                       specification.colorimetry_data)

        count_Q_as = len(Q_as)
        colours = ([[1] * 3] + [
            np.clip(XYZ_to_plotting_colourspace(x.XYZ), 0, 1)
            for x in colorimetry_data[0]
        ])

        x = (i + np.arange(
            0, (count_Q_as + 1) * (count_s + 1), (count_s + 1),
            dtype=DEFAULT_FLOAT_DTYPE)) * bar_width
        y = [s[1].Q_a for s in sorted(Q_as.items(), key=lambda s: s[0])]
        y = np.array([Q_a] + list(y))

        bars = plt.bar(
            x,
            np.abs(y),
            color=colours,
            width=bar_width,
            edgecolor=COLOUR_STYLE_CONSTANTS.colour.dark,
            label=specification.name)

        hatches = ([next(patterns) * hatching_repeat] * (count_Q_as + 1)
                   if hatching else np.where(y < 0, next(patterns),
                                             None).tolist())

        for j, bar in enumerate(bars.patches):
            bar.set_hatch(hatches[j])

        if labels:
            label_rectangles(
                y,
                bars,
                rotation='horizontal' if count_s == 1 else 'vertical',
                offset=(0 if count_s == 1 else 3 / 100 * count_s + 65 / 1000,
                        0.025),
                text_size=-5 / 7 * count_s + 12.5)

    axes.axhline(
        y=100, color=COLOUR_STYLE_CONSTANTS.colour.dark, linestyle='--')

    axes.set_xticks((np.arange(
        0, (count_Q_as + 1) * (count_s + 1), (count_s + 1),
        dtype=DEFAULT_FLOAT_DTYPE) * bar_width + (count_s * bar_width / 2)),
                    ['Qa'] + [
                        'Q{0}'.format(index + 1)
                        for index in range(0, count_Q_as + 1, 1)
                    ])
    axes.set_yticks(range(0, 100 + y_ticks_interval, y_ticks_interval))

    aspect = 1 / (120 / (bar_width + len(Q_as) + bar_width * 2))
    bounding_box = (-bar_width, ((count_Q_as + 1) * (count_s + 1)) / 2, 0, 120)

    settings = {
        'axes': axes,
        'aspect': aspect,
        'bounding_box': bounding_box,
        'legend': hatching,
        'title': 'Colour Quality',
    }
    settings.update(kwargs)

    return render(**settings)


@override_style()
def plot_single_sd_colour_rendering_index_bars(sd, **kwargs):
    """
    Plots the *Colour Rendering Index* (CRI) of given illuminant or light
    source spectral distribution.

    Parameters
    ----------
    sd : SpectralDistribution
        Illuminant or light source spectral distribution to plot the
        *Colour Rendering Index* (CRI).

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.quality.plot_colour_quality_bars`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.
    labels : bool, optional
        {:func:`colour.plotting.quality.plot_colour_quality_bars`},
        Add labels above bars.
    hatching : bool or None, optional
        {:func:`colour.plotting.quality.plot_colour_quality_bars`},
        Use hatching for the bars.
    hatching_repeat : int, optional
        {:func:`colour.plotting.quality.plot_colour_quality_bars`},
        Hatching pattern repeat.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> from colour import ILLUMINANTS_SDS
    >>> illuminant = ILLUMINANTS_SDS['FL2']
    >>> plot_single_sd_colour_rendering_index_bars(illuminant)
    ... # doctest: +SKIP

    .. image:: ../_static/Plotting_\
Plot_Single_SD_Colour_Rendering_Index_Bars.png
        :align: center
        :alt: plot_single_sd_colour_rendering_index_bars
    """

    return plot_multi_sds_colour_rendering_indexes_bars([sd], **kwargs)


@override_style()
def plot_multi_sds_colour_rendering_indexes_bars(sds, **kwargs):
    """
    Plots the *Colour Rendering Index* (CRI) of given illuminants or light
    sources spectral distributions.

    Parameters
    ----------
    sds : array_like
        Array of illuminants or light sources spectral distributions to
        plot the *Colour Rendering Index* (CRI).

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.quality.plot_colour_quality_bars`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.
    labels : bool, optional
        {:func:`colour.plotting.quality.plot_colour_quality_bars`},
        Add labels above bars.
    hatching : bool or None, optional
        {:func:`colour.plotting.quality.plot_colour_quality_bars`},
        Use hatching for the bars.
    hatching_repeat : int, optional
        {:func:`colour.plotting.quality.plot_colour_quality_bars`},
        Hatching pattern repeat.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> from colour import (ILLUMINANTS_SDS,
    ...                     LIGHT_SOURCES_SDS)
    >>> illuminant = ILLUMINANTS_SDS['FL2']
    >>> light_source = LIGHT_SOURCES_SDS['Kinoton 75P']
    >>> plot_multi_sds_colour_rendering_indexes_bars(
    ...     [illuminant, light_source])  # doctest: +SKIP

    .. image:: ../_static/Plotting_\
Plot_Multi_SDs_Colour_Rendering_Indexes_Bars.png
        :align: center
        :alt: plot_multi_sds_colour_rendering_indexes_bars
    """

    settings = dict(kwargs)
    settings.update({'standalone': False})

    specifications = [
        colour_rendering_index(sd, additional_data=True) for sd in sds
    ]

    # *colour rendering index* colorimetry data tristimulus values are
    # computed in [0, 100] domain however `plot_colour_quality_bars` expects
    # [0, 1] domain. As we want to keep `plot_colour_quality_bars` definition
    # agnostic from the colour quality data, we update the test sd
    # colorimetry data tristimulus values domain.
    for specification in specifications:
        colorimetry_data = specification.colorimetry_data
        for i, c_d in enumerate(colorimetry_data[0]):
            colorimetry_data[0][i] = TCS_ColorimetryData(
                c_d.name, c_d.XYZ / 100, c_d.uv, c_d.UVW)

    _figure, axes = plot_colour_quality_bars(specifications, **settings)

    title = 'Colour Rendering Index - {0}'.format(', '.join(
        [sd.strict_name for sd in sds]))

    settings = {'axes': axes, 'title': title}
    settings.update(kwargs)

    return render(**settings)


@override_style()
def plot_single_sd_colour_quality_scale_bars(sd, **kwargs):
    """
    Plots the *Colour Quality Scale* (CQS) of given illuminant or light source
    spectral distribution.

    Parameters
    ----------
    sd : SpectralDistribution
        Illuminant or light source spectral distribution to plot the
        *Colour Quality Scale* (CQS).

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.quality.plot_colour_quality_bars`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.
    labels : bool, optional
        {:func:`colour.plotting.quality.plot_colour_quality_bars`},
        Add labels above bars.
    hatching : bool or None, optional
        {:func:`colour.plotting.quality.plot_colour_quality_bars`},
        Use hatching for the bars.
    hatching_repeat : int, optional
        {:func:`colour.plotting.quality.plot_colour_quality_bars`},
        Hatching pattern repeat.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> from colour import ILLUMINANTS_SDS
    >>> illuminant = ILLUMINANTS_SDS['FL2']
    >>> plot_single_sd_colour_quality_scale_bars(illuminant)
    ... # doctest: +SKIP

    .. image:: ../_static/Plotting_\
Plot_Single_SD_Colour_Quality_Scale_Bars.png
        :align: center
        :alt: plot_single_sd_colour_quality_scale_bars
    """

    return plot_multi_sds_colour_quality_scales_bars([sd], **kwargs)


@override_style()
def plot_multi_sds_colour_quality_scales_bars(sds, **kwargs):
    """
    Plots the *Colour Quality Scale* (CQS) of given illuminants or light
    sources spectral distributions.

    Parameters
    ----------
    sds : array_like
        Array of illuminants or light sources spectral distributions to
        plot the *Colour Quality Scale* (CQS).

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.quality.plot_colour_quality_bars`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.
    labels : bool, optional
        {:func:`colour.plotting.quality.plot_colour_quality_bars`},
        Add labels above bars.
    hatching : bool or None, optional
        {:func:`colour.plotting.quality.plot_colour_quality_bars`},
        Use hatching for the bars.
    hatching_repeat : int, optional
        {:func:`colour.plotting.quality.plot_colour_quality_bars`},
        Hatching pattern repeat.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> from colour import (ILLUMINANTS_SDS,
    ...                     LIGHT_SOURCES_SDS)
    >>> illuminant = ILLUMINANTS_SDS['FL2']
    >>> light_source = LIGHT_SOURCES_SDS['Kinoton 75P']
    >>> plot_multi_sds_colour_quality_scales_bars([illuminant, light_source])
    ... # doctest: +SKIP

    .. image:: ../_static/Plotting_\
Plot_Multi_SDs_Colour_Quality_Scales_Bars.png
        :align: center
        :alt: plot_multi_sds_colour_quality_scales_bars
    """

    settings = dict(kwargs)
    settings.update({'standalone': False})

    specifications = [
        colour_quality_scale(sd, additional_data=True) for sd in sds
    ]

    _figure, axes = plot_colour_quality_bars(specifications, **settings)

    title = 'Colour Quality Scale - {0}'.format(', '.join(
        [sd.strict_name for sd in sds]))

    settings = {'axes': axes, 'title': title}
    settings.update(kwargs)

    return render(**settings)
