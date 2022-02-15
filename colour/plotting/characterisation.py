"""
Characterisation Plotting
=========================

Defines the characterisation plotting objects:

-   :func:`colour.plotting.plot_single_colour_checker`
-   :func:`colour.plotting.plot_multi_colour_checkers`
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from colour.hints import Any, Dict, Sequence, Tuple, Union
from colour.characterisation import ColourChecker
from colour.models import xyY_to_XYZ
from colour.plotting import (
    CONSTANTS_COLOUR_STYLE,
    ColourSwatch,
    XYZ_to_plotting_colourspace,
    artist,
    filter_colour_checkers,
    plot_multi_colour_swatches,
    override_style,
    render,
)
from colour.utilities import attest

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "plot_single_colour_checker",
    "plot_multi_colour_checkers",
]


@override_style(
    **{
        "axes.grid": False,
        "xtick.bottom": False,
        "ytick.left": False,
        "xtick.labelbottom": False,
        "ytick.labelleft": False,
    }
)
def plot_single_colour_checker(
    colour_checker: Union[
        ColourChecker, str
    ] = "ColorChecker24 - After November 2014",
    **kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot given colour checker.

    Parameters
    ----------
    colour_checker
        Color checker to plot. ``colour_checker`` can be of any type or form
        supported by the
        :func:`colour.plotting.filter_colour_checkers` definition.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.plot_multi_colour_swatches`,
        :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> plot_single_colour_checker('ColorChecker 2005')  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_Single_Colour_Checker.png
        :align: center
        :alt: plot_single_colour_checker
    """

    return plot_multi_colour_checkers([colour_checker], **kwargs)


@override_style(
    **{
        "axes.grid": False,
        "xtick.bottom": False,
        "ytick.left": False,
        "xtick.labelbottom": False,
        "ytick.labelleft": False,
    }
)
def plot_multi_colour_checkers(
    colour_checkers: Union[
        ColourChecker, str, Sequence[Union[ColourChecker, str]]
    ],
    **kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot and compares given colour checkers.

    Parameters
    ----------
    colour_checkers
        Color checker to plot, count must be less than or equal to 2.
        ``colour_checkers`` elements can be of any type or form supported by
        the :func:`colour.plotting.filter_colour_checkers` definition.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.plot_multi_colour_swatches`,
        :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> plot_multi_colour_checkers(['ColorChecker 1976', 'ColorChecker 2005'])
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_Multi_Colour_Checkers.png
        :align: center
        :alt: plot_multi_colour_checkers
    """

    filtered_colour_checkers = list(
        filter_colour_checkers(colour_checkers).values()
    )

    attest(
        len(filtered_colour_checkers) <= 2,
        "Only two colour checkers can be compared at a time!",
    )

    _figure, axes = artist(**kwargs)

    compare_swatches = len(filtered_colour_checkers) == 2

    colour_swatches = []
    colour_checker_names = []
    for colour_checker in filtered_colour_checkers:
        colour_checker_names.append(colour_checker.name)
        for label, xyY in colour_checker.data.items():
            XYZ = xyY_to_XYZ(xyY)
            RGB = XYZ_to_plotting_colourspace(XYZ, colour_checker.illuminant)
            colour_swatches.append(
                ColourSwatch(np.clip(np.ravel(RGB), 0, 1), label.title())
            )

    if compare_swatches:
        colour_swatches = [
            swatch
            for pairs in zip(
                colour_swatches[0 : len(colour_swatches) // 2],
                colour_swatches[len(colour_swatches) // 2 :],
            )
            for swatch in pairs
        ]

    background_colour = "0.1"
    width = height = 1.0
    spacing = 0.25
    columns = 6

    settings: Dict[str, Any] = {
        "axes": axes,
        "width": width,
        "height": height,
        "spacing": spacing,
        "columns": columns,
        "direction": "-y",
        "text_kwargs": {"size": 8},
        "background_colour": background_colour,
        "compare_swatches": "Stacked" if compare_swatches else None,
    }
    settings.update(kwargs)
    settings["standalone"] = False

    plot_multi_colour_swatches(colour_swatches, **settings)

    axes.text(
        0.5,
        0.005,
        (
            f"{', '.join(colour_checker_names)} - "
            f"{CONSTANTS_COLOUR_STYLE.colour.colourspace.name} - "
            f"Colour Rendition Chart"
        ),
        transform=axes.transAxes,
        color=CONSTANTS_COLOUR_STYLE.colour.bright,
        ha="center",
        va="bottom",
        zorder=CONSTANTS_COLOUR_STYLE.zorder.midground_label,
    )

    settings.update(
        {
            "axes": axes,
            "standalone": True,
            "title": ", ".join(colour_checker_names),
        }
    )

    return render(**settings)
