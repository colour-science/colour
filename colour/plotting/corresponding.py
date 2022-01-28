# -*- coding: utf-8 -*-
"""
Corresponding Chromaticities Prediction Plotting
================================================

Defines the corresponding chromaticities prediction plotting objects:

-   :func:`colour.plotting.plot_corresponding_chromaticities_prediction`
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from colour.corresponding import (
    CorrespondingColourDataset,
    corresponding_chromaticities_prediction,
)
from colour.hints import Any, Dict, Literal, Tuple, Union
from colour.plotting import (
    CONSTANTS_COLOUR_STYLE,
    artist,
    plot_chromaticity_diagram_CIE1976UCS,
    override_style,
    render,
)
from colour.utilities import is_numeric

__author__ = "Colour Developers"
__copyright__ = "Copyright (C) 2013-2021 - Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "plot_corresponding_chromaticities_prediction",
]


@override_style()
def plot_corresponding_chromaticities_prediction(
    experiment: Union[
        Literal[1, 2, 3, 4, 6, 8, 9, 11, 12], CorrespondingColourDataset
    ] = 1,
    model: Union[
        Literal["CIE 1994", "CMCCAT2000", "Fairchild 1990", "Von Kries"], str
    ] = "Von Kries",
    chromatic_adaptation_transform: Union[
        Literal[
            "Bianco 2010",
            "Bianco PC 2010",
            "Bradford",
            "CAT02 Brill 2008",
            "CAT02",
            "CAT16",
            "CMCCAT2000",
            "CMCCAT97",
            "Fairchild",
            "Sharp",
            "Von Kries",
            "XYZ Scaling",
        ],
        str,
    ] = "CAT02",
    **kwargs: Any
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots given chromatic adaptation model corresponding chromaticities
    prediction.

    Parameters
    ----------
    experiment
        *Breneman (1987)* experiment number or
        :class:`colour.CorrespondingColourDataset` class instance.
    model
        Corresponding chromaticities prediction model name.
    chromatic_adaptation_transform
        Transformation to use with *Von Kries* chromatic adaptation model.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.diagrams.plot_chromaticity_diagram`,
        :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> plot_corresponding_chromaticities_prediction(1, 'Von Kries', 'CAT02')
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_\
Plot_Corresponding_Chromaticities_Prediction.png
        :align: center
        :alt: plot_corresponding_chromaticities_prediction
    """

    settings: Dict[str, Any] = {"uniform": True}
    settings.update(kwargs)

    _figure, axes = artist(**settings)

    name = (
        "Experiment {0}".format(experiment)
        if is_numeric(experiment)
        else experiment.name  # type: ignore[union-attr]
    )
    title = (
        (
            "Corresponding Chromaticities Prediction - {0} ({1}) - {2} - "
            "CIE 1976 UCS Chromaticity Diagram"
        ).format(model, chromatic_adaptation_transform, name)
        if model.lower() in ("von kries", "vonkries")
        else (
            "Corresponding Chromaticities Prediction - {0} - {1} - "
            "CIE 1976 UCS Chromaticity Diagram"
        ).format(model, name)
    )

    settings = {"axes": axes, "title": title}
    settings.update(kwargs)
    settings["standalone"] = False

    plot_chromaticity_diagram_CIE1976UCS(**settings)

    results = corresponding_chromaticities_prediction(
        experiment, transform=chromatic_adaptation_transform
    )

    for result in results:
        _name, uv_t, uv_m, uv_p = result
        axes.arrow(
            uv_t[0],
            uv_t[1],
            uv_p[0] - uv_t[0] - 0.1 * (uv_p[0] - uv_t[0]),
            uv_p[1] - uv_t[1] - 0.1 * (uv_p[1] - uv_t[1]),
            color=CONSTANTS_COLOUR_STYLE.colour.dark,
            head_width=0.005,
            head_length=0.005,
        )
        axes.plot(
            uv_t[0],
            uv_t[1],
            "o",
            color=CONSTANTS_COLOUR_STYLE.colour.brightest,
            markeredgecolor=CONSTANTS_COLOUR_STYLE.colour.dark,
            markersize=(
                CONSTANTS_COLOUR_STYLE.geometry.short * 6
                + CONSTANTS_COLOUR_STYLE.geometry.short * 0.75
            ),
            markeredgewidth=CONSTANTS_COLOUR_STYLE.geometry.short * 0.75,
        )
        axes.plot(
            uv_m[0],
            uv_m[1],
            "^",
            color=CONSTANTS_COLOUR_STYLE.colour.brightest,
            markeredgecolor=CONSTANTS_COLOUR_STYLE.colour.dark,
            markersize=(
                CONSTANTS_COLOUR_STYLE.geometry.short * 6
                + CONSTANTS_COLOUR_STYLE.geometry.short * 0.75
            ),
            markeredgewidth=CONSTANTS_COLOUR_STYLE.geometry.short * 0.75,
        )
        axes.plot(
            uv_p[0], uv_p[1], "^", color=CONSTANTS_COLOUR_STYLE.colour.dark
        )

    settings.update(
        {
            "standalone": True,
            "bounding_box": (-0.1, 0.7, -0.1, 0.7),
        }
    )
    settings.update(kwargs)

    return render(**settings)
