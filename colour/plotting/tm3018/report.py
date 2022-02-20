"""
ANSI/IES TM-30-18 Colour Rendition Report
=========================================

Defines the *ANSI/IES TM-30-18 Colour Rendition Report* plotting objects:

-   :func:`colour.plotting.tm3018.plot_single_sd_colour_rendition_report_full`
-   :func:`colour.plotting.
tm3018.plot_single_sd_colour_rendition_report_intermediate`
-   :func:`colour.plotting.
tm3018.plot_single_sd_colour_rendition_report_simple`
-   :func:`colour.plotting.plot_single_sd_colour_rendition_report`
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from colour.colorimetry import SpectralDistribution, sd_to_XYZ
from colour.hints import Any, Dict, Literal, Optional, Tuple, Union, cast
from colour.io import SpectralDistribution_IESTM2714
from colour.models import XYZ_to_xy, XYZ_to_Luv, Luv_to_uv
from colour.plotting.tm3018.components import (
    plot_spectra_ANSIIESTM3018,
    plot_colour_vector_graphic,
    plot_local_chroma_shifts,
    plot_local_hue_shifts,
    plot_local_colour_fidelities,
    plot_colour_fidelity_indexes,
)
from colour.quality import (
    ColourQuality_Specification_ANSIIESTM3018,
    ColourRendering_Specification_CRI,
    colour_fidelity_index_ANSIIESTM3018,
    colour_rendering_index,
)
from colour.plotting import CONSTANTS_COLOUR_STYLE, override_style, render
from colour.utilities import describe_environment, optional, validate_method

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "CONSTANT_REPORT_SIZE_FULL",
    "CONSTANT_REPORT_ROW_HEIGHT_RATIOS_FULL",
    "CONSTANT_REPORT_PADDING_FULL",
    "CONSTANT_REPORT_SIZE_INTERMEDIATE",
    "CONSTANT_REPORT_ROW_HEIGHT_RATIOS_INTERMEDIATE",
    "CONSTANT_REPORT_PADDING_INTERMEDIATE",
    "CONSTANT_REPORT_SIZE_SIMPLE",
    "CONSTANT_REPORT_ROW_HEIGHT_RATIOS_SIMPLE",
    "CONSTANT_REPORT_PADDING_SIMPLE",
    "CONSTANTS_REPORT_STYLE",
    "CONTENT_REPORT_HEADER",
    "CONTENT_REPORT_FOOTER",
    "plot_single_sd_colour_rendition_report_full",
    "plot_single_sd_colour_rendition_report_intermediate",
    "plot_single_sd_colour_rendition_report_simple",
    "plot_single_sd_colour_rendition_report",
]

# Full Report Size Constants
CONSTANT_REPORT_SIZE_FULL: Tuple = (8.27, 11.69)
"""Full report size, default to A4 paper size in inches."""

CONSTANT_REPORT_ROW_HEIGHT_RATIOS_FULL: Tuple = (1, 2, 24, 3, 1)
"""Full report size row height ratios."""

CONSTANT_REPORT_PADDING_FULL: Dict = {
    "w_pad": 20 / 100,
    "h_pad": 10 / 100,
    "hspace": 0,
    "wspace": 0,
}
"""
Full report box padding, tries to define the padding around the figure and
in-between the axes.
"""

# Intermediate Report Size Constants
CONSTANT_REPORT_SIZE_INTERMEDIATE: Tuple = (8.27, 11.69 / 2.35)
"""Intermediate report size, a window into A4 paper size in inches."""

CONSTANT_REPORT_ROW_HEIGHT_RATIOS_INTERMEDIATE: Tuple = (1, 8, 1)
"""Intermediate report size row height ratios."""

CONSTANT_REPORT_PADDING_INTERMEDIATE: Dict = {
    "w_pad": 20 / 100,
    "h_pad": 10 / 100,
    "hspace": 0,
    "wspace": 0,
}
"""
Intermediate report box padding, tries to define the padding around the figure
and in-between the axes.
"""

# Simple Report Size Constants
CONSTANT_REPORT_SIZE_SIMPLE: Tuple = (8.27, 8.27)
"""Simple report size, a window into A4 paper size in inches."""

CONSTANT_REPORT_ROW_HEIGHT_RATIOS_SIMPLE: Tuple = (1, 8, 1)
"""Simple report size row height ratios."""

CONSTANT_REPORT_PADDING_SIMPLE: Dict = {
    "w_pad": 20 / 100,
    "h_pad": 10 / 100,
    "hspace": 0,
    "wspace": 0,
}
"""
Simple report box padding, tries to define the padding around the figure
and in-between the axes.
"""

CONSTANTS_REPORT_STYLE: Dict = {
    "axes.grid": False,
    "axes.labelpad": CONSTANTS_COLOUR_STYLE.geometry.short * 3,
    "axes.labelsize": "x-small",
    "axes.labelweight": "bold",
    "legend.frameon": False,
    "xtick.labelsize": "x-small",
    "ytick.labelsize": "x-small",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": CONSTANTS_COLOUR_STYLE.geometry.long * 0.5,
    "ytick.major.size": CONSTANTS_COLOUR_STYLE.geometry.long * 0.5,
    "xtick.minor.size": CONSTANTS_COLOUR_STYLE.geometry.long * 0.25,
    "ytick.minor.size": CONSTANTS_COLOUR_STYLE.geometry.long * 0.25,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
}
"""Report style overrides."""

CONTENT_REPORT_HEADER: str = "IES TM-30-18 Colour Rendition Report"
"""Report header content, i.e. the report title."""

CONTENT_REPORT_FOOTER: str = (
    "Colours are for visual orientation purposes only. "
    "Created with Colour{0}"
)
"""Report footer content."""

_VALUE_NOT_APPLICABLE: str = "N/A"


def _plot_report_header(axes: plt.Axes) -> plt.Axes:
    """
    Plot the report header, i.e. the title, on given axes.

    Parameters
    ----------
    axes
        Axes to add the report header to.

    Returns
    -------
    :class:`matplotlib.axes._axes.Axes`
        Axes the report header was added to.
    """

    axes.set_axis_off()
    axes.text(
        0.5,
        0.5,
        CONTENT_REPORT_HEADER,
        ha="center",
        va="center",
        size="x-large",
        weight="bold",
        zorder=CONSTANTS_COLOUR_STYLE.zorder.foreground_label,
    )

    return axes


def _plot_report_footer(axes: plt.Axes) -> plt.Axes:
    """
    Plot the report footer on given axes.

    Parameters
    ----------
    axes
        Axes to add the report footer to.

    Returns
    -------
    :class:`matplotlib.axes._axes.Axes`
        Axes the report footer was added to.
    """

    try:
        describe = describe_environment(print_callable=lambda x: x)[
            "colour-science.org"
        ]["colour"]
        version = f" {describe}."
    except Exception:  # pragma: no cover
        version = "."

    axes.set_axis_off()
    axes.text(
        0.5,
        0.5,
        CONTENT_REPORT_FOOTER.format(version),
        ha="center",
        va="center",
        size="small",
        zorder=CONSTANTS_COLOUR_STYLE.zorder.foreground_label,
    )


@override_style(**CONSTANTS_REPORT_STYLE)
def plot_single_sd_colour_rendition_report_full(
    sd: SpectralDistribution,
    source: Optional[str] = None,
    date: Optional[str] = None,
    manufacturer: Optional[str] = None,
    model: Optional[str] = None,
    notes: Optional[str] = None,
    report_size: Tuple = CONSTANT_REPORT_SIZE_FULL,
    report_row_height_ratios: Tuple = CONSTANT_REPORT_ROW_HEIGHT_RATIOS_FULL,
    report_box_padding: Optional[Dict] = None,
    **kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes]:  # noqa: D405,D407,D410,D411
    """
    Generate the full *ANSI/IES TM-30-18 Colour Rendition Report* for given
    spectral distribution.

    Parameters
    ----------
    sd
        Spectral distribution of the emission source to generate the report
        for.
    source
        Emission source name, defaults to
        `colour.SpectralDistribution_IESTM2714.header.description` or
        `colour.SpectralDistribution_IESTM2714.name` properties value.
    date
        Emission source measurement date, defaults to
        `colour.SpectralDistribution_IESTM2714.header.report_date` property
        value.
    manufacturer
        Emission source manufacturer, defaults to
        `colour.SpectralDistribution_IESTM2714.header.manufacturer` property
        value.
    model
        Emission source model, defaults to
        `colour.SpectralDistribution_IESTM2714.header.catalog_number` property
        value.
    notes
        Notes pertaining to the emission source, defaults to
        `colour.SpectralDistribution_IESTM2714.header.comments` property
        value.
    report_size
        Report size, default to A4 paper size in inches.
    report_row_height_ratios
        Report size row height ratios.
    report_box_padding
        Report box padding, tries to define the padding around the figure and
        in-between the axes.

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
    >>> from colour import SDS_ILLUMINANTS
    >>> sd = SDS_ILLUMINANTS['FL2']
    >>> plot_single_sd_colour_rendition_report_full(sd)
    ... # doctest: +ELLIPSIS
    (<Figure size ... with ... Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_\
Plot_Single_SD_Colour_Rendition_Report_Full.png
        :align: center
        :alt: plot_single_sd_colour_rendition_report_full
    """

    report_box_padding = optional(
        report_box_padding, CONSTANT_REPORT_PADDING_FULL
    )

    specification: ColourQuality_Specification_ANSIIESTM3018 = cast(
        ColourQuality_Specification_ANSIIESTM3018,
        colour_fidelity_index_ANSIIESTM3018(sd, True),
    )

    sd = (
        SpectralDistribution_IESTM2714(data=sd, name=sd.name)
        if not isinstance(sd, SpectralDistribution_IESTM2714)
        else sd
    )

    NA = _VALUE_NOT_APPLICABLE

    source = optional(optional(source, sd.header.description), sd.name)
    date = optional(optional(date, sd.header.report_date), NA)
    manufacturer = optional(optional(manufacturer, sd.header.manufacturer), NA)
    model = optional(optional(model, sd.header.catalog_number), NA)
    notes = optional(optional(notes, sd.header.comments), NA)

    figure = plt.figure(figsize=report_size, constrained_layout=True)

    settings: Dict[str, Any] = dict(kwargs)
    settings["standalone"] = False
    settings["tight_layout"] = False

    gridspec_report = figure.add_gridspec(
        5, 1, height_ratios=report_row_height_ratios
    )

    # Title Row
    gridspec_title = gridspec_report[0].subgridspec(1, 1)
    axes_title = figure.add_subplot(gridspec_title[0])
    _plot_report_header(axes_title)

    # Description Rows & Columns
    gridspec_description = gridspec_report[1].subgridspec(1, 2)
    # Source & Date Column
    axes_source_date = figure.add_subplot(gridspec_description[0])
    axes_source_date.set_axis_off()
    axes_source_date.text(
        0.25,
        2 / 3,
        "Source: ",
        ha="right",
        va="center",
        size="medium",
        weight="bold",
    )
    axes_source_date.text(0.25, 2 / 3, source, va="center", size="medium")

    axes_source_date.text(
        0.25,
        1 / 3,
        "Date: ",
        ha="right",
        va="center",
        size="medium",
        weight="bold",
    )
    axes_source_date.text(0.25, 1 / 3, date, va="center", size="medium")

    # Manufacturer & Model Column
    axes_manufacturer_model = figure.add_subplot(gridspec_description[1])
    axes_manufacturer_model.set_axis_off()
    axes_manufacturer_model.text(
        0.25,
        2 / 3,
        "Manufacturer: ",
        ha="right",
        va="center",
        size="medium",
        weight="bold",
    )
    axes_manufacturer_model.text(
        0.25, 2 / 3, manufacturer, va="center", size="medium"
    )

    axes_manufacturer_model.text(
        0.25,
        1 / 3,
        "Model: ",
        ha="right",
        va="center",
        size="medium",
        weight="bold",
    )
    axes_manufacturer_model.text(
        0.25, 1 / 3, model, va="center", size="medium"
    )

    # Main Figures Rows & Columns
    gridspec_figures = gridspec_report[2].subgridspec(
        4, 2, height_ratios=[1, 1, 1, 1.5]
    )
    axes_spectra = figure.add_subplot(gridspec_figures[0, 0])
    plot_spectra_ANSIIESTM3018(specification, axes=axes_spectra, **settings)

    axes_vector_graphics = figure.add_subplot(gridspec_figures[1:3, 0])
    plot_colour_vector_graphic(
        specification, axes=axes_vector_graphics, **settings
    )

    axes_chroma_shifts = figure.add_subplot(gridspec_figures[0, 1])
    plot_local_chroma_shifts(
        specification, axes=axes_chroma_shifts, **settings
    )

    axes_hue_shifts = figure.add_subplot(gridspec_figures[1, 1])
    plot_local_hue_shifts(specification, axes=axes_hue_shifts, **settings)

    axes_colour_fidelities = figure.add_subplot(gridspec_figures[2, 1])
    plot_local_colour_fidelities(
        specification, axes=axes_colour_fidelities, x_ticker=True, **settings
    )

    # Colour Fidelity Indexes Row
    axes_colour_fidelity_indexes = figure.add_subplot(gridspec_figures[3, :])
    plot_colour_fidelity_indexes(
        specification, axes=axes_colour_fidelity_indexes, **settings
    )

    # Notes & Chromaticities / CRI Row and Columns
    gridspec_notes_chromaticities_CRI = gridspec_report[3].subgridspec(1, 2)
    axes_notes = figure.add_subplot(gridspec_notes_chromaticities_CRI[0])
    axes_notes.set_axis_off()
    axes_notes.text(
        0.25,
        1,
        "Notes: ",
        ha="right",
        va="center",
        size="medium",
        weight="bold",
    )
    axes_notes.text(0.25, 1, notes, va="center", size="medium")
    gridspec_chromaticities_CRI = gridspec_notes_chromaticities_CRI[
        1
    ].subgridspec(1, 2)

    XYZ = sd_to_XYZ(specification.sd_test)
    xy = XYZ_to_xy(XYZ)
    Luv = XYZ_to_Luv(XYZ, xy)
    uv_p = Luv_to_uv(Luv, xy)

    gridspec_chromaticities = gridspec_chromaticities_CRI[0].subgridspec(1, 1)
    axes_chromaticities = figure.add_subplot(gridspec_chromaticities[0])
    axes_chromaticities.set_axis_off()
    axes_chromaticities.text(
        0.5,
        4 / 5,
        f"$x$ {xy[0]:.4f}",
        ha="center",
        va="center",
        size="medium",
        weight="bold",
    )

    axes_chromaticities.text(
        0.5,
        3 / 5,
        f"$y$ {xy[1]:.4f}",
        ha="center",
        va="center",
        size="medium",
        weight="bold",
    )

    axes_chromaticities.text(
        0.5,
        2 / 5,
        f"$u'$ {uv_p[0]:.4f}",
        ha="center",
        va="center",
        size="medium",
        weight="bold",
    )

    axes_chromaticities.text(
        0.5,
        1 / 5,
        f"$v'$ {uv_p[1]:.4f}",
        ha="center",
        va="center",
        size="medium",
        weight="bold",
    )

    gridspec_CRI = gridspec_chromaticities_CRI[1].subgridspec(1, 1)

    CRI_spec: ColourRendering_Specification_CRI = cast(
        ColourRendering_Specification_CRI,
        colour_rendering_index(specification.sd_test, additional_data=True),
    )

    axes_CRI = figure.add_subplot(gridspec_CRI[0])
    axes_CRI.set_xticks([])
    axes_CRI.set_yticks([])
    axes_CRI.text(
        0.5,
        4 / 5,
        "CIE 13.31-1995",
        ha="center",
        va="center",
        size="medium",
        weight="bold",
    )

    axes_CRI.text(
        0.5,
        3 / 5,
        "(CRI)",
        ha="center",
        va="center",
        size="medium",
        weight="bold",
    )

    axes_CRI.text(
        0.5,
        2 / 5,
        f"$R_a$ {float(CRI_spec.Q_a):.0f}",
        ha="center",
        va="center",
        size="medium",
        weight="bold",
    )

    axes_CRI.text(
        0.5,
        1 / 5,
        f"$R_9$ {float(CRI_spec.Q_as[8].Q_a):.0f}",
        ha="center",
        va="center",
        size="medium",
        weight="bold",
    )

    gridspec_footer = gridspec_report[4].subgridspec(1, 1)
    axes_footer = figure.add_subplot(gridspec_footer[0])
    _plot_report_footer(axes_footer)

    figure.set_constrained_layout_pads(**report_box_padding)

    settings = dict(kwargs)
    settings["tight_layout"] = False

    return render(**settings)


@override_style(**CONSTANTS_REPORT_STYLE)
def plot_single_sd_colour_rendition_report_intermediate(
    sd: SpectralDistribution,
    report_size: Tuple = CONSTANT_REPORT_SIZE_INTERMEDIATE,
    report_row_height_ratios: Tuple = (
        CONSTANT_REPORT_ROW_HEIGHT_RATIOS_INTERMEDIATE
    ),
    report_box_padding: Optional[Dict] = None,
    **kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Generate the intermediate *ANSI/IES TM-30-18 Colour Rendition Report* for
    given spectral distribution.

    Parameters
    ----------
    sd
        Spectral distribution of the emission source to generate the report
        for.
    report_size
        Report size, default to A4 paper size in inches.
    report_row_height_ratios
        Report size row height ratios.
    report_box_padding
        Report box padding, tries to define the padding around the figure and
        in-between the axes.

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
    >>> from colour import SDS_ILLUMINANTS
    >>> sd = SDS_ILLUMINANTS['FL2']
    >>> plot_single_sd_colour_rendition_report_intermediate(sd)
    ... # doctest: +ELLIPSIS
    (<Figure size ... with ... Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_\
Plot_Single_SD_Colour_Rendition_Report_Intermediate.png
        :align: center
        :alt: plot_single_sd_colour_rendition_report_intermediate
    """

    report_box_padding = optional(
        report_box_padding, CONSTANT_REPORT_PADDING_INTERMEDIATE
    )

    specification: ColourQuality_Specification_ANSIIESTM3018 = cast(
        ColourQuality_Specification_ANSIIESTM3018,
        colour_fidelity_index_ANSIIESTM3018(sd, True),
    )

    figure = plt.figure(figsize=report_size, constrained_layout=True)

    settings: Dict[str, Any] = dict(kwargs)
    settings["standalone"] = False
    settings["tight_layout"] = False

    gridspec_report = figure.add_gridspec(
        3, 1, height_ratios=report_row_height_ratios
    )

    # Title Row
    gridspec_title = gridspec_report[0].subgridspec(1, 1)
    axes_title = figure.add_subplot(gridspec_title[0])
    _plot_report_header(axes_title)

    # Main Figures Rows & Columns
    gridspec_figures = gridspec_report[1].subgridspec(2, 2)

    axes_vector_graphics = figure.add_subplot(gridspec_figures[0:2, 0])
    plot_colour_vector_graphic(
        specification, axes=axes_vector_graphics, **settings
    )

    axes_chroma_shifts = figure.add_subplot(gridspec_figures[0, 1])
    plot_local_chroma_shifts(
        specification, axes=axes_chroma_shifts, **settings
    )

    axes_hue_shifts = figure.add_subplot(gridspec_figures[1, 1])
    plot_local_hue_shifts(
        specification, axes=axes_hue_shifts, x_ticker=True, **settings
    )

    gridspec_footer = gridspec_report[2].subgridspec(1, 1)
    axes_footer = figure.add_subplot(gridspec_footer[0])
    _plot_report_footer(axes_footer)

    figure.set_constrained_layout_pads(**report_box_padding)

    settings = dict(kwargs)
    settings["tight_layout"] = False

    return render(**settings)


def plot_single_sd_colour_rendition_report_simple(
    sd: SpectralDistribution,
    report_size: Tuple = CONSTANT_REPORT_SIZE_SIMPLE,
    report_row_height_ratios: Tuple = CONSTANT_REPORT_ROW_HEIGHT_RATIOS_SIMPLE,
    report_box_padding: Optional[Dict] = None,
    **kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Generate the simple *ANSI/IES TM-30-18 Colour Rendition Report* for given
    spectral distribution.

    Parameters
    ----------
    sd
        Spectral distribution of the emission source to generate the report
        for.
    report_size
        Report size, default to A4 paper size in inches.
    report_row_height_ratios
        Report size row height ratios.
    report_box_padding
        Report box padding, tries to define the padding around the figure and
        in-between the axes.

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
    >>> from colour import SDS_ILLUMINANTS
    >>> sd = SDS_ILLUMINANTS['FL2']
    >>> plot_single_sd_colour_rendition_report_simple(sd)
    ... # doctest: +ELLIPSIS
    (<Figure size ... with ... Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_\
Plot_Single_SD_Colour_Rendition_Report_Simple.png
        :align: center
        :alt: plot_single_sd_colour_rendition_report_simple
    """

    report_box_padding = optional(
        report_box_padding, CONSTANT_REPORT_PADDING_SIMPLE
    )

    specification: ColourQuality_Specification_ANSIIESTM3018 = cast(
        ColourQuality_Specification_ANSIIESTM3018,
        colour_fidelity_index_ANSIIESTM3018(sd, True),
    )

    figure = plt.figure(figsize=report_size, constrained_layout=True)

    settings: Dict[str, Any] = dict(kwargs)
    settings["standalone"] = False
    settings["tight_layout"] = False

    gridspec_report = figure.add_gridspec(
        3, 1, height_ratios=report_row_height_ratios
    )

    # Title Row
    gridspec_title = gridspec_report[0].subgridspec(1, 1)
    axes_title = figure.add_subplot(gridspec_title[0])
    _plot_report_header(axes_title)

    # Main Figures Rows & Columns
    gridspec_figures = gridspec_report[1].subgridspec(1, 1)

    axes_vector_graphics = figure.add_subplot(gridspec_figures[0, 0])
    plot_colour_vector_graphic(
        specification, axes=axes_vector_graphics, **settings
    )

    gridspec_footer = gridspec_report[2].subgridspec(1, 1)
    axes_footer = figure.add_subplot(gridspec_footer[0])
    _plot_report_footer(axes_footer)

    figure.set_constrained_layout_pads(**report_box_padding)

    settings = dict(kwargs)
    settings["tight_layout"] = False

    return render(**settings)


def plot_single_sd_colour_rendition_report(
    sd: SpectralDistribution,
    method: Union[Literal["Full", "Intermediate", "Simple"], str] = "Full",
    **kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Generate the *ANSI/IES TM-30-18 Colour Rendition Report* for given
    spectral distribution according to given method.

    Parameters
    ----------
    sd
        Spectral distribution of the emission source to generate the report
        for.
    method
        Report plotting method.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`, :func:`colour.plotting.render`,
        :func:`colour.plotting.tm3018.\
plot_single_sd_colour_rendition_report_full`, :func:`colour.plotting.tm3018.\
plot_single_sd_colour_rendition_report_intermediate`, \
:func:`colour.plotting.tm3018.plot_single_sd_colour_rendition_report_simple`}
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> from colour import SDS_ILLUMINANTS
    >>> sd = SDS_ILLUMINANTS['FL2']
    >>> plot_single_sd_colour_rendition_report(sd)
    ... # doctest: +ELLIPSIS
    (<Figure size ... with ... Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_\
Plot_Single_SD_Colour_Rendition_Report_Full.png
        :align: center
        :alt: plot_single_sd_colour_rendition_report_full

    >>> plot_single_sd_colour_rendition_report(sd, 'Intermediate')
    ... # doctest: +ELLIPSIS
    (<Figure size ... with ... Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_\
Plot_Single_SD_Colour_Rendition_Report_Intermediate.png
        :align: center
        :alt: plot_single_sd_colour_rendition_report_intermediate

    >>> plot_single_sd_colour_rendition_report(sd, 'Simple')
    ... # doctest: +ELLIPSIS
    (<Figure size ... with ... Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_\
Plot_Single_SD_Colour_Rendition_Report_Simple.png
        :align: center
        :alt: plot_single_sd_colour_rendition_report_simple
    """

    method = validate_method(method, ["Full", "Intermediate", "Simple"])

    if method == "full":
        return plot_single_sd_colour_rendition_report_full(sd, **kwargs)
    elif method == "intermediate":
        return plot_single_sd_colour_rendition_report_intermediate(
            sd, **kwargs
        )
    else:  # method == 'simple'
        return plot_single_sd_colour_rendition_report_simple(sd, **kwargs)
