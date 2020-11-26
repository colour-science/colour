# -*- coding: utf-8 -*-
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

from __future__ import division

import matplotlib.pyplot as plt
import six

from colour.colorimetry import sd_to_XYZ
from colour.io import SpectralDistribution_IESTM2714
from colour.models import XYZ_to_xy, XYZ_to_Luv, Luv_to_uv
from colour.plotting.tm3018.components import (
    plot_spectra_ANSIIESTM3018, plot_colour_vector_graphic,
    plot_local_chroma_shifts, plot_local_hue_shifts,
    plot_local_colour_fidelities, plot_colour_fidelity_indexes)
from colour.quality import (colour_fidelity_index_ANSIIESTM3018,
                            colour_rendering_index)
from colour.plotting import CONSTANTS_COLOUR_STYLE, override_style, render
from colour.utilities import describe_environment, runtime_warning

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'CONSTANT_REPORT_SIZE_FULL', 'CONSTANT_REPORT_ROW_HEIGHT_RATIOS_FULL',
    'CONSTANT_REPORT_PADDING_FULL', 'CONSTANT_REPORT_SIZE_INTERMEDIATE',
    'CONSTANT_REPORT_ROW_HEIGHT_RATIOS_INTERMEDIATE',
    'CONSTANT_REPORT_PADDING_INTERMEDIATE', 'CONSTANT_REPORT_SIZE_SIMPLE',
    'CONSTANT_REPORT_ROW_HEIGHT_RATIOS_SIMPLE',
    'CONSTANT_REPORT_PADDING_SIMPLE', 'CONSTANTS_REPORT_STYLE',
    'REPORT_HEADER_CONTENT', 'REPORT_FOOTER_CONTENT',
    'plot_single_sd_colour_rendition_report_full',
    'plot_single_sd_colour_rendition_report_intermediate',
    'plot_single_sd_colour_rendition_report_simple',
    'plot_single_sd_colour_rendition_report'
]

# Full Report Size Constants
CONSTANT_REPORT_SIZE_FULL = (8.27, 11.69)
"""
Full report size, default to A4 paper size in inches.

CONSTANT_REPORT_SIZE_FULL : tuple
"""

CONSTANT_REPORT_ROW_HEIGHT_RATIOS_FULL = (1, 2, 24, 3, 1)
"""
Full report size row height ratios.

CONSTANT_REPORT_ROW_HEIGHT_RATIOS_FULL : tuple
"""

CONSTANT_REPORT_PADDING_FULL = {
    'w_pad': 20 / 100,
    'h_pad': 10 / 100,
    'hspace': 0,
    'wspace': 0,
}
"""
Full report box padding, tries to define the padding around the figure and
in-between the axes.

CONSTANT_REPORT_PADDING_FULL : dict
"""

# Intermediate Report Size Constants
CONSTANT_REPORT_SIZE_INTERMEDIATE = (8.27, 11.69 / 2.35)
"""
Intermediate report size, a window into A4 paper size in inches.

CONSTANT_REPORT_SIZE_INTERMEDIATE : tuple
"""

CONSTANT_REPORT_ROW_HEIGHT_RATIOS_INTERMEDIATE = (1, 8, 1)
"""
Intermediate report size row height ratios.

CONSTANT_REPORT_ROW_HEIGHT_RATIOS_INTERMEDIATE : tuple
"""

CONSTANT_REPORT_PADDING_INTERMEDIATE = {
    'w_pad': 20 / 100,
    'h_pad': 10 / 100,
    'hspace': 0,
    'wspace': 0,
}
"""
Intermediate report box padding, tries to define the padding around the figure
and in-between the axes.

CONSTANT_REPORT_PADDING_INTERMEDIATE : dict
"""

# Simple Report Size Constants
CONSTANT_REPORT_SIZE_SIMPLE = (8.27, 8.27)
"""
Simple report size, a window into A4 paper size in inches.

CONSTANT_REPORT_SIZE_SIMPLE : tuple
"""

CONSTANT_REPORT_ROW_HEIGHT_RATIOS_SIMPLE = (1, 8, 1)
"""
Simple report size row height ratios.

CONSTANT_REPORT_ROW_HEIGHT_RATIOS_SIMPLE : tuple
"""

CONSTANT_REPORT_PADDING_SIMPLE = {
    'w_pad': 20 / 100,
    'h_pad': 10 / 100,
    'hspace': 0,
    'wspace': 0,
}
"""
Simple report box padding, tries to define the padding around the figure
and in-between the axes.

CONSTANT_REPORT_PADDING_SIMPLE : dict
"""

CONSTANTS_REPORT_STYLE = {
    'axes.grid': False,
    'axes.labelpad': CONSTANTS_COLOUR_STYLE.geometry.short * 3,
    'axes.labelsize': 'x-small',
    'axes.labelweight': 'bold',
    'legend.frameon': False,
    'xtick.labelsize': 'x-small',
    'ytick.labelsize': 'x-small',
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': CONSTANTS_COLOUR_STYLE.geometry.long * 0.5,
    'ytick.major.size': CONSTANTS_COLOUR_STYLE.geometry.long * 0.5,
    'xtick.minor.size': CONSTANTS_COLOUR_STYLE.geometry.long * 0.25,
    'ytick.minor.size': CONSTANTS_COLOUR_STYLE.geometry.long * 0.25,
    'xtick.minor.visible': False,
    'ytick.minor.visible': False,
}
"""
Report style overrides.

CONSTANTS_REPORT_STYLE : dict
"""

REPORT_HEADER_CONTENT = 'IES TM-30-18 Colour Rendition Report'
"""
Report header content, i.e. the report title.

REPORT_HEADER_CONTENT : unicode
"""

try:
    _COLOUR_VERSION = ' {0}.'.format(
        describe_environment(
            print_callable=lambda x: x)['colour-science.org']['colour'])
except Exception:  # noqa
    _COLOUR_VERSION = '.'

REPORT_FOOTER_CONTENT = ('Colours are for visual orientation purposes only. '
                         'Created with Colour{0}').format(_COLOUR_VERSION)
"""
Report footer content.

REPORT_FOOTER_CONTENT : unicode
"""

_NOT_APPLICABLE_VALUE = 'N/A'


def _plot_report_header(axes):
    """
    Plots the report header, i.e. the title, on given axes.

    Parameters
    ----------
    axes : Axes
        Axes to add the report header to.

    Returns
    -------
    Axes
        Axes the report header was added to.
    """

    axes.set_axis_off()
    axes.text(
        0.5,
        0.5,
        REPORT_HEADER_CONTENT,
        ha='center',
        va='center',
        size='x-large',
        weight='bold')

    return axes


def _plot_report_footer(axes):
    """
    Plots the report footer on given axes.

    Parameters
    ----------
    axes : Axes
        Axes to add the report footer to.

    Returns
    -------
    Axes
        Axes the report footer was added to.
    """

    axes.set_axis_off()
    axes.text(
        0.5,
        0.5,
        REPORT_FOOTER_CONTENT,
        ha='center',
        va='center',
        size='small')


@override_style(**CONSTANTS_REPORT_STYLE)
def plot_single_sd_colour_rendition_report_full(
        sd,
        source=None,
        date=None,
        manufacturer=None,
        model=None,
        notes=None,
        report_size=CONSTANT_REPORT_SIZE_FULL,
        report_row_height_ratios=CONSTANT_REPORT_ROW_HEIGHT_RATIOS_FULL,
        report_box_padding=None,
        **kwargs):
    """
    Generates the full *ANSI/IES TM-30-18 Colour Rendition Report* for given
    spectral distribution.

    Parameters
    ----------
    sd : SpectralDistribution or SpectralDistribution_IESTM2714
        Spectral distribution of the emission source to generate the report
        for.
    source : unicode, optional
        Emission source name, defaults to
        `colour.SpectralDistribution_IESTM2714.header.description` or
        `colour.SpectralDistribution_IESTM2714.name` properties value.
    date : unicode, optional
        Emission source measurement date, defaults to
        `colour.SpectralDistribution_IESTM2714.header.report_date` property
        value.
    manufacturer : unicode, optional
        Emission source manufacturer, defaults to
        `colour.SpectralDistribution_IESTM2714.header.manufacturer` property
        value.
    model : unicode, optional
        Emission source model, defaults to
        `colour.SpectralDistribution_IESTM2714.header.catalog_number` property
        value.
    notes : unicode, optional
        Notes pertaining to the emission source, defaults to
        `colour.SpectralDistribution_IESTM2714.header.comments` property
        value.
    report_size : array_like, optional
        Report size, default to A4 paper size in inches.
    report_row_height_ratios : array_like, optional
        Report size row height ratios.
    report_box_padding : array_like, optional
        Report box padding, tries to define the padding around the figure and
        in-between the axes.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`, :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
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

    if six.PY2:
        runtime_warning(
            'The "ANSI/IES TM-30-18 Colour Rendition Report" uses advanced '
            '"Matplotlib" layout capabilities only available for Python 3!')

        return render()

    if report_box_padding is None:
        report_box_padding = CONSTANT_REPORT_PADDING_FULL

    specification = colour_fidelity_index_ANSIIESTM3018(sd, True)

    sd = (SpectralDistribution_IESTM2714(data=sd, name=sd.name)
          if not isinstance(sd, SpectralDistribution_IESTM2714) else sd)

    source = sd.header.description if source is None else source
    source = sd.name if source is None else source
    date = sd.header.report_date if date is None else date
    date = _NOT_APPLICABLE_VALUE if date is None else date
    manufacturer = (sd.header.manufacturer
                    if manufacturer is None else manufacturer)
    manufacturer = (_NOT_APPLICABLE_VALUE
                    if manufacturer is None else manufacturer)
    model = sd.header.catalog_number if model is None else model
    model = _NOT_APPLICABLE_VALUE if model is None else model
    notes = sd.header.comments if notes is None else notes
    notes = _NOT_APPLICABLE_VALUE if notes is None else notes

    figure = plt.figure(figsize=report_size, constrained_layout=True)

    settings = kwargs.copy()
    settings['standalone'] = False
    settings['tight_layout'] = False

    gridspec_report = figure.add_gridspec(
        5, 1, height_ratios=report_row_height_ratios)

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
        'Source: ',
        ha='right',
        va='center',
        size='medium',
        weight='bold')
    axes_source_date.text(0.25, 2 / 3, source, va='center', size='medium')

    axes_source_date.text(
        0.25,
        1 / 3,
        'Date: ',
        ha='right',
        va='center',
        size='medium',
        weight='bold')
    axes_source_date.text(0.25, 1 / 3, date, va='center', size='medium')

    # Manufacturer & Model Column
    axes_manufacturer_model = figure.add_subplot(gridspec_description[1])
    axes_manufacturer_model.set_axis_off()
    axes_manufacturer_model.text(
        0.25,
        2 / 3,
        'Manufacturer: ',
        ha='right',
        va='center',
        size='medium',
        weight='bold')
    axes_manufacturer_model.text(
        0.25, 2 / 3, manufacturer, va='center', size='medium')

    axes_manufacturer_model.text(
        0.25,
        1 / 3,
        'Model: ',
        ha='right',
        va='center',
        size='medium',
        weight='bold')
    axes_manufacturer_model.text(
        0.25, 1 / 3, model, va='center', size='medium')

    # Main Figures Rows & Columns
    gridspec_figures = gridspec_report[2].subgridspec(
        4, 2, height_ratios=[1, 1, 1, 1.5])
    axes_spectra = figure.add_subplot(gridspec_figures[0, 0])
    plot_spectra_ANSIIESTM3018(specification, axes=axes_spectra, **settings)

    axes_vector_graphics = figure.add_subplot(gridspec_figures[1:3, 0])
    plot_colour_vector_graphic(
        specification, axes=axes_vector_graphics, **settings)

    axes_chroma_shifts = figure.add_subplot(gridspec_figures[0, 1])
    plot_local_chroma_shifts(
        specification, axes=axes_chroma_shifts, **settings)

    axes_hue_shifts = figure.add_subplot(gridspec_figures[1, 1])
    plot_local_hue_shifts(specification, axes=axes_hue_shifts, **settings)

    axes_colour_fidelities = figure.add_subplot(gridspec_figures[2, 1])
    plot_local_colour_fidelities(
        specification, axes=axes_colour_fidelities, x_ticker=True, **settings)

    # Colour Fidelity Indexes Row
    axes_colour_fidelity_indexes = figure.add_subplot(gridspec_figures[3, :])
    plot_colour_fidelity_indexes(
        specification, axes=axes_colour_fidelity_indexes, **settings)

    # Notes & Chromaticities / CRI Row and Columns
    gridspec_notes_chromaticities_CRI = gridspec_report[3].subgridspec(1, 2)
    axes_notes = figure.add_subplot(gridspec_notes_chromaticities_CRI[0])
    axes_notes.set_axis_off()
    axes_notes.text(
        0.25,
        1,
        'Notes: ',
        ha='right',
        va='center',
        size='medium',
        weight='bold')
    axes_notes.text(0.25, 1, notes, va='center', size='medium')
    gridspec_chromaticities_CRI = gridspec_notes_chromaticities_CRI[
        1].subgridspec(1, 2)

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
        '$x$ {:.4f}'.format(xy[0]),
        ha='center',
        va='center',
        size='medium',
        weight='bold')

    axes_chromaticities.text(
        0.5,
        3 / 5,
        '$y$ {:.4f}'.format(xy[1]),
        ha='center',
        va='center',
        size='medium',
        weight='bold')

    axes_chromaticities.text(
        0.5,
        2 / 5,
        '$u\'$ {:.4f}'.format(uv_p[0]),
        ha='center',
        va='center',
        size='medium',
        weight='bold')

    axes_chromaticities.text(
        0.5,
        1 / 5,
        '$v\'$ {:.4f}'.format(uv_p[1]),
        ha='center',
        va='center',
        size='medium',
        weight='bold')

    gridspec_CRI = gridspec_chromaticities_CRI[1].subgridspec(1, 1)

    CRI_spec = colour_rendering_index(
        specification.sd_test, additional_data=True)

    axes_CRI = figure.add_subplot(gridspec_CRI[0])
    axes_CRI.set_xticks([])
    axes_CRI.set_yticks([])
    axes_CRI.text(
        0.5,
        4 / 5,
        'CIE 13.31-1995',
        ha='center',
        va='center',
        size='medium',
        weight='bold')

    axes_CRI.text(
        0.5,
        3 / 5,
        '(CRI)',
        ha='center',
        va='center',
        size='medium',
        weight='bold')

    axes_CRI.text(
        0.5,
        2 / 5,
        '$R_a$ {:.0f}'.format(CRI_spec.Q_a),
        ha='center',
        va='center',
        size='medium',
        weight='bold')

    axes_CRI.text(
        0.5,
        1 / 5,
        '$R_9$ {:.0f}'.format(CRI_spec.Q_as[8].Q_a),
        ha='center',
        va='center',
        size='medium',
        weight='bold')

    gridspec_footer = gridspec_report[4].subgridspec(1, 1)
    axes_footer = figure.add_subplot(gridspec_footer[0])
    _plot_report_footer(axes_footer)

    figure.set_constrained_layout_pads(**report_box_padding)

    settings = kwargs.copy()
    settings['tight_layout'] = False

    return render(**settings)


@override_style(**CONSTANTS_REPORT_STYLE)
def plot_single_sd_colour_rendition_report_intermediate(
        sd,
        report_size=CONSTANT_REPORT_SIZE_INTERMEDIATE,
        report_row_height_ratios=(
            CONSTANT_REPORT_ROW_HEIGHT_RATIOS_INTERMEDIATE),
        report_box_padding=None,
        **kwargs):
    """
    Generates the intermediate *ANSI/IES TM-30-18 Colour Rendition Report* for
    given spectral distribution.

    Parameters
    ----------
    sd : SpectralDistribution or SpectralDistribution_IESTM2714
        Spectral distribution of the emission source to generate the report
        for.
    report_size : array_like, optional
        Report size, default to A4 paper size in inches.
    report_row_height_ratios : array_like, optional
        Report size row height ratios.
    report_box_padding : array_like, optional
        Report box padding, tries to define the padding around the figure and
        in-between the axes.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`, :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
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

    if six.PY2:
        runtime_warning(
            'The "ANSI/IES TM-30-18 Colour Rendition Report" uses advanced '
            '"Matplotlib" layout capabilities only available for Python 3!')

        return render()

    if report_box_padding is None:
        report_box_padding = CONSTANT_REPORT_PADDING_INTERMEDIATE

    specification = colour_fidelity_index_ANSIIESTM3018(sd, True)

    figure = plt.figure(figsize=report_size, constrained_layout=True)

    settings = kwargs.copy()
    settings['standalone'] = False
    settings['tight_layout'] = False

    gridspec_report = figure.add_gridspec(
        3, 1, height_ratios=report_row_height_ratios)

    # Title Row
    gridspec_title = gridspec_report[0].subgridspec(1, 1)
    axes_title = figure.add_subplot(gridspec_title[0])
    _plot_report_header(axes_title)

    # Main Figures Rows & Columns
    gridspec_figures = gridspec_report[1].subgridspec(2, 2)

    axes_vector_graphics = figure.add_subplot(gridspec_figures[0:2, 0])
    plot_colour_vector_graphic(
        specification, axes=axes_vector_graphics, **settings)

    axes_chroma_shifts = figure.add_subplot(gridspec_figures[0, 1])
    plot_local_chroma_shifts(
        specification, axes=axes_chroma_shifts, **settings)

    axes_hue_shifts = figure.add_subplot(gridspec_figures[1, 1])
    plot_local_hue_shifts(
        specification, axes=axes_hue_shifts, x_ticker=True, **settings)

    gridspec_footer = gridspec_report[2].subgridspec(1, 1)
    axes_footer = figure.add_subplot(gridspec_footer[0])
    _plot_report_footer(axes_footer)

    figure.set_constrained_layout_pads(**report_box_padding)

    settings = kwargs.copy()
    settings['tight_layout'] = False

    return render(**settings)


def plot_single_sd_colour_rendition_report_simple(
        sd,
        report_size=CONSTANT_REPORT_SIZE_SIMPLE,
        report_row_height_ratios=CONSTANT_REPORT_ROW_HEIGHT_RATIOS_SIMPLE,
        report_box_padding=None,
        **kwargs):
    """
    Generates the simple *ANSI/IES TM-30-18 Colour Rendition Report* for given
    spectral distribution.

    Parameters
    ----------
    sd : SpectralDistribution or SpectralDistribution_IESTM2714
        Spectral distribution of the emission source to generate the report
        for.
    report_size : array_like, optional
        Report size, default to A4 paper size in inches.
    report_row_height_ratios : array_like, optional
        Report size row height ratios.
    report_box_padding : array_like, optional
        Report box padding, tries to define the padding around the figure and
        in-between the axes.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`, :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
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

    if six.PY2:
        runtime_warning(
            'The "ANSI/IES TM-30-18 Colour Rendition Report" uses advanced '
            '"Matplotlib" layout capabilities only available for Python 3!')

        return render()

    if report_box_padding is None:
        report_box_padding = CONSTANT_REPORT_PADDING_SIMPLE

    specification = colour_fidelity_index_ANSIIESTM3018(sd, True)

    figure = plt.figure(figsize=report_size, constrained_layout=True)

    settings = kwargs.copy()
    settings['standalone'] = False
    settings['tight_layout'] = False

    gridspec_report = figure.add_gridspec(
        3, 1, height_ratios=report_row_height_ratios)

    # Title Row
    gridspec_title = gridspec_report[0].subgridspec(1, 1)
    axes_title = figure.add_subplot(gridspec_title[0])
    _plot_report_header(axes_title)

    # Main Figures Rows & Columns
    gridspec_figures = gridspec_report[1].subgridspec(1, 1)

    axes_vector_graphics = figure.add_subplot(gridspec_figures[0, 0])
    plot_colour_vector_graphic(
        specification, axes=axes_vector_graphics, **settings)

    gridspec_footer = gridspec_report[2].subgridspec(1, 1)
    axes_footer = figure.add_subplot(gridspec_footer[0])
    _plot_report_footer(axes_footer)

    figure.set_constrained_layout_pads(**report_box_padding)

    settings = kwargs.copy()
    settings['tight_layout'] = False

    return render(**settings)


def plot_single_sd_colour_rendition_report(sd, method='Full', **kwargs):
    """
    Generates the *ANSI/IES TM-30-18 Colour Rendition Report* for given
    spectral distribution according to given method.

    Parameters
    ----------
    sd : SpectralDistribution or SpectralDistribution_IESTM2714
        Spectral distribution of the emission source to generate the report
        for.
    method : unicode, optional
        **{'Full', 'Intermediate', 'Simple'}**,
        Report plotting method.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`, :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.
    source : unicode, optional
        {:func:`colour.plotting.tm3018.\
plot_single_sd_colour_rendition_report_full`},
        Emission source name, defaults to
        `colour.SpectralDistribution_IESTM2714.header.description` or
        `colour.SpectralDistribution_IESTM2714.name` properties value.
    date : unicode, optional
        {:func:`colour.plotting.tm3018.\
plot_single_sd_colour_rendition_report_full`},
        Emission source measurement date, defaults to
        `colour.SpectralDistribution_IESTM2714.header.report_date` property
        value.
    manufacturer : unicode, optional
        {:func:`colour.plotting.tm3018.\
plot_single_sd_colour_rendition_report_full`},
        Emission source manufacturer, defaults to
        `colour.SpectralDistribution_IESTM2714.header.manufacturer` property
        value.
    model : unicode, optional
        {:func:`colour.plotting.tm3018.\
plot_single_sd_colour_rendition_report_full`},
        Emission source model, defaults to
        `colour.SpectralDistribution_IESTM2714.header.catalog_number` property
        value.
    notes : unicode, optional
        {:func:`colour.plotting.tm3018.\
plot_single_sd_colour_rendition_report_full`},
        Notes pertaining to the emission source, defaults to
        `colour.SpectralDistribution_IESTM2714.header.comments` property
        value.
    report_size : array_like, optional
        {:func:`colour.plotting.tm3018.\
plot_single_sd_colour_rendition_report_full`, :func:`colour.plotting.tm3018.\
plot_single_sd_colour_rendition_report_intermediate`, \
:func:`colour.plotting.tm3018.plot_single_sd_colour_rendition_report_simple},
        Report size, default to A4 paper size in inches.
    report_row_height_ratios : array_like, optional
        {:func:`colour.plotting.tm3018.\
plot_single_sd_colour_rendition_report_full`, :func:`colour.plotting.tm3018.\
plot_single_sd_colour_rendition_report_intermediate`, \
:func:`colour.plotting.tm3018.plot_single_sd_colour_rendition_report_simple},
        Report size row height ratios.
    report_box_padding : array_like, optional
        {:func:`colour.plotting.tm3018.\
plot_single_sd_colour_rendition_report_full`, :func:`colour.plotting.tm3018.\
plot_single_sd_colour_rendition_report_intermediate`, \
:func:`colour.plotting.tm3018.plot_single_sd_colour_rendition_report_simple},
        Report box padding, tries to define the padding around the figure and
        in-between the axes.

    Returns
    -------
    tuple
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

    method = method.lower()
    if method == 'full':
        return plot_single_sd_colour_rendition_report_full(sd, **kwargs)
    elif method == 'intermediate':
        return plot_single_sd_colour_rendition_report_intermediate(
            sd, **kwargs)
    elif method == 'simple':
        return plot_single_sd_colour_rendition_report_simple(sd, **kwargs)
    else:
        raise ValueError(
            'Invalid method: "{0}", must be one of '
            '[\'Full\', \'Intermediate\', \'simple\']'.format(method))
