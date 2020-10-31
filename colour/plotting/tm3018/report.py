from __future__ import division, unicode_literals

import matplotlib.pyplot as plt

from colour.colorimetry import SDS_ILLUMINANTS, sd_to_XYZ
from colour.models import XYZ_to_xy, XYZ_to_Luv, Luv_to_uv
from colour.quality import (colour_fidelity_index_ANSIIESTM3018,
                            colour_rendering_index)
from colour.plotting import CONSTANTS_COLOUR_STYLE, override_style, render
from colour.utilities import describe_environment

from figures import (plot_spectra_ANSIIESTM3018, plot_color_vector_graphic,
                     plot_local_chroma_shifts, plot_local_hue_shifts,
                     plot_local_color_fidelities, plot_colour_fidelity_indexes)

DEFAULT_REPORT_DPI = 300

# A4 Paper Size in Inches
DEFAULT_FULL_REPORT_SIZE = (8.27, 11.69)
DEFAULT_FULL_REPORT_HEIGHT_RATIOS = (1, 2, 24, 3, 1)
DEFAULT_FULL_REPORT_PADDING = {
    'w_pad': 60 / DEFAULT_REPORT_DPI,
    'h_pad': 30 / DEFAULT_REPORT_DPI,
    'hspace': 0,
    'wspace': 0,
}
DEFAULT_INTERMEDIATE_REPORT_SIZE = (8.27, 11.69 / 3)
DEFAULT_INTERMEDIATE_REPORT_HEIGHT_RATIOS = (1, 8, 1)
DEFAULT_INTERMEDIATE_REPORT_PADDING = {
    'w_pad': 60 / DEFAULT_REPORT_DPI,
    'h_pad': 30 / DEFAULT_REPORT_DPI,
    'hspace': 0,
    'wspace': 0,
}
DEFAULT_SIMPLE_REPORT_SIZE = (8.27, 8.27)
DEFAULT_SIMPLE_REPORT_HEIGHT_RATIOS = (1, 8, 1)
DEFAULT_SIMPLE_REPORT_PADDING = {
    'w_pad': 30 / DEFAULT_REPORT_DPI,
    'h_pad': 30 / DEFAULT_REPORT_DPI,
    'hspace': 0,
    'wspace': 0,
}

DEFAULT_REPORT_STYLE_OVERRIDES = {
    'axes.grid': False,
    'axes.labelpad': CONSTANTS_COLOUR_STYLE.geometry.short,
    'axes.labelsize': 'x-small',
    'axes.labelweight': 'bold',
    'legend.frameon': False,
    'xtick.labelsize': 'x-small',
    'ytick.labelsize': 'x-small',
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': CONSTANTS_COLOUR_STYLE.geometry.long * 0.5,
    'xtick.minor.size': CONSTANTS_COLOUR_STYLE.geometry.long * 0.25,
    'ytick.major.size': CONSTANTS_COLOUR_STYLE.geometry.long * 0.5,
    'ytick.minor.size': CONSTANTS_COLOUR_STYLE.geometry.long * 0.25,
}

REPORT_HEADER_CONTENT = 'IES TM-30-18 Colour Rendition Report'
REPORT_FOOTER_CONTENT = ('Colours are for visual orientation purposes only. '
                         'Created with Colour {0}').format(
                             describe_environment(print_callable=lambda x: x)[
                                 'colour-science.org']['colour'])
DEFAULT_NOT_APPLICABLE_VALUE = 'N/A'


def _plot_report_header(axes):
    axes.set_axis_off()
    axes.text(
        0.5,
        0.5,
        REPORT_HEADER_CONTENT,
        ha='center',
        va='center',
        size='x-large',
        weight='bold')


def _plot_report_footer(axes):
    axes.set_axis_off()
    axes.text(0.5, 0.5, REPORT_FOOTER_CONTENT, ha='center', va='center')


@override_style(**DEFAULT_REPORT_STYLE_OVERRIDES)
def plot_colour_rendition_report_full_ANSIIESTM3018(
        specification,
        source=DEFAULT_NOT_APPLICABLE_VALUE,
        date=DEFAULT_NOT_APPLICABLE_VALUE,
        manufacturer=DEFAULT_NOT_APPLICABLE_VALUE,
        model=DEFAULT_NOT_APPLICABLE_VALUE,
        notes=DEFAULT_NOT_APPLICABLE_VALUE,
        report_size=DEFAULT_FULL_REPORT_SIZE,
        report_height_ratios=DEFAULT_FULL_REPORT_HEIGHT_RATIOS,
        report_box_padding=DEFAULT_FULL_REPORT_PADDING,
        **kwargs):
    figure = plt.figure(figsize=report_size, constrained_layout=True)

    settings = {'tight_layout': False}
    settings.update(**kwargs)
    settings['standalone'] = False

    gridspec_report = figure.add_gridspec(
        5, 1, height_ratios=report_height_ratios)

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
    plot_color_vector_graphic(
        specification, axes=axes_vector_graphics, **settings)

    axes_chroma_shifts = figure.add_subplot(gridspec_figures[0, 1])
    plot_local_chroma_shifts(
        specification, axes=axes_chroma_shifts, **settings)

    axes_hue_shifts = figure.add_subplot(gridspec_figures[1, 1])
    plot_local_hue_shifts(specification, axes=axes_hue_shifts, **settings)

    axes_colour_fidelities = figure.add_subplot(gridspec_figures[2, 1])
    plot_local_color_fidelities(
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

    settings = {}
    settings.update(kwargs)
    settings['tight_layout'] = False

    return render(**settings)


@override_style(**DEFAULT_REPORT_STYLE_OVERRIDES)
def plot_colour_rendition_report_intermediate_ANSIIESTM3018(
        specification,
        report_size=DEFAULT_INTERMEDIATE_REPORT_SIZE,
        report_height_ratios=DEFAULT_INTERMEDIATE_REPORT_HEIGHT_RATIOS,
        report_box_padding=DEFAULT_INTERMEDIATE_REPORT_PADDING,
        **kwargs):
    figure = plt.figure(figsize=report_size, constrained_layout=True)

    settings = {'tight_layout': False}
    settings.update(**kwargs)
    settings['standalone'] = False

    gridspec_report = figure.add_gridspec(
        3, 1, height_ratios=report_height_ratios)

    # Title Row
    gridspec_title = gridspec_report[0].subgridspec(1, 1)
    axes_title = figure.add_subplot(gridspec_title[0])
    _plot_report_header(axes_title)

    # Main Figures Rows & Columns
    gridspec_figures = gridspec_report[1].subgridspec(2, 2)

    axes_vector_graphics = figure.add_subplot(gridspec_figures[0:2, 0])
    plot_color_vector_graphic(
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

    settings = {}
    settings.update(kwargs)
    settings['tight_layout'] = False

    return render(**settings)


def plot_colour_rendition_report_simple_ANSIIESTM3018(
        specification,
        report_size=DEFAULT_SIMPLE_REPORT_SIZE,
        report_height_ratios=DEFAULT_SIMPLE_REPORT_HEIGHT_RATIOS,
        report_box_padding=DEFAULT_SIMPLE_REPORT_PADDING,
        **kwargs):
    figure = plt.figure(figsize=report_size, constrained_layout=True)

    settings = {'tight_layout': False}
    settings.update(**kwargs)
    settings['standalone'] = False

    gridspec_report = figure.add_gridspec(
        3, 1, height_ratios=report_height_ratios)

    # Title Row
    gridspec_title = gridspec_report[0].subgridspec(1, 1)
    axes_title = figure.add_subplot(gridspec_title[0])
    _plot_report_header(axes_title)

    # Main Figures Rows & Columns
    gridspec_figures = gridspec_report[1].subgridspec(1, 1)

    axes_vector_graphics = figure.add_subplot(gridspec_figures[0, 0])
    plot_color_vector_graphic(
        specification, axes=axes_vector_graphics, **settings)

    gridspec_footer = gridspec_report[2].subgridspec(1, 1)
    axes_footer = figure.add_subplot(gridspec_footer[0])
    _plot_report_footer(axes_footer)

    figure.set_constrained_layout_pads(**report_box_padding)

    settings = {}
    settings.update(kwargs)
    settings['tight_layout'] = False

    return render(**settings)


def plot_colour_rendition_report_ANSIIESTM3018(specification,
                                               report_type='Full',
                                               **kwargs):
    report_type = report_type.lower()
    if report_type == 'full':
        return plot_colour_rendition_report_full_ANSIIESTM3018(
            specification, **kwargs)
    elif report_type == 'intermediate':
        return plot_colour_rendition_report_intermediate_ANSIIESTM3018(
            specification, **kwargs)
    elif report_type == 'simple':
        return plot_colour_rendition_report_simple_ANSIIESTM3018(
            specification, **kwargs)
    else:
        raise ValueError('size must be one of \'simple\', \'intermediate\' or '
                         '\'full\'')


if __name__ == '__main__':
    from matplotlib._layoutbox import plot_children

    # colour.plotting.colour_style()

    lamp = SDS_ILLUMINANTS['FL2']

    specification = colour_fidelity_index_ANSIIESTM3018(lamp, True)
    figure, axes = plot_colour_rendition_report_ANSIIESTM3018(
        specification,
        'full',
        source='CIE FL2',
        date='2020/10/31',
        # filename='/Users/kelsolaar/Downloads/full-report.png',
        transparent_background=False,
        standalone=False)

    # plot_children(figure, figure._layoutbox, printit=False)
    plt.savefig('/Users/kelsolaar/Downloads/full-report.png', dpi=300)

    figure, axes = plot_colour_rendition_report_ANSIIESTM3018(
        specification,
        'intermediate',
        source='CIE FL2',
        date='2020/10/31',
        # filename='/Users/kelsolaar/Downloads/intermediate-report.png',
        transparent_background=False,
        standalone=False)

    # plot_children(figure, figure._layoutbox, printit=False)
    plt.savefig('/Users/kelsolaar/Downloads/intermediate-report.png', dpi=300)

    figure, axes = plot_colour_rendition_report_ANSIIESTM3018(
        specification,
        'simple',
        source='CIE FL2',
        date='2020/10/31',
        # filename='/Users/kelsolaar/Downloads/simple-report.png',
        transparent_background=False,
        standalone=False)

    # plot_children(figure, figure._layoutbox, printit=False)
    plt.savefig('/Users/kelsolaar/Downloads/simple-report.png', dpi=300)
