import os
import numpy as np
import matplotlib.pyplot as plt

from colour.colorimetry import sd_to_XYZ
from colour.io import read_image
from colour.utilities import as_float_array
from colour.plotting import (CONSTANTS_COLOUR_STYLE, artist, colour_cycle,
                             override_style, plot_image, render)
RESOURCES_DIRECTORY_ANSIIESTM3018 = os.path.join(
    os.path.dirname(__file__), 'resources')

_BIN_BAR_COLOURS = [
    '#A35C60', '#CC765E', '#CC8145', '#D8AC62', '#AC9959', '#919E5D',
    '#668B5E', '#61B290', '#7BBAA6', '#297A7E', '#55788D', '#708AB2',
    '#988CAA', '#735877', '#8F6682', '#BA7A8E'
]

_BIN_ARROW_COLOURS = [
    '#E62828', '#E74B4B', '#FB812E', '#FFB529', '#CBCA46', '#7EB94C',
    '#41C06D', '#009C7C', '#16BCB0', '#00A4BF', '#0085C3', '#3B62AA',
    '#4568AE', '#6A4E85', '#9D69A1', '#A74F81'
]

_TCS_BAR_COLOURS = [
    '#F1BDCD', '#CA6183', '#573A40', '#CD8791', '#AD3F55', '#925F62',
    '#933440', '#8C3942', '#413D3E', '#FA8070', '#C35644', '#DA604A',
    '#824E39', '#BCA89F', '#C29A89', '#8D593C', '#915E3F', '#99745B',
    '#D39257', '#D07F2C', '#FEB45F', '#EFA248', '#F0DFBD', '#FED586',
    '#D0981E', '#FED06A', '#B5AC81', '#645D37', '#EAD163', '#9E9464',
    '#EBD969', '#C4B135', '#E6DE9C', '#99912C', '#61603A', '#C2C2AF',
    '#6D703B', '#D2D7A1', '#4B5040', '#6B7751', '#D3DCC3', '#88B33A',
    '#8EBF3E', '#3E3F3D', '#65984A', '#83A96E', '#92AE86', '#91CD8E',
    '#477746', '#568C6A', '#659477', '#276E49', '#008D62', '#B6E2D4',
    '#A5D9CD', '#39C4AD', '#00A18A', '#009786', '#B4E1D9', '#CDDDDC',
    '#99C1C0', '#909FA1', '#494D4E', '#009FA8', '#32636A', '#007788',
    '#007F95', '#66A0B2', '#687D88', '#75B6DB', '#1E5574', '#AAB9C3',
    '#3091C4', '#3B3E41', '#274D72', '#376FB8', '#496692', '#3B63AC',
    '#A0AED5', '#9293C8', '#61589D', '#D4D3E5', '#ACA6CA', '#3E3B45',
    '#5F5770', '#A08CC7', '#664782', '#A77AB5', '#6A4172', '#7D4983',
    '#C4BFC4', '#937391', '#AE91AA', '#764068', '#BF93B1', '#D7A9C5',
    '#9D587F', '#CE6997', '#AE4A79'
]


@override_style()
def plot_spectra_ANSIIESTM3018(specification, **kwargs):
    """
    Plots a comparison of spectral distributions of a test and a reference
    illuminant, for use in *TM-30-18* color rendition reports.

    Parameters
    ==========
    specification : TM_30_18_Specification
        *TM-30-18* color fidelity specification.
    """

    settings = {}
    settings.update(kwargs)

    _figure, axes = artist(**settings)

    Y_reference = sd_to_XYZ(specification.sd_reference)[1]
    Y_test = sd_to_XYZ(specification.sd_test)[1]

    axes.plot(
        specification.sd_reference.wavelengths,
        specification.sd_reference.values / Y_reference,
        'black',
        label='Reference')
    axes.plot(
        specification.sd_test.wavelengths,
        specification.sd_test.values / Y_test,
        '#F05046',
        label='Test')
    axes.tick_params(axis='y', which='both', length=0)
    axes.set_yticklabels([])

    settings = {
        'axes': axes,
        'legend': True,
        'legend_columns': 2,
        'x_label': 'Wavelength (nm)',
        'y_label': 'Radiant Power\n(Equal Luminous Flux)',
    }
    settings.update(kwargs)

    return render(**settings)


def plot_color_vector_graphic(specification, **kwargs):
    """
    Plots a *Color Vector Graphic* according to *TM-30-18* recommendations.

    Parameters
    ==========
    specification : TM_30_18_Specification
        *TM-30-18* color fidelity specification.
    """

    # Background
    background_image = read_image(
        os.path.join(RESOURCES_DIRECTORY_ANSIIESTM3018, 'CVG_Background.jpg'))
    _figure, axes = plot_image(
        background_image,
        imshow_kwargs={'extent': [-1.5, 1.5, -1.5, 1.5]},
        **kwargs)

    # Lines dividing the hues in 16 equal parts along with bin numbers.
    axes.plot(0, 0, '+', color='#A6A6A6')
    for i in range(16):
        angle = 2 * np.pi * i / 16
        dx = np.cos(angle)
        dy = np.sin(angle)
        axes.plot(
            (0.15 * dx, 1.5 * dx), (0.15 * dy, 1.5 * dy),
            '--',
            color='#A6A6A6',
            lw=0.75)

        angle = 2 * np.pi * (i + 0.5) / 16
        axes.annotate(
            str(i + 1),
            color='#A6A6A6',
            ha='center',
            va='center',
            xy=(1.41 * np.cos(angle), 1.41 * np.sin(angle)),
            weight='bold',
            size=9)

    # Circles.
    circle = plt.Circle((0, 0), 1, color='black', lw=1.25, fill=False)
    axes.add_artist(circle)
    for radius in [0.8, 0.9, 1.1, 1.2]:
        circle = plt.Circle((0, 0), radius, color='white', lw=0.75, fill=False)
        axes.add_artist(circle)

    # -/+20% marks near the white circles.
    props = dict(ha='right', color='white', size=7)
    axes.annotate('-20%', xy=(0, -0.8), va='bottom', **props)
    axes.annotate('+20%', xy=(0, -1.2), va='top', **props)

    # Average "CAM02" h correlate for each bin, in radians.
    average_hues = as_float_array([
        np.mean([
            specification.colorimetry_data[1][i].CAM.h
            for i in specification.bins[j]
        ]) for j in range(16)
    ]) / 180 * np.pi
    xy_reference = np.vstack([np.cos(average_hues), np.sin(average_hues)]).T

    # Arrow offsets as defined by the standard.
    offsets = ((specification.averages_test - specification.averages_reference)
               / specification.average_norms[:, np.newaxis])
    xy_test = xy_reference + offsets

    # Arrows.
    for i in range(16):
        axes.arrow(
            xy_reference[i, 0],
            xy_reference[i, 1],
            offsets[i, 0],
            offsets[i, 1],
            length_includes_head=True,
            width=0.005,
            head_width=0.04,
            linewidth=None,
            color=_BIN_ARROW_COLOURS[i])

    # Red (test) gamut shape.
    loop = np.append(xy_test, xy_test[0, np.newaxis], axis=0)
    axes.plot(loop[:, 0], loop[:, 1], '-', color='#F05046', lw=2)

    def corner_text(label, text, ha, va):
        x = -1.44 if ha == 'left' else 1.44
        y = 1.44 if va == 'top' else -1.44
        y_text = -14 if va == 'top' else 14

        axes.annotate(
            text,
            xy=(x, y),
            color='black',
            ha=ha,
            va=va,
            weight='bold',
            size=14)
        axes.annotate(
            label,
            xy=(x, y),
            color='black',
            xytext=(0, y_text),
            textcoords='offset points',
            ha=ha,
            va=va,
            size=11)

    corner_text('$R_f$', '{:.0f}'.format(specification.R_f), 'left', 'top')
    corner_text('$R_g$', '{:.0f}'.format(specification.R_g), 'right', 'top')
    corner_text('CCT', '{:.0f} K'.format(specification.CCT), 'left', 'bottom')
    corner_text('$D_{uv}$', '{:.4f}'.format(specification.D_uv), 'right',
                'bottom')

    return render(**kwargs)


def plot_16_bars(values,
                 label_template,
                 x_ticker=False,
                 labels='vertical',
                 **kwargs):
    """
    A convenience function for plotting coloured bar graphs with labels at each
    bar.
    """

    _figure, axes = artist(**kwargs)

    axes.bar(
        np.arange(16) + 1,
        values,
        color=_BIN_BAR_COLOURS,
        width=1,
        edgecolor='black',
        linewidth=CONSTANTS_COLOUR_STYLE.geometry.short / 3)
    axes.set_xlim(0.5, 16.5)
    if x_ticker:
        axes.set_xticks(np.arange(1, 17))
        axes.set_xlabel('Hue-Angle Bin (j)')
    else:
        axes.set_xticks([])

    value_max = max(values)
    for i, value in enumerate(values):
        if labels == 'vertical':
            va, vo = (('bottom', value_max * 0.1)
                      if value > 0 else ('top', -value_max * 0.1))
            axes.annotate(
                label_template.format(value),
                xy=(i + 1, value + vo),
                rotation=90,
                fontsize='x-small',
                ha='center',
                va=va)
        elif labels == 'horizontal':
            va, vo = (('bottom', value_max * 0.1)
                      if value < 90 else ('top', -value_max * 0.1))
            axes.annotate(
                label_template.format(value + vo),
                xy=(i + 1, value),
                fontsize='x-small',
                ha='center',
                va=va)

    return render(**kwargs)


def plot_local_chroma_shifts(specification, x_ticker=False, **kwargs):
    _figure, axes = plot_16_bars(specification.R_cs, '{0:.0f}%', x_ticker,
                                 **kwargs)

    axes.set_ylim(-40, 40)
    axes.set_ylabel('Local Chroma Shift ($R_{cs,hj}$)')

    ticks = np.arange(-40, 41, 10)
    axes.set_yticks(ticks)
    axes.set_yticklabels(['{0}%'.format(value) for value in ticks])

    return render(**kwargs)


def plot_local_hue_shifts(specification, x_ticker=False, **kwargs):
    _figure, axes = plot_16_bars(specification.R_hs, '{0:.2f}', x_ticker,
                                 **kwargs)
    axes.set_ylim(-0.5, 0.5)
    axes.set_yticks(np.arange(-0.5, 0.51, 0.1))
    axes.set_ylabel('Local Hue Shift ($R_{hs,hj}$)')

    return render(**kwargs)


def plot_local_color_fidelities(specification, x_ticker=False, **kwargs):
    _figure, axes = plot_16_bars(specification.R_fs, '{:.0f}', x_ticker,
                                 'horizontal', **kwargs)
    axes.set_ylim(0, 100)
    axes.set_yticks(np.arange(0, 101, 10))
    axes.set_ylabel('Local Color Fidelity ($R_{f,hj}$)')

    return render(**kwargs)


def plot_colour_fidelity_indexes(specification, **kwargs):
    _figure, axes = artist(**kwargs)

    axes.bar(
        np.arange(99) + 1,
        specification.R_s,
        color=_TCS_BAR_COLOURS,
        width=1,
        edgecolor='black',
        linewidth=CONSTANTS_COLOUR_STYLE.geometry.short / 3)
    axes.set_xlim(0.5, 99.5)
    axes.set_ylim(0, 100)
    axes.set_yticks(np.arange(0, 101, 10))
    axes.set_ylabel('Color Sample Fidelity ($R_{f,CESi}$)')

    ticks = list(range(1, 100, 1))
    axes.set_xticks(ticks)

    labels = [
        'CES{0:02d}'.format(i) if i % 3 == 1 else '' for i in range(1, 100)
    ]
    axes.set_xticklabels(labels, rotation=90)

    return render(**kwargs)
