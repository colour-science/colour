import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from colour.colorimetry import sd_to_XYZ
from colour.utilities import as_float_array

_BIN_BAR_COLOURS = ['#a35c60', '#cc765e', '#cc8145', '#d8ac62', '#ac9959',
                    '#919e5d', '#668b5e', '#61b290', '#7bbaa6', '#297a7e',
                    '#55788d', '#708ab2', '#988caa', '#735877', '#8f6682',
                    '#ba7a8e']

_BIN_ARROW_COLOURS = ['#e62828', '#e74b4b', '#fb812e', '#ffb529', '#cbca46',
                      '#7eb94c', '#41c06d', '#009c7c', '#16bcb0', '#00a4bf',
                      '#0085c3', '#3b62aa', '#4568ae', '#6a4e85', '#9d69a1',
                      '#a74f81']

_TCS_BAR_COLOURS = [
    '#f1bdcd', '#ca6183', '#573a40', '#cd8791', '#ad3f55', '#925f62',
    '#933440', '#8c3942', '#413d3e', '#fa8070', '#c35644', '#da604a',
    '#824e39', '#bca89f', '#c29a89', '#8d593c', '#915e3f', '#99745b',
    '#d39257', '#d07f2c', '#feb45f', '#efa248', '#f0dfbd', '#fed586',
    '#d0981e', '#fed06a', '#b5ac81', '#645d37', '#ead163', '#9e9464',
    '#ebd969', '#c4b135', '#e6de9c', '#99912c', '#61603a', '#c2c2af',
    '#6d703b', '#d2d7a1', '#4b5040', '#6b7751', '#d3dcc3', '#88b33a',
    '#8ebf3e', '#3e3f3d', '#65984a', '#83a96e', '#92ae86', '#91cd8e',
    '#477746', '#568c6a', '#659477', '#276e49', '#008d62', '#b6e2d4',
    '#a5d9cd', '#39c4ad', '#00a18a', '#009786', '#b4e1d9', '#cddddc',
    '#99c1c0', '#909fa1', '#494d4e', '#009fa8', '#32636a', '#007788',
    '#007f95', '#66a0b2', '#687d88', '#75b6db', '#1e5574', '#aab9c3',
    '#3091c4', '#3b3e41', '#274d72', '#376fb8', '#496692', '#3b63ac',
    '#a0aed5', '#9293c8', '#61589d', '#d4d3e5', '#aca6ca', '#3e3b45',
    '#5f5770', '#a08cc7', '#664782', '#a77ab5', '#6a4172', '#7d4983',
    '#c4bfc4', '#937391', '#ae91aa', '#764068', '#bf93b1', '#d7a9c5',
    '#9d587f', '#ce6997', '#ae4a79']


def plot_spectra_TM_30_18(ax, spec):
    """
    Plots a comparison of spectral distributions of a test and a reference
    illuminant, for use in *TM-30-18* color rendition reports.

    Parameters
    ==========
    spec : TM_30_18_Specification
        *TM-30-18* color fidelity specification.
    """

    Y_reference = sd_to_XYZ(spec.sd_reference)[1]
    Y_test = sd_to_XYZ(spec.sd_test)[1]

    ax.plot(spec.sd_reference.wavelengths,
            spec.sd_reference.values / Y_reference,
            'k', label='Reference')
    ax.plot(spec.sd_test.wavelengths,
            spec.sd_test.values / Y_test,
            'r', label='Test')

    ax.set_yticks([])
    ax.grid()

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Radiant power')
    ax.legend()


def plot_color_vector_graphic(ax, spec):
    """
    Plots a *Color Vector Graphic* according to *TM-30-18* recommendations.

    Parameters
    ==========
    spec : TM_30_18_Specification
        *TM-30-18* color fidelity specification.
    """

    # Background
    backdrop = mpimg.imread(os.path.join(os.path.dirname(__file__),
                                         'CVG background.jpg'))
    ax.imshow(backdrop, extent=(-1.5, 1.5, -1.5, 1.5))
    ax.axis('off')

    # Lines dividing the hues in 16 equal parts along with bin numbers
    ax.plot(0, 0, '+', color='#a6a6a6')
    for i in range(16):
        angle = 2 * np.pi * i / 16
        dx = np.cos(angle)
        dy = np.sin(angle)
        ax.plot((0.15 * dx, 1.5 * dx), (0.15 * dy, 1.5 * dy), '--',
                color='#a6a6a6', lw=0.75)

        angle = 2 * np.pi * (i + 0.5) / 16
        ax.annotate(str(i + 1), color='#a6a6a6', ha='center', va='center',
                    xy=(1.41 * np.cos(angle), 1.41 * np.sin(angle)),
                    weight='bold', size=9)

    # Circles
    circle = plt.Circle((0, 0), 1, color='black', lw=1.25, fill=False)
    ax.add_artist(circle)
    for radius in [0.8, 0.9, 1.1, 1.2]:
        circle = plt.Circle((0, 0), radius, color='white', lw=0.75, fill=False)
        ax.add_artist(circle)

    # -/+20% marks near the white circles
    props = dict(ha='right', color='white', size=7)
    ax.annotate('-20%', xy=(0, -0.8), va='bottom', **props)
    ax.annotate('+20%', xy=(0, -1.2), va='top', **props)

    # Average CAM02 h correlate for each bin, in radians
    average_hues = as_float_array([np.mean([spec.colorimetry_data[1][i].CAM.h
                                   for i in spec.bins[j]])
                                  for j in range(16)]) / 180 * np.pi
    xy_reference = np.vstack([np.cos(average_hues), np.sin(average_hues)]).T

    # Arrow offsets as defined by the standard
    offsets = ((spec.averages_test - spec.averages_reference)
               / spec.average_norms[:, np.newaxis])
    xy_test = xy_reference + offsets

    # Arrows
    for i in range(16):
        ax.arrow(xy_reference[i, 0], xy_reference[i, 1], offsets[i, 0],
                 offsets[i, 1], length_includes_head=True, width=0.005,
                 head_width=0.04, linewidth=None, color=_BIN_ARROW_COLOURS[i])

    # Red (test) gamut shape
    loop = np.append(xy_test, xy_test[0, np.newaxis], axis=0)
    ax.plot(loop[:, 0], loop[:, 1], '-', color='#f05046', lw=2)

    def corner_text(label, text, ha, va):
        x = -1.44 if ha == 'left' else 1.44
        y = 1.44 if va == 'top' else -1.44
        y_text = -14 if va == 'top' else 14

        ax.annotate(text, xy=(x, y), color='black', ha=ha, va=va,
                    weight='bold', size=14)
        ax.annotate(label, xy=(x, y), color='black', xytext=(0, y_text),
                    textcoords='offset points', ha=ha, va=va, size=11)

    corner_text('$R_f$', '{:.0f}'.format(spec.R_f), 'left', 'top')
    corner_text('$R_g$', '{:.0f}'.format(spec.R_g), 'right', 'top')
    corner_text('CCT', '{:.0f} K'.format(spec.CCT), 'left', 'bottom')
    corner_text('$D_{uv}$', '{:.4f}'.format(spec.D_uv), 'right', 'bottom')


def _plot_bin_bars(ax, values, ticks, labels_format, labels='vertical'):
    """
    A convenience function for plotting coloured bar graphs with labels at each
    bar.
    """

    ax.set_axisbelow(True)  # Draw the grid behind the bars
    ax.grid(axis='y')

    ax.bar(np.arange(16) + 1, values, color=_BIN_BAR_COLOURS)
    ax.set_xlim(0.5, 16.5)
    if ticks:
        ax.set_xticks(np.arange(1, 17))
    else:
        ax.set_xticks([])

    for i, value in enumerate(values):
        if labels == 'vertical':
            va = 'bottom' if value > 0 else 'top'
            ax.annotate(labels_format.format(value), xy=(i + 1, value),
                        rotation=90, ha='center', va=va)

        elif labels == 'horizontal':
            va = 'bottom' if value < 90 else 'top'
            ax.annotate(labels_format.format(value), xy=(i + 1, value),
                        ha='center', va=va)


def plot_local_chroma_shifts(ax, spec, ticks=True):
    _plot_bin_bars(ax, spec.R_cs, ticks, '{:.0f}%')
    ax.set_ylim(-40, 40)
    ax.set_ylabel('Local Chroma Shift ($R_{cs,hj}$)')

    ticks = np.arange(-40, 41, 10)
    ax.set_yticks(ticks)
    ax.set_yticklabels(['{}%'.format(value) for value in ticks])


def plot_local_hue_shifts(ax, spec, ticks=True):
    _plot_bin_bars(ax, spec.R_hs, ticks, '{:.2f}')
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks(np.arange(-0.5, 0.51, 0.1))
    ax.set_ylabel('Local Hue Shift ($R_{hs,hj}$)')


def plot_local_color_fidelities(ax, spec, ticks=True):
    _plot_bin_bars(ax, spec.R_fs, ticks, '{:.0f}', 'horizontal')
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_xlabel('Hue-angle bin')
    ax.set_ylabel('Local Color Fidelity ($R_{f,hj}$)')


def plot_colour_fidelity_indexes(ax, spec):
    ax.set_axisbelow(True)  # Draw the grid behind the bars
    ax.grid(axis='y')

    ax.bar(np.arange(99) + 1, spec.R_s, color=_TCS_BAR_COLOURS)
    ax.set_xlim(0.5, 99.5)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_ylabel('Color Sample Fidelity ($R_{f,ces}$)')

    ticks = list(range(1, 100, 1))
    ax.set_xticks(ticks)

    labels = ['CES{:02d}'.format(i) if i % 3 == 1 else ''
              for i in range(1, 100)]
    ax.set_xticklabels(labels, rotation=90)
