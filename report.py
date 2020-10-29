import matplotlib.pyplot as plt

import colour
from colour.colorimetry import SDS_ILLUMINANTS, sd_to_XYZ
from colour.models import XYZ_to_xy, XYZ_to_Luv, Luv_to_uv
from colour.quality import (colour_fidelity_index_ANSIIESTM3018,
                            colour_rendering_index)

from _elements import (plot_spectra_TM_30_18, plot_color_vector_graphic,
                       plot_local_chroma_shifts, plot_local_hue_shifts,
                       plot_local_color_fidelities,
                       plot_colour_fidelity_indexes)


def _full_report(spec, source, date, manufacturer, model, notes=None):
    figure = plt.figure(figsize=(8.27, 11.69))

    figure.text(0.5, 0.97, 'TM-30-18 Color Rendition Report', ha='center',
                size='x-large')

    figure.text(0.250, 0.935, 'Source: ', ha='right', size='large',
                weight='bold')
    figure.text(0.250, 0.935, source, size='large')
    figure.text(0.250, 0.907, 'Date: ', ha='right', size='large',
                weight='bold')
    figure.text(0.250, 0.907, date, size='large')

    figure.text(0.700, 0.935, 'Manufacturer: ', ha='right', size='large',
                weight='bold')
    figure.text(0.700, 0.935, manufacturer, ha='left', size='large')
    figure.text(0.700, 0.907, 'Model: ', ha='right', size='large',
                weight='bold')
    figure.text(0.700, 0.907, model, size='large')

    ax = figure.add_axes((0.057, 0.767, 0.385, 0.112))
    plot_spectra_TM_30_18(ax, spec)

    ax = figure.add_axes((0.036, 0.385, 0.428, 0.333))
    plot_color_vector_graphic(ax, spec)

    ax = figure.add_axes((0.554, 0.736, 0.409, 0.141))
    plot_local_chroma_shifts(ax, spec)

    ax = figure.add_axes((0.554, 0.576, 0.409, 0.141))
    plot_local_hue_shifts(ax, spec)

    ax = figure.add_axes((0.554, 0.401, 0.409, 0.141))
    plot_local_color_fidelities(ax, spec)

    ax = figure.add_axes((0.094, 0.195, 0.870, 0.161))
    plot_colour_fidelity_indexes(ax, spec)

    if notes:
        figure.text(0.139, 0.114, 'Notes: ', ha='right', size='large',
                    weight='bold')
        figure.text(0.139, 0.114, notes, size='large')

    XYZ = sd_to_XYZ(spec.sd_test)
    xy = XYZ_to_xy(XYZ)
    Luv = XYZ_to_Luv(XYZ, xy)
    up, vp = Luv_to_uv(Luv, xy)

    figure.text(0.712, 0.111, '$x$  {:.4f}'.format(xy[0]), ha='center')
    figure.text(0.712, 0.091, '$y$  {:.4f}'.format(xy[1]), ha='center')
    figure.text(0.712, 0.071, '$u\'$  {:.4f}'.format(up), ha='center')
    figure.text(0.712, 0.051, '$v\'$  {:.4f}'.format(vp), ha='center')

    rect = plt.Rectangle((0.814, 0.035), 0.144, 0.096, color='black',
                         fill=False)
    figure.add_artist(rect)

    CRI_spec = colour_rendering_index(spec.sd_test, additional_data=True)

    figure.text(0.886, 0.111, 'CIE 13.31-1995', ha='center')
    figure.text(0.886, 0.091, '(CRI)', ha='center')
    figure.text(0.886, 0.071, '$R_a$  {:.0f}'.format(CRI_spec.Q_a),
                ha='center', weight='bold')
    figure.text(0.886, 0.051, '$R_9$  {:.0f}'.format(CRI_spec.Q_as[8].Q_a),
                ha='center', weight='bold')

    figure.text(0.500, 0.010, 'Created with Colour ' + colour.__version__,
                ha='center')


def _intermediate_report(spec, source, date, manufacturer, model, notes=None):
    figure = plt.figure(figsize=(8.27, 4.44))

    figure.text(0.500, 0.945, 'TM-30-18 Color Rendition Report', ha='center',
                size='x-large')

    ax = figure.add_axes((0.024, 0.077, 0.443, 0.833))
    plot_color_vector_graphic(ax, spec)

    ax = figure.add_axes((0.560, 0.550, 0.409, 0.342))
    plot_local_chroma_shifts(ax, spec)

    ax = figure.add_axes((0.560, 0.150, 0.409, 0.342))
    plot_local_hue_shifts(ax, spec)

    figure.text(0.500, 0.020, 'Created with Colour ' + colour.__version__,
                ha='center')


def _simple_report(spec, source, date, manufacturer, model, notes=None):
    figure = plt.figure(figsize=(4.22, 4.44))

    figure.text(0.500, 0.945, 'TM-30-18 Color Rendition Report', ha='center',
                size='x-large')

    ax = figure.add_axes((0.05, 0.05, 0.90, 0.90))
    plot_color_vector_graphic(ax, spec)

    figure.text(0.500, 0.022, 'Created with Colour ' + colour.__version__,
                ha='center')


def plot_color_rendition_report(spec, size='full', **kwargs):
    if size == 'full':
        _full_report(spec, **kwargs)
    elif size == 'intermediate':
        _intermediate_report(spec, **kwargs)
    elif size == 'simple':
        _simple_report(spec, **kwargs)
    else:
        raise ValueError('size must be one of \'simple\', \'intermediate\' or '
                         '\'full\'')


if __name__ == '__main__':
    lamp = SDS_ILLUMINANTS['FL2']

    spec = colour_fidelity_index_ANSIIESTM3018(lamp, True)
    plot_color_rendition_report(spec,
                                'intermediate',
                                source='CIE FL2',
                                date='Aug 23 2020',
                                manufacturer='N/A',
                                model='N/A')
    plt.savefig('/tmp/test.png', dpi=200)
