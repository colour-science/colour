# -*- coding: utf-8 -*-
"""
Colour Gamut Plotting
=====================

Defines the colour gamut plotting objects:

-   :func:`colour.plotting.\
plot_multi_segment_maxima_gamut_boundaries_in_hue_segments`
-   :func:`colour.plotting.plot_multi_segment_maxima_gamut_boundaries`
-   :func:`colour.plotting.plot_Jab_samples_in_segment_maxima_gamut_boundary`
"""

from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from colour.algebra import polar_to_cartesian
from colour.constants import DEFAULT_INT_DTYPE
from colour.gamut import (gamut_boundary_descriptor, polar_to_ij,
                          sample_volume_boundary_descriptor, spherical_to_Jab)
from colour.models import Jab_to_JCh
from colour.plotting import (CONSTANTS_COLOUR_STYLE, artist, override_style,
                             render)
from colour.utilities import as_float_array, tsplit, tstack, runtime_warning

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'plot_segments',
    'plot_multi_segment_maxima_gamut_boundaries_in_hue_segments',
    'plot_multi_segment_maxima_gamut_boundaries',
    'plot_Jab_samples_in_segment_maxima_gamut_boundary'
]


@override_style()
def plot_segments(segments, **kwargs):

    settings = {'uniform': True}
    settings.update(kwargs)

    _figure, axes = artist(**settings)

    rho = np.ones(segments) * 2 ** 16
    phi = np.radians(np.linspace(-180, 180, segments))

    lines = LineCollection(
        [((0, 0), a) for a in polar_to_cartesian(tstack([rho, phi]))],
        colors=CONSTANTS_COLOUR_STYLE.colour.dark,
        alpha=CONSTANTS_COLOUR_STYLE.opacity.low,
    )
    axes.add_collection(lines, autolim=False)

    settings = {'axes': axes}
    settings.update(kwargs)

    return render(**kwargs)


@override_style()
def plot_multi_segment_maxima_gamut_boundaries_in_hue_segments(
        GBD_n, angles=None, columns=None, show_segments=False, **kwargs):
    GBD_n = [as_float_array(GBD) for GBD in GBD_n]

    assert len(np.unique([GBD.shape for GBD in GBD_n])) <= 3, (
        'Gamut boundary descriptor matrices have incompatible shapes!')

    settings = {}
    settings.update(kwargs)

    if angles is not None:
        x_s = np.linspace(0, 180, GBD_n[0].shape[0])
        y_s = angles
        x_s_g, y_s_g = np.meshgrid(x_s, y_s, indexing='ij')
        theta_alpha = tstack([x_s_g, y_s_g])

        GBD_n = [
            sample_volume_boundary_descriptor(GBD, theta_alpha)
            for GBD in GBD_n
        ]

    GBD_n = [Jab_to_JCh(spherical_to_Jab(GBD_m_c)) for GBD_m_c in GBD_n]

    shape_c = GBD_n[0].shape[1]

    columns = (shape_c if columns is None else max(len(angles or []), columns))

    _figure, axes_n = plt.subplots(
        DEFAULT_INT_DTYPE(np.ceil(shape_c / columns)),
        columns,
        sharex='all',
        sharey='all',
        gridspec_kw={
            'hspace': 0,
            'wspace': 0
        },
        constrained_layout=True,
    )

    axes_n = np.ravel(axes_n)

    for i in range(shape_c):

        if show_segments:
            settings = {'axes': axes_n[i]}
            settings.update(kwargs)
            settings['standalone'] = False
            settings['tight_layout'] = False

            plot_segments(shape_c * 2 + 1, **settings)

        label = '{0:d} $^\\degree$'.format(
            DEFAULT_INT_DTYPE((
                i / shape_c * 360) if angles is None else angles[i]))

        axes_n[i].text(
            0.5,
            0.5,
            label,
            alpha=CONSTANTS_COLOUR_STYLE.opacity.low,
            fontsize='xx-large',
            horizontalalignment='center',
            verticalalignment='center',
            transform=axes_n[i].transAxes)

        if i % columns == 0:
            axes_n[i].set_ylabel('J')

        if i > shape_c - columns:
            axes_n[i].set_xlabel('C')

        for j in range(len(GBD_n)):
            axes_n[i].plot(
                GBD_n[j][..., i, 1],
                GBD_n[j][..., i, 0],
                'o-',
                label='GBD {0}'.format(j))

        if i == shape_c - 1:
            axes_n[i].legend()

    for axes in axes_n[shape_c:]:
        axes.set_visible(False)

    settings = {
        'figure_title':
            'Gamut Boundary Descriptors - {0} Hue Segments'.format(shape_c),
        'tight_layout':
            False,
    }
    settings.update(kwargs)

    return render(**settings)


@override_style()
def plot_multi_segment_maxima_gamut_boundaries(
        GBD_n,
        plane='Ja',
        show_segments=True,
        gamut_boundary_descriptor_kwargs=None,
        **kwargs):
    GBD_n = [as_float_array(GBD) for GBD in GBD_n]

    assert len(np.unique([GBD.shape for GBD in GBD_n])) <= 3, (
        'Gamut boundary descriptor matrices have incompatible shapes!')

    if gamut_boundary_descriptor_kwargs is None:
        gamut_boundary_descriptor_kwargs = {}

    shape_r, shape_c = GBD_n[0].shape[0], GBD_n[0].shape[1]

    Jab_n = [spherical_to_Jab(GBD) for GBD in GBD_n]

    settings = {'uniform': True}
    settings.update(kwargs)

    _figure, axes = artist(**settings)

    if show_segments:
        settings = {'axes': axes}
        settings.update(kwargs)
        settings['standalone'] = False

        plot_segments(shape_c + 1
                      if plane == 'ab' else shape_r * 2 + 1, **settings)

    gamut_boundary_descriptor_settings = {}
    gamut_boundary_descriptor_settings.update(gamut_boundary_descriptor_kwargs)
    if 'm' in gamut_boundary_descriptor_settings:
        gamut_boundary_descriptor_settings.pop('m')

    if 'n' in gamut_boundary_descriptor_settings:
        gamut_boundary_descriptor_settings.pop('n')

    plane = plane.lower()
    if plane == 'ab':
        x_label, y_label = 'a', 'b'

        for n, Jab in enumerate(Jab_n):
            J, a, b = tsplit(Jab)
            x, y = a, b

            if kwargs.get('show_debug_markers'):
                axes.scatter(x, y, marker='v', c='k', s=20)

            ij = polar_to_ij(
                gamut_boundary_descriptor(
                    tstack([y, x]),
                    m=shape_r,
                    n=shape_c,
                    **gamut_boundary_descriptor_settings))

            i, j = tsplit(ij.reshape([-1, 2]))

            axes.plot(
                np.hstack([i, i[0]]),
                np.hstack([j, j[0]]),
                'o-',
                alpha=0.85,
                label='GBD {0}'.format(n))
    else:
        x_label, y_label = 'a' if plane == 'ja' else 'b', 'J'

        for n, Jab in enumerate(Jab_n):
            J, a, b = tsplit(Jab)
            x, y = J, (a if plane == 'ja' else b)

            if kwargs.get('show_debug_markers'):
                axes.scatter(y, x, marker='v', c='k', s=20)

            ij = polar_to_ij(
                gamut_boundary_descriptor(
                    tstack([x, y]),
                    m=shape_r * 2,
                    n=shape_c * 2,
                    **gamut_boundary_descriptor_settings))

            i, j = tsplit(ij.reshape([-1, 2]))

            axes.plot(
                np.hstack([i, i[0]]),
                np.hstack([j, j[0]]),
                'o-',
                alpha=0.85,
                label='GBD {0}'.format(n))

    if kwargs.get('show_debug_circles'):
        if len(GBD_n) > 1:
            runtime_warning('Only displaying last "GBD" debug circles!')

        for i in range(len(x)):
            axes.add_artist(
                plt.Circle(
                    [0, 0],
                    np.linalg.norm([x[i], y[i]]),
                    fill=False,
                    alpha=CONSTANTS_COLOUR_STYLE.opacity.low))

    settings = {
        'axes':
            axes,
        'title':
            'Gamut Boundary Descriptors - {0}x{1} Segments'.format(
                shape_r, shape_c),
        'legend':
            True,
        'x_label':
            x_label,
        'y_label':
            y_label,
    }
    settings.update(kwargs)

    return render(**settings)


@override_style()
def plot_Jab_samples_in_segment_maxima_gamut_boundary(
        Jab_n,
        plane='Ja',
        gamut_boundary_descriptor_kwargs=None,
        scatter_kwargs=None,
        **kwargs):
    Jab_n = [as_float_array(Jab) for Jab in Jab_n]
    plane = plane.lower()

    if gamut_boundary_descriptor_kwargs is None:
        gamut_boundary_descriptor_kwargs = {}

    settings = {'uniform': True}
    settings.update(kwargs)

    _figure, axes = artist(**settings)

    gamut_boundary_descriptor_settings = {'E': np.array([50, 0, 0])}
    gamut_boundary_descriptor_settings.update(gamut_boundary_descriptor_kwargs)

    GBD_n = [
        gamut_boundary_descriptor(Jab, **gamut_boundary_descriptor_settings)
        for Jab in Jab_n
    ]

    settings = {
        'axes': axes,
        'standalone': False,
    }
    settings.update(kwargs)

    _figure, axes = plot_multi_segment_maxima_gamut_boundaries(
        GBD_n, plane=plane, **settings)

    for Jab in Jab_n:
        J, a, b = tsplit(Jab - gamut_boundary_descriptor_settings['E'])

        if plane == 'ab':
            x, y = a, b
        else:
            x, y = (a, J) if plane == 'ja' else (b, J)

        scatter_settings = {
            # 's': 40,
            # 'c': 'RGB',
            's': 20,
            'marker': '+',
            'alpha': 0.85,
        }
        if scatter_kwargs is not None:
            scatter_settings.update(scatter_kwargs)

        axes.scatter(x, y, **scatter_settings)

    settings = {
        'axes': axes,
        'standalone': True,
    }
    settings.update(kwargs)

    return render(**settings)
