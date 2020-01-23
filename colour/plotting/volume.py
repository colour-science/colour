# -*- coding: utf-8 -*-
"""
Colour Models Volume Plotting
=============================

Defines colour models volume and gamut plotting objects:

-   :func:`colour.plotting.plot_RGB_colourspaces_gamuts`
-   :func:`colour.plotting.plot_RGB_scatter`
"""

from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from colour.constants import DEFAULT_FLOAT_DTYPE, DEFAULT_INT_DTYPE
from colour.models import RGB_to_XYZ
from colour.models.common import (COLOURSPACE_MODELS_AXIS_LABELS,
                                  XYZ_to_colourspace_model)
from colour.plotting import (
    COLOUR_STYLE_CONSTANTS, common_colourspace_model_axis_reorder, cube,
    filter_RGB_colourspaces, filter_cmfs, grid, override_style, render)
from colour.utilities import (Structure, as_float_array, as_int_array,
                              first_item)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'nadir_grid', 'RGB_identity_cube', 'plot_RGB_colourspaces_gamuts',
    'plot_RGB_scatter'
]


def nadir_grid(limits=None, segments=10, labels=None, axes=None, **kwargs):
    """
    Returns a grid on *CIE xy* plane made of quad geometric elements and its
    associated faces and edges colours. Ticks and labels are added to the
    given axes according to the extended grid settings.

    Parameters
    ----------
    limits : array_like, optional
        Extended grid limits.
    segments : int, optional
        Edge segments count for the extended grid.
    labels : array_like, optional
        Axis labels.
    axes : matplotlib.axes.Axes, optional
        Axes to add the grid.

    Other Parameters
    ----------------
    grid_face_colours : array_like, optional
        Grid face colours array such as
        `grid_face_colours = (0.25, 0.25, 0.25)`.
    grid_edge_colours : array_like, optional
        Grid edge colours array such as
        `grid_edge_colours = (0.25, 0.25, 0.25)`.
    grid_face_alpha : numeric, optional
        Grid face opacity value such as `grid_face_alpha = 0.1`.
    grid_edge_alpha : numeric, optional
        Grid edge opacity value such as `grid_edge_alpha = 0.5`.
    x_axis_colour : array_like, optional
        *X* axis colour array such as `x_axis_colour = (0.0, 0.0, 0.0, 1.0)`.
    y_axis_colour : array_like, optional
        *Y* axis colour array such as `y_axis_colour = (0.0, 0.0, 0.0, 1.0)`.
    x_ticks_colour : array_like, optional
        *X* axis ticks colour array such as
        `x_ticks_colour = (0.0, 0.0, 0.0, 0.85)`.
    y_ticks_colour : array_like, optional
        *Y* axis ticks colour array such as
        `y_ticks_colour = (0.0, 0.0, 0.0, 0.85)`.
    x_label_colour : array_like, optional
        *X* axis label colour array such as
        `x_label_colour = (0.0, 0.0, 0.0, 0.85)`.
    y_label_colour : array_like, optional
        *Y* axis label colour array such as
        `y_label_colour = (0.0, 0.0, 0.0, 0.85)`.
    ticks_and_label_location : array_like, optional
        Location of the *X* and *Y* axis ticks and labels such as
        `ticks_and_label_location = ('-x', '-y')`.

    Returns
    -------
    tuple
        Grid quads, faces colours, edges colours.

    Examples
    --------
    >>> nadir_grid(segments=1)
    (array([[[-1.   , -1.   ,  0.   ],
            [ 1.   , -1.   ,  0.   ],
            [ 1.   ,  1.   ,  0.   ],
            [-1.   ,  1.   ,  0.   ]],
    <BLANKLINE>
           [[-1.   , -1.   ,  0.   ],
            [ 0.   , -1.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ],
            [-1.   ,  0.   ,  0.   ]],
    <BLANKLINE>
           [[-1.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ],
            [ 0.   ,  1.   ,  0.   ],
            [-1.   ,  1.   ,  0.   ]],
    <BLANKLINE>
           [[ 0.   , -1.   ,  0.   ],
            [ 1.   , -1.   ,  0.   ],
            [ 1.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ]],
    <BLANKLINE>
           [[ 0.   ,  0.   ,  0.   ],
            [ 1.   ,  0.   ,  0.   ],
            [ 1.   ,  1.   ,  0.   ],
            [ 0.   ,  1.   ,  0.   ]],
    <BLANKLINE>
           [[-1.   , -0.001,  0.   ],
            [ 1.   , -0.001,  0.   ],
            [ 1.   ,  0.001,  0.   ],
            [-1.   ,  0.001,  0.   ]],
    <BLANKLINE>
           [[-0.001, -1.   ,  0.   ],
            [ 0.001, -1.   ,  0.   ],
            [ 0.001,  1.   ,  0.   ],
            [-0.001,  1.   ,  0.   ]]]), array([[ 0.25,  0.25,  0.25,  0.1 ],
           [ 0.  ,  0.  ,  0.  ,  0.  ],
           [ 0.  ,  0.  ,  0.  ,  0.  ],
           [ 0.  ,  0.  ,  0.  ,  0.  ],
           [ 0.  ,  0.  ,  0.  ,  0.  ],
           [ 0.  ,  0.  ,  0.  ,  1.  ],
           [ 0.  ,  0.  ,  0.  ,  1.  ]]), array([[ 0.5 ,  0.5 ,  0.5 ,  0.5 ],
           [ 0.75,  0.75,  0.75,  0.25],
           [ 0.75,  0.75,  0.75,  0.25],
           [ 0.75,  0.75,  0.75,  0.25],
           [ 0.75,  0.75,  0.75,  0.25],
           [ 0.  ,  0.  ,  0.  ,  1.  ],
           [ 0.  ,  0.  ,  0.  ,  1.  ]]))
    """

    if limits is None:
        limits = np.array([[-1, 1], [-1, 1]])

    if labels is None:
        labels = ('x', 'y')

    extent = np.max(np.abs(limits[..., 1] - limits[..., 0]))

    settings = Structure(
        **{
            'grid_face_colours': (0.25, 0.25, 0.25),
            'grid_edge_colours': (0.50, 0.50, 0.50),
            'grid_face_alpha': 0.1,
            'grid_edge_alpha': 0.5,
            'x_axis_colour': (0.0, 0.0, 0.0, 1.0),
            'y_axis_colour': (0.0, 0.0, 0.0, 1.0),
            'x_ticks_colour': (0.0, 0.0, 0.0, 0.85),
            'y_ticks_colour': (0.0, 0.0, 0.0, 0.85),
            'x_label_colour': (0.0, 0.0, 0.0, 0.85),
            'y_label_colour': (0.0, 0.0, 0.0, 0.85),
            'ticks_and_label_location': ('-x', '-y')
        })
    settings.update(**kwargs)

    # Outer grid.
    quads_g = grid(
        origin=(-extent / 2, -extent / 2),
        width=extent,
        height=extent,
        height_segments=segments,
        width_segments=segments)

    RGB_g = np.ones((quads_g.shape[0], quads_g.shape[-1]))
    RGB_gf = RGB_g * settings.grid_face_colours
    RGB_gf = np.hstack([
        RGB_gf,
        np.full((RGB_gf.shape[0], 1), settings.grid_face_alpha,
                DEFAULT_FLOAT_DTYPE)
    ])
    RGB_ge = RGB_g * settings.grid_edge_colours
    RGB_ge = np.hstack([
        RGB_ge,
        np.full((RGB_ge.shape[0], 1), settings.grid_edge_alpha,
                DEFAULT_FLOAT_DTYPE)
    ])

    # Inner grid.
    quads_gs = grid(
        origin=(-extent / 2, -extent / 2),
        width=extent,
        height=extent,
        height_segments=segments * 2,
        width_segments=segments * 2)

    RGB_gs = np.ones((quads_gs.shape[0], quads_gs.shape[-1]))
    RGB_gsf = RGB_gs * 0
    RGB_gsf = np.hstack(
        [RGB_gsf,
         np.full((RGB_gsf.shape[0], 1), 0, DEFAULT_FLOAT_DTYPE)])
    RGB_gse = np.clip(RGB_gs * settings.grid_edge_colours * 1.5, 0, 1)
    RGB_gse = np.hstack(
        (RGB_gse,
         np.full((RGB_gse.shape[0], 1), settings.grid_edge_alpha / 2,
                 DEFAULT_FLOAT_DTYPE)))

    # Axis.
    thickness = extent / 1000
    quad_x = grid(
        origin=(limits[0, 0], -thickness / 2), width=extent, height=thickness)
    RGB_x = np.ones((quad_x.shape[0], quad_x.shape[-1] + 1))
    RGB_x = RGB_x * settings.x_axis_colour

    quad_y = grid(
        origin=(-thickness / 2, limits[1, 0]), width=thickness, height=extent)
    RGB_y = np.ones((quad_y.shape[0], quad_y.shape[-1] + 1))
    RGB_y = RGB_y * settings.y_axis_colour

    if axes is not None:
        # Ticks.
        x_s = 1 if '+x' in settings.ticks_and_label_location else -1
        y_s = 1 if '+y' in settings.ticks_and_label_location else -1
        for i, axis in enumerate('xy'):
            h_a = 'center' if axis == 'x' else 'left' if x_s == 1 else 'right'
            v_a = 'center'

            ticks = list(sorted(set(quads_g[..., 0, i])))
            ticks += [ticks[-1] + ticks[-1] - ticks[-2]]
            for tick in ticks:
                x = (limits[1, 1 if x_s == 1 else 0] + (x_s * extent / 25)
                     if i else tick)
                y = (tick if i else
                     limits[0, 1 if y_s == 1 else 0] + (y_s * extent / 25))

                tick = DEFAULT_INT_DTYPE(tick) if DEFAULT_FLOAT_DTYPE(
                    tick).is_integer() else tick
                c = settings['{0}_ticks_colour'.format(axis)]

                axes.text(
                    x,
                    y,
                    0,
                    tick,
                    'x',
                    horizontalalignment=h_a,
                    verticalalignment=v_a,
                    color=c,
                    clip_on=True)

        # Labels.
        for i, axis in enumerate('xy'):
            h_a = 'center' if axis == 'x' else 'left' if x_s == 1 else 'right'
            v_a = 'center'

            x = (limits[1, 1 if x_s == 1 else 0] + (x_s * extent / 10)
                 if i else 0)
            y = (0 if i else
                 limits[0, 1 if y_s == 1 else 0] + (y_s * extent / 10))

            c = settings['{0}_label_colour'.format(axis)]

            axes.text(
                x,
                y,
                0,
                labels[i],
                'x',
                horizontalalignment=h_a,
                verticalalignment=v_a,
                color=c,
                size=20,
                clip_on=True)

    quads = np.vstack([quads_g, quads_gs, quad_x, quad_y])
    RGB_f = np.vstack([RGB_gf, RGB_gsf, RGB_x, RGB_y])
    RGB_e = np.vstack([RGB_ge, RGB_gse, RGB_x, RGB_y])

    return quads, RGB_f, RGB_e


def RGB_identity_cube(plane=None,
                      width_segments=16,
                      height_segments=16,
                      depth_segments=16):
    """
    Returns an *RGB* identity cube made of quad geometric elements and its
    associated *RGB* colours.

    Parameters
    ----------
    plane : array_like, optional
        Any combination of **{'+x', '-x', '+y', '-y', '+z', '-z'}**,
        Included grids in the cube construction.
    width_segments: int, optional
        Cube segments, quad counts along the width.
    height_segments: int, optional
        Cube segments, quad counts along the height.
    depth_segments: int, optional
        Cube segments, quad counts along the depth.

    Returns
    -------
    tuple
        Cube quads, *RGB* colours.

    Examples
    --------
    >>> vertices, RGB = RGB_identity_cube(None, 1, 1, 1)
    >>> vertices
    array([[[ 0.,  0.,  0.],
            [ 1.,  0.,  0.],
            [ 1.,  1.,  0.],
            [ 0.,  1.,  0.]],
    <BLANKLINE>
           [[ 0.,  0.,  1.],
            [ 1.,  0.,  1.],
            [ 1.,  1.,  1.],
            [ 0.,  1.,  1.]],
    <BLANKLINE>
           [[ 0.,  0.,  0.],
            [ 1.,  0.,  0.],
            [ 1.,  0.,  1.],
            [ 0.,  0.,  1.]],
    <BLANKLINE>
           [[ 0.,  1.,  0.],
            [ 1.,  1.,  0.],
            [ 1.,  1.,  1.],
            [ 0.,  1.,  1.]],
    <BLANKLINE>
           [[ 0.,  0.,  0.],
            [ 0.,  1.,  0.],
            [ 0.,  1.,  1.],
            [ 0.,  0.,  1.]],
    <BLANKLINE>
           [[ 1.,  0.,  0.],
            [ 1.,  1.,  0.],
            [ 1.,  1.,  1.],
            [ 1.,  0.,  1.]]])
    >>> RGB
    array([[ 0.5,  0.5,  0. ],
           [ 0.5,  0.5,  1. ],
           [ 0.5,  0. ,  0.5],
           [ 0.5,  1. ,  0.5],
           [ 0. ,  0.5,  0.5],
           [ 1. ,  0.5,  0.5]])
    """

    quads = cube(
        plane=plane,
        width=1,
        height=1,
        depth=1,
        width_segments=width_segments,
        height_segments=height_segments,
        depth_segments=depth_segments)
    RGB = np.average(quads, axis=-2)

    return quads, RGB


@override_style()
def plot_RGB_colourspaces_gamuts(colourspaces=None,
                                 reference_colourspace='CIE xyY',
                                 segments=8,
                                 show_grid=True,
                                 grid_segments=10,
                                 show_spectral_locus=False,
                                 spectral_locus_colour=None,
                                 cmfs='CIE 1931 2 Degree Standard Observer',
                                 **kwargs):
    """
    Plots given *RGB* colourspaces gamuts in given reference colourspace.

    Parameters
    ----------
    colourspaces : array_like, optional
        *RGB* colourspaces to plot the gamuts.
    reference_colourspace : unicode, optional
        **{'CIE XYZ', 'CIE xyY', 'CIE xy', 'CIE Lab', 'CIE LCHab', 'CIE Luv',
        'CIE Luv uv', 'CIE LCHuv', 'CIE UCS', 'CIE UCS uv', 'CIE UVW',
        'DIN 99', 'Hunter Lab', 'Hunter Rdab', 'IPT', 'JzAzBz', 'OSA UCS',
        'hdr-CIELAB', 'hdr-IPT'}**,
        Reference colourspace to plot the gamuts into.
    segments : int, optional
        Edge segments count for each *RGB* colourspace cubes.
    show_grid : bool, optional
        Whether to show a grid at the bottom of the *RGB* colourspace cubes.
    grid_segments : bool, optional
        Edge segments count for the grid.
    show_spectral_locus : bool, optional
        Whether to show the spectral locus.
    spectral_locus_colour : array_like, optional
        Spectral locus colour.
    cmfs : unicode, optional
        Standard observer colour matching functions used for spectral locus.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.volume.nadir_grid`},
        Please refer to the documentation of the previously listed definitions.
    face_colours : array_like, optional
        Face colours array such as `face_colours = (None, (0.5, 0.5, 1.0))`.
    edge_colours : array_like, optional
        Edge colours array such as `edge_colours = (None, (0.5, 0.5, 1.0))`.
    face_alpha : numeric, optional
        Face opacity value such as `face_alpha = (0.5, 1.0)`.
    edge_alpha : numeric, optional
        Edge opacity value such as `edge_alpha = (0.0, 1.0)`.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> plot_RGB_colourspaces_gamuts(['ITU-R BT.709', 'ACEScg', 'S-Gamut'])
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, \
<matplotlib.axes._subplots.Axes3DSubplot object at 0x...>)

    .. image:: ../_static/Plotting_Plot_RGB_Colourspaces_Gamuts.png
        :align: center
        :alt: plot_RGB_colourspaces_gamuts
    """

    if colourspaces is None:
        colourspaces = ('ITU-R BT.709', 'ACEScg')

    colourspaces = filter_RGB_colourspaces(colourspaces).values()

    count_c = len(colourspaces)

    title = '{0} - {1} Reference Colourspace'.format(
        ', '.join([colourspace.name for colourspace in colourspaces]),
        reference_colourspace,
    )

    settings = Structure(
        **{
            'face_colours': [None] * count_c,
            'edge_colours': [None] * count_c,
            'face_alpha': [1] * count_c,
            'edge_alpha': [1] * count_c,
            'title': title,
        })
    settings.update(kwargs)

    figure = plt.figure()
    axes = figure.add_subplot(111, projection='3d')

    illuminant = COLOUR_STYLE_CONSTANTS.colour.colourspace.whitepoint

    points = np.zeros((4, 3))
    if show_spectral_locus:
        cmfs = first_item(filter_cmfs(cmfs).values())
        XYZ = cmfs.values

        points = common_colourspace_model_axis_reorder(
            XYZ_to_colourspace_model(XYZ, illuminant, reference_colourspace),
            reference_colourspace)

        points[np.isnan(points)] = 0

        c = ((0.0, 0.0, 0.0, 0.5)
             if spectral_locus_colour is None else spectral_locus_colour)

        axes.plot(
            points[..., 0], points[..., 1], points[..., 2], color=c, zorder=10)
        axes.plot(
            (points[-1][0], points[0][0]), (points[-1][1], points[0][1]),
            (points[-1][2], points[0][2]),
            color=c,
            zorder=10)

    quads, RGB_f, RGB_e = [], [], []
    for i, colourspace in enumerate(colourspaces):
        quads_c, RGB = RGB_identity_cube(
            width_segments=segments,
            height_segments=segments,
            depth_segments=segments)

        XYZ = RGB_to_XYZ(
            quads_c,
            colourspace.whitepoint,
            colourspace.whitepoint,
            colourspace.RGB_to_XYZ_matrix,
        )

        quads.extend(
            common_colourspace_model_axis_reorder(
                XYZ_to_colourspace_model(
                    XYZ,
                    colourspace.whitepoint,
                    reference_colourspace,
                ), reference_colourspace))

        if settings.face_colours[i] is not None:
            RGB = np.ones(RGB.shape) * settings.face_colours[i]

        RGB_f.extend(
            np.hstack([
                RGB,
                np.full((RGB.shape[0], 1), settings.face_alpha[i],
                        DEFAULT_FLOAT_DTYPE)
            ]))

        if settings.edge_colours[i] is not None:
            RGB = np.ones(RGB.shape) * settings.edge_colours[i]

        RGB_e.extend(
            np.hstack([
                RGB,
                np.full((RGB.shape[0], 1), settings.edge_alpha[i],
                        DEFAULT_FLOAT_DTYPE)
            ]))

    quads = as_float_array(quads)
    quads[np.isnan(quads)] = 0

    if quads.size != 0:
        for i, axis in enumerate('xyz'):
            min_a = min(np.min(quads[..., i]), np.min(points[..., i]))
            max_a = max(np.max(quads[..., i]), np.max(points[..., i]))
            getattr(axes, 'set_{}lim'.format(axis))((min_a, max_a))

    labels = np.array(
        COLOURSPACE_MODELS_AXIS_LABELS[reference_colourspace])[as_int_array(
            common_colourspace_model_axis_reorder([0, 1, 2],
                                                  reference_colourspace))]
    for i, axis in enumerate('xyz'):
        getattr(axes, 'set_{}label'.format(axis))(labels[i])

    if show_grid:
        limits = np.array([[-1.5, 1.5], [-1.5, 1.5]])

        quads_g, RGB_gf, RGB_ge = nadir_grid(limits, grid_segments, labels,
                                             axes, **settings)
        quads = np.vstack([quads_g, quads])
        RGB_f = np.vstack([RGB_gf, RGB_f])
        RGB_e = np.vstack([RGB_ge, RGB_e])

    collection = Poly3DCollection(quads)
    collection.set_facecolors(RGB_f)
    collection.set_edgecolors(RGB_e)

    axes.add_collection3d(collection)

    settings.update({
        'axes': axes,
        'axes_visible': False,
        'camera_aspect': 'equal'
    })
    settings.update(kwargs)

    return render(**settings)


@override_style()
def plot_RGB_scatter(RGB,
                     colourspace,
                     reference_colourspace='CIE xyY',
                     colourspaces=None,
                     segments=8,
                     show_grid=True,
                     grid_segments=10,
                     show_spectral_locus=False,
                     spectral_locus_colour=None,
                     points_size=12,
                     cmfs='CIE 1931 2 Degree Standard Observer',
                     **kwargs):
    """
    Plots given *RGB* colourspace array in a scatter plot.

    Parameters
    ----------
    RGB : array_like
        *RGB* colourspace array.
    colourspace : RGB_Colourspace
        *RGB* colourspace of the *RGB* array.
    reference_colourspace : unicode, optional
        **{'CIE XYZ', 'CIE xyY', 'CIE xy', 'CIE Lab', 'CIE LCHab', 'CIE Luv',
        'CIE Luv uv', 'CIE LCHuv', 'CIE UCS', 'CIE UCS uv', 'CIE UVW',
        'DIN 99', 'Hunter Lab', 'Hunter Rdab', 'IPT', 'JzAzBz', 'OSA UCS',
        'hdr-CIELAB', 'hdr-IPT'}**,
        Reference colourspace for colour conversion.
    colourspaces : array_like, optional
        *RGB* colourspaces to plot the gamuts.
    segments : int, optional
        Edge segments count for each *RGB* colourspace cubes.
    show_grid : bool, optional
        Whether to show a grid at the bottom of the *RGB* colourspace cubes.
    grid_segments : bool, optional
        Edge segments count for the grid.
    show_spectral_locus : bool, optional
        Whether to show the spectral locus.
    spectral_locus_colour : array_like, optional
        Spectral locus colour.
    points_size : numeric, optional
        Scatter points size.
    cmfs : unicode, optional
        Standard observer colour matching functions used for spectral locus.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.plot_RGB_colourspaces_gamuts`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> RGB = np.random.random((128, 128, 3))
    >>> plot_RGB_scatter(RGB, 'ITU-R BT.709')  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, \
<matplotlib.axes._subplots.Axes3DSubplot object at 0x...>)

    .. image:: ../_static/Plotting_Plot_RGB_Scatter.png
        :align: center
        :alt: plot_RGB_scatter
    """

    colourspace = first_item(filter_RGB_colourspaces(colourspace).values())

    if colourspaces is None:
        colourspaces = (colourspace.name, )

    count_c = len(colourspaces)
    settings = Structure(
        **{
            'face_colours': [None] * count_c,
            'edge_colours': [(0.25, 0.25, 0.25)] * count_c,
            'face_alpha': [0.0] * count_c,
            'edge_alpha': [0.1] * count_c,
        })
    settings.update(kwargs)
    settings['standalone'] = False

    plot_RGB_colourspaces_gamuts(
        colourspaces=colourspaces,
        reference_colourspace=reference_colourspace,
        segments=segments,
        show_grid=show_grid,
        grid_segments=grid_segments,
        show_spectral_locus=show_spectral_locus,
        spectral_locus_colour=spectral_locus_colour,
        cmfs=cmfs,
        **settings)

    XYZ = RGB_to_XYZ(RGB, colourspace.whitepoint, colourspace.whitepoint,
                     colourspace.RGB_to_XYZ_matrix)

    points = common_colourspace_model_axis_reorder(
        XYZ_to_colourspace_model(XYZ, colourspace.whitepoint,
                                 reference_colourspace), reference_colourspace)

    axes = plt.gca()
    axes.scatter(
        points[..., 0],
        points[..., 1],
        points[..., 2],
        color=np.reshape(RGB, (-1, 3)),
        s=points_size)

    settings.update({'axes': axes, 'standalone': True})
    settings.update(kwargs)

    return render(**settings)
