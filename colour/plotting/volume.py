#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Colour Models Volume Plotting
=============================

Defines colour models volume and gamut plotting objects:

-   :func:`RGB_colourspaces_gamut_plot`
"""

from __future__ import division

import matplotlib
import numpy as np
import pylab
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from colour import (
    RGB_to_XYZ,
    XYZ_to_IPT,
    XYZ_to_Lab,
    XYZ_to_Luv,
    XYZ_to_UCS,
    XYZ_to_UVW,
    XYZ_to_xyY)
from colour.plotting import (
    CHROMATICITY_DIAGRAM_DEFAULT_ILLUMINANT,
    canvas,
    cube,
    decorate,
    display,
    equal_axes3d,
    get_RGB_colourspace,
    get_cmfs)
from colour.utilities import tsplit, tstack

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['COLOURSPACE_TO_LABELS',
           'RGB_identity_cube',
           'RGB_colourspaces_gamut_plot']

COLOURSPACE_TO_LABELS = {
    'CIE XYZ': ('X', 'Y', 'Z'),
    'CIE xyY': ('x', 'y', 'Y'),
    'CIE Lab': ('a', 'b', '$L^*$'),
    'CIE Luv': ('$u^\prime$', '$v^\prime$', '$L^*$'),
    'CIE UCS': ('U', 'V', 'W'),
    'CIE UVW': ('U', 'V', 'W'),
    'IPT': ('P', 'T', 'I')}
"""
Colourspace to labels mapping.

COLOURSPACE_TO_LABELS : dict
    {'CIE XYZ', 'CIE xyY', 'CIE Lab', 'CIE Luv', 'CIE UCS', 'CIE UVW', 'IPT'}
"""


def RGB_identity_cube(plane=None,
                      width_segments=16,
                      height_segments=16,
                      depth_segments=16):
    """
    Returns an *RGB* identity cube made of quad geometric elements along its
    associated *RGB* colours.

    Parameters
    ----------
    plane : array_like, optional
        Any combination of {'+x', '-x', '+y', '-y', '+z', '-z'}
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
        Cube vertices, *RGB* colours.

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

    quads = cube(plane=plane,
                 width=1,
                 height=1,
                 depth=1,
                 width_segments=width_segments,
                 height_segments=height_segments,
                 depth_segments=depth_segments)
    RGB = np.average(quads, axis=-2)

    return quads, RGB


def RGB_colourspaces_gamut_plot(colourspaces=None,
                                reference_colourspace='CIE xyY',
                                segments=8,
                                style=None,
                                azimuth=None,
                                elevation=None,
                                equal_aspect=True,
                                spectral_locus=False,
                                spectral_locus_colour=None,
                                cmfs='CIE 1931 2 Degree Standard Observer',
                                **kwargs):
    """
    Plots given *RGB* colourspaces gamut in given reference colourspace.

    Parameters
    ----------
    colourspaces : array_like, optional
        *RGB* colourspaces to plot the gamut.
    reference_colourspace : unicode, optional
        {'CIE XYZ', 'CIE xyY', 'CIE Lab', 'CIE Luv', 'CIE UCS', 'CIE UVW',
        'IPT'}
        Reference colourspace to plot the gamut into.
    segments : int, optional
        Edge segments count for each *RGB* colourspace cubes.
    style : dict, optional
        {'face_colours', 'edge_colours', 'edge_alpha', 'face_alpha'}
        Style for each given colourspace where each key has an array_like
        value. For example given 2 colourspaces, one can define *style* such
        as ``style = { 'face_colours': (None, (0.5, 0.5, 1.0)), 'edge_colours':
        (None, (0.5, 0.5, 1.0)), 'edge_alpha': (0.5, 1.0), 'face_alpha':
        (0.0, 1.0)}``
    azimuth : numeric, optional
        Plot camera azimuth.
    elevation : numeric, optional
        Plot camera elevation.
    equal_aspect : bool, optional
        Plot axes will have equal aspect ratio.
    spectral_locus : bool, optional
        Is spectral locus line plotted.
    spectral_locus_colour : array_like, optional
        Spectral locus line colour.
    cmfs : unicode, optional
        Standard observer colour matching functions used for spectral locus.
    \*\*kwargs : \*\*
        Keywords arguments.

    Returns
    -------
    bool
        Definition success.

    Examples
    --------
    >>> c = ['Rec. 709', 'ACEScg', 'S-Gamut']
    >>> RGB_colourspaces_gamut_plot(c)  # doctest: +SKIP
    True
    """

    if colourspaces is None:
        colourspaces = ('Rec. 709', 'ACEScg')

    count_c = len(colourspaces)
    style_c = {
        'face_colours': [None] * count_c,
        'edge_colours': [None] * count_c,
        'edge_alpha': [1] * count_c,
        'face_alpha': [1] * count_c}

    if style is not None:
        style_c.update(style)

    style = style_c

    settings = {
        'title': '{0} - {1} Reference Colourspace'.format(
            ', '.join(colourspaces), reference_colourspace)}
    settings.update(kwargs)

    figure = matplotlib.pyplot.figure()
    axes = figure.add_subplot(111, projection='3d')
    axes.view_init(elev=elevation, azim=azimuth)

    illuminant = CHROMATICITY_DIAGRAM_DEFAULT_ILLUMINANT

    points = np.zeros((4, 3))
    if spectral_locus:
        cmfs = get_cmfs(cmfs)
        XYZ = cmfs.values

        if reference_colourspace == 'CIE XYZ':
            points = XYZ
        if reference_colourspace == 'CIE xyY':
            points = XYZ_to_xyY(XYZ, illuminant)
        if reference_colourspace == 'CIE Lab':
            L, a, b = tsplit(XYZ_to_Lab(XYZ, illuminant))
            points = tstack((a, b, L))
        if reference_colourspace == 'CIE Luv':
            L, u, v = tsplit(XYZ_to_Luv(XYZ, illuminant))
            points = tstack((u, v, L))
        if reference_colourspace == 'CIE UCS':
            points = XYZ_to_UCS(XYZ)
        if reference_colourspace == 'CIE UVW':
            points = XYZ_to_UVW(XYZ * 100, illuminant)
        if reference_colourspace == 'IPT':
            I, P, T = tsplit(XYZ_to_IPT(XYZ))
            points = tstack((P, T, I))

        points[np.isnan(points)] = 0

        c = ((0.0, 0.0, 0.0, 0.5)
             if spectral_locus_colour is None else
             spectral_locus_colour)

        pylab.plot(points[..., 0],
                   points[..., 1],
                   points[..., 2],
                   color=c,
                   linewidth=2,
                   zorder=1)
        pylab.plot((points[-1][0], points[0][0]),
                   (points[-1][1], points[0][1]),
                   (points[-1][2], points[0][2]),
                   color=c,
                   linewidth=2,
                   zorder=1)

    quads, RGB_f, RGB_e = [], [], []
    for i, colourspace in enumerate(colourspaces):
        colourspace, name = get_RGB_colourspace(colourspace), colourspace
        quads_c, RGB = RGB_identity_cube(width_segments=segments,
                                         height_segments=segments,
                                         depth_segments=segments)

        XYZ = RGB_to_XYZ(
            quads_c,
            illuminant,
            illuminant,
            colourspace.RGB_to_XYZ_matrix)

        if reference_colourspace == 'CIE XYZ':
            quads.extend(XYZ)
        if reference_colourspace == 'CIE xyY':
            quads.extend(XYZ_to_xyY(XYZ, illuminant))
        if reference_colourspace == 'CIE Lab':
            L, a, b = tsplit(XYZ_to_Lab(XYZ, illuminant))
            quads.extend(tstack((a, b, L)))
        if reference_colourspace == 'CIE Luv':
            L, u, v = tsplit(XYZ_to_Luv(XYZ, illuminant))
            quads.extend(tstack((u, v, L)))
        if reference_colourspace == 'CIE UCS':
            quads.extend(XYZ_to_UCS(XYZ))
        if reference_colourspace == 'CIE UVW':
            quads.extend(XYZ_to_UVW(XYZ * 100, illuminant))
        if reference_colourspace == 'IPT':
            I, P, T = tsplit(XYZ_to_IPT(XYZ))
            quads.extend(tstack((P, T, I)))

        if style['face_colours'][i] is not None:
            RGB = np.ones(RGB.shape) * style['face_colours'][i]

        RGB_f.extend(
            np.hstack(
                (RGB, np.full((RGB.shape[0], 1), style['face_alpha'][i]))))

        if style['edge_colours'][i] is not None:
            RGB = np.ones(RGB.shape) * style['edge_colours'][i]

        RGB_e.extend(
            np.hstack(
                (RGB, np.full((RGB.shape[0], 1), style['edge_alpha'][i]))))

    quads = np.asarray(quads)
    quads[np.isnan(quads)] = 0

    collection = Poly3DCollection(quads)
    collection.set_facecolors(RGB_f)
    collection.set_edgecolors(RGB_e)

    axes.add_collection3d(collection)

    labels = COLOURSPACE_TO_LABELS[reference_colourspace]
    axes.set_xlabel(labels[0])
    axes.set_ylabel(labels[1])
    axes.set_zlabel(labels[2])

    for i, axis in enumerate('xyz'):
        min_a = np.min(np.vstack((quads[..., i], points[..., i])))
        max_a = np.max(np.vstack((quads[..., i], points[..., i])))
        getattr(axes, 'set_{}lim'.format(axis))((min_a, max_a))

    equal_aspect and equal_axes3d(axes)

    decorate(**settings)

    return display(**settings)
