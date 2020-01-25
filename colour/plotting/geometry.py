# -*- coding: utf-8 -*-
"""
Geometry Plotting Utilities
===========================

Defines geometry plotting utilities objects:

-   :func:`colour.plotting.quad`
-   :func:`colour.plotting.grid`
-   :func:`colour.plotting.cube`
"""

from __future__ import division

import numpy as np

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['quad', 'grid', 'cube']


def quad(plane='xy', origin=None, width=1, height=1, depth=0):
    """
    Returns the vertices of a quad geometric element in counter-clockwise
    order.

    Parameters
    ----------
    plane : array_like, optional
        **{'xy', 'xz', 'yz'}**,
        Construction plane of the quad.
    origin: array_like, optional
        Quad origin on the construction plane.
    width: numeric, optional
        Quad width.
    height: numeric, optional
        Quad height.
    depth: numeric, optional
        Quad depth.

    Returns
    -------
    ndarray
        Quad vertices.

    Examples
    --------
    >>> quad()
    array([[0, 0, 0],
           [1, 0, 0],
           [1, 1, 0],
           [0, 1, 0]])
    """

    u, v = (0, 0) if origin is None else origin

    plane = plane.lower()
    if plane == 'xy':
        vertices = ((u, v, depth), (u + width, v, depth),
                    (u + width, v + height, depth), (u, v + height, depth))
    elif plane == 'xz':
        vertices = ((u, depth, v), (u + width, depth, v),
                    (u + width, depth, v + height), (u, depth, v + height))
    elif plane == 'yz':
        vertices = ((depth, u, v), (depth, u + width, v),
                    (depth, u + width, v + height), (depth, u, v + height))
    else:
        raise ValueError('"{0}" is not a supported plane!'.format(plane))

    return np.array(vertices)


def grid(plane='xy',
         origin=None,
         width=1,
         height=1,
         depth=0,
         width_segments=1,
         height_segments=1):
    """
    Returns the vertices of a grid made of quads.

    Parameters
    ----------
    plane : array_like, optional
        **{'xy', 'xz', 'yz'}**,
        Construction plane of the grid.
    origin: array_like, optional
        Grid origin on the construction plane.
    width: numeric, optional
        Grid width.
    height: numeric, optional
        Grid height.
    depth: numeric, optional
        Grid depth.
    width_segments: int, optional
        Grid segments, quad counts along the width.
    height_segments: int, optional
        Grid segments, quad counts along the height.

    Returns
    -------
    ndarray
        Grid vertices.

    Examples
    --------
    >>> grid(width_segments=2, height_segments=2)
    array([[[ 0. ,  0. ,  0. ],
            [ 0.5,  0. ,  0. ],
            [ 0.5,  0.5,  0. ],
            [ 0. ,  0.5,  0. ]],
    <BLANKLINE>
           [[ 0. ,  0.5,  0. ],
            [ 0.5,  0.5,  0. ],
            [ 0.5,  1. ,  0. ],
            [ 0. ,  1. ,  0. ]],
    <BLANKLINE>
           [[ 0.5,  0. ,  0. ],
            [ 1. ,  0. ,  0. ],
            [ 1. ,  0.5,  0. ],
            [ 0.5,  0.5,  0. ]],
    <BLANKLINE>
           [[ 0.5,  0.5,  0. ],
            [ 1. ,  0.5,  0. ],
            [ 1. ,  1. ,  0. ],
            [ 0.5,  1. ,  0. ]]])
    """

    u, v = (0, 0) if origin is None else origin

    w_x, h_y = width / width_segments, height / height_segments

    quads = []
    for i in range(width_segments):
        for j in range(height_segments):
            quads.append(
                quad(plane, (i * w_x + u, j * h_y + v), w_x, h_y, depth))

    return np.array(quads)


def cube(plane=None,
         origin=None,
         width=1,
         height=1,
         depth=1,
         width_segments=1,
         height_segments=1,
         depth_segments=1):
    """
    Returns the vertices of a cube made of grids.

    Parameters
    ----------
    plane : array_like, optional
        Any combination of **{'+x', '-x', '+y', '-y', '+z', '-z'}**,
        Included grids in the cube construction.
    origin: array_like, optional
        Cube origin.
    width: numeric, optional
        Cube width.
    height: numeric, optional
        Cube height.
    depth: numeric, optional
        Cube depth.
    width_segments: int, optional
        Cube segments, quad counts along the width.
    height_segments: int, optional
        Cube segments, quad counts along the height.
    depth_segments: int, optional
        Cube segments, quad counts along the depth.

    Returns
    -------
    ndarray
        Cube vertices.

    Examples
    --------
    >>> cube()
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
    """

    plane = (('+x', '-x', '+y', '-y', '+z', '-z')
             if plane is None else [p.lower() for p in plane])
    u, v, w = (0, 0, 0) if origin is None else origin

    w_s, h_s, d_s = width_segments, height_segments, depth_segments

    grids = []
    if '-z' in plane:
        grids.extend(grid('xy', (u, w), width, depth, v, w_s, d_s))
    if '+z' in plane:
        grids.extend(grid('xy', (u, w), width, depth, v + height, w_s, d_s))

    if '-y' in plane:
        grids.extend(grid('xz', (u, v), width, height, w, w_s, h_s))
    if '+y' in plane:
        grids.extend(grid('xz', (u, v), width, height, w + depth, w_s, h_s))

    if '-x' in plane:
        grids.extend(grid('yz', (w, v), depth, height, u, d_s, h_s))
    if '+x' in plane:
        grids.extend(grid('yz', (w, v), depth, height, u + width, d_s, h_s))

    return np.array(grids)
