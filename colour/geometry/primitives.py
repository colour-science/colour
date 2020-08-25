# -*- coding: utf-8 -*-
"""
Geometry Primitives
===================

Defines various geometry primitives and their generation methods:

-   :func:`colour.geometry.PLANE_TO_AXIS_MAPPING`
-   :func:`colour.geometry.primitive_grid`
-   :func:`colour.geometry.primitive_cube`
-   :func:`colour.PRIMITIVE_METHODS`
-   :func:`colour.primitive`

References
----------
-   :cite:`Cabello2015` : Cabello, R. (n.d.). PlaneGeometry.js. Retrieved May
    12, 2015, from
    https://github.com/mrdoob/three.js/blob/dev/src/geometries/PlaneGeometry.js
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.constants import DEFAULT_INT_DTYPE, DEFAULT_FLOAT_DTYPE
from colour.utilities import CaseInsensitiveMapping, filter_kwargs, ones, zeros

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'PLANE_TO_AXIS_MAPPING',
    'primitive_grid',
    'primitive_cube',
    'PRIMITIVE_METHODS',
    'primitive',
]

PLANE_TO_AXIS_MAPPING = CaseInsensitiveMapping({
    'yz': '+x',
    'zy': '-x',
    'xz': '+y',
    'zx': '-y',
    'xy': '+z',
    'yx': '-z',
})
PLANE_TO_AXIS_MAPPING.__doc__ = """
Plane to axis mapping.

PLANE_TO_AXIS_MAPPING : CaseInsensitiveMapping
    **{'-x', '+x', '-y', '+y', '-z', '+z'}**
"""


def primitive_grid(width=1,
                   height=1,
                   width_segments=1,
                   height_segments=1,
                   axis='+z'):
    """
    Generates vertices and indices for a filled and outlined grid primitive.

    Parameters
    ----------
    width : float, optional
        Grid width.
    height : float, optional
        Grid height.
    width_segments : int, optional
        Grid segments count along the width.
    height_segments : float, optional
        Grid segments count along the height.
    axis : unicode, optional
        **{'+z', '-x', '+x', '-y', '+y', '-z',
        'xy', 'xz', 'yz', 'yx', 'zx', 'zy'}**,
        Axis the primitive will be normal to, or plane the primitive will be
        co-planar with.

    Returns
    -------
    tuple
        Tuple of grid vertices, face indices to produce a filled grid and
        outline indices to produce an outline of the faces of the grid.

    References
    ----------
    :cite:`Cabello2015`

    Examples
    --------
    >>> vertices, faces, outline = primitive_grid()
    >>> print(vertices)
    [([-0.5,  0.5,  0. ], [ 0.,  1.], [ 0.,  0.,  1.], [ 0.,  1.,  0.,  1.])
     ([ 0.5,  0.5,  0. ], [ 1.,  1.], [ 0.,  0.,  1.], [ 1.,  1.,  0.,  1.])
     ([-0.5, -0.5,  0. ], [ 0.,  0.], [ 0.,  0.,  1.], [ 0.,  0.,  0.,  1.])
     ([ 0.5, -0.5,  0. ], [ 1.,  0.], [ 0.,  0.,  1.], [ 1.,  0.,  0.,  1.])]
    >>> print(faces)
    [[0 2 1]
     [2 3 1]]
    >>> print(outline)
    [[0 2]
     [2 3]
     [3 1]
     [1 0]]
    """

    axis = PLANE_TO_AXIS_MAPPING.get(axis, axis).lower()

    x_grid = width_segments
    y_grid = height_segments

    x_grid1 = x_grid + 1
    y_grid1 = y_grid + 1

    # Positions, normals and uvs.
    positions = zeros(x_grid1 * y_grid1 * 3)
    normals = zeros(x_grid1 * y_grid1 * 3)
    uvs = zeros(x_grid1 * y_grid1 * 2)

    y = np.arange(y_grid1) * height / y_grid - height / 2
    x = np.arange(x_grid1) * width / x_grid - width / 2

    positions[::3] = np.tile(x, y_grid1)
    positions[1::3] = -np.repeat(y, x_grid1)

    normals[2::3] = 1

    uvs[::2] = np.tile(np.arange(x_grid1) / x_grid, y_grid1)
    uvs[1::2] = np.repeat(1 - np.arange(y_grid1) / y_grid, x_grid1)

    # Faces and outline.
    faces, outline = [], []
    for i_y in range(y_grid):
        for i_x in range(x_grid):
            a = i_x + x_grid1 * i_y
            b = i_x + x_grid1 * (i_y + 1)
            c = (i_x + 1) + x_grid1 * (i_y + 1)
            d = (i_x + 1) + x_grid1 * i_y

            faces.extend([(a, b, d), (b, c, d)])
            outline.extend([(a, b), (b, c), (c, d), (d, a)])

    positions = np.reshape(positions, (-1, 3))
    uvs = np.reshape(uvs, (-1, 2))
    normals = np.reshape(normals, (-1, 3))

    faces = np.reshape(faces, (-1, 3)).astype(np.uint32)
    outline = np.reshape(outline, (-1, 2)).astype(np.uint32)

    if axis in ('-x', '+x'):
        shift, zero_axis = 1, 0
    elif axis in ('-y', '+y'):
        shift, zero_axis = -1, 1
    elif axis in ('-z', '+z'):
        shift, zero_axis = 0, 2

    sign = -1 if '-' in axis else 1

    positions = np.roll(positions, shift, -1)
    normals = np.roll(normals, shift, -1) * sign
    vertex_colours = np.ravel(positions)
    vertex_colours = np.hstack([
        np.reshape(
            np.interp(vertex_colours,
                      (np.min(vertex_colours), np.max(vertex_colours)),
                      (0, 1)), positions.shape),
        ones([positions.shape[0], 1])
    ])
    vertex_colours[..., zero_axis] = 0

    vertices = zeros(positions.shape[0], [
        ('position', DEFAULT_FLOAT_DTYPE, 3),
        ('uv', DEFAULT_FLOAT_DTYPE, 2),
        ('normal', DEFAULT_FLOAT_DTYPE, 3),
        ('colour', DEFAULT_FLOAT_DTYPE, 4),
    ])

    vertices['position'] = positions
    vertices['uv'] = uvs
    vertices['normal'] = normals
    vertices['colour'] = vertex_colours

    return vertices, faces, outline


def primitive_cube(width=1,
                   height=1,
                   depth=1,
                   width_segments=1,
                   height_segments=1,
                   depth_segments=1,
                   planes=None):
    """
    Generates vertices and indices for a filled and outlined cube primitive.

    Parameters
    ----------
    width : float, optional
        Cube width.
    height : float, optional
        Cube height.
    depth : float, optional
        Cube depth.
    width_segments : int, optional
        Cube segments count along the width.
    height_segments : float, optional
        Cube segments count along the height.
    depth_segments : float, optional
        Cube segments count along the depth.
    planes : array_like, optional
        **{'-x', '+x', '-y', '+y', '-z', '+z',
        'xy', 'xz', 'yz', 'yx', 'zx', 'zy'}**,
        Grid primitives to include in the cube construction.

    Returns
    -------
    tuple
        Tuple of cube vertices, face indices to produce a filled cube and
        outline indices to produce an outline of the faces of the cube.

    Examples
    --------
    >>> vertices, faces, outline = primitive_cube()
    >>> print(vertices)
    [([-0.5,  0.5, -0.5], [ 0.,  1.], [-0., -0., -1.], [ 0.,  1.,  0.,  1.])
     ([ 0.5,  0.5, -0.5], [ 1.,  1.], [-0., -0., -1.], [ 1.,  1.,  0.,  1.])
     ([-0.5, -0.5, -0.5], [ 0.,  0.], [-0., -0., -1.], [ 0.,  0.,  0.,  1.])
     ([ 0.5, -0.5, -0.5], [ 1.,  0.], [-0., -0., -1.], [ 1.,  0.,  0.,  1.])
     ([-0.5,  0.5,  0.5], [ 0.,  1.], [ 0.,  0.,  1.], [ 0.,  1.,  1.,  1.])
     ([ 0.5,  0.5,  0.5], [ 1.,  1.], [ 0.,  0.,  1.], [ 1.,  1.,  1.,  1.])
     ([-0.5, -0.5,  0.5], [ 0.,  0.], [ 0.,  0.,  1.], [ 0.,  0.,  1.,  1.])
     ([ 0.5, -0.5,  0.5], [ 1.,  0.], [ 0.,  0.,  1.], [ 1.,  0.,  1.,  1.])
     ([ 0.5, -0.5, -0.5], [ 0.,  1.], [-0., -1., -0.], [ 1.,  0.,  0.,  1.])
     ([ 0.5, -0.5,  0.5], [ 1.,  1.], [-0., -1., -0.], [ 1.,  0.,  1.,  1.])
     ([-0.5, -0.5, -0.5], [ 0.,  0.], [-0., -1., -0.], [ 0.,  0.,  0.,  1.])
     ([-0.5, -0.5,  0.5], [ 1.,  0.], [-0., -1., -0.], [ 0.,  0.,  1.,  1.])
     ([ 0.5,  0.5, -0.5], [ 0.,  1.], [ 0.,  1.,  0.], [ 1.,  1.,  0.,  1.])
     ([ 0.5,  0.5,  0.5], [ 1.,  1.], [ 0.,  1.,  0.], [ 1.,  1.,  1.,  1.])
     ([-0.5,  0.5, -0.5], [ 0.,  0.], [ 0.,  1.,  0.], [ 0.,  1.,  0.,  1.])
     ([-0.5,  0.5,  0.5], [ 1.,  0.], [ 0.,  1.,  0.], [ 0.,  1.,  1.,  1.])
     ([-0.5, -0.5,  0.5], [ 0.,  1.], [-1., -0., -0.], [ 0.,  0.,  1.,  1.])
     ([-0.5,  0.5,  0.5], [ 1.,  1.], [-1., -0., -0.], [ 0.,  1.,  1.,  1.])
     ([-0.5, -0.5, -0.5], [ 0.,  0.], [-1., -0., -0.], [ 0.,  0.,  0.,  1.])
     ([-0.5,  0.5, -0.5], [ 1.,  0.], [-1., -0., -0.], [ 0.,  1.,  0.,  1.])
     ([ 0.5, -0.5,  0.5], [ 0.,  1.], [ 1.,  0.,  0.], [ 1.,  0.,  1.,  1.])
     ([ 0.5,  0.5,  0.5], [ 1.,  1.], [ 1.,  0.,  0.], [ 1.,  1.,  1.,  1.])
     ([ 0.5, -0.5, -0.5], [ 0.,  0.], [ 1.,  0.,  0.], [ 1.,  0.,  0.,  1.])
     ([ 0.5,  0.5, -0.5], [ 1.,  0.], [ 1.,  0.,  0.], [ 1.,  1.,  0.,  1.])]
    >>> print(faces)
    [[ 1  2  0]
     [ 1  3  2]
     [ 4  6  5]
     [ 6  7  5]
     [ 9 10  8]
     [ 9 11 10]
     [12 14 13]
     [14 15 13]
     [17 18 16]
     [17 19 18]
     [20 22 21]
     [22 23 21]]
    >>> print(outline)
    [[ 0  2]
     [ 2  3]
     [ 3  1]
     [ 1  0]
     [ 4  6]
     [ 6  7]
     [ 7  5]
     [ 5  4]
     [ 8 10]
     [10 11]
     [11  9]
     [ 9  8]
     [12 14]
     [14 15]
     [15 13]
     [13 12]
     [16 18]
     [18 19]
     [19 17]
     [17 16]
     [20 22]
     [22 23]
     [23 21]
     [21 20]]
    """

    planes = (sorted(list(
        PLANE_TO_AXIS_MAPPING.values())) if planes is None else [
            PLANE_TO_AXIS_MAPPING.get(plane, plane).lower() for plane in planes
        ])

    w_s, h_s, d_s = width_segments, height_segments, depth_segments

    planes_m = []
    if '-z' in planes:
        planes_m.append(list(primitive_grid(width, depth, w_s, d_s, '-z')))
        planes_m[-1][0]['position'][..., 2] -= height / 2
        planes_m[-1][1] = np.fliplr(planes_m[-1][1])
    if '+z' in planes:
        planes_m.append(list(primitive_grid(width, depth, w_s, d_s, '+z')))
        planes_m[-1][0]['position'][..., 2] += height / 2

    if '-y' in planes:
        planes_m.append(list(primitive_grid(height, width, h_s, w_s, '-y')))
        planes_m[-1][0]['position'][..., 1] -= depth / 2
        planes_m[-1][1] = np.fliplr(planes_m[-1][1])
    if '+y' in planes:
        planes_m.append(list(primitive_grid(height, width, h_s, w_s, '+y')))
        planes_m[-1][0]['position'][..., 1] += depth / 2

    if '-x' in planes:
        planes_m.append(list(primitive_grid(depth, height, d_s, h_s, '-x')))
        planes_m[-1][0]['position'][..., 0] -= width / 2
        planes_m[-1][1] = np.fliplr(planes_m[-1][1])
    if '+x' in planes:
        planes_m.append(list(primitive_grid(depth, height, d_s, h_s, '+x')))
        planes_m[-1][0]['position'][..., 0] += width / 2

    positions = zeros([0, 3])
    uvs = zeros([0, 2])
    normals = zeros([0, 3])

    faces = zeros([0, 3], dtype=DEFAULT_INT_DTYPE)
    outline = zeros([0, 2], dtype=DEFAULT_INT_DTYPE)

    offset = 0
    for vertices_p, faces_p, outline_p in planes_m:
        positions = np.vstack([positions, vertices_p['position']])
        uvs = np.vstack([uvs, vertices_p['uv']])
        normals = np.vstack([normals, vertices_p['normal']])

        faces = np.vstack([faces, faces_p + offset])
        outline = np.vstack([outline, outline_p + offset])
        offset += vertices_p['position'].shape[0]

    vertices = zeros(positions.shape[0], [('position', DEFAULT_FLOAT_DTYPE, 3),
                                          ('uv', DEFAULT_FLOAT_DTYPE, 2),
                                          ('normal', DEFAULT_FLOAT_DTYPE, 3),
                                          ('colour', DEFAULT_FLOAT_DTYPE, 4)])

    vertex_colours = np.ravel(positions)
    vertex_colours = np.hstack([
        np.reshape(
            np.interp(vertex_colours,
                      (np.min(vertex_colours), np.max(vertex_colours)),
                      (0, 1)), positions.shape),
        ones([positions.shape[0], 1])
    ])

    vertices['position'] = positions
    vertices['uv'] = uvs
    vertices['normal'] = normals
    vertices['colour'] = vertex_colours

    return vertices, faces, outline


PRIMITIVE_METHODS = CaseInsensitiveMapping({
    'Grid': primitive_grid,
    'Cube': primitive_cube,
})
PRIMITIVE_METHODS.__doc__ = """
Supported geometry primitive generation methods.

PRIMITIVE_METHODS : CaseInsensitiveMapping
    **{'Grid', 'Cube'}**
"""


def primitive(method='Cube', **kwargs):
    """
    Returns a geometry primitive using given method.

    Parameters
    ----------
    method : unicode, optional
        **{'Cube', 'Grid'}**,
        Generation method.

    Other Parameters
    ----------------
    width : numeric, optional
        {:func:`colour.geometry.primitive_grid_mpl`,
        :func:`colour.geometry.primitive_cube_mpl`},
        Primitive width.
    height : numeric, optional
        {:func:`colour.geometry.primitive_grid_mpl`,
        :func:`colour.geometry.primitive_cube_mpl`},
        Primitive height.
    depth : numeric, optional
        {:func:`colour.geometry.primitive_grid_mpl`,
        :func:`colour.geometry.primitive_cube_mpl`},
        Primitive depth.
    width_segments
        {:func:`colour.geometry.primitive_grid_mpl`,
        :func:`colour.geometry.primitive_cube_mpl`},
        Primitive segments count along the width.
    height_segments
        {:func:`colour.geometry.primitive_grid_mpl`,
        :func:`colour.geometry.primitive_cube_mpl`},
        Primitive segments count along the height.
    depth_segments
        {:func:`colour.geometry.primitive_grid_mpl`,
        :func:`colour.geometry.primitive_cube_mpl`},
        Primitive segments count along the depth.
    planes : array_like, optional
        {:func:`colour.geometry.primitive_cube_mpl`},
        **{'-x', '+x', '-y', '+y', '-z', '+z',
        'xy', 'xz', 'yz', 'yx', 'zx', 'zy'}**,
        Included grid primitives in the cube construction.

    Returns
    -------

    References
    ----------
    :cite:`Cabello2015`

    Examples
    --------
    >>> vertices, faces, outline = primitive()
    >>> print(vertices)
    [([-0.5,  0.5, -0.5], [ 0.,  1.], [-0., -0., -1.], [ 0.,  1.,  0.,  1.])
     ([ 0.5,  0.5, -0.5], [ 1.,  1.], [-0., -0., -1.], [ 1.,  1.,  0.,  1.])
     ([-0.5, -0.5, -0.5], [ 0.,  0.], [-0., -0., -1.], [ 0.,  0.,  0.,  1.])
     ([ 0.5, -0.5, -0.5], [ 1.,  0.], [-0., -0., -1.], [ 1.,  0.,  0.,  1.])
     ([-0.5,  0.5,  0.5], [ 0.,  1.], [ 0.,  0.,  1.], [ 0.,  1.,  1.,  1.])
     ([ 0.5,  0.5,  0.5], [ 1.,  1.], [ 0.,  0.,  1.], [ 1.,  1.,  1.,  1.])
     ([-0.5, -0.5,  0.5], [ 0.,  0.], [ 0.,  0.,  1.], [ 0.,  0.,  1.,  1.])
     ([ 0.5, -0.5,  0.5], [ 1.,  0.], [ 0.,  0.,  1.], [ 1.,  0.,  1.,  1.])
     ([ 0.5, -0.5, -0.5], [ 0.,  1.], [-0., -1., -0.], [ 1.,  0.,  0.,  1.])
     ([ 0.5, -0.5,  0.5], [ 1.,  1.], [-0., -1., -0.], [ 1.,  0.,  1.,  1.])
     ([-0.5, -0.5, -0.5], [ 0.,  0.], [-0., -1., -0.], [ 0.,  0.,  0.,  1.])
     ([-0.5, -0.5,  0.5], [ 1.,  0.], [-0., -1., -0.], [ 0.,  0.,  1.,  1.])
     ([ 0.5,  0.5, -0.5], [ 0.,  1.], [ 0.,  1.,  0.], [ 1.,  1.,  0.,  1.])
     ([ 0.5,  0.5,  0.5], [ 1.,  1.], [ 0.,  1.,  0.], [ 1.,  1.,  1.,  1.])
     ([-0.5,  0.5, -0.5], [ 0.,  0.], [ 0.,  1.,  0.], [ 0.,  1.,  0.,  1.])
     ([-0.5,  0.5,  0.5], [ 1.,  0.], [ 0.,  1.,  0.], [ 0.,  1.,  1.,  1.])
     ([-0.5, -0.5,  0.5], [ 0.,  1.], [-1., -0., -0.], [ 0.,  0.,  1.,  1.])
     ([-0.5,  0.5,  0.5], [ 1.,  1.], [-1., -0., -0.], [ 0.,  1.,  1.,  1.])
     ([-0.5, -0.5, -0.5], [ 0.,  0.], [-1., -0., -0.], [ 0.,  0.,  0.,  1.])
     ([-0.5,  0.5, -0.5], [ 1.,  0.], [-1., -0., -0.], [ 0.,  1.,  0.,  1.])
     ([ 0.5, -0.5,  0.5], [ 0.,  1.], [ 1.,  0.,  0.], [ 1.,  0.,  1.,  1.])
     ([ 0.5,  0.5,  0.5], [ 1.,  1.], [ 1.,  0.,  0.], [ 1.,  1.,  1.,  1.])
     ([ 0.5, -0.5, -0.5], [ 0.,  0.], [ 1.,  0.,  0.], [ 1.,  0.,  0.,  1.])
     ([ 0.5,  0.5, -0.5], [ 1.,  0.], [ 1.,  0.,  0.], [ 1.,  1.,  0.,  1.])]
    >>> print(faces)
    [[ 1  2  0]
     [ 1  3  2]
     [ 4  6  5]
     [ 6  7  5]
     [ 9 10  8]
     [ 9 11 10]
     [12 14 13]
     [14 15 13]
     [17 18 16]
     [17 19 18]
     [20 22 21]
     [22 23 21]]
    >>> print(outline)
    [[ 0  2]
     [ 2  3]
     [ 3  1]
     [ 1  0]
     [ 4  6]
     [ 6  7]
     [ 7  5]
     [ 5  4]
     [ 8 10]
     [10 11]
     [11  9]
     [ 9  8]
     [12 14]
     [14 15]
     [15 13]
     [13 12]
     [16 18]
     [18 19]
     [19 17]
     [17 16]
     [20 22]
     [22 23]
     [23 21]
     [21 20]]
    >>> vertices, faces, outline = primitive('Grid')
    >>> print(vertices)
    [([-0.5,  0.5,  0. ], [ 0.,  1.], [ 0.,  0.,  1.], [ 0.,  1.,  0.,  1.])
     ([ 0.5,  0.5,  0. ], [ 1.,  1.], [ 0.,  0.,  1.], [ 1.,  1.,  0.,  1.])
     ([-0.5, -0.5,  0. ], [ 0.,  0.], [ 0.,  0.,  1.], [ 0.,  0.,  0.,  1.])
     ([ 0.5, -0.5,  0. ], [ 1.,  0.], [ 0.,  0.,  1.], [ 1.,  0.,  0.,  1.])]
    >>> print(faces)
    [[0 2 1]
     [2 3 1]]
    >>> print(outline)
    [[0 2]
     [2 3]
     [3 1]
     [1 0]]
    """

    function = PRIMITIVE_METHODS[method]

    return function(**filter_kwargs(function, **kwargs))
