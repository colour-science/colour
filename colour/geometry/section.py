# -*- coding: utf-8 -*-
"""
Geometry / Hull Section
=======================

Defines various objects to compute hull sections:

-   :func:`colour.geometry.hull_section`
"""

import numpy as np

from colour.algebra import linear_conversion
from colour.constants import DEFAULT_FLOAT_DTYPE
from colour.utilities import as_float_array, required

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['edges_to_chord', 'unique_vertices', 'close_chord', 'hull_section']


def edges_to_chord(edges, index=0):
    """
    Converts given edges to a chord, starting at given index.

    Parameters
    ----------
    edges : array_like
        Edges to convert to a chord.
    index : int, optional
        Index to start forming the chord at.

    Returns
    -------
    ndarray
        Chord.

    Examples
    --------
    >>> edges = np.array([
    ...     [[-0.0, -0.5, 0.0], [0.5, -0.5, 0.0]],
    ...     [[-0.5, -0.5, 0.0], [-0.0, -0.5, 0.0]],
    ...     [[0.5, 0.5, 0.0], [-0.0, 0.5, 0.0]],
    ...     [[-0.0, 0.5, 0.0], [-0.5, 0.5, 0.0]],
    ...     [[-0.5, 0.0, -0.0], [-0.5, -0.5, -0.0]],
    ...     [[-0.5, 0.5, -0.0], [-0.5, 0.0, -0.0]],
    ...     [[0.5, -0.5, -0.0], [0.5, 0.0, -0.0]],
    ...     [[0.5, 0.0, -0.0], [0.5, 0.5, -0.0]],
    ... ])
    >>> edges_to_chord(edges)
    array([[-0. , -0.5,  0. ],
           [ 0.5, -0.5,  0. ],
           [ 0.5, -0.5, -0. ],
           [ 0.5,  0. , -0. ],
           [ 0.5,  0. , -0. ],
           [ 0.5,  0.5, -0. ],
           [ 0.5,  0.5,  0. ],
           [-0. ,  0.5,  0. ],
           [-0. ,  0.5,  0. ],
           [-0.5,  0.5,  0. ],
           [-0.5,  0.5, -0. ],
           [-0.5,  0. , -0. ],
           [-0.5,  0. , -0. ],
           [-0.5, -0.5, -0. ],
           [-0.5, -0.5,  0. ],
           [-0. , -0.5,  0. ]])
    """

    edges = as_float_array(edges)
    edges = edges.tolist()

    ordered_edges = [edges.pop(index)]
    segment = np.array(ordered_edges[0][1])

    while len(edges) > 0:
        array_edges = np.array(edges)
        d_0 = np.linalg.norm(array_edges[:, 0, :] - segment, axis=1)
        d_1 = np.linalg.norm(array_edges[:, 1, :] - segment, axis=1)
        d_0_argmin, d_1_argmin = d_0.argmin(), d_1.argmin()

        if d_0[d_0_argmin] < d_1[d_1_argmin]:
            ordered_edges.append(edges.pop(d_0_argmin))
            segment = np.array(ordered_edges[-1][1])
        else:
            ordered_edges.append(edges.pop(d_1_argmin))
            segment = np.array(ordered_edges[-1][0])

    return as_float_array(ordered_edges).reshape([-1, segment.shape[-1]])


def close_chord(vertices):
    """
    Closes the chord.

    Parameters
    ----------
    vertices : array_like
        Vertices of the chord to close.

    Returns
    -------
    ndarray
        Closed chord.

    Examples
    --------
    >>> close_chord(np.array([[0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]))
    array([[ 0. ,  0.5,  0. ],
           [ 0. ,  0. ,  0.5],
           [ 0. ,  0.5,  0. ]])
    """

    return np.vstack([vertices, vertices[0]])


def unique_vertices(vertices,
                    decimals=np.finfo(DEFAULT_FLOAT_DTYPE).precision - 1):
    """
    Returns the unique vertices from given vertices.

    Parameters
    ----------
    vertices : array_like
        Vertices to return the unique vertices from.
    decimals : int, optional
        Decimals used when rounding the vertices prior to comparison.

    Returns
    -------
    ndarray
        Unique vertices.

    Notes
    -----
    -   The vertices are rounded at given ``decimals``.

    Examples
    --------
    >>> unique_vertices(
    ...     np.array([[0.0, 0.5, 0.0], [0.0, 0.0, 0.5], [0.0, 0.5, 0.0]]))
    array([[ 0. ,  0.5,  0. ],
           [ 0. ,  0. ,  0.5]])
    """

    vertices = as_float_array(vertices)

    vertices, indexes = np.unique(
        vertices.round(decimals=decimals), axis=0, return_index=True)

    return vertices[np.argsort(indexes)]


@required('trimesh')
def hull_section(hull, axis='+z', origin=0.5, normalise=False):
    """
    Computes the hull section for given axis at given origin.

    Parameters
    ----------
    hull : Trimesh
        *Trimesh* hull.
    axis : unicode, optional
        **{'+z', '+x', '+y'}**,
        Axis the hull section will be normal to.
    origin : numeric, optional
        Coordinate along ``axis`` at which to plot the hull section.
    normalise : bool, optional
        Whether to normalise ``axis`` to the extent of the hull along it.

    Returns
    -------
    ndarray
        Hull section vertices.

    Examples
    --------
    >>> from colour.geometry import primitive_cube
    >>> from colour.utilities import is_trimesh_installed
    >>> vertices, faces, outline = primitive_cube(1, 1, 1, 2, 2, 2)
    >>> if is_trimesh_installed:
    ...     import trimesh
    ...     hull = trimesh.Trimesh(vertices['position'], faces, process=False)
    ...     hull_section(hull, origin=0)
    array([[-0. , -0.5,  0. ],
           [ 0.5, -0.5,  0. ],
           [ 0.5,  0. , -0. ],
           [ 0.5,  0.5, -0. ],
           [-0. ,  0.5,  0. ],
           [-0.5,  0.5,  0. ],
           [-0.5,  0. , -0. ],
           [-0.5, -0.5, -0. ],
           [-0. , -0.5,  0. ]])
    """

    import trimesh

    if axis == '+x':
        normal, plane = np.array([1, 0, 0]), np.array([origin, 0, 0])
    elif axis == '+y':
        normal, plane = np.array([0, 1, 0]), np.array([0, origin, 0])
    elif axis == '+z':
        normal, plane = np.array([0, 0, 1]), np.array([0, 0, origin])

    if normalise:
        vertices = hull.vertices * normal
        origin = linear_conversion(
            origin, [0, 1],
            [np.min(vertices), np.max(vertices)])
        plane[plane != 0] = origin

    section = trimesh.intersections.mesh_plane(hull, normal, plane)
    if len(section) == 0:
        raise ValueError(
            'No section exists on "{0}" axis at {1} origin!'.format(
                axis, origin))
    section = close_chord(unique_vertices(edges_to_chord(section)))

    return section
