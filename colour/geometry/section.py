"""
Geometry / Hull Section
=======================

Defines various objects to compute hull sections:

-   :func:`colour.geometry.hull_section`
"""

from __future__ import annotations

import numpy as np

from colour.algebra import linear_conversion
from colour.constants import DEFAULT_FLOAT_DTYPE
from colour.hints import (
    ArrayLike,
    Boolean,
    Floating,
    Integer,
    Literal,
    NDArray,
    Union,
)
from colour.utilities import (
    as_float_array,
    as_float_scalar,
    required,
    validate_method,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "edges_to_chord",
    "unique_vertices",
    "close_chord",
    "hull_section",
]


def edges_to_chord(edges: ArrayLike, index: Integer = 0) -> NDArray:
    """
    Convert given edges to a chord, starting at given index.

    Parameters
    ----------
    edges
        Edges to convert to a chord.
    index
        Index to start forming the chord at.

    Returns
    -------
    :class:`numpy.ndarray`
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

    edge_list = as_float_array(edges).tolist()

    edges_ordered = [edge_list.pop(index)]
    segment = np.array(edges_ordered[0][1])

    while len(edge_list) > 0:
        edges_array = np.array(edge_list)
        d_0 = np.linalg.norm(edges_array[:, 0, :] - segment, axis=1)
        d_1 = np.linalg.norm(edges_array[:, 1, :] - segment, axis=1)
        d_0_argmin, d_1_argmin = d_0.argmin(), d_1.argmin()

        if d_0[d_0_argmin] < d_1[d_1_argmin]:
            edges_ordered.append(edge_list.pop(d_0_argmin))
            segment = np.array(edges_ordered[-1][1])
        else:
            edges_ordered.append(edge_list.pop(d_1_argmin))
            segment = np.array(edges_ordered[-1][0])

    return as_float_array(edges_ordered).reshape([-1, segment.shape[-1]])


def close_chord(vertices: ArrayLike) -> NDArray:
    """
    Close the chord.

    Parameters
    ----------
    vertices
        Vertices of the chord to close.

    Returns
    -------
    :class:`numpy.ndarray`
        Closed chord.

    Examples
    --------
    >>> close_chord(np.array([[0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]))
    array([[ 0. ,  0.5,  0. ],
           [ 0. ,  0. ,  0.5],
           [ 0. ,  0.5,  0. ]])
    """

    vertices = as_float_array(vertices)

    return np.vstack([vertices, vertices[0]])


def unique_vertices(
    vertices: ArrayLike,
    decimals: Integer = np.finfo(DEFAULT_FLOAT_DTYPE).precision - 1,
) -> NDArray:
    """
    Return the unique vertices from given vertices.

    Parameters
    ----------
    vertices
        Vertices to return the unique vertices from.
    decimals
        Decimals used when rounding the vertices prior to comparison.

    Returns
    -------
    :class:`numpy.ndarray`
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

    unique, indexes = np.unique(
        vertices.round(decimals=decimals), axis=0, return_index=True
    )

    return unique[np.argsort(indexes)]


@required("trimesh")
def hull_section(
    hull: trimesh.Trimesh,  # type: ignore[name-defined]  # noqa
    axis: Union[Literal["+z", "+x", "+y"], str] = "+z",
    origin: Floating = 0.5,
    normalise: Boolean = False,
) -> NDArray:
    """
    Compute the hull section for given axis at given origin.

    Parameters
    ----------
    hull
        *Trimesh* hull.
    axis
        Axis the hull section will be normal to.
    origin
        Coordinate along ``axis`` at which to plot the hull section.
    normalise
        Whether to normalise ``axis`` to the extent of the hull along it.

    Returns
    -------
    :class:`numpy.ndarray`
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

    axis = validate_method(
        axis,
        ["+z", "+x", "+y"],
        '"{0}" axis is invalid, it must be one of {1}!',
    )

    if axis == "+x":
        normal, plane = np.array([1, 0, 0]), np.array([origin, 0, 0])
    elif axis == "+y":
        normal, plane = np.array([0, 1, 0]), np.array([0, origin, 0])
    elif axis == "+z":
        normal, plane = np.array([0, 0, 1]), np.array([0, 0, origin])

    if normalise:
        vertices = hull.vertices * normal
        origin = as_float_scalar(
            linear_conversion(
                origin, [0, 1], [np.min(vertices), np.max(vertices)]
            )
        )
        plane[plane != 0] = origin

    section = trimesh.intersections.mesh_plane(hull, normal, plane)
    if len(section) == 0:
        raise ValueError(
            f'No section exists on "{axis}" axis at {origin} origin!'
        )
    section = close_chord(unique_vertices(edges_to_chord(section)))

    return section
