"""
Geometry / Hull Section
=======================

Define objects for computing hull sections in colour spaces.

This module provides functionality to compute and analyze hull sections,
which represent the boundary surfaces of colour gamuts when intersected
with the specified planes in various colour spaces.

Key Components
--------------

-   :func:`colour.geometry.hull_section`: Compute hull sections for colour
    space analysis.
"""

from __future__ import annotations

import typing

import numpy as np

from colour.algebra import linear_conversion

if typing.TYPE_CHECKING:
    from colour.hints import ArrayLike, Literal, NDArrayFloat

from colour.hints import List, cast
from colour.utilities import (
    array_namespace,
    as_float_array,
    as_float_scalar,
    required,
    validate_method,
    xp_as_float_array,
    xp_reshape,
    xp_round,
    xp_unique,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "edges_to_chord",
    "unique_vertices",
    "close_chord",
    "hull_section",
]


def edges_to_chord(edges: ArrayLike, index: int = 0) -> NDArrayFloat:
    """
    Convert specified edges to a chord, starting at specified index.

    Transforms a collection of edges into a continuous chord by
    connecting them sequentially, beginning from the specified index
    position.

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
    >>> edges = np.array(
    ...     [
    ...         [[-0.0, -0.5, 0.0], [0.5, -0.5, 0.0]],
    ...         [[-0.5, -0.5, 0.0], [-0.0, -0.5, 0.0]],
    ...         [[0.5, 0.5, 0.0], [-0.0, 0.5, 0.0]],
    ...         [[-0.0, 0.5, 0.0], [-0.5, 0.5, 0.0]],
    ...         [[-0.5, 0.0, -0.0], [-0.5, -0.5, -0.0]],
    ...         [[-0.5, 0.5, -0.0], [-0.5, 0.0, -0.0]],
    ...         [[0.5, -0.5, -0.0], [0.5, 0.0, -0.0]],
    ...         [[0.5, 0.0, -0.0], [0.5, 0.5, -0.0]],
    ...     ]
    ... )
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

    xp = array_namespace(edges)

    edge_list = cast("List[List[float]]", edges.tolist())

    edges_ordered = [edge_list.pop(index)]
    segment = xp_as_float_array(edges_ordered[0][1], xp=xp)

    while len(edge_list) > 0:
        edges = xp_as_float_array(edge_list, xp=xp)
        d_0 = xp.linalg.vector_norm(edges[:, 0, :] - segment, axis=1)
        d_1 = xp.linalg.vector_norm(edges[:, 1, :] - segment, axis=1)
        d_0_argmin, d_1_argmin = int(xp.argmin(d_0)), int(xp.argmin(d_1))

        if d_0[d_0_argmin] < d_1[d_1_argmin]:
            edges_ordered.append(edge_list.pop(d_0_argmin))
            segment = xp_as_float_array(edges_ordered[-1][1], xp=xp)
        else:
            edges_ordered.append(edge_list.pop(d_1_argmin))
            segment = xp_as_float_array(edges_ordered[-1][0], xp=xp)

    return xp_reshape(
        xp_as_float_array(edges_ordered, xp=xp, like=segment),
        (-1, segment.shape[-1]),
        xp=xp,
    )


def close_chord(vertices: ArrayLike) -> NDArrayFloat:
    """
    Close a chord by appending its first vertex to the end.

    Parameters
    ----------
    vertices
        Vertices of the chord to close.

    Returns
    -------
    :class:`numpy.ndarray`
        Closed chord with the first vertex appended to create a closed
        path.

    Examples
    --------
    >>> close_chord(np.array([[0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]))
    array([[0. , 0.5, 0. ],
           [0. , 0. , 0.5],
           [0. , 0.5, 0. ]])
    """

    vertices = as_float_array(vertices)

    xp = array_namespace(vertices)

    return xp.concat([vertices, vertices[0:1]], axis=0)


def unique_vertices(
    vertices: ArrayLike,
    decimals: int = int(np.finfo(np.float64).precision - 1),
) -> NDArrayFloat:
    """
    Return the unique vertices from the specified vertices after rounding.

    Parameters
    ----------
    vertices
        Vertices to return the unique vertices from.
    decimals
        Number of decimal places for rounding the vertices prior to
        uniqueness comparison.

    Returns
    -------
    :class:`numpy.ndarray`
        Unique vertices with duplicates removed.

    Notes
    -----
    -   The vertices are rounded to the specified number of decimal places
        before uniqueness comparison to handle floating-point precision
        issues.

    Examples
    --------
    >>> unique_vertices(np.array([[0.0, 0.5, 0.0], [0.0, 0.0, 0.5], [0.0, 0.5, 0.0]]))
    array([[0. , 0.5, 0. ],
           [0. , 0. , 0.5]])
    """

    vertices = as_float_array(vertices)

    xp = array_namespace(vertices)

    unique, indexes = xp_unique(
        xp_round(vertices, decimals=decimals, xp=xp), axis=0, return_index=True, xp=xp
    )

    return unique[xp.argsort(indexes)]


@required("trimesh")
def hull_section(
    hull: trimesh.Trimesh,  # pyright: ignore  # noqa: F821
    axis: Literal["+z", "+x", "+y"] | str = "+z",
    origin: float = 0.5,
    normalise: bool = False,
) -> NDArrayFloat:
    """
    Compute the hull section for the specified axis at the specified origin.

    Generate a cross-sectional contour of a 3D hull by intersecting it with
    a plane perpendicular to the specified axis at the specified origin
    coordinate. This operation produces vertices that define the boundary of
    the hull's intersection with the cutting plane.

    Parameters
    ----------
    hull
        *Trimesh* hull object representing the 3D geometry to section.
    axis
        Axis perpendicular to which the hull section will be computed.
        Options are "+x", "+y", or "+z".
    origin
        Coordinate along ``axis`` at which to compute the hull section.
        The value represents either an absolute position or a normalised
        position depending on the ``normalise`` parameter.
    normalise
        Whether to normalise the ``origin`` coordinate to the extent of the
        hull along the specified ``axis``. When ``True``, ``origin`` is
        interpreted as a value in [0, 1] where 0 represents the minimum
        extent and 1 represents the maximum extent along ``axis``.

    Returns
    -------
    :class:`numpy.ndarray`
        Hull section vertices forming a closed contour. The vertices are
        ordered to form a continuous path around the section boundary.

    Raises
    ------
    ValueError
        If no section exists on the specified axis at the specified origin,
        typically when the cutting plane does not intersect the hull.

    Examples
    --------
    >>> from colour.geometry import primitive_cube
    >>> from colour.utilities import is_trimesh_installed
    >>> vertices, faces, outline = primitive_cube(1, 1, 1, 2, 2, 2)
    >>> if is_trimesh_installed:
    ...     import trimesh
    ...
    ...     hull = trimesh.Trimesh(vertices["position"], faces, process=False)
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

    import trimesh.intersections  # noqa: PLC0415

    axis = validate_method(
        axis,
        ("+z", "+x", "+y"),
        '"{0}" axis is invalid, it must be one of {1}!',
    )

    if axis == "+x":
        normal, plane = as_float_array([1, 0, 0]), as_float_array([origin, 0, 0])
    elif axis == "+y":
        normal, plane = as_float_array([0, 1, 0]), as_float_array([0, origin, 0])
    elif axis == "+z":
        normal, plane = as_float_array([0, 0, 1]), as_float_array([0, 0, origin])

    xp = array_namespace(normal)

    if normalise:
        vertices = hull.vertices * normal
        origin = as_float_scalar(
            linear_conversion(origin, [0, 1], [xp.min(vertices), xp.max(vertices)])
        )
        plane = xp.where(plane != 0, origin, plane)

    section = trimesh.intersections.mesh_plane(hull, normal, plane)
    if len(section) == 0:
        error = f'No section exists on "{axis}" axis at {origin} origin!'

        raise ValueError(error)

    return close_chord(unique_vertices(edges_to_chord(section)))
