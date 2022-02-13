"""
Geometry Primitives
===================

Defines various geometry primitives and their generation methods:

-   :attr:`colour.geometry.MAPPING_PLANE_TO_AXIS`
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

from __future__ import annotations

import numpy as np

from colour.constants import DEFAULT_INT_DTYPE, DEFAULT_FLOAT_DTYPE
from colour.hints import (
    Any,
    DTypeFloating,
    DTypeInteger,
    Floating,
    Integer,
    Literal,
    NDArray,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
)
from colour.utilities import (
    CaseInsensitiveMapping,
    as_int_array,
    filter_kwargs,
    ones,
    optional,
    zeros,
    validate_method,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "MAPPING_PLANE_TO_AXIS",
    "primitive_grid",
    "primitive_cube",
    "PRIMITIVE_METHODS",
    "primitive",
]

MAPPING_PLANE_TO_AXIS: CaseInsensitiveMapping = CaseInsensitiveMapping(
    {
        "yz": "+x",
        "zy": "-x",
        "xz": "+y",
        "zx": "-y",
        "xy": "+z",
        "yx": "-z",
    }
)
MAPPING_PLANE_TO_AXIS.__doc__ = """Plane to axis mapping."""


def primitive_grid(
    width: Floating = 1,
    height: Floating = 1,
    width_segments: Integer = 1,
    height_segments: Integer = 1,
    axis: Literal[
        "-x", "+x", "-y", "+y", "-z", "+z", "xy", "xz", "yz", "yx", "zx", "zy"
    ] = "+z",
    dtype_vertices: Optional[Type[DTypeFloating]] = None,
    dtype_indexes: Optional[Type[DTypeInteger]] = None,
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Generate vertices and indexes for a filled and outlined grid primitive.

    Parameters
    ----------
    width
        Grid width.
    height
        Grid height.
    width_segments
        Grid segments count along the width.
    height_segments
        Grid segments count along the height.
    axis
        Axis the primitive will be normal to, or plane the primitive will be
        co-planar with.
    dtype_vertices
        :class:`numpy.dtype` to use for the grid vertices, default to
        the :class:`numpy.dtype` defined by the
        :attr:`colour.constant.DEFAULT_FLOAT_DTYPE` attribute.
    dtype_indexes
        :class:`numpy.dtype` to use for the grid indexes, default to
        the :class:`numpy.dtype` defined by the
        :attr:`colour.constant.DEFAULT_INT_DTYPE` attribute.

    Returns
    -------
    :class:`tuple`
        Tuple of grid vertices, face indexes to produce a filled grid and
        outline indexes to produce an outline of the faces of the grid.

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

    axis = MAPPING_PLANE_TO_AXIS.get(axis, axis).lower()

    dtype_vertices = cast(
        Type[DTypeFloating], optional(dtype_vertices, DEFAULT_FLOAT_DTYPE)
    )
    dtype_indexes = cast(
        Type[DTypeInteger], optional(dtype_indexes, DEFAULT_INT_DTYPE)
    )

    x_grid = width_segments
    y_grid = height_segments

    x_grid1 = int(x_grid + 1)
    y_grid1 = int(y_grid + 1)

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
    faces_indexes = []
    outline_indexes = []
    for i_y in range(y_grid):
        for i_x in range(x_grid):
            a = i_x + x_grid1 * i_y
            b = i_x + x_grid1 * (i_y + 1)
            c = (i_x + 1) + x_grid1 * (i_y + 1)
            d = (i_x + 1) + x_grid1 * i_y

            faces_indexes.extend([(a, b, d), (b, c, d)])
            outline_indexes.extend([(a, b), (b, c), (c, d), (d, a)])

    faces = np.reshape(as_int_array(faces_indexes, dtype_indexes), (-1, 3))
    outline = np.reshape(as_int_array(outline_indexes, dtype_indexes), (-1, 2))

    positions = np.reshape(positions, (-1, 3))
    uvs = np.reshape(uvs, (-1, 2))
    normals = np.reshape(normals, (-1, 3))

    if axis in ("-x", "+x"):
        shift, zero_axis = 1, 0
    elif axis in ("-y", "+y"):
        shift, zero_axis = -1, 1
    elif axis in ("-z", "+z"):
        shift, zero_axis = 0, 2

    sign = -1 if "-" in axis else 1

    positions = np.roll(positions, shift, -1)
    normals = np.roll(normals, shift, -1) * sign
    vertex_colours = np.ravel(positions)
    vertex_colours = np.hstack(
        [
            np.reshape(
                np.interp(
                    vertex_colours,
                    (np.min(vertex_colours), np.max(vertex_colours)),
                    (0, 1),
                ),
                positions.shape,
            ),
            ones((positions.shape[0], 1)),
        ]
    )
    vertex_colours[..., zero_axis] = 0

    vertices = zeros(
        positions.shape[0],
        [
            ("position", dtype_vertices, 3),
            ("uv", dtype_vertices, 2),
            ("normal", dtype_vertices, 3),
            ("colour", dtype_vertices, 4),
        ],  # type: ignore[arg-type]
    )

    vertices["position"] = positions
    vertices["uv"] = uvs
    vertices["normal"] = normals
    vertices["colour"] = vertex_colours

    return vertices, faces, outline


def primitive_cube(
    width: Floating = 1,
    height: Floating = 1,
    depth: Floating = 1,
    width_segments: Integer = 1,
    height_segments: Integer = 1,
    depth_segments: Integer = 1,
    planes: Optional[
        Literal[
            "-x",
            "+x",
            "-y",
            "+y",
            "-z",
            "+z",
            "xy",
            "xz",
            "yz",
            "yx",
            "zx",
            "zy",
        ]
    ] = None,
    dtype_vertices: Optional[Type[DTypeFloating]] = None,
    dtype_indexes: Optional[Type[DTypeInteger]] = None,
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Generate vertices and indexes for a filled and outlined cube primitive.

    Parameters
    ----------
    width
        Cube width.
    height
        Cube height.
    depth
        Cube depth.
    width_segments
        Cube segments count along the width.
    height_segments
        Cube segments count along the height.
    depth_segments
        Cube segments count along the depth.
    planes
        Grid primitives to include in the cube construction.
    dtype_vertices
        :class:`numpy.dtype` to use for the grid vertices, default to
        the :class:`numpy.dtype` defined by the
        :attr:`colour.constant.DEFAULT_FLOAT_DTYPE` attribute.
    dtype_indexes
        :class:`numpy.dtype` to use for the grid indexes, default to
        the :class:`numpy.dtype` defined by the
        :attr:`colour.constant.DEFAULT_INT_DTYPE` attribute.

    Returns
    -------
    :class:`tuple`
        Tuple of cube vertices, face indexes to produce a filled cube and
        outline indexes to produce an outline of the faces of the cube.

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

    axis = (
        sorted(list(MAPPING_PLANE_TO_AXIS.values()))
        if planes is None
        else [
            MAPPING_PLANE_TO_AXIS.get(plane, plane).lower() for plane in planes
        ]
    )

    dtype_vertices = cast(
        Type[DTypeFloating], optional(dtype_vertices, DEFAULT_FLOAT_DTYPE)
    )
    dtype_indexes = cast(
        Type[DTypeInteger], optional(dtype_indexes, DEFAULT_INT_DTYPE)
    )

    w_s, h_s, d_s = width_segments, height_segments, depth_segments

    planes_p = []
    if "-z" in axis:
        planes_p.append(list(primitive_grid(width, depth, w_s, d_s, "-z")))
        planes_p[-1][0]["position"][..., 2] -= height / 2
        planes_p[-1][1] = np.fliplr(planes_p[-1][1])
    if "+z" in axis:
        planes_p.append(list(primitive_grid(width, depth, w_s, d_s, "+z")))
        planes_p[-1][0]["position"][..., 2] += height / 2

    if "-y" in axis:
        planes_p.append(list(primitive_grid(height, width, h_s, w_s, "-y")))
        planes_p[-1][0]["position"][..., 1] -= depth / 2
        planes_p[-1][1] = np.fliplr(planes_p[-1][1])
    if "+y" in axis:
        planes_p.append(list(primitive_grid(height, width, h_s, w_s, "+y")))
        planes_p[-1][0]["position"][..., 1] += depth / 2

    if "-x" in axis:
        planes_p.append(list(primitive_grid(depth, height, d_s, h_s, "-x")))
        planes_p[-1][0]["position"][..., 0] -= width / 2
        planes_p[-1][1] = np.fliplr(planes_p[-1][1])
    if "+x" in axis:
        planes_p.append(list(primitive_grid(depth, height, d_s, h_s, "+x")))
        planes_p[-1][0]["position"][..., 0] += width / 2

    positions = zeros((0, 3))
    uvs = zeros((0, 2))
    normals = zeros((0, 3))

    faces = zeros((0, 3), dtype=dtype_indexes)
    outline = zeros((0, 2), dtype=dtype_indexes)

    offset = 0
    for vertices_p, faces_p, outline_p in planes_p:
        positions = np.vstack([positions, vertices_p["position"]])
        uvs = np.vstack([uvs, vertices_p["uv"]])
        normals = np.vstack([normals, vertices_p["normal"]])

        faces = np.vstack([faces, faces_p + offset])
        outline = np.vstack([outline, outline_p + offset])
        offset += vertices_p["position"].shape[0]

    vertices = zeros(
        positions.shape[0],
        [
            ("position", dtype_vertices, 3),
            ("uv", dtype_vertices, 2),
            ("normal", dtype_vertices, 3),
            ("colour", dtype_vertices, 4),
        ],  # type: ignore[arg-type]
    )

    vertex_colours = np.ravel(positions)
    vertex_colours = np.hstack(
        [
            np.reshape(
                np.interp(
                    vertex_colours,
                    (np.min(vertex_colours), np.max(vertex_colours)),
                    (0, 1),
                ),
                positions.shape,
            ),
            ones((positions.shape[0], 1)),
        ]
    )

    vertices["position"] = positions
    vertices["uv"] = uvs
    vertices["normal"] = normals
    vertices["colour"] = vertex_colours

    return vertices, faces, outline


PRIMITIVE_METHODS: CaseInsensitiveMapping = CaseInsensitiveMapping(
    {
        "Grid": primitive_grid,
        "Cube": primitive_cube,
    }
)
PRIMITIVE_METHODS.__doc__ = """
Supported geometry primitive generation methods.
"""


def primitive(
    method: Union[Literal["Cube", "Grid"], str] = "Cube", **kwargs: Any
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Return a geometry primitive using given method.

    Parameters
    ----------
    method
        Generation method.

    Other Parameters
    ----------------
    axis
        {:func:`colour.geometry.primitive_grid`},
        Axis the primitive will be normal to, or plane the primitive will be
        co-planar with.
    depth
        {:func:`colour.geometry.primitive_grid`,
        :func:`colour.geometry.primitive_cube`},
        Primitive depth.
    depth_segments
        {:func:`colour.geometry.primitive_grid`,
        :func:`colour.geometry.primitive_cube`},
        Primitive segments count along the depth.
    dtype_indexes
        {:func:`colour.geometry.primitive_grid`,
        :func:`colour.geometry.primitive_cube`},
        :class:`numpy.dtype` to use for the grid indexes, default to
        the :class:`numpy.dtype` defined by the
        :attr:`colour.constant.DEFAULT_INT_DTYPE` attribute.
    dtype_vertices
        {:func:`colour.geometry.primitive_grid`,
        :func:`colour.geometry.primitive_cube`},
        :class:`numpy.dtype` to use for the grid vertices, default to
        the :class:`numpy.dtype` defined by the
        :attr:`colour.constant.DEFAULT_FLOAT_DTYPE` attribute.
    height
        {:func:`colour.geometry.primitive_grid`,
        :func:`colour.geometry.primitive_cube`},
        Primitive height.
    planes
        {:func:`colour.geometry.primitive_cube`},
        Included grid primitives in the cube construction.
    width
        {:func:`colour.geometry.primitive_grid`,
        :func:`colour.geometry.primitive_cube`},
        Primitive width.
    width_segments
        {:func:`colour.geometry.primitive_grid`,
        :func:`colour.geometry.primitive_cube`},
        Primitive segments count along the width.
    height_segments
        {:func:`colour.geometry.primitive_grid`,
        :func:`colour.geometry.primitive_cube`},
        Primitive segments count along the height.

    Returns
    -------
    :class:`tuple`
        Tuple of primitive vertices, face indexes to produce a filled primitive
        and outline indexes to produce an outline of the faces of the
        primitive.

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

    method = validate_method(method, PRIMITIVE_METHODS)

    function = PRIMITIVE_METHODS[method]

    return function(**filter_kwargs(function, **kwargs))
