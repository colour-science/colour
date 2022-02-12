"""
Geometry Primitive Vertices
===========================

Defines various geometry primitive vertices generation methods:

-   :func:`colour.geometry.primitive_vertices_quad_mpl`
-   :func:`colour.geometry.primitive_vertices_grid_mpl`
-   :func:`colour.geometry.primitive_vertices_cube_mpl`
-   :func:`colour.geometry.primitive_vertices_sphere`
-   :func:`colour.PRIMITIVE_VERTICES_METHODS`
-   :func:`colour.primitive_vertices`
"""

from __future__ import annotations

import numpy as np

from colour.algebra import spherical_to_cartesian
from colour.geometry import MAPPING_PLANE_TO_AXIS
from colour.hints import (
    Any,
    ArrayLike,
    Boolean,
    Floating,
    Integer,
    List,
    Literal,
    NDArray,
    Optional,
    Union,
)
from colour.utilities import (
    CaseInsensitiveMapping,
    as_float_array,
    filter_kwargs,
    full,
    ones,
    tsplit,
    tstack,
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
    "primitive_vertices_quad_mpl",
    "primitive_vertices_grid_mpl",
    "primitive_vertices_cube_mpl",
    "primitive_vertices_sphere",
    "PRIMITIVE_VERTICES_METHODS",
    "primitive_vertices",
]


def primitive_vertices_quad_mpl(
    width: Floating = 1,
    height: Floating = 1,
    depth: Floating = 0,
    origin: ArrayLike = np.array([0, 0]),
    axis: Union[Literal["+z", "+x", "+y", "yz", "xz", "xy"], str] = "+z",
) -> NDArray:
    """
    Return the vertices of a quad primitive for use with *Matplotlib*
    :class:`mpl_toolkits.mplot3d.art3d.Poly3DCollection` class.

    Parameters
    ----------
    width
        Quad width.
    height
        Quad height.
    depth
        Quad depth.
    origin
        Quad origin on the construction plane.
    axis
        Axis the quad will be normal to, or plane the quad will be co-planar
        with.

    Returns
    -------
    :class:`numpy.ndarray`
        Quad primitive vertices.

    Examples
    --------
    >>> primitive_vertices_quad_mpl()
    array([[ 0.,  0.,  0.],
           [ 1.,  0.,  0.],
           [ 1.,  1.,  0.],
           [ 0.,  1.,  0.]])
    """

    axis = MAPPING_PLANE_TO_AXIS.get(axis, axis).lower()
    axis = validate_method(
        axis, ["+x", "+y", "+z"], '"{0}" axis invalid, it must be one of {1}!'
    )

    u, v = tsplit(origin)

    if axis == "+z":
        vertices = [
            (u, v, depth),
            (u + width, v, depth),
            (u + width, v + height, depth),
            (u, v + height, depth),
        ]
    elif axis == "+y":
        vertices = [
            (u, depth, v),
            (u + width, depth, v),
            (u + width, depth, v + height),
            (u, depth, v + height),
        ]
    elif axis == "+x":
        vertices = [
            (depth, u, v),
            (depth, u + width, v),
            (depth, u + width, v + height),
            (depth, u, v + height),
        ]

    return as_float_array(vertices)


def primitive_vertices_grid_mpl(
    width: Floating = 1,
    height: Floating = 1,
    depth: Floating = 0,
    width_segments: Integer = 1,
    height_segments: Integer = 1,
    origin: ArrayLike = np.array([0, 0]),
    axis: Union[Literal["+z", "+x", "+y", "yz", "xz", "xy"], str] = "+z",
) -> NDArray:
    """
    Return the vertices of a grid primitive made of quad primitives for use
    with *Matplotlib* :class:`mpl_toolkits.mplot3d.art3d.Poly3DCollection`
    class.

    Parameters
    ----------
    width
        Grid width.
    height
        Grid height.
    depth
        Grid depth.
    width_segments:
        Grid width segments, quad primitive counts along the width.
    height_segments:
        Grid height segments, quad primitive counts along the height.
    origin
        Grid origin on the construction plane.
    axis
        Axis the grid will be normal to, or plane the grid will be co-planar
        with.

    Returns
    -------
    :class:`numpy.ndarray`
        Grid primitive vertices.

    Examples
    --------
    >>> primitive_vertices_grid_mpl(width_segments=2, height_segments=2)
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

    u, v = tsplit(origin)

    w_x, h_y = width / width_segments, height / height_segments

    quads = []
    for i in range(width_segments):
        for j in range(height_segments):
            quads.append(
                primitive_vertices_quad_mpl(
                    w_x, h_y, depth, (i * w_x + u, j * h_y + v), axis
                )
            )

    return as_float_array(quads)


def primitive_vertices_cube_mpl(
    width: Floating = 1,
    height: Floating = 1,
    depth: Floating = 1,
    width_segments: Integer = 1,
    height_segments: Integer = 1,
    depth_segments: Integer = 1,
    origin: ArrayLike = np.array([0, 0, 0]),
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
) -> NDArray:
    """
    Return the vertices of a cube primitive made of grid primitives for use
    with *Matplotlib* :class:`mpl_toolkits.mplot3d.art3d.Poly3DCollection`
    class.

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
    origin
        Cube origin.
    planes
        Grid primitives to include in the cube construction.

    Returns
    -------
    :class:`numpy.ndarray`
        Cube primitive vertices.

    Examples
    --------
    >>> primitive_vertices_cube_mpl()
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

    axis = (
        sorted(list(MAPPING_PLANE_TO_AXIS.values()))
        if planes is None
        else [
            MAPPING_PLANE_TO_AXIS.get(plane, plane).lower() for plane in planes
        ]
    )

    u, v, w = tsplit(origin)

    w_s, h_s, d_s = width_segments, height_segments, depth_segments

    grids: List = []
    if "-z" in axis:
        grids.extend(
            primitive_vertices_grid_mpl(
                width, depth, v, w_s, d_s, (u, w), "+z"
            )
        )
    if "+z" in axis:
        grids.extend(
            primitive_vertices_grid_mpl(
                width, depth, v + height, w_s, d_s, (u, w), "+z"
            )
        )

    if "-y" in axis:
        grids.extend(
            primitive_vertices_grid_mpl(
                width, height, w, w_s, h_s, (u, v), "+y"
            )
        )
    if "+y" in axis:
        grids.extend(
            primitive_vertices_grid_mpl(
                width, height, w + depth, w_s, h_s, (u, v), "+y"
            )
        )

    if "-x" in axis:
        grids.extend(
            primitive_vertices_grid_mpl(
                depth, height, u, d_s, h_s, (w, v), "+x"
            )
        )
    if "+x" in axis:
        grids.extend(
            primitive_vertices_grid_mpl(
                depth, height, u + width, d_s, h_s, (w, v), "+x"
            )
        )

    return as_float_array(grids)


def primitive_vertices_sphere(
    radius: Floating = 0.5,
    segments: Integer = 8,
    intermediate: Boolean = False,
    origin: ArrayLike = np.array([0, 0, 0]),
    axis: Union[Literal["+z", "+x", "+y", "yz", "xz", "xy"], str] = "+z",
) -> NDArray:
    """
    Return the vertices of a latitude-longitude sphere primitive.

    Parameters
    ----------
    radius
        Sphere radius.
    segments
        Latitude-longitude segments, if the ``intermediate`` argument is
        *True*, then the sphere will have one less segment along its longitude.
    intermediate
        Whether to generate the sphere vertices at the center of the faces
        outlined by the segments of a regular sphere generated without
        the ``intermediate`` argument set to *True*. The resulting sphere is
        inscribed on the regular sphere faces but possesses the same poles.
    origin
        Sphere origin on the construction plane.
    axis
        Axis (or normal of the plane) the poles of the sphere will be aligned
        with.

    Returns
    -------
    :class:`numpy.ndarray`
        Sphere primitive vertices.

    Notes
    -----
    -   The sphere poles have latitude segments count - 1 co-located vertices.

    Examples
    --------
    >>> primitive_vertices_sphere(segments=4)  # doctest: +ELLIPSIS
    array([[[  0.0000000...e+00,   0.0000000...e+00,   5.0000000...e-01],
            [ -3.5355339...e-01,  -4.3297802...e-17,   3.5355339...e-01],
            [ -5.0000000...e-01,  -6.1232340...e-17,   3.0616170...e-17],
            [ -3.5355339...e-01,  -4.3297802...e-17,  -3.5355339...e-01],
            [ -6.1232340...e-17,  -7.4987989...e-33,  -5.0000000...e-01]],
    <BLANKLINE>
           [[  0.0000000...e+00,   0.0000000...e+00,   5.0000000...e-01],
            [  2.1648901...e-17,  -3.5355339...e-01,   3.5355339...e-01],
            [  3.0616170...e-17,  -5.0000000...e-01,   3.0616170...e-17],
            [  2.1648901...e-17,  -3.5355339...e-01,  -3.5355339...e-01],
            [  3.7493994...e-33,  -6.1232340...e-17,  -5.0000000...e-01]],
    <BLANKLINE>
           [[  0.0000000...e+00,   0.0000000...e+00,   5.0000000...e-01],
            [  3.5355339...e-01,   0.0000000...e+00,   3.5355339...e-01],
            [  5.0000000...e-01,   0.0000000...e+00,   3.0616170...e-17],
            [  3.5355339...e-01,   0.0000000...e+00,  -3.5355339...e-01],
            [  6.1232340...e-17,   0.0000000...e+00,  -5.0000000...e-01]],
    <BLANKLINE>
           [[  0.0000000...e+00,   0.0000000...e+00,   5.0000000...e-01],
            [  2.1648901...e-17,   3.5355339...e-01,   3.5355339...e-01],
            [  3.0616170...e-17,   5.0000000...e-01,   3.0616170...e-17],
            [  2.1648901...e-17,   3.5355339...e-01,  -3.5355339...e-01],
            [  3.7493994...e-33,   6.1232340...e-17,  -5.0000000...e-01]]])
    """

    axis = MAPPING_PLANE_TO_AXIS.get(axis, axis).lower()
    axis = validate_method(
        axis, ["+x", "+y", "+z"], '"{0}" axis invalid, it must be one of {1}!'
    )

    if not intermediate:
        theta = np.tile(
            np.radians(np.linspace(0, 180, segments + 1)),
            (int(segments) + 1, 1),
        )
        phi = np.transpose(
            np.tile(
                np.radians(np.linspace(-180, 180, segments + 1)),
                (int(segments) + 1, 1),
            )
        )
    else:
        theta = np.tile(
            np.radians(np.linspace(0, 180, segments * 2 + 1)[1::2][1:-1]),
            (int(segments) + 1, 1),
        )
        theta = np.hstack(
            [
                zeros((segments + 1, 1)),
                theta,
                full((segments + 1, 1), np.pi),
            ]
        )
        phi = np.transpose(
            np.tile(
                np.radians(np.linspace(-180, 180, segments + 1))
                + np.radians(360 / segments / 2),
                (int(segments), 1),
            )
        )

    rho = ones(phi.shape) * radius
    rho_theta_phi = tstack([rho, theta, phi])

    vertices = spherical_to_cartesian(rho_theta_phi)

    # Removing extra longitude vertices.
    vertices = vertices[:-1, :, :]

    if axis == "+z":
        pass
    elif axis == "+y":
        vertices = np.roll(vertices, 2, -1)
    elif axis == "+x":
        vertices = np.roll(vertices, 1, -1)

    vertices += origin

    return vertices


PRIMITIVE_VERTICES_METHODS: CaseInsensitiveMapping = CaseInsensitiveMapping(
    {
        "Quad MPL": primitive_vertices_quad_mpl,
        "Grid MPL": primitive_vertices_grid_mpl,
        "Cube MPL": primitive_vertices_cube_mpl,
        "Sphere": primitive_vertices_sphere,
    }
)
PRIMITIVE_VERTICES_METHODS.__doc__ = """
Supported geometry primitive vertices generation methods.
"""


def primitive_vertices(
    method: Union[
        Literal["Cube MPL", "Quad MPL", "Grid MPL", "Sphere"], str
    ] = "Cube MPL",
    **kwargs: Any,
) -> NDArray:
    """
    Return the vertices of a geometry primitive using given method.

    Parameters
    ----------
    method
        Vertices generation method.

    Other Parameters
    ----------------
    axis
        {:func:`colour.geometry.primitive_vertices_quad_mpl`,
        :func:`colour.geometry.primitive_vertices_grid_mpl`,
        :func:`colour.geometry.primitive_vertices_sphere`},
        **{'+z', '+x', '+y', 'yz', 'xz', 'xy'}**,
        Axis the primitive will be normal to, or plane the primitive will be
        co-planar with.
    depth
        {:func:`colour.geometry.primitive_vertices_quad_mpl`,
        :func:`colour.geometry.primitive_vertices_grid_mpl`,
        :func:`colour.geometry.primitive_vertices_cube_mpl`},
        Primitive depth.
    depth_segments
        {:func:`colour.geometry.primitive_vertices_grid_mpl`,
        :func:`colour.geometry.primitive_vertices_cube_mpl`},
        Primitive depth segments, quad primitive counts along the depth.
    height
        {:func:`colour.geometry.primitive_vertices_quad_mpl`,
        :func:`colour.geometry.primitive_vertices_grid_mpl`,
        :func:`colour.geometry.primitive_vertices_cube_mpl`},
        Primitive height.
    height_segments
        {:func:`colour.geometry.primitive_vertices_grid_mpl`,
        :func:`colour.geometry.primitive_vertices_cube_mpl`},
        Primitive height segments, quad primitive counts along the height.
    intermediate
        {:func:`colour.geometry.primitive_vertices_sphere`},
        Whether to generate the sphere vertices at the center of the faces
        outlined by the segments of a regular sphere generated without
        the ``intermediate`` argument set to *True*. The resulting sphere is
        inscribed on the regular sphere faces but possesses the same poles.
    origin
        {:func:`colour.geometry.primitive_vertices_quad_mpl`,
        :func:`colour.geometry.primitive_vertices_grid_mpl`,
        :func:`colour.geometry.primitive_vertices_cube_mpl`,
        :func:`colour.geometry.primitive_vertices_sphere`},
        Primitive origin on the construction plane.
    planes
        {:func:`colour.geometry.primitive_vertices_cube_mpl`},
        **{'-x', '+x', '-y', '+y', '-z', '+z',
        'xy', 'xz', 'yz', 'yx', 'zx', 'zy'}**,
        Included grid primitives in the cube construction.
    radius
        {:func:`colour.geometry.primitive_vertices_sphere`},
        Sphere radius.
    segments
        {:func:`colour.geometry.primitive_vertices_sphere`},
        Latitude-longitude segments, if the ``intermediate`` argument is
        *True*, then the sphere will have one less segment along its longitude.
    width
        {:func:`colour.geometry.primitive_vertices_quad_mpl`,
        :func:`colour.geometry.primitive_vertices_grid_mpl`,
        :func:`colour.geometry.primitive_vertices_cube_mpl`},
        Primitive width.
    width_segments
        {:func:`colour.geometry.primitive_vertices_grid_mpl`,
        :func:`colour.geometry.primitive_vertices_cube_mpl`},
        Primitive width segments, quad primitive counts along the width.

    Returns
    -------
    :class:`numpy.ndarray`
        Primitive vertices.

    Examples
    --------
    >>> primitive_vertices()
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
    >>> primitive_vertices('Quad MPL')
    array([[ 0.,  0.,  0.],
           [ 1.,  0.,  0.],
           [ 1.,  1.,  0.],
           [ 0.,  1.,  0.]])
    >>> primitive_vertices('Sphere', segments=4)  # doctest: +ELLIPSIS
    array([[[  0.0000000...e+00,   0.0000000...e+00,   5.0000000...e-01],
            [ -3.5355339...e-01,  -4.3297802...e-17,   3.5355339...e-01],
            [ -5.0000000...e-01,  -6.1232340...e-17,   3.0616170...e-17],
            [ -3.5355339...e-01,  -4.3297802...e-17,  -3.5355339...e-01],
            [ -6.1232340...e-17,  -7.4987989...e-33,  -5.0000000...e-01]],
    <BLANKLINE>
           [[  0.0000000...e+00,   0.0000000...e+00,   5.0000000...e-01],
            [  2.1648901...e-17,  -3.5355339...e-01,   3.5355339...e-01],
            [  3.0616170...e-17,  -5.0000000...e-01,   3.0616170...e-17],
            [  2.1648901...e-17,  -3.5355339...e-01,  -3.5355339...e-01],
            [  3.7493994...e-33,  -6.1232340...e-17,  -5.0000000...e-01]],
    <BLANKLINE>
           [[  0.0000000...e+00,   0.0000000...e+00,   5.0000000...e-01],
            [  3.5355339...e-01,   0.0000000...e+00,   3.5355339...e-01],
            [  5.0000000...e-01,   0.0000000...e+00,   3.0616170...e-17],
            [  3.5355339...e-01,   0.0000000...e+00,  -3.5355339...e-01],
            [  6.1232340...e-17,   0.0000000...e+00,  -5.0000000...e-01]],
    <BLANKLINE>
           [[  0.0000000...e+00,   0.0000000...e+00,   5.0000000...e-01],
            [  2.1648901...e-17,   3.5355339...e-01,   3.5355339...e-01],
            [  3.0616170...e-17,   5.0000000...e-01,   3.0616170...e-17],
            [  2.1648901...e-17,   3.5355339...e-01,  -3.5355339...e-01],
            [  3.7493994...e-33,   6.1232340...e-17,  -5.0000000...e-01]]])
    """

    method = validate_method(method, PRIMITIVE_VERTICES_METHODS)

    function = PRIMITIVE_VERTICES_METHODS[method]

    return function(**filter_kwargs(function, **kwargs))
