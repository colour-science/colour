"""
Intersection Utilities
======================

Define utilities for computing geometric intersections and line segment
operations in two-dimensional space.

References
----------
-   :cite:`Bourkea` : Bourke, P. (n.d.). Intersection point of two line
    segments in 2 dimensions. Retrieved January 15, 2016, from
    http://paulbourke.net/geometry/pointlineplane/
-   :cite:`Erdema` : Erdem, U. M. (n.d.). Fast Line Segment Intersection.
    Retrieved January 15, 2016, from
    http://www.mathworks.com/matlabcentral/fileexchange/\
27205-fast-line-segment-intersection
-   :cite:`Saeedna` : Saeedn. (n.d.). Extend a line segment a specific
    distance. Retrieved January 16, 2016, from
    http://stackoverflow.com/questions/7740507/\
extend-a-line-segment-a-specific-distance
"""

from __future__ import annotations

import typing
from dataclasses import dataclass

from colour.algebra import euclidean_distance, sdiv, sdiv_mode

if typing.TYPE_CHECKING:
    from colour.hints import ArrayLike, NDArrayBoolean, NDArrayFloat

from colour.utilities import (
    array_namespace,
    as_float_array,
    tsplit,
    tstack,
    xp_as_float_array,
    xp_reshape,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "extend_line_segment",
    "LineSegmentsIntersections_Specification",
    "intersect_line_segments",
    "intersect_ray_circle_2d",
]


def extend_line_segment(
    a: ArrayLike, b: ArrayLike, distance: float = 1
) -> NDArrayFloat:
    """
    Extend the line segment defined by point arrays :math:`a` and :math:`b` by
    the specified distance and generate the new end point.

    Parameters
    ----------
    a
        Point array :math:`a`.
    b
        Point array :math:`b`.
    distance
        Distance to extend the line segment.

    Returns
    -------
    :class:`numpy.ndarray`
        New end point.

    References
    ----------
    :cite:`Saeedna`

    Notes
    -----
    -   Input line segment points coordinates are 2d coordinates.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([0.95694934, 0.13720932])
    >>> b = np.array([0.28382835, 0.60608318])
    >>> extend_line_segment(a, b)  # doctest: +ELLIPSIS
    array([-0.5367248...,  1.1776534...])
    """

    x_a, y_a = tsplit(a)
    x_b, y_b = tsplit(b)

    d = euclidean_distance(a, b)

    with sdiv_mode():
        x_c = x_b + sdiv(x_b - x_a, d) * distance
        y_c = y_b + sdiv(y_b - y_a, d) * distance

    return tstack([x_c, y_c])


@dataclass
class LineSegmentsIntersections_Specification:
    """
    Define the specification for intersection of line segments :math:`l_1` and
    :math:`l_2` returned by the
    :func:`colour.algebra.intersect_line_segments` definition.

    Parameters
    ----------
    xy
        Array of :math:`l_1` and :math:`l_2` line segments intersections
        coordinates. Non-existing segments intersections coordinates are set
        with `np.nan`.
    intersect
        Array of *bool* indicating if line segments :math:`l_1` and
        :math:`l_2` intersect.
    parallel
        Array of :class:`bool` indicating if line segments :math:`l_1` and
        :math:`l_2` are parallel.
    coincident
        Array of :class:`bool` indicating if line segments :math:`l_1` and
        :math:`l_2` are coincident.
    """

    xy: NDArrayFloat
    intersect: NDArrayBoolean
    parallel: NDArrayBoolean
    coincident: NDArrayBoolean


def intersect_line_segments(
    l_1: ArrayLike, l_2: ArrayLike
) -> LineSegmentsIntersections_Specification:
    """
    Compute :math:`l_1` line segments intersections with :math:`l_2` line
    segments.

    Parameters
    ----------
    l_1
        :math:`l_1` line segments array, each row is a line segment such as
        (:math:`x_1`, :math:`y_1`, :math:`x_2`, :math:`y_2`) where
        (:math:`x_1`, :math:`y_1`) and (:math:`x_2`, :math:`y_2`) are
        respectively the start and end points of :math:`l_1` line segments.
    l_2
        :math:`l_2` line segments array, each row is a line segment such as
        (:math:`x_3`, :math:`y_3`, :math:`x_4`, :math:`y_4`) where
        (:math:`x_3`, :math:`y_3`) and (:math:`x_4`, :math:`y_4`) are
        respectively the start and end points of :math:`l_2` line segments.

    Returns
    -------
    :class:`colour.algebra.LineSegmentsIntersections_Specification`
        Line segments intersections specification.

    References
    ----------
    :cite:`Bourkea`, :cite:`Erdema`

    Notes
    -----
    -   Input line segments points coordinates are 2d coordinates.

    Examples
    --------
    >>> import numpy as np
    >>> l_1 = np.array(
    ...     [
    ...         [[0.15416284, 0.7400497], [0.26331502, 0.53373939]],
    ...         [[0.01457496, 0.91874701], [0.90071485, 0.03342143]],
    ...     ]
    ... )
    >>> l_2 = np.array(
    ...     [
    ...         [[0.95694934, 0.13720932], [0.28382835, 0.60608318]],
    ...         [[0.94422514, 0.85273554], [0.00225923, 0.52122603]],
    ...         [[0.55203763, 0.48537741], [0.76813415, 0.16071675]],
    ...     ]
    ... )
    >>> s = intersect_line_segments(l_1, l_2)
    >>> s.xy  # doctest: +ELLIPSIS
    array([[[       nan,        nan],
            [0.2279184..., 0.6006430...],
            [       nan,        nan]],
    <BLANKLINE>
           [[0.4281451..., 0.5055568...],
            [0.3056055..., 0.6279838...],
            [0.7578749..., 0.1761301...]]])
    >>> s.intersect
    array([[False,  True, False],
           [ True,  True,  True]])
    >>> s.parallel
    array([[False, False, False],
           [False, False, False]])
    >>> s.coincident
    array([[False, False, False],
           [False, False, False]])
    """

    l_1 = as_float_array(l_1)
    l_2 = as_float_array(l_2)

    xp = array_namespace(l_1, l_2)

    l_2 = xp_as_float_array(l_2, xp=xp, like=l_1)

    l_1 = xp_reshape(l_1, (-1, 4), xp=xp)
    l_2 = xp_reshape(l_2, (-1, 4), xp=xp)

    # ``l_1`` segments held as ``(r_1, 1)`` columns and ``l_2`` segments as
    # ``(1, r_2)`` rows; pairwise arithmetic broadcasts to ``(r_1, r_2)``
    # without materialising tiled copies of each component.
    x_1, y_1, x_2, y_2 = l_1[:, 0:1], l_1[:, 1:2], l_1[:, 2:3], l_1[:, 3:4]
    x_3, y_3, x_4, y_4 = (
        l_2[None, :, 0],
        l_2[None, :, 1],
        l_2[None, :, 2],
        l_2[None, :, 3],
    )

    x_4_x_3 = x_4 - x_3
    y_1_y_3 = y_1 - y_3
    y_4_y_3 = y_4 - y_3
    x_1_x_3 = x_1 - x_3
    x_2_x_1 = x_2 - x_1
    y_2_y_1 = y_2 - y_1

    numerator_a = x_4_x_3 * y_1_y_3 - y_4_y_3 * x_1_x_3
    numerator_b = x_2_x_1 * y_1_y_3 - y_2_y_1 * x_1_x_3
    denominator = y_4_y_3 * x_2_x_1 - x_4_x_3 * y_2_y_1
    parallel = denominator == 0

    # Parallel candidates are inactive by definition. Replacing their zero
    # denominator before division keeps invalid derivatives from contaminating
    # the backward pass through a subsequently selected intersection.
    denominator_safe = xp.where(parallel, xp.ones_like(denominator), denominator)

    with sdiv_mode("Ignore"):
        u_a = sdiv(numerator_a, denominator_safe)
        u_b = sdiv(numerator_b, denominator_safe)

    intersect = (~parallel) & (u_a >= 0) & (u_a <= 1) & (u_b >= 0) & (u_b <= 1)
    xy = tstack([x_1 + x_2_x_1 * u_a, y_1 + y_2_y_1 * u_a])
    xy = xp.where(intersect[..., None], xy, float("nan"))
    coincident = (numerator_a == 0) & (numerator_b == 0) & parallel

    return LineSegmentsIntersections_Specification(xy, intersect, parallel, coincident)


def intersect_ray_circle_2d(
    ray_origin: ArrayLike,
    ray_direction: ArrayLike,
    circle_radius: float,
) -> NDArrayFloat:
    """
    Compute the intersection distance of 2D ray(s) with a circle centred
    at the origin.

    Supports batched inputs: if *ray_origin* and *ray_direction* have
    shape ``(..., 2)``, the result has shape ``(...)``.

    Parameters
    ----------
    ray_origin
        Ray origin(s) as 2D point(s) ``[..., 2]``.
    ray_direction
        Ray direction(s) as 2D vector(s) ``[..., 2]`` (does not need to
        be normalised).
    circle_radius
        Radius of the circle centred at the origin.

    Returns
    -------
    :class:`numpy.ndarray`
        Distance(s) along the ray to the nearest forward intersection,
        or ``np.nan`` where no forward intersection exists.

    Examples
    --------
    >>> intersect_ray_circle_2d([0, 5], [0, 1], 10)
    array(5.)
    >>> intersect_ray_circle_2d([0, 15], [0, 1], 10)
    array(nan)
    """

    origin = as_float_array(ray_origin)

    xp = array_namespace(origin)

    direction = as_float_array(ray_direction)

    direction_x = direction[..., 0]
    direction_y = direction[..., 1]
    origin_x = origin[..., 0]
    origin_y = origin[..., 1]

    quadratic_a = direction_x * direction_x + direction_y * direction_y
    quadratic_b = 2.0 * (origin_x * direction_x + origin_y * direction_y)
    quadratic_c = (
        origin_x * origin_x + origin_y * origin_y - circle_radius * circle_radius
    )
    discriminant = quadratic_b * quadratic_b - 4.0 * quadratic_a * quadratic_c

    has_intersection = discriminant > 0.0
    safe_discriminant = xp.sqrt(xp.clip(discriminant, min=0.0))

    distance_1 = (-quadratic_b + safe_discriminant) / (2.0 * quadratic_a)
    distance_2 = (-quadratic_b - safe_discriminant) / (2.0 * quadratic_a)

    both_positive = (distance_1 > 0) & (distance_2 > 0)
    result = xp.where(
        both_positive,
        xp.minimum(distance_1, distance_2),
        xp.maximum(distance_1, distance_2),
    )
    forward = result > 0
    return xp.where(has_intersection & forward, result, float("nan"))
