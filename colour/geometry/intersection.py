"""
Intersection Utilities
======================

Defines the geometry intersection utilities objects.

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

import numpy as np
from dataclasses import dataclass

from colour.algebra import euclidean_distance, sdiv, sdiv_mode
from colour.hints import ArrayLike, NDArrayFloat
from colour.utilities import as_float_array, tsplit, tstack

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "extend_line_segment",
    "LineSegmentsIntersections_Specification",
    "intersect_line_segments",
]


def extend_line_segment(
    a: ArrayLike, b: ArrayLike, distance: float = 1
) -> NDArrayFloat:
    """
    Extend the line segment defined by point arrays :math:`a` and :math:`b` by
    given distance and return the new end point.

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

    xy_c = tstack([x_c, y_c])

    return xy_c


@dataclass
class LineSegmentsIntersections_Specification:
    """
    Define the specification for intersection of line segments :math:`l_1` and
    :math:`l_2` returned by :func:`colour.algebra.intersect_line_segments`
    definition.

    Parameters
    ----------
    xy
        Array of :math:`l_1` and :math:`l_2` line segments intersections
        coordinates. Non existing segments intersections coordinates are set
        with `np.nan`.
    intersect
        Array of *bool* indicating if line segments :math:`l_1` and :math:`l_2`
        intersect.
    parallel
        Array of :class:`bool` indicating if line segments :math:`l_1` and
        :math:`l_2` are parallel.
    coincident
        Array of :class:`bool` indicating if line segments :math:`l_1` and
        :math:`l_2` are coincident.
    """

    xy: NDArrayFloat
    intersect: NDArrayFloat
    parallel: NDArrayFloat
    coincident: NDArrayFloat


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
    array([[[        nan,         nan],
            [ 0.2279184...,  0.6006430...],
            [        nan,         nan]],
    <BLANKLINE>
           [[ 0.4281451...,  0.5055568...],
            [ 0.3056055...,  0.6279838...],
            [ 0.7578749...,  0.1761301...]]])
    >>> s.intersect
    array([[False,  True, False],
           [ True,  True,  True]], dtype=bool)
    >>> s.parallel
    array([[False, False, False],
           [False, False, False]], dtype=bool)
    >>> s.coincident
    array([[False, False, False],
           [False, False, False]], dtype=bool)
    """

    l_1 = as_float_array(l_1)
    l_2 = as_float_array(l_2)

    l_1 = np.reshape(l_1, (-1, 4))
    l_2 = np.reshape(l_2, (-1, 4))

    r_1, c_1 = l_1.shape[0], l_1.shape[1]
    r_2, c_2 = l_2.shape[0], l_2.shape[1]

    x_1, y_1, x_2, y_2 = (
        np.tile(l_1[:, i, None], (1, r_2)) for i in range(c_1)
    )

    l_2 = np.transpose(l_2)

    x_3, y_3, x_4, y_4 = (np.tile(l_2[i, :], (r_1, 1)) for i in range(c_2))

    x_4_x_3 = x_4 - x_3
    y_1_y_3 = y_1 - y_3
    y_4_y_3 = y_4 - y_3
    x_1_x_3 = x_1 - x_3
    x_2_x_1 = x_2 - x_1
    y_2_y_1 = y_2 - y_1

    numerator_a = x_4_x_3 * y_1_y_3 - y_4_y_3 * x_1_x_3  # pyright: ignore
    numerator_b = x_2_x_1 * y_1_y_3 - y_2_y_1 * x_1_x_3  # pyright: ignore
    denominator = y_4_y_3 * x_2_x_1 - x_4_x_3 * y_2_y_1  # pyright: ignore

    with sdiv_mode("Ignore"):
        u_a = sdiv(numerator_a, denominator)
        u_b = sdiv(numerator_b, denominator)

    intersect = np.logical_and.reduce((u_a >= 0, u_a <= 1, u_b >= 0, u_b <= 1))
    xy = tstack([x_1 + x_2_x_1 * u_a, y_1 + y_2_y_1 * u_a])
    xy[~intersect] = np.nan
    parallel = denominator == 0
    coincident = np.logical_and.reduce(
        (numerator_a == 0, numerator_b == 0, parallel)
    )

    return LineSegmentsIntersections_Specification(
        xy, intersect, parallel, coincident
    )
