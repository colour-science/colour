"""
Geometry
========

Defines the objects related to geometrical computations:

-   :func:`colour.algebra.normalise_vector`
-   :func:`colour.algebra.euclidean_distance`
-   :func:`colour.algebra.manhattan_distance`
-   :func:`colour.algebra.extend_line_segment`
-   :class:`colour.algebra.LineSegmentsIntersections_Specification`
-   :func:`colour.algebra.intersect_line_segments`
-   :func:`colour.algebra.ellipse_coefficients_general_form`
-   :func:`colour.algebra.ellipse_coefficients_canonical_form`
-   :func:`colour.algebra.point_at_angle_on_ellipse`
-   :func:`colour.algebra.ellipse_fitting_Halir1998`

References
----------
-   :cite:`Bourkea` : Bourke, P. (n.d.). Intersection point of two line
    segments in 2 dimensions. Retrieved January 15, 2016, from
    http://paulbourke.net/geometry/pointlineplane/
-   :cite:`Erdema` : Erdem, U. M. (n.d.). Fast Line Segment Intersection.
    Retrieved January 15, 2016, from
    http://www.mathworks.com/matlabcentral/fileexchange/\
27205-fast-line-segment-intersection
-   :cite:`Halir1998` : Halir, R., & Flusser, J. (1998). Numerically Stable
    Direct Least Squares Fitting Of Ellipses (pp. 1-8).
    http://citeseerx.ist.psu.edu/viewdoc/download;\
jsessionid=BEEAFC85DE53308286D626302F4A3E3C?doi=10.1.1.1.7559&rep=rep1&type=pdf
-   :cite:`Saeedna` : Saeedn. (n.d.). Extend a line segment a specific
    distance. Retrieved January 16, 2016, from
    http://stackoverflow.com/questions/7740507/\
extend-a-line-segment-a-specific-distance
-   :cite:`Wikipedia` : Wikipedia. (n.d.). Ellipse. Retrieved November 24,
    2018, from https://en.wikipedia.org/wiki/Ellipse
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from colour.hints import (
    ArrayLike,
    Floating,
    FloatingOrNDArray,
    Literal,
    NDArray,
    Union,
)
from colour.utilities import (
    CaseInsensitiveMapping,
    as_float,
    as_float_array,
    ones,
    tsplit,
    tstack,
    validate_method,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "normalise_vector",
    "euclidean_distance",
    "manhattan_distance",
    "extend_line_segment",
    "LineSegmentsIntersections_Specification",
    "intersect_line_segments",
    "ellipse_coefficients_general_form",
    "ellipse_coefficients_canonical_form",
    "point_at_angle_on_ellipse",
    "ellipse_fitting_Halir1998",
    "ELLIPSE_FITTING_METHODS",
    "ellipse_fitting",
]


def normalise_vector(a: ArrayLike) -> NDArray:
    """
    Normalise given vector :math:`a`.

    Parameters
    ----------
    a
        Vector :math:`a` to normalise.

    Returns
    -------
    :class:`numpy.ndarray`
        Normalised vector :math:`a`.

    Examples
    --------
    >>> a = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> normalise_vector(a)  # doctest: +ELLIPSIS
    array([ 0.8419703...,  0.4972256...,  0.2094102...])
    """

    a = as_float_array(a)

    return a / np.linalg.norm(a)


def euclidean_distance(a: ArrayLike, b: ArrayLike) -> FloatingOrNDArray:
    """
    Return the *Euclidean* distance between point array :math:`a` and point
    array :math:`b`.

    For a two-dimensional space, the metric is as follows:

    :math:`E_D = [(x_a - x_b)^2 + (y_a - y_b)^2]^{1/2}`

    Parameters
    ----------
    a
        Point array :math:`a`.
    b
        Point array :math:`b`.

    Returns
    -------
    :class:`np.floating` or :class:`numpy.ndarray`
        *Euclidean* distance.

    Examples
    --------
    >>> a = np.array([100.00000000, 21.57210357, 272.22819350])
    >>> b = np.array([100.00000000, 426.67945353, 72.39590835])
    >>> euclidean_distance(a, b)  # doctest: +ELLIPSIS
    451.7133019...
    """

    return as_float(
        np.linalg.norm(as_float_array(a) - as_float_array(b), axis=-1)
    )


def manhattan_distance(a: ArrayLike, b: ArrayLike) -> FloatingOrNDArray:
    """
    Return the *Manhattan* (or *City-Block*) distance between point array
    :math:`a` and point array :math:`b`.

    For a two-dimensional space, the metric is as follows:

    :math:`M_D = |x_a - x_b| + |y_a - y_b|`

    Parameters
    ----------
    a
        Point array :math:`a`.
    b
        Point array :math:`b`.

    Returns
    -------
    :class:`np.floating` or :class:`numpy.ndarray`
        *Manhattan* distance.

    Examples
    --------
    >>> a = np.array([100.00000000, 21.57210357, 272.22819350])
    >>> b = np.array([100.00000000, 426.67945353, 72.39590835])
    >>> manhattan_distance(a, b)  # doctest: +ELLIPSIS
    604.9396351...
    """

    return as_float(
        np.sum(np.abs(as_float_array(a) - as_float_array(b)), axis=-1)
    )


def extend_line_segment(
    a: ArrayLike, b: ArrayLike, distance: Floating = 1
) -> NDArray:
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

    x_c = x_b + (x_b - x_a) / d * distance
    y_c = y_b + (y_b - y_a) / d * distance

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

    xy: NDArray
    intersect: NDArray
    parallel: NDArray
    coincident: NDArray


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
    ...     [[[0.15416284, 0.7400497],
    ...       [0.26331502, 0.53373939]],
    ...      [[0.01457496, 0.91874701],
    ...       [0.90071485, 0.03342143]]]
    ... )
    >>> l_2 = np.array(
    ...     [[[0.95694934, 0.13720932],
    ...        [0.28382835, 0.60608318]],
    ...       [[0.94422514, 0.85273554],
    ...        [0.00225923, 0.52122603]],
    ...       [[0.55203763, 0.48537741],
    ...        [0.76813415, 0.16071675]]]
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

    l_1 = np.reshape(l_1, (-1, 4))
    l_2 = np.reshape(l_2, (-1, 4))

    r_1, c_1 = l_1.shape[0], l_1.shape[1]
    r_2, c_2 = l_2.shape[0], l_2.shape[1]

    x_1, y_1, x_2, y_2 = (
        np.tile(l_1[:, i, np.newaxis], (1, r_2)) for i in range(c_1)
    )

    l_2 = np.transpose(l_2)

    x_3, y_3, x_4, y_4 = (np.tile(l_2[i, :], (r_1, 1)) for i in range(c_2))

    x_4_x_3 = x_4 - x_3
    y_1_y_3 = y_1 - y_3
    y_4_y_3 = y_4 - y_3
    x_1_x_3 = x_1 - x_3
    x_2_x_1 = x_2 - x_1
    y_2_y_1 = y_2 - y_1

    numerator_a = x_4_x_3 * y_1_y_3 - y_4_y_3 * x_1_x_3
    numerator_b = x_2_x_1 * y_1_y_3 - y_2_y_1 * x_1_x_3
    denominator = y_4_y_3 * x_2_x_1 - x_4_x_3 * y_2_y_1

    u_a = numerator_a / denominator
    u_b = numerator_b / denominator

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


def ellipse_coefficients_general_form(coefficients: ArrayLike) -> NDArray:
    """
    Return the general form ellipse coefficients from given canonical form
    ellipse coefficients.

    The canonical form ellipse coefficients are as follows: the center
    coordinates :math:`x_c` and :math:`y_c`, semi-major axis length
    :math:`a_a`, semi-minor axis length :math:`a_b` and rotation angle
    :math:`\\theta` in degrees of its semi-major axis :math:`a_a`.

    Parameters
    ----------
    coefficients
        Canonical form ellipse coefficients.

    Returns
    -------
    :class:`numpy.ndarray`
        General form ellipse coefficients.

    References
    ----------
    :cite:`Wikipedia`

    Examples
    --------
    >>> coefficients = np.array([0.5, 0.5, 2, 1, 45])
    >>> ellipse_coefficients_general_form(coefficients)
    array([ 2.5, -3. ,  2.5, -1. , -1. , -3.5])
    """

    x_c, y_c, a_a, a_b, theta = tsplit(coefficients)

    theta = np.radians(theta)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_theta_2 = cos_theta**2
    sin_theta_2 = sin_theta**2
    a_a_2 = a_a**2
    a_b_2 = a_b**2

    a = a_a_2 * sin_theta_2 + a_b_2 * cos_theta_2
    b = 2 * (a_b_2 - a_a_2) * sin_theta * cos_theta
    c = a_a_2 * cos_theta_2 + a_b_2 * sin_theta_2
    d = -2 * a * x_c - b * y_c
    e = -b * x_c - 2 * c * y_c
    f = a * x_c**2 + b * x_c * y_c + c * y_c**2 - a_a_2 * a_b_2

    return np.array([a, b, c, d, e, f])


def ellipse_coefficients_canonical_form(coefficients: ArrayLike) -> NDArray:
    """
    Return the canonical form ellipse coefficients from given general form
    ellipse coefficients.

    The general form ellipse coefficients are the coefficients of the implicit
    second-order polynomial/quadratic curve expressed as follows:

    :math:`F\\left(x, y\\right)` = ax^2 + bxy + cy^2 + dx + ey + f = 0`

    with an ellipse-specific constraint such as :math:`b^2 -4ac < 0` and where
    :math:`a, b, c, d, e, f` are coefficients of the ellipse and
    :math:`F\\left(x, y\\right)` are coordinates of points lying on it.

    Parameters
    ----------
    coefficients
        General form ellipse coefficients.

    Returns
    -------
    :class:`numpy.ndarray`
        Canonical form ellipse coefficients.

    References
    ----------
    :cite:`Wikipedia`

    Examples
    --------
    >>> coefficients = np.array([ 2.5, -3.0,  2.5, -1.0, -1.0, -3.5])
    >>> ellipse_coefficients_canonical_form(coefficients)
    array([  0.5,   0.5,   2. ,   1. ,  45. ])
    """

    a, b, c, d, e, f = tsplit(coefficients)

    d_1 = b**2 - 4 * a * c
    n_p_1 = 2 * (a * e**2 + c * d**2 - b * d * e + d_1 * f)
    n_p_2 = np.sqrt((a - c) ** 2 + b**2)

    a_a = -np.sqrt(n_p_1 * (a + c + n_p_2)) / d_1
    a_b = -np.sqrt(n_p_1 * (a + c - n_p_2)) / d_1

    x_c = (2 * c * d - b * e) / d_1
    y_c = (2 * a * e - b * d) / d_1

    theta = np.select(
        [
            np.logical_and(b == 0, a < c),
            np.logical_and(b == 0, a > c),
            b != 0,
        ],
        [
            0,
            90,
            np.degrees(np.arctan((c - a - n_p_2) / b)),
        ],
    )

    return np.array([x_c, y_c, a_a, a_b, theta])


def point_at_angle_on_ellipse(
    phi: ArrayLike, coefficients: ArrayLike
) -> NDArray:
    """
    Return the coordinates of the point at angle :math:`\\phi` in degrees on
    the ellipse with given canonical form coefficients.

    Parameters
    ----------
    phi
        Point at angle :math:`\\phi` in degrees to retrieve the coordinates
        of.
    coefficients
        General form ellipse coefficients as follows: the center coordinates
        :math:`x_c` and :math:`y_c`, semi-major axis length :math:`a_a`,
        semi-minor axis length :math:`a_b` and rotation angle :math:`\\theta`
        in degrees of its semi-major axis :math:`a_a`.

    Returns
    -------
    :class:`numpy.ndarray`
        Coordinates of the point at angle :math:`\\phi`

    Examples
    --------
    >>> coefficients = np.array([0.5, 0.5, 2, 1, 45])
    >>> point_at_angle_on_ellipse(45, coefficients)  # doctest: +ELLIPSIS
    array([ 1.,  2.])
    """

    phi = np.radians(phi)
    x_c, y_c, a_a, a_b, theta = tsplit(coefficients)
    theta = np.radians(theta)

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    x = x_c + a_a * cos_theta * cos_phi - a_b * sin_theta * sin_phi
    y = y_c + a_a * sin_theta * cos_phi + a_b * cos_theta * sin_phi

    return tstack([x, y])


def ellipse_fitting_Halir1998(a: ArrayLike) -> NDArray:
    """
    Return the coefficients of the implicit second-order polynomial/quadratic
    curve that fits given point array :math:`a` using
    *Halir and Flusser (1998)* method.

    The implicit second-order polynomial is expressed as follows:

    :math:`F\\left(x, y\\right)` = ax^2 + bxy + cy^2 + dx + ey + f = 0`

    with an ellipse-specific constraint such as :math:`b^2 -4ac < 0` and where
    :math:`a, b, c, d, e, f` are coefficients of the ellipse and
    :math:`F\\left(x, y\\right)` are coordinates of points lying on it.

    Parameters
    ----------
    a
        Point array :math:`a` to be fitted.

    Returns
    -------
    :class:`numpy.ndarray`
        Coefficients of the implicit second-order polynomial/quadratic
        curve that fits given point array :math:`a`.

    References
    ----------
    :cite:`Halir1998`

    Examples
    --------
    >>> a = np.array([[2, 0], [0, 1], [-2, 0], [0, -1]])
    >>> ellipse_fitting_Halir1998(a)  # doctest: +ELLIPSIS
    array([ 0.2425356...,  0.        ,  0.9701425...,  0.        ,  0.        ,
           -0.9701425...])
    >>> ellipse_coefficients_canonical_form(ellipse_fitting_Halir1998(a))
    array([-0., -0.,  2.,  1.,  0.])
    """

    x, y = tsplit(a)

    # Quadratic part of the design matrix.
    D1 = tstack([x**2, x * y, y**2])
    # Linear part of the design matrix.
    D2 = tstack([x, y, ones(x.shape)])

    D1_T = np.transpose(D1)
    D2_T = np.transpose(D2)

    # Quadratic part of the scatter matrix.
    S1 = np.dot(D1_T, D1)
    # Combined part of the scatter matrix.
    S2 = np.dot(D1_T, D2)
    # Linear part of the scatter matrix.
    S3 = np.dot(D2_T, D2)

    T = -np.dot(np.linalg.inv(S3), np.transpose(S2))

    # Reduced scatter matrix.
    M = S1 + np.dot(S2, T)
    M = np.array([M[2, :] / 2, -M[1, :], M[0, :] / 2])

    _w, v = np.linalg.eig(M)

    A1 = v[:, np.nonzero(4 * v[0, :] * v[2, :] - v[1, :] ** 2 > 0)[0]]
    A2 = np.dot(T, A1)

    A = np.ravel([A1, A2])

    return A


ELLIPSE_FITTING_METHODS: CaseInsensitiveMapping = CaseInsensitiveMapping(
    {"Halir 1998": ellipse_fitting_Halir1998}
)
ELLIPSE_FITTING_METHODS.__doc__ = """
Supported ellipse fitting methods.

References
----------
:cite:`Halir1998`
"""


def ellipse_fitting(
    a: ArrayLike, method: Union[Literal["Halir 1998"], str] = "Halir 1998"
) -> NDArray:
    """
    Return the coefficients of the implicit second-order polynomial/quadratic
    curve that fits given point array :math:`a` using
    given method.

    The implicit second-order polynomial is expressed as follows:

    :math:`F\\left(x, y\\right)` = ax^2 + bxy + cy^2 + dx + ey + f = 0`

    with an ellipse-specific constraint such as :math:`b^2 -4ac < 0` and where
    :math:`a, b, c, d, e, f` are coefficients of the ellipse and
    :math:`F\\left(x, y\\right)` are coordinates of points lying on it.

    Parameters
    ----------
    a
        Point array :math:`a` to be fitted.
    method
        Computation method.

    Returns
    -------
    :class:`numpy.ndarray`
        Coefficients of the implicit second-order polynomial/quadratic
        curve that fits given point array :math:`a`.

    References
    ----------
    :cite:`Halir1998`

    Examples
    --------
    >>> a = np.array([[2, 0], [0, 1], [-2, 0], [0, -1]])
    >>> ellipse_fitting(a)  # doctest: +ELLIPSIS
    array([ 0.2425356...,  0.        ,  0.9701425...,  0.        ,  0.        ,
           -0.9701425...])
    >>> ellipse_coefficients_canonical_form(ellipse_fitting(a))
    array([-0., -0.,  2.,  1.,  0.])
    """

    method = validate_method(method, ELLIPSE_FITTING_METHODS)

    function = ELLIPSE_FITTING_METHODS[method]

    return function(a)
