"""
Ellipse
=======

Defines the objects related to ellipse computations:

-   :func:`colour.algebra.ellipse_coefficients_general_form`
-   :func:`colour.algebra.ellipse_coefficients_canonical_form`
-   :func:`colour.algebra.point_at_angle_on_ellipse`
-   :func:`colour.algebra.ellipse_fitting_Halir1998`

References
----------
-   :cite:`Halir1998` : Halir, R., & Flusser, J. (1998). Numerically Stable
    Direct Least Squares Fitting Of Ellipses (pp. 1-8).
    http://citeseerx.ist.psu.edu/viewdoc/download;\
jsessionid=BEEAFC85DE53308286D626302F4A3E3C?doi=10.1.1.1.7559&rep=rep1&type=pdf
-   :cite:`Wikipedia` : Wikipedia. (n.d.). Ellipse. Retrieved November 24,
    2018, from https://en.wikipedia.org/wiki/Ellipse
"""

from __future__ import annotations

import numpy as np

from colour.hints import ArrayLike, Literal, NDArrayFloat, cast
from colour.utilities import (
    CanonicalMapping,
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
    "ellipse_coefficients_general_form",
    "ellipse_coefficients_canonical_form",
    "point_at_angle_on_ellipse",
    "ellipse_fitting_Halir1998",
    "ELLIPSE_FITTING_METHODS",
    "ellipse_fitting",
]


def ellipse_coefficients_general_form(coefficients: ArrayLike) -> NDArrayFloat:
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


def ellipse_coefficients_canonical_form(
    coefficients: ArrayLike,
) -> NDArrayFloat:
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
    >>> coefficients = np.array([2.5, -3.0, 2.5, -1.0, -1.0, -3.5])
    >>> ellipse_coefficients_canonical_form(coefficients)
    array([  0.5,   0.5,   2. ,   1. ,  45. ])
    """

    a, b, c, d, e, f = tsplit(coefficients)

    d_1 = b**2 - 4 * a * c
    n_p_1 = 2 * (a * e**2 + c * d**2 - b * d * e + d_1 * f)
    n_p_2 = np.sqrt((a - c) ** 2 + b**2)

    a_a = (-np.sqrt(n_p_1 * (a + c + n_p_2))) / d_1
    a_b = (-np.sqrt(n_p_1 * (a + c - n_p_2))) / d_1

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
) -> NDArrayFloat:
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


def ellipse_fitting_Halir1998(a: ArrayLike) -> NDArrayFloat:
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

    A = cast(NDArrayFloat, np.ravel([A1, A2]))

    return A


ELLIPSE_FITTING_METHODS: CanonicalMapping = CanonicalMapping(
    {"Halir 1998": ellipse_fitting_Halir1998}
)
ELLIPSE_FITTING_METHODS.__doc__ = """
Supported ellipse fitting methods.

References
----------
:cite:`Halir1998`
"""


def ellipse_fitting(
    a: ArrayLike, method: Literal["Halir 1998"] | str = "Halir 1998"
) -> NDArrayFloat:
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

    method = validate_method(method, tuple(ELLIPSE_FITTING_METHODS))

    function = ELLIPSE_FITTING_METHODS[method]

    return function(a)
