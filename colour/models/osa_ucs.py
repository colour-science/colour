"""
Optical Society of America Uniform Colour Scales (OSA UCS)
==========================================================

Define the *OSA UCS* colourspace transformations.

-   :func:`colour.XYZ_to_OSA_UCS`
-   :func:`colour.OSA_UCS_to_XYZ`

References
----------
-   :cite:`Cao2013` : Cao, R., Trussell, H. J., & Shamey, R. (2013). Comparison
    of the performance of inverse transformation methods from OSA-UCS to
    CIEXYZ. Journal of the Optical Society of America A, 30(8), 1508.
    doi:10.1364/JOSAA.30.001508
-   :cite:`Moroney2003` : Moroney, N. (2003). A Radial Sampling of the OSA
    Uniform Color Scales. Color and Imaging Conference, 2003(1), 175-180.
    ISSN:2166-9635
-   :cite:`Schlomer2019` : Schlömer, N. (2019). On the conversion from OSA-UCS
    to CIEXYZ (Version 2). arXiv. doi:10.48550/ARXIV.1911.08323
"""

from __future__ import annotations

import typing

import numpy as np

from colour.algebra import sdiv, sdiv_mode, spow, vecmul

if typing.TYPE_CHECKING:
    from colour.hints import NDArrayFloat

from colour.hints import (  # noqa: TC001
    Domain100,
    NDArrayFloat,
    Range100,
)
from colour.models import XYZ_to_xyY
from colour.utilities import (
    from_range_100,
    to_domain_100,
    tsplit,
    tstack,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "XYZ_to_OSA_UCS",
    "OSA_UCS_to_XYZ",
]

MATRIX_XYZ_TO_RGB_OSA_UCS: NDArrayFloat = np.array(
    [
        [0.799, 0.4194, -0.1648],
        [-0.4493, 1.3265, 0.0927],
        [-0.1149, 0.3394, 0.717],
    ]
)
"""
*OSA UCS* matrix converting from *CIE XYZ* tristimulus values to *RGB*
colourspace.
"""

MATRIX_RGB_TO_XYZ_OSA_UCS: NDArrayFloat = np.linalg.inv(MATRIX_XYZ_TO_RGB_OSA_UCS)
"""
*OSA UCS* matrix converting from *RGB* colourspace to *CIE XYZ* tristimulus
values (inverse of MATRIX_XYZ_TO_RGB_OSA_UCS).
"""


def XYZ_to_OSA_UCS(XYZ: Domain100) -> Range100:
    """
    Convert from *CIE XYZ* tristimulus values under the
    *CIE 1964 10 Degree Standard Observer* to *OSA UCS* colourspace.

    The lightness axis, *L*, is typically in range [-9, 5] and centered
    around middle gray (Munsell N/6). The yellow-blue axis, *j*, is
    typically in range [-15, 15]. The red-green axis, *g*, is typically in
    range [-20, 15].

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values under the
        *CIE 1964 10 Degree Standard Observer*.

    Returns
    -------
    :class:`numpy.ndarray`
        *OSA UCS* :math:`Ljg` lightness, jaune (yellowness), and greenness.

    Notes
    -----
    +------------+-----------------------+--------------------+
    | **Domain** | **Scale - Reference** | **Scale - 1**      |
    +============+=======================+====================+
    | ``XYZ``    | 100                   | 1                  |
    +------------+-----------------------+--------------------+

    +------------+-----------------------+--------------------+
    | **Range**  | **Scale - Reference** | **Scale - 1**      |
    +============+=======================+====================+
    | ``Ljg``    | 100                   | 1                  |
    +------------+-----------------------+--------------------+

    -   *OSA UCS* uses the *CIE 1964 10 Degree Standard Observer*.

    References
    ----------
    :cite:`Cao2013`, :cite:`Moroney2003`

    Examples
    --------
    >>> import numpy as np
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952]) * 100
    >>> XYZ_to_OSA_UCS(XYZ)  # doctest: +ELLIPSIS
    array([-3.0049979...,  2.9971369..., -9.6678423...])
    """

    XYZ = to_domain_100(XYZ)
    x, y, Y = tsplit(XYZ_to_xyY(XYZ))

    Y_0 = Y * (
        4.4934 * x**2 + 4.3034 * y**2 - 4.276 * x * y - 1.3744 * x - 2.5643 * y + 1.8103
    )

    o_3 = 1 / 3
    Y_0_es = spow(Y_0, o_3) - 2 / 3
    # Gracefully handles Y_0 < 30.
    Y_0_s = Y_0 - 30
    Lambda = 5.9 * (Y_0_es + 0.042 * spow(Y_0_s, o_3))

    RGB = vecmul(MATRIX_XYZ_TO_RGB_OSA_UCS, XYZ)
    RGB_3 = spow(RGB, 1 / 3)

    with sdiv_mode():
        C = sdiv(Lambda, 5.9 * Y_0_es)

    L = (Lambda - 14.4) / spow(2, 1 / 2)
    j = C * np.dot(RGB_3, np.array([1.7, 8, -9.7]))
    g = C * np.dot(RGB_3, np.array([-13.7, 17.7, -4]))

    Ljg = tstack([L, j, g])

    return from_range_100(Ljg)


def OSA_UCS_to_XYZ(Ljg: Domain100, optimisation_kwargs: dict | None = None) -> Range100:
    """
    Convert from *OSA UCS* colourspace to *CIE XYZ* tristimulus values under
    the *CIE 1964 10 Degree Standard Observer*.

    The lightness axis, *L*, is typically in range [-9, 5] and centered
    around middle gray (Munsell N/6). The yellow-blue axis, *j*, is
    typically in range [-15, 15]. The red-green axis, *g*, is typically in
    range [-20, 15].

    Parameters
    ----------
    Ljg
        *OSA UCS* :math:`Ljg` lightness, jaune (yellowness), and greenness.
    optimisation_kwargs
        Parameters for Newton iteration. Supported parameters:

        -   *iterations_maximum*: Maximum number of iterations (default: 20).
        -   *tolerance*: Convergence tolerance (default: 1e-10).
        -   *epsilon*: Step size for numerical derivative (default: 1e-8).

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ* tristimulus values under the
        *CIE 1964 10 Degree Standard Observer*.

    Notes
    -----
    +------------+-----------------------+--------------------+
    | **Domain** | **Scale - Reference** | **Scale - 1**      |
    +============+=======================+====================+
    | ``Ljg``    | 100                   | 1                  |
    +------------+-----------------------+--------------------+

    +------------+-----------------------+--------------------+
    | **Range**  | **Scale - Reference** | **Scale - 1**      |
    +============+=======================+====================+
    | ``XYZ``    | 100                   | 1                  |
    +------------+-----------------------+--------------------+

    -   *OSA UCS* uses the *CIE 1964 10 Degree Standard Observer*.
    -   This implementation uses the improved algorithm from :cite:`Schlomer2019`
        which employs Cardano's formula for solving the cubic equation and
        Newton's method for the remaining nonlinear system.

    References
    ----------
    :cite:`Cao2013`, :cite:`Moroney2003`, :cite:`Schlomer2019`

    Examples
    --------
    >>> import numpy as np
    >>> Ljg = np.array([-3.00499790, 2.99713697, -9.66784231])
    >>> OSA_UCS_to_XYZ(Ljg)  # doctest: +ELLIPSIS
    array([ 20.654008...,  12.197225...,   5.1369520...])
    """

    Ljg = to_domain_100(Ljg)
    shape = Ljg.shape
    Ljg = np.atleast_1d(np.reshape(Ljg, (-1, 3)))

    # Default optimization settings
    settings: dict[str, typing.Any] = {
        "iterations_maximum": 20,
        "tolerance": 1e-10,
        "epsilon": 1e-8,
    }
    if optimisation_kwargs is not None:
        settings.update(optimisation_kwargs)

    L, j, g = tsplit(Ljg)

    # Step 1: Compute L' from L
    # Forward: L = (Lambda - 14.4) / sqrt(2)
    # Backward: Lambda = L * sqrt(2) + 14.4
    # But L' = Lambda in the intermediate calculation
    sqrt_2 = np.sqrt(2)
    L_prime = L * sqrt_2 + 14.4

    # Step 2: Solve for Y0 using Cardano's formula
    # Equation: 0 = f(t) = (L'/5.9 + 2/3 - t)³ - 0.042^3(t^3 - 30)
    # where t = Y0^(1/3)
    u = L_prime / 5.9 + 2.0 / 3.0
    v = 0.042**3

    # Cubic equation: at^3 + bt^2 + ct + d = 0
    a = -(v + 1)
    b = 3 * u
    c = -3 * u**2
    d = u**3 + 30 * v

    # Convert to depressed cubic: x³ + px + q = 0
    p = (3 * a * c - b**2) / (3 * a**2)
    q = (2 * b**3 - 9 * a * b * c + 27 * a**2 * d) / (27 * a**3)

    # Cardano's formula
    discriminant = (q / 2) ** 2 + (p / 3) ** 3

    with sdiv_mode():
        t = (
            -b / (3 * a)
            + spow(-q / 2 + np.sqrt(discriminant), 1.0 / 3.0)
            + spow(-q / 2 - np.sqrt(discriminant), 1.0 / 3.0)
        )

    Y0 = t**3

    # Step 3: Compute C, a, b
    with sdiv_mode():
        C = sdiv(L_prime, 5.9 * (t - 2.0 / 3.0))
        a_coef = sdiv(g, C)
        b_coef = sdiv(j, C)

    # Step 4: Solve for RGB using Newton iteration
    # Matrix A from equation (4)
    A = np.array([[-13.7, 17.7, -4.0], [1.7, 8.0, -9.7]])

    # Augment A with [1, 0, 0] to make it non-singular (set w = cbrt(R))
    A_augmented = np.vstack([A, [1.0, 0.0, 0.0]])
    A_inv = np.linalg.inv(A_augmented)

    # Initial guess for w (corresponds to cbrt(R))
    # w0 = cbrt(79.9 + 41.94) from paper
    w = np.full_like(L, (79.9 + 41.94) ** (1.0 / 3.0))

    # Newton iteration
    for _iteration in range(settings["iterations_maximum"]):
        # Solve for [cbrt(R), cbrt(G), cbrt(B)] given current w
        ab_w = np.array([a_coef, b_coef, w]).T
        RGB_cbrt = np.dot(ab_w, A_inv.T)

        RGB = RGB_cbrt**3

        XYZ = vecmul(MATRIX_RGB_TO_XYZ_OSA_UCS, RGB)
        X, Y, Z = tsplit(XYZ)

        with sdiv_mode():
            sum_XYZ = X + Y + Z
            x = sdiv(X, sum_XYZ)
            y = sdiv(Y, sum_XYZ)

        K = (
            4.4934 * x**2
            + 4.3034 * y**2
            - 4.276 * x * y
            - 1.3744 * x
            - 2.5643 * y
            + 1.8103
        )
        Y0_computed = Y * K

        error = Y0_computed - Y0
        if np.all(np.abs(error) < settings["tolerance"]):
            break

        # Newton step: compute derivative and update w
        # Derivative is computed numerically for robustness
        epsilon = settings["epsilon"]
        w_plus = w + epsilon

        ab_w_plus = np.array([a_coef, b_coef, w_plus]).T
        RGB_cbrt_plus = np.dot(ab_w_plus, A_inv.T)
        RGB_plus = RGB_cbrt_plus**3
        XYZ_plus = vecmul(MATRIX_RGB_TO_XYZ_OSA_UCS, RGB_plus)
        X_plus, Y_plus, Z_plus = tsplit(XYZ_plus)

        with sdiv_mode():
            sum_XYZ_plus = X_plus + Y_plus + Z_plus
            x_plus = sdiv(X_plus, sum_XYZ_plus)
            y_plus = sdiv(Y_plus, sum_XYZ_plus)

        K_plus = (
            4.4934 * x_plus**2
            + 4.3034 * y_plus**2
            - 4.276 * x_plus * y_plus
            - 1.3744 * x_plus
            - 2.5643 * y_plus
            + 1.8103
        )
        Y0_computed_plus = Y_plus * K_plus

        with sdiv_mode():
            derivative = sdiv(Y0_computed_plus - Y0_computed, epsilon)
            w = w - sdiv(error, derivative)

    return from_range_100(np.reshape(XYZ, shape))
