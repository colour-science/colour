"""
CIE 1994 Chromatic Adaptation Model
===================================

Define the *CIE 1994* chromatic adaptation model for predicting corresponding
colours under different viewing conditions.

-   :func:`colour.adaptation.chromatic_adaptation_CIE1994`

References
----------
-   :cite:`CIETC1-321994b` : CIE TC 1-32. (1994). CIE 109-1994 A Method of
    Predicting Corresponding Colours under Different Chromatic and Illuminance
    Adaptations. Commission Internationale de l'Eclairage.
    ISBN:978-3-900734-51-0
"""

from __future__ import annotations

import typing

import numpy as np

from colour.adaptation import CAT_VON_KRIES
from colour.algebra import sdiv, sdiv_mode, spow, vecmul

if typing.TYPE_CHECKING:
    from colour.hints import DTypeFloat, NDArray

from colour.hints import (  # noqa: TC001
    ArrayLike,
    Domain100,
    NDArrayFloat,
    Range100,
)
from colour.utilities import (
    as_float_array,
    from_range_100,
    to_domain_100,
    tsplit,
    tstack,
    usage_warning,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "MATRIX_XYZ_TO_RGB_CIE1994",
    "MATRIX_RGB_TO_XYZ_CIE1994",
    "chromatic_adaptation_CIE1994",
    "XYZ_to_RGB_CIE1994",
    "RGB_to_XYZ_CIE1994",
    "intermediate_values",
    "effective_adapting_responses",
    "beta_1",
    "beta_2",
    "exponential_factors",
    "K_coefficient",
    "corresponding_colour",
]

MATRIX_XYZ_TO_RGB_CIE1994: NDArrayFloat = CAT_VON_KRIES
"""
*CIE 1994* colour appearance model *CIE XYZ* tristimulus values to cone
responses matrix.
"""

MATRIX_RGB_TO_XYZ_CIE1994: NDArrayFloat = np.linalg.inv(MATRIX_XYZ_TO_RGB_CIE1994)
"""
*CIE 1994* colour appearance model cone responses to *CIE XYZ* tristimulus
values matrix.
"""


def chromatic_adaptation_CIE1994(
    XYZ_1: Domain100,
    xy_o1: ArrayLike,
    xy_o2: ArrayLike,
    Y_o: Domain100,
    E_o1: ArrayLike,
    E_o2: ArrayLike,
    n: ArrayLike = 1,
) -> Range100:
    """
    Adapt the specified stimulus *CIE XYZ* tristimulus values from test
    viewing conditions to reference viewing conditions using the
    *CIE 1994* chromatic adaptation model.

    Parameters
    ----------
    XYZ_1
        *CIE XYZ* tristimulus values of test sample / stimulus.
    xy_o1
        Chromaticity coordinates :math:`x_{o1}` and :math:`y_{o1}` of test
        illuminant and background.
    xy_o2
        Chromaticity coordinates :math:`x_{o2}` and :math:`y_{o2}` of
        reference illuminant and background.
    Y_o
        Luminance factor :math:`Y_o` of achromatic background as percentage
        normalised to domain [18, 100] in **'Reference'** domain-range scale.
    E_o1
        Test illuminance :math:`E_{o1}` in lux.
    E_o2
        Reference illuminance :math:`E_{o2}` in lux.
    n
        Noise component in fundamental primary system.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ* tristimulus values of the stimulus corresponding colour.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ_1``  | 100                   | 1             |
    +------------+-----------------------+---------------+
    | ``Y_o``    | 100                   | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ_2``  | 100                   | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`CIETC1-321994b`

    Examples
    --------
    >>> XYZ_1 = np.array([28.00, 21.26, 5.27])
    >>> xy_o1 = np.array([0.4476, 0.4074])
    >>> xy_o2 = np.array([0.3127, 0.3290])
    >>> Y_o = 20
    >>> E_o1 = 1000
    >>> E_o2 = 1000
    >>> chromatic_adaptation_CIE1994(XYZ_1, xy_o1, xy_o2, Y_o, E_o1, E_o2)
    ... # doctest: +ELLIPSIS
    array([ 24.0337952...,  21.1562121...,  17.6430119...])
    """

    XYZ_1 = to_domain_100(XYZ_1)
    Y_o = as_float_array(to_domain_100(Y_o))
    E_o1 = as_float_array(E_o1)
    E_o2 = as_float_array(E_o2)

    if np.any(Y_o < 18) or np.any(Y_o > 100):
        usage_warning(
            '"Y_o" luminance factor must be in [18, 100] domain, '
            "unpredictable results may occur!"
        )

    RGB_1 = XYZ_to_RGB_CIE1994(XYZ_1)

    xez_1 = intermediate_values(xy_o1)
    xez_2 = intermediate_values(xy_o2)

    RGB_o1 = effective_adapting_responses(xez_1, Y_o, E_o1)
    RGB_o2 = effective_adapting_responses(xez_2, Y_o, E_o2)

    bRGB_o1 = exponential_factors(RGB_o1)
    bRGB_o2 = exponential_factors(RGB_o2)

    K = K_coefficient(xez_1, xez_2, bRGB_o1, bRGB_o2, Y_o, n)

    RGB_2 = corresponding_colour(RGB_1, xez_1, xez_2, bRGB_o1, bRGB_o2, Y_o, K, n)
    XYZ_2 = RGB_to_XYZ_CIE1994(RGB_2)

    return from_range_100(XYZ_2)


def XYZ_to_RGB_CIE1994(XYZ: ArrayLike) -> NDArrayFloat:
    """
    Convert from *CIE XYZ* tristimulus values to cone responses using the
    *CIE 1994* colour appearance model transformation.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values.

    Returns
    -------
    :class:`numpy.ndarray`
        Cone responses.

    Examples
    --------
    >>> XYZ = np.array([28.00, 21.26, 5.27])
    >>> XYZ_to_RGB_CIE1994(XYZ)  # doctest: +ELLIPSIS
    array([ 25.8244273...,  18.6791422...,   4.8390194...])
    """

    return vecmul(MATRIX_XYZ_TO_RGB_CIE1994, XYZ)


def RGB_to_XYZ_CIE1994(RGB: ArrayLike) -> NDArrayFloat:
    """
    Convert from cone responses to *CIE XYZ* tristimulus values using the
    *CIE 1994* colour appearance model inverse transformation.

    Parameters
    ----------
    RGB
        Cone responses.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ* tristimulus values.

    Examples
    --------
    >>> RGB = np.array([25.82442730, 18.67914220, 4.83901940])
    >>> RGB_to_XYZ_CIE1994(RGB)  # doctest: +ELLIPSIS
    array([ 28.  ,  21.26,   5.27])
    """

    return vecmul(MATRIX_RGB_TO_XYZ_CIE1994, RGB)


def intermediate_values(xy_o: ArrayLike) -> NDArrayFloat:
    """
    Compute the intermediate values :math:`\\xi`, :math:`\\eta`, and
    :math:`\\zeta`.

    Parameters
    ----------
    xy_o
        Chromaticity coordinates :math:`x_o` and :math:`y_o` of the
        whitepoint.

    Returns
    -------
    :class:`numpy.ndarray`
        Intermediate values :math:`\\xi`, :math:`\\eta`, and
        :math:`\\zeta`.

    Examples
    --------
    >>> xy_o = np.array([0.4476, 0.4074])
    >>> intermediate_values(xy_o)  # doctest: +ELLIPSIS
    array([ 1.1185719...,  0.9329553...,  0.3268087...])
    """

    x_o, y_o = tsplit(xy_o)

    # Computing :math:`\\xi` :math:`\\eta`, :math:`\\zeta` values.
    xi = (0.48105 * x_o + 0.78841 * y_o - 0.08081) / y_o
    eta = (-0.27200 * x_o + 1.11962 * y_o + 0.04570) / y_o
    zeta = (0.91822 * (1 - x_o - y_o)) / y_o

    return tstack([xi, eta, zeta])


def effective_adapting_responses(
    xez: ArrayLike, Y_o: ArrayLike, E_o: ArrayLike
) -> NDArrayFloat:
    """
    Compute the effective adapting responses in the fundamental primary system
    of the test or reference field.

    Parameters
    ----------
    xez
        Intermediate values :math:`\\xi`, :math:`\\eta`, :math:`\\zeta`.
    Y_o
        Luminance factor :math:`Y_o` of achromatic background as percentage
        normalised to domain [18, 100] in **'Reference'** domain-range scale.
    E_o
        Test or reference illuminance :math:`E_o` in lux.

    Returns
    -------
    :class:`numpy.ndarray`
        Effective adapting responses.

    Examples
    --------
    >>> xez = np.array([1.11857195, 0.93295530, 0.32680879])
    >>> E_o = 1000
    >>> Y_o = 20
    >>> effective_adapting_responses(xez, Y_o, E_o)  # doctest: +ELLIPSIS
    array([ 71.2105020...,  59.3937790...,  20.8052937...])
    """

    xez = as_float_array(xez)
    Y_o = as_float_array(Y_o)
    E_o = as_float_array(E_o)

    return ((Y_o[..., None] * E_o[..., None]) / (100 * np.pi)) * xez


@typing.overload
def beta_1(x: float | DTypeFloat) -> DTypeFloat: ...
@typing.overload
def beta_1(x: NDArray) -> NDArrayFloat: ...
@typing.overload
def beta_1(x: ArrayLike) -> DTypeFloat | NDArrayFloat: ...
def beta_1(x: ArrayLike) -> DTypeFloat | NDArrayFloat:
    """
    Compute the exponent :math:`\\beta_1` for the middle and long-wavelength
    sensitive cones.

    Parameters
    ----------
    x
        Middle and long-wavelength sensitive cone response.

    Returns
    -------
    :class:`numpy.ndarray`
        Exponent :math:`\\beta_1`.

    Examples
    --------
    >>> beta_1(318.323316315)  # doctest: +ELLIPSIS
    4.6106222...
    """

    x_p = spow(x, 0.4495)

    return (x_p * 6.362 + 6.469) / (x_p + 6.469)


@typing.overload
def beta_2(x: float | DTypeFloat) -> DTypeFloat: ...
@typing.overload
def beta_2(x: NDArray) -> NDArrayFloat: ...
@typing.overload
def beta_2(x: ArrayLike) -> DTypeFloat | NDArrayFloat: ...
def beta_2(x: ArrayLike) -> DTypeFloat | NDArrayFloat:
    """
    Compute the exponent :math:`\\beta_2` for the short-wavelength sensitive
    cones.

    Parameters
    ----------
    x
        Short-wavelength sensitive cone response.

    Returns
    -------
    :class:`numpy.ndarray`
        Exponent :math:`\\beta_2`.

    Examples
    --------
    >>> beta_2(318.323316315)  # doctest: +ELLIPSIS
    4.6522416...
    """

    x_p = spow(x, 0.5128)

    return (x_p * 8.091 + 8.414) * 0.7844 / (x_p + 8.414)


def exponential_factors(RGB_o: ArrayLike) -> NDArrayFloat:
    """
    Compute the chromatic adaptation exponential factors
    :math:`\\beta_1(R_o)`, :math:`\\beta_1(G_o)` and :math:`\\beta_2(B_o)` of
    the specified cone responses.

    Parameters
    ----------
    RGB_o
        Cone responses.

    Returns
    -------
    :class:`numpy.ndarray`
        Chromatic adaptation exponential factors :math:`\\beta_1(R_o)`,
        :math:`\\beta_1(G_o)` and :math:`\\beta_2(B_o)`.

    Examples
    --------
    >>> RGB_o = np.array([318.32331631, 318.30352317, 318.23283482])
    >>> exponential_factors(RGB_o)  # doctest: +ELLIPSIS
    array([ 4.6106222...,  4.6105892...,  4.6520698...])
    """

    R_o, G_o, B_o = tsplit(RGB_o)

    bR_o = beta_1(R_o)
    bG_o = beta_1(G_o)
    bB_o = beta_2(B_o)

    return tstack([bR_o, bG_o, bB_o])


def K_coefficient(
    xez_1: ArrayLike,
    xez_2: ArrayLike,
    bRGB_o1: ArrayLike,
    bRGB_o2: ArrayLike,
    Y_o: ArrayLike,
    n: ArrayLike = 1,
) -> NDArrayFloat:
    """
    Compute the coefficient :math:`K` for correcting the difference between
    the test and reference illuminances.

    Parameters
    ----------
    xez_1
        Intermediate values :math:`\\xi_1`, :math:`\\eta_1`, :math:`\\zeta_1`
        for the test illuminant and background.
    xez_2
        Intermediate values :math:`\\xi_2`, :math:`\\eta_2`, :math:`\\zeta_2`
        for the reference illuminant and background.
    bRGB_o1
        Chromatic adaptation exponential factors :math:`\\beta_1(R_{o1})`,
        :math:`\\beta_1(G_{o1})` and :math:`\\beta_2(B_{o1})` of test sample.
    bRGB_o2
        Chromatic adaptation exponential factors :math:`\\beta_1(R_{o2})`,
        :math:`\\beta_1(G_{o2})` and :math:`\\beta_2(B_{o2})` of reference
        sample.
    Y_o
        Luminance factor :math:`Y_o` of achromatic background as percentage
        normalised to domain [18, 100] in **'Reference'** domain-range scale.
    n
        Noise component in fundamental primary system.

    Returns
    -------
    :class:`numpy.ndarray`
        Coefficient :math:`K`.

    Examples
    --------
    >>> xez_1 = np.array([1.11857195, 0.93295530, 0.32680879])
    >>> xez_2 = np.array([1.00000372, 1.00000176, 0.99999461])
    >>> bRGB_o1 = np.array([3.74852518, 3.63920879, 2.78924811])
    >>> bRGB_o2 = np.array([3.68102374, 3.68102256, 3.56557351])
    >>> Y_o = 20
    >>> K_coefficient(xez_1, xez_2, bRGB_o1, bRGB_o2, Y_o)
    1.0
    """

    xi_1, eta_1, _zeta_1 = tsplit(xez_1)
    xi_2, eta_2, _zeta_2 = tsplit(xez_2)
    bR_o1, bG_o1, _bB_o1 = tsplit(bRGB_o1)
    bR_o2, bG_o2, _bB_o2 = tsplit(bRGB_o2)
    Y_o = as_float_array(Y_o)
    n = as_float_array(n)

    K = spow((Y_o * xi_1 + n) / (20 * xi_1 + n), (2 / 3) * bR_o1) / spow(
        (Y_o * xi_2 + n) / (20 * xi_2 + n), (2 / 3) * bR_o2
    )

    K *= spow((Y_o * eta_1 + n) / (20 * eta_1 + n), (1 / 3) * bG_o1) / spow(
        (Y_o * eta_2 + n) / (20 * eta_2 + n), (1 / 3) * bG_o2
    )

    return K


def corresponding_colour(
    RGB_1: ArrayLike,
    xez_1: ArrayLike,
    xez_2: ArrayLike,
    bRGB_o1: ArrayLike,
    bRGB_o2: ArrayLike,
    Y_o: ArrayLike,
    K: ArrayLike,
    n: ArrayLike = 1,
) -> NDArrayFloat:
    """
    Compute corresponding colour cone responses of the specified test sample cone
    responses :math:`RGB_1` under chromatic adaptation.

    Parameters
    ----------
    RGB_1
        Test sample cone responses :math:`RGB_1`.
    xez_1
        Intermediate values :math:`\\xi_1`, :math:`\\eta_1`, :math:`\\zeta_1`
        for test illuminant and background.
    xez_2
        Intermediate values :math:`\\xi_2`, :math:`\\eta_2`, :math:`\\zeta_2`
        for reference illuminant and background.
    bRGB_o1
        Chromatic adaptation exponential factors :math:`\\beta_1(R_{o1})`,
        :math:`\\beta_1(G_{o1})` and :math:`\\beta_2(B_{o1})` of test
        sample.
    bRGB_o2
        Chromatic adaptation exponential factors :math:`\\beta_1(R_{o2})`,
        :math:`\\beta_1(G_{o2})` and :math:`\\beta_2(B_{o2})` of reference
        sample.
    Y_o
        Luminance factor :math:`Y_o` of achromatic background as percentage
        normalised to domain [18, 100] in **'Reference'** domain-range
        scale.
    K
        Coefficient :math:`K`.
    n
        Noise component in fundamental primary system.

    Returns
    -------
    :class:`numpy.ndarray`
        Corresponding colour cone responses of the specified test sample cone
        responses.

    Examples
    --------
    >>> RGB_1 = np.array([25.82442730, 18.67914220, 4.83901940])
    >>> xez_1 = np.array([1.11857195, 0.93295530, 0.32680879])
    >>> xez_2 = np.array([1.00000372, 1.00000176, 0.99999461])
    >>> bRGB_o1 = np.array([3.74852518, 3.63920879, 2.78924811])
    >>> bRGB_o2 = np.array([3.68102374, 3.68102256, 3.56557351])
    >>> Y_o = 20
    >>> K = 1
    >>> corresponding_colour(RGB_1, xez_1, xez_2, bRGB_o1, bRGB_o2, Y_o, K)
    ... # doctest: +ELLIPSIS
    array([ 23.1636901...,  20.0211948...,  16.2001664...])
    """

    R_1, G_1, B_1 = tsplit(RGB_1)
    xi_1, eta_1, zeta_1 = tsplit(xez_1)
    xi_2, eta_2, zeta_2 = tsplit(xez_2)
    bR_o1, bG_o1, bB_o1 = tsplit(bRGB_o1)
    bR_o2, bG_o2, bB_o2 = tsplit(bRGB_o2)
    Y_o = as_float_array(Y_o)
    K = as_float_array(K)
    n = as_float_array(n)

    def RGB_c(
        x_1: NDArrayFloat,
        x_2: NDArrayFloat,
        y_1: NDArrayFloat,
        y_2: NDArrayFloat,
        z: NDArrayFloat,
        n: NDArrayFloat,
    ) -> NDArrayFloat:
        """Compute the corresponding colour cone responses component."""

        with sdiv_mode():
            return (Y_o * x_2 + n) * spow(K, sdiv(1, y_2)) * spow(
                (z + n) / (Y_o * x_1 + n), sdiv(y_1, y_2)
            ) - n

    R_2 = RGB_c(xi_1, xi_2, bR_o1, bR_o2, R_1, n)
    G_2 = RGB_c(eta_1, eta_2, bG_o1, bG_o2, G_1, n)
    B_2 = RGB_c(zeta_1, zeta_2, bB_o1, bB_o2, B_1, n)

    return tstack([R_2, G_2, B_2])
