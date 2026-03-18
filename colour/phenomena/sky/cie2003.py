"""
CIE Standard General Sky
========================

Implement the *CIE Standard General Sky* luminance distribution model as
defined in *CIE S 011/E:2003* and *ISO 15469:2004(E)*.

-   :func:`colour.phenomena.sky_luminance_gradation_CIE2003`
-   :func:`colour.phenomena.sky_scattering_indicatrix_CIE2003`
-   :func:`colour.phenomena.sky_luminance_distribution_CIE2003`

References
----------
-   :cite:`CIE2003` : CIE. (2003). International Organization for
    Standardization. (2004). INTERNATIONAL STANDARD ISO 15469 CIE S 011/E -
    Spatial distribution of daylight - CIE standard general sky.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass

import numpy as np

if typing.TYPE_CHECKING:
    from colour.hints import ArrayLike, NDArrayFloat

from colour.utilities import MixinDataclassIterable, as_float, as_float_array

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "SkyType_CIE2003",
    "CIE_STANDARD_SKY_PARAMETERS",
    "sky_luminance_gradation_CIE2003",
    "sky_scattering_indicatrix_CIE2003",
    "sky_luminance_distribution_CIE2003",
    "sky_luminance_distribution_overcast_CIE2003",
]


@dataclass(frozen=True)
class SkyType_CIE2003(MixinDataclassIterable):
    """
    Define parameters for a *CIE Standard General Sky* type.

    Parameters
    ----------
    gradation_group
        Gradation function group (I-VI).
    indicatrix_group
        Indicatrix function group (1-6).
    a
        Gradation parameter :math:`a`.
    b
        Gradation parameter :math:`b`.
    c
        Indicatrix parameter :math:`c`.
    d
        Indicatrix parameter :math:`d`.
    e
        Indicatrix parameter :math:`e`.
    description
        Description of luminance distribution.

    References
    ----------
    :cite:`CIE2003`
    """

    gradation_group: int
    indicatrix_group: int
    a: float
    b: float
    c: float
    d: float
    e: float
    description: str


CIE_STANDARD_SKY_PARAMETERS: dict[int, SkyType_CIE2003] = {
    1: SkyType_CIE2003(
        1,
        1,
        4.0,
        -0.70,
        0,
        -1.0,
        0,
        "CIE Standard Overcast Sky, "
        "steep luminance gradation towards zenith, azimuthal uniformity",
    ),
    2: SkyType_CIE2003(
        1,
        2,
        4.0,
        -0.70,
        2,
        -1.5,
        0.15,
        "Overcast, with steep luminance gradation "
        "and slight brightening towards the sun",
    ),
    3: SkyType_CIE2003(
        2,
        1,
        1.1,
        -0.8,
        0,
        -1.0,
        0,
        "Overcast, moderately graded with azimuthal uniformity",
    ),
    4: SkyType_CIE2003(
        2,
        2,
        1.1,
        -0.8,
        2,
        -1.5,
        0.15,
        "Overcast, moderately graded and slight brightening towards the sun",
    ),
    5: SkyType_CIE2003(
        3,
        1,
        0,
        -1.0,
        0,
        -1.0,
        0,
        "Sky of uniform luminance",
    ),
    6: SkyType_CIE2003(
        3,
        2,
        0,
        -1.0,
        2,
        -1.5,
        0.15,
        "Partly cloudy sky, no gradation towards zenith, "
        "slight brightening towards the sun",
    ),
    7: SkyType_CIE2003(
        3,
        3,
        0,
        -1.0,
        5,
        -2.5,
        0.30,
        "Partly cloudy sky, no gradation towards zenith, brighter circumsolar region",
    ),
    8: SkyType_CIE2003(
        3,
        4,
        0,
        -1.0,
        10,
        -3.0,
        0.45,
        "Partly cloudy sky, no gradation towards zenith, distinct solar corona",
    ),
    9: SkyType_CIE2003(
        4,
        2,
        -1.0,
        -0.55,
        2,
        -1.5,
        0.15,
        "Partly cloudy, with the obscured sun",
    ),
    10: SkyType_CIE2003(
        4,
        3,
        -1.0,
        -0.55,
        5,
        -2.5,
        0.30,
        "Partly cloudy, with brighter circumsolar region",
    ),
    11: SkyType_CIE2003(
        4,
        4,
        -1.0,
        -0.55,
        10,
        -3.0,
        0.45,
        "White-blue sky with distinct solar corona",
    ),
    12: SkyType_CIE2003(
        5,
        4,
        -1.0,
        -0.32,
        10,
        -3.0,
        0.45,
        "CIE Standard Clear Sky, low luminance turbidity",
    ),
    13: SkyType_CIE2003(
        5,
        5,
        -1.0,
        -0.32,
        16,
        -3.0,
        0.30,
        "CIE Standard Clear Sky, polluted atmosphere",
    ),
    14: SkyType_CIE2003(
        6,
        5,
        -1.0,
        -0.15,
        16,
        -3.0,
        0.30,
        "Cloudless turbid sky with broad solar corona",
    ),
    15: SkyType_CIE2003(
        6,
        6,
        -1.0,
        -0.15,
        24,
        -2.8,
        0.15,
        "White-blue turbid sky with broad solar corona",
    ),
}
"""
*CIE Standard General Sky* parameters for 15 sky types.

References
----------
:cite:`CIE2003`
"""


def sky_luminance_gradation_CIE2003(
    Z: ArrayLike,
    a: ArrayLike,
    b: ArrayLike,
) -> NDArrayFloat:
    """
    Compute the *CIE Standard General Sky* luminance gradation function
    :math:`\\varphi(Z)` at the given zenith angle :math:`Z`.

    Parameters
    ----------
    Z
        Zenith angle :math:`Z` in radians.
    a
        Gradation parameter :math:`a`.
    b
        Gradation parameter :math:`b`.

    Returns
    -------
    :class:`numpy.ndarray`
        Luminance gradation :math:`\\varphi(Z)`.

    References
    ----------
    :cite:`CIE2003`

    Examples
    --------
    >>> sky_luminance_gradation_CIE2003(0, 4.0, -0.70)  # doctest: +ELLIPSIS
    np.float64(2.9863412...)
    >>> sky_luminance_gradation_CIE2003(  # doctest: +ELLIPSIS
    ...     np.pi / 2, 4.0, -0.70
    ... )
    np.float64(1.0)
    """

    Z = as_float_array(Z)
    a = as_float_array(a)
    b = as_float_array(b)

    phi = np.where(
        np.pi / 2 > Z,
        1 + a * np.exp(b / np.cos(Z)),
        1.0,
    )

    return as_float(phi)


def sky_scattering_indicatrix_CIE2003(
    chi: ArrayLike,
    c: ArrayLike,
    d: ArrayLike,
    e: ArrayLike,
) -> NDArrayFloat:
    """
    Compute the *CIE Standard General Sky* scattering indicatrix function
    :math:`f(\\chi)` at the given angular distance :math:`\\chi` between a sky
    element and the sun.

    Parameters
    ----------
    chi
        Angular distance :math:`\\chi` between the sky element and the sun
        in radians.
    c
        Indicatrix parameter :math:`c`.
    d
        Indicatrix parameter :math:`d`.
    e
        Indicatrix parameter :math:`e`.

    Returns
    -------
    :class:`numpy.ndarray`
        Scattering indicatrix :math:`f(\\chi)`.

    References
    ----------
    :cite:`CIE2003`

    Examples
    --------
    >>> sky_scattering_indicatrix_CIE2003(  # doctest: +ELLIPSIS
    ...     np.radians(30), 10, -3.0, 0.45
    ... )
    np.float64(3.3264628...)
    """

    chi = as_float_array(chi)
    c = as_float_array(c)
    d = as_float_array(d)
    e = as_float_array(e)

    f = 1 + c * (np.exp(d * chi) - np.exp(d * np.pi / 2)) + e * np.cos(chi) ** 2

    return as_float(f)


def sky_luminance_distribution_CIE2003(
    sky_type: int,
    Z: ArrayLike,
    alpha: ArrayLike,
    Z_s: ArrayLike,
    alpha_s: ArrayLike,
) -> NDArrayFloat:
    """
    Compute the relative sky luminance :math:`L_{\\alpha} / L_z` for the given
    *CIE Standard General Sky* type.

    The relative luminance at any point is the ratio of the luminance of an
    arbitrary sky element to the zenith luminance :math:`L_z`:

    :math:`L_{\\alpha} / L_z = f(\\chi) \\cdot \\varphi(Z) / \
(f(Z_s) \\cdot \\varphi(0))`

    Parameters
    ----------
    sky_type
        *CIE Standard General Sky* type (1-15).
    Z
        Zenith angle :math:`Z` of the sky element in radians.
    alpha
        Azimuth angle :math:`\\alpha` of the sky element in radians
        (clockwise from north).
    Z_s
        Zenith angle :math:`Z_s` of the sun in radians.
    alpha_s
        Azimuth angle :math:`\\alpha_s` of the sun in radians
        (clockwise from north).

    Returns
    -------
    :class:`numpy.ndarray`
        Relative sky luminance :math:`L_{\\alpha} / L_z`.

    Raises
    ------
    ValueError
        If ``sky_type`` is not in range [1, 15].

    References
    ----------
    :cite:`CIE2003`

    Examples
    --------
    >>> sky_luminance_distribution_CIE2003(  # doctest: +ELLIPSIS
    ...     1, np.radians(45), np.radians(180), np.radians(30), np.radians(0)
    ... )
    np.float64(0.8325846...)
    >>> sky_luminance_distribution_CIE2003(  # doctest: +ELLIPSIS
    ...     12, np.radians(45), np.radians(180), np.radians(30), np.radians(0)
    ... )
    np.float64(0.4544524...)
    """

    if sky_type not in CIE_STANDARD_SKY_PARAMETERS:
        error = f'"sky_type" must be in range [1, 15], got "{sky_type}".'
        raise ValueError(error)

    Z = as_float_array(Z)
    alpha = as_float_array(alpha)
    Z_s = as_float_array(Z_s)
    alpha_s = as_float_array(alpha_s)

    parameters = CIE_STANDARD_SKY_PARAMETERS[sky_type]
    a = parameters.a
    b = parameters.b
    c = parameters.c
    d = parameters.d
    e = parameters.e

    # Angular distance between sky element and the sun (Equation 1).
    chi = np.arccos(
        np.cos(Z_s) * np.cos(Z)
        + np.sin(Z_s) * np.sin(Z) * np.cos(np.abs(alpha - alpha_s))
    )

    # Relative luminance (Equations 3-7).
    f_chi = sky_scattering_indicatrix_CIE2003(chi, c, d, e)
    phi_Z = sky_luminance_gradation_CIE2003(Z, a, b)

    f_Z_s = sky_scattering_indicatrix_CIE2003(Z_s, c, d, e)
    phi_0 = sky_luminance_gradation_CIE2003(0, a, b)

    L_ratio = (f_chi * phi_Z) / (f_Z_s * phi_0)

    return as_float(L_ratio)


def sky_luminance_distribution_overcast_CIE2003(
    Z: ArrayLike,
) -> NDArrayFloat:
    """
    Compute the relative sky luminance :math:`L_{\\alpha} / L_z` for the
    *CIE Traditional Overcast Sky* (the 16th sky type).

    :math:`L_{\\alpha}(\\gamma) / L_{zenith} = (1 + 2 \\sin \\gamma) / 3`

    where :math:`\\gamma` is the angle of elevation of the sky element above
    the horizon.

    Parameters
    ----------
    Z
        Zenith angle :math:`Z` of the sky element in radians.

    Returns
    -------
    :class:`numpy.ndarray`
        Relative sky luminance :math:`L_{\\alpha} / L_z`.

    References
    ----------
    :cite:`CIE2003`

    Examples
    --------
    >>> sky_luminance_distribution_overcast_CIE2003(0)  # doctest: +ELLIPSIS
    np.float64(1.0)
    >>> sky_luminance_distribution_overcast_CIE2003(  # doctest: +ELLIPSIS
    ...     np.pi / 2
    ... )
    np.float64(0.3333333...)
    """

    Z = as_float_array(Z)

    gamma = np.pi / 2 - Z

    return as_float((1 + 2 * np.sin(gamma)) / 3)
