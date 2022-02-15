"""
Helmholtz—Kohlrausch Effect
===========================

Defines the following methods for estimating Helmholtz-Kohlrausch effect (HKE):

-   :attr:`colour.HKE_NAYATANI1997_METHODS`: Nayatani HKE computation methods,
    choice between variable achromatic colour ('VAC') and variable chromatic
    colour ('VCC').
-   :func:`colour.HelmholtzKohlrausch_effect_object_Nayatani1997`:
    *Nayatani (1997)* HKE estimation for object colours.
-   :func:`colour.HelmholtzKohlrausch_effect_luminous_Nayatani1997`:
    *Nayatani (1997)* HKE estimation for luminous colours.
-   :func:`colour.appearance.coefficient_q_Nayatani1997`:
    Calculates :math:`WI` coefficient for *Nayatani 1997* HKE estimation.
-   :func:`colour.appearance.coefficient_K_Br_Nayatani1997`:
    Calculates :math:`K_{Br}` coefficient for *Nayatani 1997* HKE estimation.

References
----------
-   :cite:`Nayatani1997` : Nayatani, Y. (1997). Simple estimation methods for
    the Helmholtz—Kohlrausch effect. Color Research & Application, 22(6),
    385-401. doi:10.1002/(SICI)1520-6378(199712)22:6<385::AID-COL6>3.0.CO;2-R
"""

from __future__ import annotations

import numpy as np

from colour.algebra import spow
from colour.hints import (
    ArrayLike,
    FloatingOrArrayLike,
    FloatingOrNDArray,
    Literal,
    Union,
)
from colour.utilities import (
    CaseInsensitiveMapping,
    as_float_array,
    tsplit,
    validate_method,
)

__author__ = "Ilia Sibiryakov"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "HKE_NAYATANI1997_METHODS",
    "HelmholtzKohlrausch_effect_object_Nayatani1997",
    "HelmholtzKohlrausch_effect_luminous_Nayatani1997",
    "coefficient_q_Nayatani1997",
    "coefficient_K_Br_Nayatani1997",
]

HKE_NAYATANI1997_METHODS = CaseInsensitiveMapping(
    {
        "VAC": -0.1340,
        "VCC": -0.8660,
    }
)
HKE_NAYATANI1997_METHODS.__doc__ = """
*Nayatani (1997)* *HKE* computation methods, choice between variable achromatic
colour ('VAC') and variable chromatic colour ('VCC')

References
----------
:cite:`Nayatani1997`
"""


def HelmholtzKohlrausch_effect_object_Nayatani1997(
    uv: ArrayLike,
    uv_c: ArrayLike,
    L_a: FloatingOrArrayLike,
    method: Union[Literal["VAC", "VCC"], str] = "VCC",
) -> FloatingOrNDArray:
    """
    Return the *HKE* value for object colours using *Nayatani (1997)* method.

    Parameters
    ----------
    uv
        *CIE uv* chromaticity coordinates of samples.
    uv_c
        *CIE uv* chromaticity coordinates of reference white.
    L_a
        Adapting luminance in :math:`cd/m^2`.
    method
        Which estimation method to use, *VCC* or *VAC*.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Luminance factor (:math:`\\Gamma`) value(s) computed with Nayatani
        object colour estimation method.

    References
    ----------
    :cite:`Nayatani1997`

    Examples
    --------
    >>> import colour
    >>> white = colour.xy_to_Luv_uv(colour.temperature.CCT_to_xy_CIE_D(6504))
    >>> colours = colour.XYZ_to_xy(
    ...     [colour.wavelength_to_XYZ(430 + i * 50) for i in range(5)])
    >>> L_adapting = 65
    >>> HelmholtzKohlrausch_effect_object_Nayatani1997(  # doctest: +ELLIPSIS
    ...     colour.xy_to_Luv_uv(colours), white, L_adapting)
    array([ 2.2468383...,  1.4619799...,  1.1801658...,  0.9031318...,  \
1.7999376...])
    """

    u, v = tsplit(uv)
    u_c, v_c = tsplit(uv_c)

    method = validate_method(method, HKE_NAYATANI1997_METHODS)

    K_Br = coefficient_K_Br_Nayatani1997(L_a)
    q = coefficient_q_Nayatani1997(np.arctan2(v - v_c, u - u_c))
    S_uv = 13 * np.sqrt((u - u_c) ** 2 + (v - v_c) ** 2)

    return 1 + (HKE_NAYATANI1997_METHODS[method] * q + 0.0872 * K_Br) * S_uv


def HelmholtzKohlrausch_effect_luminous_Nayatani1997(
    uv: ArrayLike,
    uv_c: ArrayLike,
    L_a: FloatingOrArrayLike,
    method: Union[Literal["VAC", "VCC"], str] = "VCC",
) -> FloatingOrNDArray:
    """
    Return the *HKE* factor for luminous colours using *Nayatani (1997)* method.

    Parameters
    ----------
    uv
        *CIE uv* chromaticity coordinates of samples.
    uv_c
        *CIE uv* chromaticity coordinates of reference white.
    L_a
        Adapting luminance in :math:`cd/m^2`.
    method
        Which estimation method to use, *VCC* or *VAC*.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Luminance factor (:math:`\\Gamma`) value(s) computed with Nayatani
        luminous colour estimation method.

    References
    ----------
    :cite:`Nayatani1997`

    Examples
    --------
    >>> import colour
    >>> white = colour.xy_to_Luv_uv(colour.temperature.CCT_to_xy_CIE_D(6504))
    >>> colours = colour.XYZ_to_xy(
    ...     [colour.wavelength_to_XYZ(430 + i * 50) for i in range(5)])
    >>> L_adapting = 65
    >>> HelmholtzKohlrausch_effect_luminous_Nayatani1997(  # doctest: +ELLIPSIS
    ...     colour.xy_to_Luv_uv(colours), white, L_adapting)
    array([ 7.4460471...,  2.4767159...,  1.4723422...,  0.7938695...,  \
4.1828629...])
    """

    return (
        0.4462
        * (
            HelmholtzKohlrausch_effect_object_Nayatani1997(
                uv, uv_c, L_a, method
            )
            + 0.3086
        )
        ** 3
    )


def coefficient_q_Nayatani1997(
    theta: FloatingOrArrayLike,
) -> FloatingOrNDArray:
    """
    Return the :math:`q(\\theta)` coefficient for *Nayatani (1997)* *HKE*
    computations.

    The hue angle :math:`\\theta` can be computed as follows:

    :math:`tan^{-1}\\cfrac{v' - v'_c}{u' - u'_c}`

    where :math:`u'` and :math:`v'` are the CIE 1976 chromaticity coordinates
    of the test chromatic light and :math:`u'_c` and :math:`v'_c` are the CIE
    1976 chromaticity coordinates of the reference white light.

    Parameters
    ----------
    theta
        Hue angle (:math:`\\theta`) in radians.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        :math:`q` coefficient for *Nayatani (1997)* *HKE* methods.

    References
    ----------
    :cite:`Nayatani1997`

    Examples
    --------
    This recreates *FIG. A-1*.

    >>> import matplotlib.pyplot as plt
    >>> angles = [(np.pi * 2 / 100 * i) for i in range(100)]
    >>> q_values = coefficient_q_Nayatani1997(angles)
    >>> plt.plot(np.array(angles), q_values / (np.pi * 2) * 180)
    ... # doctest: +ELLIPSIS
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.show()  # doctest: +SKIP
    """

    theta = as_float_array(theta)

    theta_2, theta_3, theta_4 = 2 * theta, 3 * theta, 4 * theta

    return (
        -0.01585
        - 0.03017 * np.cos(theta)
        - 0.04556 * np.cos(theta_2)
        - 0.02667 * np.cos(theta_3)
        - 0.00295 * np.cos(theta_4)
        + 0.14592 * np.sin(theta)
        + 0.05084 * np.sin(theta_2)
        - 0.01900 * np.sin(theta_3)
        - 0.00764 * np.sin(theta_4)
    )


def coefficient_K_Br_Nayatani1997(
    L_a: FloatingOrArrayLike,
) -> FloatingOrNDArray:
    """
    Return the :math:`K_{Br}` coefficient for *Nayatani (1997)* *HKE*
    computations.

    Parameters
    ----------
    L_a
        Adapting luminance in :math:`cd/m^2`.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        :math:`K_{Br}` coefficient for *Nayatani (1997)* *HKE* methods.

    Notes
    -----
    -   The :math:`K_{Br}` coefficient is normalised to unity around
        :math:`63.66cd/m^2`.

    References
    ----------
    :cite:`Nayatani1997`

    Examples
    --------
    >>> L_a_values = [10 + i * 20 for i in range(5)]
    >>> coefficient_K_Br_Nayatani1997(L_a_values)  # doctest: +ELLIPSIS
    array([ 0.7134481...,  0.8781172...,  0.9606248...,  1.0156689...,  \
1.0567008...])
    >>> coefficient_K_Br_Nayatani1997(63.66)  # doctest: +ELLIPSIS
    1.0001284...
    """

    L_a_4495 = spow(L_a, 0.4495)

    return 0.2717 * (6.469 + 6.362 * L_a_4495) / (6.469 + L_a_4495)
