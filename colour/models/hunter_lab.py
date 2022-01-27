# -*- coding: utf-8 -*-
"""
Hunter L,a,b Colour Scale
=========================

Defines the *Hunter L,a,b* colour scale transformations:

-   :func:`colour.XYZ_to_K_ab_HunterLab1966`
-   :func:`colour.XYZ_to_Hunter_Lab`
-   :func:`colour.Hunter_Lab_to_XYZ`

References
----------
-   :cite:`HunterLab2008b` : HunterLab. (2008). Hunter L,a,b Color Scale.
    http://www.hunterlab.se/wp-content/uploads/2012/11/Hunter-L-a-b.pdf
-   :cite:`HunterLab2008c` : HunterLab. (2008). Illuminant Factors in
    Universal Software and EasyMatch Coatings.
    https://support.hunterlab.com/hc/en-us/article_attachments/201437785/\
an02_02.pdf
"""

from __future__ import annotations

import numpy as np

from colour.colorimetry import TVS_ILLUMINANTS_HUNTERLAB
from colour.hints import ArrayLike, NDArray
from colour.utilities import from_range_100, to_domain_100, tsplit, tstack

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'XYZ_to_K_ab_HunterLab1966',
    'XYZ_to_Hunter_Lab',
    'Hunter_Lab_to_XYZ',
]


def XYZ_to_K_ab_HunterLab1966(XYZ: ArrayLike) -> NDArray:
    """
    Converts from *whitepoint* *CIE XYZ* tristimulus values to
    *Hunter L,a,b* :math:`K_{a}` and :math:`K_{b}` chromaticity
    coefficients.

    Parameters
    ----------
    XYZ
        *Whitepoint* *CIE XYZ* tristimulus values.

    Returns
    -------
    :class:`numpy.ndarray`
        *Hunter L,a,b* :math:`K_{a}` and :math:`K_{b}` chromaticity
        coefficients.

    References
    ----------
    :cite:`HunterLab2008c`

    Examples
    --------
    >>> XYZ = np.array([109.850, 100.000, 35.585])
    >>> XYZ_to_K_ab_HunterLab1966(XYZ)  # doctest: +ELLIPSIS
    array([ 185.2378721...,   38.4219142...])
    """

    X, _Y, Z = tsplit(XYZ)

    K_a = 175 * np.sqrt(X / 98.043)
    K_b = 70 * np.sqrt(Z / 118.115)

    K_ab = tstack([K_a, K_b])

    return K_ab


def XYZ_to_Hunter_Lab(
        XYZ: ArrayLike,
        XYZ_n: ArrayLike = TVS_ILLUMINANTS_HUNTERLAB[
            'CIE 1931 2 Degree Standard Observer']['D65'].XYZ_n,
        K_ab: ArrayLike = TVS_ILLUMINANTS_HUNTERLAB[
            'CIE 1931 2 Degree Standard Observer']['D65'].K_ab) -> NDArray:
    """
    Converts from *CIE XYZ* tristimulus values to *Hunter L,a,b* colour scale.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values.
    XYZ_n
        Reference *illuminant* tristimulus values.
    K_ab
        Reference *illuminant* chromaticity coefficients, if ``K_ab`` is set to
        *None* it will be computed using
        :func:`colour.XYZ_to_K_ab_HunterLab1966`.

    Returns
    -------
    :class:`numpy.ndarray`
        *Hunter L,a,b* colour scale array.

    Notes
    -----

    +------------+-----------------------+-----------------+
    | **Domain** | **Scale - Reference** | **Scale - 1**   |
    +============+=======================+=================+
    | ``XYZ``    | [0, 100]              | [0, 1]          |
    +------------+-----------------------+-----------------+
    | ``XYZ_n``  | [0, 100]              | [0, 1]          |
    +------------+-----------------------+-----------------+

    +------------+-----------------------+-----------------+
    | **Range**  | **Scale - Reference** | **Scale - 1**   |
    +============+=======================+=================+
    | ``Lab``    | ``L`` : [0, 100]      | ``L`` : [0, 1]  |
    |            |                       |                 |
    |            | ``a`` : [-100, 100]   | ``a`` : [-1, 1] |
    |            |                       |                 |
    |            | ``b`` : [-100, 100]   | ``b`` : [-1, 1] |
    +------------+-----------------------+-----------------+

    References
    ----------
    :cite:`HunterLab2008b`

    Examples
    --------
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952]) * 100
    >>> D65 = TVS_ILLUMINANTS_HUNTERLAB[
    ...     'CIE 1931 2 Degree Standard Observer']['D65']
    >>> XYZ_to_Hunter_Lab(XYZ, D65.XYZ_n, D65.K_ab)   # doctest: +ELLIPSIS
    array([ 34.9245257...,  47.0618985...,  14.3861510...])
    """

    X, Y, Z = tsplit(to_domain_100(XYZ))
    X_n, Y_n, Z_n = tsplit(to_domain_100(XYZ_n))
    K_a, K_b = (tsplit(XYZ_to_K_ab_HunterLab1966(XYZ_n))
                if K_ab is None else tsplit(K_ab))

    Y_Y_n = Y / Y_n
    sqrt_Y_Y_n = np.sqrt(Y_Y_n)

    L = 100 * sqrt_Y_Y_n
    a = K_a * ((X / X_n - Y_Y_n) / sqrt_Y_Y_n)
    b = K_b * ((Y_Y_n - Z / Z_n) / sqrt_Y_Y_n)

    Lab = tstack([L, a, b])

    return from_range_100(Lab)


def Hunter_Lab_to_XYZ(
        Lab: ArrayLike,
        XYZ_n: ArrayLike = TVS_ILLUMINANTS_HUNTERLAB[
            'CIE 1931 2 Degree Standard Observer']['D65'].XYZ_n,
        K_ab: ArrayLike = TVS_ILLUMINANTS_HUNTERLAB[
            'CIE 1931 2 Degree Standard Observer']['D65'].K_ab) -> NDArray:
    """
    Converts from *Hunter L,a,b* colour scale to *CIE XYZ* tristimulus values.

    Parameters
    ----------
    Lab
        *Hunter L,a,b* colour scale array.
    XYZ_n
        Reference *illuminant* tristimulus values.
    K_ab
        Reference *illuminant* chromaticity coefficients, if ``K_ab`` is set to
        *None* it will be computed using
        :func:`colour.XYZ_to_K_ab_HunterLab1966`.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ* tristimulus values.

    Notes
    -----

    +------------+-----------------------+-----------------+
    | **Domain** | **Scale - Reference** | **Scale - 1**   |
    +============+=======================+=================+
    | ``Lab``    | ``L`` : [0, 100]      | ``L`` : [0, 1]  |
    |            |                       |                 |
    |            | ``a`` : [-100, 100]   | ``a`` : [-1, 1] |
    |            |                       |                 |
    |            | ``b`` : [-100, 100]   | ``b`` : [-1, 1] |
    +------------+-----------------------+-----------------+
    | ``XYZ_n``  | [0, 100]              | [0, 1]          |
    +------------+-----------------------+-----------------+

    +------------+-----------------------+-----------------+
    | **Range**  | **Scale - Reference** | **Scale - 1**   |
    +============+=======================+=================+
    | ``XYZ``    | [0, 100]              | [0, 1]          |
    +------------+-----------------------+-----------------+

    References
    ----------
    :cite:`HunterLab2008b`

    Examples
    --------
    >>> Lab = np.array([34.92452577, 47.06189858, 14.38615107])
    >>> D65 = TVS_ILLUMINANTS_HUNTERLAB[
    ...     'CIE 1931 2 Degree Standard Observer']['D65']
    >>> Hunter_Lab_to_XYZ(Lab, D65.XYZ_n, D65.K_ab)
    array([ 20.654008,  12.197225,   5.136952])
    """

    L, a, b = tsplit(to_domain_100(Lab))
    X_n, Y_n, Z_n = tsplit(to_domain_100(XYZ_n))
    K_a, K_b = (tsplit(XYZ_to_K_ab_HunterLab1966(XYZ_n))
                if K_ab is None else tsplit(K_ab))

    L_100 = L / 100
    L_100_2 = L_100 ** 2

    Y = L_100_2 * Y_n
    X = ((a / K_a) * L_100 + L_100_2) * X_n
    Z = -((b / K_b) * L_100 - L_100_2) * Z_n

    XYZ = tstack([X, Y, Z])

    return from_range_100(XYZ)
