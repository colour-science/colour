# -*- coding: utf-8 -*-
"""
Hunter Rd,a,b Colour Scale
==========================

Defines the *Hunter Rd,a,b* colour scale transformations:

-   :func:`colour.XYZ_to_Hunter_Rdab`
-   :func:`colour.Hunter_Rdab_to_XYZ`

References
----------
-   :cite:`HunterLab2012a` : HunterLab. (2012). Hunter Rd,a,b Color Scale -
    History and Application.
    https://hunterlabdotcom.files.wordpress.com/2012/07/\
an-1016-hunter-rd-a-b-color-scale-update-12-07-03.pdf
"""

from __future__ import division, unicode_literals

from colour.colorimetry import TVS_ILLUMINANT_HUNTERLAB
from colour.models import XYZ_to_K_ab_HunterLab1966
from colour.utilities import from_range_100, to_domain_100, tsplit, tstack

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['XYZ_to_Hunter_Rdab', 'Hunter_Rdab_to_XYZ']


def XYZ_to_Hunter_Rdab(XYZ,
                       XYZ_n=TVS_ILLUMINANT_HUNTERLAB[
                           'CIE 1931 2 Degree Standard Observer']['D65'].XYZ_n,
                       K_ab=TVS_ILLUMINANT_HUNTERLAB[
                           'CIE 1931 2 Degree Standard Observer']['D65'].K_ab):
    """
    Converts from *CIE XYZ* tristimulus values to *Hunter Rd,a,b* colour scale.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values.
    XYZ_n : array_like, optional
        Reference *illuminant* tristimulus values.
    K_ab : array_like, optional
        Reference *illuminant* chromaticity coefficients, if ``K_ab`` is set to
        *None* it will be computed using
        :func:`colour.XYZ_to_K_ab_HunterLab1966`.

    Returns
    -------
    ndarray
        *Hunter Rd,a,b* colour scale array.

    Notes
    -----

    +------------+------------------------+--------------------+
    | **Domain** | **Scale - Reference**  | **Scale - 1**      |
    +============+========================+====================+
    | ``XYZ``    | [0, 100]               | [0, 1]             |
    +------------+------------------------+--------------------+
    | ``XYZ_n``  | [0, 100]               | [0, 1]             |
    +------------+------------------------+--------------------+

    +------------+------------------------+--------------------+
    | **Range**  | **Scale - Reference**  | **Scale - 1**      |
    +============+========================+====================+
    | ``R_d_ab`` | ``R_d``  : [0, 100]    | ``R_d`` : [0, 1]   |
    |            |                        |                    |
    |            | ``a_Rd`` : [-100, 100] | ``a_Rd`` : [-1, 1] |
    |            |                        |                    |
    |            | ``b_Rd`` : [-100, 100] | ``b_Rd`` : [-1, 1] |
    +------------+------------------------+--------------------+

    References
    ----------
    :cite:`HunterLab2012a`

    Examples
    --------
    >>> import colour.ndarray as np
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952]) * 100
    >>> D65 = TVS_ILLUMINANT_HUNTERLAB[
    ...     'CIE 1931 2 Degree Standard Observer']['D65']
    >>> XYZ_to_Hunter_Rdab(XYZ, D65.XYZ_n, D65.K_ab)
    ... # doctest: +ELLIPSIS
    array([ 12.197225 ...,  57.1253787...,  17.4624134...])
    """

    X, Y, Z = tsplit(to_domain_100(XYZ))
    X_n, Y_n, Z_n = tsplit(to_domain_100(XYZ_n))
    K_a, K_b = (tsplit(XYZ_to_K_ab_HunterLab1966(XYZ_n))
                if K_ab is None else tsplit(K_ab))

    f = 0.51 * ((21 + 0.2 * Y) / (1 + 0.2 * Y))
    Y_Yn = Y / Y_n

    R_d = Y
    a_Rd = K_a * f * (X / X_n - Y_Yn)
    b_Rd = K_b * f * (Y_Yn - Z / Z_n)

    R_d_ab = tstack([R_d, a_Rd, b_Rd])

    return from_range_100(R_d_ab)


def Hunter_Rdab_to_XYZ(R_d_ab,
                       XYZ_n=TVS_ILLUMINANT_HUNTERLAB[
                           'CIE 1931 2 Degree Standard Observer']['D65'].XYZ_n,
                       K_ab=TVS_ILLUMINANT_HUNTERLAB[
                           'CIE 1931 2 Degree Standard Observer']['D65'].K_ab):
    """
    Converts from *Hunter Rd,a,b* colour scale to *CIE XYZ* tristimulus values.

    Parameters
    ----------
    R_d_ab : array_like
        *Hunter Rd,a,b* colour scale array.
    XYZ_n : array_like, optional
        Reference *illuminant* tristimulus values.
    K_ab : array_like, optional
        Reference *illuminant* chromaticity coefficients, if ``K_ab`` is set to
        *None* it will be computed using
        :func:`colour.XYZ_to_K_ab_HunterLab1966`.

    Returns
    -------
    ndarray
        *CIE XYZ* tristimulus values.

    Notes
    -----

    +------------+------------------------+--------------------+
    | **Domain** | **Scale - Reference**  | **Scale - 1**      |
    +============+========================+====================+
    | ``R_d_ab`` | ``R_d``  : [0, 100]    | ``R_d`` : [0, 1]   |
    |            |                        |                    |
    |            | ``a_Rd`` : [-100, 100] | ``a_Rd`` : [-1, 1] |
    |            |                        |                    |
    |            | ``b_Rd`` : [-100, 100] | ``b_Rd`` : [-1, 1] |
    +------------+------------------------+--------------------+
    | ``XYZ_n``  | [0, 100]               | [0, 1]             |
    +------------+------------------------+--------------------+

    +------------+------------------------+--------------------+
    | **Range**  | **Scale - Reference**  | **Scale - 1**      |
    +============+========================+====================+
    | ``XYZ``    | [0, 100]               | [0, 1]             |
    +------------+------------------------+--------------------+

    References
    ----------
    :cite:`HunterLab2012a`

    Examples
    --------
    >>> import colour.ndarray as np
    >>> R_d_ab = np.array([12.19722500, 57.12537874, 17.46241341])
    >>> D65 = TVS_ILLUMINANT_HUNTERLAB[
    ...     'CIE 1931 2 Degree Standard Observer']['D65']
    >>> Hunter_Rdab_to_XYZ(R_d_ab, D65.XYZ_n, D65.K_ab)
    array([ 20.654008,  12.197225,   5.136952])
    """

    R_d, a_Rd, b_Rd = tsplit(to_domain_100(R_d_ab))
    X_n, Y_n, Z_n = tsplit(to_domain_100(XYZ_n))
    K_a, K_b = (tsplit(XYZ_to_K_ab_HunterLab1966(XYZ_n))
                if K_ab is None else tsplit(K_ab))

    f = 0.51 * ((21 + 0.2 * R_d) / (1 + 0.2 * R_d))
    Rd_Yn = R_d / Y_n
    X = (a_Rd / (K_a * f) + Rd_Yn) * X_n
    Z = -(b_Rd / (K_b * f) - Rd_Yn) * Z_n

    XYZ = tstack([X, R_d, Z])

    return from_range_100(XYZ)
