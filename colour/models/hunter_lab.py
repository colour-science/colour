#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hunter L,a,b Colour Scale
=========================

Defines the *Hunter L,a,b* colour scale transformations:

-   :func:`XYZ_to_K_ab_HunterLab1966`
-   :func:`XYZ_to_Hunter_Lab`
-   :func:`Hunter_Lab_to_XYZ`

See Also
--------
`Hunter L,a,b Colour Scale Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/hunter_lab.ipynb>`_

References
----------
.. [1]  HunterLab. (2008). Hunter L,a,b Color Scale. Retrieved from
        http://www.hunterlab.se/wp-content/uploads/2012/11/Hunter-L-a-b.pdf
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import HUNTERLAB_ILLUMINANTS
from colour.utilities import tsplit, tstack

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['XYZ_to_K_ab_HunterLab1966',
           'XYZ_to_Hunter_Lab',
           'Hunter_Lab_to_XYZ']


def XYZ_to_K_ab_HunterLab1966(XYZ):
    """
    Converts from *whitepoint* *CIE XYZ* tristimulus values to
    *Hunter L,a,b* :math:`K_{a}` and :math:`K_{b}` chromaticity
    coefficients.

    Parameters
    ----------
    XYZ : array_like
        *Whitepoint* *CIE XYZ* tristimulus values.

    Returns
    -------
    ndarray
        *Hunter L,a,b* :math:`K_{a}` and :math:`K_{b}` chromaticity
        coefficients.

    Notes
    -----
    -   Input *CIE XYZ* tristimulus values are in domain [0, 100].

    References
    ----------
    .. [2]  HunterLab. (2008). Illuminant Factors in Universal Software and
            EasyMatch Coatings. Retrieved from
            https://support.hunterlab.com/hc/en-us/article_attachments/\
201437785/an02_02.pdf

    Examples
    --------
    >>> XYZ = np.array([109.850, 100.000, 35.585])
    >>> XYZ_to_K_ab_HunterLab1966(XYZ)  # doctest: +ELLIPSIS
    array([ 185.2378721...,   38.4219142...])
    """

    X, _Y, Z = tsplit(XYZ)

    K_a = 175 * np.sqrt(X / 98.043)
    K_b = 70 * np.sqrt(Z / 118.115)

    K_ab = tstack((K_a, K_b))

    return K_ab


def XYZ_to_Hunter_Lab(
        XYZ,
        XYZ_n=HUNTERLAB_ILLUMINANTS[
            'CIE 1931 2 Degree Standard Observer']['D50'].XYZ_n,
        K_ab=HUNTERLAB_ILLUMINANTS[
            'CIE 1931 2 Degree Standard Observer']['D50'].K_ab):
    """
    Converts from *CIE XYZ* tristimulus values to *Hunter L,a,b* colour scale.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values.
    XYZ_n : array_like, optional
        Reference *illuminant* tristimulus values.
    K_ab : array_like, optional
        Reference *illuminant* chromaticity coefficients, if `K_ab` is set to
        `None` it will be computed using :func:`XYZ_to_K_ab_HunterLab1966`.

    Returns
    -------
    ndarray
        *Hunter L,a,b* colour scale array.

    Notes
    -----
    -   Input *CIE XYZ* and reference *illuminant* tristimulus values are in
        domain [0, 100].
    -   Output *Lightness* :math:`L^*` is in range [0, 100].

    Examples
    --------
    >>> XYZ = np.array([0.07049534, 0.10080000, 0.09558313]) * 100
    >>> D50 = HUNTERLAB_ILLUMINANTS[
    ...     'CIE 1931 2 Degree Standard Observer']['D50']
    >>> XYZ_to_Hunter_Lab(XYZ, D50.XYZ_n, D50.K_ab)   # doctest: +ELLIPSIS
    array([ 31.7490157..., -15.1146262...,  -2.7866075...])
    """

    X, Y, Z = tsplit(XYZ)
    X_n, Y_n, Z_n = tsplit(XYZ_n)
    K_a, K_b = (tsplit(XYZ_to_K_ab_HunterLab1966(XYZ_n))
                if K_ab is None else
                tsplit(K_ab))

    Y_Y_n = Y / Y_n
    sqrt_Y_Y_n = np.sqrt(Y_Y_n)

    L = 100 * sqrt_Y_Y_n
    a = K_a * ((X / X_n - Y_Y_n) / sqrt_Y_Y_n)
    b = K_b * ((Y_Y_n - Z / Z_n) / sqrt_Y_Y_n)

    Lab = tstack((L, a, b))

    return Lab


def Hunter_Lab_to_XYZ(
        Lab,
        XYZ_n=HUNTERLAB_ILLUMINANTS[
            'CIE 1931 2 Degree Standard Observer']['D50'].XYZ_n,
        K_ab=HUNTERLAB_ILLUMINANTS[
            'CIE 1931 2 Degree Standard Observer']['D50'].K_ab):
    """
    Converts from *Hunter L,a,b* colour scale to *CIE XYZ* tristimulus values.

    Parameters
    ----------
    Lab : array_like
        *Hunter L,a,b* colour scale array.
    XYZ_n : array_like, optional
        Reference *illuminant* tristimulus values.
    K_ab : array_like, optional
        Reference *illuminant* chromaticity coefficients, if `K_ab` is set to
        `None` it will be computed using :func:`XYZ_to_K_ab_HunterLab1966`.

    Returns
    -------
    ndarray
        *CIE XYZ* tristimulus values.

    Notes
    -----
    -   Input *Lightness* :math:`L^*` is in domain [0, 100].
    -   Input *CIE XYZ* and reference *illuminant* tristimulus values are in
        domain [0, 100].
    -   Output *CIE XYZ* tristimulus values are in range [0, 100].

    Examples
    --------
    >>> Lab = np.array([31.74901573, -15.11462629, -2.78660758])
    >>> D50 = HUNTERLAB_ILLUMINANTS[
    ...     'CIE 1931 2 Degree Standard Observer']['D50']
    >>> Hunter_Lab_to_XYZ(Lab, D50.XYZ_n, D50.K_ab)   # doctest: +ELLIPSIS
    array([  7.049534,  10.08    ,   9.558313])
    """

    L, a, b = tsplit(Lab)
    X_n, Y_n, Z_n = tsplit(XYZ_n)
    K_a, K_b = (tsplit(XYZ_to_K_ab_HunterLab1966(XYZ_n))
                if K_ab is None else
                tsplit(K_ab))

    L_100 = L / 100
    L_100_2 = L_100 ** 2

    Y = L_100_2 * Y_n
    X = ((a / K_a) * L_100 + L_100_2) * X_n
    Z = -((b / K_b) * L_100 - L_100_2) * Z_n

    XYZ = tstack((X, Y, Z))

    return XYZ
