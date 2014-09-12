#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
IPT Colourspace
===============

Defines the *IPT* colourspace transformations:

-   :func:`XYZ_to_IPT`
-   :func:`IPT_to_XYZ`

And computation of correlates:

-   :func:`IPT_hue_angle`

References
----------
.. [1]  **Mark D. Fairchild**, *Color Appearance Models, 3nd Edition*,
        The Wiley-IS&T Series in Imaging Science and Technology,
        published June 2013, ASIN: B00DAYO8E2, locations 8492-8567
"""

from __future__ import division, unicode_literals

import numpy as np

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['IPT_XYZ_TO_LMS_MATRIX',
           'IPT_LMS_TO_XYZ_MATRIX',
           'IPT_LMS_TO_IPT_MATRIX',
           'IPT_IPT_TO_LMS_MATRIX',
           'XYZ_to_IPT',
           'IPT_to_XYZ',
           'IPT_hue_angle']

IPT_XYZ_TO_LMS_MATRIX = np.array([
    [0.4002, 0.7075, -0.0807],
    [-0.2280, 1.1500, 0.0612],
    [0.0000, 0.0000, 0.9184]])
"""
*CIE XYZ* colourspace to *IPT* colourspace normalised cone responses matrix.

IPT_XYZ_TO_LMS_MATRIX : array_like, (3, 3)
"""

IPT_LMS_TO_XYZ_MATRIX = np.around(
    np.linalg.inv(IPT_XYZ_TO_LMS_MATRIX),
    decimals=4)
"""
*IPT* colourspace normalised cone responses to *CIE XYZ* colourspace matrix.

Notes
-----
-   This matrix has been rounded on purpose to 4 decimals so in order to keep
    available precision consistent between transformations.

IPT_LMS_TO_XYZ_MATRIX : array_like, (3, 3)
"""

IPT_LMS_TO_IPT_MATRIX = np.array([
    [0.4000, 0.4000, 0.2000],
    [4.4550, -4.8510, 0.3960],
    [0.8056, 0.3572, -1.1628]])
"""
*IPT* colourspace normalised cone responses to *IPT* colourspace matrix.

IPT_LMS_TO_IPT_MATRIX : array_like, (3, 3)
"""

IPT_IPT_TO_LMS_MATRIX = np.around(
    np.linalg.inv(IPT_LMS_TO_IPT_MATRIX),
    decimals=4)
"""
*IPT* colourspace to *IPT* colourspace normalised cone responses matrix.

Notes
-----
-   This matrix has been rounded on purpose to 4 decimals so in order to keep
    available precision consistent between transformations.

IPT_IPT_TO_LMS_MATRIX : array_like, (3, 3)
"""


def XYZ_to_IPT(XYZ):
    """
    Converts from *CIE XYZ* colourspace to *IPT* colourspace. [1]_

    Parameters
    ----------
    XYZ : array_like, (3,)
        *CIE XYZ* colourspace matrix.

    Returns
    -------
    ndarray, (3,)
        *IPT* colourspace matrix.

    Notes
    -----
    -   Input *CIE XYZ* colourspace matrix needs to be adapted for
        *CIE Standard Illuminant D Series* *D65*.

    Examples
    --------
    >>> XYZ = np.array([0.07049534, 0.1008, 0.09558313])
    >>> XYZ_to_IPT(XYZ)  # doctest: +ELLIPSIS
    array([ 0.3657112..., -0.1111479...,  0.0159474...])
    """

    LMS = np.dot(IPT_XYZ_TO_LMS_MATRIX, XYZ)
    LMS_prime = np.sign(LMS) * np.abs(LMS) ** 0.43
    IPT = np.dot(IPT_LMS_TO_IPT_MATRIX, LMS_prime)

    return IPT


def IPT_to_XYZ(IPT):
    """
    Converts from *IPT* colourspace to *CIE XYZ* colourspace. [1]_

    Parameters
    ----------
    IPT : array_like, (3,)
        *IPT* colourspace matrix.

    Returns
    -------
    ndarray, (3,)
        *CIE XYZ* colourspace matrix.

     Examples
    --------
    >>> IPT = np.array([0.36571124, -0.11114798, 0.01594746])
    >>> IPT_to_XYZ(IPT)  # doctest: +ELLIPSIS
    array([ 0.0704827...,  0.1007966...,  0.0955884...])
    """

    LMS = np.dot(IPT_IPT_TO_LMS_MATRIX, IPT)
    LMS_prime = np.sign(LMS) * np.abs(LMS) ** (1 / 0.43)
    XYZ = np.dot(IPT_LMS_TO_XYZ_MATRIX, LMS_prime)

    return XYZ


def IPT_hue_angle(IPT):
    """
    Computes the hue angle from *IPT* colourspace. [1]_

    Parameters
    ----------
    IPT : array_like, (3,)
        *IPT* colourspace matrix.

    Returns
    -------
    numeric
        Hue angle.

    Examples
    --------
    >>> IPT = np.array([0.07049534, 0.1008, 0.09558313])
    >>> IPT_hue_angle(IPT)  # doctest: +ELLIPSIS
    0.7588396...
    """

    return np.arctan2(IPT[2], IPT[1])
