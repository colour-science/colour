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
.. [1]  Fairchild, M. D. (2013). IPT Colourspace. In Color Appearance Models
        (3rd ed., pp. 8492–8567). Wiley. ISBN:B00DAYO8E2
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

IPT_LMS_TO_XYZ_MATRIX = np.linalg.inv(IPT_XYZ_TO_LMS_MATRIX)
"""
*IPT* colourspace normalised cone responses to *CIE XYZ* colourspace matrix.

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

IPT_IPT_TO_LMS_MATRIX = np.linalg.inv(IPT_LMS_TO_IPT_MATRIX)
"""
*IPT* colourspace to *IPT* colourspace normalised cone responses matrix.

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
    >>> XYZ = np.array([0.96907232, 1, 1.12179215])
    >>> XYZ_to_IPT(XYZ)  # doctest: +ELLIPSIS
    array([ 1.0030082...,  0.0190691..., -0.0136929...])
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
    >>> IPT = np.array([1.00300825, 0.01906918, -0.01369292])
    >>> IPT_to_XYZ(IPT)  # doctest: +ELLIPSIS
    array([ 0.9690723...,  1.        ,  1.1217921...])
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
    >>> IPT_hue_angle(([0.96907232, 1, 1.12179215]))  # doctest: +ELLIPSIS
    0.8427358...
    """

    return np.arctan2(IPT[2], IPT[1])
