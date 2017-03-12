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
        (3rd ed., pp. 6197–6223). Wiley. ISBN:B00DAYO8E2
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.utilities import dot_vector, tsplit

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
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
*CIE XYZ* tristimulus values to normalised cone responses matrix.

IPT_XYZ_TO_LMS_MATRIX : array_like, (3, 3)
"""

IPT_LMS_TO_XYZ_MATRIX = np.linalg.inv(IPT_XYZ_TO_LMS_MATRIX)
"""
Normalised cone responses to *CIE XYZ* tristimulus values matrix.

IPT_LMS_TO_XYZ_MATRIX : array_like, (3, 3)
"""

IPT_LMS_TO_IPT_MATRIX = np.array([
    [0.4000, 0.4000, 0.2000],
    [4.4550, -4.8510, 0.3960],
    [0.8056, 0.3572, -1.1628]])
"""
Normalised cone responses to *IPT* colourspace matrix.

IPT_LMS_TO_IPT_MATRIX : array_like, (3, 3)
"""

IPT_IPT_TO_LMS_MATRIX = np.linalg.inv(IPT_LMS_TO_IPT_MATRIX)
"""
*IPT* colourspace to normalised cone responses matrix.

IPT_IPT_TO_LMS_MATRIX : array_like, (3, 3)
"""


def XYZ_to_IPT(XYZ):
    """
    Converts from *CIE XYZ* tristimulus values to *IPT* colourspace.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values.

    Returns
    -------
    ndarray
        *IPT* colourspace array.

    Notes
    -----
    -   Input *CIE XYZ* tristimulus values needs to be adapted for
        *CIE Standard Illuminant D Series* *D65*.

    Examples
    --------
    >>> XYZ = np.array([0.96907232, 1.00000000, 1.12179215])
    >>> XYZ_to_IPT(XYZ)  # doctest: +ELLIPSIS
    array([ 1.0030082...,  0.0190691..., -0.0136929...])
    """

    LMS = dot_vector(IPT_XYZ_TO_LMS_MATRIX, XYZ)
    LMS_prime = np.sign(LMS) * np.abs(LMS) ** 0.43
    IPT = dot_vector(IPT_LMS_TO_IPT_MATRIX, LMS_prime)

    return IPT


def IPT_to_XYZ(IPT):
    """
    Converts from *IPT* colourspace to *CIE XYZ* tristimulus values.

    Parameters
    ----------
    IPT : array_like
        *IPT* colourspace array.

    Returns
    -------
    ndarray
        *CIE XYZ* tristimulus values.

    Examples
    --------
    >>> IPT = np.array([1.00300825, 0.01906918, -0.01369292])
    >>> IPT_to_XYZ(IPT)  # doctest: +ELLIPSIS
    array([ 0.9690723...,  1.        ,  1.1217921...])
    """

    LMS = dot_vector(IPT_IPT_TO_LMS_MATRIX, IPT)
    LMS_prime = np.sign(LMS) * np.abs(LMS) ** (1 / 0.43)
    XYZ = dot_vector(IPT_LMS_TO_XYZ_MATRIX, LMS_prime)

    return XYZ


def IPT_hue_angle(IPT):
    """
    Computes the hue angle in degrees from *IPT* colourspace.

    Parameters
    ----------
    IPT : array_like
        *IPT* colourspace array.

    Returns
    -------
    numeric or ndarray
        Hue angle in degrees.

    Examples
    --------
    >>> IPT = np.array([0.96907232, 1, 1.12179215])
    >>> IPT_hue_angle(IPT)  # doctest: +ELLIPSIS
    48.2852074...
    """

    _I, P, T = tsplit(IPT)

    hue = np.degrees(np.arctan2(T, P)) % 360

    return hue
