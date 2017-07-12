#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
hdr-IPT Colourspace
===================

Defines the *hdr-IPT* colourspace transformations:

-   :func:`XYZ_to_hdr_IPT`
-   :func:`hdr_IPT_to_XYZ`

See Also
--------
`hdr-IPT Colourspace Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/hdr_IPT.ipynb>`_

References
----------
.. [1]  Fairchild, M. D., & Wyble, D. R. (2010). hdr-CIELAB and hdr-IPT:
        Simple Models for Describing the Color of High-Dynamic-Range and
        Wide-Color-Gamut Images. In Proc. of Color and Imaging Conference
        (pp. 322–326). ISBN:9781629932156
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import lightness_Fairchild2010, luminance_Fairchild2010
from colour.models.ipt import (IPT_XYZ_TO_LMS_MATRIX, IPT_LMS_TO_XYZ_MATRIX,
                               IPT_LMS_TO_IPT_MATRIX, IPT_IPT_TO_LMS_MATRIX)
from colour.utilities import dot_vector

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['XYZ_to_hdr_IPT', 'hdr_IPT_to_XYZ', 'exponent_hdr_IPT']


def XYZ_to_hdr_IPT(XYZ, Y_s=0.2, Y_abs=100):
    """
    Converts from *CIE XYZ* tristimulus values to *hdr-IPT* colourspace.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values.
    Y_s : numeric or array_like
        Relative luminance :math:`Y_s` of the surround in domain [0, 1].
    Y_abs : numeric or array_like
        Absolute luminance :math:`Y_{abs}` of the scene diffuse white in
        :math:`cd/m^2`.

    Returns
    -------
    ndarray
        *hdr-IPT* colourspace array.

    Notes
    -----
    -   Input *CIE XYZ* tristimulus values needs to be adapted for
        *CIE Standard Illuminant D Series* *D65*.

    Examples
    --------
    >>> XYZ = np.array([0.96907232, 1.00000000, 1.12179215])
    >>> XYZ_to_hdr_IPT(XYZ)  # doctest: +ELLIPSIS
    array([ 94.6592917...,   0.3804177...,  -0.2673118...])
    """

    e = exponent_hdr_IPT(Y_s, Y_abs)[..., np.newaxis]

    LMS = dot_vector(IPT_XYZ_TO_LMS_MATRIX, XYZ)
    LMS_prime = np.sign(LMS) * np.abs(lightness_Fairchild2010(LMS, e))
    IPT = dot_vector(IPT_LMS_TO_IPT_MATRIX, LMS_prime)

    return IPT


def hdr_IPT_to_XYZ(IPT_hdr, Y_s=0.2, Y_abs=100):
    """
    Converts from *hdr-IPT* colourspace to *CIE XYZ* tristimulus values.

    Parameters
    ----------
    IPT_hdr : array_like
        *hdr-IPT* colourspace array.
    Y_s : numeric or array_like
        Relative luminance :math:`Y_s` of the surround in domain [0, 1].
    Y_abs : numeric or array_like
        Absolute luminance :math:`Y_{abs}` of the scene diffuse white in
        :math:`cd/m^2`.

    Returns
    -------
    ndarray
        *CIE XYZ* tristimulus values.

    Examples
    --------
    >>> IPT_hdr = np.array([94.65929175, 0.38041773, -0.26731187])
    >>> hdr_IPT_to_XYZ(IPT_hdr)  # doctest: +ELLIPSIS
    array([ 0.9690723...,  1.        ,  1.1217921...])
    """

    e = exponent_hdr_IPT(Y_s, Y_abs)[..., np.newaxis]

    LMS = dot_vector(IPT_IPT_TO_LMS_MATRIX, IPT_hdr)
    LMS_prime = np.sign(LMS) * np.abs(luminance_Fairchild2010(LMS, e))
    XYZ = dot_vector(IPT_LMS_TO_XYZ_MATRIX, LMS_prime)

    return XYZ


def exponent_hdr_IPT(Y_s, Y_abs):
    """
    Computes *hdr-IPT* colourspace *Lightness* :math:`\epsilon` exponent.

    Parameters
    ----------
    Y_s : numeric or array_like
        Relative luminance :math:`Y_s` of the surround in range [0, 1].
    Y_abs : numeric or array_like
        Absolute luminance :math:`Y_{abs}` of the scene diffuse white in
        :math:`cd/m^2`.

    Returns
    -------
    array_like
        *hdr-IPT* colourspace *Lightness* :math:`\epsilon` exponent.

    Examples
    --------
    >>> exponent_hdr_IPT(0.2, 100)  # doctest: +ELLIPSIS
    1.6891383...
    """

    Y_s = np.asarray(Y_s)
    Y_abs = np.asarray(Y_abs)

    lf = np.log(318) / np.log(Y_abs)
    sf = 1.25 - 0.25 * (Y_s / 0.184)
    epsilon = 1.38 * sf * lf

    return epsilon
