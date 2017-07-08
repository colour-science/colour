#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
hdr-CIELAB Colourspace
======================

Defines the *hdr-CIELAB* colourspace transformations:

-   :func:`XYZ_to_hdr_CIELab`
-   :func:`hdr_CIELab_to_XYZ`

See Also
--------
`hdr-CIELAB Colourspace Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/hdr_cie_lab.ipynb>`_

References
----------
.. [1]  Fairchild, M. D., & Wyble, D. R. (2010). hdr-CIELAB and hdr-IPT:
        Simple Models for Describing the Color of High-Dynamic-Range and
        Wide-Color-Gamut Images. In Proc. of Color and Imaging Conference
        (pp. 322–326). ISBN:9781629932156
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import (ILLUMINANTS, lightness_Fairchild2010,
                                luminance_Fairchild2010)
from colour.models import xy_to_xyY, xyY_to_XYZ
from colour.utilities import tsplit, tstack

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['XYZ_to_hdr_CIELab', 'hdr_CIELab_to_XYZ', 'exponent_hdr_CIELab']


def XYZ_to_hdr_CIELab(
        XYZ,
        illuminant=ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D50'],
        Y_s=0.2,
        Y_abs=100):
    """
    Converts from *CIE XYZ* tristimulus values to *hdr-CIELAB* colourspace.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values.
    illuminant : array_like, optional
        Reference *illuminant* *xy* chromaticity coordinates or *CIE xyY*
        colourspace array.
    Y_s : numeric or array_like
        Relative luminance :math:`Y_s` of the surround in domain [0, 1].
    Y_abs : numeric or array_like
        Absolute luminance :math:`Y_{abs}` of the scene diffuse white in
        :math:`cd/m^2`.

    Returns
    -------
    ndarray
        *hdr-CIELAB* colourspace array.

    Notes
    -----
    -   Conversion to polar coordinates to compute the *chroma* :math:`C_{hdr}`
        and *hue* :math:`h_{hdr}` correlates can be safely performed with
        :func:`colour.Lab_to_LCHab` definition.
    -   Conversion to cartesian coordinates from the *Lightness*
        :math:`L_{hdr}`, *chroma* :math:`C_{hdr}` and *hue* :math:`h_{hdr}`
        correlates can be safely performed with :func:`colour.LCHab_to_Lab`
        definition.
    -   Input *CIE XYZ* tristimulus values are in domain [0, math:`\infty`].
    -   Input *illuminant* *xy* chromaticity coordinates or *CIE xyY*
        colourspace array are in domain [0, :math:`\infty`].

    Examples
    --------
    >>> XYZ = np.array([0.07049534, 0.10080000, 0.09558313])
    >>> XYZ_to_hdr_CIELab(XYZ)  # doctest: +ELLIPSIS
    array([ 24.9020664..., -46.8312760..., -10.14274843])
    """

    X, Y, Z = tsplit(XYZ)
    X_n, Y_n, Z_n = tsplit(xyY_to_XYZ(xy_to_xyY(illuminant)))

    e = exponent_hdr_CIELab(Y_s, Y_abs)

    L_hdr = lightness_Fairchild2010(Y / Y_n, e)
    a_hdr = 5 * (lightness_Fairchild2010(X / X_n, e) - L_hdr)
    b_hdr = 2 * (L_hdr - lightness_Fairchild2010(Z / Z_n, e))

    Lab_hdr = tstack((L_hdr, a_hdr, b_hdr))

    return Lab_hdr


def hdr_CIELab_to_XYZ(
        Lab_hdr,
        illuminant=ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D50'],
        Y_s=0.2,
        Y_abs=100):
    """
    Converts from *hdr-CIELAB* colourspace to *CIE XYZ* tristimulus values.

    Parameters
    ----------
    Lab_hdr : array_like
        *hdr-CIELAB* colourspace array.
    illuminant : array_like, optional
        Reference *illuminant* *xy* chromaticity coordinates or *CIE xyY*
        colourspace array.
    Y_s : numeric or array_like
        Relative luminance :math:`Y_s` of the surround in domain [0, 1].
    Y_abs : numeric or array_like
        Absolute luminance :math:`Y_{abs}` of the scene diffuse white in
        :math:`cd/m^2`.

    Returns
    -------
    ndarray
        *CIE XYZ* tristimulus values.

    Notes
    -----
    -   Input *illuminant* *xy* chromaticity coordinates or *CIE xyY*
        colourspace array are in domain [0, :math:`\infty`].
    -   Output *CIE XYZ* tristimulus values are in range [0, math:`\infty`].

    Examples
    --------
    >>> Lab_hdr = np.array([24.90206646, -46.83127607, -10.14274843])
    >>> hdr_CIELab_to_XYZ(Lab_hdr)  # doctest: +ELLIPSIS
    array([ 0.0704953...,  0.1008    ,  0.0955831...])
    """

    L_hdr, a_hdr, b_hdr = tsplit(Lab_hdr)
    X_n, Y_n, Z_n = tsplit(xyY_to_XYZ(xy_to_xyY(illuminant)))

    e = exponent_hdr_CIELab(Y_s, Y_abs)

    Y = luminance_Fairchild2010(L_hdr, e) * Y_n
    X = luminance_Fairchild2010((a_hdr + 5 * L_hdr) / 5, e) * X_n
    Z = luminance_Fairchild2010((-b_hdr + 2 * L_hdr) / 2, e) * Z_n

    XYZ = tstack((X, Y, Z))

    return XYZ


def exponent_hdr_CIELab(Y_s, Y_abs):
    """
    Computes *hdr-CIELAB* colourspace *Lightness* :math:`\epsilon` exponent.

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
        *hdr-CIELAB* colourspace *Lightness* :math:`\epsilon` exponent.

    Examples
    --------
    >>> exponent_hdr_CIELab(0.2, 100)  # doctest: +ELLIPSIS
    1.8360198...
    """

    Y_s = np.asarray(Y_s)
    Y_abs = np.asarray(Y_abs)

    lf = np.log(318) / np.log(Y_abs)
    sf = 1.25 - 0.25 * (Y_s / 0.184)
    epsilon = 1.50 * sf * lf

    return epsilon
