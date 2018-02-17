# -*- coding: utf-8 -*-
"""
hdr-CIELAB Colourspace
======================

Defines the *hdr-CIELAB* colourspace transformations:

-   :func:`colour.XYZ_to_hdr_CIELab`
-   :func:`colour.hdr_CIELab_to_XYZ`

See Also
--------
`hdr-CIELAB Colourspace Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/hdr_cie_lab.ipynb>`_

References
----------
-   :cite:`Fairchild2010` : Fairchild, M. D., & Wyble, D. R. (2010).
    hdr-CIELAB and hdr-IPT: Simple Models for Describing the Color of
    High-Dynamic-Range and Wide-Color-Gamut Images. In Proc. of Color and
    Imaging Conference (pp. 322-326). ISBN:9781629932156
-   :cite:`Fairchild2011` : Fairchild, M. D., & Chen, P. (2011). Brightness,
    lightness, and specifying color in high-dynamic-range scenes and images.
    In S. P. Farnand & F. Gaykema (Eds.), Proc. SPIE 7867, Image Quality and
    System Performance VIII (p. 78670O). doi:10.1117/12.872075
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import (
    ILLUMINANTS, lightness_Fairchild2010, lightness_Fairchild2011,
    luminance_Fairchild2010, luminance_Fairchild2011)
from colour.models import xy_to_xyY, xyY_to_XYZ
from colour.utilities import tsplit, tstack
from colour.utilities.documentation import DocstringTuple

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'HDR_CIELAB_METHODS', 'exponent_hdr_CIELab', 'XYZ_to_hdr_CIELab',
    'hdr_CIELab_to_XYZ'
]

HDR_CIELAB_METHODS = DocstringTuple(('Fairchild 2010', 'Fairchild 2011'))
HDR_CIELAB_METHODS.__doc__ = """
Supported *hdr-CIELAB* colourspace computation methods.

References
----------
-   :cite:`Fairchild2010`
-   :cite:`Fairchild2011`

HDR_CIELAB_METHODS : tuple
    **{'Fairchild 2011', 'Fairchild 2010'}**
"""


def exponent_hdr_CIELab(Y_s, Y_abs, method='Fairchild 2011'):
    """
    Computes *hdr-CIELAB* colourspace *Lightness* :math:`\epsilon` exponent
    using *Fairchild and Wyble (2010)* or *Fairchild and Chen (2011)* method.

    Parameters
    ----------
    Y_s : numeric or array_like
        Relative luminance :math:`Y_s` of the surround in range [0, 1].
    Y_abs : numeric or array_like
        Absolute luminance :math:`Y_{abs}` of the scene diffuse white in
        :math:`cd/m^2`.
    method : unicode, optional
        **{'Fairchild 2011', 'Fairchild 2010'}**,
        Computation method.

    Returns
    -------
    array_like
        *hdr-CIELAB* colourspace *Lightness* :math:`\epsilon` exponent.

    Examples
    --------
    >>> exponent_hdr_CIELab(0.2, 100)  # doctest: +ELLIPSIS
    0.7099276...
    >>> exponent_hdr_CIELab(0.2, 100, method='Fairchild 2010')
    ... # doctest: +ELLIPSIS
    1.8360198...
    """

    Y_s = np.asarray(Y_s)
    Y_abs = np.asarray(Y_abs)

    method_l = method.lower()
    assert method.lower() in [
        m.lower() for m in HDR_CIELAB_METHODS
    ], ('"{0}" method is invalid, must be one of {1}!'.format(
        method, HDR_CIELAB_METHODS))

    if method_l == 'fairchild 2010':
        epsilon = 1.50
    else:
        epsilon = 0.58

    sf = 1.25 - 0.25 * (Y_s / 0.184)
    lf = np.log(318) / np.log(Y_abs)
    epsilon *= sf * lf

    return epsilon


def XYZ_to_hdr_CIELab(
        XYZ,
        illuminant=ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D50'],
        Y_s=0.2,
        Y_abs=100,
        method='Fairchild 2011'):
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
    method : unicode, optional
        **{'Fairchild 2011', 'Fairchild 2010'}**,
        Computation method.

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

    References
    ----------
    -   :cite:`Fairchild2010`
    -   :cite:`Fairchild2011`

    Examples
    --------
    >>> XYZ = np.array([0.07049534, 0.10080000, 0.09558313])
    >>> XYZ_to_hdr_CIELab(XYZ)  # doctest: +ELLIPSIS
    array([ 26.4646106..., -24.613326 ...,  -4.8479681...])
    >>> XYZ_to_hdr_CIELab(XYZ, method='Fairchild 2010')  # doctest: +ELLIPSIS
    array([ 24.9020664..., -46.8312760..., -10.1427484...])
    """

    X, Y, Z = tsplit(XYZ)
    X_n, Y_n, Z_n = tsplit(xyY_to_XYZ(xy_to_xyY(illuminant)))

    method_l = method.lower()
    assert method.lower() in [
        m.lower() for m in HDR_CIELAB_METHODS
    ], ('"{0}" method is invalid, must be one of {1}!'.format(
        method, HDR_CIELAB_METHODS))

    if method_l == 'fairchild 2010':
        lightness_callable = lightness_Fairchild2010
    else:
        lightness_callable = lightness_Fairchild2011

    e = exponent_hdr_CIELab(Y_s, Y_abs, method)

    L_hdr = lightness_callable(Y / Y_n, e)
    a_hdr = 5 * (lightness_callable(X / X_n, e) - L_hdr)
    b_hdr = 2 * (L_hdr - lightness_callable(Z / Z_n, e))

    Lab_hdr = tstack((L_hdr, a_hdr, b_hdr))

    return Lab_hdr


def hdr_CIELab_to_XYZ(
        Lab_hdr,
        illuminant=ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D50'],
        Y_s=0.2,
        Y_abs=100,
        method='Fairchild 2011'):
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
    method : unicode, optional
        **{'Fairchild 2011', 'Fairchild 2010'}**,
        Computation method.

    Returns
    -------
    ndarray
        *CIE XYZ* tristimulus values.

    Notes
    -----
    -   Input *illuminant* *xy* chromaticity coordinates or *CIE xyY*
        colourspace array are in domain [0, :math:`\infty`].
    -   Output *CIE XYZ* tristimulus values are in range [0, math:`\infty`].

    References
    ----------
    -   :cite:`Fairchild2010`
    -   :cite:`Fairchild2011`

    Examples
    --------
    >>> Lab_hdr = np.array([26.46461067, -24.613326, -4.84796811])
    >>> hdr_CIELab_to_XYZ(Lab_hdr)  # doctest: +ELLIPSIS
    array([ 0.0704953...,  0.1008    ,  0.0955831...])
    >>> Lab_hdr = np.array([24.90206646, -46.83127607, -10.14274843])
    >>> hdr_CIELab_to_XYZ(Lab_hdr, method='Fairchild 2010')
    ... # doctest: +ELLIPSIS
    array([ 0.0704953...,  0.1008    ,  0.0955831...])
    """

    L_hdr, a_hdr, b_hdr = tsplit(Lab_hdr)
    X_n, Y_n, Z_n = tsplit(xyY_to_XYZ(xy_to_xyY(illuminant)))

    method_l = method.lower()
    assert method.lower() in [
        m.lower() for m in HDR_CIELAB_METHODS
    ], ('"{0}" method is invalid, must be one of {1}!'.format(
        method, HDR_CIELAB_METHODS))

    if method_l == 'fairchild 2010':
        luminance_callable = luminance_Fairchild2010
    else:
        luminance_callable = luminance_Fairchild2011

    e = exponent_hdr_CIELab(Y_s, Y_abs, method)

    Y = luminance_callable(L_hdr, e) * Y_n
    X = luminance_callable((a_hdr + 5 * L_hdr) / 5, e) * X_n
    Z = luminance_callable((-b_hdr + 2 * L_hdr) / 2, e) * Z_n

    XYZ = tstack((X, Y, Z))

    return XYZ
