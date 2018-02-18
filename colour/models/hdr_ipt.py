# -*- coding: utf-8 -*-
"""
hdr-IPT Colourspace
===================

Defines the *hdr-IPT* colourspace transformations:

-   :func:`colour.XYZ_to_hdr_IPT`
-   :func:`colour.hdr_IPT_to_XYZ`

See Also
--------
`hdr-IPT Colourspace Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/hdr_ipt.ipynb>`_

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
    lightness_Fairchild2010, lightness_Fairchild2011, luminance_Fairchild2010,
    luminance_Fairchild2011)
from colour.models.ipt import (IPT_XYZ_TO_LMS_MATRIX, IPT_LMS_TO_XYZ_MATRIX,
                               IPT_LMS_TO_IPT_MATRIX, IPT_IPT_TO_LMS_MATRIX)
from colour.utilities import dot_vector
from colour.utilities.documentation import DocstringTuple

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'HDR_IPT_METHODS', 'exponent_hdr_IPT', 'XYZ_to_hdr_IPT', 'hdr_IPT_to_XYZ'
]

HDR_IPT_METHODS = DocstringTuple(('Fairchild 2010', 'Fairchild 2011'))
HDR_IPT_METHODS.__doc__ = """
Supported *hdr-IPT* colourspace computation methods.

References
----------
-   :cite:`Fairchild2010`
-   :cite:`Fairchild2011`

HDR_IPT_METHODS : tuple
    **{'Fairchild 2011', 'Fairchild 2010'}**
"""


def exponent_hdr_IPT(Y_s, Y_abs, method='Fairchild 2011'):
    """
    Computes *hdr-IPT* colourspace *Lightness* :math:`\epsilon` exponent using
    *Fairchild and Wyble (2010)* or *Fairchild and Chen (2011)* method.

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
        *hdr-IPT* colourspace *Lightness* :math:`\epsilon` exponent.

    Examples
    --------
    >>> exponent_hdr_IPT(0.2, 100)  # doctest: +ELLIPSIS
    0.7221678...
    >>> exponent_hdr_IPT(0.2, 100, method='Fairchild 2010')
    ... # doctest: +ELLIPSIS
    1.6891383...
    """

    Y_s = np.asarray(Y_s)
    Y_abs = np.asarray(Y_abs)

    method_l = method.lower()
    assert method.lower() in [
        m.lower() for m in HDR_IPT_METHODS
    ], ('"{0}" method is invalid, must be one of {1}!'.format(
        method, HDR_IPT_METHODS))

    if method_l == 'fairchild 2010':
        epsilon = 1.38
    else:
        epsilon = 0.59

    lf = np.log(318) / np.log(Y_abs)
    sf = 1.25 - 0.25 * (Y_s / 0.184)
    epsilon *= sf * lf

    return epsilon


def XYZ_to_hdr_IPT(XYZ, Y_s=0.2, Y_abs=100, method='Fairchild 2011'):
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
    method : unicode, optional
        **{'Fairchild 2011', 'Fairchild 2010'}**,
        Computation method.

    Returns
    -------
    ndarray
        *hdr-IPT* colourspace array.

    Notes
    -----
    -   Input *CIE XYZ* tristimulus values needs to be adapted for
        *CIE Standard Illuminant D Series* *D65*.

    References
    ----------
    -   :cite:`Fairchild2010`
    -   :cite:`Fairchild2011`

    Examples
    --------
    >>> XYZ = np.array([0.96907232, 1.00000000, 1.12179215])
    >>> XYZ_to_hdr_IPT(XYZ)  # doctest: +ELLIPSIS
    array([ 93.5317473...,   1.8564156...,  -1.3292254...])
    >>> XYZ_to_hdr_IPT(XYZ, method='Fairchild 2010')  # doctest: +ELLIPSIS
    array([ 94.6592917...,   0.3804177...,  -0.2673118...])
    """

    method_l = method.lower()
    assert method.lower() in [
        m.lower() for m in HDR_IPT_METHODS
    ], ('"{0}" method is invalid, must be one of {1}!'.format(
        method, HDR_IPT_METHODS))

    if method_l == 'fairchild 2010':
        lightness_callable = lightness_Fairchild2010
    else:
        lightness_callable = lightness_Fairchild2011

    e = exponent_hdr_IPT(Y_s, Y_abs, method)[..., np.newaxis]

    LMS = dot_vector(IPT_XYZ_TO_LMS_MATRIX, XYZ)
    LMS_prime = np.sign(LMS) * np.abs(lightness_callable(LMS, e))
    IPT = dot_vector(IPT_LMS_TO_IPT_MATRIX, LMS_prime)

    return IPT


def hdr_IPT_to_XYZ(IPT_hdr, Y_s=0.2, Y_abs=100, method='Fairchild 2011'):
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
    method : unicode, optional
        **{'Fairchild 2011', 'Fairchild 2010'}**,
        Computation method.

    Returns
    -------
    ndarray
        *CIE XYZ* tristimulus values.

    References
    ----------
    -   :cite:`Fairchild2010`
    -   :cite:`Fairchild2011`

    Examples
    --------
    >>> IPT_hdr = np.array([93.53174734, 1.85641567, -1.32922546])
    >>> hdr_IPT_to_XYZ(IPT_hdr)  # doctest: +ELLIPSIS
    array([ 0.9690723...,  1.        ,  1.1217921...])
    >>> IPT_hdr = np.array([94.65929175, 0.38041773, -0.26731187])
    >>> hdr_IPT_to_XYZ(IPT_hdr, method='Fairchild 2010')
    ... # doctest: +ELLIPSIS
    array([ 0.9690723...,  1.        ,  1.1217921...])
    """

    method_l = method.lower()
    assert method.lower() in [
        m.lower() for m in HDR_IPT_METHODS
    ], ('"{0}" method is invalid, must be one of {1}!'.format(
        method, HDR_IPT_METHODS))

    if method_l == 'fairchild 2010':
        luminance_callable = luminance_Fairchild2010
    else:
        luminance_callable = luminance_Fairchild2011

    e = exponent_hdr_IPT(Y_s, Y_abs, method)[..., np.newaxis]

    LMS = dot_vector(IPT_IPT_TO_LMS_MATRIX, IPT_hdr)
    LMS_prime = np.sign(LMS) * np.abs(luminance_callable(LMS, e))
    XYZ = dot_vector(IPT_LMS_TO_XYZ_MATRIX, LMS_prime)

    return XYZ
