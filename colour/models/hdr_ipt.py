# -*- coding: utf-8 -*-
"""
hdr-IPT Colourspace
===================

Defines the *hdr-IPT* colourspace transformations:

-   :attr:`colour.HDR_IPT_METHODS`: Supported *hdr-IPT* colourspace computation
    methods.
-   :func:`colour.XYZ_to_hdr_IPT`
-   :func:`colour.hdr_IPT_to_XYZ`

References
----------
-   :cite:`Fairchild2010` : Fairchild, M. D., & Wyble, D. R. (2010). hdr-CIELAB
    and hdr-IPT: Simple Models for Describing the Color of High-Dynamic-Range
    and Wide-Color-Gamut Images. Proc. of Color and Imaging Conference,
    322-326. ISBN:978-1-62993-215-6
-   :cite:`Fairchild2011` : Fairchild, M. D., & Chen, P. (2011). Brightness,
    lightness, and specifying color in high-dynamic-range scenes and images. In
    S. P. Farnand & F. Gaykema (Eds.), Proc. SPIE 7867, Image Quality and
    System Performance VIII (p. 78670O). doi:10.1117/12.872075
"""

from __future__ import division, unicode_literals

import colour.ndarray as np

from colour.colorimetry import (
    lightness_Fairchild2010, lightness_Fairchild2011, luminance_Fairchild2010,
    luminance_Fairchild2011)
from colour.models.ipt import (MATRIX_IPT_XYZ_TO_LMS, MATRIX_IPT_LMS_TO_XYZ,
                               MATRIX_IPT_LMS_TO_IPT, MATRIX_IPT_IPT_TO_LMS)
from colour.utilities import (as_float_array, domain_range_scale, from_range_1,
                              from_range_100, to_domain_1, to_domain_100,
                              dot_vector)
from colour.utilities.documentation import (DocstringTuple,
                                            is_documentation_building)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'HDR_IPT_METHODS', 'exponent_hdr_IPT', 'XYZ_to_hdr_IPT', 'hdr_IPT_to_XYZ'
]

HDR_IPT_METHODS = ('Fairchild 2010', 'Fairchild 2011')
if is_documentation_building():  # pragma: no cover
    HDR_IPT_METHODS = DocstringTuple(HDR_IPT_METHODS)
    HDR_IPT_METHODS.__doc__ = """
Supported *hdr-IPT* colourspace computation methods.

References
----------
:cite:`Fairchild2010`, :cite:`Fairchild2011`

HDR_IPT_METHODS : tuple
    **{'Fairchild 2011', 'Fairchild 2010'}**
"""


def exponent_hdr_IPT(Y_s, Y_abs, method='Fairchild 2011'):
    """
    Computes *hdr-IPT* colourspace *Lightness* :math:`\\epsilon` exponent using
    *Fairchild and Wyble (2010)* or *Fairchild and Chen (2011)* method.

    Parameters
    ----------
    Y_s : numeric or array_like
        Relative luminance :math:`Y_s` of the surround.
    Y_abs : numeric or array_like
        Absolute luminance :math:`Y_{abs}` of the scene diffuse white in
        :math:`cd/m^2`.
    method : unicode, optional
        **{'Fairchild 2011', 'Fairchild 2010'}**,
        Computation method.

    Returns
    -------
    array_like
        *hdr-IPT* colourspace *Lightness* :math:`\\epsilon` exponent.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``Y_s``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    Examples
    --------
    >>> exponent_hdr_IPT(0.2, 100)  # doctest: +ELLIPSIS
    0.4820209...
    >>> exponent_hdr_IPT(0.2, 100, method='Fairchild 2010')
    ... # doctest: +ELLIPSIS
    1.6891383...
    """

    Y_s = to_domain_1(Y_s)
    Y_abs = as_float_array(Y_abs)

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
    if method_l == 'fairchild 2010':
        epsilon *= sf * lf
    else:
        epsilon /= sf * lf

    return epsilon


def XYZ_to_hdr_IPT(XYZ, Y_s=0.2, Y_abs=100, method='Fairchild 2011'):
    """
    Converts from *CIE XYZ* tristimulus values to *hdr-IPT* colourspace.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values.
    Y_s : numeric or array_like
        Relative luminance :math:`Y_s` of the surround.
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

    +-------------+-------------------------+---------------------+
    | **Domain**  | **Scale - Reference**   | **Scale - 1**       |
    +=============+=========================+=====================+
    | ``XYZ``     | [0, 1]                  | [0, 1]              |
    +-------------+-------------------------+---------------------+
    | ``Y_s``     | [0, 1]                  | [0, 1]              |
    +-------------+-------------------------+---------------------+

    +-------------+-------------------------+---------------------+
    | **Range**   | **Scale - Reference**   | **Scale - 1**       |
    +=============+=========================+=====================+
    | ``IPT_hdr`` | ``I_hdr`` : [0, 100]    | ``I_hdr`` : [0, 1]  |
    |             |                         |                     |
    |             | ``P_hdr`` : [-100, 100] | ``P_hdr`` : [-1, 1] |
    |             |                         |                     |
    |             | ``T_hdr`` : [-100, 100] | ``T_hdr`` : [-1, 1] |
    +-------------+-------------------------+---------------------+

    -   Input *CIE XYZ* tristimulus values needs to be adapted for
        *CIE Standard Illuminant D Series* *D65*.

    References
    ----------
    :cite:`Fairchild2010`, :cite:`Fairchild2011`

    Examples
    --------
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> XYZ_to_hdr_IPT(XYZ)  # doctest: +ELLIPSIS
    array([ 48.3937634...,  42.4499020...,  22.0195403...])
    >>> XYZ_to_hdr_IPT(XYZ, method='Fairchild 2010')  # doctest: +ELLIPSIS
    array([ 30.0287314...,  83.9384506...,  34.9028738...])
    """

    XYZ = to_domain_1(XYZ)

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

    LMS = dot_vector(MATRIX_IPT_XYZ_TO_LMS, XYZ)

    # Domain and range scaling has already be handled.
    with domain_range_scale('ignore'):
        LMS_prime = np.sign(LMS) * np.abs(lightness_callable(LMS, e))

    IPT_hdr = dot_vector(MATRIX_IPT_LMS_TO_IPT, LMS_prime)

    return from_range_100(IPT_hdr)


def hdr_IPT_to_XYZ(IPT_hdr, Y_s=0.2, Y_abs=100, method='Fairchild 2011'):
    """
    Converts from *hdr-IPT* colourspace to *CIE XYZ* tristimulus values.

    Parameters
    ----------
    IPT_hdr : array_like
        *hdr-IPT* colourspace array.
    Y_s : numeric or array_like
        Relative luminance :math:`Y_s` of the surround.
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

    +-------------+-------------------------+---------------------+
    | **Domain**  | **Scale - Reference**   | **Scale - 1**       |
    +=============+=========================+=====================+
    | ``IPT_hdr`` | ``I_hdr`` : [0, 100]    | ``I_hdr`` : [0, 1]  |
    |             |                         |                     |
    |             | ``P_hdr`` : [-100, 100] | ``P_hdr`` : [-1, 1] |
    |             |                         |                     |
    |             | ``T_hdr`` : [-100, 100] | ``T_hdr`` : [-1, 1] |
    +-------------+-------------------------+---------------------+
    | ``Y_s``     | [0, 1]                  | [0, 1]              |
    +-------------+-------------------------+---------------------+

    +-------------+-------------------------+---------------------+
    | **Range**   | **Scale - Reference**   | **Scale - 1**       |
    +=============+=========================+=====================+
    | ``XYZ``     | [0, 1]                  | [0, 1]              |
    +-------------+-------------------------+---------------------+

    References
    ----------
    :cite:`Fairchild2010`, :cite:`Fairchild2011`

    Examples
    --------
    >>> IPT_hdr = np.array([48.39376346, 42.44990202, 22.01954033])
    >>> hdr_IPT_to_XYZ(IPT_hdr)  # doctest: +ELLIPSIS
    array([ 0.2065400...,  0.1219722...,  0.0513695...])
    >>> IPT_hdr = np.array([30.02873147, 83.93845061, 34.90287382])
    >>> hdr_IPT_to_XYZ(IPT_hdr, method='Fairchild 2010')
    ... # doctest: +ELLIPSIS
    array([ 0.2065400...,  0.1219722...,  0.0513695...])
    """

    IPT_hdr = to_domain_100(IPT_hdr)

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

    LMS = dot_vector(MATRIX_IPT_IPT_TO_LMS, IPT_hdr)

    # Domain and range scaling has already be handled.
    with domain_range_scale('ignore'):
        LMS_prime = np.sign(LMS) * np.abs(luminance_callable(LMS, e))

    XYZ = dot_vector(MATRIX_IPT_LMS_TO_XYZ, LMS_prime)

    return from_range_1(XYZ)
