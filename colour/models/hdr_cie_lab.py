# -*- coding: utf-8 -*-
"""
hdr-CIELAB Colourspace
======================

Defines the *hdr-CIELAB* colourspace transformations:

-   :attr:`colour.HDR_CIELAB_METHODS`: Supported *hdr-CIELAB* colourspace
    computation methods.
-   :func:`colour.XYZ_to_hdr_CIELab`
-   :func:`colour.hdr_CIELab_to_XYZ`

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

from __future__ import annotations

import numpy as np

from colour.colorimetry import (
    CCS_ILLUMINANTS,
    lightness_Fairchild2010,
    lightness_Fairchild2011,
    luminance_Fairchild2010,
    luminance_Fairchild2011,
)
from colour.hints import (
    ArrayLike,
    FloatingOrArrayLike,
    FloatingOrNDArray,
    Literal,
    NDArray,
    Tuple,
    Union,
)
from colour.models import xy_to_xyY, xyY_to_XYZ
from colour.utilities import (
    as_float_array,
    domain_range_scale,
    from_range_1,
    from_range_100,
    to_domain_1,
    to_domain_100,
    tsplit,
    tstack,
    validate_method,
)
from colour.utilities.documentation import (
    DocstringTuple,
    is_documentation_building,
)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'HDR_CIELAB_METHODS',
    'exponent_hdr_CIELab',
    'XYZ_to_hdr_CIELab',
    'hdr_CIELab_to_XYZ',
]

HDR_CIELAB_METHODS: Tuple = ('Fairchild 2010', 'Fairchild 2011')
if is_documentation_building():  # pragma: no cover
    HDR_CIELAB_METHODS = DocstringTuple(HDR_CIELAB_METHODS)
    HDR_CIELAB_METHODS.__doc__ = """
Supported *hdr-CIELAB* colourspace computation methods.

References
----------
:cite:`Fairchild2010`, :cite:`Fairchild2011`
"""


def exponent_hdr_CIELab(
        Y_s: FloatingOrArrayLike,
        Y_abs: FloatingOrArrayLike,
        method: Union[Literal['Fairchild 2011', 'Fairchild 2010'],
                      str] = 'Fairchild 2011') -> FloatingOrNDArray:
    """
    Computes *hdr-CIELAB* colourspace *Lightness* :math:`\\epsilon` exponent
    using *Fairchild and Wyble (2010)* or *Fairchild and Chen (2011)* method.

    Parameters
    ----------
    Y_s
        Relative luminance :math:`Y_s` of the surround.
    Y_abs
        Absolute luminance :math:`Y_{abs}` of the scene diffuse white in
        :math:`cd/m^2`.
    method
        Computation method.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        *hdr-CIELAB* colourspace *Lightness* :math:`\\epsilon` exponent.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``Y_s``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    Examples
    --------
    >>> exponent_hdr_CIELab(0.2, 100)  # doctest: +ELLIPSIS
    0.4738510...
    >>> exponent_hdr_CIELab(0.2, 100, method='Fairchild 2010')
    ... # doctest: +ELLIPSIS
    1.8360198...
    """

    Y_s = to_domain_1(Y_s)
    Y_abs = as_float_array(Y_abs)
    method = validate_method(method, HDR_CIELAB_METHODS)

    if method == 'fairchild 2010':
        epsilon = 1.50
    else:
        epsilon = 0.58

    sf = 1.25 - 0.25 * (Y_s / 0.184)
    lf = np.log(318) / np.log(Y_abs)
    if method == 'fairchild 2010':
        epsilon *= sf * lf
    else:
        epsilon /= sf * lf

    return epsilon


def XYZ_to_hdr_CIELab(
        XYZ: ArrayLike,
        illuminant: ArrayLike = CCS_ILLUMINANTS[
            'CIE 1931 2 Degree Standard Observer']['D65'],
        Y_s: FloatingOrArrayLike = 0.2,
        Y_abs: FloatingOrArrayLike = 100,
        method: Union[Literal['Fairchild 2011', 'Fairchild 2010'],
                      str] = 'Fairchild 2011') -> NDArray:
    """
    Converts from *CIE XYZ* tristimulus values to *hdr-CIELAB* colourspace.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values.
    illuminant
        Reference *illuminant* *CIE xy* chromaticity coordinates or *CIE xyY*
        colourspace array.
    Y_s
        Relative luminance :math:`Y_s` of the surround.
    Y_abs
        Absolute luminance :math:`Y_{abs}` of the scene diffuse white in
        :math:`cd/m^2`.
    method
        Computation method.

    Returns
    -------
    :class:`numpy.ndarray`
        *hdr-CIELAB* colourspace array.

    Notes
    -----

    +----------------+-------------------------+---------------------+
    | **Domain**     | **Scale - Reference**   | **Scale - 1**       |
    +================+=========================+=====================+
    | ``XYZ``        | [0, 1]                  | [0, 1]              |
    +----------------+-------------------------+---------------------+
    | ``illuminant`` | [0, 1]                  | [0, 1]              |
    +----------------+-------------------------+---------------------+
    | ``Y_s``        | [0, 1]                  | [0, 1]              |
    +----------------+-------------------------+---------------------+

    +----------------+-------------------------+---------------------+
    | **Range**      | **Scale - Reference**   | **Scale - 1**       |
    +================+=========================+=====================+
    | ``Lab_hdr``    | ``L_hdr`` : [0, 100]    | ``L_hdr`` : [0, 1]  |
    |                |                         |                     |
    |                | ``a_hdr`` : [-100, 100] | ``a_hdr`` : [-1, 1] |
    |                |                         |                     |
    |                | ``b_hdr`` : [-100, 100] | ``b_hdr`` : [-1, 1] |
    +----------------+-------------------------+---------------------+

    -   Conversion to polar coordinates to compute the *chroma* :math:`C_{hdr}`
        and *hue* :math:`h_{hdr}` correlates can be safely performed with
        :func:`colour.Lab_to_LCHab` definition.
    -   Conversion to cartesian coordinates from the *Lightness*
        :math:`L_{hdr}`, *chroma* :math:`C_{hdr}` and *hue* :math:`h_{hdr}`
        correlates can be safely performed with :func:`colour.LCHab_to_Lab`
        definition.

    References
    ----------
    :cite:`Fairchild2010`, :cite:`Fairchild2011`

    Examples
    --------
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> XYZ_to_hdr_CIELab(XYZ)  # doctest: +ELLIPSIS
    array([ 51.8700206...,  60.4763385...,  32.1455191...])
    >>> XYZ_to_hdr_CIELab(XYZ, method='Fairchild 2010')  # doctest: +ELLIPSIS
    array([  31.9962111...,  128.0076303...,   48.7695230...])
    """

    X, Y, Z = tsplit(to_domain_1(XYZ))
    method = validate_method(method, HDR_CIELAB_METHODS)

    X_n, Y_n, Z_n = tsplit(xyY_to_XYZ(xy_to_xyY(illuminant)))

    if method == 'fairchild 2010':
        lightness_callable = lightness_Fairchild2010
    else:
        lightness_callable = lightness_Fairchild2011

    e = exponent_hdr_CIELab(Y_s, Y_abs, method)

    # Domain and range scaling has already be handled.
    with domain_range_scale('ignore'):
        L_hdr = lightness_callable(Y / Y_n, e)
        a_hdr = 5 * (lightness_callable(X / X_n, e) - L_hdr)
        b_hdr = 2 * (L_hdr - lightness_callable(Z / Z_n, e))

    Lab_hdr = tstack([L_hdr, a_hdr, b_hdr])

    return from_range_100(Lab_hdr)


def hdr_CIELab_to_XYZ(
        Lab_hdr: ArrayLike,
        illuminant: ArrayLike = CCS_ILLUMINANTS[
            'CIE 1931 2 Degree Standard Observer']['D65'],
        Y_s: FloatingOrArrayLike = 0.2,
        Y_abs: FloatingOrArrayLike = 100,
        method: Union[Literal['Fairchild 2011', 'Fairchild 2010'],
                      str] = 'Fairchild 2011') -> NDArray:
    """
    Converts from *hdr-CIELAB* colourspace to *CIE XYZ* tristimulus values.

    Parameters
    ----------
    Lab_hdr
        *hdr-CIELAB* colourspace array.
    illuminant
        Reference *illuminant* *CIE xy* chromaticity coordinates or *CIE xyY*
        colourspace array.
    Y_s
        Relative luminance :math:`Y_s` of the surround.
    Y_abs
        Absolute luminance :math:`Y_{abs}` of the scene diffuse white in
        :math:`cd/m^2`.
    method
        Computation method.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ* tristimulus values.

    Notes
    -----

    +----------------+-------------------------+---------------------+
    | **Domain**     | **Scale - Reference**   | **Scale - 1**       |
    +================+=========================+=====================+
    | ``Lab_hdr``    | ``L_hdr`` : [0, 100]    | ``L_hdr`` : [0, 1]  |
    |                |                         |                     |
    |                | ``a_hdr`` : [-100, 100] | ``a_hdr`` : [-1, 1] |
    |                |                         |                     |
    |                | ``b_hdr`` : [-100, 100] | ``b_hdr`` : [-1, 1] |
    +----------------+-------------------------+---------------------+
    | ``illuminant`` | [0, 1]                  | [0, 1]              |
    +----------------+-------------------------+---------------------+
    | ``Y_s``        | [0, 1]                  | [0, 1]              |
    +----------------+-------------------------+---------------------+

    +----------------+-------------------------+---------------------+
    | **Range**      | **Scale - Reference**   | **Scale - 1**       |
    +================+=========================+=====================+
    | ``XYZ``        | [0, 1]                  | [0, 1]              |
    +----------------+-------------------------+---------------------+

    References
    ----------
    :cite:`Fairchild2010`, :cite:`Fairchild2011`

    Examples
    --------
    >>> Lab_hdr = np.array([51.87002062, 60.4763385, 32.14551912])
    >>> hdr_CIELab_to_XYZ(Lab_hdr)  # doctest: +ELLIPSIS
    array([ 0.2065400...,  0.1219722...,  0.0513695...])
    >>> Lab_hdr = np.array([31.99621114, 128.00763036, 48.76952309])
    >>> hdr_CIELab_to_XYZ(Lab_hdr, method='Fairchild 2010')
    ... # doctest: +ELLIPSIS
    array([ 0.2065400...,  0.1219722...,  0.0513695...])
    """

    L_hdr, a_hdr, b_hdr = tsplit(to_domain_100(Lab_hdr))
    method = validate_method(method, HDR_CIELAB_METHODS)

    X_n, Y_n, Z_n = tsplit(xyY_to_XYZ(xy_to_xyY(illuminant)))

    if method == 'fairchild 2010':
        luminance_callable = luminance_Fairchild2010
    else:
        luminance_callable = luminance_Fairchild2011

    e = exponent_hdr_CIELab(Y_s, Y_abs, method)

    # Domain and range scaling has already been handled.
    with domain_range_scale('ignore'):
        Y = luminance_callable(L_hdr, e) * Y_n
        X = luminance_callable((a_hdr + 5 * L_hdr) / 5, e) * X_n
        Z = luminance_callable((-b_hdr + 2 * L_hdr) / 2, e) * Z_n

    XYZ = tstack([X, Y, Z])

    return from_range_1(XYZ)
