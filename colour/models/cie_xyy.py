# -*- coding: utf-8 -*-
"""
Tristimulus Values, CIE xyY Colourspace and Chromaticity Coordinates
====================================================================

Defines the *CIE xyY* colourspace transformations:

-   :func:`colour.XYZ_to_xyY`
-   :func:`colour.xyY_to_XYZ`
-   :func:`colour.xyY_to_xy`
-   :func:`colour.xy_to_xyY`
-   :func:`colour.XYZ_to_xy`
-   :func:`colour.xy_to_XYZ`

References
----------
-   :cite:`Lindbloom2003e` : Lindbloom, B. (2003). XYZ to xyY. Retrieved
    February 24, 2014, from http://www.brucelindbloom.com/Eqn_XYZ_to_xyY.html
-   :cite:`Lindbloom2009d` : Lindbloom, B. (2009). xyY to XYZ. Retrieved
    February 24, 2014, from http://www.brucelindbloom.com/Eqn_xyY_to_XYZ.html
-   :cite:`Wikipedia2005` : Wikipedia. (2005). CIE 1931 color space. Retrieved
    February 24, 2014, from http://en.wikipedia.org/wiki/CIE_1931_color_space
"""

from __future__ import annotations

import numpy as np

from colour.colorimetry import CCS_ILLUMINANTS
from colour.hints import ArrayLike, Floating, NDArray
from colour.utilities import (
    as_float_array,
    as_float_scalar,
    from_range_1,
    full,
    to_domain_1,
    tsplit,
    tstack,
    zeros,
)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'XYZ_to_xyY',
    'xyY_to_XYZ',
    'xy_to_xyY',
    'xyY_to_xy',
    'XYZ_to_xy',
    'xy_to_XYZ',
]


def XYZ_to_xyY(XYZ: ArrayLike,
               illuminant: ArrayLike = CCS_ILLUMINANTS[
                   'CIE 1931 2 Degree Standard Observer']['D65']) -> NDArray:
    """
    Converts from *CIE XYZ* tristimulus values to *CIE xyY* colourspace and
    reference *illuminant*.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values.
    illuminant
        Reference *illuminant* chromaticity coordinates.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE xyY* colourspace array.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``xyY``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Lindbloom2003e`, :cite:`Wikipedia2005`

    Examples
    --------
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> XYZ_to_xyY(XYZ)  # doctest: +ELLIPSIS
    array([ 0.5436955...,  0.3210794...,  0.1219722...])
    """

    XYZ = to_domain_1(XYZ)
    X, Y, Z = tsplit(XYZ)
    xy_w = as_float_array(illuminant)

    XYZ_n = zeros(XYZ.shape)
    XYZ_n[..., 0:2] = xy_w

    xyY = np.where(
        np.all(XYZ == 0, axis=-1)[..., np.newaxis],
        XYZ_n,
        tstack([
            X / (X + Y + Z),
            Y / (X + Y + Z),
            from_range_1(Y),
        ]),
    )

    return xyY


def xyY_to_XYZ(xyY: ArrayLike) -> NDArray:
    """
    Converts from *CIE xyY* colourspace to *CIE XYZ* tristimulus values.

    Parameters
    ----------
    xyY
        *CIE xyY* colourspace array.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ* tristimulus values.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``xyY``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Lindbloom2009d`, :cite:`Wikipedia2005`

    Examples
    --------
    >>> xyY = np.array([0.54369557, 0.32107944, 0.12197225])
    >>> xyY_to_XYZ(xyY)  # doctest: +ELLIPSIS
    array([ 0.2065400...,  0.1219722...,  0.0513695...])
    """

    x, y, Y = tsplit(xyY)
    Y = to_domain_1(Y)

    XYZ = np.where(
        (y == 0)[..., np.newaxis],
        tstack([y, y, y]),
        tstack([x * Y / y, Y, (1 - x - y) * Y / y]),
    )

    return from_range_1(XYZ)


def xyY_to_xy(xyY: ArrayLike) -> NDArray:
    """
    Converts from *CIE xyY* colourspace to *CIE xy* chromaticity coordinates.

    ``xyY`` argument with last dimension being equal to 2 will be assumed to be
    a *CIE xy* chromaticity coordinates argument and will be returned directly
    by the definition.

    Parameters
    ----------
    xyY
        *CIE xyY* colourspace array or *CIE xy* chromaticity coordinates.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE xy* chromaticity coordinates.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``xyY``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Wikipedia2005`

    Examples
    --------
    >>> xyY = np.array([0.54369557, 0.32107944, 0.12197225])
    >>> xyY_to_xy(xyY)  # doctest: +ELLIPSIS
    array([ 0.54369557...,  0.32107944...])
    >>> xy = np.array([0.54369557, 0.32107944])
    >>> xyY_to_xy(xy)  # doctest: +ELLIPSIS
    array([ 0.54369557...,  0.32107944...])
    """

    xyY = as_float_array(xyY)

    # Assuming ``xyY`` is actually a *CIE xy* chromaticity coordinates argument
    # and returning it directly.
    if xyY.shape[-1] == 2:
        return xyY

    xy = xyY[..., 0:2]

    return xy


def xy_to_xyY(xy: ArrayLike, Y: Floating = 1) -> NDArray:
    """
    Converts from *CIE xy* chromaticity coordinates to *CIE xyY* colourspace by
    extending the array last dimension with given :math:`Y` *luminance*.

    ``xy`` argument with last dimension being equal to 3 will be assumed to be
    a *CIE xyY* colourspace array argument and will be returned directly by the
    definition.

    Parameters
    ----------
    xy
        *CIE xy* chromaticity coordinates or *CIE xyY* colourspace array.
    Y
        Optional :math:`Y` *luminance* value used to construct the *CIE xyY*
        colourspace array, the default :math:`Y` *luminance* value is 1.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE xyY* colourspace array.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``xy``     | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``xyY``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    -   This definition is a convenient object provided to implement support of
        illuminant argument *luminance* value in various :mod:`colour.models`
        package objects such as :func:`colour.Lab_to_XYZ` or
        :func:`colour.Luv_to_XYZ`.

    References
    ----------
    :cite:`Wikipedia2005`

    Examples
    --------
    >>> xy = np.array([0.54369557, 0.32107944])
    >>> xy_to_xyY(xy)  # doctest: +ELLIPSIS
    array([ 0.5436955...,  0.3210794...,  1.        ])
    >>> xy = np.array([0.54369557, 0.32107944, 1.00000000])
    >>> xy_to_xyY(xy)  # doctest: +ELLIPSIS
    array([ 0.5436955...,  0.3210794...,  1.        ])
    >>> xy = np.array([0.54369557, 0.32107944])
    >>> xy_to_xyY(xy, 100)  # doctest: +ELLIPSIS
    array([   0.5436955...,    0.3210794...,  100.        ])
    """

    xy = as_float_array(xy)
    Y = as_float_scalar(to_domain_1(Y))

    # Assuming ``xy`` is actually a *CIE xyY* colourspace array argument and
    # returning it directly.
    if xy.shape[-1] == 3:
        return xy

    x, y = tsplit(xy)

    xyY = tstack([x, y, full(x.shape, Y)])

    return from_range_1(xyY, np.array([1, 1, 100]))


def XYZ_to_xy(XYZ: ArrayLike,
              illuminant: ArrayLike = CCS_ILLUMINANTS[
                  'CIE 1931 2 Degree Standard Observer']['D65']) -> NDArray:
    """
    Returns the *CIE xy* chromaticity coordinates from given *CIE XYZ*
    tristimulus values.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values.
    illuminant
        Reference *illuminant* chromaticity coordinates.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE xy* chromaticity coordinates.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Wikipedia2005`

    Examples
    --------
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> XYZ_to_xy(XYZ)  # doctest: +ELLIPSIS
    array([ 0.5436955...,  0.3210794...])
    """

    return xyY_to_xy(XYZ_to_xyY(XYZ, illuminant))


def xy_to_XYZ(xy: ArrayLike) -> NDArray:
    """
    Returns the *CIE XYZ* tristimulus values from given *CIE xy* chromaticity
    coordinates.

    Parameters
    ----------
    xy
        *CIE xy* chromaticity coordinates.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ* tristimulus values.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``xy``     | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Wikipedia2005`

    Examples
    --------
    >>> xy = np.array([0.54369557, 0.32107944])
    >>> xy_to_XYZ(xy)  # doctest: +ELLIPSIS
    array([ 1.6933366...,  1.        ,  0.4211574...])
    """

    return xyY_to_XYZ(xy_to_xyY(xy))
