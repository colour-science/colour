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
-   :cite:`CIETC1-482004h` : CIE TC 1-48. (2004). CIE 015:2004 Colorimetry,
    3rd Edition. In CIE 015:2004 Colorimetry, 3rd Edition. Commission
    Internationale de l'Eclairage. ISBN:978-3-901906-33-6
-   :cite:`Wikipedia2005` : Wikipedia. (2005). CIE 1931 color space. Retrieved
    February 24, 2014, from http://en.wikipedia.org/wiki/CIE_1931_color_space
"""

from __future__ import annotations

import numpy as np

from colour.hints import ArrayLike, NDArrayFloat
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

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "XYZ_to_xyY",
    "xyY_to_XYZ",
    "xy_to_xyY",
    "xyY_to_xy",
    "XYZ_to_xy",
    "xy_to_XYZ",
]


def XYZ_to_xyY(XYZ: ArrayLike) -> NDArrayFloat:
    """
    Convert from *CIE XYZ* tristimulus values to *CIE xyY* colourspace.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values.

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
    :cite:`CIETC1-482004h`, :cite:`Wikipedia2005`

    Examples
    --------
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> XYZ_to_xyY(XYZ)  # doctest: +ELLIPSIS
    array([ 0.5436955...,  0.3210794...,  0.1219722...])
    """

    XYZ = to_domain_1(XYZ)

    X, Y, Z = tsplit(XYZ)

    xyY = zeros(XYZ.shape)
    xyY[..., 0:2] = 0

    m_xyY = ~np.all(XYZ == 0, axis=-1)
    X_Y_Z = (X + Y + Z)[m_xyY]

    xyY[m_xyY] = (
        tstack(
            [
                X[m_xyY] / X_Y_Z,
                Y[m_xyY] / X_Y_Z,
                from_range_1(Y[m_xyY]),
            ]
        ),
    )

    return xyY


def xyY_to_XYZ(xyY: ArrayLike) -> NDArrayFloat:
    """
    Convert from *CIE xyY* colourspace to *CIE XYZ* tristimulus values.

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
    :cite:`CIETC1-482004h`, :cite:`Wikipedia2005`

    Examples
    --------
    >>> xyY = np.array([0.54369557, 0.32107944, 0.12197225])
    >>> xyY_to_XYZ(xyY)  # doctest: +ELLIPSIS
    array([ 0.2065400...,  0.1219722...,  0.0513695...])
    """

    xyY = as_float_array(xyY)

    x, y, Y = tsplit(xyY)
    Y = to_domain_1(Y)

    XYZ = zeros(xyY.shape)
    m_XYZ = ~(y == 0)

    Y_y = Y[m_XYZ] / y[m_XYZ]

    XYZ[m_XYZ] = tstack(
        [x[m_XYZ] * Y_y, Y[m_XYZ], (1 - x[m_XYZ] - y[m_XYZ]) * Y_y]
    )

    return from_range_1(XYZ)


def xyY_to_xy(xyY: ArrayLike) -> NDArrayFloat:
    """
    Convert from *CIE xyY* colourspace to *CIE xy* chromaticity coordinates.

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
    :cite:`CIETC1-482004h`, :cite:`Wikipedia2005`

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


def xy_to_xyY(xy: ArrayLike, Y: float = 1) -> NDArrayFloat:
    """
    Convert from *CIE xy* chromaticity coordinates to *CIE xyY* colourspace by
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
    :cite:`CIETC1-482004h`, :cite:`Wikipedia2005`

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


def XYZ_to_xy(XYZ: ArrayLike) -> NDArrayFloat:
    """
    Return the *CIE xy* chromaticity coordinates from given *CIE XYZ*
    tristimulus values.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values.

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
    :cite:`CIETC1-482004h`, :cite:`Wikipedia2005`

    Examples
    --------
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> XYZ_to_xy(XYZ)  # doctest: +ELLIPSIS
    array([ 0.5436955...,  0.3210794...])
    """

    return xyY_to_xy(XYZ_to_xyY(XYZ))


def xy_to_XYZ(xy: ArrayLike) -> NDArrayFloat:
    """
    Return the *CIE XYZ* tristimulus values from given *CIE xy* chromaticity
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
    :cite:`CIETC1-482004h`, :cite:`Wikipedia2005`

    Examples
    --------
    >>> xy = np.array([0.54369557, 0.32107944])
    >>> xy_to_XYZ(xy)  # doctest: +ELLIPSIS
    array([ 1.6933366...,  1.        ,  0.4211574...])
    """

    return xyY_to_XYZ(xy_to_xyY(xy))
