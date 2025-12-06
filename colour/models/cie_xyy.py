"""
Tristimulus Values, CIE xyY Colourspace and Chromaticity Coordinates
====================================================================

Define the *CIE xyY* colourspace transformations.

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

from colour.algebra import sdiv, sdiv_mode
from colour.hints import (  # noqa: TC001
    ArrayLike,
    Domain1,
    NDArrayFloat,
    Range1,
)
from colour.utilities import as_float_array, from_range_1, to_domain_1, tsplit, tstack

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
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


def XYZ_to_xyY(XYZ: Domain1) -> Range1:
    """
    Convert from *CIE XYZ* tristimulus values to *CIE xyY* colourspace.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE xyY* colourspace array with chromaticity coordinates
        :math:`x, y` and luminance :math:`Y`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | 1                     | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``xyY``    | 1                     | 1             |
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

    X_Y_Z = X + Y + Z

    with sdiv_mode():
        x = sdiv(X, X_Y_Z)
        y = sdiv(Y, X_Y_Z)

    return tstack([x, y, from_range_1(Y)])


def xyY_to_XYZ(xyY: Domain1) -> Range1:
    """
    Convert from *CIE xyY* colourspace to *CIE XYZ* tristimulus values.

    Parameters
    ----------
    xyY
        *CIE xyY* colourspace array with chromaticity coordinates
        :math:`x, y` and luminance :math:`Y`.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ* tristimulus values.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``xyY``    | 1                     | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | 1                     | 1             |
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

    with sdiv_mode():
        Y_y = sdiv(Y, y)

    XYZ = tstack([x * Y_y, Y, (1 - (x + y)) * Y_y])

    return from_range_1(XYZ)


def xyY_to_xy(xyY: Domain1) -> NDArrayFloat:
    """
    Convert from *CIE xyY* colourspace to *CIE xy* chromaticity
    coordinates.

    ``xyY`` argument with last dimension being equal to 2 will be
    assumed to be a *CIE xy* chromaticity coordinates argument and will
    be returned directly by the definition.

    Parameters
    ----------
    xyY
        *CIE xyY* colourspace array or *CIE xy* chromaticity
        coordinates.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE xy* chromaticity coordinates.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``xyY``    | 1                     | 1             |
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

    return xyY[..., 0:2]


def xy_to_xyY(xy: ArrayLike, Y: Domain1 = 1) -> Range1:
    """
    Convert from *CIE xy* chromaticity coordinates to *CIE xyY*
    colourspace by extending the array's last dimension with the
    specified :math:`Y` *luminance*.

    ``xy`` argument with last dimension equal to 3 will be assumed
    to be a *CIE xyY* colourspace array and will be returned
    directly by the definition.

    Parameters
    ----------
    xy
        *CIE xy* chromaticity coordinates or *CIE xyY* colourspace
        array.
    Y
        Optional :math:`Y` *luminance* value used to construct the
        *CIE xyY* colourspace array. The default :math:`Y`
        *luminance* value is 1.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE xyY* colourspace array.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``xy``     | 1                     | 1             |
    +------------+-----------------------+---------------+
    | ``Y``      | 1                     | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``xyY``    | 1                     | 1             |
    +------------+-----------------------+---------------+

    -   This definition is a convenient object provided to
        implement support of illuminant argument luminance value
        in various :mod:`colour.models` package objects such as
        :func:`colour.Lab_to_XYZ` or :func:`colour.Luv_to_XYZ`.

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
    Y = as_float_array(to_domain_1(Y))

    # Assuming ``xy`` is actually a *CIE xyY* colourspace array argument and
    # returning it directly.
    if xy.shape[-1] == 3:
        return xy

    x, y = tsplit(xy)

    xyY = tstack([x, y, np.resize(Y, x.shape)])

    return from_range_1(xyY, np.array([1, 1, 100]))


def XYZ_to_xy(XYZ: Domain1) -> NDArrayFloat:
    """
    Convert from *CIE XYZ* tristimulus values to *CIE xy* chromaticity
    coordinates.

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
    | ``XYZ``    | 1                     | 1             |
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


def xy_to_XYZ(xy: ArrayLike) -> Range1:
    """
    Convert from *CIE xy* chromaticity coordinates to *CIE XYZ* tristimulus values.

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
    | ``xy``     | 1                     | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | 1                     | 1             |
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
