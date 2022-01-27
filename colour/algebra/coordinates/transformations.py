# -*- coding: utf-8 -*-
"""
Coordinates System Transformations
==================================

Defines the objects to apply transformations on coordinates systems.

The following transformations are available:

-   :func:`colour.algebra.cartesian_to_spherical`: Cartesian to spherical
    transformation.
-   :func:`colour.algebra.spherical_to_cartesian`: Spherical to cartesian
    transformation.
-   :func:`colour.algebra.cartesian_to_polar`: Cartesian to polar
    transformation.
-   :func:`colour.algebra.polar_to_cartesian`: Polar to cartesian
    transformation.
-   :func:`colour.algebra.cartesian_to_cylindrical`: Cartesian to cylindrical
    transformation.
-   :func:`colour.algebra.cylindrical_to_cartesian`: Cylindrical to cartesian
    transformation.

References
----------
-   :cite:`Wikipedia2005a` : Wikipedia. (2005). ISO 31-11. Retrieved July 31,
    2016, from https://en.wikipedia.org/wiki/ISO_31-11
-   :cite:`Wikipedia2006` : Wikipedia. (2006). List of common coordinate
    transformations. Retrieved July 18, 2014, from
    http://en.wikipedia.org/wiki/List_of_common_coordinate_transformations
"""

from __future__ import annotations

import numpy as np

from colour.hints import ArrayLike, NDArray
from colour.utilities import as_float_array, tsplit, tstack

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'cartesian_to_spherical',
    'spherical_to_cartesian',
    'cartesian_to_polar',
    'polar_to_cartesian',
    'cartesian_to_cylindrical',
    'cylindrical_to_cartesian',
]


def cartesian_to_spherical(a: ArrayLike) -> NDArray:
    """
    Transforms given cartesian coordinates array :math:`xyz` to spherical
    coordinates array :math:`\\rho\\theta\\phi` (radial distance, inclination
    or elevation and azimuth).

    Parameters
    ----------
    a
        Cartesian coordinates array :math:`xyz` to transform.

    Returns
    -------
    :class:`numpy.ndarray`
        Spherical coordinates array :math:`\\rho\\theta\\phi`, :math:`\\rho` is
        in range [0, +inf], :math:`\\theta` is in range [0, pi] radians, i.e.
        [0, 180] degrees, and :math:`\\phi` is in range [-pi, pi] radians, i.e.
        [-180, 180] degrees.

    References
    ----------
    :cite:`Wikipedia2006`, :cite:`Wikipedia2005a`

    Examples
    --------
    >>> a = np.array([3, 1, 6])
    >>> cartesian_to_spherical(a)  # doctest: +ELLIPSIS
    array([ 6.7823299...,  0.4850497...,  0.3217505...])
    """

    x, y, z = tsplit(a)

    rho = np.linalg.norm(a, axis=-1)
    theta = np.arccos(z / rho)
    phi = np.arctan2(y, x)

    rtp = tstack([rho, theta, phi])

    return rtp


def spherical_to_cartesian(a: ArrayLike) -> NDArray:
    """
    Transforms given spherical coordinates array :math:`\\rho\\theta\\phi`
    (radial distance, inclination or elevation and azimuth) to cartesian
    coordinates array :math:`xyz`.

    Parameters
    ----------
    a
        Spherical coordinates array :math:`\\rho\\theta\\phi` to transform,
        :math:`\\rho` is in range [0, +inf], :math:`\\theta` is in range
        [0, pi] radians, i.e. [0, 180] degrees, and :math:`\\phi` is in range
        [-pi, pi] radians, i.e. [-180, 180] degrees.

    Returns
    -------
    :class:`numpy.ndarray`
        Cartesian coordinates array :math:`xyz`.

    References
    ----------
    :cite:`Wikipedia2006`, :cite:`Wikipedia2005a`

    Examples
    --------
    >>> a = np.array([6.78232998, 0.48504979, 0.32175055])
    >>> spherical_to_cartesian(a)  # doctest: +ELLIPSIS
    array([ 3.0000000...,  0.9999999...,  5.9999999...])
    """

    rho, theta, phi = tsplit(a)

    x = rho * np.sin(theta) * np.cos(phi)
    y = rho * np.sin(theta) * np.sin(phi)
    z = rho * np.cos(theta)

    xyz = tstack([x, y, z])

    return xyz


def cartesian_to_polar(a: ArrayLike) -> NDArray:
    """
    Transforms given cartesian coordinates array :math:`xy` to polar
    coordinates array :math:`\\rho\\phi` (radial coordinate, angular
    coordinate).

    Parameters
    ----------
    a
        Cartesian coordinates array :math:`xy` to transform.

    Returns
    -------
    :class:`numpy.ndarray`
        Polar coordinates array :math:`\\rho\\phi`, :math:`\\rho` is
        in range [0, +inf], :math:`\\phi` is in range [-pi, pi] radians, i.e.
        [-180, 180] degrees.

    References
    ----------
    :cite:`Wikipedia2006`, :cite:`Wikipedia2005a`

    Examples
    --------
    >>> a = np.array([3, 1])
    >>> cartesian_to_polar(a)  # doctest: +ELLIPSIS
    array([ 3.1622776...,  0.3217505...])
    """

    x, y = tsplit(a)

    rho = np.hypot(x, y)
    phi = np.arctan2(y, x)

    return tstack([rho, phi])


def polar_to_cartesian(a: ArrayLike) -> NDArray:
    """
    Transforms given polar coordinates array :math:`\\rho\\phi` (radial
    coordinate, angular coordinate) to cartesian coordinates array :math:`xy`.

    Parameters
    ----------
    a
        Polar coordinates array :math:`\\rho\\phi` to transform, :math:`\\rho`
        is in range [0, +inf], :math:`\\phi` is in range [-pi, pi] radians
        i.e. [-180, 180] degrees.

    Returns
    -------
    :class:`numpy.ndarray`
        Cartesian coordinates array :math:`xy`.

    References
    ----------
    :cite:`Wikipedia2006`, :cite:`Wikipedia2005a`

    Examples
    --------
    >>> a = np.array([3.16227766, 0.32175055])
    >>> polar_to_cartesian(a)  # doctest: +ELLIPSIS
    array([ 3.        ,  0.9999999...])
    """

    rho, phi = tsplit(a)

    x = rho * np.cos(phi)
    y = rho * np.sin(phi)

    return tstack([x, y])


def cartesian_to_cylindrical(a: ArrayLike) -> NDArray:
    """
    Transforms given cartesian coordinates array :math:`xyz` to cylindrical
    coordinates array :math:`\\rho\\phi z` (radial distance, azimuth and
    height).

    Parameters
    ----------
    a
        Cartesian coordinates array :math:`xyz` to transform.

    Returns
    -------
    :class:`numpy.ndarray`
        Cylindrical coordinates array :math:`\\rho\\phi z`, :math:`\\rho` is in
        range [0, +inf], :math:`\\phi` is in range [-pi, pi] radians i.e.
        [-180, 180] degrees, :math:`z` is in range [0, +inf].

    References
    ----------
    :cite:`Wikipedia2006`, :cite:`Wikipedia2005a`

    Examples
    --------
    >>> a = np.array([3, 1, 6])
    >>> cartesian_to_cylindrical(a)  # doctest: +ELLIPSIS
    array([ 3.1622776...,  0.3217505...,  6.        ])
    """

    a = as_float_array(a)

    rho, phi = tsplit(cartesian_to_polar(a[..., 0:2]))

    return tstack([rho, phi, a[..., -1]])


def cylindrical_to_cartesian(a: ArrayLike) -> NDArray:
    """
    Transforms given cylindrical coordinates array :math:`\\rho\\phi z`
    (radial distance, azimuth and height) to cartesian coordinates array
    :math:`xyz`.

    Parameters
    ----------
    a
        Cylindrical coordinates array :math:`\\rho\\phi z` to transform,
        :math:`\\rho` is in range [0, +inf], :math:`\\phi` is in range
        [-pi, pi] radians i.e. [-180, 180] degrees, :math:`z` is in range
        [0, +inf].

    Returns
    -------
    :class:`numpy.ndarray`
        Cartesian coordinates array :math:`xyz`.

    References
    ----------
    :cite:`Wikipedia2006`, :cite:`Wikipedia2005a`

    Examples
    --------
    >>> a = np.array([3.16227766, 0.32175055, 6.00000000])
    >>> cylindrical_to_cartesian(a)  # doctest: +ELLIPSIS
    array([ 3.        ,  0.9999999...,  6.        ])
    """

    a = as_float_array(a)

    x, y = tsplit(polar_to_cartesian(a[..., 0:2]))

    return tstack([x, y, a[..., -1]])
