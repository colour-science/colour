#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Coordinates System Transformations
==================================

Defines objects to apply transformations on coordinates systems.

The following transformations are available:

-   :func:`cartesian_to_spherical`: Cartesian to Spherical transformation.
-   :func:`spherical_to_cartesian`: Spherical to Cartesian transformation.
-   :func:`cartesian_to_polar`: Cartesian to Polar transformation.
-   :func:`polar_to_cartesian`: Polar to Cartesian transformation.
-   :func:`cartesian_to_cylindrical`: Cartesian to Cylindrical transformation.
-   :func:`cylindrical_to_cartesian`: Cylindrical to Cartesian transformation.

References
----------
.. [1]  Wikipedia. (n.d.). List of common coordinate transformations.
        Retrieved from http://en.wikipedia.org/wiki/\
List_of_common_coordinate_transformations
.. [2]  Wikipedia. (n.d.). ISO 31-11. misc. Retrieved from
        https://en.wikipedia.org/wiki/ISO_31-11
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.utilities import tsplit, tstack

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['cartesian_to_spherical',
           'spherical_to_cartesian',
           'cartesian_to_polar',
           'polar_to_cartesian',
           'cartesian_to_cylindrical',
           'cylindrical_to_cartesian']


def cartesian_to_spherical(a):
    """
    Transforms given Cartesian coordinates array :math:`xyz` to Spherical
    coordinates array :math:`\\rho\\theta\phi` (radial distance, inclination or
    elevation and azimuth).

    Parameters
    ----------
    a : array_like
        Cartesian coordinates array :math:`xyz` to transform.

    Returns
    -------
    ndarray
        Spherical coordinates array :math:`\\rho\\theta\phi`.

    See Also
    --------
    spherical_to_cartesian, cartesian_to_polar, polar_to_cartesian,
    cartesian_to_cylindrical, cylindrical_to_cartesian

    Examples
    --------
    >>> a = np.array([3, 1, 6])
    >>> cartesian_to_spherical(a)  # doctest: +ELLIPSIS
    array([ 6.7823299...,  1.0857465...,  0.3217505...])
    """

    x, y, z = tsplit(a)

    rho = np.linalg.norm(a, axis=-1)
    theta = np.arctan2(z, np.linalg.norm(tstack((x, y)), axis=-1))
    phi = np.arctan2(y, x)

    rtp = tstack((rho, theta, phi))

    return rtp


def spherical_to_cartesian(a):
    """
    Transforms given Spherical coordinates array :math:`\\rho\\theta\phi`
    (radial distance, inclination or elevation and azimuth) to Cartesian
    coordinates array :math:`xyz`.

    Parameters
    ----------
    a : array_like
        Spherical coordinates array :math:`\\rho\\theta\phi` to transform.

    Returns
    -------
    ndarray
        Cartesian coordinates array :math:`xyz`.

    See Also
    --------
    cartesian_to_spherical, cartesian_to_polar, polar_to_cartesian,
    cartesian_to_cylindrical, cylindrical_to_cartesian

    Examples
    --------
    >>> a = np.array([6.78232998, 1.08574654, 0.32175055])
    >>> spherical_to_cartesian(a)  # doctest: +ELLIPSIS
    array([ 3.        ,  0.9999999...,  6.        ])
    """

    rho, theta, phi = tsplit(a)

    x = rho * np.cos(theta) * np.cos(phi)
    y = rho * np.cos(theta) * np.sin(phi)
    z = rho * np.sin(theta)

    xyz = tstack((x, y, z))

    return xyz


def cartesian_to_polar(a):
    """
    Transforms given Cartesian coordinates array :math:`xy` to Polar
    coordinates array :math:`\\rho\phi` (radial coordinate, angular
    coordinate).

    Parameters
    ----------
    a : array_like
        Cartesian coordinates array :math:`xy` to transform.

    Returns
    -------
    ndarray
        Polar coordinates array :math:`\\rho\phi`.

    See Also
    --------
    cartesian_to_spherical, spherical_to_cartesian, polar_to_cartesian,
    cartesian_to_cylindrical, cylindrical_to_cartesian

    Examples
    --------
    >>> a = np.array([3, 1])
    >>> cartesian_to_polar(a)  # doctest: +ELLIPSIS
    array([ 3.1622776...,  0.3217505...])
    """

    x, y = tsplit(a)

    rho = np.hypot(x, y)
    phi = np.arctan2(y, x)

    return tstack((rho, phi))


def polar_to_cartesian(a):
    """
    Transforms given Polar coordinates array :math:`\\rho\phi` (radial
    coordinate, angular coordinate) to Cartesian coordinates array :math:`xy`.

    Parameters
    ----------
    a : array_like
        Polar coordinates array :math:`\\rho\phi` to transform.

    Returns
    -------
    ndarray
        Cartesian coordinates array :math:`xy`.

    See Also
    --------
    cartesian_to_spherical, spherical_to_cartesian, cartesian_to_polar,
    cartesian_to_cylindrical, cylindrical_to_cartesian

    Examples
    --------
    >>> a = np.array([3.16227766, 0.32175055])
    >>> polar_to_cartesian(a)  # doctest: +ELLIPSIS
    array([ 3.        ,  0.9999999...])
    """

    rho, phi = tsplit(a)

    x = rho * np.cos(phi)
    y = rho * np.sin(phi)

    return tstack((x, y))


def cartesian_to_cylindrical(a):
    """
    Transforms given Cartesian coordinates array :math:`xyz` to Cylindrical
    coordinates array :math:`\\rho\phi z` (azimuth, radial distance and
    height).

    Parameters
    ----------
    a : array_like
        Cartesian coordinates array :math:`xyz` to transform.

    Returns
    -------
    ndarray
        Cylindrical coordinates array :math:`\\rho\phi z`.

    See Also
    --------
    cartesian_to_spherical, spherical_to_cartesian, cartesian_to_polar,
    polar_to_cartesian, cylindrical_to_cartesian

    Examples
    --------
    >>> a = np.array([3, 1, 6])
    >>> cartesian_to_cylindrical(a)  # doctest: +ELLIPSIS
    array([ 3.1622776...,  0.3217505...,  6.        ])
    """

    a = np.asarray(a)

    rho, phi = tsplit(cartesian_to_polar(a[..., 0:2]))

    return tstack((rho, phi, a[..., -1]))


def cylindrical_to_cartesian(a):
    """
    Transforms given Cylindrical coordinates array :math:`\\rho\phi z`
    (azimuth, radial distance and height) to Cartesian coordinates array
    :math:`xyz`.

    Parameters
    ----------
    a : array_like
        Cylindrical coordinates array :math:`\\rho\phi z` to transform.

    Returns
    -------
    ndarray
        Cartesian coordinates array :math:`xyz`.

    See Also
    --------
    cartesian_to_spherical, spherical_to_cartesian, cartesian_to_polar,
    polar_to_cartesian, cartesian_to_cylindrical

    Examples
    --------
    >>> a = np.array([3.16227766, 0.32175055, 6.00000000])
    >>> cylindrical_to_cartesian(a)  # doctest: +ELLIPSIS
    array([ 3.        ,  0.9999999...,  6.        ])
    """

    a = np.asarray(a)

    x, y = tsplit(polar_to_cartesian(a[..., 0:2]))

    return tstack((x, y, a[..., -1]))
