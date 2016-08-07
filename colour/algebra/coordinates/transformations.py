#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Coordinates System Transformations
==================================

Defines objects to apply transformations on coordinates systems.

The following transformations are available:

-   :func:`cartesian_to_spherical`: Cartesian to Spherical transformation.
-   :func:`spherical_to_cartesian`: Spherical to Cartesian transformation.
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
__copyright__ = 'Copyright (C) 2013-2016 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['cartesian_to_spherical',
           'spherical_to_cartesian',
           'cartesian_to_cylindrical',
           'cylindrical_to_cartesian']


def cartesian_to_spherical(a):
    """
    Transforms given Cartesian coordinates array :math:`xyz` to Spherical
    coordinates array :math:`\rho\theta\phi` (radial distance, inclination or
    elevation and azimuth).

    Parameters
    ----------
    a : array_like
        Cartesian coordinates array :math:`xyz` to transform.

    Returns
    -------
    ndarray
        Spherical coordinates array :math:`\rho\theta\phi`.

    See Also
    --------
    spherical_to_cartesian, cartesian_to_cylindrical, cylindrical_to_cartesian

    Examples
    --------
    >>> a = np.array([3, 1, 6])
    >>> cartesian_to_spherical(a)  # doctest: +ELLIPSIS
    array([ 6.7823299...,  1.0857465...,  0.3217505...])
    """

    x, y, z = tsplit(a)

    r = np.linalg.norm(a, axis=-1)
    theta = np.arctan2(z, np.linalg.norm(tstack((x, y)), axis=-1))
    phi = np.arctan2(y, x)

    rtp = tstack((r, theta, phi))

    return rtp


def spherical_to_cartesian(a):
    """
    Transforms given Spherical coordinates array :math:`\rho\theta\phi`
    (radial distance, inclination or elevation and azimuth) to Cartesian
    coordinates array :math:`xyz`.

    Parameters
    ----------
    a : array_like
        Spherical coordinates array :math:`\rho\theta\phi` to transform.

    Returns
    -------
    ndarray
        Cartesian coordinates array :math:`xyz`.

    See Also
    --------
    cartesian_to_spherical, cartesian_to_cylindrical, cylindrical_to_cartesian

    Examples
    --------
    >>> a = np.array([6.78232998, 1.08574654, 0.32175055])
    >>> spherical_to_cartesian(a)  # doctest: +ELLIPSIS
    array([ 3.        ,  0.9999999...,  6.        ])
    """

    r, theta, phi = tsplit(a)

    x = r * np.cos(theta) * np.cos(phi)
    y = r * np.cos(theta) * np.sin(phi)
    z = r * np.sin(theta)

    xyz = tstack((x, y, z))

    return xyz


def cartesian_to_cylindrical(a):
    """
    Transforms given Cartesian coordinates array :math:`xyz` to Cylindrical
    coordinates array :math:`\phi\rho\z` (radial distance, azimuth, and
    height).

    Parameters
    ----------
    a : array_like
        Cartesian coordinates array :math:`xyz` to transform.

    Returns
    -------
    ndarray
        Cylindrical coordinates array :math:`\phi\rho\z`.

    See Also
    --------
    cartesian_to_spherical, spherical_to_cartesian, cylindrical_to_cartesian

    Examples
    --------
    >>> a = np.array([3, 1, 6])
    >>> cartesian_to_cylindrical(a)  # doctest: +ELLIPSIS
    array([ 0.3217505...,  3.1622776...,  6.        ])
    """

    x, y, z = tsplit(a)

    phi = np.arctan2(y, x)
    rho = np.linalg.norm(tstack((x, y)), axis=-1)

    return tstack((phi, rho, z))


def cylindrical_to_cartesian(a):
    """
    Transforms given Cylindrical coordinates array :math:`\phi\rho\z` (radial
    distance, azimuth and height) to Cartesian coordinates array :math:`xyz`.

    Parameters
    ----------
    a : array_like
        Cylindrical coordinates array :math:`\phi\rho\z` to transform.

    Returns
    -------
    ndarray
        Cartesian coordinates array :math:`xyz`.

    See Also
    --------
    cartesian_to_spherical, spherical_to_cartesian, cartesian_to_cylindrical

    Examples
    --------
    >>> a = np.array([0.32175055, 3.16227766, 6.00000000])
    >>> cylindrical_to_cartesian(a)  # doctest: +ELLIPSIS
    array([ 3.        ,  0.9999999...,  6.        ])
    """

    phi, rho, z = tsplit(a)

    x = rho * np.cos(phi)
    y = rho * np.sin(phi)

    return tstack((x, y, z))
