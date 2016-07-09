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
        Retrieved from
        http://en.wikipedia.org/wiki/List_of_common_coordinate_transformations
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
    Transforms given Cartesian coordinates array to Spherical coordinates.

    Parameters
    ----------
    a : array_like
        Cartesian coordinates array (x, y, z) to transform.

    Returns
    -------
    ndarray
        Spherical coordinates array (r, theta, phi).

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
    Transforms given Spherical coordinates array to Cartesian coordinates.

    Parameters
    ----------
    a : array_like
        Spherical coordinates array (r, theta, phi) to transform.

    Returns
    -------
    ndarray
        Cartesian coordinates array (x, y, z).

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
    Transforms given Cartesian coordinates array to Cylindrical coordinates.

    Parameters
    ----------
    a : array_like
        Cartesian coordinates array (x, y, z) to transform.

    Returns
    -------
    ndarray
        Cylindrical coordinates array (z, theta, rho).

    See Also
    --------
    cartesian_to_spherical, spherical_to_cartesian, cylindrical_to_cartesian

    Examples
    --------
    >>> a = np.array([3, 1, 6])
    >>> cartesian_to_cylindrical(a)  # doctest: +ELLIPSIS
    array([ 6.        ,  0.3217505...,  3.1622776...])
    """

    x, y, z = tsplit(a)

    theta = np.arctan2(y, x)
    rho = np.linalg.norm(tstack((x, y)), axis=-1)

    return tstack((z, theta, rho))


def cylindrical_to_cartesian(a):
    """
    Transforms given Cylindrical coordinates array to Cartesian coordinates.

    Parameters
    ----------
    a : array_like
        Cylindrical coordinates array (z, theta, rho) to transform.

    Returns
    -------
    ndarray
        Cartesian coordinates array (x, y, z).

    See Also
    --------
    cartesian_to_spherical, spherical_to_cartesian, cartesian_to_cylindrical

    Examples
    --------
    >>> a = np.array([6.00000000, 0.32175055, 3.16227766])
    >>> cylindrical_to_cartesian(a)  # doctest: +ELLIPSIS
    array([ 3.        ,  0.9999999...,  6.        ])
    """

    z, theta, rho = tsplit(a)

    x = rho * np.cos(theta)
    y = rho * np.sin(theta)

    return tstack((x, y, z))
