# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**coordinates.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package algebra coordinates transformations objects.

**Others:**

"""

from __future__ import unicode_literals

import math
import numpy as np

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["cartesian_to_spherical",
           "spherical_to_cartesian",
           "cartesian_to_cylindrical",
           "cylindrical_to_cartesian"]


def cartesian_to_spherical(vector):
    """
    Transforms given Cartesian coordinates vector to Spherical coordinates.

    Usage::

        >>> cartesian_to_spherical(np.array([3, 1, 6]))
        [6.78232998  1.08574654  0.32175055]

    :param vector: Cartesian coordinates vector (x, y, z) to transform.
    :type vector: array_like
    :return: Spherical coordinates vector (r, theta, phi).
    :rtype: ndarray
    """

    r = np.linalg.norm(vector)
    x, y, z = np.ravel(vector)

    theta = math.atan2(z, np.linalg.norm((x, y)))
    phi = math.atan2(y, x)

    return np.array((r, theta, phi))


def spherical_to_cartesian(vector):
    """
    Transforms given Spherical coordinates vector to Cartesian coordinates.

    Usage::

        >>> spherical_to_cartesian(np.array([6.78232998, 1.08574654, 0.32175055]))
        [ 3.          0.99999999  6.        ]

    :param vector: Spherical coordinates vector (r, theta, phi) to transform.
    :type vector: array_like
    :return: Cartesian coordinates vector (x, y, z).
    :rtype: ndarray
    """

    r, theta, phi = np.ravel(vector)

    x = r * math.cos(theta) * math.cos(phi)
    y = r * math.cos(theta) * math.sin(phi)
    z = r * math.sin(theta)

    return np.array((x, y, z))


def cartesian_to_cylindrical(vector):
    """
    Transforms given Cartesian coordinates vector to Cylindrical coordinates.

    Usage::

        >>> cartesian_to_cylindrical(np.array([3, 1, 6]))
        [ 6.          0.32175055  3.16227766]

    :param vector: Cartesian coordinates vector (x, y, z) to transform.
    :type vector: array_like
    :return: Cylindrical coordinates vector (z, theta, rho).
    :rtype: ndarray
    """

    x, y, z = np.ravel(vector)

    theta = math.atan2(y, x)
    rho = np.linalg.norm((x, y))

    return np.array((z, theta, rho))


def cylindrical_to_cartesian(vector):
    """
    Transforms given Cylindrical coordinates vector to Cartesian coordinates.

    Usage::

        >>> cylindrical_to_cartesian(np.array([6., 0.32175055, 3.16227766]))
        [ 3.          0.99999999  6.        ]

    :param vector: Cylindrical coordinates vector (z, theta, rho) to transform.
    :type vector: array_like
    :return: Cartesian coordinates vector (x, y, z).
    :rtype: ndarray
    """

    z, theta, rho = np.ravel(vector)

    x = rho * math.cos(theta)
    y = rho * math.sin(theta)

    return np.array((x, y, z))