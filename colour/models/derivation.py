#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RGB Colourspace Derivation
==========================

Defines objects related to *RGB* colourspace derivation, essentially
calculating the normalised primary matrix for given *RGB* colourspace primaries
and whitepoint.

References
----------
.. [1]  `RP 177-1993 SMPTE RECOMMENDED PRACTICE - Television Color Equations
        <http://car.france3.mars.free.fr/HD/INA-%2026%20jan%2006/SMPTE%20normes%20et%20confs/rp177.pdf>`_
        (Last accessed 24 February 2014)
"""

from __future__ import unicode_literals

import numpy as np

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['xy_to_z',
           'get_normalised_primary_matrix',
           'get_RGB_luminance_equation',
           'get_RGB_luminance']


def xy_to_z(xy):
    """
    Returns the *z* coordinate using given *xy* chromaticity coordinates.

    Parameters
    ----------
    xy : array_like
        *xy* chromaticity coordinates.

    Returns
    -------
    float
        *z* coordinate.

    References
    ----------
    .. [2]  `RP 177-1993 SMPTE RECOMMENDED PRACTICE -
            Television Color Equations: 3.3.2
            <http://car.france3.mars.free.fr/HD/INA-%2026%20jan%2006/SMPTE%20normes%20et%20confs/rp177.pdf>`_

    Examples
    --------
    >>> colour.models.derivation.xy_to_z((0.25, 0.25))
    0.5
    """

    return 1 - xy[0] - xy[1]


def get_normalised_primary_matrix(primaries, whitepoint):
    """
    Returns the *normalised primary matrix* using given *primaries* and
    *whitepoint* matrices.

    Parameters
    ----------
    primaries : array_like
        Primaries chromaticity coordinate matrix, (3, 2).
    whitepoint : array_like
        Illuminant / whitepoint chromaticity coordinates.

    Returns
    -------
    ndarray, (3, 3)
        Normalised primary matrix.

    References
    ----------
    .. [3]  `RP 177-1993 SMPTE RECOMMENDED PRACTICE -
            Television Color Equations: 3.3.2 - 3.3.6
            <http://car.france3.mars.free.fr/HD/INA-%2026%20jan%2006/SMPTE%20normes%20et%20confs/rp177.pdf>`_

    Examples
    --------
    >>> primaries = np.array([0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700])
    >>> whitepoint = (0.32168, 0.33767)
    >>> colour.get_normalised_primary_matrix(primaries, whitepoint)
    array([[  9.52552396e-01,   0.00000000e+00,   9.36786317e-05],
           [  3.43966450e-01,   7.28166097e-01,  -7.21325464e-02],
           [  0.00000000e+00,   0.00000000e+00,   1.00882518e+00]])
    """

    # Add *z* coordinates to the primaries and transposing the matrix.
    primaries = primaries.reshape((3, 2))
    z = np.array([xy_to_z(np.ravel(primary)) for primary in primaries])
    primaries = np.hstack((primaries, z.reshape((3, 1))))

    primaries = np.transpose(primaries)

    whitepoint = np.array([
        whitepoint[0] / whitepoint[1],
        1,
        xy_to_z(whitepoint) / whitepoint[1]]).reshape((3, 1))

    coefficients = np.dot(np.linalg.inv(primaries), whitepoint)
    coefficients = np.diagflat(coefficients)

    npm = np.dot(primaries, coefficients)

    return npm


def get_RGB_luminance_equation(primaries, whitepoint):
    """
    Returns the *luminance equation* from given *primaries* and *whitepoint*
    matrices.

    Parameters
    ----------
    primaries : array_like, (3, 2)
        Primaries chromaticity coordinate matrix.
    whitepoint : array_like
        Illuminant / whitepoint chromaticity coordinates.

    Returns
    -------
    unicode
        *Luminance* equation.

    References
    ----------
    .. [4]  `RP 177-1993 SMPTE RECOMMENDED PRACTICE -
            Television Color Equations: 3.3.8
            <http://car.france3.mars.free.fr/HD/INA-%2026%20jan%2006/SMPTE%20normes%20et%20confs/rp177.pdf>`_

    Examples
    --------
    >>> primaries = np.array([0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700])
    >>> whitepoint = (0.32168, 0.33767)
    >>> colour.get_RGB_luminance_equation(primaries, whitepoint)
    Y = 0.343966449765(R) + 0.728166096613(G) + -0.0721325463786(B)
    """

    return 'Y = {0}(R) + {1}(G) + {2}(B)'.format(
        *np.ravel(get_normalised_primary_matrix(primaries, whitepoint))[3:6])


def get_RGB_luminance(RGB, primaries, whitepoint):
    """
    Returns the *luminance* :math:`y` of given *RGB* components from given
    *primaries* and *whitepoint* matrices.

    Parameters
    ----------
    RGB : array_like, (3,)
        *RGB* chromaticity coordinate matrix.
    primaries : array_like, (3, 2)
        Primaries chromaticity coordinate matrix.
    whitepoint : array_like
        Illuminant / whitepoint chromaticity coordinates.

    Returns
    -------
    float
        *Luminance* :math:`y`.

    References
    ----------
    .. [5]  `RP 177-1993 SMPTE RECOMMENDED PRACTICE -
            Television Color Equations: 3.3.3 - 3.3.6
            <http://car.france3.mars.free.fr/HD/INA-%2026%20jan%2006/SMPTE%20normes%20et%20confs/rp177.pdf>`_
    Examples
    --------
    >>> RGB = np.array([40.6, 4.2, 67.4])
    >>> primaries = np.array([0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700])
    >>> whitepoint = (0.32168, 0.33767)
    >>> colour.get_RGB_luminance(primaries, whitepoint)
    12.1616018403
    """

    R, G, B = np.ravel(RGB)
    X, Y, Z = np.ravel(get_normalised_primary_matrix(primaries,
                                                     whitepoint))[3:6]

    return X * R + Y * G + Z * B