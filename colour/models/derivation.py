#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RGB Colourspace Derivation
==========================

Defines objects related to *RGB* colourspace derivation, essentially
calculating the normalised primary matrix for given *RGB* colourspace primaries
and whitepoint.

See Also
--------
`RGB Colourspaces IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/master/notebooks/models/rgb.ipynb>`_  # noqa

References
----------
.. [1]  SMPTE. (1993). Derivation of Basic Television Color Equations. In
        RP 177:1993 (Vol. RP 177:199). doi:10.5594/S9781614821915
"""

from __future__ import division, unicode_literals

import numpy as np

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['xy_to_z',
           'normalised_primary_matrix',
           'RGB_luminance_equation',
           'RGB_luminance']


def xy_to_z(xy):
    """
    Returns the *z* coordinate using given *xy* chromaticity coordinates.

    Parameters
    ----------
    xy : array_like
        *xy* chromaticity coordinates.

    Returns
    -------
    numeric
        *z* coordinate.

    Examples
    --------
    >>> xy_to_z((0.25, 0.25))
    0.5
    """

    return 1 - xy[0] - xy[1]


def normalised_primary_matrix(primaries, whitepoint):
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

    Examples
    --------
    >>> pms = np.array([0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700])
    >>> whitepoint = (0.32168, 0.33767)
    >>> normalised_primary_matrix(pms, whitepoint)  # doctest: +ELLIPSIS
    array([[  9.5255239...e-01,   0.0000000...e+00,   9.3678631...e-05],
           [  3.4396645...e-01,   7.2816609...e-01,  -7.2132546...e-02],
           [  0.0000000...e+00,   0.0000000...e+00,   1.0088251...e+00]])
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


def RGB_luminance_equation(primaries, whitepoint):
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

    Examples
    --------
    >>> pms = np.array([0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700])
    >>> whitepoint = (0.32168, 0.33767)
    >>> # Doctests skip for Python 2.x compatibility.
    >>> RGB_luminance_equation(pms, whitepoint)  # doctest: +SKIP
    'Y = 0.3439664...(R) + 0.7281660...(G) + -0.0721325...(B)'
    """

    return 'Y = {0}(R) + {1}(G) + {2}(B)'.format(
        *np.ravel(normalised_primary_matrix(primaries, whitepoint))[3:6])


def RGB_luminance(RGB, primaries, whitepoint):
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
    numeric
        *Luminance* :math:`y`.

    Examples
    --------
    >>> RGB = np.array([40.6, 4.2, 67.4])
    >>> pms = np.array([0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700])
    >>> whitepoint = (0.32168, 0.33767)
    >>> RGB_luminance(RGB, pms, whitepoint)  # doctest: +ELLIPSIS
    12.1616018...
    """

    R, G, B = np.ravel(RGB)
    X, Y, Z = np.ravel(normalised_primary_matrix(primaries,
                                                 whitepoint))[3:6]

    return X * R + Y * G + Z * B
