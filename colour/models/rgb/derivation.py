#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RGB Colourspace Derivation
==========================

Defines objects related to *RGB* colourspace derivation, essentially
calculating the normalised primary matrix for given *RGB* colourspace primaries
and whitepoint:

-   :func:`normalised_primary_matrix`
-   :func:`chromatically_adapted_primaries`
-   :func:`primaries_whitepoint`
-   :func:`RGB_luminance_equation`
-   :func:`RGB_luminance`

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
.. [1]  Society of Motion Picture and Television Engineers. (1993). Derivation
        of Basic Television Color Equations. In RP 177:1993 (Vol. RP 177:199).
        doi:10.5594/S9781614821915
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.adaptation import chromatic_adaptation_VonKries
from colour.models import XYZ_to_xy, XYZ_to_xyY, xy_to_XYZ
from colour.utilities import tsplit

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['xy_to_z',
           'normalised_primary_matrix',
           'chromatically_adapted_primaries',
           'primaries_whitepoint',
           'RGB_luminance_equation',
           'RGB_luminance']


def xy_to_z(xy):
    """
    Returns the *z* coordinate using given :math:`xy` chromaticity coordinates.

    Parameters
    ----------
    xy : array_like
        :math:`xy` chromaticity coordinates.

    Returns
    -------
    numeric
        *z* coordinate.

    Examples
    --------
    >>> xy_to_z(np.array([0.25, 0.25]))
    0.5
    """

    x, y = tsplit(xy)

    z = 1 - x - y

    return z


def normalised_primary_matrix(primaries, whitepoint):
    """
    Returns the *normalised primary matrix* using given *primaries* and
    *whitepoint* :math:`xy` chromaticity coordinates.

    Parameters
    ----------
    primaries : array_like, (3, 2)
        Primaries :math:`xy` chromaticity coordinates.
    whitepoint : array_like
        Illuminant / whitepoint :math:`xy` chromaticity coordinates.

    Returns
    -------
    ndarray, (3, 3)
        *Normalised primary matrix*.

    Examples
    --------
    >>> p = np.array([0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700])
    >>> whitepoint = np.array([0.32168, 0.33767])
    >>> normalised_primary_matrix(p, whitepoint)  # doctest: +ELLIPSIS
    array([[  9.5255239...e-01,   0.0000000...e+00,   9.3678631...e-05],
           [  3.4396645...e-01,   7.2816609...e-01,  -7.2132546...e-02],
           [  0.0000000...e+00,   0.0000000...e+00,   1.0088251...e+00]])
    """

    primaries = np.reshape(primaries, (3, 2))

    z = xy_to_z(primaries)[..., np.newaxis]
    primaries = np.transpose(np.hstack((primaries, z)))

    whitepoint = xy_to_XYZ(whitepoint)

    coefficients = np.dot(np.linalg.inv(primaries), whitepoint)
    coefficients = np.diagflat(coefficients)

    npm = np.dot(primaries, coefficients)

    return npm


def chromatically_adapted_primaries(primaries,
                                    whitepoint_t,
                                    whitepoint_r,
                                    chromatic_adaptation_transform='CAT02'):
    """
    Chromatically adapts given *primaries* :math:`xy` chromaticity coordinates
    from test `whitepoint_t` to reference `whitepoint_r`.


    Parameters
    ----------
    primaries : array_like, (3, 2)
        Primaries :math:`xy` chromaticity coordinates.
    whitepoint_t : array_like
        Test illuminant / whitepoint :math:`xy` chromaticity coordinates.
    whitepoint_r : array_like
        Reference illuminant / whitepoint :math:`xy` chromaticity coordinates.
    chromatic_adaptation_transform : unicode, optional
        **{'CAT02', 'XYZ Scaling', 'Von Kries', 'Bradford', 'Sharp',
        'Fairchild', 'CMCCAT97', 'CMCCAT2000', 'CAT02_BRILL_CAT', 'Bianco',
        'Bianco PC'}**,
        *Chromatic adaptation* transform.

    Returns
    -------
    ndarray
        Chromatically adapted primaries :math:`xy` chromaticity coordinates.

    Examples
    --------
    >>> p = np.array([0.64, 0.33, 0.30, 0.60, 0.15, 0.06])
    >>> whitepoint_t = np.array([0.31270, 0.32900])
    >>> whitepoint_r = np.array([0.34570, 0.35850])
    >>> chromatic_adaptation_transform = 'Bradford'
    >>> chromatically_adapted_primaries(  # doctest: +ELLIPSIS
    ...     p,
    ...     whitepoint_t,
    ...     whitepoint_r,
    ...     chromatic_adaptation_transform)
    array([[ 0.6484414...,  0.3308533...],
           [ 0.3211951...,  0.5978443...],
           [ 0.1558932...,  0.0660492...]])
    """

    primaries = np.reshape(primaries, (3, 2))

    XYZ_a = chromatic_adaptation_VonKries(
        xy_to_XYZ(primaries),
        xy_to_XYZ(whitepoint_t),
        xy_to_XYZ(whitepoint_r),
        chromatic_adaptation_transform)

    P_a = XYZ_to_xyY(XYZ_a)[..., 0:2]

    return P_a


def primaries_whitepoint(npm):
    """
    Returns the *primaries* and *whitepoint* :math:`xy` chromaticity
    coordinates using given *normalised primary matrix*.

    Parameters
    ----------
    npm : array_like, (3, 3)
        *Normalised primary matrix*.

    Returns
    -------
    tuple
        *Primaries* and *whitepoint* :math:`xy` chromaticity coordinates.

    References
    ----------
    .. [2]  Trieu, T. (2015). Private Discussion with Mansencal, T.

    Examples
    --------
    >>> npm = np.array([[9.52552396e-01, 0.00000000e+00, 9.36786317e-05],
    ...                 [3.43966450e-01, 7.28166097e-01, -7.21325464e-02],
    ...                 [0.00000000e+00, 0.00000000e+00, 1.00882518e+00]])
    >>> p, w = primaries_whitepoint(npm)
    >>> p  # doctest: +ELLIPSIS
    array([[  7.3470000...e-01,   2.6530000...e-01],
           [  0.0000000...e+00,   1.0000000...e+00],
           [  1.0000000...e-04,  -7.7000000...e-02]])
    >>> w # doctest: +ELLIPSIS
    array([ 0.32168,  0.33767])
    """

    npm = npm.reshape((3, 3))

    primaries = XYZ_to_xy(
        np.transpose(np.dot(npm, np.identity(3))))
    whitepoint = np.squeeze(XYZ_to_xy(
        np.transpose(np.dot(npm, np.ones((3, 1))))))

    # TODO: Investigate if we return an ndarray here with primaries and
    # whitepoint stacked together.
    return primaries, whitepoint


def RGB_luminance_equation(primaries, whitepoint):
    """
    Returns the *luminance equation* from given *primaries* and *whitepoint*.

    Parameters
    ----------
    primaries : array_like, (3, 2)
        Primaries chromaticity coordinates.
    whitepoint : array_like
        Illuminant / whitepoint chromaticity coordinates.

    Returns
    -------
    unicode
        *Luminance* equation.

    Examples
    --------
    >>> p = np.array([0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700])
    >>> whitepoint = np.array([0.32168, 0.33767])
    >>> # Doctests skip for Python 2.x compatibility.
    >>> RGB_luminance_equation(p, whitepoint)  # doctest: +SKIP
    'Y = 0.3439664...(R) + 0.7281660...(G) + -0.0721325...(B)'
    """

    return 'Y = {0}(R) + {1}(G) + {2}(B)'.format(
        *np.ravel(normalised_primary_matrix(primaries, whitepoint))[3:6])


def RGB_luminance(RGB, primaries, whitepoint):
    """
    Returns the *luminance* :math:`Y` of given *RGB* components from given
    *primaries* and *whitepoint*.

    Parameters
    ----------
    RGB : array_like
        *RGB* chromaticity coordinate matrix.
    primaries : array_like, (3, 2)
        Primaries chromaticity coordinate matrix.
    whitepoint : array_like
        Illuminant / whitepoint chromaticity coordinates.

    Returns
    -------
    numeric or ndarray
        *Luminance* :math:`Y`.

    Examples
    --------
    >>> RGB = np.array([40.6, 4.2, 67.4])
    >>> p = np.array([0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700])
    >>> whitepoint = np.array([0.32168, 0.33767])
    >>> RGB_luminance(RGB, p, whitepoint)  # doctest: +ELLIPSIS
    12.1616018...
    """

    Y = np.sum(
        normalised_primary_matrix(primaries, whitepoint)[1] * RGB, axis=-1)

    return Y
