#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Von Kries Chromatic Adaptation Model
====================================

Defines Von Kries chromatic adaptation model objects:

-   :func:`chromatic_adaptation_matrix_VonKries`
-   :func:`chromatic_adaptation_VonKries`

See Also
--------
`Chromatic Adaptation IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/master/notebooks/adaptation/vonkries.ipynb>`_  # noqa

References
----------
.. [1]  Fairchild, M. D. (2013). Chromatic Adaptation Models. In Color
        Appearance Models (3rd ed., pp. 4179–4252). Wiley. ASIN:B00DAYO8E2
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.adaptation import CHROMATIC_ADAPTATION_TRANSFORMS
from colour.utilities import row_as_diagonal

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['chromatic_adaptation_matrix_VonKries',
           'chromatic_adaptation_VonKries']


def chromatic_adaptation_matrix_VonKries(XYZ_w, XYZ_wr, transform='CAT02'):
    """
    Returns the *chromatic adaptation* matrix from test viewing conditions
    *CIE XYZ_w* colourspace matrix to reference viewing conditions *CIE XYZ_wr*
    colourspace matrix.

    Parameters
    ----------
    XYZ_w : array_like
        Test viewing condition *CIE XYZ* colourspace matrix.
    XYZ_wr : array_like
        Reference viewing condition *CIE XYZ* colourspace matrix.
    transform : unicode, optional
        {'CAT02', 'XYZ Scaling', 'Von Kries', 'Bradford', 'Sharp', 'Fairchild,
        'CMCCAT97', 'CMCCAT2000', 'CAT02_BRILL_CAT', 'Bianco', 'Bianco PC'},
        Chromatic adaptation transform.

    Returns
    -------
    ndarray, (3, 3)
        Chromatic adaptation matrix.

    Raises
    ------
    KeyError
        If chromatic adaptation method is not defined.

    Examples
    --------
    >>> XYZ_w = np.array([1.09846607, 1., 0.3558228])
    >>> XYZ_wr = np.array([0.95042855, 1., 1.08890037])
    >>> chromatic_adaptation_matrix_VonKries(XYZ_w, XYZ_wr)  # noqa  # doctest: +ELLIPSIS
    array([[ 0.8687653..., -0.1416539...,  0.3871961...],
           [-0.1030072...,  1.0584014...,  0.1538646...],
           [ 0.0078167...,  0.0267875...,  2.9608177...]])

    Using Bradford method:

    >>> XYZ_w = np.array([1.09846607, 1., 0.3558228])
    >>> XYZ_wr = np.array([0.95042855, 1., 1.08890037])
    >>> method = 'Bradford'
    >>> chromatic_adaptation_matrix_VonKries(XYZ_w, XYZ_wr, method)  # noqa  # doctest: +ELLIPSIS
    array([[ 0.8446794..., -0.1179355...,  0.3948940...],
           [-0.1366408...,  1.1041236...,  0.1291981...],
           [ 0.0798671..., -0.1349315...,  3.1928829...]])
    """

    M = CHROMATIC_ADAPTATION_TRANSFORMS.get(transform)

    if M is None:
        raise KeyError(
            '"{0}" chromatic adaptation transform is not defined! Supported '
            'methods: "{1}".'.format(transform,
                                     CHROMATIC_ADAPTATION_TRANSFORMS.keys()))

    rgb_w = np.einsum('...i,...ij->...j', XYZ_w, np.transpose(M))
    rgb_wr = np.einsum('...i,...ij->...j', XYZ_wr, np.transpose(M))

    D = rgb_wr / rgb_w

    D = row_as_diagonal(D)

    cat = np.einsum('...ij,...jk->...ik', np.linalg.inv(M), D)
    cat = np.einsum('...ij,...jk->...ik', cat, M)

    return cat


def chromatic_adaptation_VonKries(XYZ, XYZ_w, XYZ_wr, transform='CAT02'):
    """
    Adapts given *CIE XYZ* colourspace stimulus from test viewing conditions
    *CIE XYZ_w* colourspace matrix to reference viewing conditions *CIE XYZ_wr*
    colourspace matrix. [6]_

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* colourspace stimulus to adapt.
    XYZ_w : array_like
        Test viewing condition *CIE XYZ* colourspace whitepoint matrix.
    XYZ_wr : array_like
        Reference viewing condition *CIE XYZ* colourspace whitepoint matrix.
    transform : unicode, optional
        {'CAT02', 'XYZ Scaling', 'Von Kries', 'Bradford', 'Sharp', 'Fairchild,
        'CMCCAT97', 'CMCCAT2000', 'CAT02_BRILL_CAT', 'Bianco', 'Bianco PC'},
        Chromatic adaptation transform.

    Returns
    -------
    ndarray
        *CIE XYZ_c* colourspace matrix of the stimulus corresponding colour.

    Examples
    --------
    >>> XYZ = np.array([0.07049534, 0.1008, 0.09558313])
    >>> XYZ_w = np.array([1.09846607, 1., 0.3558228])
    >>> XYZ_wr = np.array([0.95042855, 1., 1.08890037])
    >>> chromatic_adaptation_VonKries(XYZ, XYZ_w, XYZ_wr)  # doctest: +ELLIPSIS
    array([ 0.0839746...,  0.1141321...,  0.2862554...])

    Using Bradford method:

    >>> XYZ = np.array([0.07049534, 0.1008, 0.09558313])
    >>> XYZ_w = np.array([1.09846607, 1., 0.3558228])
    >>> XYZ_wr = np.array([0.95042855, 1., 1.08890037])
    >>> method = 'Bradford'
    >>> chromatic_adaptation_VonKries(XYZ, XYZ_w, XYZ_wr, method)  # noqa  # doctest: +ELLIPSIS
    array([ 0.0854032...,  0.1140122...,  0.2972149...])
    """

    cat = chromatic_adaptation_matrix_VonKries(XYZ_w, XYZ_wr, transform)
    XYZ_a = np.einsum('...ij,...j->...i', cat, XYZ)

    return XYZ_a
