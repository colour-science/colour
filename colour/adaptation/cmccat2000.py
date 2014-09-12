#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CMCCAT2000 Chromatic Adaptation Model
=====================================

Defines *CMCCAT2000* chromatic adaptation model objects:

-   :class:`CMCCAT2000_InductionFactors`
-   :attr:`CIECAM02_VIEWING_CONDITIONS`
-   :class:`CMCCAT2000_VIEWING_CONDITIONS`
-   :func:`CMCCAT2000_forward`
-   :func:`CMCCAT2000_reverse`

See Also
--------
`CMCCAT2000 Chromatic Adaptation Model IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/master/notebooks/adaptation/cmccat2000.ipynb>`_  # noqa

References
----------
.. [1]  **Changjun Li, M. Ronnier Luo*, Bryan Rigg andRobert W. G. Hunt**,
        *CMC 2000 chromatic adaptation transform: CMCCAT2000*,
        *Color Research & Application, Volume 27, Issue 1, pages 49–58,
        February 2002*,
        DOI: http://dx.doi.org/10.1002/col.10005
.. [2]  **Stephen Westland, Caterina Ripamonti, Vien Cheung**,
        *Computational Colour Science Using MATLAB, 2nd Edition*,
        The Wiley-IS&T Series in Imaging Science and Technology,
        published July 2012, ISBN-13: 978-0-470-66569-5, page  38.
"""

from __future__ import division, unicode_literals

import math
import numpy as np
from collections import namedtuple

from colour.adaptation.cat import CMCCAT2000_CAT
from colour.utilities import CaseInsensitiveMapping

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['CMCCAT2000_INVERSE_CAT',
           'CMCCAT2000_InductionFactors',
           'CMCCAT2000_VIEWING_CONDITIONS',
           'CMCCAT2000_forward',
           'CMCCAT2000_reverse']

CMCCAT2000_INVERSE_CAT = np.linalg.inv(CMCCAT2000_CAT)
"""
Inverse *CMCCAT2000_CAT* chromatic adaptation transform.

CMCCAT2000_INVERSE_CAT : array_like, (3, 3)
"""


class CMCCAT2000_InductionFactors(
    namedtuple('CMCCAT2000_InductionFactors',
               ('F',))):
    """
    *CMCCAT2000* chromatic adaptation model induction factors.

    Parameters
    ----------
    F : numeric
        :math:`F` surround condition.
    """


CMCCAT2000_VIEWING_CONDITIONS = CaseInsensitiveMapping(
    {'Average': CMCCAT2000_InductionFactors(1.),
     'Dim': CMCCAT2000_InductionFactors(0.8),
     'Dark': CMCCAT2000_InductionFactors(0.8)})
"""
Reference *CMCCAT2000* chromatic adaptation model viewing conditions.

CMCCAT2000_VIEWING_CONDITIONS : CaseInsensitiveMapping
    ('Average', 'Dim', 'Dark')
"""


def CMCCAT2000_forward(XYZ,
                       XYZ_w,
                       XYZ_wr,
                       L_A1=100,
                       L_A2=100,
                       surround=CMCCAT2000_VIEWING_CONDITIONS.get('Average')):
    """
    Adapts given *CIE XYZ* colourspace stimulus from test viewing conditions
    *CIE XYZ_w* colourspace matrix to reference viewing conditions *CIE XYZ_wr*
    colourspace matrix using *CMCCAT2000* forward chromatic adaptation model.

    Parameters
    ----------
    XYZ : array_like, (3,)
        *CIE XYZ* colourspace stimulus to adapt.
    XYZ_w : array_like, (3,)
        Source viewing condition *CIE XYZ* colourspace matrix.
    XYZ_wr : array_like, (3,)
        Target viewing condition *CIE XYZ* colourspace matrix.
    L_A1 : numeric, optional
        Luminance of test adapting field :math`L_{A1}` in :math:`cd/m^2`.
    L_A2 : numeric, optional
        Luminance of reference adapting field :math`L_{A2}` in :math:`cd/m^2`.
    surround : CMCCAT2000_InductionFactors, optional
        Surround viewing conditions induction factors.

    Returns
    -------
    ndarray, (3,)
        *CIE XYZ_c* colourspace matrix of the stimulus corresponding colour.

    Warning
    -------
    The output domain of that definition is non standard!

    Notes
    -----
    -   Input *CIE XYZ*, *CIE XYZ_w* and *CIE XYZ_wr* colourspace matrices are
        in domain [0, 100].
    -   Output *CIE XYZ_c* colourspace matrix is in domain [0, 100].

    Examples
    --------
    >>> XYZ = np.array([22.48, 22.74, 8.54])
    >>> XYZ_w = np.array([111.15, 100.00, 35.20])
    >>> XYZ_wr = np.array([94.81, 100.00, 107.30])
    >>> CMCCAT2000_forward(XYZ, XYZ_w, XYZ_wr, 200, 200)  # doctest: +ELLIPSIS
    array([ 19.5269832...,  23.0683396...,  24.9717522...])
    """

    XYZ, XYZ_w, XYZ_wr = np.ravel(XYZ), np.ravel(XYZ_w), np.ravel(XYZ_wr)

    RGB = np.dot(CMCCAT2000_CAT, XYZ.reshape(3, 1))
    RGB_w = np.dot(CMCCAT2000_CAT, XYZ_w.reshape(3, 1))
    RGB_wr = np.dot(CMCCAT2000_CAT, XYZ_wr.reshape(3, 1))

    D = (surround.F *
         (0.08 * math.log10(0.5 * (L_A1 + L_A2)) +
          0.76 - 0.45 * (L_A1 - L_A2) / (L_A1 + L_A2)))

    D = 0 if D < 0 else 1 if D > 1 else D
    a = D * XYZ_w[1] / XYZ_wr[1]

    RGB_c = RGB * (a * (RGB_wr / RGB_w) + 1 - D)
    XYZ_c = np.dot(CMCCAT2000_INVERSE_CAT, RGB_c)

    return np.ravel(XYZ_c)


def CMCCAT2000_reverse(XYZ_c,
                       XYZ_w,
                       XYZ_wr,
                       L_A1=100,
                       L_A2=100,
                       surround=CMCCAT2000_VIEWING_CONDITIONS.get('Average')):
    """
    Adapts given *CIE XYZ* colourspace matrix of the stimulus corresponding
    colour from reference viewing conditions *CIE XYZ_wr* colourspace matrix to
    test viewing conditions *CIE XYZ_w* colourspace matrix using *CMCCAT2000*
    reverse chromatic adaptation model.

    Parameters
    ----------
    XYZ : array_like, (3,)
        *CIE XYZ* colourspace stimulus to adapt.
    XYZ_w : array_like, (3,)
        Source viewing condition *CIE XYZ* colourspace whitepoint matrix.
    XYZ_wr : array_like, (3,)
        Target viewing condition *CIE XYZ* colourspace whitepoint matrix.
    L_A1 : numeric, optional
        Luminance of test adapting field :math`L_{A1}` in :math:`cd/m^2`.
    L_A2 : numeric, optional
        Luminance of reference adapting field :math`L_{A2}` in :math:`cd/m^2`.
    surround : CMCCAT2000_InductionFactors, optional
        Surround viewing conditions induction factors.

    Returns
    -------
    ndarray, (3,)
        *CIE XYZ_c* stimulus colourspace matrix.

    Warning
    -------
    The output domain of that definition is non standard!

    Notes
    -----
    -   Input *CIE XYZ_c*, *CIE XYZ_w* and *CIE XYZ_wr* colourspace matrices
        are in domain [0, 100].
    -   Output *CIE XYZ* colourspace matrix is in domain [0, 100].

    Examples
    --------
    >>> XYZ_c = np.array([19.53, 23.07, 24.97])
    >>> XYZ_w = np.array([111.15, 100.00, 35.20])
    >>> XYZ_wr = np.array([94.81, 100.00, 107.30])
    >>> CMCCAT2000_reverse(XYZ_c, XYZ_w, XYZ_wr, 200, 200)  # noqa  # doctest: +ELLIPSIS
    array([ 22.4839876...,  22.7419485...,   8.5393392...])
    """

    XYZ_c, XYZ_w, XYZ_wr = np.ravel(XYZ_c), np.ravel(XYZ_w), np.ravel(XYZ_wr)

    RGB_c = np.dot(CMCCAT2000_CAT, XYZ_c.reshape(3, 1))
    RGB_w = np.dot(CMCCAT2000_CAT, XYZ_w.reshape(3, 1))
    RGB_wr = np.dot(CMCCAT2000_CAT, XYZ_wr.reshape(3, 1))

    D = (surround.F *
         (0.08 * math.log10(0.5 * (L_A1 + L_A2)) +
          0.76 - 0.45 * (L_A1 - L_A2) / (L_A1 + L_A2)))

    D = 0 if D < 0 else 1 if D > 1 else D
    a = D * XYZ_w[1] / XYZ_wr[1]

    RGB = RGB_c / (a * (RGB_wr / RGB_w) + 1 - D)
    XYZ = np.dot(CMCCAT2000_INVERSE_CAT, RGB)

    return np.ravel(XYZ)
