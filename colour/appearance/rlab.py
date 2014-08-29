#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RLAB Colour Appearance Model
============================

Defines *RLAB* colour appearance model objects:

-   :attr:`RLAB_Specification`:
-   :func:`XYZ_to_RLAB`

References
----------
.. [1]  **Mark D. Fairchild**,
        *Refinement of the RLAB color space*,
        *Color Research & Application, Volume 21, Issue 5, pages 338–346,
        October 1996*,
        https://ritdml.rit.edu/bitstream/handle/1850/7857/MFairchildArticle12-06-1998.pdf  # noqa
        (Last accessed 16 August 2014)
.. [2]  **Mark D. Fairchild**, *Color Appearance Models, 3nd Edition*,
        The Wiley-IS&T Series in Imaging Science and Technology,
        published June 2013, ASIN: B00DAYO8E2,
        locations 6019-6178.
"""

from __future__ import division, unicode_literals

import math
import numpy as np
from collections import namedtuple

from colour.appearance.hunt import XYZ_to_rgb
from colour.appearance.hunt import XYZ_TO_HPE_MATRIX
from colour.utilities import CaseInsensitiveMapping

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'GPL V3.0 - http://www.gnu.org/licenses/'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['R_MATRIX',
           'RLAB_VIEWING_CONDITIONS',
           'RLAB_ReferenceSpecification',
           'RLAB_Specification',
           'XYZ_to_RLAB']

R_MATRIX = np.array(
    [[1.9569, -1.1882, 0.2313],
     [0.3612, 0.6388, 0.0000],
     [0.0000, 0.0000, 1.0000]])
"""
*RLAB* colour appearance model precomputed helper matrix.

R_MATRIX : array_like, (3, 3)
"""

RLAB_VIEWING_CONDITIONS = CaseInsensitiveMapping(
    {'Average': 1 / 2.3,
     'Dim': 1 / 2.9,
     'Dark': 1 / 3.5})
"""
Reference *RLAB* colour appearance model viewing conditions.

RLAB_VIEWING_CONDITIONS : dict
('Average', 'Dim', 'Dark')
"""


class RLAB_ReferenceSpecification(
    namedtuple('RLAB_ReferenceSpecification',
               ('LR', 'CR', 'hR', 'sR', 'HR', 'aR', 'bR'))):
    """
    Defines the *RLAB* colour appearance model reference specification.

    This specification has field names consistent with **Mark D. Fairchild**
    reference.

    Parameters
    ----------
    LR : numeric
        Correlate of *Lightness* :math:`L^R`.
    CR : numeric
        Correlate of *achromatic chroma* :math:`C^R`.
    hR : numeric
        *Hue* angle :math:`h^R` in degrees.
    sR : numeric
        Correlate of *saturation* :math:`s^R`.
    HR : numeric
        *Hue* :math:`h` composition :math:`H^R`.
    aR : numeric
        Red–green chromatic response :math:`a^R`.
    bR : numeric
        Yellow–blue chromatic response :math:`b^R`.
    """


class RLAB_Specification(
    namedtuple('RLAB_Specification',
               ('J', 'C', 'h', 's', 'HC', 'a', 'b'))):
    """
    Defines the *RLAB* colour appearance model specification.

    This specification has field names consistent with the remaining colour
    appearance models in :mod:`colour.appearance` but diverge from
    **Mark D. Fairchild** reference.

    Parameters
    ----------
    J : numeric
        Correlate of *Lightness* :math:`L^R`.
    C : numeric
        Correlate of *achromatic chroma* :math:`C^R`.
    h : numeric
        *Hue* angle :math:`h^R` in degrees.
    s : numeric
        Correlate of *saturation* :math:`s^R`.
    HC : numeric
        *Hue* :math:`h` composition :math:`H^C`.
    a : numeric
        Red–green chromatic response :math:`a^R`.
    b : numeric
        Yellow–blue chromatic response :math:`b^R`.
    """


def XYZ_to_RLAB(XYZ, XYZ_n, Y_n, sigma, D):
    """
    Computes the RLAB model color appearance correlates.

    Parameters
    ----------
    XYZ : array_like, (3, n)
        *CIE XYZ* colourspace matrix of test sample / stimulus in domain
        [0, 100].
    XYZ_n : array_like, (3,)
        *CIE XYZ* colourspace matrix of reference white in domain [0, 100].
    Y_n : numeric
        Absolute adapting luminance in :math:`cd/m^2`.
    sigma : numeric
        Relative luminance of the surround, see :attr:`RLAB_VIEWING_CONDITIONS`
        for reference.
    D : numeric
        *Discounting-the-Illuminant* factor in domain [0, 1].

    Returns
    -------
    RLAB_Specification
        *RLAB* colour appearance model specification.

    Warning
    -------
    The input domain of that definition is non standard!

    Notes
    -----
    -   Input *CIE XYZ* colourspace matrix is in domain [0, 100].
    -   Input *CIE XYZ_n* colourspace matrix is in domain [0, 100].

    Examples
    --------
    >>> XYZ = np.array([19.01, 20, 21.78])
    >>> XYZ_n = np.array([109.85, 100, 35.58])
    >>> Y_n = 31.83
    >>> sigma = 0.4347
    >>> D = 1.0
    >>> XYZ_to_RLAB(XYZ, XYZ_n, Y_n, sigma, D)  # doctest: +ELLIPSIS
    RLAB_Specification(J=49.8413019..., C=54.8643626..., h=286.4866886..., s=1.1007810..., HC=None, a=15.5700988..., b=-52.6086524...)
    """

    X, Y, Z = np.ravel(XYZ)

    # Converting to cone responses.
    LMS = XYZ_to_rgb(XYZ)
    LMS_n = XYZ_to_rgb(XYZ_n)

    # Computing the :math:`A` matrix.
    LMS_l_E = (3 * LMS_n) / (LMS_n[0] + LMS_n[1] + LMS_n[2])
    LMS_p_L = ((1 + (Y_n ** (1 / 3)) + LMS_l_E) /
               (1 + (Y_n ** (1 / 3)) + (1 / LMS_l_E)))
    LMS_a_L = (LMS_p_L + D * (1 - LMS_p_L)) / LMS_n

    # Special handling here to allow *array_like* variable input.
    if len(np.shape(X)) == 0:
        # *numeric* case.
        # Implementation as per reference.
        aR = np.diag(LMS_a_L)
        XYZ_ref = R_MATRIX.dot(aR).dot(XYZ_TO_HPE_MATRIX).dot(XYZ)
    else:
        # *array_like* case.
        # Constructing huge multidimensional arrays might not be the best idea,
        # we handle each input dimension separately.

        # First figure out how many values we have to deal with.
        dimension = len(X)
        # Then create the output array that will be filled layer by layer.
        XYZ_ref = np.zeros((3, dimension))
        for layer in range(dimension):
            aR = np.diag(LMS_a_L[..., layer])
            XYZ_ref[..., layer] = (
                R_MATRIX.dot(aR).dot(XYZ_TO_HPE_MATRIX).dot(XYZ[..., layer]))

    X_ref, Y_ref, Z_ref = XYZ_ref

    # -------------------------------------------------------------------------
    # Computing the correlate of *Lightness* :math:`L^R`.
    # -------------------------------------------------------------------------
    LR = 100 * (Y_ref ** sigma)

    # Computing opponent colour dimensions :math:`a^R` and :math:`b^R`.
    aR = 430 * ((X_ref ** sigma) - (Y_ref ** sigma))
    bR = 170 * ((Y_ref ** sigma) - (Z_ref ** sigma))

    # -------------------------------------------------------------------------
    # Computing the *hue* angle :math:`h^R`.
    # -------------------------------------------------------------------------
    hR = math.degrees(np.arctan2(bR, aR)) % 360
    # TODO: Implement hue composition computation.

    # -------------------------------------------------------------------------
    # Computing the correlate of *chroma* :math:`C^R`.
    # -------------------------------------------------------------------------
    CR = np.sqrt((aR ** 2) + (bR ** 2))

    # -------------------------------------------------------------------------
    # Computing the correlate of *saturation* :math:`s^R`.
    # -------------------------------------------------------------------------
    sR = CR / LR

    return RLAB_Specification(LR, CR, hR, sR, None, aR, bR)
