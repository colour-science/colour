#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RLAB Colour Appearance Model
============================

Defines RLAB colour appearance model objects:

-   :attr:`RLAB_VIEWING_CONDITIONS`
-   :attr:`RLAB_D_FACTOR`
-   :class:`RLAB_Specification`
-   :func:`XYZ_to_RLAB`

See Also
--------
`RLAB Colour Appearance Model IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/master/notebooks/appearance/rlab.ipynb>`_  # noqa

References
----------
.. [1]  Fairchild, M. D. (1996). Refinement of the RLAB color space. Color
        Research & Application, 21(5), 338–346.
        doi:10.1002/(SICI)1520-6378(199610)21:5<338::AID-COL3>3.0.CO;2-Z
.. [2]  Fairchild, M. D. (2013). The RLAB Model. In Color Appearance Models
        (3rd ed., pp. 5563–5824). Wiley. ASIN:B00DAYO8E2
"""

from __future__ import division, unicode_literals

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
           'RLAB_D_FACTOR',
           'RLAB_ReferenceSpecification',
           'RLAB_Specification',
           'XYZ_to_RLAB']

R_MATRIX = np.array(
    [[1.9569, -1.1882, 0.2313],
     [0.3612, 0.6388, 0.0000],
     [0.0000, 0.0000, 1.0000]])
"""
RLAB colour appearance model precomputed helper matrix.

R_MATRIX : array_like, (3, 3)
"""

RLAB_VIEWING_CONDITIONS = CaseInsensitiveMapping(
    {'Average': 1 / 2.3,
     'Dim': 1 / 2.9,
     'Dark': 1 / 3.5})
"""
Reference RLAB colour appearance model viewing conditions.

RLAB_VIEWING_CONDITIONS : CaseInsensitiveMapping
    {'Average', 'Dim', 'Dark'}
"""

RLAB_D_FACTOR = CaseInsensitiveMapping(
    {'Hard Copy Images': 1,
     'Soft Copy Images': 0,
     'Projected Transparencies, Dark Room': 0.5})
"""
RLAB colour appearance model *Discounting-the-Illuminant* factor values.

RLAB_D_FACTOR : CaseInsensitiveMapping
    {'Hard Copy Images',
    'Soft Copy Images',
    'Projected Transparencies, Dark Room'}

Aliases:

-   'hard_cp_img': 'Hard Copy Images'
-   'soft_cp_img': 'Soft Copy Images'
-   'projected_dark': 'Projected Transparencies, Dark Room'
"""
RLAB_D_FACTOR['hard_cp_img'] = (
    RLAB_D_FACTOR['Hard Copy Images'])
RLAB_D_FACTOR['soft_cp_img'] = (
    RLAB_D_FACTOR['Soft Copy Images'])
RLAB_D_FACTOR['projected_dark'] = (
    RLAB_D_FACTOR['Projected Transparencies, Dark Room'])


class RLAB_ReferenceSpecification(
    namedtuple('RLAB_ReferenceSpecification',
               ('LR', 'CR', 'hR', 'sR', 'HR', 'aR', 'bR'))):
    """
    Defines the RLAB colour appearance model reference specification.

    This specification has field names consistent with Fairchild (2013)
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
    Defines the RLAB colour appearance model specification.

    This specification has field names consistent with the remaining colour
    appearance models in :mod:`colour.appearance` but diverge from Fairchild
    (2013) reference.

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


def XYZ_to_RLAB(XYZ,
                XYZ_n,
                Y_n,
                sigma=RLAB_VIEWING_CONDITIONS.get('Average'),
                D=RLAB_D_FACTOR.get('Hard Copy Images')):
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
    sigma : numeric, optional
        Relative luminance of the surround, see :attr:`RLAB_VIEWING_CONDITIONS`
        for reference.
    D : numeric, optional
        *Discounting-the-Illuminant* factor in domain [0, 1].

    Returns
    -------
    RLAB_Specification
        RLAB colour appearance model specification.

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
    >>> sigma = RLAB_VIEWING_CONDITIONS['Average']
    >>> D = RLAB_D_FACTOR['Hard Copy Images']
    >>> XYZ_to_RLAB(XYZ, XYZ_n, Y_n, sigma, D)  # doctest: +ELLIPSIS
    RLAB_Specification(J=49.8347069..., C=54.8700585..., h=286.4860208..., s=1.1010410..., HC=None, a=15.5711021..., b=-52.6142956...)
    """

    X, Y, Z = np.ravel(XYZ)

    # Converting to cone responses.
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
        XYZ_ref = np.dot(np.dot(np.dot(R_MATRIX, aR), XYZ_TO_HPE_MATRIX), XYZ)
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
                np.dot(np.dot(np.dot(R_MATRIX, aR), XYZ_TO_HPE_MATRIX),
                       XYZ[..., layer]))

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
    hR = np.degrees(np.arctan2(bR, aR)) % 360
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
