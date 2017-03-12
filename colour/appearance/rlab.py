#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RLAB Colour Appearance Model
============================

Defines *RLAB* colour appearance model objects:

-   :attr:`RLAB_VIEWING_CONDITIONS`
-   :attr:`RLAB_D_FACTOR`
-   :class:`RLAB_Specification`
-   :func:`XYZ_to_RLAB`

See Also
--------
`RLAB Colour Appearance Model Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/appearance/rlab.ipynb>`_

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

from colour.appearance.hunt import XYZ_TO_HPE_MATRIX, XYZ_to_rgb
from colour.utilities import (
    CaseInsensitiveMapping,
    dot_matrix,
    dot_vector,
    tsplit,
    row_as_diagonal)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
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
*RLAB* colour appearance model precomputed helper matrix.

R_MATRIX : array_like, (3, 3)
"""

RLAB_VIEWING_CONDITIONS = CaseInsensitiveMapping(
    {'Average': 1 / 2.3,
     'Dim': 1 / 2.9,
     'Dark': 1 / 3.5})
"""
Reference *RLAB* colour appearance model viewing conditions.

RLAB_VIEWING_CONDITIONS : CaseInsensitiveMapping
    **{'Average', 'Dim', 'Dark'}**
"""

RLAB_D_FACTOR = CaseInsensitiveMapping(
    {'Hard Copy Images': 1,
     'Soft Copy Images': 0,
     'Projected Transparencies, Dark Room': 0.5})
"""
*RLAB* colour appearance model *Discounting-the-Illuminant* factor values.

RLAB_D_FACTOR : CaseInsensitiveMapping
    **{'Hard Copy Images',
    'Soft Copy Images',
    'Projected Transparencies, Dark Room'}**

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
    Defines the *RLAB* colour appearance model reference specification.

    This specification has field names consistent with *Fairchild (2013)*
    reference.

    Parameters
    ----------
    LR : numeric or array_like
        Correlate of *Lightness* :math:`L^R`.
    CR : numeric or array_like
        Correlate of *achromatic chroma* :math:`C^R`.
    hR : numeric or array_like
        *Hue* angle :math:`h^R` in degrees.
    sR : numeric or array_like
        Correlate of *saturation* :math:`s^R`.
    HR : numeric or array_like
        *Hue* :math:`h` composition :math:`H^R`.
    aR : numeric or array_like
        Red–green chromatic response :math:`a^R`.
    bR : numeric or array_like
        Yellow–blue chromatic response :math:`b^R`.
    """


class RLAB_Specification(
    namedtuple('RLAB_Specification',
               ('J', 'C', 'h', 's', 'HC', 'a', 'b'))):
    """
    Defines the *RLAB* colour appearance model specification.

    This specification has field names consistent with the remaining colour
    appearance models in :mod:`colour.appearance` but diverge from
    *Fairchild (2013)* reference.

    Parameters
    ----------
    J : numeric or array_like
        Correlate of *Lightness* :math:`L^R`.
    C : numeric or array_like
        Correlate of *achromatic chroma* :math:`C^R`.
    h : numeric or array_like
        *Hue* angle :math:`h^R` in degrees.
    s : numeric or array_like
        Correlate of *saturation* :math:`s^R`.
    HC : numeric or array_like
        *Hue* :math:`h` composition :math:`H^C`.
    a : numeric or array_like
        Red–green chromatic response :math:`a^R`.
    b : numeric or array_like
        Yellow–blue chromatic response :math:`b^R`.

    Notes
    -----
    -   This specification is the one used in the current model implementation.
    """


def XYZ_to_RLAB(XYZ,
                XYZ_n,
                Y_n,
                sigma=RLAB_VIEWING_CONDITIONS['Average'],
                D=RLAB_D_FACTOR['Hard Copy Images']):
    """
    Computes the *RLAB* model color appearance correlates.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values of test sample / stimulus in domain
        [0, 100].
    XYZ_n : array_like
        *CIE XYZ* tristimulus values of reference white in domain [0, 100].
    Y_n : numeric or array_like
        Absolute adapting luminance in :math:`cd/m^2`.
    sigma : numeric or array_like, optional
        Relative luminance of the surround, see :attr:`RLAB_VIEWING_CONDITIONS`
        for reference.
    D : numeric or array_like, optional
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
    -   Input *CIE XYZ* tristimulus values are in domain [0, 100].
    -   Input *CIE XYZ_n* tristimulus values are in domain [0, 100].

    Examples
    --------
    >>> XYZ = np.array([19.01, 20.00, 21.78])
    >>> XYZ_n = np.array([109.85, 100, 35.58])
    >>> Y_n = 31.83
    >>> sigma = RLAB_VIEWING_CONDITIONS['Average']
    >>> D = RLAB_D_FACTOR['Hard Copy Images']
    >>> XYZ_to_RLAB(XYZ, XYZ_n, Y_n, sigma, D)  # doctest: +ELLIPSIS
    RLAB_Specification(J=49.8347069..., C=54.8700585..., h=286.4860208..., \
s=1.1010410..., HC=None, a=15.5711021..., b=-52.6142956...)
    """

    Y_n = np.asarray(Y_n)
    D = np.asarray(D)
    sigma = np.asarray(sigma)

    # Converting to cone responses.
    LMS_n = XYZ_to_rgb(XYZ_n)

    # Computing the :math:`A` matrix.
    LMS_l_E = (3 * LMS_n) / (LMS_n[0] + LMS_n[1] + LMS_n[2])
    LMS_p_L = ((1 + (Y_n[..., np.newaxis] ** (1 / 3)) + LMS_l_E) /
               (1 + (Y_n[..., np.newaxis] ** (1 / 3)) + (1 / LMS_l_E)))
    LMS_a_L = (LMS_p_L + D[..., np.newaxis] * (1 - LMS_p_L)) / LMS_n

    aR = row_as_diagonal(LMS_a_L)
    M = dot_matrix(dot_matrix(R_MATRIX, aR), XYZ_TO_HPE_MATRIX)
    XYZ_ref = dot_vector(M, XYZ)

    X_ref, Y_ref, Z_ref = tsplit(XYZ_ref)

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
    CR = np.hypot(aR, bR)

    # -------------------------------------------------------------------------
    # Computing the correlate of *saturation* :math:`s^R`.
    # -------------------------------------------------------------------------
    sR = CR / LR

    return RLAB_Specification(LR, CR, hR, sR, None, aR, bR)
