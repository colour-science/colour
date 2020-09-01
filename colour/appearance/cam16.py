# -*- coding: utf-8 -*-
"""
CAM16 Colour Appearance Model
=============================

Defines *CAM16* colour appearance model objects:

-   :class:`colour.appearance.InductionFactors_CAM16`
-   :attr:`colour.VIEWING_CONDITIONS_CAM16`
-   :class:`colour.CAM_Specification_CAM16`
-   :func:`colour.XYZ_to_CAM16`
-   :func:`colour.CAM16_to_XYZ`

References
----------
-   :cite:`Li2017` : Li, C., Li, Z., Wang, Z., Xu, Y., Luo, M. R., Cui, G.,
    Melgosa, M., Brill, M. H., & Pointer, M. (2017). Comprehensive color
    solutions: CAM16, CAT16, and CAM16-UCS. Color Research & Application,
    42(6), 703-718. doi:10.1002/col.22131
"""

from __future__ import division, unicode_literals

import colour.ndarray as np
from collections import namedtuple

from colour.algebra import spow
from colour.appearance.ciecam02 import (
    VIEWING_CONDITIONS_CIECAM02, P, achromatic_response_forward,
    achromatic_response_inverse, brightness_correlate, chroma_correlate,
    colourfulness_correlate, degree_of_adaptation, eccentricity_factor,
    hue_angle, hue_quadrature, lightness_correlate,
    opponent_colour_dimensions_forward, opponent_colour_dimensions_inverse,
    post_adaptation_non_linear_response_compression_forward,
    post_adaptation_non_linear_response_compression_inverse,
    post_adaptation_non_linear_response_compression_matrix,
    saturation_correlate, temporary_magnitude_quantity_inverse,
    viewing_condition_dependent_parameters)
from colour.utilities import (CaseInsensitiveMapping, as_float, as_float_array,
                              as_namedtuple, dot_vector, from_range_100,
                              from_range_degrees, ones, to_domain_100,
                              to_domain_degrees, tsplit)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'MATRIX_16', 'MATRIX_INVERSE_16', 'InductionFactors_CAM16',
    'VIEWING_CONDITIONS_CAM16', 'CAM_Specification_CAM16', 'XYZ_to_CAM16',
    'CAM16_to_XYZ'
]

MATRIX_16 = np.array([
    [0.401288, 0.650173, -0.051461],
    [-0.250268, 1.204414, 0.045854],
    [-0.002079, 0.048952, 0.953127],
])
"""
Adaptation matrix :math:`M_{16}`.

MATRIX_16 : array_like, (3, 3)
"""

MATRIX_INVERSE_16 = np.linalg.inv(MATRIX_16)
"""
Inverse adaptation matrix :math:`M^{-1}_{16}`.

MATRIX_INVERSE_16 : array_like, (3, 3)
"""


class InductionFactors_CAM16(
        namedtuple('InductionFactors_CAM16', ('F', 'c', 'N_c'))):
    """
    *CAM16* colour appearance model induction factors.

    Parameters
    ----------
    F : numeric or array_like
        Maximum degree of adaptation :math:`F`.
    c : numeric or array_like
        Exponential non linearity :math:`c`.
    N_c : numeric or array_like
        Chromatic induction factor :math:`N_c`.

    References
    ----------
    :cite:`Li2017`
    """


VIEWING_CONDITIONS_CAM16 = CaseInsensitiveMapping(VIEWING_CONDITIONS_CIECAM02)
VIEWING_CONDITIONS_CAM16.__doc__ = """
Reference *CAM16* colour appearance model viewing conditions.

References
----------
:cite:`Li2017`

VIEWING_CONDITIONS_CAM16 : CaseInsensitiveMapping
    **{'Average', 'Dim', 'Dark'}**
"""


class CAM_Specification_CAM16(
        namedtuple('CAM_Specification_CAM16',
                   ('J', 'C', 'h', 's', 'Q', 'M', 'H', 'HC'))):
    """
    Defines the *CAM16* colour appearance model specification.

    Parameters
    ----------
    J : numeric or array_like
        Correlate of *Lightness* :math:`J`.
    C : numeric or array_like
        Correlate of *chroma* :math:`C`.
    h : numeric or array_like
        *Hue* angle :math:`h` in degrees.
    s : numeric or array_like
        Correlate of *saturation* :math:`s`.
    Q : numeric or array_like
        Correlate of *brightness* :math:`Q`.
    M : numeric or array_like
        Correlate of *colourfulness* :math:`M`.
    H : numeric or array_like
        *Hue* :math:`h` quadrature :math:`H`.
    HC : numeric or array_like
        *Hue* :math:`h` composition :math:`H^C`.

    References
    ----------
    :cite:`Li2017`
    """

    def __new__(cls,
                J=None,
                C=None,
                h=None,
                s=None,
                Q=None,
                M=None,
                H=None,
                HC=None):
        """
        Returns a new instance of the :class:`colour.CAM_Specification_CAM16`
        class.
        """

        return super(CAM_Specification_CAM16, cls).__new__(
            cls, J, C, h, s, Q, M, H, HC)


def XYZ_to_CAM16(XYZ,
                 XYZ_w,
                 L_A,
                 Y_b,
                 surround=VIEWING_CONDITIONS_CAM16['Average'],
                 discount_illuminant=False):
    """
    Computes the *CAM16* colour appearance model correlates from given
    *CIE XYZ* tristimulus values.

    This is the *forward* implementation.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values of test sample / stimulus.
    XYZ_w : array_like
        *CIE XYZ* tristimulus values of reference white.
    L_A : numeric or array_like
        Adapting field *luminance* :math:`L_A` in :math:`cd/m^2`, (often taken
        to be 20% of the luminance of a white object in the scene).
    Y_b : numeric or array_like
        Relative luminance of background :math:`Y_b` in :math:`cd/m^2`.
    surround : InductionFactors_CAM16, optional
        Surround viewing conditions induction factors.
    discount_illuminant : bool, optional
        Truth value indicating if the illuminant should be discounted.

    Returns
    -------
    CAM_Specification_CAM16
        *CAM16* colour appearance model specification.

    Notes
    -----

    +---------------------------+-----------------------+---------------+
    | **Domain**                | **Scale - Reference** | **Scale - 1** |
    +===========================+=======================+===============+
    | ``XYZ``                   | [0, 100]              | [0, 1]        |
    +---------------------------+-----------------------+---------------+
    | ``XYZ_w``                 | [0, 100]              | [0, 1]        |
    +---------------------------+-----------------------+---------------+

    +-------------------------------+-----------------------+---------------+
    | **Range**                     | **Scale - Reference** | **Scale - 1** |
    +===============================+=======================+===============+
    | ``CAM_Specification_CAM16.J`` | [0, 100]              | [0, 1]        |
    +-------------------------------+-----------------------+---------------+
    | ``CAM_Specification_CAM16.C`` | [0, 100]              | [0, 1]        |
    +-------------------------------+-----------------------+---------------+
    | ``CAM_Specification_CAM16.h`` | [0, 360]              | [0, 1]        |
    +-------------------------------+-----------------------+---------------+
    | ``CAM_Specification_CAM16.s`` | [0, 100]              | [0, 1]        |
    +-------------------------------+-----------------------+---------------+
    | ``CAM_Specification_CAM16.Q`` | [0, 100]              | [0, 1]        |
    +-------------------------------+-----------------------+---------------+
    | ``CAM_Specification_CAM16.M`` | [0, 100]              | [0, 1]        |
    +-------------------------------+-----------------------+---------------+
    | ``CAM_Specification_CAM16.H`` | [0, 360]              | [0, 1]        |
    +-------------------------------+-----------------------+---------------+

    References
    ----------
    :cite:`Li2017`

    Examples
    --------
    >>> XYZ = np.array([19.01, 20.00, 21.78])
    >>> XYZ_w = np.array([95.05, 100.00, 108.88])
    >>> L_A = 318.31
    >>> Y_b = 20.0
    >>> surround = VIEWING_CONDITIONS_CAM16['Average']
    >>> XYZ_to_CAM16(XYZ, XYZ_w, L_A, Y_b, surround)  # doctest: +ELLIPSIS
    CAM_Specification_CAM16(J=41.7312079..., C=0.1033557..., \
h=217.0679597..., s=2.3450150..., Q=195.3717089..., M=0.1074367..., \
H=275.5949861..., HC=None)
    """

    XYZ = to_domain_100(XYZ)
    XYZ_w = to_domain_100(XYZ_w)
    _X_w, Y_w, _Z_w = tsplit(XYZ_w)
    L_A = as_float_array(L_A)
    Y_b = as_float_array(Y_b)

    # Step 0
    # Converting *CIE XYZ* tristimulus values to sharpened *RGB* values.
    RGB_w = dot_vector(MATRIX_16, XYZ_w)

    # Computing degree of adaptation :math:`D`.
    D = (np.clip(degree_of_adaptation(surround.F, L_A), 0, 1)
         if not discount_illuminant else ones(L_A.shape))

    n, F_L, N_bb, N_cb, z = tsplit(
        viewing_condition_dependent_parameters(Y_b, Y_w, L_A))

    D_RGB = (D[..., np.newaxis] * Y_w[..., np.newaxis] / RGB_w + 1 -
             D[..., np.newaxis])
    RGB_wc = D_RGB * RGB_w

    # Applying forward post-adaptation non linear response compression.
    RGB_aw = post_adaptation_non_linear_response_compression_forward(
        RGB_wc, F_L)

    # Computing achromatic responses for the whitepoint.
    A_w = achromatic_response_forward(RGB_aw, N_bb)

    # Step 1
    # Converting *CIE XYZ* tristimulus values to sharpened *RGB* values.
    RGB = dot_vector(MATRIX_16, XYZ)

    # Step 2
    RGB_c = D_RGB * RGB

    # Step 3
    # Applying forward post-adaptation non linear response compression.
    RGB_a = post_adaptation_non_linear_response_compression_forward(RGB_c, F_L)

    # Step 4
    # Converting to preliminary cartesian coordinates.
    a, b = tsplit(opponent_colour_dimensions_forward(RGB_a))

    # Computing the *hue* angle :math:`h`.
    h = hue_angle(a, b)

    # Step 5
    # Computing eccentricity factor *e_t*.
    e_t = eccentricity_factor(h)

    # Computing hue :math:`h` quadrature :math:`H`.
    H = hue_quadrature(h)
    # TODO: Compute hue composition.

    # Step 6
    # Computing achromatic responses for the stimulus.
    A = achromatic_response_forward(RGB_a, N_bb)

    # Step 7
    # Computing the correlate of *Lightness* :math:`J`.
    J = lightness_correlate(A, A_w, surround.c, z)

    # Step 8
    # Computing the correlate of *brightness* :math:`Q`.
    Q = brightness_correlate(surround.c, J, A_w, F_L)

    # Step 9
    # Computing the correlate of *chroma* :math:`C`.
    C = chroma_correlate(J, n, surround.N_c, N_cb, e_t, a, b, RGB_a)

    # Computing the correlate of *colourfulness* :math:`M`.
    M = colourfulness_correlate(C, F_L)

    # Computing the correlate of *saturation* :math:`s`.
    s = saturation_correlate(M, Q)

    if np.__name__ == 'cupy':
        J = as_float(J)
        C = as_float(C)
        h = as_float(h)
        s = as_float(s)
        Q = as_float(Q)
        M = as_float(M)
        H = as_float(H)

    return CAM_Specification_CAM16(
        from_range_100(J), from_range_100(C), from_range_degrees(h),
        from_range_100(s), from_range_100(Q), from_range_100(M),
        from_range_degrees(H), None)


def CAM16_to_XYZ(specification,
                 XYZ_w,
                 L_A,
                 Y_b,
                 surround=VIEWING_CONDITIONS_CAM16['Average'],
                 discount_illuminant=False):
    """
    Converts from *CAM16* specification to *CIE XYZ* tristimulus values.

    This is the *inverse* implementation.

    Parameters
    ----------
    specification : CAM_Specification_CAM16
        *CAM16* colour appearance model specification. Correlate of
        *Lightness* :math:`J`, correlate of *chroma* :math:`C` or correlate of
        *colourfulness* :math:`M` and *hue* angle :math:`h` in degrees must be
        specified, e.g. :math:`JCh` or :math:`JMh`.
    XYZ_w : array_like
        *CIE XYZ* tristimulus values of reference white.
    L_A : numeric or array_like
        Adapting field *luminance* :math:`L_A` in :math:`cd/m^2`, (often taken
        to be 20% of the luminance of a white object in the scene).
    Y_b : numeric or array_like
        Relative luminance of background :math:`Y_b` in :math:`cd/m^2`.
    surround : InductionFactors_CAM16, optional
        Surround viewing conditions.
    discount_illuminant : bool, optional
        Discount the illuminant.

    Returns
    -------
    XYZ : ndarray
        *CIE XYZ* tristimulus values.

    Raises
    ------
    ValueError
        If neither *C* or *M* correlates have been defined in the
        ``CAM_Specification_CAM16`` argument.

    Notes
    -----

    +-------------------------------+-----------------------+---------------+
    | **Domain**                    | **Scale - Reference** | **Scale - 1** |
    +===============================+=======================+===============+
    | ``CAM_Specification_CAM16.J`` | [0, 100]              | [0, 1]        |
    +-------------------------------+-----------------------+---------------+
    | ``CAM_Specification_CAM16.C`` | [0, 100]              | [0, 1]        |
    +-------------------------------+-----------------------+---------------+
    | ``CAM_Specification_CAM16.h`` | [0, 360]              | [0, 1]        |
    +-------------------------------+-----------------------+---------------+
    | ``CAM_Specification_CAM16.s`` | [0, 100]              | [0, 1]        |
    +-------------------------------+-----------------------+---------------+
    | ``CAM_Specification_CAM16.Q`` | [0, 100]              | [0, 1]        |
    +-------------------------------+-----------------------+---------------+
    | ``CAM_Specification_CAM16.M`` | [0, 100]              | [0, 1]        |
    +-------------------------------+-----------------------+---------------+
    | ``CAM_Specification_CAM16.H`` | [0, 360]              | [0, 1]        |
    +-------------------------------+-----------------------+---------------+
    | ``XYZ_w``                     | [0, 100]              | [0, 1]        |
    +-------------------------------+-----------------------+---------------+

    +---------------------------+-----------------------+---------------+
    | **Range**                 | **Scale - Reference** | **Scale - 1** |
    +===========================+=======================+===============+
    | ``XYZ``                   | [0, 100]              | [0, 1]        |
    +---------------------------+-----------------------+---------------+

    -   ``CAM_Specification_CAM16`` can also be passed as a compatible argument
        to :func:`colour.utilities.as_namedtuple` definition.

    References
    ----------
    :cite:`Li2017`

    Examples
    --------
    >>> specification = CAM_Specification_CAM16(J=41.731207905126638,
    ...                                     C=0.103355738709070,
    ...                                     h=217.067959767393010)
    >>> XYZ_w = np.array([95.05, 100.00, 108.88])
    >>> L_A = 318.31
    >>> Y_b = 20.0
    >>> CAM16_to_XYZ(specification, XYZ_w, L_A, Y_b)  # doctest: +ELLIPSIS
    array([ 19.01...,  20...  ,  21.78...])
    """

    J, C, h, _s, _Q, M, _H, _HC = as_namedtuple(specification,
                                                CAM_Specification_CAM16)
    J = to_domain_100(J)
    C = to_domain_100(C) if C is not None else C
    h = to_domain_degrees(h)
    M = to_domain_100(M) if M is not None else M
    L_A = as_float_array(L_A)
    XYZ_w = to_domain_100(XYZ_w)
    _X_w, Y_w, _Z_w = tsplit(XYZ_w)

    # Step 0
    # Converting *CIE XYZ* tristimulus values to sharpened *RGB* values.
    RGB_w = dot_vector(MATRIX_16, XYZ_w)

    # Computing degree of adaptation :math:`D`.
    D = (np.clip(degree_of_adaptation(surround.F, L_A), 0, 1)
         if not discount_illuminant else ones(L_A.shape))

    n, F_L, N_bb, N_cb, z = tsplit(
        viewing_condition_dependent_parameters(Y_b, Y_w, L_A))

    D_RGB = (D[..., np.newaxis] * Y_w[..., np.newaxis] / RGB_w + 1 -
             D[..., np.newaxis])
    RGB_wc = D_RGB * RGB_w

    # Applying forward post-adaptation non linear response compression.
    RGB_aw = post_adaptation_non_linear_response_compression_forward(
        RGB_wc, F_L)

    # Computing achromatic responses for the whitepoint.
    A_w = achromatic_response_forward(RGB_aw, N_bb)

    # Step 1
    if C is None and M is not None:
        C = M / spow(F_L, 0.25)
    elif C is None:
        raise ValueError('Either "C" or "M" correlate must be defined in '
                         'the "CAM_Specification_CAM16" argument!')

    # Step 2
    # Computing temporary magnitude quantity :math:`t`.
    t = temporary_magnitude_quantity_inverse(C, J, n)

    # Computing eccentricity factor *e_t*.
    e_t = eccentricity_factor(h)

    # Computing achromatic response :math:`A` for the stimulus.
    A = achromatic_response_inverse(A_w, J, surround.c, z)

    # Computing *P_1* to *P_3*.
    P_n = P(surround.N_c, N_cb, e_t, t, A, N_bb)
    _P_1, P_2, _P_3 = tsplit(P_n)

    # Step 3
    # Computing opponent colour dimensions :math:`a` and :math:`b`.
    a, b = tsplit(opponent_colour_dimensions_inverse(P_n, h))

    # Step 4
    # Computing post-adaptation non linear response compression matrix.
    RGB_a = post_adaptation_non_linear_response_compression_matrix(P_2, a, b)

    # Step 5
    # Applying inverse post-adaptation non linear response compression.
    RGB_c = post_adaptation_non_linear_response_compression_inverse(RGB_a, F_L)

    # Step 6
    RGB = RGB_c / D_RGB

    # Step 7
    XYZ = dot_vector(MATRIX_INVERSE_16, RGB)

    return from_range_100(XYZ)
