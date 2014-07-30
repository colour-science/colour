# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**ciecam02.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package *CIECAM02* colour appearance model objects.

**Others:**

"""

from __future__ import unicode_literals

import bisect
import math
import numpy

import colour.computation.chromatic_adaptation
from collections import namedtuple

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["CIECAM02_SURROUND_FCNC",
           "CIECAM02_VIEWING_CONDITION_PARAMETERS",
           "HPE",
           "CAT02_CAT_INVERSE",
           "HUE_DATA_FOR_HUE_QUADRATURE",
           "CIECAM02_JQCMSHH",
           "get_luminance_level_adaptation_factor",
           "get_chromatic_induction_factors",
           "get_base_exponential_non_linearity",
           "get_degree_of_adaptation",
           "apply_full_chromatic_adaptation",
           "RGB_to_HPE",
           "apply_post_adaptation_non_linear_response_compression",
           "get_opponent_colour_dimensions",
           "get_hue_quadrature",
           "get_eccentricity_factor",
           "get_achromatic_response",
           "get_lightness_correlate",
           "get_brightness_correlate",
           "get_chroma_correlate",
           "get_colourfulness_correlate",
           "get_saturation_correlate",
           "XYZ_to_CIECAM02"]

CIECAM02_SURROUND_FCNC = namedtuple("CIECAM02_Surround", ("F", "c", "Nc"))

CIECAM02_VIEWING_CONDITION_PARAMETERS = {
    "Average": CIECAM02_SURROUND_FCNC(1., 0.69, 1.),
    "Dim": CIECAM02_SURROUND_FCNC(0.9, 0.59, 0.95),
    "Dark": CIECAM02_SURROUND_FCNC(0.8, 0.525, 0.8)}

HPE = numpy.array([[0.38971, 0.68898, -0.07868],
                   [-0.22981, 1.18340, 0.04641],
                   [0.00000, 0.00000, 1.00000]])

CAT02_CAT_INVERSE = numpy.linalg.inv(colour.computation.chromatic_adaptation.CAT02_CAT)

HUE_DATA_FOR_HUE_QUADRATURE = {
    "hi": numpy.array([20.14, 90.00, 164.25, 237.53, 380.14]),
    "ei": numpy.array([0.8, 0.7, 1.0, 1.2, 0.8]),
    "Hi": numpy.array([0.0, 100.0, 200.0, 300.0, 400.0])}

CIECAM02_JQCMSHH = namedtuple("CIECAM02_JQCMshH", ("J", "Q", "C", "M", "s", "h", "H"))


def get_luminance_level_adaptation_factor(LA):
    """
    Returns the *luminance* level adaptation factor *FL*.

    Usage::

        >>> get_luminance_level_adaptation_factor(318.31)
        1.16754446415

    :param LA: Adapting field *luminance* in cd/m2.
    :type LA: float
    :return: *Luminance* level adaptation factor *FL*
    :rtype: float
    """

    k = 1. / (5. * LA + 1)
    k4 = k ** 4
    FL = 0.2 * k4 * (5. * LA) + 0.1 * (1. - k4) ** 2 * (5. * LA) ** (1. / 3.)
    return FL


def get_chromatic_induction_factors(n):
    """
    Returns the chromatic induction factors *Nbb* and *Ncb*.

    Usage::

        >>> get_chromatic_induction_factors(0.2)
        (1.0003040045593807, 1.0003040045593807)

    :param n: Function of the luminance factor of the background.
    :type n: float
    :return: Chromatic induction factors *Nbb* and *Ncb*.
    :rtype: tuple
    """

    Nbb = Ncb = 0.725 * (1. / n) ** 0.2
    return Nbb, Ncb


def get_base_exponential_non_linearity(n):
    """
    Returns the base exponential non linearity.

    Usage::

        >>> get_base_exponential_non_linearity(0.2)
        1.9272135955

    :param n: Function of the luminance factor of the background.
    :type n: float
    :return: Base exponential non linearity.
    :rtype: float
    """

    z = 1.48 + math.sqrt(n)
    return z


def get_degree_of_adaptation(F, LA):
    """
    Returns the degree of adaptation *D* from given surround maximum degree of adaptation *F* and
    adapting field *luminance* in cd/m2.

    Usage::

        >>> get_degree_of_adaptation(1.0, 318.31)
        0.994468780088

    :param F: Surround maximum degree of adaptation.
    :type F: float
    :param LA: Adapting field *luminance* in cd/m2.
    :type LA: float
    :return: Degree of adaptation.
    :rtype: float
    """

    D = F * (1. - (1. / 3.6) * numpy.exp((-LA - 42.) / 92.))
    return D


def apply_full_chromatic_adaptation(RGB, RGBw, Yw, D):
    """
    Applies full chromatic adaptation to given *CMCCAT2000* transform sharpened *RGB* matrix
    using given *CMCCAT2000* transform sharpened whitepoint *RGBw* matrix.

    Usage::

        >>> RGB = numpy.array([18.985456, 20.707422, 21.747482])
        >>> RGBw = numpy.array([94.930528, 103.536988, 108.717742])
        >>> Yw = 100.0
        >>> D = 0.994468780088
        >>> apply_full_chromatic_adaptation(1.0, 318.31)
        array([[ 19.99370783]
              [ 20.00393634]
              [ 20.01326387]])

    :param RGB: *CMCCAT2000* transform sharpened *RGB* matrix.
    :type RGB: array_like
    :param RGBw: *CMCCAT2000* transform sharpened whitepoint *RGBw* matrix.
    :type RGBw: array_like
    :param Yw: Whitepoint *Y* value to ensure that the adaptation is independent of the luminance factor of the adopted white point.
    :type Yw: float
    :param D: Degree of adaptation.
    :type D: float
    :return: Adapted *RGB* matrix.
    :rtype: ndarray (3, 1)
    """

    R, G, B = numpy.ravel(RGB)
    Rw, Gw, Bw = numpy.ravel(RGBw)

    equation = lambda x, y: ((Yw * D / y) + 1 - D) * x

    Rc = equation(R, Rw)
    Gc = equation(G, Gw)
    Bc = equation(B, Bw)

    return numpy.array([Rc, Gc, Bc]).reshape((3, 1))


def RGB_to_HPE(RGB):
    """
    Converts given *CMCCAT2000* transform sharpened *RGB* matrix to *Hunt-Pointer-Estevez* colourspace matrix.

    Usage::

        >>> RGB_to_HPE(numpy.array([19.99370783, 20.00393634, 20.01326387]))
        array([[ 19.99693975]
              [ 20.00186123]
              [ 20.0135053 ]])

    :param RGB: *CMCCAT2000* transform sharpened *RGB* matrix.
    :type RGB: array_like
    :return: *Hunt-Pointer-Estevez* colourspace matrix.
    :rtype: ndarray (3, 1)
    """

    pyb = numpy.dot(numpy.dot(HPE, CAT02_CAT_INVERSE), RGB)
    return pyb.reshape((3, 1))


def apply_post_adaptation_non_linear_response_compression(RGB, FL):
    """
    Applies post adaptation non linear response compression on given *CMCCAT2000* transform sharpened *RGB* matrix.

    Usage::

        >>> RGB = numpy.array([19.99693975, 20.00186123, 20.0135053])
        >>> FL = 1.16754446415
        >>> apply_post_adaptation_non_linear_response_compression(RGB, FL)
        array([[ 7.9463202 ]
               [ 7.94711528]
               [ 7.94899595]])

    :param RGB: *CMCCAT2000* transform sharpened *RGB* matrix.
    :type RGB: array_like
    :return: Compressed *CMCCAT2000* transform sharpened *RGB* matrix.
    :rtype: ndarray (3, 1)
    """

    # TODO: Check for negative values and their handling.
    RGBc = (((400. * (FL * RGB / 100) ** 0.42) / (27.13 + (FL * RGB / 100) ** 0.42)) + 0.1).reshape((3, 1))
    return RGBc


def get_opponent_colour_dimensions(RGB):
    """
    Returns opponent colour dimensions from given compressed *CMCCAT2000* transform sharpened *RGB* matrix.

    Usage::

        >>> get_opponent_colour_dimensions(numpy.array([7.9463202, 7.94711528,7.94899595]))
        (-0.000624112068243, -0.000506270106773)

    :param RGB: Compressed *CMCCAT2000* transform sharpened *RGB* matrix.
    :type RGB: array_like
    :return: Opponent colour dimensions
    :rtype: tuple
    """

    R, G, B = numpy.ravel(RGB)

    a = R - 12. * G / 11. + B / 11.
    B = (R + G - 2. * B) / 9.

    return a, B


def get_hue_quadrature(h):
    """
    Returns the hue quadrature from given hue angle in degrees.

    Usage::

        >>> get_hue_quadrature(-140.951567342)
        278.060735856

    :param h: Hue angle in degrees.
    :type h: float
    :return: Hue quadrature.
    :rtype: float
    """

    hi = HUE_DATA_FOR_HUE_QUADRATURE.get("hi")
    ei = HUE_DATA_FOR_HUE_QUADRATURE.get("ei")
    Hi = HUE_DATA_FOR_HUE_QUADRATURE.get("Hi")

    hp = h + 360 if h < hi[0] else h
    index = bisect.bisect_left(hi, hp) - 1

    H = Hi[index] + \
        ((100 * (hp - hi[index]) / ei[index]) / ((hp - hi[index]) / ei[index] + (hi[index + 1] - hp) / ei[index + 1]))

    return H


def get_eccentricity_factor(h):
    """
    Returns the eccentricity factor from given hue angle

    Usage::

        >>> get_eccentricity_factor(-140.951567342)
        1.17400547285

    :param h: Hue angle in degrees.
    :type h: float
    :return: Eccentricity factor.
    :rtype: float
    """

    et = 1. / 4. * (math.cos(2. + h * math.pi / 180.) + 3.8)
    return et


def get_achromatic_response(RGB, Nbb):
    """
    Returns the achromatic response *A* from given compressed *CMCCAT2000* transform sharpened *RGB* matrix
    and *Nbb* chromatic induction factor.

    Usage::

        >>> get_achromatic_response(numpy.array([7.9463202, 7.94711528,7.94899595]), 1.0003040045593807)
        23.9394809667

    :param RGB: Compressed *CMCCAT2000* transform sharpened *RGB* matrix.
    :type RGB: array_like
    :param Nbb: Chromatic induction factor.
    :type Nbb: float
    :return: Achromatic response.
    :rtype: float
    """

    R, G, B = numpy.ravel(RGB)

    A = (2. * R + G + (1. / 20.) * B - 0.305) * Nbb
    return A


def get_lightness_correlate(A, Aw, c, z):
    """
    Returns the *Lightness* correlate *J*.

    Usage::

        >>> get_lightness_correlate(23.9394809667, 46.1882087914, 0.69, 1.9272135955)
        41.7310911324

    :param A: Achromatic response for the stimulus.
    :type A: float
    :param Aw: Achromatic response for the whitepoint.
    :type Aw: float
    :param c: Surround exponential non linearity.
    :type c: float
    :param z: Base exponential non linearity.
    :type z: float
    :return: *Lightness* correlate *J*.
    :rtype: float
    """

    J = 100. * (A / Aw) ** (c * z)
    return J


def get_brightness_correlate(c, J, Aw, FL):
    """
    Returns the *brightness* correlate *Q*.

    Usage::

        >>> get_brightness_correlate(0.69, 41.7310911325, 46.1882087914, 1.16754446415)
        195.371325966

    :param c: Surround exponential non linearity.
    :type c: float
    :param J: *Lightness* correlate *J*.
    :type J: float
    :param Aw: Achromatic response for the whitepoint.
    :type Aw: float
    :param FL: *Luminance* level adaptation factor *FL*.
    :type FL: float
    :return: *Brightness* correlate *Q*.
    :rtype: float
    """

    Q = (4. / c) * math.sqrt(J / 100.) * (Aw + 4) * FL ** 0.25
    return Q


def get_chroma_correlate(J, n, Nc, Ncb, et, a, b, RGBa):
    """
    Returns the *chroma* correlate *C*.

    Usage::

        >>> J = 41.7310911325
        >>> n = 0.2
        >>> Nc = 1.0
        >>> Ncb = 1.00030400456
        >>> et = 1.17400547285
        >>> a = -0.000624112068243
        >>> b = -0.000506270106773
        >>> RGBa = numpy.array([7.9463202, 7.94711528,7.94899595])
        >>> get_chroma_correlate(J, n, Nc, Ncb, et, a, b, RGBa)
        0.104707757171

    :param J: *Lightness* correlate *J*.
    :type J: float
    :param n: Function of the luminance factor of the background.
    :type n: float
    :param Nc: Surround chromatic induction factor.
    :type Nc: float
    :param Ncb: Chromatic induction factor.
    :type Ncb: float
    :param et: Eccentricity factor.
    :type et: float
    :param a: Opponent colour dimension *a*.
    :type a: float
    :param b: Opponent colour dimension *b*.
    :type b: float
    :param RGBa: Compressed stimulus *CMCCAT2000* transform sharpened *RGB* matrix.
    :type RGBa: array_like
    :return: *Chroma* correlate *C*.
    :rtype: float
    """

    Ra, Ga, Ba = numpy.ravel(RGBa)

    t = ((50000. / 13.) * Nc * Ncb) * (et * (a ** 2 + b ** 2) ** 0.5) / (Ra + Ga + 21. * Ba / 20.)
    C = t ** 0.9 * (J / 100.) ** 0.5 * (1.64 - 0.29 ** n) ** 0.73
    return C


def get_colourfulness_correlate(C, FL):
    """
    Returns the *colourfulness* correlate *M*.

    Usage::

        >>> get_colourfulness_correlate(0.104707757171, 1.16754446415)
        0.108842175669

    :param C: *Chroma* correlate *C*.
    :type C: float
    :param FL: *Luminance* level adaptation factor *FL*.
    :type FL: float
    :return: *Colourfulness* correlate *M*.
    :rtype: float
    """

    M = C * FL ** 0.25
    return M


def get_saturation_correlate(M, Q):
    """
    Returns the *saturation* correlate *s*.

    Usage::

        >>> get_saturation_correlate(0.108842175669, 195.371325966)
        2.36030537392

    :param M: *Colourfulness* correlate *M*.
    :type M: float
    :param Q: *Brightness* correlate *C*.
    :type Q: float
    :return: *Saturation* correlate *s*.
    :rtype: float
    """

    s = 100. * (M / Q) ** 0.5
    return s


def XYZ_to_CIECAM02(XYZ,
                    XYZw,
                    Yb,
                    LA,
                    surround=CIECAM02_VIEWING_CONDITION_PARAMETERS.get("Average"),
                    discount_illuminant=False):
    """
    Converts given *CIE XYZ* colourspace matrix to *CIECAM02* representation.

    References:

    -  **Mark D. Fairchild**, *Color Appearance Models, 2nd Edition*, \
    The Wiley-IS&T Series in Imaging Science and Technology, published 19 November 2004, ISBN-13: 978-0470012161, Pages 265-277.
    -  **Stephen Westland, Caterina Ripamonti, Vien Cheung**, *Computational Colour Science Using MATLAB, 2nd Edition*, \
    The Wiley-IS&T Series in Imaging Science and Technology, published July 2012, ISBN-13: 978-0-470-66569-5, Pages 88-92.
    -  `The CIECAM02 Color Appearance Model <http://rit-mcsl.org/fairchild/PDFs/PRO19.pdf>`_

    Usage::

    :param XYZ: *CIE XYZ* colourspace stimulus matrix.
    :type XYZ: array_like
    :param XYZw: *CIE XYZ* colourspace whitepoint matrix.
    :type XYZw: array_like
    :param Yb: Adapting field *Y* tristimulus value.
    :type Yb: float
    :param LA: Adapting field *luminance* in cd/m2.
    :type LA: float
    :param surround: Surround viewing conditions.
    :type surround: CIECAM02_Surround
    :param discount_illuminant: Discount the illuminant.
    :type discount_illuminant: bool
    :return: *CIECAM02* representation.
    :rtype: CIECAM02_JQCMshH
    """

    XYZ = numpy.array(XYZ).reshape((3, 1))
    XYZw = numpy.array(XYZw).reshape((3, 1))
    X, Y, Z = numpy.ravel(XYZ)
    Xw, Yw, Zw = numpy.ravel(XYZw)

    n = Yb / Yw

    FL = get_luminance_level_adaptation_factor(LA)
    Nbb, Ncb = get_chromatic_induction_factors(n)
    z = get_base_exponential_non_linearity(n)

    # Step 1: Converting *CIE XYZ* colourspace matrices to *CMCCAT2000* transform sharpened *RGB* values.
    RGB = numpy.dot(colour.computation.chromatic_adaptation.CAT02_CAT, XYZ)
    RGBw = numpy.dot(colour.computation.chromatic_adaptation.CAT02_CAT, XYZw)

    # Step 2: Calculating degree of adaptation *D*.
    D = get_degree_of_adaptation(surround.F, LA) if not discount_illuminant else 1.

    # Step 3: Calculation full chromatic adaptation.
    RGBc = apply_full_chromatic_adaptation(RGB, RGBw, Yw, D)
    RGBwc = apply_full_chromatic_adaptation(RGBw, RGBw, Yw, D)

    # Step 4: Converting to *Hunt-Pointer-Estevez* colourspace.
    RGBp = RGB_to_HPE(RGBc)
    RGBpw = RGB_to_HPE(RGBwc)

    # Step 5: Apply post-adaptation non linear response compression.
    RGBa = apply_post_adaptation_non_linear_response_compression(RGBp, FL)
    RGBaw = apply_post_adaptation_non_linear_response_compression(RGBpw, FL)

    # Step 6: Convert to Preliminary Cartesian coordinates.
    a, b = get_opponent_colour_dimensions(RGBa)
    h = math.degrees(math.atan2(b, a))

    # Step 7: Compute hue quadrature *hue*.
    H = get_hue_quadrature(h)

    # Step 8: Compute eccentricity factor *et*.
    et = get_eccentricity_factor(h)

    # Step 9: Compute achromatic responses for the stimulus.
    A = get_achromatic_response(RGBa, Nbb)
    Aw = get_achromatic_response(RGBaw, Nbb)

    # Step 10: Compute the correlate of *Lightness* *J*.
    J = get_lightness_correlate(A, Aw, surround.c, z)

    # Step 11: Compute the correlate of *brightness* *Q*.
    Q = get_brightness_correlate(surround.c, J, Aw, FL)

    # Step 12: Compute the correlate of *chroma* *C*.
    C = get_chroma_correlate(J, n, surround.Nc, Ncb, et, a, b, RGBa)

    # Step 13: Compute the correlate of *colourfulness* *M*.
    M = get_colourfulness_correlate(C, FL)

    # Step 14: Compute the correlate of *saturation* *s*.
    s = get_saturation_correlate(M, Q)

    return CIECAM02_JQCMSHH(J, Q, C, M, s, h, H)


# print XYZ_to_CIECAM02([11.18882, 9.30994, 3.21014],
# [96.4296, 100, 82.49],
# Yb=20,
# LA=1000)

print XYZ_to_CIECAM02([19.01, 20.00, 21.78],
                      [95.05, 100.00, 108.88],
                      Yb=20.0,
                      LA=318.31)