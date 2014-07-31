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
import numpy as np
from collections import namedtuple

import colour.colorimetry.chromatic_adaptation
import colour.utilities.decorators

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
           "CIECAM02_JCHQMSH",
           "get_luminance_level_adaptation_factor",
           "get_chromatic_induction_factors",
           "get_base_exponential_non_linearity",
           "get_viewing_condition_dependent_parameters",
           "get_degree_of_adaptation",
           "apply_forward_full_chromatic_adaptation",
           "RGB_to_HPE",
           "apply_forward_post_adaptation_non_linear_response_compression",
           "get_forward_opponent_colour_dimensions",
           "get_hue_quadrature",
           "get_forward_eccentricity_factor",
           "get_forward_achromatic_response",
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

HPE = np.array([[0.38971, 0.68898, -0.07868],
                   [-0.22981, 1.18340, 0.04641],
                   [0.00000, 0.00000, 1.00000]])

HPE_INVERSE = np.linalg.inv(HPE)

CAT02_CAT_INVERSE = np.linalg.inv(colour.colorimetry.chromatic_adaptation.CAT02_CAT)

HUE_DATA_FOR_HUE_QUADRATURE = {
    "hi": np.array([20.14, 90.00, 164.25, 237.53, 380.14]),
    "ei": np.array([0.8, 0.7, 1.0, 1.2, 0.8]),
    "Hi": np.array([0.0, 100.0, 200.0, 300.0, 400.0])}

CIECAM02_JCHQMSH = namedtuple("CIECAM02_JChQMsH", ("J", "C", "h", "Q", "M", "s", "H"))

_CIECAM02_VIEWING_CONDITION_DEPENDENT_PARAMETERS_CACHE = {}

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

    References:

    -  **Mark D. Fairchild**, *Color Appearance Models, 2nd Edition*, \
    The Wiley-IS&T Series in Imaging Science and Technology, published 19 November 2004, ISBN-13: 978-0470012161, Pages 265-277.
    -  **Stephen Westland, Caterina Ripamonti, Vien Cheung**, *Computational Colour Science Using MATLAB, 2nd Edition*, \
    The Wiley-IS&T Series in Imaging Science and Technology, published July 2012, ISBN-13: 978-0-470-66569-5, Pages 88-92.
    -  `The CIECAM02 Color Appearance Model <http://rit-mcsl.org/fairchild/PDFs/PRO19.pdf>`_  (Last accessed 30 July 2014)
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

    References:

    -  **Mark D. Fairchild**, *Color Appearance Models, 2nd Edition*, \
    The Wiley-IS&T Series in Imaging Science and Technology, published 19 November 2004, ISBN-13: 978-0470012161, Pages 265-277.
    -  **Stephen Westland, Caterina Ripamonti, Vien Cheung**, *Computational Colour Science Using MATLAB, 2nd Edition*, \
    The Wiley-IS&T Series in Imaging Science and Technology, published July 2012, ISBN-13: 978-0-470-66569-5, Pages 88-92.
    -  `The CIECAM02 Color Appearance Model <http://rit-mcsl.org/fairchild/PDFs/PRO19.pdf>`_  (Last accessed 30 July 2014)
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

    References:

    -  **Mark D. Fairchild**, *Color Appearance Models, 2nd Edition*, \
    The Wiley-IS&T Series in Imaging Science and Technology, published 19 November 2004, ISBN-13: 978-0470012161, Pages 265-277.
    -  **Stephen Westland, Caterina Ripamonti, Vien Cheung**, *Computational Colour Science Using MATLAB, 2nd Edition*, \
    The Wiley-IS&T Series in Imaging Science and Technology, published July 2012, ISBN-13: 978-0-470-66569-5, Pages 88-92.
    -  `The CIECAM02 Color Appearance Model <http://rit-mcsl.org/fairchild/PDFs/PRO19.pdf>`_  (Last accessed 30 July 2014)
    """

    z = 1.48 + math.sqrt(n)
    return z


@colour.utilities.decorators.memoize(_CIECAM02_VIEWING_CONDITION_DEPENDENT_PARAMETERS_CACHE)
def get_viewing_condition_dependent_parameters(Yb, Yw, LA):
    """
    Returns the viewing condition dependent parameters.

    Usage::

        >>> get_viewing_condition_dependent_parameters(20.0, 100.0, 318.31)
        (0.2, 1.16754446415, 1.00030400456, 1.00030400456, 1.9272135955)

    :param Yb: Adapting field *Y* tristimulus value.
    :type Yb: float
    :param Yw: Whitepoint *Y* tristimulus value.
    :type Yw: float
    :param LA: Adapting field *luminance* in cd/m2.
    :type LA: float
    :return: Viewing condition dependent parameters.
    :rtype: tuple
    """

    n = Yb / Yw

    FL = get_luminance_level_adaptation_factor(LA)
    Nbb, Ncb = get_chromatic_induction_factors(n)
    z = get_base_exponential_non_linearity(n)

    return n, FL, Nbb, Ncb, z


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

    References:

    -  **Mark D. Fairchild**, *Color Appearance Models, 2nd Edition*, \
    The Wiley-IS&T Series in Imaging Science and Technology, published 19 November 2004, ISBN-13: 978-0470012161, Pages 265-277.
    -  **Stephen Westland, Caterina Ripamonti, Vien Cheung**, *Computational Colour Science Using MATLAB, 2nd Edition*, \
    The Wiley-IS&T Series in Imaging Science and Technology, published July 2012, ISBN-13: 978-0-470-66569-5, Pages 88-92.
    -  `The CIECAM02 Color Appearance Model <http://rit-mcsl.org/fairchild/PDFs/PRO19.pdf>`_  (Last accessed 30 July 2014)
    """

    D = F * (1. - (1. / 3.6) * np.exp((-LA - 42.) / 92.))
    return D


def apply_forward_full_chromatic_adaptation(RGB, RGBw, Yw, D):
    """
    Applies full chromatic adaptation to given *CMCCAT2000* transform sharpened *RGB* matrix
    using given *CMCCAT2000* transform sharpened whitepoint *RGBw* matrix.

    Usage::

        >>> RGB = np.array([18.985456, 20.707422, 21.747482])
        >>> RGBw = np.array([94.930528, 103.536988, 108.717742])
        >>> Yw = 100.0
        >>> D = 0.994468780088
        >>> apply_forward_full_chromatic_adaptation(1.0, 318.31)
        array([[ 19.99370783]
              [ 20.00393634]
              [ 20.01326387]])

    :param RGB: *CMCCAT2000* transform sharpened *RGB* matrix.
    :type RGB: array_like
    :param RGBw: *CMCCAT2000* transform sharpened whitepoint *RGBw* matrix.
    :type RGBw: array_like
    :param Yw: Whitepoint *Y* tristimulus value.
    :type Yw: float
    :param D: Degree of adaptation.
    :type D: float
    :return: Adapted *RGB* matrix.
    :rtype: ndarray (3, 1)

    References:

    -  **Mark D. Fairchild**, *Color Appearance Models, 2nd Edition*, \
    The Wiley-IS&T Series in Imaging Science and Technology, published 19 November 2004, ISBN-13: 978-0470012161, Pages 265-277.
    -  **Stephen Westland, Caterina Ripamonti, Vien Cheung**, *Computational Colour Science Using MATLAB, 2nd Edition*, \
    The Wiley-IS&T Series in Imaging Science and Technology, published July 2012, ISBN-13: 978-0-470-66569-5, Pages 88-92.
    -  `The CIECAM02 Color Appearance Model <http://rit-mcsl.org/fairchild/PDFs/PRO19.pdf>`_  (Last accessed 30 July 2014)
    """

    R, G, B = np.ravel(RGB)
    Rw, Gw, Bw = np.ravel(RGBw)

    equation = lambda x, y: ((Yw * D / y) + 1 - D) * x

    Rc = equation(R, Rw)
    Gc = equation(G, Gw)
    Bc = equation(B, Bw)

    return np.array([Rc, Gc, Bc]).reshape((3, 1))


def apply_reverse_full_chromatic_adaptation(RGB, RGBw, Yw, D):
    R, G, B = np.ravel(RGB)
    Rw, Gw, Bw = np.ravel(RGBw)

    equation = lambda x, y: x / (Yw * (D / y) + 1. - D)

    Rc = equation(R, Rw)
    Gc = equation(G, Gw)
    Bc = equation(B, Bw)

    return np.array([Rc, Gc, Bc]).reshape((3, 1))


def RGB_to_HPE(RGB):
    """
    Converts given *CMCCAT2000* transform sharpened *RGB* matrix to *Hunt-Pointer-Estevez* colourspace matrix.

    Usage::

        >>> RGB_to_HPE(np.array([19.99370783, 20.00393634, 20.01326387]))
        array([[ 19.99693975]
              [ 20.00186123]
              [ 20.0135053 ]])

    :param RGB: *CMCCAT2000* transform sharpened *RGB* matrix.
    :type RGB: array_like
    :return: *Hunt-Pointer-Estevez* colourspace matrix.
    :rtype: ndarray (3, 1)

    References:

    -  **Mark D. Fairchild**, *Color Appearance Models, 2nd Edition*, \
    The Wiley-IS&T Series in Imaging Science and Technology, published 19 November 2004, ISBN-13: 978-0470012161, Pages 265-277.
    -  **Stephen Westland, Caterina Ripamonti, Vien Cheung**, *Computational Colour Science Using MATLAB, 2nd Edition*, \
    The Wiley-IS&T Series in Imaging Science and Technology, published July 2012, ISBN-13: 978-0-470-66569-5, Pages 88-92.
    -  `The CIECAM02 Color Appearance Model <http://rit-mcsl.org/fairchild/PDFs/PRO19.pdf>`_  (Last accessed 30 July 2014)
    """

    pyb = np.dot(np.dot(HPE, CAT02_CAT_INVERSE), RGB)
    return pyb.reshape((3, 1))


def HPE_to_RGB(pyb):
    RGB = np.dot(np.dot(colour.colorimetry.chromatic_adaptation.CAT02_CAT, HPE_INVERSE), pyb)
    return RGB.reshape((3, 1))


def apply_forward_post_adaptation_non_linear_response_compression(RGB, FL):
    """
    Returns given *CMCCAT2000* transform sharpened *RGB* matrix with post adaptation non linear response compression \
    for forward *CIECAM02* implementation.

    Usage::

        >>> RGB = np.array([19.99693975, 20.00186123, 20.0135053])
        >>> FL = 1.16754446415
        >>> apply_forward_post_adaptation_non_linear_response_compression(RGB, FL)
        array([[ 7.9463202 ]
               [ 7.94711528]
               [ 7.94899595]])

    :param RGB: *CMCCAT2000* transform sharpened *RGB* matrix.
    :type RGB: array_like
    :return: Compressed *CMCCAT2000* transform sharpened *RGB* matrix.
    :rtype: ndarray (3, 1)

    References:

    -  **Mark D. Fairchild**, *Color Appearance Models, 2nd Edition*, \
    The Wiley-IS&T Series in Imaging Science and Technology, published 19 November 2004, ISBN-13: 978-0470012161, Pages 265-277.
    -  **Stephen Westland, Caterina Ripamonti, Vien Cheung**, *Computational Colour Science Using MATLAB, 2nd Edition*, \
    The Wiley-IS&T Series in Imaging Science and Technology, published July 2012, ISBN-13: 978-0-470-66569-5, Pages 88-92.
    -  `The CIECAM02 Color Appearance Model <http://rit-mcsl.org/fairchild/PDFs/PRO19.pdf>`_  (Last accessed 30 July 2014)
    """

    # TODO: Check for negative values and their handling.
    RGBc = (((400. * (FL * RGB / 100) ** 0.42) / (27.13 + (FL * RGB / 100) ** 0.42)) + 0.1).reshape((3, 1))
    return RGBc


def apply_reverse_post_adaptation_non_linear_response_compression(RGB, FL):
    RGBp = (np.sign(RGB - 0.1) * \
            (100. / FL) * ((27.13 * np.abs(RGB - 0.1)) / (400 - np.abs(RGB - 0.1))) ** (1 / 0.42))
    return RGBp


def get_forward_opponent_colour_dimensions(RGB):
    """
    Returns opponent colour dimensions from given compressed *CMCCAT2000* transform sharpened *RGB* matrix \
    for forward *CIECAM02* implementation

    Usage::

        >>> get_forward_opponent_colour_dimensions(np.array([7.9463202, 7.94711528,7.94899595]))
        (-0.000624112068243, -0.000506270106773)

    :param RGB: Compressed *CMCCAT2000* transform sharpened *RGB* matrix.
    :type RGB: array_like
    :return: Opponent colour dimensions
    :rtype: tuple

    References:

    -  **Mark D. Fairchild**, *Color Appearance Models, 2nd Edition*, \
    The Wiley-IS&T Series in Imaging Science and Technology, published 19 November 2004, ISBN-13: 978-0470012161, Pages 265-277.
    -  **Stephen Westland, Caterina Ripamonti, Vien Cheung**, *Computational Colour Science Using MATLAB, 2nd Edition*, \
    The Wiley-IS&T Series in Imaging Science and Technology, published July 2012, ISBN-13: 978-0-470-66569-5, Pages 88-92.
    -  `The CIECAM02 Color Appearance Model <http://rit-mcsl.org/fairchild/PDFs/PRO19.pdf>`_  (Last accessed 30 July 2014)
    """

    R, G, B = np.ravel(RGB)

    a = R - 12. * G / 11. + B / 11.
    b = (R + G - 2. * B) / 9.

    return a, b


def get_reverse_opponent_colour_dimensions(p, hr):
    p1, p2, p3 = p

    sin_hr, cos_hr = math.sin(hr), math.cos(hr)
    p4 = p1 / sin_hr
    p5 = p1 / cos_hr
    n = p2 * (2. + p3) * (460. / 1403.)

    if abs(sin_hr) >= abs(cos_hr):
        b = n / (p4 + (2. + p3) * (220. / 1403.) * (cos_hr / sin_hr) - (27. / 1403.) + p3 * (6300. / 1403))
        a = b * (cos_hr / sin_hr)
    else:
        a = n / (p5 + (2. + p3) * (220. / 1403.) - ((27. / 1403.) - p3 * (6300. / 1403.)) * (sin_hr / cos_hr))
        b = a * (sin_hr / cos_hr)
    return a, b


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

    References:

    -  **Mark D. Fairchild**, *Color Appearance Models, 2nd Edition*, \
    The Wiley-IS&T Series in Imaging Science and Technology, published 19 November 2004, ISBN-13: 978-0470012161, Pages 265-277.
    -  **Stephen Westland, Caterina Ripamonti, Vien Cheung**, *Computational Colour Science Using MATLAB, 2nd Edition*, \
    The Wiley-IS&T Series in Imaging Science and Technology, published July 2012, ISBN-13: 978-0-470-66569-5, Pages 88-92.
    -  `The CIECAM02 Color Appearance Model <http://rit-mcsl.org/fairchild/PDFs/PRO19.pdf>`_  (Last accessed 30 July 2014)
    """

    hi = HUE_DATA_FOR_HUE_QUADRATURE.get("hi")
    ei = HUE_DATA_FOR_HUE_QUADRATURE.get("ei")
    Hi = HUE_DATA_FOR_HUE_QUADRATURE.get("Hi")

    hp = h + 360 if h < hi[0] else h
    index = bisect.bisect_left(hi, hp) - 1

    H = Hi[index] + \
        ((100 * (hp - hi[index]) / ei[index]) / ((hp - hi[index]) / ei[index] + (hi[index + 1] - hp) / ei[index + 1]))

    return H


def get_forward_eccentricity_factor(h):
    """
    Returns the eccentricity factor from given hue angle for forward *CIECAM02* implementation.

    Usage::

        >>> get_forward_eccentricity_factor(-140.951567342)
        1.17400547285

    :param h: Hue angle in degrees.
    :type h: float
    :return: Eccentricity factor.
    :rtype: float

    References:

    -  **Mark D. Fairchild**, *Color Appearance Models, 2nd Edition*, \
    The Wiley-IS&T Series in Imaging Science and Technology, published 19 November 2004, ISBN-13: 978-0470012161, Pages 265-277.
    -  **Stephen Westland, Caterina Ripamonti, Vien Cheung**, *Computational Colour Science Using MATLAB, 2nd Edition*, \
    The Wiley-IS&T Series in Imaging Science and Technology, published July 2012, ISBN-13: 978-0-470-66569-5, Pages 88-92.
    -  `The CIECAM02 Color Appearance Model <http://rit-mcsl.org/fairchild/PDFs/PRO19.pdf>`_  (Last accessed 30 July 2014)
    """

    et = 1. / 4. * (math.cos(2. + h * math.pi / 180.) + 3.8)
    return et


def get_reverse_eccentricity_factor(h):
    et = (math.cos(h * math.pi / 180. + 2) + 3.8) / 4.
    return et


def get_forward_achromatic_response(RGB, Nbb):
    """
    Returns the achromatic response *A* from given compressed *CMCCAT2000* transform sharpened *RGB* matrix
    and *Nbb* chromatic induction factor for forward *CIECAM02* implementation

    Usage::

        >>> get_forward_achromatic_response(np.array([7.9463202, 7.94711528,7.94899595]), 1.0003040045593807)
        23.9394809667

    :param RGB: Compressed *CMCCAT2000* transform sharpened *RGB* matrix.
    :type RGB: array_like
    :param Nbb: Chromatic induction factor.
    :type Nbb: float
    :return: Achromatic response.
    :rtype: float

    References:

    -  **Mark D. Fairchild**, *Color Appearance Models, 2nd Edition*, \
    The Wiley-IS&T Series in Imaging Science and Technology, published 19 November 2004, ISBN-13: 978-0470012161, Pages 265-277.
    -  **Stephen Westland, Caterina Ripamonti, Vien Cheung**, *Computational Colour Science Using MATLAB, 2nd Edition*, \
    The Wiley-IS&T Series in Imaging Science and Technology, published July 2012, ISBN-13: 978-0-470-66569-5, Pages 88-92.
    -  `The CIECAM02 Color Appearance Model <http://rit-mcsl.org/fairchild/PDFs/PRO19.pdf>`_  (Last accessed 30 July 2014)
    """

    R, G, B = np.ravel(RGB)

    A = (2. * R + G + (1. / 20.) * B - 0.305) * Nbb
    return A


def get_reverse_achromatic_response(Aw, J, c, z):
    A = Aw * (J / 100.) ** (1. / (c * z))
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

    References:

    -  **Mark D. Fairchild**, *Color Appearance Models, 2nd Edition*, \
    The Wiley-IS&T Series in Imaging Science and Technology, published 19 November 2004, ISBN-13: 978-0470012161, Pages 265-277.
    -  **Stephen Westland, Caterina Ripamonti, Vien Cheung**, *Computational Colour Science Using MATLAB, 2nd Edition*, \
    The Wiley-IS&T Series in Imaging Science and Technology, published July 2012, ISBN-13: 978-0-470-66569-5, Pages 88-92.
    -  `The CIECAM02 Color Appearance Model <http://rit-mcsl.org/fairchild/PDFs/PRO19.pdf>`_  (Last accessed 30 July 2014)
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

    References:

    -  **Mark D. Fairchild**, *Color Appearance Models, 2nd Edition*, \
    The Wiley-IS&T Series in Imaging Science and Technology, published 19 November 2004, ISBN-13: 978-0470012161, Pages 265-277.
    -  **Stephen Westland, Caterina Ripamonti, Vien Cheung**, *Computational Colour Science Using MATLAB, 2nd Edition*, \
    The Wiley-IS&T Series in Imaging Science and Technology, published July 2012, ISBN-13: 978-0-470-66569-5, Pages 88-92.
    -  `The CIECAM02 Color Appearance Model <http://rit-mcsl.org/fairchild/PDFs/PRO19.pdf>`_  (Last accessed 30 July 2014)
    """

    Q = (4. / c) * math.sqrt(J / 100.) * (Aw + 4) * FL ** 0.25
    return Q


def get_forward_temporary_magniture_quantity(Nc, Ncb, et, a, b, RGBa):
    Ra, Ga, Ba = np.ravel(RGBa)
    t = ((50000. / 13.) * Nc * Ncb) * (et * (a ** 2 + b ** 2) ** 0.5) / (Ra + Ga + 21. * Ba / 20.)
    return t


def get_reverse_temporary_magniture_quantity(C, J, n):
    t = (C / (math.sqrt(J / 100.) * (1.64 - 0.29 ** n) ** 0.73)) ** (1. / 0.9)
    return t


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
        >>> RGBa = np.array([7.9463202, 7.94711528,7.94899595])
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

    References:

    -  **Mark D. Fairchild**, *Color Appearance Models, 2nd Edition*, \
    The Wiley-IS&T Series in Imaging Science and Technology, published 19 November 2004, ISBN-13: 978-0470012161, Pages 265-277.
    -  **Stephen Westland, Caterina Ripamonti, Vien Cheung**, *Computational Colour Science Using MATLAB, 2nd Edition*, \
    The Wiley-IS&T Series in Imaging Science and Technology, published July 2012, ISBN-13: 978-0-470-66569-5, Pages 88-92.
    -  `The CIECAM02 Color Appearance Model <http://rit-mcsl.org/fairchild/PDFs/PRO19.pdf>`_  (Last accessed 30 July 2014)
    """

    t = get_forward_temporary_magniture_quantity(Nc, Ncb, et, a, b, RGBa)
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

    References:

    -  **Mark D. Fairchild**, *Color Appearance Models, 2nd Edition*, \
    The Wiley-IS&T Series in Imaging Science and Technology, published 19 November 2004, ISBN-13: 978-0470012161, Pages 265-277.
    -  **Stephen Westland, Caterina Ripamonti, Vien Cheung**, *Computational Colour Science Using MATLAB, 2nd Edition*, \
    The Wiley-IS&T Series in Imaging Science and Technology, published July 2012, ISBN-13: 978-0-470-66569-5, Pages 88-92.
    -  `The CIECAM02 Color Appearance Model <http://rit-mcsl.org/fairchild/PDFs/PRO19.pdf>`_  (Last accessed 30 July 2014)
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

    References:

    -  **Mark D. Fairchild**, *Color Appearance Models, 2nd Edition*, \
    The Wiley-IS&T Series in Imaging Science and Technology, published 19 November 2004, ISBN-13: 978-0470012161, Pages 265-277.
    -  **Stephen Westland, Caterina Ripamonti, Vien Cheung**, *Computational Colour Science Using MATLAB, 2nd Edition*, \
    The Wiley-IS&T Series in Imaging Science and Technology, published July 2012, ISBN-13: 978-0-470-66569-5, Pages 88-92.
    -  `The CIECAM02 Color Appearance Model <http://rit-mcsl.org/fairchild/PDFs/PRO19.pdf>`_  (Last accessed 30 July 2014)
    """

    s = 100. * (M / Q) ** 0.5
    return s


def get_p(Nc, Ncb, et, t, A, Nbb):
    p1 = ((50000. / 13.) * Nc * Ncb * et) / t
    p2 = A / Nbb + 0.305
    p3 = 21. / 20.

    return p1, p2, p3


def get_post_adaptation_non_linear_response_compression_matrix(p2, a, b):
    Ra = (460. * p2 + 451. * a + 288. * b) / 1403.
    Ga = (460. * p2 - 891. * a - 261. * b) / 1403.
    Ba = (460. * p2 - 220. * a - 6300. * b) / 1403.

    return np.ravel([Ra, Ga, Ba]).reshape((3, 1))


def XYZ_to_CIECAM02(XYZ,
                    XYZw,
                    LA,
                    Yb,
                    surround=CIECAM02_VIEWING_CONDITION_PARAMETERS.get("Average"),
                    discount_illuminant=False):
    """
    Converts given *CIE XYZ* colourspace matrix to *CIECAM02* representation.

    Usage::

        >>> XYZ = np.array([19.01, 20.00, 21.78])
        >>> XYZw = np.array([95.05, 100.00, 108.88])
        >>> LA = 318.31
        >>> Yb = 20.0
        >>> XYZ_to_CIECAM02(XYZ, XYZw, LA, Yb)
        CIECAM02_JChQMsH(J=41.731091132513917, C=0.1047077571711053, h=-140.9515673417281, Q=195.37132596607671, M=0.1088421756692261, s=2.3603053739204447, H=278.06073585662813)

    :param XYZ: *CIE XYZ* colourspace stimulus matrix.
    :type XYZ: array_like
    :param XYZw: *CIE XYZ* colourspace whitepoint matrix.
    :type XYZw: array_like
    :param LA: Adapting field *luminance* in cd/m2.
    :type LA: float
    :param Yb: Adapting field *Y* tristimulus value.
    :type Yb: float
    :param surround: Surround viewing conditions.
    :type surround: CIECAM02_Surround
    :param discount_illuminant: Discount the illuminant.
    :type discount_illuminant: bool
    :return: *CIECAM02* representation.
    :rtype: CIECAM02_JChQMsH

    :warning: The arguments domains of that definition are non standard!
    :note: Input *CIE XYZ* colourspace matrix is in domain [0, 100].
    :note: Input *CIE XYZw* colourspace matrix is in domain [0, 100].

    References:

    -  **Mark D. Fairchild**, *Color Appearance Models, 2nd Edition*, \
    The Wiley-IS&T Series in Imaging Science and Technology, published 19 November 2004, ISBN-13: 978-0470012161, Pages 265-277.
    -  **Stephen Westland, Caterina Ripamonti, Vien Cheung**, *Computational Colour Science Using MATLAB, 2nd Edition*, \
    The Wiley-IS&T Series in Imaging Science and Technology, published July 2012, ISBN-13: 978-0-470-66569-5, Pages 88-92.
    -  `The CIECAM02 Color Appearance Model <http://rit-mcsl.org/fairchild/PDFs/PRO19.pdf>`_  (Last accessed 30 July 2014)
    """

    XYZ = np.array(XYZ).reshape((3, 1))
    XYZw = np.array(XYZw).reshape((3, 1))
    X, Y, Z = np.ravel(XYZ)
    Xw, Yw, Zw = np.ravel(XYZw)

    n, FL, Nbb, Ncb, z = get_viewing_condition_dependent_parameters(Yb, Yw, LA)

    # Converting *CIE XYZ* colourspace matrices to *CMCCAT2000* transform sharpened *RGB* values.
    RGB = np.dot(colour.colorimetry.chromatic_adaptation.CAT02_CAT, XYZ)
    RGBw = np.dot(colour.colorimetry.chromatic_adaptation.CAT02_CAT, XYZw)

    # Computing degree of adaptation *D*.
    D = get_degree_of_adaptation(surround.F, LA) if not discount_illuminant else 1.

    # Computing full chromatic adaptation.
    RGBc = apply_forward_full_chromatic_adaptation(RGB, RGBw, Yw, D)
    RGBwc = apply_forward_full_chromatic_adaptation(RGBw, RGBw, Yw, D)

    # Converting to *Hunt-Pointer-Estevez* colourspace.
    RGBp = RGB_to_HPE(RGBc)
    RGBpw = RGB_to_HPE(RGBwc)

    # Applying forward post-adaptation non linear response compression.
    RGBa = apply_forward_post_adaptation_non_linear_response_compression(RGBp, FL)
    RGBaw = apply_forward_post_adaptation_non_linear_response_compression(RGBpw, FL)

    # Converting to preliminary cartesian coordinates.
    a, b = get_forward_opponent_colour_dimensions(RGBa)
    h = math.degrees(math.atan2(b, a))

    # Computing hue quadrature *hue*.
    H = get_hue_quadrature(h)

    # Computing eccentricity factor *et*.
    et = get_forward_eccentricity_factor(h)

    # Computing achromatic responses for the stimulus and the whitepoint.
    A = get_forward_achromatic_response(RGBa, Nbb)
    Aw = get_forward_achromatic_response(RGBaw, Nbb)

    # Computing the correlate of *Lightness* *J*.
    J = get_lightness_correlate(A, Aw, surround.c, z)

    # Computing the correlate of *brightness* *Q*.
    Q = get_brightness_correlate(surround.c, J, Aw, FL)

    # Computing the correlate of *chroma* *C*.
    C = get_chroma_correlate(J, n, surround.Nc, Ncb, et, a, b, RGBa)

    # Computing the correlate of *colourfulness* *M*.
    M = get_colourfulness_correlate(C, FL)

    # Computing the correlate of *saturation* *s*.
    s = get_saturation_correlate(M, Q)

    return CIECAM02_JCHQMSH(J, C, h, Q, M, s, H)


def CIECAM02_to_XYZ(JChQMsH,
                    XYZw,
                    LA,
                    Yb,
                    surround=CIECAM02_VIEWING_CONDITION_PARAMETERS.get("Average"),
                    discount_illuminant=False):
    XYZw = np.array(XYZw).reshape((3, 1))
    Xw, Yw, Zw = np.ravel(XYZw)

    n, FL, Nbb, Ncb, z = get_viewing_condition_dependent_parameters(Yb, Yw, LA)

    J, C, h, Q, M, s, H = JChQMsH


    # Converting *CIE XYZ* colourspace matrices to *CMCCAT2000* transform sharpened *RGB* values.
    RGBw = np.dot(colour.colorimetry.chromatic_adaptation.CAT02_CAT, XYZw)

    # Computing degree of adaptation *D*.
    D = get_degree_of_adaptation(surround.F, LA) if not discount_illuminant else 1.

    # Calculation full chromatic adaptation.
    RGBwc = apply_forward_full_chromatic_adaptation(RGBw, RGBw, Yw, D)

    # Converting to *Hunt-Pointer-Estevez* colourspace.
    RGBpw = RGB_to_HPE(RGBwc)

    # Applying post-adaptation non linear response compression.
    RGBaw = apply_forward_post_adaptation_non_linear_response_compression(RGBpw, FL)

    # Computing achromatic responses for the stimulus and the whitepoint.
    Aw = get_forward_achromatic_response(RGBaw, Nbb)

    # Computing temporary magnitude quantity *t*.
    t = get_reverse_temporary_magniture_quantity(C, J, n)

    # Computing eccentricity factor *et*.
    et = get_reverse_eccentricity_factor(h)

    # Computing achromatic response *A* for the stimulus.
    A = get_reverse_achromatic_response(Aw, J, surround.c, z)

    # Computing *p1* to *p3*.
    p1, p2, p3 = get_p(surround.Nc, Ncb, et, t, A, Nbb)

    # Computing opponent colour dimensions *a* and *b*.
    hr = math.radians(h)
    a, b = get_reverse_opponent_colour_dimensions((p1, p2, p3), hr)

    # Computing post-adaptation non linear response compression matrix.
    RGBa = get_post_adaptation_non_linear_response_compression_matrix(p2, a, b)

    # Applying reverse post-adaptation non linear response compression.
    RGBp = apply_reverse_post_adaptation_non_linear_response_compression(RGBa, FL)

    # Converting to *Hunt-Pointer-Estevez* colourspace.
    RGBc = HPE_to_RGB(RGBp)

    # Applying reverse full chromatic adaptation.
    RGB = apply_reverse_full_chromatic_adaptation(RGBc, RGBw, Yw, D)

    # Converting *CMCCAT2000* transform sharpened *RGB* values to *CIE XYZ* colourspace matrices.
    XYZ = np.dot(CAT02_CAT_INVERSE, RGB)

    return XYZ

# print XYZ_to_CIECAM02([11.18882, 9.30994, 3.21014],
# [96.4296, 100, 82.49],
# Yb=20,
# LA=1000)

print XYZ_to_CIECAM02([19.01, 20.00, 21.78],
                      [95.05, 100.00, 108.88],
                      LA=318.31,
                      Yb=20.0)

# CIECAM02_JChQMsH(J=41.731091132513917, C=0.1047077571711053, h=-140.9515673417281, Q=195.37132596607671, M=0.1088421756692261, s=2.3603053739204447, H=278.06073585662813)
# CIECAM02_JChQMsH(J=41.731091132513917, C=0.1047077571711053, h=-140.9515673417281, Q=195.37132596607671, M=0.1088421756692261, s=2.3603053739204447, H=278.06073585662813)


print CIECAM02_to_XYZ(CIECAM02_JCHQMSH(J=41.731091132513917,
                                       C=0.1047077571711053,
                                       h=-140.9515673417281,
                                       Q=195.37132596607671,
                                       M=0.1088421756692261,
                                       s=2.3603053739204447,
                                       H=278.06073585662813),
                      [95.05, 100.00, 108.88],
                      LA=318.31,
                      Yb=20.0)
