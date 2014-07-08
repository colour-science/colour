#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**deprecated.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package colour deprecated objects.
    Those objects are only provided for convenience and completeness.

**Others:**

"""

from __future__ import unicode_literals

import math
import numpy

import colour.utilities.verbose

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["RGB_to_HSV",
           "HSV_to_RGB",
           "RGB_to_HSL",
           "HSL_to_RGB",
           "RGB_to_CMY",
           "CMY_to_RGB",
           "CMY_to_CMYK",
           "CMYK_to_CMY",
           "RGB_to_HEX",
           "HEX_to_RGB"]

LOGGER = colour.utilities.verbose.install_logger()


def RGB_to_HSV(RGB):
    """
    Converts from *RGB* colourspace to *HSV* colourspace.

    References:

    -  http://alvyray.com/Papers/CG/color78.pdf
    -  http://www.easyrgb.com/index.php?X=MATH&H=20#text20

    Usage::

        >>> RGB_to_HSV(numpy.matrix([0.49019607843137253, 0.9803921568627451, 0.25098039215686274]).reshape((3, 1)))
        matrix([[ 0.27867384],
                [ 0.744     ],
                [ 0.98039216]])

    :param RGB: *RGB* colourspace matrix.
    :type RGB: matrix (3x1)
    :return: *HSV* matrix.
    :rtype: matrix (3x1)

    :note: *RGB* is in domain [0, 1].
    :note: *HSV* is in domain [0, 1].
    """

    R, G, B = numpy.ravel(RGB)

    minimum = min(R, G, B)
    maximum = max(R, G, B)
    delta = maximum - minimum

    V = maximum

    if delta == 0:
        H = 0.
        S = 0.
    else:

        S = delta / maximum

        delta_R = (((maximum - R) / 6.) + (delta / 2.)) / delta
        delta_G = (((maximum - G) / 6.) + (delta / 2.)) / delta
        delta_B = (((maximum - B) / 6.) + (delta / 2.)) / delta

        if R == maximum:
            H = delta_B - delta_G
        elif G == maximum:
            H = (1. / 3.) + delta_R - delta_B
        elif B == maximum:
            H = (2. / 3.) + delta_G - delta_R

        if H < 0:
            H += 1
        if H > 1:
            H -= 1

    return numpy.matrix([H, S, V]).reshape((3, 1))


def HSV_to_RGB(HSV):
    """
    Converts from *HSV* colourspace to *RGB* colourspace.

    References:

    -  http://alvyray.com/Papers/CG/color78.pdf
    -  http://www.easyrgb.com/index.php?X=MATH&H=21#text21

    Usage::

        >>> HSV_to_RGB(numpy.matrix([0.27867384, 0.744, 0.98039216]).reshape((3, 1)))
        matrix([[ 0.49019606]
                [ 0.98039216]
                [ 0.25098039]])

    :param HSV: *HSV* colourspace matrix.
    :type HSV: matrix (3x1)
    :return: *RGB* matrix.
    :rtype: matrix (3x1)

    :note: *HSV* is in domain [0, 1].
    :note: *RGB* is in domain [0, 1].
    """

    H, S, V = numpy.ravel(HSV)

    if S == 0:
        R = V
        G = V
        B = V
    else:
        h = H * 6.
        if h == 6.:
            h = 0

        i = math.floor(h)
        j = V * (1. - S)
        k = V * (1. - S * (h - i))
        l = V * (1. - S * (1. - (h - i)))
        if i == 0:
            R = V
            G = l
            B = j
        elif i == 1:
            R = k
            G = V
            B = j
        elif i == 2:
            R = j
            G = V
            B = l
        elif i == 3:
            R = j
            G = k
            B = V
        elif i == 4:
            R = l
            G = j
            B = V
        elif i == 5:
            R = V
            G = j
            B = k

    return numpy.matrix([R, G, B]).reshape((3, 1))


def RGB_to_HSL(RGB):
    """
    Converts from *RGB* colourspace to *HSL* colourspace.

    References:

    -  http://alvyray.com/Papers/CG/color78.pdf
    -  http://www.easyrgb.com/index.php?X=MATH&H=18#text18

    Usage::

        >>> RGB_to_HSL(numpy.matrix([0.49019607843137253, 0.9803921568627451, 0.25098039215686274]).reshape((3, 1)))
        matrix([[ 0.27867384]
                [ 0.94897959]
                [ 0.61568627]])

    :param RGB: *RGB* colourspace matrix.
    :type RGB: matrix (3x1)
    :return: *HSL* matrix.
    :rtype: matrix (3x1)

    :note: *RGB* is in domain [0, 1].
    :note: *HSL* is in domain [0, 1].
    """

    R, G, B = numpy.ravel(RGB)

    minimum = min(R, G, B)
    maximum = max(R, G, B)
    delta = maximum - minimum

    L = (maximum + minimum) / 2

    if delta == 0:
        H = 0.
        S = 0.
    else:

        S = delta / (maximum + minimum) if L < 0.5 else delta / (2 - maximum - minimum)

        delta_R = (((maximum - R) / 6.) + (delta / 2.)) / delta
        delta_G = (((maximum - G) / 6.) + (delta / 2.)) / delta
        delta_B = (((maximum - B) / 6.) + (delta / 2.)) / delta

        if R == maximum:
            H = delta_B - delta_G
        elif G == maximum:
            H = (1. / 3.) + delta_R - delta_B
        elif B == maximum:
            H = (2. / 3.) + delta_G - delta_R

        if H < 0:
            H += 1
        if H > 1:
            H -= 1

    return numpy.matrix([H, S, L]).reshape((3, 1))


def HSL_to_RGB(HSL):
    """
    Converts from *HSL* colourspace to *RGB* colourspace.

    References:

    -  http://alvyray.com/Papers/CG/color78.pdf
    -  http://www.easyrgb.com/index.php?X=MATH&H=19#text19

    Usage::

        >>> HSL_to_RGB(numpy.matrix([0.27867384, 0.94897959, 0.61568627]).reshape((3, 1)))
        matrix([[ 0.49019605]
                [ 0.98039216]
                [ 0.25098038]])

    :param HSL: *HSL* colourspace matrix.
    :type HSL: matrix (3x1)
    :return: *RGB* matrix.
    :rtype: matrix (3x1)

    :note: *HSL* is in domain [0, 1].
    :note: *RGB* is in domain [0, 1].
    """

    H, S, L = numpy.ravel(HSL)

    if S == 1:
        R = L
        G = L
        B = L
    else:
        def H_to_RGB(vi, vj, vH):
            if vH < 0:
                vH += 1
            if vH > 1:
                vH -= 1
            if 6 * vH < 1:
                return vi + (vj - vi) * 6. * vH
            if 2 * vH < 1:
                return vj
            if 3 * vH < 2:
                return vi + (vj - vi) * ((2. / 3.) - vH) * 6.
            return vi

        j = L * (1. + S) if L < 0.5 else (L + S) - (S * L)
        i = 2 * L - j

        R = H_to_RGB(i, j, H + (1. / 3.))
        G = H_to_RGB(i, j, H)
        B = H_to_RGB(i, j, H - (1. / 3.))

    return numpy.matrix([R, G, B]).reshape((3, 1))


def RGB_to_CMY(RGB):
    """
    Converts from *RGB* colourspace to *CMY* colourspace.

    References:

    -  http://www.easyrgb.com/index.php?X=MATH&H=11#text11

    Usage::

        >>> RGB_to_CMY(numpy.matrix([0.49019607843137253, 0.9803921568627451, 0.25098039215686274]).reshape((3, 1)))
        matrix([[ 0.50980392]
                [ 0.01960784]
                [ 0.74901961]])

    :param RGB: *RGB* colourspace matrix.
    :type RGB: matrix (3x1)
    :return: *CMY* matrix.
    :rtype: matrix (3x1)
    """

    R, G, B = numpy.ravel(RGB)
    return numpy.matrix([1. - R, 1. - G, 1. - B]).reshape((3, 1))


def CMY_to_RGB(CMY):
    """
    Converts from *CMY* colourspace to *CMY* colourspace.

    References:

    -  http://www.easyrgb.com/index.php?X=MATH&H=12#text12

    Usage::

        >>> CMY_to_RGB(numpy.matrix([0.50980392, 0.01960784, 0.74901961]).reshape((3, 1)))
        matrix([[ 0.49019608]
                [ 0.98039216]
                [ 0.25098039]])

    :param CMY: *CMY* colourspace matrix.
    :type CMY: matrix (3x1)
    :return: *RGB* matrix.
    :rtype: matrix (3x1)
    """

    C, M, Y = numpy.ravel(CMY)
    return numpy.matrix([1. - C, 1. - M, 1. - Y]).reshape((3, 1))


def CMY_to_CMYK(CMY):
    """
    Converts from *CMY* colourspace to *CMYK* colourspace.

    References:

    -  http://www.easyrgb.com/index.php?X=MATH&H=13#text13

    Usage::

        >>> CMY_to_CMYK(numpy.matrix([0.50980392, 0.01960784, 0.74901961]).reshape((3, 1)))
        matrix([[ 0.5       ]
                [ 0.        ]
                [ 0.744     ]
                [ 0.01960784]])

    :param CMY: *CMY* colourspace matrix.
    :type CMY: matrix (3x1)
    :return: *CMYK* matrix.
    :rtype: matrix (4x1)
    """

    C, M, Y = numpy.ravel(CMY)

    K = 1.

    if C < K:
        K = C
    if M < K:
        K = M
    if Y < K:
        K = Y
    if K == 1:
        C = 0.
        M = 0.
        Y = 0.
    else:
        C = (C - K) / (1. - K)
        M = (M - K) / (1. - K)
        Y = (Y - K) / (1. - K)

    return numpy.matrix([C, M, Y, K]).reshape((4, 1))


def CMYK_to_CMY(CMYK):
    """
    Converts from *CMYK* colourspace to *CMY* colourspace.

    References:

    -  http://www.easyrgb.com/index.php?X=MATH&H=14#text14

    Usage::

        >>> CMYK_to_CMY(numpy.matrix([0.5, 0.,0.744, 0.01960784]).reshape((4, 1)))
        matrix([[ 0.50980392]
                [ 0.01960784]
                [ 0.74901961]])

    :param CMYK: *CMYK* colourspace matrix.
    :type CMYK: matrix (4x1)
    :return: *CMY* matrix.
    :rtype: matrix (3x1)
    """

    C, M, Y, K = numpy.ravel(CMYK)

    return numpy.matrix([C * (1. - K) + K, M * (1. - K) + K, Y * (1. - K) + K]).reshape((3, 1))


def RGB_to_HEX(RGB):
    """
    Converts from *RGB* colourspace to hex triplet representation.

    Usage::

        >>> RGB_to_HEX(numpy.matrix([0.66666667, 0.86666667, 1.]).reshape((3, 1)))
        #aaddff

    :param RGB: *RGB* colourspace matrix.
    :type RGB: matrix (3x1)
    :return: Hex triplet representation.
    :rtype: unicode

    :note: *RGB* is in domain [0, 1].
    """

    RGB = numpy.ravel(RGB)
    R, G, B = map(int, RGB * 255.)
    return "#{0:02x}{1:02x}{2:02x}".format(R, G, B)


def HEX_to_RGB(HEX):
    """
    Converts from hex triplet representation to *RGB* colourspace.

    Usage::

        >>> HEX_to_RGB("#aaddff")
        [[ 0.66666667]
        [ 0.86666667]
        [ 1.        ]]

    :param HEX: Hex triplet representation.
    :type HEX: unicode
    :return: *RGB* colourspace matrix.
    :rtype: matrix (3x1)

    :note: *RGB* is in domain [0, 1].
    """

    HEX = HEX.lstrip("#")
    length = len(HEX)
    return numpy.matrix([int(HEX[i:i + length / 3], 16) for i in range(0, length, length / 3)]).reshape((3, 1)) / 255.