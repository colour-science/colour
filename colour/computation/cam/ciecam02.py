# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**ciecam02.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package *CIECAM02* objects.

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

__all__ = []

CIECAM02_SURROUND_FCNC = namedtuple("Surround", ("F", "c", "Nc"))

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
    k = 1. / (5. * LA + 1)
    k4 = k ** 4
    FL = 0.2 * k4 * (5. * LA) + 0.1 * (1. - k4) ** 2 * (5. * LA) ** (1. / 3.)
    return FL


def get_induction_factors(n):
    N_bb = N_cb = 0.725 * (1. / n) ** 0.2
    return N_bb, N_cb


def get_base_exponential_nonlinearity(n):
    z = 1.48 + math.sqrt(n)
    return z


def get_degree_of_adapation(F, LA):
    D = F * (1. - (1. / 3.6) * numpy.exp((-LA - 42.) / 92.))
    return D


def get_full_chromatic_adaptation(RGB, RGBw, Yw, D):
    R, G, B = numpy.ravel(RGB)
    Rw, Gw, Bw = numpy.ravel(RGBw)

    equation = lambda x, y: ((Yw * D / y) + 1 - D) * x

    Rc = equation(R, Rw)
    Gc = equation(G, Gw)
    Bc = equation(B, Bw)

    return numpy.array([Rc, Gc, Bc]).reshape((3, 1))


def RGB_to_HPE(RGB):
    pyb = numpy.dot(numpy.dot(HPE, CAT02_CAT_INVERSE), RGB)
    return pyb


def apply_post_adaptation_non_linear_response_compression(pyb, FL):
    return ((400. * (FL * pyb / 100) ** 0.42) / (27.13 + (FL * pyb / 100) ** 0.42)) + 0.1


def get_opponent_color_dimensions(pyb):
    p, y, b = numpy.ravel(pyb)

    a = p - 12. * y / 11. + b / 11.
    b = (p + y - 2. * b) / 9.

    return a, b


def get_hue_quadrature(h):
    hi = HUE_DATA_FOR_HUE_QUADRATURE.get("hi")
    ei = HUE_DATA_FOR_HUE_QUADRATURE.get("ei")
    Hi = HUE_DATA_FOR_HUE_QUADRATURE.get("Hi")

    hp = h + 360 if h < hi[0] else h
    index = bisect.bisect_left(hi, hp) - 1

    H = Hi[index] + \
        ((100 * (hp - hi[index]) / ei[index]) / ((hp - hi[index]) / ei[index] + (hi[index + 1] - hp) / ei[index + 1]))

    return H

def get_eccentricity_factor(hue):
    return 1. / 4. * (math.cos(2. + hue * math.pi / 180.) + 3.8)


def get_achromatic_response(pyb, Nbb):
    p, y, b = numpy.ravel(pyb)

    A = (2. * p + y + (1. / 20.) * b - 0.305) * Nbb
    return A


def get_lightness_correlate(A, Aa, c, z):
    J = 100. * (A / Aa) ** (c * z)
    return J


def get_brightness_correlate(c, J, Aw, FL):
    Q = (4. / c) * math.sqrt(J / 100.) * (Aw + 4) * FL ** 0.25
    return Q


def get_chroma_correlate(J, n, Nc, Ncb, et, a, b, pyba):
    pa, ya, ba = numpy.ravel(pyba)

    t = ((50000. / 13.) * Nc * Ncb) * (et * (a ** 2 + b ** 2) ** 0.5) / (pa + ya + 21. * ba / 20.)
    C = t ** 0.9 * (J / 100.) ** 0.5 * (1.64 - 0.29 ** n) ** 0.73

    return C


def get_colourfulness_correlate(C, FL):
    M = C * FL ** 0.25
    return M


def get_saturation_correlate(M, Q):
    s = 100. * (M / Q) ** 0.5
    return s


def XYZ_to_CIECAM02(XYZ,
                     XYZw,
                     Yb,
                     LA,
                     surround=CIECAM02_VIEWING_CONDITION_PARAMETERS.get("Average"),
                     discount_illuminant=False):
    XYZ = numpy.array(XYZ).reshape((3, 1))
    XYZw = numpy.array(XYZw).reshape((3, 1))
    X, Y, Z = numpy.ravel(XYZ)
    Xw, Yw, Zw = numpy.ravel(XYZw)

    n = Yb / Yw

    FL = get_luminance_level_adaptation_factor(LA)
    Nbb, Ncb = get_induction_factors(n)
    z = get_base_exponential_nonlinearity(n)

    # Step 1: Converting *CIE XYZ* colourspace matrices to sharpened *RGB* values.
    RGB = numpy.dot(colour.computation.chromatic_adaptation.CAT02_CAT, XYZ)
    RGBw = numpy.dot(colour.computation.chromatic_adaptation.CAT02_CAT, XYZw)

    # Step 2: Calculating degree of adaptation *D*.
    D = get_degree_of_adapation(surround.F, LA) if not discount_illuminant else 1.

    # Step 3: Calculation full chromatic adaptation.
    RGBc = get_full_chromatic_adaptation(RGB, RGBw, Yw, D)
    RGBwc = get_full_chromatic_adaptation(RGBw, RGBw, Yw, D)

    # Step 4: Converting to *Hunt-Pointer-Estevez* colourspace.
    RGBp = RGB_to_HPE(RGBc)
    RGBpw = RGB_to_HPE(RGBwc)

    # Step 5: Apply post-adaptation non linear response compression.
    RGBa = apply_post_adaptation_non_linear_response_compression(RGBp, FL)
    RGBaw = apply_post_adaptation_non_linear_response_compression(RGBpw, FL)

    # Step 6: Convert to Preliminary Cartesian coordinates.
    a, b = get_opponent_color_dimensions(RGBa)
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


print XYZ_to_CIECAM02([11.18882, 9.30994, 3.21014],
                       [96.4296, 100, 82.49],
                       Yb=20,
                       LA=1000)

print XYZ_to_CIECAM02([19.31, 23.93, 10.14],
                       [98.88, 90, 32.03],
                       Yb=18,
                       LA=200)