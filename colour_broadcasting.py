#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals

import numpy as np

from colour.adaptation import VON_KRIES_CAT
from colour.utilities import tstack, tsplit, warning

CIE1994_XYZ_TO_RGB_MATRIX = VON_KRIES_CAT
CIE1994_RGB_TO_XYZ_MATRIX = np.linalg.inv(CIE1994_XYZ_TO_RGB_MATRIX)

# #############################################################################
# #############################################################################
# ## n-1 Pseudo Numeric Input
# #############################################################################
# #############################################################################
def chromatic_adaptation_CIE1994(XYZ_1,
                                 xy_o1,
                                 xy_o2,
                                 Y_o,
                                 E_o1,
                                 E_o2,
                                 n=1):
    Y_o = np.asarray(Y_o)
    E_o1 = np.asarray(E_o1)
    E_o2 = np.asarray(E_o2)

    if np.any(Y_o < 18) or np.any(Y_o > 100):
        warning(('"Y_o" luminance factor must be in [18, 100] domain, '
                 'unpredictable results may occur!'))

    RGB_1 = XYZ_to_RGB_cie1994(XYZ_1)

    xez_1 = intermediate_values(xy_o1)
    xez_2 = intermediate_values(xy_o2)

    RGB_o1 = effective_adapting_responses(xez_1, Y_o, E_o1)
    RGB_o2 = effective_adapting_responses(xez_2, Y_o, E_o2)

    bRGB_o1 = exponential_factors(RGB_o1)
    bRGB_o2 = exponential_factors(RGB_o2)

    K = K_coefficient(xez_1, xez_2, bRGB_o1, bRGB_o2, Y_o, n)

    RGB_2 = corresponding_colour(
        RGB_1, xez_1, xez_2, bRGB_o1, bRGB_o2, Y_o, K, n)
    XYZ_2 = RGB_to_XYZ_cie1994(RGB_2)

    return XYZ_2


def XYZ_to_RGB_cie1994(XYZ):
    return np.einsum('...ij,...j->...i', CIE1994_XYZ_TO_RGB_MATRIX, XYZ)


def RGB_to_XYZ_cie1994(RGB):
    return np.einsum('...ij,...j->...i', CIE1994_RGB_TO_XYZ_MATRIX, RGB)


def intermediate_values(xy_o):
    x_o, y_o = tsplit(xy_o)

    # Computing :math:`\xi`, :math:`\eta`, :math:`\zeta` values.
    xi = (0.48105 * x_o + 0.78841 * y_o - 0.08081) / y_o
    eta = (-0.27200 * x_o + 1.11962 * y_o + 0.04570) / y_o
    zeta = (0.91822 * (1 - x_o - y_o)) / y_o

    xez = tstack((xi, eta, zeta))

    return xez


def effective_adapting_responses(xez, Y_o, E_o):
    xez = np.asarray(xez)
    Y_o = np.asarray(Y_o)
    E_o = np.asarray(E_o)

    RGB_o = ((Y_o[..., np.newaxis] * E_o[..., np.newaxis]) / (
    100 * np.pi)) * xez

    return RGB_o


def beta_1(x):
    return (6.469 + 6.362 * (x ** 0.4495)) / (6.469 + (x ** 0.4495))


def beta_2(x):
    return 0.7844 * (8.414 + 8.091 * (x ** 0.5128)) / (8.414 + (x ** 0.5128))


def exponential_factors(RGB_o):
    R_o, G_o, B_o = tsplit(RGB_o)

    bR_o = beta_1(R_o)
    bG_o = beta_1(G_o)
    bB_o = beta_2(B_o)

    bRGB_o = tstack((bR_o, bG_o, bB_o))

    return bRGB_o


def K_coefficient(xez_1, xez_2, bRGB_o1, bRGB_o2, Y_o, n=1):
    xi_1, eta_1, zeta_1 = tsplit(xez_1)
    xi_2, eta_2, zeta_2 = tsplit(xez_2)
    bR_o1, bG_o1, bB_o1 = tsplit(bRGB_o1)
    bR_o2, bG_o2, bB_o2 = tsplit(bRGB_o2)
    Y_o = np.asarray(Y_o)

    K = (((Y_o * xi_1 + n) / (20 * xi_1 + n)) ** ((2 / 3) * bR_o1) /
         ((Y_o * xi_2 + n) / (20 * xi_2 + n)) ** ((2 / 3) * bR_o2))

    K *= (((Y_o * eta_1 + n) / (20 * eta_1 + n)) ** ((1 / 3) * bG_o1) /
          ((Y_o * eta_2 + n) / (20 * eta_2 + n)) ** ((1 / 3) * bG_o2))

    return K


def corresponding_colour(RGB_1, xez_1, xez_2, bRGB_o1, bRGB_o2, Y_o, K, n=1):
    R_1, G_1, B_1 = tsplit(RGB_1)
    xi_1, eta_1, zeta_1 = tsplit(xez_1)
    xi_2, eta_2, zeta_2 = tsplit(xez_2)
    bR_o1, bG_o1, bB_o1 = tsplit(bRGB_o1)
    bR_o2, bG_o2, bB_o2 = tsplit(bRGB_o2)
    Y_o = np.asarray(Y_o)
    K = np.asarray(K)

    RGBc = lambda x1, x2, y1, y2, z: (
        (Y_o * x2 + n) * K ** (1 / y2) *
        ((z + n) / (Y_o * x1 + n)) ** (y1 / y2) - n)

    R_2 = RGBc(xi_1, xi_2, bR_o1, bR_o2, R_1)
    G_2 = RGBc(eta_1, eta_2, bG_o1, bG_o2, G_1)
    B_2 = RGBc(zeta_1, zeta_2, bB_o1, bB_o2, B_1)

    RGB_2 = tstack((R_2, G_2, B_2))

    return RGB_2


print('n-1 Pseudo Numeric Input\n')

print('1d array input:')
XYZ_1 = np.array([28.0, 21.26, 5.27])
xy_o1 = np.array([0.4476, 0.4074])
xy_o2 = np.array([0.3127, 0.3290])
Y_o = 20
E_o1 = 1000
E_o2 = 1000
print(chromatic_adaptation_CIE1994(
    XYZ_1, xy_o1, xy_o2, Y_o, E_o1, E_o2))

print('\n')

print('1.5d array input:')
XYZ_1 = np.tile(XYZ_1, (6, 1))
print(chromatic_adaptation_CIE1994(
    XYZ_1, xy_o1, xy_o2, Y_o, E_o1, E_o2))

print('\n')

print('2d array input:')
xy_o1 = np.tile(xy_o1, (6, 1))
xy_o2 = np.tile(xy_o2, (6, 1))
Y_o = np.tile(Y_o, 6)
E_o1 = np.tile(E_o1, 6)
E_o2 = np.tile(E_o2, 6)
print(chromatic_adaptation_CIE1994(
    XYZ_1, xy_o1, xy_o2, Y_o, E_o1, E_o2))

print('\n')

print('3d array input:')
XYZ_1 = np.reshape(XYZ_1, (2, 3, 3))
xy_o1 = np.reshape(xy_o1, (2, 3, 2))
xy_o2 = np.reshape(xy_o2, (2, 3, 2))
Y_o = np.reshape(Y_o, (2, 3))
E_o1 = np.reshape(E_o1, (2, 3))
E_o2 = np.reshape(E_o2, (2, 3))
print(chromatic_adaptation_CIE1994(
    XYZ_1, xy_o1, xy_o2, Y_o, E_o1, E_o2))

print('\n')

print('4d array input:')
XYZ_1 = np.reshape(XYZ_1, (1, 2, 3, 3))
xy_o1 = np.reshape(xy_o1, (1, 2, 3, 2))
xy_o2 = np.reshape(xy_o2, (1, 2, 3, 2))
Y_o = np.reshape(Y_o, (1, 2, 3))
E_o1 = np.reshape(E_o1, (1, 2, 3))
E_o2 = np.reshape(E_o2, (1, 2, 3))
print(chromatic_adaptation_CIE1994(
    XYZ_1, xy_o1, xy_o2, Y_o, E_o1, E_o2))

print('\n')

print('*' * 79)

print('\n')

# #############################################################################
# #############################################################################
# ## n Pseudo Numeric Input
# #############################################################################
# #############################################################################
def chromatic_adaptation_CIE1994(XYZ_1,
                                 xy_o1,
                                 xy_o2,
                                 Y_o,
                                 E_o1,
                                 E_o2,
                                 n=1):
    Y_o = np.asarray(Y_o)
    E_o1 = np.asarray(E_o1)
    E_o2 = np.asarray(E_o2)

    if np.any(Y_o < 18) or np.any(Y_o > 100):
        warning(('"Y_o" luminance factor must be in [18, 100] domain, '
                 'unpredictable results may occur!'))

    RGB_1 = XYZ_to_RGB_cie1994(XYZ_1)

    xez_1 = intermediate_values(xy_o1)
    xez_2 = intermediate_values(xy_o2)

    RGB_o1 = effective_adapting_responses(xez_1, Y_o, E_o1)
    RGB_o2 = effective_adapting_responses(xez_2, Y_o, E_o2)

    bRGB_o1 = exponential_factors(RGB_o1)
    bRGB_o2 = exponential_factors(RGB_o2)

    K = K_coefficient(xez_1, xez_2, bRGB_o1, bRGB_o2, Y_o, n)

    RGB_2 = corresponding_colour(
        RGB_1, xez_1, xez_2, bRGB_o1, bRGB_o2, Y_o, K, n)
    XYZ_2 = RGB_to_XYZ_cie1994(RGB_2)

    return XYZ_2


def XYZ_to_RGB_cie1994(XYZ):
    return np.einsum('...ij,...j->...i', CIE1994_XYZ_TO_RGB_MATRIX, XYZ)


def RGB_to_XYZ_cie1994(RGB):
    return np.einsum('...ij,...j->...i', CIE1994_RGB_TO_XYZ_MATRIX, RGB)


def intermediate_values(xy_o):
    x_o, y_o = tsplit(xy_o)

    # Computing :math:`\xi`, :math:`\eta`, :math:`\zeta` values.
    xi = (0.48105 * x_o + 0.78841 * y_o - 0.08081) / y_o
    eta = (-0.27200 * x_o + 1.11962 * y_o + 0.04570) / y_o
    zeta = (0.91822 * (1 - x_o - y_o)) / y_o

    xez = tstack((xi, eta, zeta))

    return xez


def effective_adapting_responses(xez, Y_o, E_o):
    xez = np.asarray(xez)
    Y_o = np.asarray(Y_o)
    E_o = np.asarray(E_o)

    RGB_o = ((Y_o * E_o) / (100 * np.pi)) * xez

    return RGB_o


def beta_1(x):
    return (6.469 + 6.362 * (x ** 0.4495)) / (6.469 + (x ** 0.4495))


def beta_2(x):
    return 0.7844 * (8.414 + 8.091 * (x ** 0.5128)) / (8.414 + (x ** 0.5128))


def exponential_factors(RGB_o):
    R_o, G_o, B_o = tsplit(RGB_o)

    bR_o = beta_1(R_o)
    bG_o = beta_1(G_o)
    bB_o = beta_2(B_o)

    bRGB_o = tstack((bR_o, bG_o, bB_o))

    return bRGB_o


def K_coefficient(xez_1, xez_2, bRGB_o1, bRGB_o2, Y_o, n=1):
    xi_1, eta_1, zeta_1 = tsplit(xez_1)
    xi_2, eta_2, zeta_2 = tsplit(xez_2)
    bR_o1, bG_o1, bB_o1 = tsplit(bRGB_o1)
    bR_o2, bG_o2, bB_o2 = tsplit(bRGB_o2)
    Y_o = np.asarray(Y_o)

    K = (((Y_o * xi_1 + n) / (20 * xi_1 + n)) ** ((2 / 3) * bR_o1) /
         ((Y_o * xi_2 + n) / (20 * xi_2 + n)) ** ((2 / 3) * bR_o2))

    K *= (((Y_o * eta_1 + n) / (20 * eta_1 + n)) ** ((1 / 3) * bG_o1) /
          ((Y_o * eta_2 + n) / (20 * eta_2 + n)) ** ((1 / 3) * bG_o2))

    return K


def corresponding_colour(RGB_1, xez_1, xez_2, bRGB_o1, bRGB_o2, Y_o, K, n=1):
    R_1, G_1, B_1 = tsplit(RGB_1)
    xi_1, eta_1, zeta_1 = tsplit(xez_1)
    xi_2, eta_2, zeta_2 = tsplit(xez_2)
    bR_o1, bG_o1, bB_o1 = tsplit(bRGB_o1)
    bR_o2, bG_o2, bB_o2 = tsplit(bRGB_o2)
    Y_o = np.asarray(Y_o)
    K = np.asarray(K)

    RGBc = lambda x1, x2, y1, y2, z: (
        (Y_o * x2 + n) * K ** (1 / y2) *
        ((z + n) / (Y_o * x1 + n)) ** (y1 / y2) - n)

    R_2 = RGBc(xi_1, xi_2, bR_o1, bR_o2, R_1)
    G_2 = RGBc(eta_1, eta_2, bG_o1, bG_o2, G_1)
    B_2 = RGBc(zeta_1, zeta_2, bB_o1, bB_o2, B_1)

    RGB_2 = tstack((R_2, G_2, B_2))

    return RGB_2


print('n Pseudo Numeric Input\n')

print('1d array input:')
XYZ_1 = np.array([28.0, 21.26, 5.27])
xy_o1 = np.array([0.4476, 0.4074])
xy_o2 = np.array([0.3127, 0.3290])
Y_o = 20
E_o1 = 1000
E_o2 = 1000
print(chromatic_adaptation_CIE1994(
    XYZ_1, xy_o1, xy_o2, Y_o, E_o1, E_o2))

print('\n')

print('1.5d array input:')
XYZ_1 = np.tile(XYZ_1, (6, 1))
print(chromatic_adaptation_CIE1994(
    XYZ_1, xy_o1, xy_o2, Y_o, E_o1, E_o2))

print('\n')

print('2d array input:')
xy_o1 = np.tile(xy_o1, (6, 1))
xy_o2 = np.tile(xy_o2, (6, 1))
Y_o = np.tile(Y_o, (6, 1))
E_o1 = np.tile(E_o1, (6, 1))
E_o2 = np.tile(E_o2, (6, 1))
print(chromatic_adaptation_CIE1994(
    XYZ_1, xy_o1, xy_o2, Y_o, E_o1, E_o2))

print('\n')

print('3d array input:')
XYZ_1 = np.reshape(XYZ_1, (2, 3, 3))
xy_o1 = np.reshape(xy_o1, (2, 3, 2))
xy_o2 = np.reshape(xy_o2, (2, 3, 2))
Y_o = np.reshape(Y_o, (2, 3, 1))
E_o1 = np.reshape(E_o1, (2, 3, 1))
E_o2 = np.reshape(E_o2, (2, 3, 1))
print(chromatic_adaptation_CIE1994(
    XYZ_1, xy_o1, xy_o2, Y_o, E_o1, E_o2))

print('\n')

print('4d array input:')
XYZ_1 = np.reshape(XYZ_1, (1, 2, 3, 3))
xy_o1 = np.reshape(xy_o1, (1, 2, 3, 2))
xy_o2 = np.reshape(xy_o2, (1, 2, 3, 2))
Y_o = np.reshape(Y_o, (1, 2, 3, 1))
E_o1 = np.reshape(E_o1, (1, 2, 3, 1))
E_o2 = np.reshape(E_o2, (1, 2, 3, 1))
print(chromatic_adaptation_CIE1994(
    XYZ_1, xy_o1, xy_o2, Y_o, E_o1, E_o2))