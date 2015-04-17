#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NOTE: This is a work in progress and will be injected into the API.

Defines *Colour* vectorised prototype definitions. They are designed to be
compatible with the existing API while accepting arrays with n-dimensions.

Helpers
-------

# (.*)_analysis\(\)$
$1_analysis()

"""

from __future__ import division, with_statement

import functools
import numpy as np
import timeit
import warnings
from pprint import pprint

from colour.utilities import (
    handle_numpy_errors,
    ignore_numpy_errors,
    is_iterable,
    is_numeric,
    message_box,
    tstack,
    tsplit,
    row_as_diagonal,
    warning)

# np.random.seed(64)

DATA_HD1 = np.random.rand(1920 * 1080, 3)
DATA_HD2 = np.random.rand(1920 * 1080, 3)
DATA_HD3 = np.random.rand(1920 * 1080, 3)

DATA_VGA1 = np.random.rand(320 * 200, 3)
DATA_VGA2 = np.random.rand(320 * 200, 3)
DATA_VGA3 = np.random.rand(320 * 200, 3)

DATA1, DATA2, DATA3 = DATA_VGA1, DATA_VGA2, DATA_VGA3

# Warnings supression.
np.seterr(all='ignore')
warnings.filterwarnings('ignore')

# #############################################################################
# #############################################################################
# ## colour.adaptation.cie1994
# #############################################################################
# #############################################################################
from colour.adaptation import *
from colour.adaptation.cie1994 import *


def chromatic_adaptation_CIE1994_2d(XYZ_1):
    xy_o1 = (0.4476, 0.4074)
    xy_o2 = (0.3127, 0.3290)
    Y_o = 20
    E_o1 = 1000
    E_o2 = 1000

    for i in range(len(XYZ_1)):
        chromatic_adaptation_CIE1994(XYZ_1[i], xy_o1, xy_o2, Y_o, E_o1, E_o2)


def chromatic_adaptation_CIE1994_vectorise(XYZ_1,
                                           xy_o1,
                                           xy_o2,
                                           Y_o,
                                           E_o1,
                                           E_o2,
                                           n=1):
    if np.any(Y_o < 18) or np.any(Y_o > 100):
        warning(('"Y_o" luminance factor must be in [18, 100] domain, '
                 'unpredictable results may occur!'))

    RGB_1 = XYZ_to_RGB_cie1994_vectorise(XYZ_1)

    xez_1 = intermediate_values_vectorise(xy_o1)
    xez_2 = intermediate_values_vectorise(xy_o2)

    RGB_o1 = effective_adapting_responses_vectorise(xez_1, Y_o, E_o1)
    RGB_o2 = effective_adapting_responses_vectorise(xez_2, Y_o, E_o2)

    bRGB_o1 = exponential_factors_vectorise(RGB_o1)
    bRGB_o2 = exponential_factors_vectorise(RGB_o2)

    K = K_coefficient_vectorise(xez_1, xez_2, bRGB_o1, bRGB_o2, Y_o, n)

    RGB_2 = corresponding_colour_vectorise(
        RGB_1, xez_1, xez_2, bRGB_o1, bRGB_o2, Y_o, K, n)
    XYZ_2 = RGB_to_XYZ_cie1994_vectorise(RGB_2)

    return XYZ_2


def XYZ_to_RGB_cie1994_vectorise(XYZ):
    return np.einsum('...ij,...j->...i', CIE1994_XYZ_TO_RGB_MATRIX, XYZ)


def RGB_to_XYZ_cie1994_vectorise(RGB):
    return np.einsum('...ij,...j->...i', CIE1994_RGB_TO_XYZ_MATRIX, RGB)


def intermediate_values_vectorise(xy_o):
    x_o, y_o = tsplit(xy_o)

    # Computing :math:`\xi`, :math:`\eta`, :math:`\zeta` values.
    xi = (0.48105 * x_o + 0.78841 * y_o - 0.08081) / y_o
    eta = (-0.27200 * x_o + 1.11962 * y_o + 0.04570) / y_o
    zeta = (0.91822 * (1 - x_o - y_o)) / y_o

    xez = tstack((xi, eta, zeta))

    return xez


def effective_adapting_responses_vectorise(xez, Y_o, E_o):
    # TODO: Mention *xez* place change.
    xez = np.asarray(xez)
    Y_o = np.asarray(Y_o)
    E_o = np.asarray(E_o)

    RGB_o = ((Y_o * E_o) / (100 * np.pi)) * xez

    return RGB_o


def beta_1_vectorise(x):
    return (6.469 + 6.362 * (x ** 0.4495)) / (6.469 + (x ** 0.4495))


def beta_2_vectorise(x):
    return 0.7844 * (8.414 + 8.091 * (x ** 0.5128)) / (8.414 + (x ** 0.5128))


def exponential_factors_vectorise(RGB_o):
    R_o, G_o, B_o = tsplit(RGB_o)

    bR_o = beta_1_vectorise(R_o)
    bG_o = beta_1_vectorise(G_o)
    bB_o = beta_2_vectorise(B_o)

    bRGB_o = tstack((bR_o, bG_o, bB_o))

    return bRGB_o


def K_coefficient_vectorise(xez_1, xez_2, bRGB_o1, bRGB_o2, Y_o, n=1):
    # TODO: Mention *Y_o* place change.

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


def corresponding_colour_vectorise(
        RGB_1, xez_1, xez_2, bRGB_o1, bRGB_o2, Y_o, K, n=1):
    # TODO: Mention *Y_o* place change.

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


def chromatic_adaptation_CIE1994_analysis():
    message_box('chromatic_adaptation_CIE1994')

    print('Reference:')
    XYZ_1 = np.array([28.0, 21.26, 5.27])
    xy_o1 = np.array([0.4476, 0.4074])
    xy_o2 = np.array([0.3127, 0.3290])
    Y_o = 20
    E_o1 = 1000
    E_o2 = 1000
    print(chromatic_adaptation_CIE1994(XYZ_1, xy_o1, xy_o2, Y_o, E_o1, E_o2))

    print('\n')

    print('1d array input:')
    print(chromatic_adaptation_CIE1994_vectorise(
        XYZ_1, xy_o1, xy_o2, Y_o, E_o1, E_o2))

    print('\n')

    print('2d array input:')
    XYZ_1 = np.tile(XYZ_1, (6, 1))
    print(chromatic_adaptation_CIE1994_vectorise(
        XYZ_1, xy_o1, xy_o2, Y_o, E_o1, E_o2))

    print('\n')

    print('3d array input:')
    XYZ_1 = np.reshape(XYZ_1, (2, 3, 3))
    print(chromatic_adaptation_CIE1994_vectorise(
        XYZ_1, xy_o1, xy_o2, Y_o, E_o1, E_o2))

    print('\n')


# chromatic_adaptation_CIE1994_analysis()


def chromatic_adaptation_CIE1994_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    xy_o1 = np.array([0.4476, 0.4074])
    xy_o2 = np.array([0.3127, 0.3290])
    Y_o = 20
    E_o1 = 1000
    E_o2 = 1000

    times = timeit.Timer(
        functools.partial(chromatic_adaptation_CIE1994_2d,
                          DATA_HD1)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(chromatic_adaptation_CIE1994_vectorise,
                          DATA_HD1, xy_o1, xy_o2, Y_o, E_o1, E_o2)
    ).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('chromatic_adaptation_CIE1994\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# chromatic_adaptation_CIE1994_profile()

# #############################################################################
# #############################################################################
# ## colour.adaptation.cmccat2000
# #############################################################################
# #############################################################################
from colour.adaptation.cmccat2000 import *


def CMCCAT2000_forward_2d(XYZ):
    XYZ_w = np.array([111.15, 100.00, 35.20])
    XYZ_wr = np.array([94.81, 100.00, 107.30])
    L_A1 = 200
    L_A2 = 200

    for i in range(len(XYZ)):
        CMCCAT2000_forward(XYZ[i], XYZ_w, XYZ_wr, L_A1, L_A2)


def CMCCAT2000_forward_vectorise(XYZ,
                                 XYZ_w,
                                 XYZ_wr,
                                 L_A1,
                                 L_A2,
                                 surround=CMCCAT2000_VIEWING_CONDITIONS.get(
                                     'Average')):
    L_A1 = np.asarray(L_A1)
    L_A2 = np.asarray(L_A2)

    RGB = np.einsum('...ij,...j->...i', CMCCAT2000_CAT, XYZ)
    RGB_w = np.einsum('...ij,...j->...i', CMCCAT2000_CAT, XYZ_w)
    RGB_wr = np.einsum('...ij,...j->...i', CMCCAT2000_CAT, XYZ_wr)

    D = (surround.F *
         (0.08 * np.log10(0.5 * (L_A1 + L_A2)) +
          0.76 - 0.45 * (L_A1 - L_A2) / (L_A1 + L_A2)))

    D = np.clip(D, 0, 1)
    a = D * XYZ_w[..., 1] / XYZ_wr[..., 1]

    RGB_c = (RGB *
             (a[..., np.newaxis] * (RGB_wr / RGB_w) + 1 - D[..., np.newaxis]))
    XYZ_c = np.einsum('...ij,...j->...i', CMCCAT2000_INVERSE_CAT, RGB_c)

    return XYZ_c


def CMCCAT2000_forward_analysis():
    message_box('CMCCAT2000_forward')

    print('Reference:')
    XYZ = np.array([22.48, 22.74, 8.54])
    XYZ_w = np.array([111.15, 100.00, 35.20])
    XYZ_wr = np.array([94.81, 100.00, 107.30])
    L_A1 = 200
    L_A2 = 200
    print(CMCCAT2000_forward(XYZ, XYZ_w, XYZ_wr, L_A1, L_A2))

    print('\n')

    print('1d array input:')
    print(CMCCAT2000_forward_vectorise(XYZ, XYZ_w, XYZ_wr, L_A1, L_A2))

    print('\n')

    print('2d array input:')
    XYZ = np.tile(XYZ, (6, 1))
    print(CMCCAT2000_forward_vectorise(XYZ, XYZ_w, XYZ_wr, L_A1, L_A2))

    print('\n')

    print('3d array input:')
    XYZ = np.reshape(XYZ, (2, 3, 3))
    print(CMCCAT2000_forward_vectorise(XYZ, XYZ_w, XYZ_wr, L_A1, L_A2))

    print('\n')


# CMCCAT2000_forward_analysis()


def CMCCAT2000_forward_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    XYZ_w = np.array([111.15, 100.00, 35.20])
    XYZ_wr = np.array([94.81, 100.00, 107.30])
    L_A1 = 200
    L_A2 = 200

    times = timeit.Timer(
        functools.partial(
            CMCCAT2000_forward_2d,
            DATA_HD1)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            CMCCAT2000_forward_vectorise,
            DATA_HD1, XYZ_w, XYZ_wr, L_A1, L_A2)
    ).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('CMCCAT2000_forward\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# CMCCAT2000_forward_profile()


def CMCCAT2000_reverse_2d(XYZ):
    XYZ_w = np.array([111.15, 100.00, 35.20])
    XYZ_wr = np.array([94.81, 100.00, 107.30])
    L_A1 = 200
    L_A2 = 200

    for i in range(len(XYZ)):
        CMCCAT2000_reverse(XYZ[i], XYZ_w, XYZ_wr, L_A1, L_A2)


def CMCCAT2000_reverse_vectorise(XYZ_c,
                                 XYZ_w,
                                 XYZ_wr,
                                 L_A1,
                                 L_A2,
                                 surround=CMCCAT2000_VIEWING_CONDITIONS.get(
                                     'Average')):
    L_A1 = np.asarray(L_A1)
    L_A2 = np.asarray(L_A2)

    RGB_c = np.einsum('...ij,...j->...i', CMCCAT2000_CAT, XYZ_c)
    RGB_w = np.einsum('...ij,...j->...i', CMCCAT2000_CAT, XYZ_w)
    RGB_wr = np.einsum('...ij,...j->...i', CMCCAT2000_CAT, XYZ_wr)

    D = (surround.F *
         (0.08 * np.log10(0.5 * (L_A1 + L_A2)) +
          0.76 - 0.45 * (L_A1 - L_A2) / (L_A1 + L_A2)))

    D = np.clip(D, 0, 1)
    a = D * XYZ_w[..., 1] / XYZ_wr[..., 1]

    RGB = (RGB_c /
           (a[..., np.newaxis] * (RGB_wr / RGB_w) + 1 - D[..., np.newaxis]))
    XYZ = np.einsum('...ij,...j->...i', CMCCAT2000_INVERSE_CAT, RGB)

    return XYZ


def CMCCAT2000_reverse_analysis():
    message_box('CMCCAT2000_reverse')

    print('Reference:')
    XYZ = np.array([22.48, 22.74, 8.54])
    XYZ_w = np.array([111.15, 100.00, 35.20])
    XYZ_wr = np.array([94.81, 100.00, 107.30])
    L_A1 = 200
    L_A2 = 200
    print(CMCCAT2000_reverse(XYZ, XYZ_w, XYZ_wr, L_A1, L_A2))

    print('\n')

    print('1d array input:')
    print(CMCCAT2000_reverse_vectorise(XYZ, XYZ_w, XYZ_wr, L_A1, L_A2))

    print('\n')

    print('2d array input:')
    XYZ = np.tile(XYZ, (6, 1))
    print(CMCCAT2000_reverse_vectorise(XYZ, XYZ_w, XYZ_wr, L_A1, L_A2))

    print('\n')

    print('3d array input:')
    XYZ = np.reshape(XYZ, (2, 3, 3))
    print(CMCCAT2000_reverse_vectorise(XYZ, XYZ_w, XYZ_wr, L_A1, L_A2))

    print('\n')


# CMCCAT2000_reverse_analysis()


def CMCCAT2000_reverse_profile(repeat_a=3, number_a=5, repeat_b=3,
                               number_b=10):
    XYZ_w = np.array([111.15, 100.00, 35.20])
    XYZ_wr = np.array([94.81, 100.00, 107.30])
    L_A1 = 200
    L_A2 = 200

    times = timeit.Timer(
        functools.partial(
            CMCCAT2000_reverse_2d,
            DATA_HD1)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            CMCCAT2000_reverse_vectorise,
            DATA_HD1, XYZ_w, XYZ_wr, L_A1, L_A2)
    ).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('CMCCAT2000_reverse\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# CMCCAT2000_reverse_profile()

# #############################################################################
# #############################################################################
# ## colour.adaptation.fairchild1990
# #############################################################################
# #############################################################################
from colour.adaptation.fairchild1990 import *


def chromatic_adaptation_Fairchild1990_2d(XYZ_1):
    XYZ_n = np.array([111.15, 100.00, 35.20])
    XYZ_r = np.array([94.81, 100.00, 107.30])
    Y_n = 200

    for i in range(len(XYZ_1)):
        chromatic_adaptation_Fairchild1990(XYZ_1[i], XYZ_n, XYZ_r, Y_n)


def chromatic_adaptation_Fairchild1990_vectorise(XYZ_1,
                                                 XYZ_n,
                                                 XYZ_r,
                                                 Y_n,
                                                 discount_illuminant=False):
    XYZ_1 = np.asarray(XYZ_1)
    XYZ_n = np.asarray(XYZ_n)
    XYZ_r = np.asarray(XYZ_r)
    Y_n = np.asarray(Y_n)

    LMS_1 = np.einsum('...ij,...j->...i',
                      FAIRCHILD1990_XYZ_TO_RGB_MATRIX,
                      XYZ_1)
    LMS_n = np.einsum('...ij,...j->...i',
                      FAIRCHILD1990_XYZ_TO_RGB_MATRIX,
                      XYZ_n)
    LMS_r = np.einsum('...ij,...j->...i',
                      FAIRCHILD1990_XYZ_TO_RGB_MATRIX,
                      XYZ_r)

    p_LMS = degrees_of_adaptation_vectorise(LMS_1,
                                            Y_n,
                                            discount_illuminant=discount_illuminant)

    a_LMS_1 = p_LMS / LMS_n
    a_LMS_2 = p_LMS / LMS_r

    A_1 = row_as_diagonal(a_LMS_1)
    A_2 = row_as_diagonal(a_LMS_2)

    LMSp_1 = np.einsum('...ij,...j->...i', A_1, LMS_1)

    c = 0.219 - 0.0784 * np.log10(Y_n)
    C = row_as_diagonal(tstack((c, c, c)))

    LMS_a = np.einsum('...ij,...j->...i', C, LMSp_1)
    LMSp_2 = np.einsum('...ij,...j->...i', np.linalg.inv(C), LMS_a)

    LMS_c = np.einsum('...ij,...j->...i', np.linalg.inv(A_2), LMSp_2)
    XYZ_c = np.einsum('...ij,...j->...i',
                      FAIRCHILD1990_RGB_TO_XYZ_MATRIX, LMS_c)

    return XYZ_c


def XYZ_to_RGB_fairchild1990_vectorise(XYZ):
    return np.einsum('...ij,...j->...i', FAIRCHILD1990_XYZ_TO_RGB_MATRIX, XYZ)


def RGB_to_XYZ_fairchild1990_vectorise(RGB):
    return np.einsum('...ij,...j->...i', FAIRCHILD1990_RGB_TO_XYZ_MATRIX, RGB)


def degrees_of_adaptation_vectorise(LMS, Y_n, v=1 / 3,
                                    discount_illuminant=False):
    LMS = np.asarray(LMS)
    if discount_illuminant:
        return np.ones(LMS.shape)

    Y_n = np.asarray(Y_n)
    v = np.asarray(v)

    L, M, S = tsplit(LMS)

    LMS_E = np.einsum('...ij,...j->...i',
                      VON_KRIES_CAT,
                      np.ones(LMS.shape))  # E illuminant.
    L_E, M_E, S_E = tsplit(LMS_E)

    Ye_n = Y_n ** v

    f_E = lambda x, y: (3 * (x / y)) / (L / L_E + M / M_E + S / S_E)
    f_P = lambda x: (1 + Ye_n + x) / (1 + Ye_n + 1 / x)

    p_L = f_P(f_E(L, L_E))
    p_M = f_P(f_E(M, M_E))
    p_S = f_P(f_E(S, S_E))

    p_LMS = tstack((p_L, p_M, p_S))

    return p_LMS


def chromatic_adaptation_Fairchild1990_analysis():
    message_box('chromatic_adaptation_Fairchild1990')

    print('Reference:')
    XYZ_1 = np.array([19.53, 23.07, 24.97])
    XYZ_n = np.array([111.15, 100.00, 35.20])
    XYZ_r = np.array([94.81, 100.00, 107.30])
    Y_n = 200
    print(chromatic_adaptation_Fairchild1990(XYZ_1, XYZ_n, XYZ_r, Y_n))

    print('\n')

    print('1d array input:')
    print(chromatic_adaptation_Fairchild1990_vectorise(
        XYZ_1, XYZ_n, XYZ_r, Y_n))

    print('\n')

    print('2d array input:')
    XYZ_1 = np.tile(XYZ_1, (6, 1))
    print(chromatic_adaptation_Fairchild1990_vectorise(
        XYZ_1, XYZ_n, XYZ_r, Y_n))

    print('\n')

    print('3d array input:')
    XYZ_1 = np.reshape(XYZ_1, (2, 3, 3))
    print(chromatic_adaptation_Fairchild1990_vectorise(
        XYZ_1, XYZ_n, XYZ_r, Y_n))

    print('\n')


# chromatic_adaptation_Fairchild1990_analysis()

def chromatic_adaptation_Fairchild1990_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    XYZ_n = np.array([111.15, 100.00, 35.20])
    XYZ_r = np.array([94.81, 100.00, 107.30])
    Y_n = 200

    times = timeit.Timer(
        functools.partial(
            chromatic_adaptation_Fairchild1990_2d,
            DATA_HD1)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            chromatic_adaptation_Fairchild1990_vectorise,
            DATA_HD1, XYZ_n, XYZ_r, Y_n)
    ).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('chromatic_adaptation_Fairchild1990\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# chromatic_adaptation_Fairchild1990_profile()

# #############################################################################
# #############################################################################
# ## colour.adaptation.vonkries
# #############################################################################
# #############################################################################

# #############################################################################
# # ### colour.chromatic_adaptation_matrix_VonKries
# #############################################################################
from colour.adaptation.vonkries import *


def chromatic_adaptation_matrix_VonKries_2d(data1, data2):
    for i in range(len(data1)):
        chromatic_adaptation_matrix_VonKries(data1[i], data2[i])


def chromatic_adaptation_matrix_VonKries_vectorise(XYZ_w,
                                                   XYZ_wr,
                                                   transform='CAT02'):
    M = CHROMATIC_ADAPTATION_TRANSFORMS.get(transform)

    if M is None:
        raise KeyError(
            '"{0}" chromatic adaptation transform is not defined! Supported '
            'methods: "{1}".'.format(transform,
                                     CHROMATIC_ADAPTATION_TRANSFORMS.keys()))

    rgb_w = np.einsum('...i,...ij->...j', XYZ_w, np.transpose(M))
    rgb_wr = np.einsum('...i,...ij->...j', XYZ_wr, np.transpose(M))

    D = rgb_wr / rgb_w

    D = row_as_diagonal(D)

    cat = np.einsum('...ij,...jk->...ik', np.linalg.inv(M), D)
    cat = np.einsum('...ij,...jk->...ik', cat, M)

    return cat


def chromatic_adaptation_matrix_VonKries_analysis():
    message_box('chromatic_adaptation_matrix_VonKries')

    print('Reference:')
    XYZ_w = np.array([1.09846607, 1., 0.3558228])
    XYZ_wr = np.array([0.95042855, 1., 1.08890037])
    print(chromatic_adaptation_matrix_VonKries(XYZ_w, XYZ_wr))

    print('\n')

    print('1d array input:')
    print(chromatic_adaptation_matrix_VonKries_vectorise(XYZ_w, XYZ_wr))

    print('\n')

    print('2d array input:')
    XYZ_w = np.tile(XYZ_w, (6, 1))
    XYZ_wr = np.tile(XYZ_wr, (6, 1))
    print(chromatic_adaptation_matrix_VonKries_vectorise(XYZ_w, XYZ_wr))

    print('\n')

    print('3d array input:')
    XYZ_w = np.reshape(XYZ_w, (2, 3, 3))
    XYZ_wr = np.reshape(XYZ_wr, (2, 3, 3))
    print(chromatic_adaptation_matrix_VonKries_vectorise(XYZ_w, XYZ_wr))

    print('\n')


# chromatic_adaptation_matrix_VonKries_analysis()


def chromatic_adaptation_matrix_VonKries_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            chromatic_adaptation_matrix_VonKries_2d,
            DATA_HD1, DATA_HD2)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            chromatic_adaptation_matrix_VonKries_vectorise,
            DATA_HD1, DATA_HD2)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('chromatic_adaptation_matrix_VonKries\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# chromatic_adaptation_matrix_VonKries_profile()

# #############################################################################
# # ### colour.chromatic_adaptation_VonKries
# #############################################################################


def chromatic_adaptation_VonKries_2d(data1, data2, data3):
    for i in range(len(data1)):
        chromatic_adaptation_VonKries(data1[i], data2[i], data3[i])


def chromatic_adaptation_VonKries_vectorise(XYZ,
                                            XYZ_w,
                                            XYZ_wr,
                                            transform='CAT02'):
    cat = chromatic_adaptation_matrix_VonKries_vectorise(XYZ_w, XYZ_wr,
                                                         transform)
    XYZ_a = np.einsum('...ij,...j->...i', cat, XYZ)

    return XYZ_a


def chromatic_adaptation_VonKries_analysis():
    message_box('chromatic_adaptation_VonKries')

    print('Reference:')
    XYZ = np.array([0.07049534, 0.1008, 0.09558313])
    XYZ_w = np.array([1.09846607, 1., 0.3558228])
    XYZ_wr = np.array([0.95042855, 1., 1.08890037])
    print(chromatic_adaptation_VonKries(XYZ, XYZ_w, XYZ_wr))

    print('\n')

    print('1d array input:')
    print(chromatic_adaptation_VonKries_vectorise(XYZ, XYZ_w, XYZ_wr))

    print('\n')

    print('2d array input:')
    XYZ = np.tile(XYZ, (6, 1))
    XYZ_w = np.tile(XYZ_w, (6, 1))
    XYZ_wr = np.tile(XYZ_wr, (6, 1))
    print(chromatic_adaptation_VonKries_vectorise(XYZ, XYZ_w, XYZ_wr))

    print('\n')

    print('3d array input:')
    XYZ = np.reshape(XYZ, (2, 3, 3))
    XYZ_w = np.reshape(XYZ_w, (2, 3, 3))
    XYZ_wr = np.reshape(XYZ_wr, (2, 3, 3))
    print(chromatic_adaptation_VonKries_vectorise(XYZ,
                                                  XYZ_w,
                                                  XYZ_wr))

    print('\n')


# chromatic_adaptation_VonKries_analysis()


def chromatic_adaptation_VonKries_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            chromatic_adaptation_VonKries_2d,
            DATA_HD1, DATA_HD2, DATA_HD3)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            chromatic_adaptation_VonKries_vectorise,
            DATA_HD1, DATA_HD2, DATA_HD3)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('chromatic_adaptation_VonKries\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# chromatic_adaptation_VonKries_profile()


# #############################################################################
# #############################################################################
# ## colour.algebra.coordinates.transformations
# #############################################################################
# #############################################################################

# #############################################################################
# # ### colour.cartesian_to_spherical
# #############################################################################
from colour.algebra.coordinates.transformations import *


def cartesian_to_spherical_2d(vectors):
    for vector in vectors:
        cartesian_to_spherical(vector)


def cartesian_to_spherical_vectorise(vector):
    x, y, z = tsplit(vector)

    r = np.linalg.norm(vector, axis=-1)
    theta = np.arctan2(z, np.linalg.norm(tstack((x, y)), axis=-1))
    phi = np.arctan2(y, x)

    rtp = tstack((r, theta, phi))

    return rtp


def cartesian_to_spherical_analysis():
    message_box('cartesian_to_spherical')

    print('Reference:')
    vector = np.array([3, 1, 6])
    print(cartesian_to_spherical(vector))

    print('\n')

    print('1d array input:')
    print(cartesian_to_spherical_vectorise(vector))

    print('\n')

    print('2d array input:')
    vector = np.tile(vector, (6, 1))
    print(cartesian_to_spherical_vectorise(vector))

    print('\n')

    print('3d array input:')
    vector = np.reshape(vector, (2, 3, 3))
    print(cartesian_to_spherical_vectorise(vector))

    print('\n')


# cartesian_to_spherical_analysis()


def cartesian_to_spherical_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            cartesian_to_spherical_2d,
            DATA_HD1)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            cartesian_to_spherical_vectorise,
            DATA_HD1)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('cartesian_to_spherical\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# cartesian_to_spherical_profile()

# #############################################################################
# # ### colour.spherical_to_cartesian
# #############################################################################


def spherical_to_cartesian_2d(vectors):
    for vector in vectors:
        spherical_to_cartesian(vector)


def spherical_to_cartesian_vectorise(vector):
    r, theta, phi = tsplit(vector)

    x = r * np.cos(theta) * np.cos(phi)
    y = r * np.cos(theta) * np.sin(phi)
    z = r * np.sin(theta)

    xyz = tstack((x, y, z))

    return xyz


def spherical_to_cartesian_analysis():
    message_box('spherical_to_cartesian')

    print('Reference:')
    vector = np.array([6.78232998, 1.08574654, 0.32175055])
    print(spherical_to_cartesian(vector))

    print('\n')

    print('1d array input:')
    print(spherical_to_cartesian_vectorise(vector))

    print('\n')

    print('2d array input:')
    vector = np.tile(vector, (6, 1))
    print(spherical_to_cartesian_vectorise(vector))

    print('\n')

    print('3d array input:')
    vector = np.reshape(vector, (2, 3, 3))
    print(spherical_to_cartesian_vectorise(vector))

    print('\n')


# spherical_to_cartesian_analysis()


def spherical_to_cartesian_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            spherical_to_cartesian_2d,
            DATA_HD1)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            spherical_to_cartesian_vectorise,
            DATA_HD1)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('spherical_to_cartesian\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# spherical_to_cartesian_profile()

# #############################################################################
# # ### colour.cartesian_to_cylindrical
# #############################################################################


def cartesian_to_cylindrical_2d(vectors):
    for vector in vectors:
        cartesian_to_cylindrical(vector)


def cartesian_to_cylindrical_vectorise(vector):
    x, y, z = tsplit(vector)

    theta = np.arctan2(y, x)
    rho = np.linalg.norm(tstack((x, y)), axis=-1)

    return tstack((z, theta, rho))


def cartesian_to_cylindrical_analysis():
    message_box('cartesian_to_cylindrical')

    print('Reference:')
    vector = np.array([3, 1, 6])
    print(cartesian_to_cylindrical(vector))

    print('\n')

    print('1d array input:')
    print(cartesian_to_cylindrical_vectorise(vector))

    print('\n')

    print('2d array input:')
    vector = np.tile(vector, (6, 1))
    print(cartesian_to_cylindrical_vectorise(vector))

    print('\n')

    print('3d array input:')
    vector = np.reshape(vector, (2, 3, 3))
    print(cartesian_to_cylindrical_vectorise(vector))

    print('\n')


# cartesian_to_cylindrical_analysis()


def cartesian_to_cylindrical_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            cartesian_to_cylindrical_2d,
            DATA_HD1)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            cartesian_to_cylindrical_vectorise,
            DATA_HD1)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('cartesian_to_cylindrical\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# cartesian_to_cylindrical_profile()

# #############################################################################
# # ### colour.cylindrical_to_cartesian
# #############################################################################


def cylindrical_to_cartesian_2d(vectors):
    for vector in vectors:
        cylindrical_to_cartesian(vector)


def cylindrical_to_cartesian_vectorise(vector):
    z, theta, rho = tsplit(vector)

    x = rho * np.cos(theta)
    y = rho * np.sin(theta)

    return tstack((x, y, z))


def cylindrical_to_cartesian_analysis():
    message_box('cylindrical_to_cartesian')

    print('Reference:')
    vector = np.array([6, 0.32175055, 3.16227766])
    print(cylindrical_to_cartesian(vector))

    print('\n')

    print('1d array input:')
    print(cylindrical_to_cartesian_vectorise(vector))

    print('\n')

    print('2d array input:')
    vector = np.tile(vector, (6, 1))
    print(cylindrical_to_cartesian_vectorise(vector))

    print('\n')

    print('3d array input:')
    vector = np.reshape(vector, (2, 3, 3))
    print(cylindrical_to_cartesian_vectorise(vector))

    print('\n')


# cylindrical_to_cartesian_analysis()


def cylindrical_to_cartesian_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            cylindrical_to_cartesian_2d,
            DATA_HD1)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            cylindrical_to_cartesian_vectorise,
            DATA_HD1)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('cylindrical_to_cartesian\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# cylindrical_to_cartesian_profile()

# #############################################################################
# #############################################################################
# ## colour.appearance.atd95
# #############################################################################
# #############################################################################
from colour.appearance.atd95 import *


def XYZ_to_ATD95_2d(XYZ, XYZ_0, Y_0, k_1, k_2):
    for i in range(len(XYZ)):
        XYZ_to_ATD95(XYZ[i], XYZ_0, Y_0, k_1, k_2)


def XYZ_to_ATD95_vectorise(XYZ, XYZ_0, Y_0, k_1, k_2, sigma=300):
    Y_0 = np.asarray(Y_0)
    k_1 = np.asarray(k_1)
    k_2 = np.asarray(k_2)

    XYZ = luminance_to_retinal_illuminance_vectorise(
        XYZ, Y_0[..., np.newaxis])
    XYZ_0 = luminance_to_retinal_illuminance_vectorise(
        XYZ_0, Y_0[..., np.newaxis])

    # Computing adaptation model.
    LMS = XYZ_to_LMS_ATD95_vectorise(XYZ)
    XYZ_a = k_1[..., np.newaxis] * XYZ + k_2[..., np.newaxis] * XYZ_0
    LMS_a = XYZ_to_LMS_ATD95_vectorise(XYZ_a)

    LMS_g = LMS * (sigma / (sigma + LMS_a))

    # Computing opponent colour dimensions.
    A_1, T_1, D_1, A_2, T_2, D_2 = tsplit(
        opponent_colour_dimensions_vectorise(LMS_g))

    # -------------------------------------------------------------------------
    # Computing the correlate of *brightness* :math:`Br`.
    # -------------------------------------------------------------------------
    Br = (A_1 ** 2 + T_1 ** 2 + D_1 ** 2) ** 0.5

    # -------------------------------------------------------------------------
    # Computing the correlate of *saturation* :math:`C`.
    # -------------------------------------------------------------------------
    C = (T_2 ** 2 + D_2 ** 2) ** 0.5 / A_2

    # -------------------------------------------------------------------------
    # Computing the *hue* :math:`H`.
    # -------------------------------------------------------------------------
    H = T_2 / D_2

    return ATD95_Specification(H, C, Br, A_1, T_1, D_1, A_2, T_2, D_2)


def luminance_to_retinal_illuminance_vectorise(XYZ, Y_c):
    XYZ = np.asarray(XYZ)
    Y_c = np.asarray(Y_c)

    return 18. * (Y_c * XYZ / 100.) ** 0.8


def XYZ_to_LMS_ATD95_vectorise(XYZ):
    X, Y, Z = tsplit(XYZ)

    L = ((0.66 * (0.2435 * X + 0.8524 * Y - 0.0516 * Z)) ** 0.7) + 0.024
    M = ((-0.3954 * X + 1.1642 * Y + 0.0837 * Z) ** 0.7) + 0.036
    S = ((0.43 * (0.04 * Y + 0.6225 * Z)) ** 0.7) + 0.31

    return tstack((L, M, S))


def opponent_colour_dimensions_vectorise(LMS_g):
    L_g, M_g, S_g = tsplit(LMS_g)

    A_1i = 3.57 * L_g + 2.64 * M_g
    T_1i = 7.18 * L_g - 6.21 * M_g
    D_1i = -0.7 * L_g + 0.085 * M_g + S_g
    A_2i = 0.09 * A_1i
    T_2i = 0.43 * T_1i + 0.76 * D_1i
    D_2i = D_1i

    A_1 = final_response_vectorise(A_1i)
    T_1 = final_response_vectorise(T_1i)
    D_1 = final_response_vectorise(D_1i)
    A_2 = final_response_vectorise(A_2i)
    T_2 = final_response_vectorise(T_2i)
    D_2 = final_response_vectorise(D_2i)

    return tstack((A_1, T_1, D_1, A_2, T_2, D_2))


def final_response_vectorise(value):
    return value / (200 + abs(value))


def XYZ_to_ATD95_analysis():
    message_box('XYZ_to_ATD95')

    XYZ = np.array([19.01, 20.00, 21.78])
    XYZ_0 = np.array([95.05, 100.00, 108.88])
    Y_0 = 318.31
    k_1 = 0.0
    k_2 = 50.0

    print('Reference:')
    print(XYZ_to_ATD95(XYZ, XYZ_0, Y_0, k_1, k_2))

    print('\n')

    print('1d array input:')
    print(XYZ_to_ATD95_vectorise(XYZ, XYZ_0, Y_0, k_1, k_2))

    print('\n')

    print('1.5d array input:')
    XYZ = np.tile(XYZ, (6, 1))
    print(XYZ_to_ATD95_vectorise(XYZ, XYZ_0, Y_0, k_1, k_2))

    print('\n')

    print('2d array input:')
    XYZ_0 = np.tile(XYZ_0, (6, 1))
    Y_0 = np.tile(Y_0, 6)
    k_1 = np.tile(k_1, 6)
    k_2 = np.tile(k_2, 6)
    print(XYZ_to_ATD95_vectorise(XYZ, XYZ_0, Y_0, k_1, k_2))

    print('\n')

    print('3d array input:')
    XYZ = np.reshape(XYZ, (2, 3, 3))
    XYZ_0 = np.reshape(XYZ_0, (2, 3, 3))
    Y_0 = np.reshape(Y_0, (2, 3))
    k_1 = np.reshape(k_1, (2, 3))
    k_2 = np.reshape(k_2, (2, 3))
    print(XYZ_to_ATD95_vectorise(XYZ, XYZ_0, Y_0, k_1, k_2))

    print('\n')


# XYZ_to_ATD95_analysis()


def XYZ_to_ATD95_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    XYZ_0 = np.array([95.05, 100.00, 108.88])
    Y_0 = 318.31
    k_1 = 0.0
    k_2 = 50.0

    times = timeit.Timer(
        functools.partial(
            XYZ_to_ATD95_2d,
            DATA_HD1, XYZ_0, Y_0, k_1, k_2)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            XYZ_to_ATD95_vectorise,
            DATA_HD1, XYZ_0, Y_0, k_1, k_2)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('XYZ_to_ATD95\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# XYZ_to_ATD95_profile()

# #############################################################################
# #############################################################################
# ## colour.appearance.hunt
# #############################################################################
# #############################################################################
from colour.appearance.hunt import *


def XYZ_to_Hunt_2d(XYZ, XYZ_w, XYZ_b, L_A, surround, CCT_w):
    for i in range(len(XYZ)):
        XYZ_to_Hunt(XYZ[i], XYZ_w, XYZ_b, L_A, surround, CCT_w)


def XYZ_to_Hunt_vectorise(XYZ,
                          XYZ_w,
                          XYZ_b,
                          L_A,
                          surround=HUNT_VIEWING_CONDITIONS.get(
                              'Normal Scenes'),
                          L_AS=None,
                          CCT_w=None,
                          XYZ_p=None,
                          p=None,
                          S=None,
                          S_W=None,
                          helson_judd_effect=False,
                          discount_illuminant=True):
    X, Y, Z = tsplit(XYZ)
    X_b, Y_b, Z_b = tsplit(XYZ_b)
    X_w, Y_w, Z_w = tsplit(XYZ_w)

    # Arguments handling.
    if XYZ_p is not None:
        X_p, Y_p, Z_p = tsplit(XYZ_p)
    else:
        X_p = X_b
        Y_p = Y_b
        Z_p = Y_b
        warning('Unspecified proximal field "XYZ_p" argument, using '
                'background "XYZ_b" as approximation!')

    if surround.N_cb is None:
        N_cb = 0.725 * (Y_w / Y_b) ** 0.2
        warning('Unspecified "N_cb" argument, using approximation: '
                '"{0}"'.format(N_cb))
    if surround.N_bb is None:
        N_bb = 0.725 * (Y_w / Y_b) ** 0.2
        warning('Unspecified "N_bb" argument, using approximation: '
                '"{0}"'.format(N_bb))

    if L_AS is None and CCT_w is None:
        raise ValueError('Either the scotopic luminance "L_AS" of the '
                         'illuminant or its correlated colour temperature '
                         '"CCT_w" must be specified!')
    if L_AS is None:
        L_AS = illuminant_scotopic_luminance_vectorise(L_A, CCT_w)
        warning('Unspecified "L_AS" argument, using approximation from "CCT": '
                '"{0}"'.format(L_AS))

    if S is None != S_W is None:
        raise ValueError('Either both stimulus scotopic response "S" and '
                         'reference white scotopic response "S_w" arguments '
                         'need to be specified or none of them!')
    elif S is None and S_W is None:
        S = Y
        S_W = Y_w
        warning('Unspecified stimulus scotopic response "S" and reference '
                'white scotopic response "S_w" arguments, using '
                'approximation: "{0}", "{1}"'.format(S, S_W))

    if p is None:
        warning('Unspecified simultaneous contrast / assimilation "p" '
                'argument, model will not account for simultaneous chromatic '
                'contrast!')

    XYZ_p = tstack((X_p, Y_p, Z_p))

    # Computing luminance level adaptation factor :math:`F_L`.
    F_L = luminance_level_adaptation_factor_vectorise(L_A)

    # Computing test sample chromatic adaptation.
    rgb_a = chromatic_adaptation_vectorise(XYZ,
                                           XYZ_w,
                                           XYZ_b,
                                           L_A,
                                           F_L,
                                           XYZ_p,
                                           p,
                                           helson_judd_effect,
                                           discount_illuminant)

    # Computing reference white chromatic adaptation.
    rgb_aw = chromatic_adaptation_vectorise(XYZ_w,
                                            XYZ_w,
                                            XYZ_b,
                                            L_A,
                                            F_L,
                                            XYZ_p,
                                            p,
                                            helson_judd_effect,
                                            discount_illuminant)

    # Computing opponent colour dimensions.
    # Computing achromatic post adaptation signals.
    A_a = achromatic_post_adaptation_signal_vectorise(rgb_a)
    A_aw = achromatic_post_adaptation_signal_vectorise(rgb_aw)

    # Computing colour difference signals.
    C = colour_difference_signals_vectorise(rgb_a)
    C_w = colour_difference_signals_vectorise(rgb_aw)

    # -------------------------------------------------------------------------
    # Computing the *hue* angle :math:`h_s`.
    # -------------------------------------------------------------------------
    h = hue_angle_vectorise(C)
    hue_w = hue_angle_vectorise(C_w)
    # TODO: Implement hue quadrature & composition computation.

    # -------------------------------------------------------------------------
    # Computing the correlate of *saturation* :math:`s`.
    # -------------------------------------------------------------------------
    # Computing eccentricity factors.
    e_s = eccentricity_factor_vectorise(h)

    # Computing low luminance tritanopia factor :math:`F_t`.
    F_t = low_luminance_tritanopia_factor_vectorise(L_A)

    M_yb = yellowness_blueness_response_vectorise(C, e_s, surround.N_c, N_cb,
                                                  F_t)
    M_rg = redness_greenness_response_vectorise(C, e_s, surround.N_c, N_cb)
    M_yb_w = yellowness_blueness_response_vectorise(C_w, e_s, surround.N_c,
                                                    N_cb, F_t)
    M_rg_w = redness_greenness_response_vectorise(C_w, e_s, surround.N_c, N_cb)

    # Computing overall chromatic response.
    M = overall_chromatic_response_vectorise(M_yb, M_rg)
    M_w = overall_chromatic_response_vectorise(M_yb_w, M_rg_w)

    s = saturation_correlate_vectorise(M, rgb_a)

    # -------------------------------------------------------------------------
    # Computing the correlate of *brightness* :math:`Q`.
    # -------------------------------------------------------------------------
    # Computing achromatic signal :math:`A`.
    A = achromatic_signal_vectorise(L_AS, S, S_W, N_bb, A_a)
    A_w = achromatic_signal_vectorise(L_AS, S_W, S_W, N_bb, A_aw)

    Q = brightness_correlate_vectorise(A, A_w, M, surround.N_b)
    brightness_w = brightness_correlate_vectorise(A_w, A_w, M_w, surround.N_b)
    # TODO: Implement whiteness-blackness :math:`Q_{wb}` computation.

    # -------------------------------------------------------------------------
    # Computing the correlate of *Lightness* :math:`J`.
    # -------------------------------------------------------------------------
    J = lightness_correlate_vectorise(Y_b, Y_w, Q, brightness_w)

    # -------------------------------------------------------------------------
    # Computing the correlate of *chroma* :math:`C_{94}`.
    # -------------------------------------------------------------------------
    C_94 = chroma_correlate_vectorise(s,
                                      Y_b,
                                      Y_w,
                                      Q,
                                      brightness_w)

    # -------------------------------------------------------------------------
    # Computing the correlate of *colourfulness* :math:`M_{94}`.
    # -------------------------------------------------------------------------
    M_94 = colourfulness_correlate_vectorise(F_L, C_94)

    return Hunt_Specification(J, C_94, h, s, Q, M_94, None, None)


def luminance_level_adaptation_factor_vectorise(L_A):
    L_A = np.asarray(L_A)

    k = 1 / (5 * L_A + 1)
    k4 = k ** 4
    F_L = (0.2
           * k4 * (5 * L_A) + 0.1 * (1 - k4) ** 2 * (5 * L_A) ** (1 / 3))

    return F_L


def illuminant_scotopic_luminance_vectorise(L_A, CCT):
    L_A = np.asarray(L_A)
    CCT = np.asarray(CCT)

    CCT = 2.26 * L_A * ((CCT / 4000) - 0.4) ** (1 / 3)

    return CCT


def XYZ_to_rgb_vectorise(XYZ):
    return np.einsum('...ij,...j->...i', XYZ_TO_HPE_MATRIX, XYZ)


def f_n_vectorise(x):
    x = np.asarray(x)

    x_m = 40 * ((x ** 0.73) / (x ** 0.73 + 2))

    return x_m


def chromatic_adaptation_vectorise(XYZ,
                                   XYZ_w,
                                   XYZ_b,
                                   L_A,
                                   F_L,
                                   XYZ_p=None,
                                   p=None,
                                   helson_judd_effect=False,
                                   discount_illuminant=True):
    XYZ_w = np.asarray(XYZ_w)
    XYZ_b = np.asarray(XYZ_b)
    L_A = np.asarray(L_A)
    F_L = np.asarray(F_L)

    rgb = XYZ_to_rgb_vectorise(XYZ)
    rgb_w = XYZ_to_rgb_vectorise(XYZ_w)
    Y_w = XYZ_w[..., 1]
    Y_b = XYZ_b[..., 1]

    h_rgb = 3 * rgb_w / np.sum(rgb_w, axis=-1)[..., np.newaxis]

    # Computing chromatic adaptation factors.
    if not discount_illuminant:
        F_rgb = ((1 + (L_A ** (1 / 3)) + h_rgb) /
                 (1 + (L_A ** (1 / 3)) + (1 / h_rgb)))
    else:
        F_rgb = np.ones(h_rgb.shape)

    # Computing Helson-Judd effect parameters.
    if helson_judd_effect:
        D_rgb = (f_n((Y_b / Y_w) * F_L * F_rgb[..., 1]) -
                 f_n((Y_b / Y_w) * F_L * F_rgb))
        # assert D_pyb[1] == 0
    else:
        D_rgb = np.zeros(F_rgb.shape)

    # Computing cone bleach factors.
    B_rgb = (10 ** 7) / ((10 ** 7) + 5 * L_A[..., np.newaxis] * (rgb_w / 100))

    # Computing adjusted reference white signals.
    if XYZ_p is not None and p is not None:
        rgb_p = XYZ_to_rgb_vectorise(XYZ_p)
        rgb_w = adjusted_reference_white_signals_vectorise(rgb_p, B_rgb, rgb_w,
                                                           p)

    # Computing adapted cone responses.
    rgb_a = 1 + B_rgb * (
        f_n_vectorise(F_L[..., np.newaxis] * F_rgb * rgb / rgb_w) + D_rgb)

    return rgb_a


def adjusted_reference_white_signals_vectorise(rgb_p, rgb_b, rgb_w, p):
    rgb_p = np.asarray(rgb_p)
    rgb_b = np.asarray(rgb_b)
    rgb_w = np.asarray(rgb_w)
    p = np.asarray(p)

    p_rgb = rgb_p / rgb_b
    rgb_w = (rgb_w * (((1 - p) * p_rgb + (1 + p) / p_rgb) ** 0.5) /
             (((1 + p) * p_rgb + (1 - p) / p_rgb) ** 0.5))

    return rgb_w


def achromatic_post_adaptation_signal_vectorise(rgb):
    r, g, b = tsplit(rgb)

    A = 2 * r + g + (1 / 20) * b - 3.05 + 1

    return A


def colour_difference_signals_vectorise(rgb):
    r, g, b = tsplit(rgb)

    C_1 = r - g
    C_2 = g - b
    C_3 = b - r

    C = tstack((C_1, C_2, C_3))

    return C


def hue_angle_vectorise(C):
    C_1, C_2, C_3 = tsplit(C)
    hue = (180 * np.arctan2(0.5 * (C_2 - C_3) / 4.5,
                            C_1 - (C_2 / 11)) / np.pi) % 360
    return hue


def eccentricity_factor_vectorise(hue):
    hue = np.asarray(hue)

    h_s = HUE_DATA_FOR_HUE_QUADRATURE.get('h_s')
    e_s = HUE_DATA_FOR_HUE_QUADRATURE.get('e_s')

    x = np.interp(hue, h_s, e_s)
    x = np.where(hue < 20.14, 0.856 - (hue / 20.14) * 0.056, x)
    x = np.where(hue > 237.53, 0.856 + 0.344 * (360 - hue) / (360 - 237.53), x)

    return x


def low_luminance_tritanopia_factor_vectorise(L_A):
    L_A = np.asarray(L_A)

    F_t = L_A / (L_A + 0.1)

    return F_t


def yellowness_blueness_response_vectorise(C, e_s, N_c, N_cb, F_t):
    C_1, C_2, C_3 = tsplit(C)
    e_s = np.asarray(e_s)
    N_c = np.asarray(N_c)
    N_cb = np.asarray(N_cb)
    F_t = np.asarray(F_t)

    M_yb = (100 * (0.5 * (C_2 - C_3) / 4.5) *
            (e_s * (10 / 13) * N_c * N_cb * F_t))

    return M_yb


def redness_greenness_response_vectorise(C, e_s, N_c, N_cb):
    C_1, C_2, C_3 = tsplit(C)
    e_s = np.asarray(e_s)
    N_c = np.asarray(N_c)
    N_cb = np.asarray(N_cb)

    M_rg = 100 * (C_1 - (C_2 / 11)) * (e_s * (10 / 13) * N_c * N_cb)

    return M_rg


def overall_chromatic_response_vectorise(M_yb, M_rg):
    M_yb = np.asarray(M_yb)
    M_rg = np.asarray(M_rg)

    M = ((M_yb ** 2) + (M_rg ** 2)) ** 0.5

    return M


def saturation_correlate_vectorise(M, rgb_a):
    M = np.asarray(M)
    rgb_a = np.asarray(rgb_a)

    s = 50 * M / np.sum(rgb_a, axis=-1)

    return s


def achromatic_signal_vectorise(L_AS, S, S_W, N_bb, A_a):
    L_AS = np.asarray(L_AS)
    S = np.asarray(S)
    S_W = np.asarray(S_W)
    N_bb = np.asarray(N_bb)
    A_a = np.asarray(A_a)

    j = 0.00001 / ((5 * L_AS / 2.26) + 0.00001)

    # Computing scotopic luminance level adaptation factor :math:`F_{LS}`.
    F_LS = 3800 * (j ** 2) * (5 * L_AS / 2.26)
    F_LS += 0.2 * ((1 - (j ** 2)) ** 0.4) * ((5 * L_AS / 2.26) ** (1 / 6))

    # Computing cone bleach factors :math:`B_S`.
    B_S = 0.5 / (1 + 0.3 * ((5 * L_AS / 2.26) * (S / S_W)) ** 0.3)
    B_S += 0.5 / (1 + 5 * (5 * L_AS / 2.26))

    # Computing adapted scotopic signal :math:`A_S`.
    A_S = (f_n(F_LS * S / S_W) * 3.05 * B_S) + 0.3

    # Computing achromatic signal :math:`A`.
    A = N_bb * (A_a - 1 + A_S - 0.3 + np.sqrt((1 + (0.3 ** 2))))

    return A


def brightness_correlate_vectorise(A, A_w, M, N_b):
    A = np.asarray(A)
    A_w = np.asarray(A_w)
    M = np.asarray(M)
    N_b = np.asarray(N_b)

    N_1 = ((7 * A_w) ** 0.5) / (5.33 * N_b ** 0.13)
    N_2 = (7 * A_w * N_b ** 0.362) / 200

    Q = ((7 * (A + (M / 100))) ** 0.6) * N_1 - N_2
    return Q


def lightness_correlate_vectorise(Y_b, Y_w, Q, Q_w):
    Y_b = np.asarray(Y_b)
    Y_w = np.asarray(Y_w)
    Q = np.asarray(Q)
    Q_w = np.asarray(Q_w)

    Z = 1 + (Y_b / Y_w) ** 0.5
    J = 100 * (Q / Q_w) ** Z

    return J


def chroma_correlate_vectorise(s, Y_b, Y_w, Q, Q_w):
    s = np.asarray(s)
    Y_b = np.asarray(Y_b)
    Y_w = np.asarray(Y_w)
    Q = np.asarray(Q)
    Q_w = np.asarray(Q_w)

    C_94 = (2.44 * (s ** 0.69) *
            ((Q / Q_w) ** (Y_b / Y_w)) *
            (1.64 - 0.29 ** (Y_b / Y_w)))

    return C_94


def colourfulness_correlate_vectorise(F_L, C_94):
    F_L = np.asarray(F_L)
    C_94 = np.asarray(C_94)

    M_94 = F_L ** 0.15 * C_94

    return M_94


def XYZ_to_Hunt_analysis():
    message_box('XYZ_to_Hunt')

    XYZ = np.array([19.01, 20.00, 21.78])
    XYZ_w = np.array([95.05, 100.00, 108.88])
    XYZ_b = np.array([95.05, 100.00, 108.88])
    L_A = 318.31
    surround = HUNT_VIEWING_CONDITIONS['Normal Scenes']
    CCT_w = 6504.0

    print('Reference:')
    print(XYZ_to_Hunt(XYZ, XYZ_w, XYZ_b, L_A, surround, CCT_w=CCT_w))

    print('\n')

    print('1d array input:')
    print(XYZ_to_Hunt_vectorise(XYZ, XYZ_w, XYZ_b, L_A, surround, CCT_w=CCT_w))

    print('\n')
    print('1.5d array input:')
    XYZ = np.tile(XYZ, (6, 1))
    print(XYZ_to_Hunt_vectorise(XYZ, XYZ_w, XYZ_b, L_A, surround, CCT_w=CCT_w))

    print('\n')

    print('2d array input:')
    XYZ_w = np.tile(XYZ_w, (6, 1))
    XYZ_b = np.tile(XYZ_b, (6, 1))
    L_A = np.tile(L_A, 6)
    CCT_w = np.tile(CCT_w, 6)
    print(XYZ_to_Hunt_vectorise(XYZ, XYZ_w, XYZ_b, L_A, surround, CCT_w=CCT_w))

    print('\n')

    print('3d array input:')
    XYZ = np.reshape(XYZ, (2, 3, 3))
    XYZ_w = np.reshape(XYZ_w, (2, 3, 3))
    XYZ_b = np.reshape(XYZ_b, (2, 3, 3))
    L_A = np.reshape(L_A, (2, 3))
    CCT_w = np.reshape(CCT_w, (2, 3))
    print(XYZ_to_Hunt_vectorise(XYZ, XYZ_w, XYZ_b, L_A, surround, CCT_w=CCT_w))

    print('\n')


# XYZ_to_Hunt_analysis()


def XYZ_to_Hunt_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    XYZ_w = np.array([95.05, 100.00, 108.88])
    XYZ_b = np.array([95.05, 100.00, 108.88])
    L_A = 318.31
    surround = HUNT_VIEWING_CONDITIONS['Normal Scenes']
    CCT_w = 6504.0

    times = timeit.Timer(
        functools.partial(
            XYZ_to_Hunt_2d,
            DATA_HD1, XYZ_w, XYZ_b, L_A, surround, CCT_w)).repeat(
        repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            XYZ_to_Hunt_vectorise,
            DATA_HD1, XYZ_w, XYZ_b, L_A, surround, CCT_w)).repeat(
        repeat_b, number_b)

    b = min(times) / number_b

    print('XYZ_to_Hunt\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# XYZ_to_Hunt_profile(number_a=3)

# #############################################################################
# #############################################################################
# ## colour.appearance.ciecam02
# #############################################################################
# #############################################################################
from colour.appearance.ciecam02 import *


def XYZ_to_CIECAM02_2d(XYZ, XYZ_w, L_A, Y_b):
    for i in range(len(XYZ)):
        XYZ_to_CIECAM02(XYZ[i], XYZ_w, L_A, Y_b)


def CIECAM02_to_XYZ_2d(J, C, h, XYZ_w, L_A, Y_b):
    for i in range(len(J)):
        CIECAM02_to_XYZ(J[i][0], C, h, XYZ_w, L_A, Y_b)


def XYZ_to_CIECAM02_vectorise(XYZ,
                              XYZ_w,
                              L_A,
                              Y_b,
                              surround=CIECAM02_VIEWING_CONDITIONS.get(
                                  'Average'),
                              discount_illuminant=False):
    X_w, Y_w, Z_w = tsplit(XYZ_w)
    L_A = np.asarray(L_A)
    Y_b = np.asarray(Y_b)

    n, F_L, N_bb, N_cb, z = tsplit(
        viewing_condition_dependent_parameters_vectorise(Y_b, Y_w, L_A))

    # Converting *CIE XYZ* tristimulus values matrices to CMCCAT2000 transform
    # sharpened *RGB* values.
    RGB = np.einsum('...ij,...j->...i', CAT02_CAT, XYZ)
    RGB_w = np.einsum('...ij,...j->...i', CAT02_CAT, XYZ_w)

    # Computing degree of adaptation :math:`D`.
    D = degree_of_adaptation_vectorise(surround.F,
                                       L_A) if not discount_illuminant else 1

    # Computing full chromatic adaptation.
    RGB_c = full_chromatic_adaptation_forward_vectorise(RGB, RGB_w,
                                                        Y_w[..., np.newaxis],
                                                        D[..., np.newaxis])
    RGB_wc = full_chromatic_adaptation_forward_vectorise(RGB_w, RGB_w,
                                                         Y_w[..., np.newaxis],
                                                         D[..., np.newaxis])

    # Converting to *Hunt-Pointer-Estevez* colourspace.
    RGB_p = RGB_to_rgb_vectorise(RGB_c)
    RGB_pw = RGB_to_rgb_vectorise(RGB_wc)

    # Applying forward post-adaptation non linear response compression.
    RGB_a = post_adaptation_non_linear_response_compression_forward_vectorise(
        RGB_p, F_L[..., np.newaxis])
    RGB_aw = post_adaptation_non_linear_response_compression_forward_vectorise(
        RGB_pw, F_L[..., np.newaxis])

    # Converting to preliminary cartesian coordinates.
    a, b = tsplit(opponent_colour_dimensions_forward_vectorise(RGB_a))

    # -------------------------------------------------------------------------
    # Computing the *hue* angle :math:`h`.
    h = hue_angle_vectorise(a, b)
    # -------------------------------------------------------------------------
    # Computing hue :math:`h` quadrature :math:`H`.
    H = hue_quadrature_vectorise(h)
    # TODO: Compute hue composition.

    # Computing eccentricity factor *e_t*.
    e_t = eccentricity_factor_vectorise(h)

    # Computing achromatic responses for the stimulus and the whitepoint.
    A = achromatic_response_forward_vectorise(RGB_a, N_bb)
    A_w = achromatic_response_forward_vectorise(RGB_aw, N_bb)

    # -------------------------------------------------------------------------
    # Computing the correlate of *Lightness* :math:`J`.
    # -------------------------------------------------------------------------
    J = lightness_correlate_vectorise(A, A_w, surround.c, z)

    # -------------------------------------------------------------------------
    # Computing the correlate of *brightness* :math:`Q`.
    # -------------------------------------------------------------------------
    Q = brightness_correlate_vectorise(surround.c, J, A_w, F_L)

    # -------------------------------------------------------------------------
    # Computing the correlate of *chroma* :math:`C`.
    # -------------------------------------------------------------------------
    C = chroma_correlate_vectorise(J, n, surround.N_c, N_cb, e_t, a, b, RGB_a)

    # -------------------------------------------------------------------------
    # Computing the correlate of *colourfulness* :math:`M`.
    # -------------------------------------------------------------------------
    M = colourfulness_correlate_vectorise(C, F_L)

    # -------------------------------------------------------------------------
    # Computing the correlate of *saturation* :math:`s`.
    # -------------------------------------------------------------------------
    s = saturation_correlate_vectorise(M, Q)

    return CIECAM02_Specification(J, C, h, s, Q, M, H, None)


def CIECAM02_to_XYZ_vectorise(J,
                              C,
                              h,
                              XYZ_w,
                              L_A,
                              Y_b,
                              surround=CIECAM02_VIEWING_CONDITIONS.get(
                                  'Average'),
                              discount_illuminant=False):
    X_w, Y_w, Zw = tsplit(XYZ_w)

    n, F_L, N_bb, N_cb, z = tsplit(
        viewing_condition_dependent_parameters_vectorise(
            Y_b, Y_w, L_A))

    # Converting *CIE XYZ* tristimulus values matrices to CMCCAT2000 transform
    # sharpened *RGB* values.
    RGB_w = np.einsum('...ij,...j->...i', CAT02_CAT, XYZ_w)

    # Computing degree of adaptation :math:`D`.
    D = degree_of_adaptation_vectorise(surround.F,
                                       L_A) if not discount_illuminant else 1

    # Computation full chromatic adaptation.
    RGB_wc = full_chromatic_adaptation_forward_vectorise(RGB_w, RGB_w,
                                                         Y_w[..., np.newaxis],
                                                         D[..., np.newaxis])

    # Converting to *Hunt-Pointer-Estevez* colourspace.
    RGB_pw = RGB_to_rgb_vectorise(RGB_wc)

    # Applying post-adaptation non linear response compression.
    RGB_aw = post_adaptation_non_linear_response_compression_forward_vectorise(
        RGB_pw, F_L[..., np.newaxis])

    # Computing achromatic responses for the stimulus and the whitepoint.
    A_w = achromatic_response_forward_vectorise(RGB_aw, N_bb)

    # Computing temporary magnitude quantity :math:`t`.
    t = temporary_magnitude_quantity_reverse_vectorise(C, J, n)

    # Computing eccentricity factor *e_t*.
    e_t = eccentricity_factor_vectorise(h)

    # Computing achromatic response :math:`A` for the stimulus.
    A = achromatic_response_reverse_vectorise(A_w, J, surround.c, z)

    # Computing *P_1* to *P_3*.
    P = P_vectorise(surround.N_c, N_cb, e_t, t, A, N_bb)
    P_1, P_2, P_3 = tsplit(P)

    # Computing opponent colour dimensions :math:`a` and :math:`b`.
    a, b = tsplit(opponent_colour_dimensions_reverse_vectorise(P, h))

    # Computing post-adaptation non linear response compression matrix.
    RGB_a = post_adaptation_non_linear_response_compression_matrix_vectorise(
        P_2, a, b)

    # Applying reverse post-adaptation non linear response compression.
    RGB_p = post_adaptation_non_linear_response_compression_reverse_vectorise(
        RGB_a, F_L[..., np.newaxis])

    # Converting to *Hunt-Pointer-Estevez* colourspace.
    RGB_c = rgb_to_RGB_vectorise(RGB_p)

    # Applying reverse full chromatic adaptation.
    RGB = full_chromatic_adaptation_reverse_vectorise(RGB_c, RGB_w,
                                                      Y_w[..., np.newaxis],
                                                      D[..., np.newaxis])

    # Converting CMCCAT2000 transform sharpened *RGB* values to *CIE XYZ*
    # tristimulus values.
    XYZ = np.einsum('...ij,...j->...i', CAT02_INVERSE_CAT, RGB)

    return XYZ


def chromatic_induction_factors_vectorise(n):
    n = np.asarray(n)

    N_cbb = 0.725 * (1 / n) ** 0.2
    N_cbb = tstack((N_cbb, N_cbb))

    return N_cbb


def base_exponential_non_linearity_vectorise(n):
    n = np.asarray(n)

    z = 1.48 + np.sqrt(n)

    return z


def viewing_condition_dependent_parameters_vectorise(Y_b, Y_w, L_A):
    Y_b = np.asarray(Y_b)
    Y_w = np.asarray(Y_w)

    n = Y_b / Y_w

    F_L = luminance_level_adaptation_factor_vectorise(L_A)
    N_bb, N_cb = tsplit(chromatic_induction_factors_vectorise(n))
    z = base_exponential_non_linearity_vectorise(n)

    return tstack((n, F_L, N_bb, N_cb, z))


def degree_of_adaptation_vectorise(F, L_A):
    F = np.asarray(F)
    L_A = np.asarray(L_A)

    D = F * (1 - (1 / 3.6) * np.exp((-L_A - 42) / 92))

    return D


def full_chromatic_adaptation_forward_vectorise(RGB, RGB_w, Y_w, D):
    RGB = np.asarray(RGB)
    RGB_w = np.asarray(RGB_w)
    Y_w = np.asarray(Y_w)
    D = np.asarray(D)

    RGB_c = ((Y_w * D / RGB_w) + 1 - D) * RGB

    return RGB_c


def full_chromatic_adaptation_reverse_vectorise(RGB, RGB_w, Y_w, D):
    RGB = np.asarray(RGB)
    RGB_w = np.asarray(RGB_w)
    Y_w = np.asarray(Y_w)
    D = np.asarray(D)

    RGB_c = RGB / (Y_w * (D / RGB_w) + 1 - D)

    return RGB_c


def RGB_to_rgb_vectorise(RGB):
    rgb = np.einsum('...ij,...j->...i',
                    np.einsum('...ij,...jk->...ik', XYZ_TO_HPE_MATRIX,
                              CAT02_INVERSE_CAT),
                    RGB)

    return rgb


def rgb_to_RGB_vectorise(rgb):
    RGB = np.einsum('...ij,...j->...i',
                    np.einsum('...ij,...jk->...ik', CAT02_CAT,
                              HPE_TO_XYZ_MATRIX),
                    rgb)

    return RGB


def post_adaptation_non_linear_response_compression_forward_vectorise(RGB,
                                                                      F_L):
    RGB = np.asarray(RGB)
    F_L = np.asarray(F_L)

    # TODO: Check for negative values and their handling.
    RGB_c = ((((400 * (F_L * RGB / 100) ** 0.42) /
               (27.13 + (F_L * RGB / 100) ** 0.42))) + 0.1)

    return RGB_c


def post_adaptation_non_linear_response_compression_reverse_vectorise(RGB,
                                                                      F_L):
    RGB = np.asarray(RGB)
    F_L = np.asarray(F_L)

    RGB_p = ((np.sign(RGB - 0.1) *
              (100 / F_L) * ((27.13 * np.abs(RGB - 0.1)) /
                             (400 - np.abs(RGB - 0.1))) ** (1 / 0.42)))

    return RGB_p


def opponent_colour_dimensions_forward_vectorise(RGB):
    R, G, B = tsplit(RGB)

    a = R - 12 * G / 11 + B / 11
    b = (R + G - 2 * B) / 9

    ab = tstack((a, b))

    return ab


def opponent_colour_dimensions_reverse_vectorise(P, h):
    P_1, P_2, P_3 = tsplit(P)
    hr = np.radians(h)

    sin_hr = np.sin(hr)
    cos_hr = np.cos(hr)

    P_4 = P_1 / sin_hr
    P_5 = P_1 / cos_hr
    n = P_2 * (2 + P_3) * (460 / 1403)

    a = np.zeros(hr.shape)
    b = np.zeros(hr.shape)

    b = np.where(np.abs(sin_hr) >= np.abs(cos_hr),
                 (n / (P_4 + (2 + P_3) * (220 / 1403) * (cos_hr / sin_hr) -
                       (27 / 1403) + P_3 * (6300 / 1403))),
                 b)

    a = np.where(np.abs(sin_hr) >= np.abs(cos_hr), b * (cos_hr / sin_hr), a)

    a = np.where(np.abs(sin_hr) < np.abs(cos_hr),
                 (n / (P_5 + (2 + P_3) * (220 / 1403) -
                       ((27 / 1403) - P_3 * (6300 / 1403)) *
                       (sin_hr / cos_hr))),
                 a)

    b = np.where(np.abs(sin_hr) < np.abs(cos_hr), a * (sin_hr / cos_hr), b)

    ab = tstack((a, b))

    return ab


def hue_angle_vectorise(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    h = np.degrees(np.arctan2(b, a)) % 360
    return h


def hue_quadrature_vectorise(h):
    h = np.nan_to_num(h)

    h_i = HUE_DATA_FOR_HUE_QUADRATURE.get('h_i')
    e_i = HUE_DATA_FOR_HUE_QUADRATURE.get('e_i')
    H_i = HUE_DATA_FOR_HUE_QUADRATURE.get('H_i')

    i = np.searchsorted(h_i, h, side='left') - 1

    h_ii = h_i[i]
    e_ii = e_i[i]
    H_ii = H_i[i]
    h_ii1 = h_i[i + 1]
    e_ii1 = e_i[i + 1]

    H = H_ii + ((100 * (h - h_ii) / e_ii) /
                ((h - h_ii) / e_ii + (h_ii1 - h) / e_ii1))
    H = np.where(h < 20.14,
                 385.9 + (14.1 * h / 0.856) / (h / 0.856 + (20.14 - h) / 0.8),
                 H)
    H = np.where(h >= 237.53,
                 H_ii + ((85.9 * (h - h_ii) / e_ii) /
                         ((h - h_ii) / e_ii + (360 - h) / 0.856)),
                 H)
    return H


def eccentricity_factor_vectorise(h):
    h = np.asarray(h)

    e_t = 1 / 4 * (np.cos(2 + h * np.pi / 180) + 3.8)

    return e_t


def achromatic_response_forward_vectorise(RGB, N_bb):
    R, G, B = tsplit(RGB)

    A = (2 * R + G + (1 / 20) * B - 0.305) * N_bb

    return A


def achromatic_response_reverse_vectorise(A_w, J, c, z):
    A_w = np.asarray(A_w)
    J = np.asarray(J)
    c = np.asarray(c)
    z = np.asarray(z)

    A = A_w * (J / 100) ** (1 / (c * z))

    return A


def lightness_correlate_vectorise(A, A_w, c, z):
    A = np.asarray(A)
    A_w = np.asarray(A_w)
    c = np.asarray(c)
    z = np.asarray(z)

    J = 100 * (A / A_w) ** (c * z)

    return J


def brightness_correlate_vectorise(c, J, A_w, F_L):
    c = np.asarray(c)
    J = np.asarray(J)
    A_w = np.asarray(A_w)
    F_L = np.asarray(F_L)

    Q = (4 / c) * np.sqrt(J / 100) * (A_w + 4) * F_L ** 0.25

    return Q


def temporary_magnitude_quantity_forward_vectorise(N_c, N_cb, e_t, a, b,
                                                   RGB_a):
    N_c = np.asarray(N_c)
    N_cb = np.asarray(N_cb)
    e_t = np.asarray(e_t)
    a = np.asarray(a)
    b = np.asarray(b)
    Ra, Ga, Ba = tsplit(RGB_a)

    t = ((50000 / 13) * N_c * N_cb) * (e_t * (a ** 2 + b ** 2) ** 0.5) / (
        Ra + Ga + 21 * Ba / 20)

    return t


def temporary_magnitude_quantity_reverse_vectorise(C, J, n):
    C = np.asarray(C)
    J = np.asarray(J)
    n = np.asarray(n)

    t = (C / (np.sqrt(J / 100) * (1.64 - 0.29 ** n) ** 0.73)) ** (1 / 0.9)

    return t


def chroma_correlate_vectorise(J, n, N_c, N_cb, e_t, a, b, RGB_a):
    J = np.asarray(J)
    n = np.asarray(n)

    t = temporary_magnitude_quantity_forward_vectorise(N_c, N_cb, e_t, a, b,
                                                       RGB_a)
    C = t ** 0.9 * (J / 100) ** 0.5 * (1.64 - 0.29 ** n) ** 0.73

    return C


def colourfulness_correlate_vectorise(C, F_L):
    C = np.asarray(C)
    F_L = np.asarray(F_L)

    M = C * F_L ** 0.25

    return M


def saturation_correlate_vectorise(M, Q):
    M = np.asarray(M)
    Q = np.asarray(Q)

    s = 100 * (M / Q) ** 0.5

    return s


def P_vectorise(N_c, N_cb, e_t, t, A, N_bb):
    N_c = np.asarray(N_c)
    N_cb = np.asarray(N_cb)
    e_t = np.asarray(e_t)
    t = np.asarray(t)
    A = np.asarray(A)
    N_bb = np.asarray(N_bb)

    P_1 = ((50000 / 13) * N_c * N_cb * e_t) / t
    P_2 = A / N_bb + 0.305
    P_3 = np.ones(P_1.shape) * (21 / 20)

    P = tstack((P_1, P_2, P_3))

    return P


def post_adaptation_non_linear_response_compression_matrix_vectorise(P_2, a,
                                                                     b):
    P_2 = np.asarray(P_2)
    a = np.asarray(a)
    b = np.asarray(b)

    R_a = (460 * P_2 + 451 * a + 288 * b) / 1403
    G_a = (460 * P_2 - 891 * a - 261 * b) / 1403
    B_a = (460 * P_2 - 220 * a - 6300 * b) / 1403

    RGB_a = tstack((R_a, G_a, B_a))

    return RGB_a


def XYZ_to_CIECAM02_analysis():
    message_box('XYZ_to_CIECAM02')

    XYZ = np.array([19.01, 20.00, 21.78])
    XYZ_w = np.array([95.05, 100.00, 108.88])
    L_A = 318.31
    Y_b = 20.0
    surround = CIECAM02_VIEWING_CONDITIONS['Average']

    print('Reference:')
    print(XYZ_to_CIECAM02(XYZ, XYZ_w, L_A, Y_b, surround))

    print('\n')

    print('1d array input:')
    print(XYZ_to_CIECAM02_vectorise(XYZ, XYZ_w, L_A, Y_b, surround))

    print('\n')

    print('1.5d array input:')
    XYZ = np.tile(XYZ, (6, 1))
    print(XYZ_to_CIECAM02_vectorise(XYZ, XYZ_w, L_A, Y_b, surround))

    print('\n')

    print('2d array input:')
    XYZ_w = np.tile(XYZ_w, (6, 1))
    L_A = np.tile(L_A, 6)
    Y_b = np.tile(Y_b, 6)
    print(XYZ_to_CIECAM02_vectorise(XYZ, XYZ_w, L_A, Y_b, surround))

    print('\n')

    print('3d array input:')
    XYZ = np.reshape(XYZ, (2, 3, 3))
    XYZ_w = np.reshape(XYZ_w, (2, 3, 3))
    L_A = np.reshape(L_A, (2, 3))
    Y_b = np.reshape(Y_b, (2, 3))
    print(XYZ_to_CIECAM02_vectorise(XYZ, XYZ_w, L_A, Y_b, surround))

    print('\n')


# XYZ_to_CIECAM02_analysis()


def XYZ_to_CIECAM02_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    XYZ_w = np.array([95.05, 100.00, 108.88])
    L_A = 318.31
    Y_b = 20.0

    times = timeit.Timer(
        functools.partial(
            XYZ_to_CIECAM02_2d,
            DATA_VGA1, XYZ_w, L_A, Y_b)).repeat(
        repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            XYZ_to_CIECAM02_vectorise,
            DATA_VGA1, XYZ_w, L_A, Y_b)).repeat(
        repeat_b, number_b)

    b = min(times) / number_b

    print('XYZ_to_CIECAM02\t{0}\t{1}\t{2}'.format(
        len(DATA_VGA1), a, b))


# XYZ_to_CIECAM02_profile()


def CIECAM02_to_XYZ_analysis():
    message_box('CIECAM02_to_XYZ')

    J = 42.53
    C = 51.92
    h = 248.9
    XYZ_w = np.array([109.85, 100, 35.58])
    L_A = 31.83
    Y_b = 20.0

    print('Reference:')
    print(CIECAM02_to_XYZ(J, C, h, XYZ_w, L_A, Y_b))

    print('\n')

    print('Numeric input:')
    print(CIECAM02_to_XYZ_vectorise(J, C, h, XYZ_w, L_A, Y_b))

    print('\n')

    print('1d array input:')
    J = np.tile(J, 6)
    C = np.tile(C, 6)
    h = np.tile(h, 6)
    XYZ_w = np.tile(XYZ_w, (6, 1))
    L_A = np.tile(L_A, 6)
    Y_b = np.tile(Y_b, 6)
    print(CIECAM02_to_XYZ_vectorise(J, C, h, XYZ_w, L_A, Y_b))

    print('\n')

    print('2d array input:')
    J = np.reshape(J, (2, 3))
    C = np.reshape(C, (2, 3))
    h = np.reshape(h, (2, 3))
    XYZ_w = np.reshape(XYZ_w, (2, 3, 3))
    L_A = np.reshape(L_A, (2, 3))
    Y_b = np.reshape(Y_b, (2, 3))
    print(CIECAM02_to_XYZ_vectorise(J, C, h, XYZ_w, L_A, Y_b))

    print('\n')


# CIECAM02_to_XYZ_analysis()


def CIECAM02_to_XYZ_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    C = 0.1047077571711053
    h = 219.0484326582719
    XYZ_w = np.array([95.05, 100.00, 108.88])
    L_A = 318.31
    Y_b = 20.0

    times = timeit.Timer(
        functools.partial(
            CIECAM02_to_XYZ_2d,
            DATA_VGA1, C, h, XYZ_w, L_A, Y_b)).repeat(
        repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            CIECAM02_to_XYZ_vectorise,
            DATA_VGA1, C, h, XYZ_w, L_A, Y_b)).repeat(
        repeat_b, number_b)

    b = min(times) / number_b

    print('CIECAM02_to_XYZ\t{0}\t{1}\t{2}'.format(
        len(DATA_VGA1), a, b))


# CIECAM02_to_XYZ_profile()


# #############################################################################
# #############################################################################
# ## colour.appearance.llab
# #############################################################################
# #############################################################################
from colour.appearance.llab import *


def XYZ_to_LLAB_2d(XYZ, XYZ_0, Y_b, L, surround):
    for i in range(len(XYZ)):
        XYZ_to_LLAB(XYZ[i], XYZ_0, Y_b, L, surround)


def XYZ_to_LLAB_vectorise(
        XYZ,
        XYZ_0,
        Y_b,
        L,
        surround=LLAB_VIEWING_CONDITIONS.get(
            'Reference Samples & Images, Average Surround, Subtending < 4')):
    X, Y, Z = tsplit(XYZ)
    RGB = XYZ_to_RGB_LLAB_vectorise(XYZ)
    RGB_0 = XYZ_to_RGB_LLAB_vectorise(XYZ_0)

    # Reference illuminant *CIE Standard Illuminant D Series* *D65*.
    XYZ_0r = np.array([95.05, 100, 108.88])
    RGB_0r = XYZ_to_RGB_LLAB_vectorise(XYZ_0r)

    # Computing chromatic adaptation.
    XYZ_r = chromatic_adaptation_vectorise(RGB, RGB_0, RGB_0r, Y, surround.D)

    # -------------------------------------------------------------------------
    # Computing the correlate of *Lightness* :math:`L_L`.
    # -------------------------------------------------------------------------
    # Computing opponent colour dimensions.
    L_L, a, b = tsplit(opponent_colour_dimensions_vectorise(XYZ_r,
                                                            Y_b,
                                                            surround.F_S,
                                                            surround.F_L))

    # Computing perceptual correlates.
    # -------------------------------------------------------------------------
    # Computing the correlate of *chroma* :math:`Ch_L`.
    # -------------------------------------------------------------------------
    Ch_L = chroma_correlate_vectorise(a, b)

    # -------------------------------------------------------------------------
    # Computing the correlate of *colourfulness* :math:`C_L`.
    # -------------------------------------------------------------------------
    C_L = colourfulness_correlate_vectorise(L, L_L, Ch_L, surround.F_C)

    # -------------------------------------------------------------------------
    # Computing the correlate of *saturation* :math:`s_L`.
    # -------------------------------------------------------------------------
    s_L = saturation_correlate_vectorise(Ch_L, L_L)

    # -------------------------------------------------------------------------
    # Computing the *hue* angle :math:`h_L`.
    # -------------------------------------------------------------------------
    h_L = hue_angle_vectorise(a, b)
    h_Lr = np.radians(h_L)
    # TODO: Implement hue composition computation.


    # -------------------------------------------------------------------------
    # Computing final opponent signals.
    # -------------------------------------------------------------------------
    A_L, B_L = tsplit(final_opponent_signals_vectorise(C_L, h_Lr))

    return LLAB_Specification(L_L, Ch_L, h_L, s_L, C_L, None, A_L, B_L)


def XYZ_to_RGB_LLAB_vectorise(XYZ):
    X, Y, Z = tsplit(XYZ)

    Y = tstack((Y, Y, Y))
    XYZ_n = XYZ / Y

    return np.einsum('...ij,...j->...i', LLAB_XYZ_TO_RGB_MATRIX, XYZ_n)


def chromatic_adaptation_vectorise(RGB, RGB_0, RGB_0r, Y, D=1):
    R, G, B = tsplit(RGB)
    R_0, G_0, B_0 = tsplit(RGB_0)
    R_0r, G_0r, B_0r = tsplit(RGB_0r)
    Y = np.asarray(Y)

    beta = (B_0 / B_0r) ** 0.0834

    R_r = (D * (R_0r / R_0) + 1 - D) * R
    G_r = (D * (G_0r / G_0) + 1 - D) * G
    B_r = (D * (B_0r / (B_0 ** beta)) + 1 - D) * (abs(B) ** beta)

    RGB_r = tstack((R_r, G_r, B_r))

    Y = tstack((Y, Y, Y))

    XYZ_r = np.einsum('...ij,...j->...i', LLAB_RGB_TO_XYZ_MATRIX, RGB_r * Y)

    return XYZ_r


def f_vectorise(x, F_S):
    x = np.asarray(x)
    F_S = np.asarray(F_S)

    x_m = np.where(x > 0.008856,
                   x ** (1 / F_S),
                   ((((0.008856 ** (1 / F_S)) -
                      (16 / 116)) / 0.008856) * x + (16 / 116)))
    return x_m


def opponent_colour_dimensions_vectorise(XYZ, Y_b, F_S, F_L):
    X, Y, Z = tsplit(XYZ)
    Y_b = np.asarray(Y_b)
    F_S = np.asarray(F_S)
    F_L = np.asarray(F_L)

    # Account for background lightness contrast.
    z = 1 + F_L * ((Y_b / 100) ** 0.5)

    # Computing modified *CIE Lab* colourspace matrix.
    L = 116 * (f_vectorise(Y / 100, F_S) ** z) - 16
    a = 500 * (f_vectorise(X / 95.05, F_S) - f_vectorise(Y / 100, F_S))
    b = 200 * (f_vectorise(Y / 100, F_S) - f_vectorise(Z / 108.88, F_S))

    Lab = tstack((L, a, b))

    return Lab


def hue_angle_vectorise(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    h_L = np.degrees(np.arctan2(b, a)) % 360

    return h_L


def chroma_correlate_vectorise(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    c = (a ** 2 + b ** 2) ** 0.5
    Ch_L = 25 * np.log(1 + 0.05 * c)

    return Ch_L


def colourfulness_correlate_vectorise(L, L_L, Ch_L, F_C):
    L = np.asarray(L)
    L_L = np.asarray(L_L)
    Ch_L = np.asarray(Ch_L)
    F_C = np.asarray(F_C)

    S_C = 1 + 0.47 * np.log10(L) - 0.057 * np.log10(L) ** 2
    S_M = 0.7 + 0.02 * L_L - 0.0002 * L_L ** 2
    C_L = Ch_L * S_M * S_C * F_C

    return C_L


def saturation_correlate_vectorise(Ch_L, L_L):
    Ch_L = np.asarray(Ch_L)
    L_L = np.asarray(L_L)

    S_L = Ch_L / L_L

    return S_L


def final_opponent_signals_vectorise(C_L, h_L):
    C_L = np.asarray(C_L)
    h_L = np.asarray(h_L)

    A_L = C_L * np.cos(h_L)
    B_L = C_L * np.sin(h_L)

    AB_L = tstack((A_L, B_L))

    return AB_L


def XYZ_to_LLAB_analysis():
    message_box('XYZ_to_LLAB')

    XYZ = np.array([19.01, 20, 21.78])
    XYZ_0 = np.array([95.05, 100, 108.88])
    Y_b = 20.0
    L = 318.31
    surround = LLAB_VIEWING_CONDITIONS['ref_average_4_minus']

    print('Reference:')
    print(XYZ_to_LLAB(XYZ, XYZ_0, Y_b, L, surround))

    print('\n')

    print('1d array input:')
    print(XYZ_to_LLAB_vectorise(XYZ, XYZ_0, Y_b, L, surround))

    print('\n')

    print('1.5d array input:')
    XYZ = np.tile(XYZ, (6, 1))
    print(XYZ_to_LLAB_vectorise(XYZ, XYZ_0, Y_b, L, surround))

    print('\n')

    print('2d array input:')
    XYZ_0 = np.tile(XYZ_0, (6, 1))
    Y_b = np.tile(Y_b, 6)
    L = np.tile(L, 6)
    print(XYZ_to_LLAB_vectorise(XYZ, XYZ_0, Y_b, L, surround))

    print('\n')

    print('3d array input:')
    XYZ = np.reshape(XYZ, (2, 3, 3))
    XYZ_0 = np.reshape(XYZ_0, (2, 3, 3))
    Y_b = np.reshape(Y_b, (2, 3))
    L = np.reshape(L, (2, 3))
    print(XYZ_to_LLAB_vectorise(XYZ, XYZ_0, Y_b, L, surround))

    print('\n')


# XYZ_to_LLAB_analysis()


def XYZ_to_LLAB_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    XYZ_0 = np.array([95.05, 100, 108.88])
    Y_b = 20.0
    L = 318.31
    surround = LLAB_VIEWING_CONDITIONS['ref_average_4_minus']

    times = timeit.Timer(
        functools.partial(
            XYZ_to_LLAB_2d,
            DATA_VGA1, XYZ_0, Y_b, L, surround)).repeat(
        repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            XYZ_to_LLAB_vectorise,
            DATA_VGA1, XYZ_0, Y_b, L, surround)).repeat(
        repeat_b, number_b)

    b = min(times) / number_b

    print('XYZ_to_LLAB\t{0}\t{1}\t{2}'.format(
        len(DATA_VGA1), a, b))


# XYZ_to_LLAB_profile()

# #############################################################################
# #############################################################################
# ## colour.appearance.nayatani95
# #############################################################################
# #############################################################################
from colour.appearance.nayatani95 import *
from colour.models import XYZ_to_xy


def XYZ_to_Nayatani95_2d(XYZ, XYZ_n, Y_o, E_o, E_or):
    for i in range(len(XYZ)):
        XYZ_to_Nayatani95(XYZ[i], XYZ_n, Y_o, E_o, E_or)


def XYZ_to_Nayatani95_vectorise(XYZ,
                                XYZ_n,
                                Y_o,
                                E_o,
                                E_or,
                                n=1):
    if np.any(Y_o < 18) or np.any(Y_o > 100):
        warning(('"Y_o" luminance factor must be in [18, 100] domain, '
                 'unpredictable results may occur!'))

    X, Y, Z = tsplit(XYZ)
    Y_o = np.asarray(Y_o)
    E_o = np.asarray(E_o)
    E_or = np.asarray(E_or)

    # Computing adapting luminance :math:`L_o` and normalising luminance
    # :math:`L_{or}` in in :math:`cd/m^2`.
    # L_o = illuminance_to_luminance(E_o, Y_o)
    L_or = illuminance_to_luminance_vectorise(E_or, Y_o)

    # Computing :math:`\xi`, :math:`\eta`, :math:`\zeta` values.
    xez = intermediate_values_vectorise(XYZ_to_xy(XYZ_n))
    xi, eta, zeta = tsplit(xez)

    # Computing adapting field cone responses.
    RGB_o = (((Y_o[..., np.newaxis] * E_o[..., np.newaxis]) /
              (100 * np.pi)) * xez)

    # Computing stimulus cone responses.
    RGB = XYZ_to_RGB_Nayatani95_vectorise(XYZ)
    R, G, B = tsplit(RGB)

    # Computing exponential factors of the chromatic adaptation.
    bRGB_o = exponential_factors_vectorise(RGB_o)
    bL_or = beta_1_vectorise(L_or)

    # Computing scaling coefficients :math:`e(R)` and :math:`e(G)`
    eR = scaling_coefficient_vectorise(R, xi)
    eG = scaling_coefficient_vectorise(G, eta)

    # Computing opponent colour dimensions.
    # Computing achromatic response :math:`Q`:
    Q_response = achromatic_response_vectorise(RGB, bRGB_o, xez,
                                               bL_or, eR, eG, n)

    # Computing tritanopic response :math:`t`:
    t_response = tritanopic_response_vectorise(RGB, bRGB_o, xez, n)

    # Computing protanopic response :math:`p`:
    p_response = protanopic_response_vectorise(RGB, bRGB_o, xez, n)

    # -------------------------------------------------------------------------
    # Computing the correlate of *brightness* :math:`B_r`.
    # -------------------------------------------------------------------------
    B_r = brightness_correlate_vectorise(bRGB_o, bL_or, Q_response)

    # Computing *brightness* :math:`B_{rw}` of ideal white.
    brightness_ideal_white = ideal_white_brightness_correlate_vectorise(bRGB_o,
                                                                        xez,
                                                                        bL_or,
                                                                        n)

    # -------------------------------------------------------------------------
    # Computing the correlate of achromatic *Lightness* :math:`L_p^\star`.
    # -------------------------------------------------------------------------
    Lstar_P = (
        achromatic_lightness_correlate_vectorise(Q_response))

    # -------------------------------------------------------------------------
    # Computing the correlate of normalised achromatic *Lightness*
    # :math:`L_n^\star`.
    # -------------------------------------------------------------------------
    Lstar_N = (
        normalised_achromatic_lightness_correlate_vectorise(B_r,
                                                            brightness_ideal_white))

    # -------------------------------------------------------------------------
    # Computing the *hue* angle :math:`\\theta`.
    # -------------------------------------------------------------------------
    theta = hue_angle_vectorise(p_response, t_response)
    # TODO: Implement hue quadrature & composition computation.

    # -------------------------------------------------------------------------
    # Computing the correlate of *saturation* :math:`S`.
    # -------------------------------------------------------------------------
    S_RG, S_YB = saturation_components_vectorise(theta, bL_or,
                                                 t_response,
                                                 p_response)
    S = saturation_correlate_vectorise(S_RG, S_YB)

    # -------------------------------------------------------------------------
    # Computing the correlate of *chroma* :math:`C`.
    # -------------------------------------------------------------------------
    C_RG, C_YB = chroma_components_vectorise(Lstar_P, S_RG, S_YB)
    C = chroma_correlate_vectorise(Lstar_P, S)

    # -------------------------------------------------------------------------
    # Computing the correlate of *colourfulness* :math:`M`.
    # -------------------------------------------------------------------------
    # TODO: Investigate components usage.
    # M_RG, M_YB = colourfulness_components(C_RG, C_YB,
    # brightness_ideal_white)
    M = colourfulness_correlate_vectorise(C, brightness_ideal_white)

    return Nayatani95_Specification(Lstar_P,
                                    C,
                                    theta,
                                    S,
                                    B_r,
                                    M,
                                    None,
                                    None,
                                    Lstar_N)


def illuminance_to_luminance_vectorise(E, Y_f):
    E = np.asarray(E)
    Y_f = np.asarray(Y_f)

    return Y_f * E / (100 * np.pi)


def XYZ_to_RGB_Nayatani95_vectorise(XYZ):
    return np.einsum('...ij,...j->...i', NAYATANI95_XYZ_TO_RGB_MATRIX, XYZ)


def scaling_coefficient_vectorise(x, y):
    return np.where(x >= (20 * y), 1.758, 1)


def achromatic_response_vectorise(RGB, bRGB_o, xez, bL_or, eR, eG, n=1):
    R, G, B = tsplit(RGB)
    bR_o, bG_o, bB_o = tsplit(bRGB_o)
    xi, eta, zeta = tsplit(xez)
    bL_or = np.asarray(bL_or)
    eR = np.asarray(eR)
    eG = np.asarray(eG)

    Q = (2 / 3) * bR_o * eR * np.log10((R + n) / (20 * xi + n))
    Q += (1 / 3) * bG_o * eG * np.log10((G + n) / (20 * eta + n))
    Q *= 41.69 / bL_or

    return Q


def tritanopic_response_vectorise(RGB, bRGB_o, xez, n=1):
    R, G, B = tsplit(RGB)
    bR_o, bG_o, bB_o = tsplit(bRGB_o)
    xi, eta, zeta = tsplit(xez)

    t = (1 / 1) * bR_o * np.log10((R + n) / (20 * xi + n))
    t += - (12 / 11) * bG_o * np.log10((G + n) / (20 * eta + n))
    t += (1 / 11) * bB_o * np.log10((B + n) / (20 * zeta + n))

    return t


def protanopic_response_vectorise(RGB, bRGB_o, xez, n=1):
    R, G, B = tsplit(RGB)
    bR_o, bG_o, bB_o = tsplit(bRGB_o)
    xi, eta, zeta = tsplit(xez)

    p = (1 / 9) * bR_o * np.log10((R + n) / (20 * xi + n))
    p += (1 / 9) * bG_o * np.log10((G + n) / (20 * eta + n))
    p += - (2 / 9) * bB_o * np.log10((B + n) / (20 * zeta + n))

    return p


def brightness_correlate_vectorise(bRGB_o, bL_or, Q):
    bR_o, bG_o, bB_o = tsplit(bRGB_o)
    bL_or = np.asarray(bL_or)
    Q = np.asarray(Q)

    B_r = (50 / bL_or) * ((2 / 3) * bR_o + (1 / 3) * bG_o) + Q

    return B_r


def ideal_white_brightness_correlate_vectorise(bRGB_o, xez, bL_or, n=1):
    bR_o, bG_o, bB_o = tsplit(bRGB_o)
    xi, eta, zeta = tsplit(xez)
    bL_or = np.asarray(bL_or)

    B_rw = (2 / 3) * bR_o * 1.758 * np.log10((100 * xi + n) / (20 * xi + n))
    B_rw += (1 / 3) * bG_o * 1.758 * np.log10((100 * eta + n) / (20 * eta + n))
    B_rw *= 41.69 / bL_or
    B_rw += (50 / bL_or) * (2 / 3) * bR_o
    B_rw += (50 / bL_or) * (1 / 3) * bG_o

    return B_rw


def achromatic_lightness_correlate_vectorise(Q):
    B_r = np.asarray(Q)

    return Q + 50


def normalised_achromatic_lightness_correlate_vectorise(B_r, B_rw):
    B_r = np.asarray(B_r)
    B_rw = np.asarray(B_rw)

    return 100 * (B_r / B_rw)


def hue_angle_vectorise(p, t):
    p = np.asarray(p)
    t = np.asarray(t)

    h_L = np.degrees(np.arctan2(p, t)) % 360

    return h_L


def chromatic_strength_function_vectorise(theta):
    h = np.asarray(E_s)

    E_s = 0.9394
    E_s += - 0.2478 * np.sin(1 * theta)
    E_s += - 0.0743 * np.sin(2 * theta)
    E_s += + 0.0666 * np.sin(3 * theta)
    E_s += - 0.0186 * np.sin(4 * theta)
    E_s += - 0.0055 * np.cos(1 * theta)
    E_s += - 0.0521 * np.cos(2 * theta)
    E_s += - 0.0573 * np.cos(3 * theta)
    E_s += - 0.0061 * np.cos(4 * theta)

    return E_s


def saturation_components_vectorise(h, bL_or, t, p):
    h = np.asarray(h)
    bL_or = np.asarray(bL_or)
    t = np.asarray(t)
    p = np.asarray(p)

    E_s = chromatic_strength_function(np.radians(h))
    S_RG = (488.93 / bL_or) * E_s * t
    S_YB = (488.93 / bL_or) * E_s * p

    return S_RG, S_YB


def saturation_correlate_vectorise(S_RG, S_YB):
    S_RG = np.asarray(S_RG)
    S_YB = np.asarray(S_YB)

    S = np.sqrt((S_RG ** 2) + (S_YB ** 2))

    return S


def chroma_components_vectorise(Lstar_P, S_RG, S_YB):
    Lstar_P = np.asarray(Lstar_P)
    S_RG = np.asarray(S_RG)
    S_YB = np.asarray(S_YB)

    C_RG = ((Lstar_P / 50) ** 0.7) * S_RG
    C_YB = ((Lstar_P / 50) ** 0.7) * S_YB

    return C_RG, C_YB


def chroma_correlate_vectorise(Lstar_P, S):
    Lstar_P = np.asarray(Lstar_P)
    S = np.asarray(S)

    C = ((Lstar_P / 50) ** 0.7) * S

    return C


def colourfulness_components_vectorise(C_RG, C_YB, B_rw):
    C_RG = np.asarray(C_RG)
    C_YB = np.asarray(C_YB)
    B_rw = np.asarray(B_rw)

    M_RG = C_RG * B_rw / 100
    M_YB = C_YB * B_rw / 100

    return M_RG, M_YB


def colourfulness_correlate_vectorise(C, B_rw):
    C = np.asarray(C)
    B_rw = np.asarray(B_rw)

    M = C * B_rw / 100
    return M


def XYZ_to_Nayatani95_analysis():
    message_box('XYZ_to_Nayatani95')

    XYZ = np.array([19.01, 20, 21.78])
    XYZ_n = np.array([95.05, 100, 108.88])
    Y_o = 20.0
    E_o = 5000.0
    E_or = 1000.0

    print('Reference:')
    print(XYZ_to_Nayatani95(XYZ, XYZ_n, Y_o, E_o, E_or))

    print('\n')

    print('1d array input:')
    print(XYZ_to_Nayatani95_vectorise(XYZ, XYZ_n, Y_o, E_o, E_or))

    print('\n')

    print('1.5d array input:')
    XYZ = np.tile(XYZ, (6, 1))
    print(XYZ_to_Nayatani95_vectorise(XYZ, XYZ_n, Y_o, E_o, E_or))

    print('\n')

    print('2d array input:')
    XYZ_n = np.tile(XYZ_n, (6, 1))
    Y_o = np.tile(Y_o, 6)
    E_o = np.tile(E_o, 6)
    E_or = np.tile(E_or, 6)
    print(XYZ_to_Nayatani95_vectorise(XYZ, XYZ_n, Y_o, E_o, E_or))

    print('\n')

    print('3d array input:')
    XYZ = np.reshape(XYZ, (2, 3, 3))
    XYZ_n = np.reshape(XYZ_n, (2, 3, 3))
    Y_o = np.reshape(Y_o, (2, 3))
    E_o = np.reshape(E_o, (2, 3))
    E_or = np.reshape(E_or, (2, 3))
    print(XYZ_to_Nayatani95_vectorise(XYZ, XYZ_n, Y_o, E_o, E_or))

    print('\n')


# XYZ_to_Nayatani95_analysis()


def XYZ_to_Nayatani95_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    XYZ_n = np.array([95.05, 100, 108.88])
    Y_o = 20.0
    E_o = 5000.0
    E_or = 1000.0

    times = timeit.Timer(
        functools.partial(
            XYZ_to_Nayatani95_2d,
            DATA_VGA1, XYZ_n, Y_o, E_o, E_or)).repeat(
        repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            XYZ_to_Nayatani95_vectorise,
            DATA_VGA1, XYZ_n, Y_o, E_o, E_or)).repeat(
        repeat_b, number_b)

    b = min(times) / number_b

    print('XYZ_to_Nayatani95\t{0}\t{1}\t{2}'.format(
        len(DATA_VGA1), a, b))


# XYZ_to_Nayatani95_profile()

# #############################################################################
# #############################################################################
# ## colour.appearance.rlab
# #############################################################################
# #############################################################################
from colour.appearance.rlab import *


def XYZ_to_RLAB_2d(XYZ, XYZ_n, Y_n, sigma, D):
    for i in range(len(XYZ)):
        XYZ_to_RLAB(XYZ[i], XYZ_n, Y_n, sigma, D)


def XYZ_to_RLAB_vectorise(XYZ,
                          XYZ_n,
                          Y_n,
                          sigma=RLAB_VIEWING_CONDITIONS.get('Average'),
                          D=RLAB_D_FACTOR.get('Hard Copy Images')):
    Y_n = np.asarray(Y_n)
    D = np.asarray(D)
    sigma = np.asarray(sigma)

    # Converting to cone responses.
    LMS_n = XYZ_to_rgb_vectorise(XYZ_n)

    # Computing the :math:`A` matrix.
    LMS_l_E = (3 * LMS_n) / (LMS_n[0] + LMS_n[1] + LMS_n[2])
    LMS_p_L = ((1 + (Y_n[..., np.newaxis] ** (1 / 3)) + LMS_l_E) /
               (1 + (Y_n[..., np.newaxis] ** (1 / 3)) + (1 / LMS_l_E)))
    LMS_a_L = (LMS_p_L + D[..., np.newaxis] * (1 - LMS_p_L)) / LMS_n

    aR = row_as_diagonal(LMS_a_L)
    # XYZ_ref = np.dot(np.dot(np.dot(R_MATRIX, aR), XYZ_TO_HPE_MATRIX), XYZ)
    XYZ_ref = np.einsum('...ij,...j->...i',
                        np.einsum('...ij,...jk->...ik',
                                  np.einsum('...ij,...jk->...ik', R_MATRIX,
                                            aR),
                                  XYZ_TO_HPE_MATRIX),
                        XYZ)

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
    CR = np.sqrt((aR ** 2) + (bR ** 2))

    # -------------------------------------------------------------------------
    # Computing the correlate of *saturation* :math:`s^R`.
    # -------------------------------------------------------------------------
    sR = CR / LR

    return RLAB_Specification(LR, CR, hR, sR, None, aR, bR)


def XYZ_to_RLAB_analysis():
    message_box('XYZ_to_RLAB')

    XYZ = np.array([19.01, 20, 21.78])
    XYZ_n = np.array([109.85, 100, 35.58])
    Y_n = 31.83
    sigma = RLAB_VIEWING_CONDITIONS['Average']
    D = RLAB_D_FACTOR['Hard Copy Images']

    print('Reference:')
    print(XYZ_to_RLAB(XYZ, XYZ_n, Y_n, sigma, D))

    print('\n')

    print('1d array input:')
    print(XYZ_to_RLAB_vectorise(XYZ, XYZ_n, Y_n, sigma, D))

    print('\n')

    print('2d array input:')
    XYZ = np.tile(XYZ, (6, 1))
    print(XYZ_to_RLAB_vectorise(XYZ, XYZ_n, Y_n, sigma, D))

    print('\n')

    print('3d array input:')
    XYZ = np.reshape(XYZ, (2, 3, 3))
    print(XYZ_to_RLAB_vectorise(XYZ, XYZ_n, Y_n, sigma, D))

    print('\n')


# XYZ_to_RLAB_analysis()


def XYZ_to_RLAB_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    XYZ_n = np.array([109.85, 100, 35.58])
    Y_n = 31.83
    sigma = RLAB_VIEWING_CONDITIONS['Average']
    D = RLAB_D_FACTOR['Hard Copy Images']

    times = timeit.Timer(
        functools.partial(
            XYZ_to_RLAB_2d,
            DATA_VGA1, XYZ_n, Y_n, sigma, D)).repeat(
        repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            XYZ_to_RLAB_vectorise,
            DATA_VGA1, XYZ_n, Y_n, sigma, D)).repeat(
        repeat_b, number_b)

    b = min(times) / number_b

    print('XYZ_to_RLAB\t{0}\t{1}\t{2}'.format(
        len(DATA_VGA1), a, b))


# XYZ_to_RLAB_profile()

# #############################################################################
# #############################################################################
# ## colour.colorimetry.lightness
# #############################################################################
# #############################################################################

# #############################################################################
# # ### colour.lightness_Glasser1958
# #############################################################################
from colour.colorimetry.lightness import *

D1 = np.linspace(0, 100, 1000000)


def lightness_Glasser1958_2d(Y):
    for Y_ in Y:
        lightness_Glasser1958(Y_)


def lightness_Glasser1958_vectorise(Y, **kwargs):
    Y = np.asarray(Y)

    L = 25.29 * (Y ** (1 / 3)) - 18.38

    return L


def lightness_Glasser1958_analysis():
    message_box('lightness_Glasser1958')

    print('Reference:')
    print(lightness_Glasser1958(10.08))

    print('\n')

    print('Numeric input:')
    print(lightness_Glasser1958_vectorise(10.08))

    print('\n')

    print('0d array input:')
    print(lightness_Glasser1958_vectorise(np.array(10.08)))

    print('\n')

    print('1d array input:')
    Y = [10.08] * 6
    print(lightness_Glasser1958_vectorise(Y))

    print('\n')

    print('2d array input:')
    Y = np.reshape(np.array(Y), (2, 3))
    print(lightness_Glasser1958_vectorise(Y))

    print('\n')

    print('3d array input:')
    Y = np.reshape(np.array(Y), (2, 3, 1))
    print(lightness_Glasser1958_vectorise(Y))

    print('\n')


# lightness_Glasser1958_analysis()


def lightness_Glasser1958_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            lightness_Glasser1958_2d,
            D1)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            lightness_Glasser1958_vectorise,
            D1)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('lightness_Glasser1958\t{0}\t{1}\t{2}'.format(
        len(D1), a, b))


# lightness_Glasser1958_profile()

# #############################################################################
# # ### colour.lightness_Wyszecki1963
# #############################################################################
def lightness_Wyszecki1963_2d(Y):
    for Y_ in Y:
        lightness_Wyszecki1963(Y_)


def lightness_Wyszecki1963_vectorise(Y, **kwargs):
    Y = np.asarray(Y)

    if np.any(Y < 1) or np.any(Y > 98):
        warning(('"W*" Lightness computation is only applicable for '
                 '1% < "Y" < 98%, unpredictable results may occur!'))

    W = 25 * (Y ** (1 / 3)) - 17

    return W


def lightness_Wyszecki1963_analysis():
    message_box('lightness_Wyszecki1963')

    print('Reference:')
    print(lightness_Wyszecki1963(10.08))

    print('\n')

    print('Numeric input:')
    print(lightness_Wyszecki1963_vectorise(10.08))

    print('\n')

    print('0d array input:')
    print(lightness_Wyszecki1963_vectorise(np.array(10.08)))

    print('\n')

    print('1d array input:')
    Y = [10.08] * 6
    print(lightness_Wyszecki1963_vectorise(Y))

    print('\n')

    print('2d array input:')
    Y = np.reshape(np.array(Y), (2, 3))
    print(lightness_Wyszecki1963_vectorise(Y))

    print('\n')

    print('3d array input:')
    Y = np.reshape(np.array(Y), (2, 3, 1))
    print(lightness_Wyszecki1963_vectorise(Y))

    print('\n')


# lightness_Wyszecki1963_analysis()


def lightness_Wyszecki1963_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            lightness_Wyszecki1963_2d,
            D1)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            lightness_Wyszecki1963_vectorise,
            D1)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('lightness_Wyszecki1963\t{0}\t{1}\t{2}'.format(
        len(D1), a, b))


# lightness_Wyszecki1963_profile()

# #############################################################################
# # ### colour.lightness_1976
# #############################################################################
def lightness_1976_2d(Y):
    Lstar = []
    for Y_ in Y:
        Lstar.append(lightness_1976(Y_))
    return Lstar


from colour.constants import CIE_E, CIE_K


def lightness_1976_vectorise(Y, Y_n=100):
    Y = np.asarray(Y)
    Y_n = np.asarray(Y_n)

    Lstar = Y / Y_n

    Lstar = np.where(Lstar <= CIE_E,
                     CIE_K * Lstar,
                     116 * Lstar ** (1 / 3) - 16)

    return Lstar


def lightness_1976_analysis():
    message_box('lightness_1976')

    print('Reference:')
    print(lightness_1976(10.08, 100))

    print('\n')

    print('Numeric input:')
    print(lightness_1976_vectorise(10.08, 100))

    print('\n')

    print('0d array input:')
    print(lightness_1976_vectorise(np.array(10.08), np.array(100)))

    print('\n')

    print('1d array input:')
    Y = [10.08] * 6
    print(lightness_1976_vectorise(Y))

    print('\n')

    print('2d array input:')
    Y = np.reshape(Y, (2, 3))
    print(lightness_1976_vectorise(Y))

    print('\n')

    print('3d array input:')
    Y = np.reshape(Y, (2, 3, 1))
    print(lightness_1976_vectorise(Y))

    print('\n')

    Y = np.linspace(0, 100, 10000)
    np.testing.assert_almost_equal(lightness_1976_2d(Y),
                                   lightness_1976_vectorise(Y))


# lightness_1976_analysis()


def lightness_1976_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            lightness_1976_2d,
            D1)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            lightness_1976_vectorise,
            D1)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('lightness_1976\t{0}\t{1}\t{2}'.format(
        len(D1), a, b))


# lightness_1976_profile()

# #############################################################################
# #############################################################################
# ## colour.colorimetry.luminance
# #############################################################################
# #############################################################################

# #############################################################################
# # ### colour.luminance_Newhall1943
# #############################################################################
from colour.colorimetry.luminance import *

L = np.linspace(0, 100, 1000000)


def luminance_Newhall1943_2d(L):
    for L_ in L:
        luminance_Newhall1943(L_)


def luminance_Newhall1943_vectorise(V, **kwargs):
    V = np.asarray(V)

    R_Y = (1.2219 * V - 0.23111 * (V * V) + 0.23951 * (V ** 3) - 0.021009 *
           (V ** 4) + 0.0008404 * (V ** 5))

    return R_Y


def luminance_Newhall1943_analysis():
    message_box('luminance_Newhall1943')

    print('Reference:')
    print(luminance_Newhall1943(3.74629715382))

    print('\n')

    print('Numeric input:')
    print(luminance_Newhall1943_vectorise(3.74629715382))

    print('\n')

    print('0d array input:')
    print(luminance_Newhall1943_vectorise(np.array(3.74629715382)))

    print('\n')

    print('1d array input:')
    V = [3.74629715382] * 6
    print(luminance_Newhall1943_vectorise(V))

    print('\n')

    print('2d array input:')
    V = np.reshape(np.array(V), (2, 3))
    print(luminance_Newhall1943_vectorise(V))

    print('\n')

    print('3d array input:')
    V = np.reshape(np.array(V), (2, 3, 1))
    print(luminance_Newhall1943_vectorise(V))

    print('\n')


# luminance_Newhall1943_analysis()


def luminance_Newhall1943_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            luminance_Newhall1943_2d,
            D1)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            luminance_Newhall1943_vectorise,
            D1)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('luminance_Newhall1943\t{0}\t{1}\t{2}'.format(
        len(D1), a, b))


# luminance_Newhall1943_profile()

# #############################################################################
# # ### colour.luminance_ASTMD153508
# #############################################################################


def luminance_ASTMD153508_2d(L):
    for L_ in L:
        luminance_ASTMD153508(L_)


def luminance_ASTMD153508_vectorise(V, **kwargs):
    V = np.asarray(V)

    Y = (1.1914 * V - 0.22533 * (V ** 2) + 0.23352 * (V ** 3) - 0.020484 *
         (V ** 4) + 0.00081939 * (V ** 5))

    return Y


def luminance_ASTMD153508_analysis():
    message_box('luminance_ASTMD153508')

    print('Reference:')
    print(luminance_ASTMD153508(3.74629715382))

    print('\n')

    print('Numeric input:')
    print(luminance_ASTMD153508_vectorise(3.74629715382))

    print('\n')

    print('0d array input:')
    print(luminance_ASTMD153508_vectorise(np.array(3.74629715382)))

    print('\n')

    print('1d array input:')
    V = [3.74629715382] * 6
    print(luminance_ASTMD153508_vectorise(V))

    print('\n')

    print('2d array input:')
    V = np.reshape(np.array(V), (2, 3))
    print(luminance_ASTMD153508_vectorise(V))

    print('\n')

    print('3d array input:')
    V = np.reshape(np.array(V), (2, 3, 1))
    print(luminance_ASTMD153508_vectorise(V))

    print('\n')


# luminance_ASTMD153508_analysis()


def luminance_ASTMD153508_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            luminance_ASTMD153508_2d,
            D1)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            luminance_ASTMD153508_vectorise,
            D1)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('luminance_ASTMD153508\t{0}\t{1}\t{2}'.format(
        len(D1), a, b))


# luminance_ASTMD153508_profile()


# #############################################################################
# # ### colour.luminance_1976
# #############################################################################

def luminance_1976_2d(L):
    Y = []
    for L_ in L:
        Y.append(luminance_1976(L_))
    return Y


def luminance_1976_vectorise(Lstar, Y_n=100):
    Lstar = np.asarray(Lstar)
    Y_n = np.asarray(Y_n)

    Y = np.where(Lstar > CIE_K * CIE_E,
                 Y_n * ((Lstar + 16) / 116) ** 3,
                 Y_n * (Lstar / CIE_K))

    return Y


def luminance_1976_analysis():
    message_box('luminance_1976')

    print('Reference:')
    print(luminance_1976(37.9856290977))

    print('\n')

    print('Numeric input:')
    print(luminance_1976_vectorise(37.9856290977))

    print('\n')

    print('0d array input:')
    print(luminance_1976_vectorise(np.array(37.9856290977)))

    print('\n')

    print('1d array input:')
    Lstar = [37.9856290977] * 6
    print(luminance_1976_vectorise(Lstar))

    print('\n')

    print('2d array input:')
    Lstar = np.reshape(np.array(Lstar), (2, 3))
    print(luminance_1976_vectorise(Lstar))

    print('\n')

    print('3d array input:')
    Lstar = np.reshape(np.array(Lstar), (2, 3, 1))
    print(luminance_1976_vectorise(Lstar))

    print('\n')

    Lstar = np.linspace(0, 100, 10000)
    np.testing.assert_almost_equal(luminance_1976_2d(Lstar),
                                   luminance_1976_vectorise(Lstar))


# luminance_1976_analysis()


def luminance_1976_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            luminance_1976_2d,
            D1)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            luminance_1976_vectorise,
            D1)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('luminance_1976\t{0}\t{1}\t{2}'.format(
        len(D1), a, b))


# luminance_1976_profile()


# #############################################################################
# #############################################################################
# ## colour.colorimetry.spectrum
# #############################################################################
# #############################################################################

# #############################################################################
# # ### colour.SpectralShape
# #############################################################################
from colour.colorimetry.spectrum import *


def SpectralShape__contains__(self, wavelength):
    return np.all(np.in1d(np.ravel(wavelength), self.range()))


SpectralShape.__contains__ = SpectralShape__contains__


def SpectralShape__contains__analysis():
    message_box('SpectralShape.__contains__')

    print(380 in SpectralShape(360, 830, 1))

    print('\n')

    print((380, 480) in SpectralShape(360, 830, 1))

    print('\n')

    print((380, 480.5) in SpectralShape(360, 830, 1))

    print('\n')


# SpectralShape__contains__analysis()

# #############################################################################
# # ### colour.SpectralPowerDistribution
# #############################################################################

def SpectralPowerDistribution__getitem__(self, wavelength):
    if type(wavelength) is slice:
        return self.values[wavelength]
    else:
        wavelength = np.asarray(wavelength)

        value = [self.data.__getitem__(x) for x in np.ravel(wavelength)]
        value = np.reshape(value, wavelength.shape)

        return value


SpectralPowerDistribution.__getitem__ = SpectralPowerDistribution__getitem__


def SpectralPowerDistribution__getitem__analysis():
    message_box('SpectralPowerDistribution.__getitem___')

    data = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
    spd = SpectralPowerDistribution('Spd', data)

    print(spd[510])

    print('\n')

    print(spd[np.array(510)])

    print('\n')

    print(spd[np.array([510])])

    print('\n')

    print(spd[[510, 520]])

    print('\n')

    print(spd[np.array([510, 520])])

    print('\n')

    print(spd[np.array([[510], [520]])])

    print('\n')

    print(spd[np.array([[[510, 520]], [[530, 540]]])])

    print('\n')

    print(spd[0:-1])

    print('\n')


# SpectralPowerDistribution__getitem__analysis()


def SpectralPowerDistribution__setitem__(self, wavelength, value):
    # TODO: Mention implicit resize.
    if is_numeric(wavelength) or is_iterable(wavelength):
        wavelengths = np.ravel(wavelength)
    elif type(wavelength) is slice:
        wavelengths = self.wavelengths[wavelength]
    else:
        raise NotImplementedError(
            '"{0}" type is not supported for indexing!'.format(
                type(wavelength)))

    values = np.resize(value, wavelengths.shape)
    for i in range(len(wavelengths)):
        # self.data ===> self.__data
        self.data.__setitem__(wavelengths[i], values[i])


SpectralPowerDistribution.__setitem__ = SpectralPowerDistribution__setitem__


def SpectralPowerDistribution__setitem__analysis():
    message_box('SpectralPowerDistribution.__setitem__')

    spd = SpectralPowerDistribution('Spd', {})

    spd[510] = 49.67
    pprint(list(spd.items))

    print('\n')

    spd[[520, 530]] = (69.59, 81.73)
    pprint(list(spd.items))

    print('\n')

    spd[[540, 550]] = 88.19
    pprint(list(spd.items))

    print('\n')

    spd[:] = 49.67
    pprint(list(spd.items))

    print('\n')

    spd[0:3] = 69.59
    pprint(list(spd.items))

    print('\n')


# SpectralPowerDistribution__setitem__analysis()

def SpectralPowerDistribution_get(self, wavelength, default=None):
    wavelength = np.asarray(wavelength)

    value = [self.data.get(x, default) for x in np.ravel(wavelength)]
    value = np.reshape(value, wavelength.shape)

    return value


SpectralPowerDistribution.get = SpectralPowerDistribution_get


def SpectralPowerDistribution_get_analysis():
    message_box('SpectralPowerDistribution.get')

    data = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
    spd = SpectralPowerDistribution('Spd', data)

    print(spd.get(510))

    print('\n')

    print(spd.get(np.array(510)))

    print('\n')

    print(spd.get(np.array([510])))

    print('\n')

    print(spd.get([510, 520]))

    print('\n')

    print(spd.get(np.array([510, 520])))

    print('\n')

    print(spd.get(np.array([[510], [520]])))

    print('\n')

    print(spd.get([510, 520, 521]))

    print('\n')


# SpectralPowerDistribution_get_analysis()


def SpectralPowerDistribution__contains__(self, wavelength):
    return np.all(np.in1d(np.ravel(wavelength), self.wavelengths))


SpectralPowerDistribution.__contains__ = SpectralPowerDistribution__contains__


def SpectralPowerDistribution__contains__analysis():
    message_box('SpectralPowerDistribution.__contains__')

    data = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
    spd = SpectralPowerDistribution('Spd', data)

    print(510 in spd)

    print('\n')

    print([510, 520] in spd)

    print('\n')

    print([510, 520, 521] in spd)

    print('\n')


# SpectralPowerDistribution__contains__analysis()

# #############################################################################
# # ### colour.TriSpectralPowerDistribution
# #############################################################################


def TriSpectralPowerDistribution__getitem__(self, wavelength):
    value = tstack((np.asarray(self.x[wavelength]),
                    np.asarray(self.y[wavelength]),
                    np.asarray(self.z[wavelength])))

    return value


TriSpectralPowerDistribution.__getitem__ = TriSpectralPowerDistribution__getitem__


def TriSpectralPowerDistribution__getitem__analysis():
    message_box('TriSpectralPowerDistribution.__getitem__')

    x_bar = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
    y_bar = {510: 90.56, 520: 87.34, 530: 45.76, 540: 23.45}
    z_bar = {510: 12.43, 520: 23.15, 530: 67.98, 540: 90.28}
    data = {'x_bar': x_bar, 'y_bar': y_bar, 'z_bar': z_bar}
    mpg = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
    tri_spd = TriSpectralPowerDistribution('Tri Spd', data, mpg)

    print(tri_spd[510])

    print('\n')

    print(tri_spd[np.array(510)])

    print('\n')

    print(tri_spd[[510, 520]])

    print('\n')

    print(tri_spd[np.array([[510, 520]])])

    print('\n')

    print(tri_spd[np.array([[510], [520]])])

    print('\n')

    print(tri_spd[0:-1])

    print('\n')


# TriSpectralPowerDistribution__getitem__analysis()


def TriSpectralPowerDistribution__setitem__(self, wavelength, value):
    # TODO: Mention implicit resize.
    if is_numeric(wavelength) or is_iterable(wavelength):
        wavelengths = np.ravel(wavelength)
    elif type(wavelength) is slice:
        wavelengths = self.wavelengths[wavelength]
    else:
        raise NotImplementedError(
            '"{0}" type is not supported for indexing!'.format(
                type(wavelength)))

    value = np.resize(value, (wavelengths.shape[0], 3))

    self.x.__setitem__(wavelengths, value[..., 0])
    self.y.__setitem__(wavelengths, value[..., 1])
    self.z.__setitem__(wavelengths, value[..., 2])


TriSpectralPowerDistribution.__setitem__ = TriSpectralPowerDistribution__setitem__


def TriSpectralPowerDistribution__setitem__analysis():
    message_box('TriSpectralPowerDistribution.__setitem__')

    data = {'x_bar': {}, 'y_bar': {}, 'z_bar': {}}
    mpg = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
    tri_spd = TriSpectralPowerDistribution('Tri Spd', data, mpg)

    tri_spd[510] = 49.67
    pprint(list(tri_spd.items))

    print('\n')

    tri_spd[[520, 530]] = (69.59, 81.73)
    pprint(list(tri_spd.items))

    print('\n')

    tri_spd[[540, 550]] = ((49.67, 69.59, 81.73), (81.73, 69.59, 49.67))
    pprint(list(tri_spd.items))

    print('\n')

    tri_spd[:] = 49.67
    pprint(list(tri_spd.items))

    print('\n')

    tri_spd[0:3] = ((81.73, 69.59, 49.67), (49.67, 69.59, 81.73))
    pprint(list(tri_spd.items))

    print('\n')


# TriSpectralPowerDistribution__setitem__analysis()


def TriSpectralPowerDistribution_get(self, wavelength, default=None):
    wavelength = np.asarray(wavelength)

    value = np.asarray([(self.x.get(x, default),
                         self.y.get(x, default),
                         self.z.get(x, default))
                        for x in np.ravel(wavelength)])

    value = np.reshape(value, wavelength.shape + (3,))

    return value


TriSpectralPowerDistribution.get = TriSpectralPowerDistribution_get


def TriSpectralPowerDistribution_get_analysis():
    message_box('TriSpectralPowerDistribution.get')

    x_bar = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
    y_bar = {510: 90.56, 520: 87.34, 530: 45.76, 540: 23.45}
    z_bar = {510: 12.43, 520: 23.15, 530: 67.98, 540: 90.28}
    data = {'x_bar': x_bar, 'y_bar': y_bar, 'z_bar': z_bar}
    mpg = {'x': 'x_bar', 'y': 'y_bar', 'z': 'z_bar'}
    tri_spd = TriSpectralPowerDistribution('Tri Spd', data, mpg)

    print(tri_spd.get(510))

    print('\n')

    print(tri_spd.get(np.array(510)))

    print('\n')

    print(tri_spd.get(np.array([510])))

    print('\n')

    print(tri_spd.get([510, 520]))

    print('\n')

    print(tri_spd.get(np.array([510, 520])))

    print('\n')

    print(tri_spd.get(np.array([[510], [520]])))

    print('\n')

    print(tri_spd.get([510, 520, 521]))

    print('\n')


# TriSpectralPowerDistribution_get_analysis()

# #############################################################################
# #############################################################################
# ## colour.colorimetry.transformations
# #############################################################################
# #############################################################################

# #############################################################################
# # ### colour.RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs
# #############################################################################
from colour import PHOTOPIC_LEFS, RGB_CMFS
from colour.colorimetry.transformations import *


def RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs_vectorise(wavelength):
    cmfs = RGB_CMFS.get('Wright & Guild 1931 2 Degree RGB CMFs')

    try:
        rgb_bar = cmfs[wavelength]
    except KeyError as error:
        raise KeyError(('"{0} nm" wavelength not available in "{1}" colour '
                        'matching functions with "{2}" shape!').format(
            error.args[0], cmfs.name, cmfs.shape))

    rgb = rgb_bar / np.sum(rgb_bar)

    M1 = np.array([[0.49000, 0.31000, 0.20000],
                   [0.17697, 0.81240, 0.01063],
                   [0.00000, 0.01000, 0.99000]])

    M2 = np.array([[0.66697, 1.13240, 1.20063],
                   [0.66697, 1.13240, 1.20063],
                   [0.66697, 1.13240, 1.20063]])

    xyz = np.einsum('...ij,...j->...i', M1, rgb)
    xyz /= np.einsum('...ij,...j->...i', M2, rgb)

    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]

    V = PHOTOPIC_LEFS.get('CIE 1924 Photopic Standard Observer').clone()
    V.align(cmfs.shape)
    L = V[wavelength]

    x_bar = x / y * L
    y_bar = L
    z_bar = z / y * L

    xyz_bar = tstack((np.asarray(x_bar),
                      np.asarray(y_bar),
                      np.asarray(z_bar)))

    return xyz_bar


def RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs_analysis():
    message_box('RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs')

    print('Reference:')
    print(RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(700))

    print('\n')

    print('Numeric input:')
    print(RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs_vectorise(700))

    print('\n')

    print('0d array input:')
    print(RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs_vectorise(np.array(700)))

    print('\n')

    print('1d array input:')
    wl = [700] * 6
    print(RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs_vectorise(wl))

    print('\n')

    print('2d array input:')
    wl = np.reshape(np.array(wl), (2, 3))
    print(RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs_vectorise(wl))

    print('\n')

    print('3d array input:')
    wl = np.reshape(np.array(wl), (2, 3, 1))
    print(RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs_vectorise(wl))

    print('\n')


# RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs_analysis()

# #############################################################################
# # ### colour.RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs
# #############################################################################


def RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs_vectorise(wavelength):
    cmfs = RGB_CMFS.get('Stiles & Burch 1959 10 Degree RGB CMFs')

    try:
        rgb_bar = cmfs[wavelength]
    except KeyError as error:
        raise KeyError(('"{0} nm" wavelength not available in "{1}" colour '
                        'matching functions with "{2}" shape!').format(
            error.args[0], cmfs.name, cmfs.shape))

    M = np.array([[0.341080, 0.189145, 0.387529],
                  [0.139058, 0.837460, 0.073316],
                  [0.000000, 0.039553, 2.026200]])

    xyz_bar = np.einsum('...ij,...j->...i', M, rgb_bar)

    return xyz_bar


def RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs_analysis():
    message_box('RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs')

    print('Reference:')
    print(RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(700))

    print('\n')

    print('Numeric input:')
    print(RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs_vectorise(700))

    print('\n')

    print('0d array input:')
    print(RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs_vectorise(np.array(700)))

    print('\n')

    print('1d array input:')
    wl = [700] * 6
    print(RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs_vectorise(wl))

    print('\n')

    print('2d array input:')
    wl = np.reshape(np.array(wl), (2, 3))
    print(RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs_vectorise(wl))

    print('\n')

    print('3d array input:')
    wl = np.reshape(np.array(wl), (2, 3, 1))
    print(RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs_vectorise(wl))

    print('\n')


# RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs_analysis()

# #############################################################################
# # ### colour.RGB_10_degree_cmfs_to_LMS_10_degree_cmfs
# #############################################################################

def RGB_10_degree_cmfs_to_LMS_10_degree_cmfs_vectorise(wavelength):
    cmfs = RGB_CMFS.get('Stiles & Burch 1959 10 Degree RGB CMFs')

    try:
        rgb_bar = cmfs[wavelength]
    except KeyError as error:
        raise KeyError(('"{0} nm" wavelength not available in "{1}" colour '
                        'matching functions with "{2}" shape!').format(
            error.args[0], cmfs.name, cmfs.shape))

    M = np.array([[0.1923252690, 0.749548882, 0.0675726702],
                  [0.0192290085, 0.940908496, 0.113830196],
                  [0.0000000000, 0.0105107859, 0.991427669]])

    lms_bar = np.einsum('...ij,...j->...i', M, rgb_bar)
    lms_bar[..., -1][np.asarray(np.asarray(wavelength) > 505)] = 0

    return lms_bar


def RGB_10_degree_cmfs_to_LMS_10_degree_cmfs_analysis():
    message_box('RGB_10_degree_cmfs_to_LMS_10_degree_cmfs')

    print('Reference:')
    print(RGB_10_degree_cmfs_to_LMS_10_degree_cmfs(700))

    print('\n')

    print('Numeric input:')
    print(RGB_10_degree_cmfs_to_LMS_10_degree_cmfs_vectorise(700))

    print('\n')

    print('0d array input:')
    print(RGB_10_degree_cmfs_to_LMS_10_degree_cmfs_vectorise(np.array(700)))

    print('\n')

    print('1d array input:')
    wl = [700] * 6
    print(RGB_10_degree_cmfs_to_LMS_10_degree_cmfs_vectorise(wl))

    print('\n')

    print('2d array input:')
    wl = np.reshape(np.array(wl), (2, 3))
    print(RGB_10_degree_cmfs_to_LMS_10_degree_cmfs_vectorise(wl))

    print('\n')

    print('2d array input:')
    wl = np.reshape(np.array(wl), (2, 3, 1))
    print(RGB_10_degree_cmfs_to_LMS_10_degree_cmfs_vectorise(wl))

    print('\n')


# RGB_10_degree_cmfs_to_LMS_10_degree_cmfs_analysis()

# #############################################################################
# # ### colour.LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs
# #############################################################################
from colour import LMS_CMFS


def LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs_vectorise(wavelength):
    cmfs = LMS_CMFS.get('Stockman & Sharpe 2 Degree Cone Fundamentals')

    try:
        lms_bar = cmfs[wavelength]
    except KeyError as error:
        raise KeyError(('"{0} nm" wavelength not available in "{1}" colour '
                        'matching functions with "{2}" shape!').format(
            error.args[0], cmfs.name, cmfs.shape))

    M = np.array([[1.94735469, -1.41445123, 0.36476327],
                  [0.68990272, 0.34832189, 0.00000000],
                  [0.00000000, 0.00000000, 1.93485343]])

    xyz_bar = np.einsum('...ij,...j->...i', M, lms_bar)

    return xyz_bar


def LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs_analysis():
    message_box('LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs')

    print('Reference:')
    print(LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs(700))

    print('\n')

    print('Numeric input:')
    print(LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs_vectorise(700))

    print('\n')

    print('0d array input:')
    print(LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs_vectorise(np.array(700)))

    print('\n')

    print('1d array input:')
    wl = [700] * 6
    print(LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs_vectorise(wl))

    print('\n')

    print('2d array input:')
    wl = np.reshape(np.array(wl), (2, 3))
    print(LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs_vectorise(wl))

    print('\n')

    print('3d array input:')
    wl = np.reshape(np.array(wl), (2, 3, 1))
    print(LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs_vectorise(wl))

    print('\n')


# LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs_analysis()

# #############################################################################
# # ### colour.LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs
# #############################################################################


def LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs_vectorise(wavelength):
    cmfs = LMS_CMFS.get('Stockman & Sharpe 10 Degree Cone Fundamentals')

    try:
        lms_bar = cmfs[wavelength]
    except KeyError as error:
        raise KeyError(('"{0} nm" wavelength not available in "{1}" colour '
                        'matching functions with "{2}" shape!').format(
            error.args[0], cmfs.name, cmfs.shape))

    M = np.array([[1.93986443, -1.34664359, 0.43044935],
                  [0.69283932, 0.34967567, 0.00000000],
                  [0.00000000, 0.00000000, 2.14687945]])

    xyz_bar = np.einsum('...ij,...j->...i', M, lms_bar)

    return xyz_bar


def LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs_analysis():
    message_box('LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs')

    print('Reference:')
    print(LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs(700))

    print('\n')

    print('Numeric input:')
    print(LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs_vectorise(700))

    print('\n')

    print('0d array input:')
    print(LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs_vectorise(np.array(700)))

    print('\n')

    print('1d array input:')
    wl = [700] * 6
    print(LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs_vectorise(wl))

    print('\n')

    print('2d array input:')
    wl = np.reshape(np.array(wl), (2, 3))
    print(LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs_vectorise(wl))

    print('\n')

    print('3d array input:')
    wl = np.reshape(np.array(wl), (2, 3, 1))
    print(LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs_vectorise(wl))

    print('\n')


# LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs_analysis()

# #############################################################################
# #############################################################################
# ## colour.colorimetry.tristimulus
# #############################################################################
# #############################################################################

# #############################################################################
# # ### colour.wavelength_to_XYZ
# #############################################################################
from colour import (
    STANDARD_OBSERVERS_CMFS,
    SpragueInterpolator,
    SplineInterpolator)
from colour.colorimetry.tristimulus import *

WAVELENGTHS = np.linspace(400, 700, 1000)


def wavelength_to_XYZ_2d(wavelengths):
    for wavelength in wavelengths:
        wavelength_to_XYZ(wavelength)


def wavelength_to_XYZ_vectorise(wavelength,
                                cmfs=STANDARD_OBSERVERS_CMFS.get(
                                    'CIE 1931 2 Degree Standard Observer')):
    cmfs_shape = cmfs.shape
    if (np.min(wavelength) < cmfs_shape.start or
                np.max(wavelength) > cmfs_shape.end):
        raise ValueError(
            '"{0} nm" wavelength is not in "[{1}, {2}]" domain!'.format(
                wavelength, cmfs_shape.start, cmfs_shape.end))

    if wavelength not in cmfs:
        wavelengths, values, = cmfs.wavelengths, cmfs.values
        interpolator = (SpragueInterpolator
                        if cmfs.is_uniform() else
                        SplineInterpolator)

        interpolators = [interpolator(wavelengths, values[:, i])
                         for i in range(values.shape[-1])]

        XYZ = np.dstack([interpolator(np.ravel(wavelength))
                         for interpolator in interpolators])
    else:
        XYZ = cmfs.get(wavelength)

    XYZ = np.reshape(XYZ, np.asarray(wavelength).shape + (3,))

    return XYZ


def wavelength_to_XYZ_analysis():
    message_box('wavelength_to_XYZ')

    print('Reference:')
    print(wavelength_to_XYZ(480))

    print('\n')

    print('Numeric input:')
    print(wavelength_to_XYZ_vectorise(480))

    print('\n')

    print('0d array input:')
    print(wavelength_to_XYZ_vectorise(np.array(480)))

    print('\n')

    print('1d array input:')
    print(wavelength_to_XYZ_vectorise([480] * 6))

    print('\n')

    print('1d array input:')
    print(wavelength_to_XYZ_vectorise(np.array([480] * 5 + [480.5])))

    print('\n')

    print('2d array input:')
    print(wavelength_to_XYZ_vectorise(
        np.array([[480] * 3, [480] * 2 + [480.5]])))

    print('\n')

    print('3d array input:')
    print(wavelength_to_XYZ_vectorise(
        np.array([[[480] * 3], [[480] * 2 + [480.5]]])))

    print('\n')


# wavelength_to_XYZ_analysis()


def wavelength_to_XYZ_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            wavelength_to_XYZ_2d,
            WAVELENGTHS)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            wavelength_to_XYZ_vectorise,
            WAVELENGTHS)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('wavelength_to_XYZ\t{0}\t{1}\t{2}'.format(
        len(WAVELENGTHS), a, b))


# wavelength_to_XYZ_profile()

# #############################################################################
# #############################################################################
# ## colour.colorimetry.blackbody
# #############################################################################
# #############################################################################

# #############################################################################
# # ### colour.planck_law
# #############################################################################
from colour.colorimetry.blackbody import *
from colour.colorimetry.spectrum import *

WAVELENGTHS = np.linspace(1, 15000, 100000) * 1e-9


def planck_law_2d(wavelengths):
    for wavelength in wavelengths:
        planck_law(wavelength, 5500)


@handle_numpy_errors(over='ignore')
def planck_law_vectorise(wavelength, temperature, c1=C1, c2=C2, n=N):
    l = np.asarray(wavelength)
    t = np.asarray(temperature)

    p = (((c1 * n ** -2 * l ** -5) / np.pi) *
         (np.exp(c2 / (n * l * t)) - 1) ** -1)

    return p


def planck_law_analysis():
    message_box('planck_law')

    print('Reference:')
    print(planck_law(500 * 1e-9, 5500))

    print('\n')

    print('Numeric input:')
    print(planck_law_vectorise(500 * 1e-9, 5500))

    print('\n')

    print('0d array input:')
    print(planck_law_vectorise(np.array(500 * 1e-9), 5500))

    print('\n')

    print('1d array input:')
    wl = [500 * 1e-9] * 6
    print(planck_law_vectorise(wl, 5500))

    print('\n')

    print('2d array input:')
    wl = np.reshape(wl, (2, 3))
    print(planck_law_vectorise(wl, 5500))

    print('\n')

    print('3d array input:')
    wl = np.reshape(wl, (2, 3, 1))
    print(planck_law_vectorise(wl, 5500))

    print('\n')


# planck_law_analysis()



def planck_law_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            planck_law_2d,
            WAVELENGTHS)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            planck_law_vectorise,
            WAVELENGTHS, 5500)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('planck_law\t{0}\t{1}\t{2}'.format(
        len(WAVELENGTHS), a, b))


# planck_law_profile()


def blackbody_spd_vectorise(temperature,
                            shape=DEFAULT_SPECTRAL_SHAPE,
                            c1=C1,
                            c2=C2,
                            n=N):
    wavelengths = shape.range()
    return SpectralPowerDistribution(
        name='{0}K Blackbody'.format(temperature),
        data=dict(
            zip(wavelengths,
                planck_law_vectorise(
                    wavelengths * 1e-9, temperature, c1, c2, n))))


def blackbody_spd_analysis():
    message_box('blackbody_spd')

    print(blackbody_spd_vectorise(5000).values)


# blackbody_spd_analysis()


# #############################################################################
# #############################################################################
# ## colour.colorimetry.whiteness
# #############################################################################
# #############################################################################

# #############################################################################
# # ### colour.whiteness_Berger1959
# #############################################################################
from colour.colorimetry.whiteness import *


def whiteness_Berger1959_2d(XYZ):
    XYZ_0 = np.array([94.80966767, 100., 107.30513595])

    for i in range(len(XYZ)):
        whiteness_Berger1959(XYZ[i], XYZ_0)


def whiteness_Berger1959_vectorise(XYZ, XYZ_0):
    X, Y, Z = tsplit(XYZ)
    X_0, Y_0, Z_0 = tsplit(XYZ_0)

    WI = 0.333 * Y + 125 * (Z / Z_0) - 125 * (X / X_0)

    return WI


def whiteness_Berger1959_analysis():
    message_box('whiteness_Berger1959')

    print('Reference:')
    XYZ = np.array([95., 100., 105.])
    XYZ_0 = np.array([94.80966767, 100., 107.30513595])
    print(whiteness_Berger1959(XYZ, XYZ_0))

    print('\n')

    print('1d array input:')
    print(whiteness_Berger1959_vectorise(XYZ, XYZ_0))

    print('\n')

    print('2d array input:')
    XYZ = np.tile(XYZ, (6, 1))
    print(whiteness_Berger1959_vectorise(XYZ, XYZ_0))

    print('\n')

    print('3d array input:')
    XYZ = np.reshape(XYZ, (2, 3, 3))
    print(whiteness_Berger1959_vectorise(XYZ, XYZ_0))

    print('\n')


# whiteness_Berger1959_analysis()


def whiteness_Berger1959_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    XYZ_0 = np.array([94.80966767, 100., 107.30513595])

    times = timeit.Timer(
        functools.partial(
            whiteness_Berger1959_2d,
            DATA_HD1)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            whiteness_Berger1959_vectorise,
            DATA_HD1, XYZ_0)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('whiteness_Berger1959\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# whiteness_Berger1959_profile()

# #############################################################################
# # ### colour.whiteness_Taube1960
# #############################################################################


def whiteness_Taube1960_2d(XYZ):
    XYZ_0 = np.array([94.80966767, 100., 107.30513595])

    for i in range(len(XYZ)):
        whiteness_Taube1960(XYZ[i], XYZ_0)


def whiteness_Taube1960_vectorise(XYZ, XYZ_0):
    X, Y, Z = tsplit(XYZ)
    X_0, Y_0, Z_0 = tsplit(XYZ_0)

    WI = 400 * (Z / Z_0) - 3 * Y

    return WI


def whiteness_Taube1960_analysis():
    message_box('whiteness_Taube1960')

    print('Reference:')
    XYZ = np.array([95., 100., 105.])
    XYZ_0 = np.array([94.80966767, 100., 107.30513595])
    print(whiteness_Taube1960(XYZ, XYZ_0))

    print('\n')

    print('1d array input:')
    print(whiteness_Taube1960_vectorise(XYZ, XYZ_0))

    print('\n')

    print('2d array input:')
    XYZ = np.tile(XYZ, (6, 1))
    print(whiteness_Taube1960_vectorise(XYZ, XYZ_0))

    print('\n')

    print('3d array input:')
    XYZ = np.reshape(XYZ, (2, 3, 3))
    print(whiteness_Taube1960_vectorise(XYZ, XYZ_0))

    print('\n')


# whiteness_Taube1960_analysis()


def whiteness_Taube1960_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    XYZ_0 = np.array([94.80966767, 100., 107.30513595])

    times = timeit.Timer(
        functools.partial(
            whiteness_Taube1960_2d,
            DATA_HD1)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            whiteness_Taube1960_vectorise,
            DATA_HD1, XYZ_0)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('whiteness_Taube1960\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# whiteness_Taube1960_profile()

# #############################################################################
# # ### colour.whiteness_Stensby1968
# #############################################################################


def whiteness_Stensby1968_2d(Lab):
    for i in range(len(Lab)):
        whiteness_Stensby1968(Lab[i])


def whiteness_Stensby1968_vectorise(Lab):
    L, a, b = tsplit(Lab)

    WI = L - 3 * b + 3 * a

    return WI


def whiteness_Stensby1968_analysis():
    message_box('whiteness_Stensby1968')

    print('Reference:')
    Lab = np.array([100., -2.46875131, -16.72486654])
    print(whiteness_Stensby1968(Lab))

    print('\n')

    print('1d array input:')
    print(whiteness_Stensby1968_vectorise(Lab))

    print('\n')

    print('2d array input:')
    Lab = np.tile(Lab, (6, 1))
    print(whiteness_Stensby1968_vectorise(Lab))

    print('\n')

    print('3d array input:')
    Lab = np.reshape(Lab, (2, 3, 3))
    print(whiteness_Stensby1968_vectorise(Lab))

    print('\n')


# whiteness_Stensby1968_analysis()


def whiteness_Stensby1968_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            whiteness_Stensby1968_2d,
            DATA_HD1)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            whiteness_Stensby1968_vectorise,
            DATA_HD1)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('whiteness_Stensby1968\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# whiteness_Stensby1968_profile()

# #############################################################################
# # ### colour.whiteness_ASTM313
# #############################################################################


def whiteness_ASTM313_2d(XYZ):
    for i in range(len(XYZ)):
        whiteness_ASTM313(XYZ[i])


def whiteness_ASTM313_vectorise(XYZ):
    X, Y, Z = tsplit(XYZ)

    WI = 3.388 * Z - 3 * Y

    return WI


def whiteness_ASTM313_analysis():
    message_box('whiteness_ASTM313')

    print('Reference:')
    XYZ = np.array([95., 100., 105.])
    print(whiteness_ASTM313(XYZ))

    print('\n')

    print('1d array input:')
    print(whiteness_ASTM313_vectorise(XYZ))

    print('\n')

    print('2d array input:')
    XYZ = np.tile(XYZ, (6, 1))
    print(whiteness_ASTM313_vectorise(XYZ))

    print('\n')

    print('3d array input:')
    XYZ = np.reshape(XYZ, (2, 3, 3))
    print(whiteness_ASTM313_vectorise(XYZ))

    print('\n')


# whiteness_ASTM313_analysis()


def whiteness_ASTM313_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            whiteness_ASTM313_2d,
            DATA_HD1)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            whiteness_ASTM313_vectorise,
            DATA_HD1)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('whiteness_ASTM313\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# whiteness_ASTM313_profile()

# #############################################################################
# # ### colour.whiteness_Ganz1979
# #############################################################################


def whiteness_Ganz1979_2d(xy):
    Y = 100
    for i in range(len(xy)):
        whiteness_Ganz1979(xy[i], Y)


def whiteness_Ganz1979_vectorise(xy, Y):
    x, y = tsplit(xy)
    Y = np.asarray(Y)

    W = Y - 1868.322 * x - 3695.690 * y + 1809.441
    T = -1001.223 * x + 748.366 * y + 68.261

    WT = tstack((W, T))

    return WT


def whiteness_Ganz1979_analysis():
    message_box('whiteness_Ganz1979')

    print('Reference:')
    xy = (0.3167, 0.3334)
    Y = 100
    print(whiteness_Ganz1979(xy, Y))

    print('\n')

    print('1d array input:')
    print(whiteness_Ganz1979_vectorise(xy, Y))

    print('\n')

    print('2d array input:')
    xy = np.tile(xy, (6, 1))
    print(whiteness_Ganz1979_vectorise(xy, Y))

    print('\n')

    print('3d array input:')
    xy = np.reshape(xy, (2, 3, 2))
    print(whiteness_Ganz1979_vectorise(xy, Y))

    print('\n')


# whiteness_Ganz1979_analysis()


def whiteness_Ganz1979_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    Y = 100

    times = timeit.Timer(
        functools.partial(
            whiteness_Ganz1979_2d,
            DATA_HD1[..., 0:2])).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            whiteness_Ganz1979_vectorise,
            DATA_HD1[..., 0:2], 100)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('whiteness_Ganz1979\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1[..., 0:2]), a, b))


# whiteness_Ganz1979_profile()

# #############################################################################
# # ### colour.whiteness_CIE2004
# #############################################################################


def whiteness_CIE2004_2d(xy):
    Y = 100
    xy_n = (0.3139, 0.3311)

    for i in range(len(xy)):
        whiteness_CIE2004(xy[i], Y, xy_n)


def whiteness_CIE2004_vectorise(xy,
                                Y,
                                xy_n,
                                observer='CIE 1931 2 Degree Standard Observer'):
    x, y = tsplit(xy)
    Y = np.asarray(Y)
    x_n, y_n = tsplit(xy_n)

    W = Y + 800 * (x_n - x) + 1700 * (y_n - y)
    T = (1000 if '1931' in observer else 900) * (x_n - x) - 650 * (y_n - y)

    WT = tstack((W, T))

    return WT


def whiteness_CIE2004_analysis():
    message_box('whiteness_CIE2004')

    print('Reference:')
    xy = (0.3167, 0.3334)
    Y = 100
    xy_n = (0.3139, 0.3311)
    print(whiteness_CIE2004(xy, Y, xy_n))

    print('\n')

    print('1d array input:')
    print(whiteness_CIE2004_vectorise(xy, Y, xy_n))

    print('\n')

    print('2d array input:')
    xy = np.tile(xy, (6, 1))
    xy_n = np.tile(xy_n, (6, 1))
    print(whiteness_CIE2004_vectorise(xy, Y, xy_n))

    print('\n')

    print('3d array input:')
    xy = np.reshape(xy, (2, 3, 2))
    xy_n = np.reshape(xy_n, (2, 3, 2))
    print(whiteness_CIE2004_vectorise(xy, Y, xy_n))

    print('\n')


# whiteness_CIE2004_analysis()


def whiteness_CIE2004_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    Y = 100
    xy_n = (0.3139, 0.3311)

    times = timeit.Timer(
        functools.partial(
            whiteness_CIE2004_2d,
            DATA_HD1[..., 0:2])).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            whiteness_CIE2004_vectorise,
            DATA_HD1[..., 0:2], Y, xy_n)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('whiteness_CIE2004\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1[..., 0:2]), a, b))


# whiteness_CIE2004_profile()

# #############################################################################
# #############################################################################
# ## colour.difference.delta_e
# #############################################################################
# #############################################################################

# #############################################################################
# ## colour.delta_E_CIE1976
# #############################################################################
from colour.difference.delta_e import *


def delta_E_CIE1976_2d(Lab1, Lab2):
    for i in range(len(Lab1)):
        delta_E_CIE1976(Lab1[i], Lab2[i])


def delta_E_CIE1976_vectorise(Lab1, Lab2, **kwargs):
    delta_E = np.linalg.norm(np.asarray(Lab1) - np.asarray(Lab2), axis=-1)

    return delta_E


def delta_E_CIE1976_analysis():
    message_box('delta_E_CIE1976')

    print('Reference:')
    Lab1 = np.array([100, 21.57210357, 272.2281935])
    Lab2 = np.array([100, 426.67945353, 72.39590835])
    print(delta_E_CIE1976(Lab1, Lab2))

    print('\n')

    print('1d array input:')
    print(delta_E_CIE1976_vectorise(Lab1, Lab2))

    print('\n')

    print('2d array input:')
    Lab1 = np.tile(Lab1, (6, 1))
    Lab2 = np.tile(Lab2, (6, 1))
    print(delta_E_CIE1976_vectorise(Lab1, Lab2))

    print('\n')

    print('3d array input:')
    Lab1 = np.reshape(Lab1, (2, 3, 3))
    Lab2 = np.reshape(Lab2, (2, 3, 3))
    print(delta_E_CIE1976_vectorise(Lab1, Lab2))

    print('\n')


# delta_E_CIE1976_analysis()


def delta_E_CIE1976_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            delta_E_CIE1976_2d,
            DATA_HD1, DATA_HD2)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            delta_E_CIE1976_vectorise,
            DATA_HD1, DATA_HD2)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('delta_E_CIE1976\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# delta_E_CIE1976_profile()

# #############################################################################
# # ### colour.delta_E_CIE1994
# #############################################################################


def delta_E_CIE1994_2d(Lab1, Lab2):
    for i in range(len(Lab1)):
        delta_E_CIE1994(Lab1[i], Lab2[i])


def delta_E_CIE1994_vectorise(Lab1, Lab2, textiles=True, **kwargs):
    k1 = 0.048 if textiles else 0.045
    k2 = 0.014 if textiles else 0.015
    kL = 2 if textiles else 1
    kC = 1
    kH = 1

    L1, a1, b1 = tsplit(Lab1)
    L2, a2, b2 = tsplit(Lab2)

    C1 = np.sqrt(a1 ** 2 + b1 ** 2)
    C2 = np.sqrt(a2 ** 2 + b2 ** 2)

    sL = 1
    sC = 1 + k1 * C1
    sH = 1 + k2 * C1

    delta_L = L1 - L2
    delta_C = C1 - C2
    delta_A = a1 - a2
    delta_B = b1 - b2

    delta_H = np.sqrt(delta_A ** 2 + delta_B ** 2 - delta_C ** 2)

    L = (delta_L / (kL * sL)) ** 2
    C = (delta_C / (kC * sC)) ** 2
    H = (delta_H / (kH * sH)) ** 2

    delta_E = np.sqrt(L + C + H)

    return delta_E


def delta_E_CIE1994_analysis():
    message_box('delta_E_CIE1994')

    print('Reference:')
    Lab1 = np.array([100, 21.57210357, 272.2281935])
    Lab2 = np.array([100, 426.67945353, 72.39590835])
    print(delta_E_CIE1994(Lab1, Lab2))

    print('\n')

    print('1d array input:')
    print(delta_E_CIE1994_vectorise(Lab1, Lab2))

    print('\n')

    print('2d array input:')
    Lab1 = np.tile(Lab1, (6, 1))
    Lab2 = np.tile(Lab2, (6, 1))
    print(delta_E_CIE1994_vectorise(Lab1, Lab2))

    print('\n')

    print('3d array input:')
    Lab1 = np.reshape(Lab1, (2, 3, 3))
    Lab2 = np.reshape(Lab2, (2, 3, 3))
    print(delta_E_CIE1994_vectorise(Lab1, Lab2))

    print('\n')


# delta_E_CIE1994_analysis()


def delta_E_CIE1994_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            delta_E_CIE1994_2d,
            DATA_HD1, DATA_HD2)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            delta_E_CIE1994_vectorise,
            DATA_HD1, DATA_HD2)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('delta_E_CIE1994\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# delta_E_CIE1994_profile()

# #############################################################################
# # ### colour.delta_E_CIE2000
# #############################################################################


def delta_E_CIE2000_2d(Lab1, Lab2):
    delta_E = []
    for i in range(len(Lab1)):
        delta_E.append(delta_E_CIE2000(Lab1[i], Lab2[i]))
    return delta_E


def delta_E_CIE2000_vectorise(Lab1, Lab2, **kwargs):
    kL = 1
    kC = 1
    kH = 1

    L1, a1, b1 = tsplit(Lab1)
    L2, a2, b2 = tsplit(Lab2)

    l_bar_prime = 0.5 * (L1 + L2)

    c1 = np.sqrt(a1 * a1 + b1 * b1)
    c2 = np.sqrt(a2 * a2 + b2 * b2)

    c_bar = 0.5 * (c1 + c2)
    c_bar7 = np.power(c_bar, 7)

    g = 0.5 * (1 - np.sqrt(c_bar7 / (c_bar7 + 25 ** 7)))

    a1_prime = a1 * (1 + g)
    a2_prime = a2 * (1 + g)
    c1_prime = np.sqrt(a1_prime * a1_prime + b1 * b1)
    c2_prime = np.sqrt(a2_prime * a2_prime + b2 * b2)
    c_bar_prime = 0.5 * (c1_prime + c2_prime)

    h1_prime = np.asarray(np.rad2deg(np.arctan2(b1, a1_prime)))
    h1_prime[np.asarray(h1_prime < 0.0)] += 360

    h2_prime = np.asarray(np.rad2deg(np.arctan2(b2, a2_prime)))
    h2_prime[np.asarray(h2_prime < 0.0)] += 360

    h_bar_prime = np.where(np.fabs(h1_prime - h2_prime) <= 180,
                           0.5 * (h1_prime + h2_prime),
                           (0.5 * (h1_prime + h2_prime + 360)))

    t = (1 - 0.17 * np.cos(np.deg2rad(h_bar_prime - 30)) +
         0.24 * np.cos(np.deg2rad(2 * h_bar_prime)) +
         0.32 * np.cos(np.deg2rad(3 * h_bar_prime + 6)) -
         0.20 * np.cos(np.deg2rad(4 * h_bar_prime - 63)))

    h = h2_prime - h1_prime
    delta_h_prime = np.where(h2_prime <= h1_prime, h - 360, h + 360)
    delta_h_prime = np.where(np.fabs(h) <= 180, h, delta_h_prime)

    delta_L_prime = L2 - L1
    delta_C_prime = c2_prime - c1_prime
    delta_H_prime = (2 * np.sqrt(c1_prime * c2_prime) *
                     np.sin(np.deg2rad(0.5 * delta_h_prime)))

    sL = 1 + ((0.015 * (l_bar_prime - 50) * (l_bar_prime - 50)) /
              np.sqrt(20 + (l_bar_prime - 50) * (l_bar_prime - 50)))
    sC = 1 + 0.045 * c_bar_prime
    sH = 1 + 0.015 * c_bar_prime * t

    delta_theta = (30 * np.exp(-((h_bar_prime - 275) / 25) *
                               ((h_bar_prime - 275) / 25)))

    c_bar_prime7 = c_bar_prime ** 7

    rC = np.sqrt(c_bar_prime7 / (c_bar_prime7 + 25 ** 7))
    rT = -2 * rC * np.sin(np.deg2rad(2 * delta_theta))

    delta_E = np.sqrt(
        (delta_L_prime / (kL * sL)) * (delta_L_prime / (kL * sL)) +
        (delta_C_prime / (kC * sC)) * (delta_C_prime / (kC * sC)) +
        (delta_H_prime / (kH * sH)) * (delta_H_prime / (kH * sH)) +
        (delta_C_prime / (kC * sC)) * (delta_H_prime / (kH * sH)) * rT)

    return delta_E


def delta_E_CIE2000_analysis():
    message_box('delta_E_CIE2000')

    print('Reference:')
    Lab1 = np.array([100, 21.57210357, 272.2281935])
    Lab2 = np.array([100, 426.67945353, 72.39590835])
    print(delta_E_CIE2000(Lab1, Lab2))

    print('\n')

    print('1d array input:')
    print(delta_E_CIE2000_vectorise(Lab1, Lab2))

    print('\n')

    print('2d array input:')
    Lab1 = np.tile(Lab1, (6, 1))
    Lab2 = np.tile(Lab2, (6, 1))
    print(delta_E_CIE2000_vectorise(Lab1, Lab2))

    print('\n')

    print('3d array input:')
    Lab1 = np.reshape(Lab1, (2, 3, 3))
    Lab2 = np.reshape(Lab2, (2, 3, 3))
    print(delta_E_CIE2000_vectorise(Lab1, Lab2))

    print('\n')

    np.testing.assert_almost_equal(
        np.ravel(delta_E_CIE2000_2d(DATA1 * 360, DATA2 * 360)),
        np.ravel(delta_E_CIE2000_vectorise(DATA1 * 360, DATA2 * 360)))


# delta_E_CIE2000_analysis()


def delta_E_CIE2000_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            delta_E_CIE2000_2d,
            DATA_HD1, DATA_HD2)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            delta_E_CIE2000_vectorise,
            DATA_HD1, DATA_HD2)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('delta_E_CIE2000\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# delta_E_CIE2000_profile(3, 3)

# #############################################################################
# # ### colour.delta_E_CMC
# #############################################################################


def delta_E_CMC_2d(Lab1, Lab2):
    delta_E = []
    for i in range(len(Lab1)):
        delta_E.append(delta_E_CMC(Lab1[i], Lab2[i]))
    return delta_E


def delta_E_CMC_vectorise(Lab1, Lab2, l=2, c=1):
    L1, a1, b1 = tsplit(Lab1)
    L2, a2, b2 = tsplit(Lab2)

    c1 = np.sqrt(a1 * a1 + b1 * b1)
    c2 = np.sqrt(a2 * a2 + b2 * b2)
    sl = np.where(L1 < 16, 0.511, (0.040975 * L1) / (1 + 0.01765 * L1))
    sc = 0.0638 * c1 / (1 + 0.0131 * c1) + 0.638
    h1 = np.where(c1 < 0.000001, 0, np.rad2deg(np.arctan2(b1, a1)))

    while np.any(h1 < 0):
        h1[h1 < 0] += 360

    while np.any(h1 >= 360):
        h1[h1 >= 360] -= 360

    t = np.where(np.logical_and(h1 >= 164, h1 <= 345),
                 0.56 + np.fabs(0.2 * np.cos(np.deg2rad(h1 + 168))),
                 0.36 + np.fabs(0.4 * np.cos(np.deg2rad(h1 + 35))))

    c4 = c1 * c1 * c1 * c1
    f = np.sqrt(c4 / (c4 + 1900))
    sh = sc * (f * t + 1 - f)

    delta_L = L1 - L2
    delta_C = c1 - c2
    delta_A = a1 - a2
    delta_B = b1 - b2
    delta_H2 = delta_A * delta_A + delta_B * delta_B - delta_C * delta_C

    v1 = delta_L / (l * sl)
    v2 = delta_C / (c * sc)
    v3 = sh

    delta_E = np.sqrt(v1 * v1 + v2 * v2 + (delta_H2 / (v3 * v3)))

    return delta_E


def delta_E_CMC_analysis():
    message_box('delta_E_CMC')

    print('Reference:')
    Lab1 = np.array([100, 21.57210357, 272.2281935])
    Lab2 = np.array([100, 426.67945353, 72.39590835])
    print(delta_E_CMC(Lab1, Lab2))

    print('\n')

    print('1d array input:')
    print(delta_E_CMC_vectorise(Lab1, Lab2))

    print('\n')

    print('2d array input:')
    Lab1 = np.tile(Lab1, (6, 1))
    Lab2 = np.tile(Lab2, (6, 1))
    print(delta_E_CMC_vectorise(Lab1, Lab2))

    print('\n')

    print('3d array input:')
    Lab1 = np.reshape(Lab1, (2, 3, 3))
    Lab2 = np.reshape(Lab2, (2, 3, 3))
    print(delta_E_CMC_vectorise(Lab1, Lab2))

    print('\n')

    np.testing.assert_almost_equal(
        np.ravel(delta_E_CMC_2d(DATA1 * 360, DATA2 * 360)),
        np.ravel(delta_E_CMC_vectorise(DATA1 * 360, DATA2 * 360)))


# delta_E_CMC_analysis()


def delta_E_CMC_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            delta_E_CMC_2d,
            DATA_HD1, DATA_HD2)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            delta_E_CMC_vectorise,
            DATA_HD1, DATA_HD2)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('delta_E_CMC\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# delta_E_CMC_profile()

# #############################################################################
# #############################################################################
# ## colour.models.cie_xyy
# #############################################################################
# #############################################################################

# #############################################################################
# # ### colour.XYZ_to_xyY
# #############################################################################
from colour import ILLUMINANTS
from colour.models.cie_xyy import *


def XYZ_to_xyY_2d(XYZ):
    xyY = []
    for i in range(len(XYZ)):
        xyY.append(XYZ_to_xyY(XYZ[i]))
    return xyY


@handle_numpy_errors(divide='ignore', invalid='ignore')
def XYZ_to_xyY_vectorise(XYZ,
                         illuminant=ILLUMINANTS.get(
                             'CIE 1931 2 Degree Standard Observer').get(
                             'D50')):
    XYZ = np.asarray(XYZ)
    X, Y, Z = tsplit(XYZ)
    xy_w = np.asarray(illuminant)

    XYZ_n = np.zeros(XYZ.shape)
    XYZ_n[..., 0:2] = xy_w

    xyY = np.where(
        XYZ == 0,
        XYZ_n,
        tstack((X / (X + Y + Z), Y / (X + Y + Z), Y)))

    return xyY


def XYZ_to_xyY_analysis():
    message_box('XYZ_to_xyY')

    print('Reference:')
    XYZ = np.array([0.07049534, 0.1008, 0.09558313])
    print(XYZ_to_xyY(XYZ))

    print('\n')

    print('1d array input:')
    print(XYZ_to_xyY_vectorise(XYZ))

    print('\n')

    print('2d array input:')
    XYZ = np.tile(XYZ, (6, 1))
    print(XYZ_to_xyY_vectorise(XYZ))

    print('\n')

    XYZ = np.tile((0, 0, 0), (6, 1))
    print(XYZ_to_xyY_vectorise(XYZ))

    print('\n')

    print('3d array input:')
    XYZ = np.reshape(np.tile([0.07049534, 0.1008, 0.09558313], (6, 1)),
                     (2, 3, 3))
    print(XYZ_to_xyY_vectorise(XYZ))

    print('\n')

    np.testing.assert_almost_equal(
        np.ravel(XYZ_to_xyY_2d(DATA1)),
        np.ravel(XYZ_to_xyY_vectorise(DATA1)))


# XYZ_to_xyY_analysis()


def XYZ_to_xyY_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            XYZ_to_xyY_2d,
            DATA_HD1)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            XYZ_to_xyY_vectorise,
            DATA_HD1)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('XYZ_to_xyY\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# XYZ_to_xyY_profile()

# #############################################################################
# # ### colour.xyY_to_XYZ
# #############################################################################


def xyY_to_XYZ_2d(xyY):
    XYZ = []
    for i in range(len(xyY)):
        XYZ.append(xyY_to_XYZ(xyY[i]))
    return XYZ


@handle_numpy_errors(divide='ignore')
def xyY_to_XYZ_vectorise(xyY):
    x, y, Y = tsplit(xyY)

    XYZ = np.where((y == 0)[..., np.newaxis],
                   tstack((y, y, y)),
                   tstack((x * Y / y, Y, (1 - x - y) * Y / y)))

    return XYZ


def xyY_to_XYZ_analysis():
    message_box('xyY_to_XYZ')

    print('Reference:')
    xyY = np.array([0.26414772, 0.37770001, 0.1008])
    print(xyY_to_XYZ(xyY))

    print('\n')

    print('1d array input:')
    print(xyY_to_XYZ_vectorise(xyY))

    print('\n')

    print('2d array input:')
    xyY = np.tile(xyY, (6, 1))
    print(xyY_to_XYZ_vectorise(xyY))

    print('\n')

    print('3d array input:')
    xyY = np.reshape(xyY, (2, 3, 3))
    print(xyY_to_XYZ_vectorise(xyY))

    print('\n')

    np.testing.assert_almost_equal(
        np.ravel(xyY_to_XYZ_2d(DATA1)),
        np.ravel(xyY_to_XYZ_vectorise(DATA1)))


# xyY_to_XYZ_analysis()


def xyY_to_XYZ_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            xyY_to_XYZ_2d,
            DATA_HD1)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            xyY_to_XYZ_vectorise,
            DATA_HD1)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('xyY_to_XYZ\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# xyY_to_XYZ_profile()

# #############################################################################
# # ### colour.xy_to_XYZ
# #############################################################################


def xy_to_XYZ_2d(xy):
    for i in range(len(xy)):
        xy_to_XYZ(xy[i])


def xy_to_XYZ_vectorise(xy):
    x, y = tsplit(xy)

    xyY = tstack((x, y, np.ones(x.shape)))
    XYZ = xyY_to_XYZ_vectorise(xyY)

    return XYZ


def xy_to_XYZ_analysis():
    message_box('xy_to_XYZ')

    print('Reference:')
    xy = (0.26414772236966133, 0.37770000704815188)
    print(xy_to_XYZ(xy))

    print('\n')

    print('1d array input:')
    print(xy_to_XYZ_vectorise(xy))

    print('\n')

    print('2d array input:')
    xy = np.tile(xy, (6, 1))
    print(xy_to_XYZ_vectorise(xy))

    print('\n')

    print('3d array input:')
    xy = np.reshape(xy, (2, 3, 2))
    print(xy_to_XYZ_vectorise(xy))

    print('\n')


# xy_to_XYZ_analysis()


def xy_to_XYZ_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            xy_to_XYZ_2d,
            DATA_HD1[..., 0:2])).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            xy_to_XYZ_vectorise,
            DATA_HD1[..., 0:2])).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('xy_to_XYZ\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1[..., 0:2]), a, b))


# xy_to_XYZ_profile()


# #############################################################################
# # ### colour.XYZ_to_xy
# #############################################################################


def XYZ_to_xy_2d(XYZ):
    for i in range(len(XYZ)):
        XYZ_to_xy(XYZ[i])


def XYZ_to_xy_vectorise(XYZ,
                        illuminant=ILLUMINANTS.get(
                            'CIE 1931 2 Degree Standard Observer').get('D50')):
    xyY = XYZ_to_xyY_vectorise(XYZ, illuminant)
    xy = xyY[..., 0:2]

    return xy


def XYZ_to_xy_analysis():
    message_box('XYZ_to_xy')

    print('Reference:')
    XYZ = np.array([0.07049534, 0.1008, 0.09558313])
    print(XYZ_to_xy(XYZ))

    print('\n')

    print('1d array input:')
    print(XYZ_to_xy_vectorise(XYZ))

    print('\n')

    print('2d array input:')
    XYZ = np.tile(XYZ, (6, 1))
    print(XYZ_to_xy_vectorise(XYZ))

    print('\n')

    print('3d array input:')
    XYZ = np.reshape(XYZ, (2, 3, 3))
    print(XYZ_to_xy_vectorise(XYZ))

    print('\n')


# XYZ_to_xy_analysis()


def XYZ_to_xy_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            XYZ_to_xy_2d,
            DATA_HD1)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            XYZ_to_xy_vectorise,
            DATA_HD1)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('XYZ_to_xy\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# XYZ_to_xy_profile()

# #############################################################################
# #############################################################################
# ## colour.models.cie_lab
# #############################################################################
# #############################################################################

# #############################################################################
# # ### colour.XYZ_to_Lab
# #############################################################################
from colour.models.cie_lab import *


def XYZ_to_Lab_2d(XYZ):
    Lab = []
    for i in range(len(XYZ)):
        Lab.append(XYZ_to_Lab(XYZ[i]))
    return Lab


def XYZ_to_Lab_vectorise(XYZ,
                         illuminant=ILLUMINANTS.get(
                             'CIE 1931 2 Degree Standard Observer').get(
                             'D50')):
    XYZ = np.asarray(XYZ)
    XYZ_r = xy_to_XYZ_vectorise(illuminant)

    XYZ_f = XYZ / XYZ_r

    XYZ_f = np.where(XYZ_f > CIE_E,
                     np.power(XYZ_f, 1 / 3),
                     (CIE_K * XYZ_f + 16) / 116)

    X_f, Y_f, Z_f = tsplit(XYZ_f)

    L = 116 * Y_f - 16
    a = 500 * (X_f - Y_f)
    b = 200 * (Y_f - Z_f)

    Lab = tstack((L, a, b))

    return Lab


def XYZ_to_Lab_analysis():
    message_box('XYZ_to_Lab')

    print('Reference:')
    XYZ = np.array([0.07049534, 0.1008, 0.09558313])
    print(XYZ_to_Lab(XYZ))

    print('\n')

    print('1d array input:')
    print(XYZ_to_Lab_vectorise(XYZ))

    print('\n')

    print('2d array input:')
    XYZ = np.tile(XYZ, (6, 1))
    print(XYZ_to_Lab_vectorise(XYZ))

    print('\n')

    print('3d array input:')
    XYZ = np.reshape(XYZ, (2, 3, 3))
    print(XYZ_to_Lab_vectorise(XYZ))

    print('\n')

    np.testing.assert_almost_equal(
        np.ravel(XYZ_to_Lab_2d(DATA1)),
        np.ravel(XYZ_to_Lab_vectorise(DATA1)))


# XYZ_to_Lab_analysis()


def XYZ_to_Lab_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            XYZ_to_Lab_2d,
            DATA_HD1)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            XYZ_to_Lab_vectorise,
            DATA_HD1)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('XYZ_to_Lab\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# XYZ_to_Lab_profile()

# #############################################################################
# # ### colour.Lab_to_XYZ
# #############################################################################


def Lab_to_XYZ_2d(Lab):
    XYZ = []
    for i in range(len(Lab)):
        XYZ.append(Lab_to_XYZ(Lab[i]))
    return XYZ


def Lab_to_XYZ_vectorise(Lab,
                         illuminant=ILLUMINANTS.get(
                             'CIE 1931 2 Degree Standard Observer').get(
                             'D50')):
    L, a, b = tsplit(Lab)
    XYZ_r = xy_to_XYZ_vectorise(illuminant)

    f_y = (L + 16) / 116
    f_x = a / 500 + f_y
    f_z = f_y - b / 200

    x_r = np.where(f_x ** 3 > CIE_E, f_x ** 3, (116 * f_x - 16) / CIE_K)
    y_r = np.where(L > CIE_K * CIE_E, ((L + 16) / 116) ** 3, L / CIE_K)
    z_r = np.where(f_z ** 3 > CIE_E, f_z ** 3, (116 * f_z - 16) / CIE_K)

    XYZ = tstack((x_r, y_r, z_r)) * XYZ_r

    return XYZ


def Lab_to_XYZ_analysis():
    message_box('Lab_to_XYZ')

    print('Reference:')
    Lab = np.array([37.9856291, -23.62302887, -4.41417036])
    print(Lab_to_XYZ(Lab))

    print('\n')

    print('1d array input:')
    print(Lab_to_XYZ_vectorise(Lab))

    print('\n')

    print('2d array input:')
    Lab = np.tile(Lab, (6, 1))
    print(Lab_to_XYZ_vectorise(Lab))

    print('\n')

    print('3d array input:')
    Lab = np.reshape(Lab, (2, 3, 3))
    print(Lab_to_XYZ_vectorise(Lab))

    print('\n')

    np.testing.assert_almost_equal(
        np.ravel(Lab_to_XYZ_2d(DATA1 * 360)),
        np.ravel(Lab_to_XYZ_vectorise(DATA1 * 360)))


# Lab_to_XYZ_analysis()


def Lab_to_XYZ_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            Lab_to_XYZ_2d,
            DATA_HD1)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            Lab_to_XYZ_vectorise,
            DATA_HD1)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('Lab_to_XYZ\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# Lab_to_XYZ_profile()

# #############################################################################
# # ### colour.Lab_to_LCHab
# #############################################################################


def Lab_to_LCHab_2d(Lab):
    for i in range(len(Lab)):
        Lab_to_LCHab(Lab[i])


def Lab_to_LCHab_vectorise(Lab):
    L, a, b = tsplit(Lab)

    H = np.asarray(180 * np.arctan2(b, a) / np.pi)
    H[np.asarray(H < 0)] += 360

    LCHab = tstack((L, np.sqrt(a ** 2 + b ** 2), H))

    return LCHab


def Lab_to_LCHab_analysis():
    message_box('Lab_to_LCHab')

    print('Reference:')
    Lab = np.array([37.9856291, -23.62302887, -4.41417036])
    print(Lab_to_LCHab(Lab))

    print('\n')

    print('1d array input:')
    print(Lab_to_LCHab_vectorise(Lab))

    print('\n')

    print('2d array input:')
    Lab = np.tile(Lab, (6, 1))
    print(Lab_to_LCHab_vectorise(Lab))

    print('\n')

    print('2d array input:')
    Lab = np.reshape(Lab, (2, 3, 3))
    print(Lab_to_LCHab_vectorise(Lab))

    print('\n')


# Lab_to_LCHab_analysis()


def Lab_to_LCHab_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            Lab_to_LCHab_2d,
            DATA_HD1)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            Lab_to_LCHab_vectorise,
            DATA_HD1)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('Lab_to_LCHab\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# Lab_to_LCHab_profile()

# #############################################################################
# # ### colour.LCHab_to_Lab
# #############################################################################


def LCHab_to_Lab_2d(LCHab):
    for i in range(len(LCHab)):
        LCHab_to_Lab(LCHab[i])


def LCHab_to_Lab_vectorise(LCHab):
    L, C, H = tsplit(LCHab)

    return tstack((L,
                   C * np.cos(np.radians(H)),
                   C * np.sin(np.radians(H))))


def LCHab_to_Lab_analysis():
    message_box('LCHab_to_Lab')

    print('Reference:')
    LCHab = np.array([37.9856291, 24.03190365, 190.58415972])
    print(LCHab_to_Lab(LCHab))

    print('\n')

    print('1d array input:')
    print(LCHab_to_Lab_vectorise(LCHab))

    print('\n')

    print('2d array input:')
    LCHab = np.tile(LCHab, (6, 1))
    print(LCHab_to_Lab_vectorise(LCHab))

    print('\n')

    print('3d array input:')
    LCHab = np.reshape(LCHab, (2, 3, 3))
    print(LCHab_to_Lab_vectorise(LCHab))

    print('\n')


# LCHab_to_Lab_analysis()


def LCHab_to_Lab_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            LCHab_to_Lab_2d,
            DATA_HD1)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            LCHab_to_Lab_vectorise,
            DATA_HD1)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('LCHab_to_Lab\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# LCHab_to_Lab_profile()


# #############################################################################
# #############################################################################
# ## colour.models.cie_luv
# #############################################################################
# #############################################################################

# #############################################################################
# # ### colour.XYZ_to_Luv
# #############################################################################
from colour.models.cie_luv import *


def XYZ_to_Luv_2d(XYZ):
    Luv = []
    for i in range(len(XYZ)):
        Luv.append(XYZ_to_Luv(XYZ[i]))
    return Luv


def XYZ_to_Luv_vectorise(XYZ,
                         illuminant=ILLUMINANTS.get(
                             'CIE 1931 2 Degree Standard Observer').get(
                             'D50')):
    X, Y, Z = tsplit(XYZ)
    X_r, Y_r, Z_r = tsplit(xy_to_XYZ_vectorise(illuminant))

    y_r = Y / Y_r

    L = np.where(y_r > CIE_E, 116 * y_r ** (1 / 3) - 16, CIE_K * y_r)

    u = (13 * L * ((4 * X / (X + 15 * Y + 3 * Z)) -
                   (4 * X_r / (X_r + 15 * Y_r + 3 * Z_r))))
    v = (13 * L * ((9 * Y / (X + 15 * Y + 3 * Z)) -
                   (9 * Y_r / (X_r + 15 * Y_r + 3 * Z_r))))

    Luv = tstack((L, u, v))

    return Luv


def XYZ_to_Luv_analysis():
    message_box('XYZ_to_Luv')

    print('Reference:')
    XYZ = np.array([0.07049534, 0.1008, 0.09558313])
    print(XYZ_to_Luv(XYZ))

    print('\n')

    print('1d array input:')
    print(XYZ_to_Luv_vectorise(XYZ))

    print('\n')

    print('2d array input:')
    XYZ = np.tile(XYZ, (6, 1))
    print(XYZ_to_Luv_vectorise(XYZ))

    print('\n')

    print('2d array input:')
    XYZ = np.reshape(XYZ, (2, 3, 3))
    print(XYZ_to_Luv_vectorise(XYZ))

    print('\n')

    np.testing.assert_almost_equal(
        np.ravel(XYZ_to_Luv_2d(DATA1)),
        np.ravel(XYZ_to_Luv_vectorise(DATA1)))


# XYZ_to_Luv_analysis()


def XYZ_to_Luv_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            XYZ_to_Luv_2d,
            DATA_HD1)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            XYZ_to_Luv_vectorise,
            DATA_HD1)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('XYZ_to_Luv\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# XYZ_to_Luv_profile()

# #############################################################################
# # ### colour.Luv_to_XYZ
# #############################################################################


def Luv_to_XYZ_2d(Luv):
    XYZ = []
    for i in range(len(Luv)):
        XYZ.append(Luv_to_XYZ(Luv[i]))
    return XYZ


def Luv_to_XYZ_vectorise(Luv,
                         illuminant=ILLUMINANTS.get(
                             'CIE 1931 2 Degree Standard Observer').get(
                             'D50')):
    L, u, v = tsplit(Luv)
    X_r, Y_r, Z_r = tsplit(xy_to_XYZ_vectorise(illuminant))

    Y = np.where(L > CIE_E * CIE_K, ((L + 16) / 116) ** 3, L / CIE_K)

    a = 1 / 3 * ((52 * L / (u + 13 * L *
                            (4 * X_r / (X_r + 15 * Y_r + 3 * Z_r)))) - 1)
    b = -5 * Y
    c = -1 / 3.0
    d = Y * (39 * L / (v + 13 * L *
                       (9 * Y_r / (X_r + 15 * Y_r + 3 * Z_r))) - 5)

    X = (d - b) / (a - c)
    Z = X * a + b

    XYZ = tstack((X, Y, Z))

    return XYZ


def Luv_to_XYZ_analysis():
    message_box('Luv_to_XYZ')

    print('Reference:')
    Luv = np.array([37.9856291, -28.79229446, -1.3558195])
    print(Luv_to_XYZ(Luv))

    print('\n')

    print('1d array input:')
    print(Luv_to_XYZ_vectorise(Luv))

    print('\n')

    print('2d array input:')
    Luv = np.tile(Luv, (6, 1))
    print(Luv_to_XYZ_vectorise(Luv))

    print('\n')

    print('3d array input:')
    Luv = np.reshape(Luv, (2, 3, 3))
    print(Luv_to_XYZ_vectorise(Luv))

    print('\n')

    np.testing.assert_almost_equal(
        np.ravel(Luv_to_XYZ_2d(DATA1)),
        np.ravel(Luv_to_XYZ_vectorise(DATA1)))


# Luv_to_XYZ_analysis()


def Luv_to_XYZ_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            Luv_to_XYZ_2d,
            DATA_HD1)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            Luv_to_XYZ_vectorise,
            DATA_HD1)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('Luv_to_XYZ\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# Luv_to_XYZ_profile()

# #############################################################################
# # ### colour.Luv_to_uv
# #############################################################################


def Luv_to_uv_2d(Luv):
    for i in range(len(Luv)):
        Luv_to_uv(Luv[i])


def Luv_to_uv_vectorise(Luv,
                        illuminant=ILLUMINANTS.get(
                            'CIE 1931 2 Degree Standard Observer').get('D50')):
    X, Y, Z = tsplit(Luv_to_XYZ_vectorise(Luv, illuminant))

    uv = tstack((4 * X / (X + 15 * Y + 3 * Z),
                 9 * Y / (X + 15 * Y + 3 * Z)))

    return uv


def Luv_to_uv_analysis():
    message_box('Luv_to_uv')

    print('Reference:')
    Luv = np.array([37.9856291, -28.79229446, -1.3558195])
    print(Luv_to_uv(Luv))

    print('\n')

    print('1d array input:')
    print(Luv_to_uv_vectorise(Luv))

    print('\n')

    print('2d array input:')
    Luv = np.tile(Luv, (6, 1))
    print(Luv_to_uv_vectorise(Luv))

    print('\n')

    print('3d array input:')
    Luv = np.reshape(Luv, (2, 3, 3))
    print(Luv_to_uv_vectorise(Luv))

    print('\n')


# Luv_to_uv_analysis()


def Luv_to_uv_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            Luv_to_uv_2d,
            DATA_HD1)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            Luv_to_uv_vectorise,
            DATA_HD1)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('Luv_to_uv\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# Luv_to_uv_profile()

# #############################################################################
# # ### colour.Luv_uv_to_xy
# #############################################################################


def Luv_uv_to_xy_2d(uv):
    for i in range(len(uv)):
        Luv_uv_to_xy(uv[i])


def Luv_uv_to_xy_vectorise(uv):
    u, v = tsplit(uv)

    xy = tstack((9 * u / (6 * u - 16 * v + 12),
                 4 * v / (6 * u - 16 * v + 12)))

    return xy


def Luv_uv_to_xy_analysis():
    message_box('Luv_uv_to_xy')

    print('Reference:')
    uv = np.array([0.15085309882985695, 0.48532970854318019])
    print(Luv_uv_to_xy(uv))

    print('\n')

    print('1d array input:')
    print(Luv_uv_to_xy_vectorise(uv))

    print('\n')

    print('2d array input:')
    uv = np.tile(uv, (6, 1))
    print(Luv_uv_to_xy_vectorise(uv))

    print('\n')

    print('3d array input:')
    uv = np.reshape(uv, (2, 3, 2))
    print(Luv_uv_to_xy_vectorise(uv))

    print('\n')


# Luv_uv_to_xy_analysis()


def Luv_uv_to_xy_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            Luv_uv_to_xy_2d,
            DATA_HD1[..., 0:2])).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            Luv_uv_to_xy_vectorise,
            DATA_HD1[..., 0:2])).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('Luv_uv_to_xy\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1[..., 0:2]), a, b))


# Luv_uv_to_xy_profile()

# #############################################################################
# # ### colour.Luv_to_LCHuv
# #############################################################################


def Luv_to_LCHuv_2d(Luv):
    for i in range(len(Luv)):
        Luv_to_LCHuv(Luv[i])


def Luv_to_LCHuv_vectorise(Luv):
    L, u, v = tsplit(Luv)

    H = np.asarray(180 * np.arctan2(v, u) / np.pi)
    H[np.asarray(H < 0)] += 360

    LCHuv = tstack((L, np.sqrt(u ** 2 + v ** 2), H))

    return LCHuv


def Luv_to_LCHuv_analysis():
    message_box('Luv_to_LCHuv')

    print('Reference:')
    Luv = np.array([37.9856291, -28.79229446, -1.3558195])
    print(Luv_to_LCHuv(Luv))

    print('\n')

    print('1d array input:')
    print(Luv_to_LCHuv_vectorise(Luv))

    print('\n')

    print('2d array input:')
    Luv = np.tile(Luv, (6, 1))
    print(Luv_to_LCHuv_vectorise(Luv))

    print('\n')

    print('3d array input:')
    Luv = np.reshape(Luv, (2, 3, 3))
    print(Luv_to_LCHuv_vectorise(Luv))

    print('\n')


# Luv_to_LCHuv_analysis()


def Luv_to_LCHuv_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            Luv_to_LCHuv_2d,
            DATA_HD1)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            Luv_to_LCHuv_vectorise,
            DATA_HD1)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('Luv_to_LCHuv\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# Luv_to_LCHuv_profile()

# #############################################################################
# # ### colour.LCHuv_to_Luv
# #############################################################################


def LCHuv_to_Luv_2d(LCHuv):
    for i in range(len(LCHuv)):
        LCHuv_to_Luv(LCHuv[i])


def LCHuv_to_Luv_vectorise(LCHuv):
    L, C, H = tsplit(LCHuv)

    Luv = tstack((L, C * np.cos(np.radians(H)), C * np.sin(np.radians(H))))

    return Luv


def LCHuv_to_Luv_analysis():
    message_box('LCHuv_to_Luv')

    print('Reference:')
    LCHuv = np.array([37.9856291, 28.82419933, 182.69604747])
    print(LCHuv_to_Luv(LCHuv))

    print('\n')

    print('1d array input:')
    print(LCHuv_to_Luv_vectorise(LCHuv))

    print('\n')

    print('2d array input:')
    LCHuv = np.tile(LCHuv, (6, 1))
    print(LCHuv_to_Luv_vectorise(LCHuv))

    print('\n')

    print('3d array input:')
    LCHuv = np.reshape(LCHuv, (2, 3, 3))
    print(LCHuv_to_Luv_vectorise(LCHuv))

    print('\n')


# LCHuv_to_Luv_analysis()


def LCHuv_to_Luv_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            LCHuv_to_Luv_2d,
            DATA_HD1)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            LCHuv_to_Luv_vectorise,
            DATA_HD1)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('LCHuv_to_Luv\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# LCHuv_to_Luv_profile()

# #############################################################################
# #############################################################################
# ## colour.models.cie_ucs
# #############################################################################
# #############################################################################

# #############################################################################
# # ### colour.XYZ_to_UCS
# #############################################################################
from colour.models.cie_ucs import *


def XYZ_to_UCS_2d(XYZ):
    for i in range(len(XYZ)):
        XYZ_to_UCS(XYZ[i])


def XYZ_to_UCS_vectorise(XYZ):
    X, Y, Z = tsplit(XYZ)

    UVW = tstack((2 / 3 * X, Y, 1 / 2 * (-X + 3 * Y + Z)))

    return UVW


def XYZ_to_UCS_analysis():
    message_box('XYZ_to_UCS')

    print('Reference:')
    XYZ = np.array([0.07049534, 0.1008, 0.09558313])
    print(XYZ_to_UCS(XYZ))

    print('\n')

    print('1d array input:')
    print(XYZ_to_UCS_vectorise(XYZ))

    print('\n')

    print('2d array input:')
    XYZ = np.tile(XYZ, (6, 1))
    print(XYZ_to_UCS_vectorise(XYZ))

    print('\n')

    print('3d array input:')
    XYZ = np.reshape(XYZ, (2, 3, 3))
    print(XYZ_to_UCS_vectorise(XYZ))

    print('\n')


# XYZ_to_UCS_analysis()


def XYZ_to_UCS_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            XYZ_to_UCS_2d,
            DATA_HD1)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            XYZ_to_UCS_vectorise,
            DATA_HD1)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('XYZ_to_UCS\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# XYZ_to_UCS_profile()

# #############################################################################
# # ### colour.UCS_to_XYZ
# #############################################################################


def UCS_to_XYZ_2d(UVW):
    for i in range(len(UVW)):
        UCS_to_XYZ(UVW[i])


def UCS_to_XYZ_vectorise(UVW):
    U, V, W = tsplit(UVW)

    XYZ = tstack((3 / 2 * U, V, 3 / 2 * U - (3 * V) + (2 * W)))

    return XYZ


def UCS_to_XYZ_analysis():
    message_box('UCS_to_XYZ')

    print('Reference:')
    UVW = np.array([0.04699689, 0.1008, 0.1637439])
    print(UCS_to_XYZ(UVW))

    print('\n')

    print('1d array input:')
    print(UCS_to_XYZ_vectorise(UVW))

    print('\n')

    print('2d array input:')
    UVW = np.tile(UVW, (6, 1))
    print(UCS_to_XYZ_vectorise(UVW))

    print('\n')

    print('3d array input:')
    UVW = np.reshape(UVW, (2, 3, 3))
    print(UCS_to_XYZ_vectorise(UVW))

    print('\n')


# UCS_to_XYZ_analysis()


def UCS_to_XYZ_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            UCS_to_XYZ_2d,
            DATA_HD1)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            UCS_to_XYZ_vectorise,
            DATA_HD1)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('UCS_to_XYZ\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# UCS_to_XYZ_profile()

# #############################################################################
# # ### colour.UCS_to_uv
# #############################################################################


def UCS_to_uv_2d(UVW):
    for i in range(len(UVW)):
        UCS_to_uv(UVW[i])


def UCS_to_uv_vectorise(UVW):
    U, V, W = tsplit(UVW)

    uv = tstack((U / (U + V + W), V / (U + V + W)))

    return uv


def UCS_to_uv_analysis():
    message_box('UCS_to_uv')

    print('Reference:')
    UVW = np.array([0.04699689, 0.1008, 0.1637439])
    print(UCS_to_uv(UVW))

    print('\n')

    print('1d array input:')
    print(UCS_to_uv_vectorise(UVW))

    print('\n')

    print('2d array input:')
    UVW = np.tile(UVW, (6, 1))
    print(UCS_to_uv_vectorise(UVW))

    print('\n')

    print('3d array input:')
    UVW = np.reshape(UVW, (2, 3, 3))
    print(UCS_to_uv_vectorise(UVW))

    print('\n')


# UCS_to_uv_analysis()


def UCS_to_uv_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            UCS_to_uv_2d,
            DATA_HD1)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            UCS_to_uv_vectorise,
            DATA_HD1)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('UCS_to_uv\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# UCS_to_uv_profile()

# #############################################################################
# # ### colour.UCS_uv_to_xy
# #############################################################################


def UCS_uv_to_xy_2d(uv):
    for i in range(len(uv)):
        UCS_uv_to_xy(uv[i])


def UCS_uv_to_xy_vectorise(uv):
    u, v = tsplit(uv)

    xy = tstack((3 * u / (2 * u - 8 * v + 4), 2 * v / (2 * u - 8 * v + 4)))

    return xy


def UCS_uv_to_xy_analysis():
    message_box('UCS_uv_to_xy')

    print('Reference:')
    uv = np.array([0.15085308732766581, 0.3235531372954405])
    print(UCS_uv_to_xy(uv))

    print('\n')

    print('1d array input:')
    print(UCS_uv_to_xy_vectorise(uv))

    print('\n')

    print('2d array input:')
    uv = np.tile(uv, (6, 1))
    print(UCS_uv_to_xy_vectorise(uv))

    print('\n')

    print('3d array input:')
    uv = np.reshape(uv, (2, 3, 2))
    print(UCS_uv_to_xy_vectorise(uv))

    print('\n')


# UCS_uv_to_xy_analysis()


def UCS_uv_to_xy_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            UCS_uv_to_xy_2d,
            DATA_HD1[..., 0:2])).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            UCS_uv_to_xy_vectorise,
            DATA_HD1[..., 0:2])).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('UCS_uv_to_xy\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1[..., 0:2]), a, b))


# UCS_uv_to_xy_profile()

# #############################################################################
# #############################################################################
# ## colour.models.cie_uvw
# #############################################################################
# #############################################################################

# #############################################################################
# # ### colour.XYZ_to_UVW
# #############################################################################
from colour.models.cie_uvw import *


def XYZ_to_UVW_2d(XYZ):
    for i in range(len(XYZ)):
        XYZ_to_UVW(XYZ[i])


def XYZ_to_UVW_vectorise(XYZ,
                         illuminant=ILLUMINANTS.get(
                             'CIE 1931 2 Degree Standard Observer').get(
                             'D50')):
    xyY = XYZ_to_xyY_vectorise(XYZ, illuminant)
    x, y, Y = tsplit(xyY)

    u, v = tsplit(UCS_to_uv_vectorise(XYZ_to_UCS_vectorise(XYZ)))
    u_0, v_0 = tsplit(UCS_to_uv_vectorise(XYZ_to_UCS_vectorise(
        xy_to_XYZ_vectorise(illuminant))))

    W = 25 * Y ** (1 / 3) - 17
    U = 13 * W * (u - u_0)
    V = 13 * W * (v - v_0)

    UVW = tstack((U, V, W))

    return UVW


def XYZ_to_UVW_analysis():
    message_box('XYZ_to_UVW')

    print('Reference:')
    XYZ = np.array([0.07049534, 0.1008, 0.09558313])
    print(XYZ_to_UVW(XYZ))

    print('\n')

    print('1d array input:')
    print(XYZ_to_UVW_vectorise(XYZ))

    print('\n')

    print('2d array input:')
    XYZ = np.tile(XYZ, (6, 1))
    print(XYZ_to_UVW_vectorise(XYZ))

    print('\n')

    print('3d array input:')
    XYZ = np.reshape(XYZ, (2, 3, 3))
    print(XYZ_to_UVW_vectorise(XYZ))

    print('\n')


# XYZ_to_UVW_analysis()


def XYZ_to_UVW_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            XYZ_to_UVW_2d,
            DATA_HD1)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            XYZ_to_UVW_vectorise,
            DATA_HD1)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('XYZ_to_UVW\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# XYZ_to_UVW_profile()

# #############################################################################
# #############################################################################
# ## colour.models.deprecated
# #############################################################################
# #############################################################################

# #############################################################################
# # ### colour.models.deprecated.RGB_to_HSV
# #############################################################################
from colour.models.deprecated import *


def RGB_to_HSV_2d(RGB):
    HSV = []
    for i in range(len(RGB)):
        HSV.append(RGB_to_HSV(RGB[i]))
    return HSV


@handle_numpy_errors(divide='ignore', invalid='ignore')
def RGB_to_HSV_vectorise(RGB):
    minimum = np.amin(RGB, -1)
    maximum = np.amax(RGB, -1)
    delta = np.ptp(RGB, -1)

    V = maximum

    R, G, B = tsplit(RGB)

    S = np.asarray(delta / maximum)
    S[np.asarray(delta == 0)] = 0

    delta_R = (((maximum - R) / 6) + (delta / 2)) / delta
    delta_G = (((maximum - G) / 6) + (delta / 2)) / delta
    delta_B = (((maximum - B) / 6) + (delta / 2)) / delta

    H = delta_B - delta_G
    H = np.where(G == maximum, (1 / 3) + delta_R - delta_B, H)
    H = np.where(B == maximum, (2 / 3) + delta_G - delta_R, H)
    H[np.asarray(H < 0)] += 1
    H[np.asarray(H > 1)] -= 1
    H[np.asarray(delta == 0)] = 0

    HSV = tstack((H, S, V))

    return HSV


def RGB_to_HSV_analysis():
    message_box('RGB_to_HSV')

    print('Reference:')
    RGB = np.array([0.49019608, 0.98039216, 0.25098039])
    print(RGB_to_HSV(RGB))

    print('\n')

    print('1d array input:')
    print(RGB_to_HSV_vectorise(RGB))

    print('\n')

    print('2d array input:')
    RGB = np.tile(RGB, (6, 1))
    print(RGB_to_HSV_vectorise(RGB))

    print('\n')

    print('3d array input:')
    RGB = np.reshape(RGB, (2, 3, 3))
    print(RGB_to_HSV_vectorise(RGB))

    print('\n')

    np.testing.assert_almost_equal(
        np.ravel(RGB_to_HSV_2d(DATA1)),
        np.ravel(RGB_to_HSV_vectorise(DATA1)))


# RGB_to_HSV_analysis()


def RGB_to_HSV_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            RGB_to_HSV_2d,
            DATA_HD1)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            RGB_to_HSV_vectorise,
            DATA_HD1)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('RGB_to_HSV\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# RGB_to_HSV_profile()

# #############################################################################
# # ### colour.models.deprecated.HSV_to_RGB
# #############################################################################
def HSV_to_RGB_2d(HSV):
    RGB = []
    for i in range(len(HSV)):
        RGB.append(HSV_to_RGB(HSV[i]))
    return RGB


def HSV_to_RGB_vectorise(HSV):
    H, S, V = tsplit(HSV)

    h = np.asarray(H * 6)
    h[np.asarray(h == 6)] = 0

    i = np.floor(h)
    j = V * (1 - S)
    k = V * (1 - S * (h - i))
    l = V * (1 - S * (1 - (h - i)))

    i = tstack((i, i, i)).astype(np.uint8)
    RGB = np.choose(i,
                    (tstack((V, l, j)),
                     tstack((k, V, j)),
                     tstack((j, V, l)),
                     tstack((j, k, V)),
                     tstack((l, j, V)),
                     tstack((V, j, k))))

    return RGB


def HSV_to_RGB_analysis():
    message_box('HSV_to_RGB')

    print('Reference:')
    HSV = np.array([0.27867383, 0.744, 0.98039216])
    print(HSV_to_RGB(HSV))

    print('\n')

    print('1d array input:')
    print(HSV_to_RGB_vectorise(HSV))

    print('\n')

    print('2d array input:')
    HSV = np.tile(HSV, (6, 1))
    print(HSV_to_RGB_vectorise(HSV))

    print('\n')

    print('3d array input:')
    HSV = np.reshape(HSV, (2, 3, 3))
    print(HSV_to_RGB_vectorise(HSV))

    print('\n')

    np.testing.assert_almost_equal(
        np.ravel(HSV_to_RGB_2d(DATA1)),
        np.ravel(HSV_to_RGB_vectorise(DATA1)))


# HSV_to_RGB_analysis()


def HSV_to_RGB_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            HSV_to_RGB_2d,
            DATA_HD1)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            HSV_to_RGB_vectorise,
            DATA_HD1)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('HSV_to_RGB\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# HSV_to_RGB_profile()

# #############################################################################
# # ### colour.models.deprecated.RGB_to_HSL
# #############################################################################


def RGB_to_HSL_2d(RGB):
    HSL = []
    for i in range(len(RGB)):
        HSL.append(RGB_to_HSL(RGB[i]))
    return HSL


@handle_numpy_errors(divide='ignore', invalid='ignore')
def RGB_to_HSL_vectorise(RGB):
    minimum = np.amin(RGB, -1)
    maximum = np.amax(RGB, -1)
    delta = np.ptp(RGB, -1)

    R, G, B = tsplit(RGB)

    L = (maximum + minimum) / 2

    S = np.where(L < 0.5,
                 delta / (maximum + minimum),
                 delta / (2 - maximum - minimum))
    S[np.asarray(delta == 0)] = 0

    delta_R = (((maximum - R) / 6) + (delta / 2)) / delta
    delta_G = (((maximum - G) / 6) + (delta / 2)) / delta
    delta_B = (((maximum - B) / 6) + (delta / 2)) / delta

    H = delta_B - delta_G
    H = np.where(G == maximum, (1 / 3) + delta_R - delta_B, H)
    H = np.where(B == maximum, (2 / 3) + delta_G - delta_R, H)
    H[np.asarray(H < 0)] += 1
    H[np.asarray(H > 1)] -= 1
    H[np.asarray(delta == 0)] = 0

    HSL = tstack((H, S, L))

    return HSL


def RGB_to_HSL_analysis():
    message_box('RGB_to_HSL')

    print('Reference:')
    RGB = np.array([0.49019608, 0.98039216, 0.25098039])
    print(RGB_to_HSL(RGB))

    print('\n')

    print('1d array input:')
    print(RGB_to_HSL_vectorise(RGB))

    print('\n')

    print('2d array input:')
    RGB = np.tile(RGB, (6, 1))
    print(RGB_to_HSL_vectorise(RGB))

    print('\n')

    print('3d array input:')
    RGB = np.reshape(RGB, (2, 3, 3))
    print(RGB_to_HSL_vectorise(RGB))

    print('\n')

    np.testing.assert_almost_equal(
        np.ravel(RGB_to_HSL_2d(DATA1)),
        np.ravel(RGB_to_HSL_vectorise(DATA1)))


# RGB_to_HSL_analysis()


def RGB_to_HSL_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            RGB_to_HSL_2d,
            DATA_HD1)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            RGB_to_HSL_vectorise,
            DATA_HD1)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('RGB_to_HSL\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# RGB_to_HSL_profile()

# #############################################################################
# # ### colour.models.deprecated.HSL_to_RGB
# #############################################################################


def HSL_to_RGB_2d(HSL):
    RGB = []
    for i in range(len(HSL)):
        RGB.append(HSL_to_RGB(HSL[i]))
    return RGB


def HSL_to_RGB_vectorise(HSL):
    H, S, L = tsplit(HSL)

    def H_to_RGB(vi, vj, vH):
        """
        Converts *hue* value to *RGB* colourspace.
        """

        vH = np.asarray(vH)

        vH[np.asarray(vH < 0)] += 1
        vH[np.asarray(vH > 1)] -= 1

        v = np.full(vi.shape, np.nan)

        v = np.where(np.logical_and(6 * vH < 1, np.isnan(v)),
                     vi + (vj - vi) * 6 * vH,
                     v)
        v = np.where(np.logical_and(2 * vH < 1, np.isnan(v)),
                     vj,
                     v)
        v = np.where(np.logical_and(3 * vH < 2, np.isnan(v)),
                     vi + (vj - vi) * ((2 / 3) - vH) * 6,
                     v)
        v = np.where(np.isnan(v), vi, v)

        return v

    j = np.where(L < 0.5, L * (1 + S), (L + S) - (S * L))
    i = 2 * L - j

    R = H_to_RGB(i, j, H + (1 / 3))
    G = H_to_RGB(i, j, H)
    B = H_to_RGB(i, j, H - (1 / 3))

    R = np.where(S == 1, L, R)
    G = np.where(S == 1, L, G)
    B = np.where(S == 1, L, B)

    RGB = tstack((R, G, B))

    return RGB


def HSL_to_RGB_analysis():
    message_box('HSL_to_RGB')

    print('Reference:')
    HSL = np.array([0.27867383, 0.9489796, 0.61568627])
    print(HSL_to_RGB(HSL))

    print('\n')

    print('1d array input:')
    print(HSL_to_RGB_vectorise(HSL))

    print('\n')

    print('2d array input:')
    HSL = np.tile(HSL, (6, 1))
    print(HSL_to_RGB_vectorise(HSL))

    print('\n')

    print('3d array input:')
    HSL = np.reshape(HSL, (2, 3, 3))
    print(HSL_to_RGB_vectorise(HSL))

    print('\n')

    np.testing.assert_almost_equal(
        np.ravel(HSL_to_RGB_2d(DATA1)),
        np.ravel(HSL_to_RGB_vectorise(DATA1)))


# HSL_to_RGB_analysis()


def HSL_to_RGB_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            HSL_to_RGB_2d,
            DATA_HD1)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            HSL_to_RGB_vectorise,
            DATA_HD1)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('HSL_to_RGB\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# HSL_to_RGB_profile()

# #############################################################################
# # ### colour.models.deprecated.RGB_to_CMY
# #############################################################################


def RGB_to_CMY_2d(RGB):
    for i in range(len(RGB)):
        RGB_to_CMY(RGB[i])


def RGB_to_CMY_vectorise(RGB):
    CMY = 1 - np.asarray(RGB)

    return CMY


def RGB_to_CMY_analysis():
    message_box('RGB_to_CMY')

    print('Reference:')
    RGB = np.array([0.49019608, 0.98039216, 0.25098039])
    print(RGB_to_CMY(RGB))

    print('\n')

    print('1d array input:')
    print(RGB_to_CMY_vectorise(RGB))

    print('\n')

    print('2d array input:')
    RGB = np.tile(RGB, (6, 1))
    print(RGB_to_CMY_vectorise(RGB))

    print('\n')

    print('3d array input:')
    RGB = np.reshape(RGB, (2, 3, 3))
    print(RGB_to_CMY_vectorise(RGB))

    print('\n')


# RGB_to_CMY_analysis()


def RGB_to_CMY_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            RGB_to_CMY_2d,
            DATA_HD1)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            RGB_to_CMY_vectorise,
            DATA_HD1)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('RGB_to_CMY\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# RGB_to_CMY_profile()

# #############################################################################
# # ### colour.models.deprecated.CMY_to_RGB
# #############################################################################


def CMY_to_RGB_2d(CMY):
    for i in range(len(CMY)):
        CMY_to_RGB(CMY[i])


def CMY_to_RGB_vectorise(CMY):
    RGB = 1 - np.asarray(CMY)

    return RGB


def CMY_to_RGB_analysis():
    message_box('CMY_to_RGB')

    print('Reference:')
    CMY = np.array([0.50980392, 0.01960784, 0.74901961])
    print(CMY_to_RGB(CMY))

    print('\n')

    print('1d array input:')
    print(CMY_to_RGB_vectorise(CMY))

    print('\n')

    print('2d array input:')
    CMY = np.tile(CMY, (6, 1))
    print(CMY_to_RGB_vectorise(CMY))

    print('\n')

    print('3d array input:')
    CMY = np.reshape(CMY, (2, 3, 3))
    print(CMY_to_RGB_vectorise(CMY))

    print('\n')


# CMY_to_RGB_analysis()


def CMY_to_RGB_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            CMY_to_RGB_2d,
            DATA_HD1)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            CMY_to_RGB_vectorise,
            DATA_HD1)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('CMY_to_RGB\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# CMY_to_RGB_profile()

# #############################################################################
# # ### colour.models.deprecated.CMY_to_CMYK
# #############################################################################


def CMY_to_CMYK_2d(CMY):
    CMYK = []
    for i in range(len(CMY)):
        CMYK.append(CMY_to_CMYK(CMY[i]))
    return CMYK


def CMY_to_CMYK_vectorise(CMY):
    C, M, Y = tsplit(CMY)

    K = np.ones(C.shape)
    K = np.where(C < K, C, K)
    K = np.where(M < K, M, K)
    K = np.where(Y < K, Y, K)

    C = np.asarray((C - K) / (1 - K))
    M = np.asarray((M - K) / (1 - K))
    Y = np.asarray((Y - K) / (1 - K))

    C[np.asarray(K == 1)] = 0
    M[np.asarray(K == 1)] = 0
    Y[np.asarray(K == 1)] = 0

    CMYK = tstack((C, M, Y, K))

    return CMYK


def CMY_to_CMYK_analysis():
    message_box('CMY_to_CMYK')

    print('Reference:')
    CMY = np.array([0.49019608, 0.98039216, 0.25098039])
    print(CMY_to_CMYK(CMY))

    print('\n')

    print('1d array input:')
    print(CMY_to_CMYK_vectorise(CMY))

    print('\n')

    print('2d array input:')
    CMY = np.tile(CMY, (6, 1))
    print(CMY_to_CMYK_vectorise(CMY))

    print('\n')

    print('3d array input:')
    CMY = np.reshape(CMY, (2, 3, 3))
    print(CMY_to_CMYK_vectorise(CMY))

    print('\n')

    np.testing.assert_almost_equal(
        np.ravel(CMY_to_CMYK_2d(DATA1)),
        np.ravel(CMY_to_CMYK_vectorise(DATA1)))


# CMY_to_CMYK_analysis()


def CMY_to_CMYK_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            CMY_to_CMYK_2d,
            DATA_HD1)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            CMY_to_CMYK_vectorise,
            DATA_HD1)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('CMY_to_CMYK\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# CMY_to_CMYK_profile()

# #############################################################################
# # ### colour.models.deprecated.CMYK_to_CMY
# #############################################################################


def CMYK_to_CMY_2d(CMYK):
    for i in range(len(CMYK)):
        CMYK_to_CMY(CMYK[i])


def CMYK_to_CMY_vectorise(CMYK):
    C, M, Y, K = tsplit(CMYK)

    CMY = tstack((C * (1 - K) + K,
                  M * (1 - K) + K,
                  Y * (1 - K) + K))

    return CMY


def CMYK_to_CMY_analysis():
    message_box('CMYK_to_CMY')

    print('Reference:')
    CMYK = np.array([0.31937173, 0.97382199, 0., 0.25098039])
    print(CMYK_to_CMY(CMYK))

    print('\n')

    print('1d array input:')
    print(CMYK_to_CMY_vectorise(CMYK))

    print('\n')

    print('2d array input:')
    CMYK = np.tile(CMYK, (6, 1))
    print(CMYK_to_CMY_vectorise(CMYK))

    print('\n')

    print('3d array input:')
    CMYK = np.reshape(CMYK, (2, 3, 4))
    print(CMYK_to_CMY_vectorise(CMYK))

    print('\n')


# CMYK_to_CMY_analysis()


def CMYK_to_CMY_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    D1 = tstack((DATA_HD1[..., 0],
                 DATA_HD1[..., 0],
                 DATA_HD1[..., 0],
                 DATA_HD1[..., 0]))
    times = timeit.Timer(
        functools.partial(
            CMYK_to_CMY_2d,
            D1)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            CMYK_to_CMY_vectorise,
            D1)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('CMYK_to_CMY\t{0}\t{1}\t{2}'.format(
        len(D1), a, b))


# CMYK_to_CMY_profile()

# #############################################################################
# #############################################################################
# ## colour.models.derivation
# #############################################################################
# #############################################################################
# #############################################################################
# # ### colour.normalised_primary_matrix
# #############################################################################
from colour.models.derivation import *


def xy_to_z_vectorise(xy):
    x, y = tsplit(xy)

    z = 1 - x - y

    return z


def normalised_primary_matrix_vectorise(primaries, whitepoint):
    primaries = np.reshape(primaries, (3, 2))

    z = xy_to_z_vectorise(primaries)[..., np.newaxis]
    primaries = np.transpose(np.hstack((primaries, z)))

    whitepoint = xy_to_XYZ_vectorise(whitepoint)

    coefficients = np.dot(np.linalg.inv(primaries), whitepoint)
    coefficients = np.diagflat(coefficients)

    npm = np.dot(primaries, coefficients)

    return npm


def normalised_primary_matrix_analysis():
    message_box('normalised_primary_matrix')

    print('Reference:')
    P = np.array([0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700])
    W = (0.32168, 0.33767)
    print(normalised_primary_matrix(P, W))

    print('\n')

    print('Refactor:')
    print(normalised_primary_matrix_vectorise(P, W))

    print('\n')


# normalised_primary_matrix_analysis()

# #############################################################################
# # ### colour.primaries_whitepoint
# #############################################################################
def primaries_whitepoint_vectorise(npm):
    npm = npm.reshape((3, 3))

    primaries = XYZ_to_xy_vectorise(
        np.transpose(np.dot(npm, np.identity(3))))
    whitepoint = XYZ_to_xy_vectorise(
        np.transpose(np.dot(npm, np.ones((3, 1)))))

    # TODO: Should we return a tuple or stack the whitepoint chromaticity
    # coordinates to the primaries.
    return primaries, whitepoint


def primaries_whitepoint_analysis():
    message_box('primaries_whitepoint')

    print('Reference:')
    npm = np.array([[9.52552396e-01, 0.00000000e+00, 9.36786317e-05],
                    [3.43966450e-01, 7.28166097e-01, -7.21325464e-02],
                    [0.00000000e+00, 0.00000000e+00, 1.00882518e+00]])
    print(primaries_whitepoint(npm))

    print('\n')

    print('Refactor:')
    print(primaries_whitepoint_vectorise(npm))

    print('\n')


# primaries_whitepoint_analysis()

# #############################################################################
# # ### colour.RGB_luminance
# #############################################################################


def RGB_luminance_2d(RGB):
    for i in range(len(RGB)):
        RGB_luminance(RGB[i],
                      np.array([0.73470, 0.26530,
                                0.00000, 1.00000,
                                0.00010, -0.07700]),
                      (0.32168, 0.33767))


def RGB_luminance_vectorise(RGB, primaries, whitepoint):
    R, G, B = tsplit(RGB)

    X, Y, Z = np.ravel(normalised_primary_matrix_vectorise(primaries,
                                                           whitepoint))[3:6]
    L = X * R + Y * G + Z * B

    return L


def RGB_luminance_analysis():
    message_box('RGB_luminance')

    print('Reference:')
    RGB = np.array([40.6, 4.2, 67.4])
    P = np.array([0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700])
    W = (0.32168, 0.33767)
    print(RGB_luminance(RGB, P, W))

    print('\n')

    print('1d array input:')
    print(RGB_luminance_vectorise(RGB, P, W))

    print('\n')

    print('2d array input:')
    RGB = np.tile(RGB, (6, 1))
    print(RGB_luminance_vectorise(RGB, P, W))

    print('\n')

    print('3d array input:')
    RGB = np.reshape(RGB, (2, 3, 3))
    print(RGB)
    print(RGB_luminance_vectorise(RGB, P, W))

    print('\n')


# RGB_luminance_analysis()


def RGB_luminance_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    P = np.array([0.73470, 0.26530,
                  0.00000, 1.00000,
                  0.00010, -0.07700]),
    W = (0.32168, 0.33767)

    times = timeit.Timer(
        functools.partial(
            RGB_luminance_2d,
            DATA_HD1)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            RGB_luminance_vectorise,
            DATA_HD1, P, W)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('RGB_luminance\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# RGB_luminance_profile()

# #############################################################################
# #############################################################################
# ### colour.models.ipt
# #############################################################################
# #############################################################################

# #############################################################################
# ### colour.XYZ_to_IPT
# #############################################################################
from colour.models.ipt import *


def XYZ_to_IPT_2d(XYZ):
    for i in range(len(XYZ)):
        XYZ_to_IPT(XYZ[i])


def XYZ_to_IPT_vectorise(XYZ):
    LMS = np.einsum('...ij,...j->...i', IPT_XYZ_TO_LMS_MATRIX, XYZ)
    LMS_prime = np.sign(LMS) * np.abs(LMS) ** 0.43
    IPT = np.einsum('...ij,...j->...i', IPT_LMS_TO_IPT_MATRIX, LMS_prime)

    return IPT


def XYZ_to_IPT_analysis():
    message_box('XYZ_to_IPT')

    print('Reference:')
    XYZ = np.array([0.96907232, 1, 1.12179215])
    print(XYZ_to_IPT(XYZ))

    print('\n')

    print('1d array input:')
    print(XYZ_to_IPT_vectorise(XYZ))

    print('\n')

    print('2d array input:')
    XYZ = np.tile(XYZ, (6, 1))
    print(XYZ_to_IPT_vectorise(XYZ))

    print('\n')

    print('3d array input:')
    XYZ = np.reshape(XYZ, (2, 3, 3))
    print(XYZ_to_IPT_vectorise(XYZ))

    print('\n')


# XYZ_to_IPT_analysis()


def XYZ_to_IPT_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            XYZ_to_IPT_2d,
            DATA_HD1)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            XYZ_to_IPT_vectorise,
            DATA_HD1)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('XYZ_to_IPT\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# XYZ_to_IPT_profile()

# #############################################################################
# #############################################################################
# ### colour.IPT_to_XYZ
# #############################################################################
# #############################################################################


def IPT_to_XYZ_2d(IPT):
    for i in range(len(IPT)):
        IPT_to_XYZ(IPT[i])


def IPT_to_XYZ_vectorise(IPT):
    LMS = np.einsum('...ij,...j->...i', IPT_IPT_TO_LMS_MATRIX, IPT)
    LMS_prime = np.sign(LMS) * np.abs(LMS) ** (1 / 0.43)
    XYZ = np.einsum('...ij,...j->...i', IPT_LMS_TO_XYZ_MATRIX, LMS_prime)

    return XYZ


def IPT_to_XYZ_analysis():
    message_box('IPT_to_XYZ')

    print('Reference:')
    IPT = np.array([1.00300825, 0.01906918, -0.01369292])
    print(IPT_to_XYZ(IPT))

    print('\n')

    print('1d array input:')
    print(IPT_to_XYZ_vectorise(IPT))

    print('\n')

    print('2d array input:')
    IPT = np.tile(IPT, (6, 1))
    print(IPT_to_XYZ_vectorise(IPT))

    print('\n')

    print('3d array input:')
    IPT = np.reshape(IPT, (2, 3, 3))
    print(IPT_to_XYZ_vectorise(IPT))

    print('\n')


# IPT_to_XYZ_analysis()


def IPT_to_XYZ_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            IPT_to_XYZ_2d,
            DATA_HD1)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            IPT_to_XYZ_vectorise,
            DATA_HD1)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('IPT_to_XYZ\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# IPT_to_XYZ_profile()

# #############################################################################
# ### colour.IPT_hue_angle
# #############################################################################


def IPT_hue_angle_2d(IPT):
    for i in range(len(IPT)):
        IPT_hue_angle(IPT[i])


def IPT_hue_angle_vectorise(IPT):
    I, P, T = tsplit(IPT)

    hue = np.arctan2(T, P)

    return hue


def IPT_hue_angle_analysis():
    message_box('IPT_hue_angle')

    print('Reference:')
    IPT = np.array([0.96907232, 1., 1.12179215])
    print(IPT_hue_angle(IPT))

    print('\n')

    print('1d array input:')
    print(IPT_hue_angle_vectorise(IPT))

    print('\n')

    print('2d array input:')
    IPT = np.tile(IPT, (6, 1))
    print(IPT_hue_angle_vectorise(IPT))

    print('\n')

    print('3d array input:')
    IPT = np.reshape(IPT, (2, 3, 3))
    print(IPT_hue_angle_vectorise(IPT))

    print('\n')


# IPT_hue_angle_analysis()


def IPT_hue_angle_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            IPT_hue_angle_2d,
            DATA_HD1)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            IPT_hue_angle_vectorise,
            DATA_HD1)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('IPT_hue_angle\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# IPT_hue_angle_profile()

# #############################################################################
# #############################################################################
# ### colour.models.log
# #############################################################################
# #############################################################################

# #############################################################################
# ### colour.linear_to_cineon
# #############################################################################
from colour.models.log import *


DATA = np.linspace(0, 1, 1000000)


def linear_to_cineon_2d(value):
    for i in range(len(value)):
        linear_to_cineon(value[i])


def linear_to_cineon_vectorise(value,
                               black_offset=10 ** ((95 - 685) / 300),
                               **kwargs):
    value = np.asarray(value)

    return ((685 + 300 *
             np.log10(value * (1 - black_offset) + black_offset)) / 1023)


def linear_to_cineon_analysis():
    message_box('linear_to_cineon')

    print('Reference:')
    print(linear_to_cineon(0.18))

    print('\n')

    print('Numeric input:')
    print(linear_to_cineon_vectorise(0.18))

    print('\n')

    print('0d array input:')
    print(linear_to_cineon_vectorise(np.array(0.18)))

    print('\n')

    print('1d array input:')
    linear = [0.18] * 6
    print(linear_to_cineon_vectorise(linear))

    print('\n')

    print('2d array input:')
    linear = np.reshape(linear, (2, 3))
    print(linear_to_cineon_vectorise(linear))

    print('\n')

    print('3d array input:')
    linear = np.reshape(linear, (2, 3, 1))
    print(linear_to_cineon_vectorise(linear))

    print('\n')


# linear_to_cineon_analysis()


def linear_to_cineon_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            linear_to_cineon_2d,
            DATA_HD1[..., 0])).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            linear_to_cineon_vectorise,
            DATA_HD1[..., 0])).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('linear_to_cineon\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1[..., 0]), a, b))


# linear_to_cineon_profile()

# #############################################################################
# ### colour.cineon_to_linear
# #############################################################################


def cineon_to_linear_2d(value):
    for i in range(len(value)):
        cineon_to_linear(value[i])


def cineon_to_linear_vectorise(value,
                               black_offset=10 ** ((95 - 685) / 300),
                               **kwargs):
    value = np.asarray(value)

    return ((10 ** ((1023 * value - 685) / 300) - black_offset) /
            (1 - black_offset))


def cineon_to_linear_analysis():
    message_box('cineon_to_linear')

    print('Reference:')
    print(cineon_to_linear(0.5))

    print('\n')

    print('Numeric input:')
    print(cineon_to_linear_vectorise(0.5))

    print('\n')

    print('0d array input:')
    print(cineon_to_linear_vectorise(np.array(0.5)))

    print('\n')

    print('1d array input:')
    log = [0.5] * 6
    print(cineon_to_linear_vectorise(log))

    print('\n')

    print('2d array input:')
    log = np.reshape(log, (2, 3))
    print(cineon_to_linear_vectorise(log))

    print('\n')

    print('3d array input:')
    log = np.reshape(log, (2, 3, 1))
    print(cineon_to_linear_vectorise(log))

    print('\n')


# cineon_to_linear_analysis()


def cineon_to_linear_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            cineon_to_linear_2d,
            DATA_HD1[..., 0])).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            cineon_to_linear_vectorise,
            DATA_HD1[..., 0])).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('cineon_to_linear\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1[..., 0]), a, b))


# cineon_to_linear_profile()

# #############################################################################
# ### colour.linear_to_panalog
# #############################################################################


def linear_to_panalog_2d(value):
    for i in range(len(value)):
        linear_to_panalog(value[i])


def linear_to_panalog_vectorise(value,
                                black_offset=10 ** ((64 - 681) / 444),
                                **kwargs):
    value = np.asarray(value)

    return ((681 + 444 *
             np.log10(value * (1 - black_offset) + black_offset)) / 1023)


def linear_to_panalog_analysis():
    message_box('linear_to_panalog')

    print('Reference:')
    print(linear_to_panalog(0.18))

    print('\n')

    print('Numeric input:')
    print(linear_to_panalog_vectorise(0.18))

    print('\n')

    print('0d array input:')
    print(linear_to_panalog_vectorise(np.array(0.18)))

    print('\n')

    print('1d array input:')
    linear = [0.18] * 6
    print(linear_to_panalog_vectorise(linear))

    print('\n')

    print('2d array input:')
    linear = np.reshape(linear, (2, 3))
    print(linear_to_panalog_vectorise(linear))

    print('\n')

    print('3d array input:')
    linear = np.reshape(linear, (2, 3, 1))
    print(linear_to_panalog_vectorise(linear))

    print('\n')


# linear_to_panalog_analysis()


def linear_to_panalog_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            linear_to_panalog_2d,
            DATA_HD1[..., 0])).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            linear_to_panalog_vectorise,
            DATA_HD1[..., 0])).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('linear_to_panalog\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1[..., 0]), a, b))


# linear_to_panalog_profile()

# #############################################################################
# ### colour.panalog_to_linear
# #############################################################################


def panalog_to_linear_2d(value):
    for i in range(len(value)):
        panalog_to_linear(value[i])


def panalog_to_linear_vectorise(value,
                                black_offset=10 ** ((64 - 681) / 444),
                                **kwargs):
    value = np.asarray(value)

    return ((10 ** ((1023 * value - 681) / 444) - black_offset) /
            (1 - black_offset))


def panalog_to_linear_analysis():
    message_box('panalog_to_linear')

    print('Reference:')
    print(panalog_to_linear(0.5))

    print('\n')

    print('Numeric input:')
    print(panalog_to_linear_vectorise(0.5))

    print('\n')

    print('0d array input:')
    print(panalog_to_linear_vectorise(np.array(0.5)))

    print('\n')

    print('1d array input:')
    log = [0.5] * 6
    print(panalog_to_linear_vectorise(log))

    print('\n')

    print('2d array input:')
    log = np.reshape(log, (2, 3))
    print(panalog_to_linear_vectorise(log))

    print('\n')

    print('3d array input:')
    log = np.reshape(log, (2, 3, 1))
    print(panalog_to_linear_vectorise(log))

    print('\n')


# panalog_to_linear_analysis()


def panalog_to_linear_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            panalog_to_linear_2d,
            DATA_HD1[..., 0])).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            panalog_to_linear_vectorise,
            DATA_HD1[..., 0])).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('panalog_to_linear\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1[..., 0]), a, b))


# panalog_to_linear_profile()

# #############################################################################
# ### colour.linear_to_red_log
# #############################################################################


def linear_to_red_log_2d(value):
    for i in range(len(value)):
        linear_to_red_log(value[i])


def linear_to_red_log_vectorise(value,
                                black_offset=10 ** ((0 - 1023) / 511),
                                **kwargs):
    value = np.asarray(value)

    return ((1023 +
             511 * np.log10(value * (1 - black_offset) + black_offset)) / 1023)


def linear_to_red_log_analysis():
    message_box('linear_to_red_log')

    print('Reference:')
    print(linear_to_red_log(0.18))

    print('\n')

    print('Numeric input:')
    print(linear_to_red_log_vectorise(0.18))

    print('\n')

    print('0d array input:')
    print(linear_to_red_log_vectorise(np.array(0.18)))

    print('\n')

    print('1d array input:')
    linear = [0.18] * 6
    print(linear_to_red_log_vectorise(linear))

    print('\n')

    print('2d array input:')
    linear = np.reshape(linear, (2, 3))
    print(linear_to_red_log_vectorise(linear))

    print('\n')

    print('3d array input:')
    linear = np.reshape(linear, (2, 3, 1))
    print(linear_to_red_log_vectorise(linear))

    print('\n')


# linear_to_red_log_analysis()


def linear_to_red_log_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            linear_to_red_log_2d,
            DATA_HD1[..., 0])).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            linear_to_red_log_vectorise,
            DATA_HD1[..., 0])).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('linear_to_red_log\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1[..., 0]), a, b))


# linear_to_red_log_profile()

# #############################################################################
# ### colour.red_log_to_linear
# #############################################################################


def red_log_to_linear_2d(value):
    for i in range(len(value)):
        red_log_to_linear(value[i])


def red_log_to_linear_vectorise(value,
                                black_offset=10 ** ((0 - 1023) / 511),
                                **kwargs):
    value = np.asarray(value)

    return (((10 **
              ((1023 * value - 1023) / 511)) - black_offset) /
            (1 - black_offset))


def red_log_to_linear_analysis():
    message_box('red_log_to_linear')

    print('Reference:')
    print(red_log_to_linear(0.5))

    print('\n')

    print('Numeric input:')
    print(red_log_to_linear_vectorise(0.5))

    print('\n')

    print('1d array input:')
    print(red_log_to_linear_vectorise(np.array(0.5)))

    print('\n')

    print('1d array input:')
    log = [0.5] * 6
    print(red_log_to_linear_vectorise(log))

    print('\n')

    print('2d array input:')
    log = np.reshape(log, (2, 3))
    print(red_log_to_linear_vectorise(log))

    print('\n')

    print('3d array input:')
    log = np.reshape(log, (2, 3, 1))
    print(red_log_to_linear_vectorise(log))

    print('\n')


# red_log_to_linear_analysis()


def red_log_to_linear_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            red_log_to_linear_2d,
            DATA_HD1[..., 0])).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            red_log_to_linear_vectorise,
            DATA_HD1[..., 0])).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('red_log_to_linear\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1[..., 0]), a, b))


# red_log_to_linear_profile()

# #############################################################################
# ### colour.linear_to_viper_log
# #############################################################################


def linear_to_viper_log_2d(value):
    for i in range(len(value)):
        linear_to_viper_log(value[i])


def linear_to_viper_log_vectorise(value, **kwargs):
    value = np.asarray(value)

    return (1023 + 500 * np.log10(value)) / 1023


def linear_to_viper_log_analysis():
    message_box('linear_to_viper_log')

    print('Reference:')
    print(linear_to_viper_log(0.18))

    print('\n')

    print('Numeric input:')
    print(linear_to_viper_log_vectorise(0.18))

    print('\n')

    print('1d array input:')
    print(linear_to_viper_log_vectorise(np.array(0.18)))

    print('\n')

    print('1d array input:')
    linear = [0.18] * 6
    print(linear_to_viper_log_vectorise(linear))

    print('\n')

    print('2d array input:')
    linear = np.reshape(linear, (2, 3))
    print(linear_to_viper_log_vectorise(linear))

    print('\n')

    print('3d array input:')
    linear = np.reshape(linear, (2, 3, 1))
    print(linear_to_viper_log_vectorise(linear))

    print('\n')


# linear_to_viper_log_analysis()


def linear_to_viper_log_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            linear_to_viper_log_2d,
            DATA_HD1[..., 0])).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            linear_to_viper_log_vectorise,
            DATA_HD1[..., 0])).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('linear_to_viper_log\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1[..., 0]), a, b))


# linear_to_viper_log_profile()

# #############################################################################
# ### colour.viper_log_to_linear
# #############################################################################


def viper_log_to_linear_2d(value):
    for i in range(len(value)):
        viper_log_to_linear(value[i])


def viper_log_to_linear_vectorise(value, **kwargs):
    value = np.asarray(value)

    return 10 ** ((1023 * value - 1023) / 500)


def viper_log_to_linear_analysis():
    message_box('viper_log_to_linear')

    print('Reference:')
    print(viper_log_to_linear(0.5))

    print('\n')

    print('Numeric input:')
    print(viper_log_to_linear_vectorise(0.5))

    print('\n')

    print('0d array input:')
    print(viper_log_to_linear_vectorise(np.array(0.5)))

    print('\n')

    print('1d array input:')
    log = [0.5] * 6
    print(viper_log_to_linear_vectorise(log))

    print('\n')

    print('2d array input:')
    log = np.reshape(log, (2, 3))
    print(viper_log_to_linear_vectorise(log))

    print('\n')

    print('3d array input:')
    log = np.reshape(log, (2, 3, 1))
    print(viper_log_to_linear_vectorise(log))

    print('\n')


# viper_log_to_linear_analysis()


def viper_log_to_linear_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            viper_log_to_linear_2d,
            DATA_HD1[..., 0])).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            viper_log_to_linear_vectorise,
            DATA_HD1[..., 0])).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('viper_log_to_linear\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1[..., 0]), a, b))


# viper_log_to_linear_profile()

# #############################################################################
# ### colour.linear_to_pivoted_log
# #############################################################################


def linear_to_pivoted_log_2d(value):
    for i in range(len(value)):
        linear_to_pivoted_log(value[i])


def linear_to_pivoted_log_vectorise(value,
                                    log_reference=445,
                                    linear_reference=0.18,
                                    negative_gamma=0.6,
                                    density_per_code_value=0.002):
    value = np.asarray(value)

    return ((log_reference + np.log10(value / linear_reference) /
             (density_per_code_value / negative_gamma)) / 1023)


def linear_to_pivoted_log_analysis():
    message_box('linear_to_pivoted_log')

    print('Reference:')
    print(linear_to_pivoted_log(0.18))

    print('\n')

    print('Numeric input:')
    print(linear_to_pivoted_log_vectorise(0.18))

    print('\n')

    print('0d array input:')
    print(linear_to_pivoted_log_vectorise(np.array(0.18)))

    print('\n')

    print('1d array input:')
    linear = [0.18] * 6
    print(linear_to_pivoted_log_vectorise(linear))

    print('\n')

    print('2d array input:')
    linear = np.reshape(linear, (2, 3))
    print(linear_to_pivoted_log_vectorise(linear))

    print('\n')

    print('3d array input:')
    linear = np.reshape(linear, (2, 3, 1))
    print(linear_to_pivoted_log_vectorise(linear))

    print('\n')


# linear_to_pivoted_log_analysis()

def linear_to_pivoted_log_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            linear_to_pivoted_log_2d,
            DATA_HD1[..., 0])).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            linear_to_pivoted_log_vectorise,
            DATA_HD1[..., 0])).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('linear_to_pivoted_log\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1[..., 0]), a, b))


# linear_to_pivoted_log_profile()

# #############################################################################
# ### colour.pivoted_log_to_linear
# #############################################################################


def pivoted_log_to_linear_2d(value):
    for i in range(len(value)):
        pivoted_log_to_linear(value[i])


def pivoted_log_to_linear_vectorise(value,
                                    log_reference=445,
                                    linear_reference=0.18,
                                    negative_gamma=0.6,
                                    density_per_code_value=0.002):
    value = np.asarray(value)

    return (10 ** ((value * 1023 - log_reference) *
                   (density_per_code_value / negative_gamma)) *
            linear_reference)


def pivoted_log_to_linear_analysis():
    message_box('pivoted_log_to_linear')

    print('Reference:')
    print(pivoted_log_to_linear(0.5))

    print('\n')

    print('Numeric input:')
    print(pivoted_log_to_linear_vectorise(0.5))

    print('\n')

    print('0d array input:')
    print(pivoted_log_to_linear_vectorise(np.array(0.5)))

    print('\n')

    print('1d array input:')
    log = [0.5] * 6
    print(pivoted_log_to_linear_vectorise(log))

    print('\n')

    print('2d array input:')
    log = np.reshape(log, (2, 3))
    print(pivoted_log_to_linear_vectorise(log))

    print('\n')

    print('3d array input:')
    log = np.reshape(log, (2, 3, 1))
    print(pivoted_log_to_linear_vectorise(log))

    print('\n')


# pivoted_log_to_linear_analysis()


def pivoted_log_to_linear_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            pivoted_log_to_linear_2d,
            DATA_HD1[..., 0])).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            pivoted_log_to_linear_vectorise,
            DATA_HD1[..., 0])).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('pivoted_log_to_linear\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1[..., 0]), a, b))


# pivoted_log_to_linear_profile()

# #############################################################################
# ### colour.linear_to_c_log
# #############################################################################


def linear_to_c_log_2d(value):
    for i in range(len(value)):
        linear_to_c_log(value[i])


def linear_to_c_log_vectorise(value, **kwargs):
    value = np.asarray(value)

    return 0.529136 * np.log10(10.1596 * value + 1) + 0.0730597


def linear_to_c_log_analysis():
    message_box('linear_to_c_log')

    print('Reference:')
    print(linear_to_c_log(0.18))

    print('\n')

    print('Numeric input:')
    print(linear_to_c_log_vectorise(0.18))

    print('\n')

    print('0d array input:')
    print(linear_to_c_log_vectorise(np.array(0.18)))

    print('\n')

    print('1d array input:')
    linear = [0.18] * 6
    print(linear_to_c_log_vectorise(linear))

    print('\n')

    print('2d array input:')
    linear = np.reshape(linear, (2, 3))
    print(linear_to_c_log_vectorise(linear))

    print('\n')

    print('3d array input:')
    linear = np.reshape(linear, (2, 3, 1))
    print(linear_to_c_log_vectorise(linear))

    print('\n')


# linear_to_c_log_analysis()


def linear_to_c_log_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            linear_to_c_log_2d,
            DATA_HD1[..., 0])).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            linear_to_c_log_vectorise,
            DATA_HD1[..., 0])).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('linear_to_c_log\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1[..., 0]), a, b))


# linear_to_c_log_profile()

# #############################################################################
# ### colour.c_log_to_linear
# #############################################################################


def c_log_to_linear_2d(value):
    for i in range(len(value)):
        c_log_to_linear(value[i])


def c_log_to_linear_vectorise(value, **kwargs):
    value = np.asarray(value)

    return (-0.071622555735168 *
            (1.3742747797867 - np.exp(1) ** (4.3515940948906 * value)))


def c_log_to_linear_analysis():
    message_box('c_log_to_linear')

    print('Reference:')
    print(c_log_to_linear(0.5))

    print('\n')

    print('Numeric input:')
    print(c_log_to_linear_vectorise(0.5))

    print('\n')

    print('0d array input:')
    print(c_log_to_linear_vectorise(np.array(0.5)))

    print('\n')

    print('1d array input:')
    log = [0.5] * 6
    print(c_log_to_linear_vectorise(log))

    print('\n')

    print('2d array input:')
    log = np.reshape(log, (2, 3))
    print(c_log_to_linear_vectorise(log))

    print('\n')

    print('3d array input:')
    log = np.reshape(log, (2, 3, 1))
    print(c_log_to_linear_vectorise(log))

    print('\n')


# c_log_to_linear_analysis()


def c_log_to_linear_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            c_log_to_linear_2d,
            DATA_HD1[..., 0])).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            c_log_to_linear_vectorise,
            DATA_HD1[..., 0])).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('c_log_to_linear\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1[..., 0]), a, b))


# c_log_to_linear_profile()

# #############################################################################
# #############################################################################
# ### colour.models.rgb
# #############################################################################
# #############################################################################

# #############################################################################
# ### OECF / OECF_i
# #############################################################################
from colour.models.dataset.aces import (
    _aces_cc_transfer_function,
    _aces_cc_inverse_transfer_function)
from colour.models.dataset.aces import *


RGB = np.array([0.86969452, 1.00516431, 1.41715848])
RGB_t = np.tile(RGB, (6, 1)).reshape(2, 3, 3)


def _aces_cc_transfer_function_vectorise(value):
    value = np.asarray(value)

    output = np.where(value < 0,
                      (np.log2(2 ** -15 * 0.5) + 9.72) / 17.52,
                      (np.log2(2 ** -16 + value * 0.5) + 9.72) / 17.52)
    output = np.where(value >= 2 ** -15,
                      (np.log2(value) + 9.72) / 17.52,
                      output)

    return output


def _aces_cc_transfer_function_analysis():
    message_box('_aces_cc_transfer_function')

    print(_aces_cc_transfer_function(RGB[0]))

    print(_aces_cc_transfer_function_vectorise(RGB[0]))

    print(_aces_cc_transfer_function_vectorise(RGB))

    print(_aces_cc_transfer_function_vectorise(RGB_t))

    print('\n')

    np.testing.assert_almost_equal(
        [_aces_cc_transfer_function(x) for x in np.ravel(DATA1)],
        np.ravel(_aces_cc_transfer_function_vectorise(DATA1)))


# _aces_cc_transfer_function_analysis()


def _aces_cc_inverse_transfer_function_vectorise(value):
    value = np.asarray(value)

    output = np.where(value < (9.72 - 15) / 17.52,
                      (2 ** (value * 17.52 - 9.72) - 2 ** -16) * 2,
                      2 ** (value * 17.52 - 9.72))
    output = np.where(value >= (np.log2(65504) + 9.72) / 17.52,
                      65504,
                      output)

    return output


def _aces_cc_inverse_transfer_function_analysis():
    message_box('_aces_cc_inverse_transfer_function')

    print(_aces_cc_inverse_transfer_function(RGB[0]))

    print(_aces_cc_inverse_transfer_function_vectorise(RGB[0]))

    print(_aces_cc_inverse_transfer_function_vectorise(RGB))

    print(_aces_cc_inverse_transfer_function_vectorise(RGB_t))

    print('\n')

    np.testing.assert_almost_equal(
        [_aces_cc_inverse_transfer_function(x) for x in np.ravel(DATA1)],
        np.ravel(_aces_cc_inverse_transfer_function_vectorise(DATA1)))


# _aces_cc_inverse_transfer_function_analysis()

from colour.models.dataset.aces import (
    _aces_proxy_transfer_function,
    _aces_proxy_inverse_transfer_function)


def _aces_proxy_transfer_function_vectorise(value, bit_depth='10 Bit'):
    value = np.asarray(value)

    constants = ACES_PROXY_CONSTANTS.get(bit_depth)

    CV_min = np.resize(constants.CV_min, value.shape)
    CV_max = np.resize(constants.CV_max, value.shape)

    float_2_cv = lambda x: np.maximum(CV_min, np.minimum(CV_max, np.round(x)))

    output = np.where(value > 2 ** -9.72,
                      float_2_cv((np.log2(value) + constants.mid_log_offset) *
                                 constants.steps_per_stop + constants.mid_CV_offset),
                      np.resize(CV_min, value.shape))
    return output


def _aces_proxy_transfer_function_analysis():
    message_box('_aces_proxy_transfer_function')

    print(_aces_proxy_transfer_function(RGB[0]))

    print(_aces_proxy_transfer_function_vectorise(RGB[0]))

    print(_aces_proxy_transfer_function_vectorise(RGB))

    print(_aces_proxy_transfer_function_vectorise(RGB_t))

    print('\n')

    np.testing.assert_almost_equal(
        [_aces_proxy_transfer_function(x) for x in np.ravel(DATA1)],
        np.ravel(_aces_proxy_transfer_function_vectorise(DATA1)))


# _aces_proxy_transfer_function_analysis()


def _aces_proxy_inverse_transfer_function_vectorise(value, bit_depth='10 Bit'):
    value = np.asarray(value)

    constants = ACES_PROXY_CONSTANTS.get(bit_depth)

    return (2 ** (((value - constants.mid_CV_offset) /
                   constants.steps_per_stop - constants.mid_log_offset)))


def _aces_proxy_inverse_transfer_function_analysis():
    message_box('_aces_proxy_inverse_transfer_function')

    print(_aces_proxy_inverse_transfer_function(RGB[0]))

    print(_aces_proxy_inverse_transfer_function_vectorise(RGB[0]))

    print(_aces_proxy_inverse_transfer_function_vectorise(RGB))

    print(_aces_proxy_inverse_transfer_function_vectorise(RGB_t))

    print('\t')


# _aces_proxy_inverse_transfer_function_analysis()

from colour.models.dataset.adobe_rgb_1998 import (
    _adobe_rgb_1998_transfer_function,
    _adobe_rgb_1998_inverse_transfer_function)
from colour.models.dataset.adobe_rgb_1998 import *


def _adobe_rgb_1998_transfer_function_vectorise(value):
    # Also valid for:
    # _adobe_wide_gamut_rgb_transfer_function
    value = np.asarray(value)

    return value ** (1 / (563 / 256))


def _adobe_rgb_1998_transfer_function_analysis():
    message_box('_adobe_rgb_1998_transfer_function')

    print(_adobe_rgb_1998_transfer_function(RGB[0]))

    print(_adobe_rgb_1998_transfer_function_vectorise(RGB[0]))

    print(_adobe_rgb_1998_transfer_function_vectorise(RGB))

    print(_adobe_rgb_1998_transfer_function_vectorise(RGB_t))

    print('\t')


# _adobe_rgb_1998_transfer_function_analysis()


def _adobe_rgb_1998_inverse_transfer_function_vectorise(value):
    # Also valid for:
    # _adobe_wide_gamut_rgb_inverse_transfer_function
    value = np.asarray(value)

    return value ** (563 / 256)


def _adobe_rgb_1998_inverse_transfer_function_analysis():
    message_box('_adobe_rgb_1998_inverse_transfer_function')

    print(_adobe_rgb_1998_inverse_transfer_function(RGB[0]))

    print(_adobe_rgb_1998_inverse_transfer_function_vectorise(RGB[0]))

    print(_adobe_rgb_1998_inverse_transfer_function_vectorise(RGB))

    print(_adobe_rgb_1998_inverse_transfer_function_vectorise(RGB_t))

    print('\n')


# _adobe_rgb_1998_inverse_transfer_function_analysis()

from colour.models.dataset.alexa_wide_gamut_rgb import (
    _alexa_wide_gamut_rgb_transfer_function,
    _alexa_wide_gamut_rgb_inverse_transfer_function)
from colour.models.dataset.alexa_wide_gamut_rgb import *


def _alexa_wide_gamut_rgb_transfer_function_vectorise(
        value,
        firmware='SUP 3.x',
        method='Linear Scene Exposure Factor',
        EI=800):
    value = np.asarray(value)

    cut, a, b, c, d, e, f, _ = ALEXA_LOG_C_CURVE_CONVERSION_DATA.get(
        firmware).get(method).get(EI)

    return np.where(value > cut,
                    c * np.log10(a * value + b) + d,
                    e * value + f)


def _alexa_wide_gamut_rgb_transfer_function_analysis():
    message_box('_alexa_wide_gamut_rgb_transfer_function')

    print(_alexa_wide_gamut_rgb_transfer_function(RGB[0]))

    print(_alexa_wide_gamut_rgb_transfer_function_vectorise(RGB[0]))

    print(_alexa_wide_gamut_rgb_transfer_function_vectorise(RGB))

    print(_alexa_wide_gamut_rgb_transfer_function_vectorise(RGB_t))

    print('\n')

    np.testing.assert_almost_equal(
        [_alexa_wide_gamut_rgb_transfer_function(x) for x in np.ravel(DATA1)],
        np.ravel(_alexa_wide_gamut_rgb_transfer_function_vectorise(DATA1)))


# _alexa_wide_gamut_rgb_transfer_function_analysis()


def _alexa_wide_gamut_rgb_inverse_transfer_function_vectorise(
        value,
        firmware='SUP 3.x',
        method='Linear Scene Exposure Factor',
        EI=800):
    value = np.asarray(value)

    cut, a, b, c, d, e, f, _ = (
        ALEXA_LOG_C_CURVE_CONVERSION_DATA.get(firmware).get(method).get(EI))

    return np.where(value > e * cut + f,
                    (np.power(10., (value - d) / c) - b) / a,
                    (value - f) / e)


def _alexa_wide_gamut_rgb_inverse_transfer_function_analysis():
    message_box('_alexa_wide_gamut_rgb_inverse_transfer_function')

    print(_alexa_wide_gamut_rgb_inverse_transfer_function(RGB[0]))

    print(_alexa_wide_gamut_rgb_inverse_transfer_function_vectorise(RGB[0]))

    print(_alexa_wide_gamut_rgb_inverse_transfer_function_vectorise(RGB))

    print(_alexa_wide_gamut_rgb_inverse_transfer_function_vectorise(RGB_t))

    print('\n')

    np.testing.assert_almost_equal(
        [_alexa_wide_gamut_rgb_inverse_transfer_function(x) for x in
         np.ravel(DATA1)],
        np.ravel(
            _alexa_wide_gamut_rgb_inverse_transfer_function_vectorise(DATA1)))


# _alexa_wide_gamut_rgb_inverse_transfer_function_analysis()

from colour.models.dataset.apple_rgb import (
    _apple_rgb_transfer_function,
    _apple_rgb_inverse_transfer_function)
from colour.models.dataset.apple_rgb import *


def _apple_rgb_transfer_function_vectorise(value):
    # Also valid for:
    # _color_match_rgb_transfer_function
    value = np.asarray(value)

    return value ** (1 / 1.8)


def _apple_rgb_transfer_function_function_analysis():
    message_box('_apple_rgb_transfer_function')

    print(_apple_rgb_transfer_function(RGB[0]))

    print(_apple_rgb_transfer_function_vectorise(RGB[0]))

    print(_apple_rgb_transfer_function_vectorise(RGB))

    print(_apple_rgb_transfer_function_vectorise(RGB_t))

    print('\n')


# _apple_rgb_transfer_function_function_analysis()


def _apple_rgb_inverse_transfer_function_vectorise(value):
    # Also valid for:
    # _color_match_rgb_inverse_transfer_function
    value = np.asarray(value)

    return value ** 1.8


def _apple_rgb_inverse_transfer_function_analysis():
    message_box('_apple_rgb_inverse_transfer_function')

    print(_apple_rgb_inverse_transfer_function(RGB[0]))

    print(_apple_rgb_inverse_transfer_function_vectorise(RGB[0]))

    print(_apple_rgb_inverse_transfer_function_vectorise(RGB))

    print(_apple_rgb_inverse_transfer_function_vectorise(RGB_t))

    print('\n')


# _apple_rgb_inverse_transfer_function_analysis()

from colour.models.dataset.best_rgb import (
    _best_rgb_transfer_function,
    _best_rgb_inverse_transfer_function)
from colour.models.dataset.best_rgb import *


def _best_rgb_transfer_function_vectorise(value):
    # Also valid for:
    # _beta_rgb_transfer_function
    # _cie_rgb_transfer_function
    # _don_rgb_4_transfer_function
    # _ekta_space_ps_5_transfer_function
    # _max_rgb_transfer_function
    # _ntsc_rgb_transfer_function
    # _russell_rgb_transfer_function
    # _smpte_c_rgb_transfer_function
    # _xtreme_rgb_transfer_function
    value = np.asarray(value)

    return value ** (1 / 2.2)


def _best_rgb_transfer_function_analysis():
    message_box('_best_rgb_transfer_function')

    print(_best_rgb_transfer_function(RGB[0]))

    print(_best_rgb_transfer_function_vectorise(RGB[0]))

    print(_best_rgb_transfer_function_vectorise(RGB))

    print(_best_rgb_transfer_function_vectorise(RGB_t))

    print('\n')


# _best_rgb_transfer_function_analysis()


def _best_rgb_inverse_transfer_function_vectorise(value):
    # Also valid for:
    # _beta_rgb_inverse_transfer_function
    # _cie_rgb_inverse_transfer_function
    # _don_rgb_4_inverse_transfer_function
    # _ekta_space_ps_5_inverse_transfer_function
    # _max_rgb_inverse_transfer_function
    # _ntsc_rgb_inverse_transfer_function
    # _russell_rgb_inverse_transfer_function
    # _smpte_c_rgb_inverse_transfer_function
    # _xtreme_rgb_inverse_transfer_function
    value = np.asarray(value)

    return value ** 2.2


def _best_rgb_inverse_transfer_function_analysis():
    message_box('_best_rgb_inverse_transfer_function')

    print(_best_rgb_inverse_transfer_function(RGB[0]))

    print(_best_rgb_inverse_transfer_function_vectorise(RGB[0]))

    print(_best_rgb_inverse_transfer_function_vectorise(RGB))

    print(_best_rgb_inverse_transfer_function_vectorise(RGB_t))

    print('\n')


# _best_rgb_inverse_transfer_function_analysis()

from colour.models.dataset.dci_p3 import (
    _dci_p3_transfer_function,
    _dci_p3_inverse_transfer_function)
from colour.models.dataset.dci_p3 import *


def _dci_p3_transfer_function_vectorise(value):
    value = np.asarray(value)

    return 4095 * (value / 52.37) ** (1 / 2.6)


def _dci_p3_transfer_function_analysis():
    message_box('_dci_p3_transfer_function')

    print(_dci_p3_transfer_function(RGB[0]))

    print(_dci_p3_transfer_function_vectorise(RGB[0]))

    print(_dci_p3_transfer_function_vectorise(RGB))

    print(_dci_p3_transfer_function_vectorise(RGB_t))

    print('\n')


# _dci_p3_transfer_function_analysis()


def _dci_p3_inverse_transfer_function_vectorise(value):
    value = np.asarray(value)

    return 52.37 * (value / 4095) ** 2.6


def _dci_p3_inverse_transfer_function_analysis():
    message_box('_dci_p3_inverse_transfer_function')

    print(_dci_p3_inverse_transfer_function(RGB[0]))

    print(_dci_p3_inverse_transfer_function_vectorise(RGB[0]))

    print(_dci_p3_inverse_transfer_function_vectorise(RGB))

    print(_dci_p3_inverse_transfer_function_vectorise(RGB_t))

    print('\n')


# _dci_p3_inverse_transfer_function_analysis()

from colour.models.dataset.pal_secam_rgb import (
    _pal_secam_rgb_transfer_function,
    _pal_secam_rgb_inverse_transfer_function)
from colour.models.dataset.pal_secam_rgb import *


def _pal_secam_rgb_transfer_function_vectorise(value):
    value = np.asarray(value)

    return value ** (1 / 2.8)


def _pal_secam_rgb_transfer_function_analysis():
    message_box('_pal_secam_rgb_transfer_function')

    print(_pal_secam_rgb_transfer_function(RGB[0]))

    print(_pal_secam_rgb_transfer_function_vectorise(RGB[0]))

    print(_pal_secam_rgb_transfer_function_vectorise(RGB))

    print(_pal_secam_rgb_transfer_function_vectorise(RGB_t))

    print('\n')


# _pal_secam_rgb_transfer_function_analysis()


def _pal_secam_rgb_inverse_transfer_function_vectorise(value):
    value = np.asarray(value)

    return value ** 2.8


def _pal_secam_rgb_inverse_transfer_function_analysis():
    message_box('_pal_secam_rgb_inverse_transfer_function')

    print(_pal_secam_rgb_inverse_transfer_function(RGB[0]))

    print(_pal_secam_rgb_inverse_transfer_function_vectorise(RGB[0]))

    print(_pal_secam_rgb_inverse_transfer_function_vectorise(RGB))

    print(_pal_secam_rgb_inverse_transfer_function_vectorise(RGB_t))

    print('\n')


# _pal_secam_rgb_inverse_transfer_function_analysis()

from colour.models.dataset.prophoto_rgb import (
    _prophoto_rgb_transfer_function,
    _prophoto_rgb_inverse_transfer_function)
from colour.models.dataset.prophoto_rgb import *


def _prophoto_rgb_transfer_function_vectorise(value):
    value = np.asarray(value)

    return np.where(value < 0.001953,
                    value * 16,
                    value ** (1 / 1.8))


def _prophoto_rgb_transfer_function_analysis():
    message_box('_prophoto_rgb_transfer_function')

    print(_prophoto_rgb_transfer_function_vectorise(RGB))

    print(_prophoto_rgb_transfer_function_vectorise(RGB_t))

    print('\n')

    np.testing.assert_almost_equal(
        [_prophoto_rgb_transfer_function(x) for x in np.ravel(DATA1)],
        np.ravel(_prophoto_rgb_transfer_function_vectorise(DATA1)))


# _prophoto_rgb_transfer_function_analysis()


def _prophoto_rgb_inverse_transfer_function_vectorise(value):
    value = np.asarray(value)

    return np.where(
        value < _prophoto_rgb_transfer_function_vectorise(0.001953),
        value / 16,
        value ** 1.8)


def _prophoto_rgb_inverse_transfer_function_analysis():
    message_box('_prophoto_rgb_inverse_transfer_function')

    print(_prophoto_rgb_inverse_transfer_function_vectorise(RGB))

    print(_prophoto_rgb_inverse_transfer_function_vectorise(RGB_t))

    print('\n')

    np.testing.assert_almost_equal(
        [_prophoto_rgb_inverse_transfer_function(x) for x in np.ravel(DATA1)],
        np.ravel(_prophoto_rgb_inverse_transfer_function_vectorise(DATA1)))


# _prophoto_rgb_inverse_transfer_function_analysis()

from colour.models.dataset.rec_709 import (
    _rec_709_transfer_function,
    _rec_709_inverse_transfer_function)
from colour.models.dataset.rec_709 import *


def _rec_709_transfer_function_vectorise(value):
    value = np.asarray(value)

    return np.where(value < 0.018,
                    value * 4.5,
                    1.099 * (value ** 0.45) - 0.099)


def _rec_709_transfer_function_analysis():
    message_box('_rec_709_transfer_function')

    print(_rec_709_transfer_function_vectorise(RGB))

    print(_rec_709_transfer_function_vectorise(RGB_t))

    print('\n')

    np.testing.assert_almost_equal(
        [_rec_709_transfer_function(x) for x in np.ravel(DATA1)],
        np.ravel(_rec_709_transfer_function_vectorise(DATA1)))


# _rec_709_transfer_function_analysis()


def _rec_709_inverse_transfer_function_vectorise(value):
    value = np.asarray(value)

    return np.where(value < _rec_709_transfer_function_vectorise(0.018),
                    value / 4.5,
                    ((value + 0.099) / 1.099) ** (1 / 0.45))


def _rec_709_inverse_transfer_function_analysis():
    message_box('_rec_709_inverse_transfer_function')

    print(_rec_709_inverse_transfer_function_vectorise(RGB))

    print(_rec_709_inverse_transfer_function_vectorise(RGB_t))

    print('\n')

    np.testing.assert_almost_equal(
        [_rec_709_inverse_transfer_function(x) for x in np.ravel(DATA1)],
        np.ravel(_rec_709_inverse_transfer_function_vectorise(DATA1)))


# _rec_709_inverse_transfer_function_analysis()

from colour.models.dataset.rec_2020 import (
    _rec_2020_transfer_function,
    _rec_2020_inverse_transfer_function)
from colour.models.dataset.rec_2020 import *


def _rec_2020_transfer_function_vectorise(value, is_10_bits_system=True):
    value = np.asarray(value)

    a = REC_2020_CONSTANTS.alpha(is_10_bits_system)
    b = REC_2020_CONSTANTS.beta(is_10_bits_system)
    return np.where(value < b,
                    value * 4.5,
                    a * (value ** 0.45) - (a - 1))


def _rec_2020_transfer_function_analysis():
    message_box('_rec_2020_transfer_function')

    print(_rec_2020_transfer_function_vectorise(RGB))

    print(_rec_2020_transfer_function_vectorise(RGB_t))

    print('\n')

    np.testing.assert_almost_equal(
        [_rec_2020_transfer_function(x) for x in np.ravel(DATA1)],
        np.ravel(_rec_2020_transfer_function_vectorise(DATA1)))


# _rec_2020_transfer_function_analysis()


def _rec_2020_inverse_transfer_function_vectorise(value,
                                                  is_10_bits_system=True):
    value = np.asarray(value)

    a = REC_2020_CONSTANTS.alpha(is_10_bits_system)
    b = REC_2020_CONSTANTS.beta(is_10_bits_system)
    return np.where(value < _rec_2020_transfer_function_vectorise(b),
                    value / 4.5,
                    ((value + (a - 1)) / a) ** (1 / 0.45))


def _rec_2020_inverse_transfer_function_analysis():
    message_box('_rec_2020_inverse_transfer_function')

    print(_rec_2020_inverse_transfer_function_vectorise(RGB))

    print(_rec_2020_inverse_transfer_function_vectorise(RGB_t))

    print('\n')

    np.testing.assert_almost_equal(
        [_rec_2020_inverse_transfer_function(x) for x in np.ravel(DATA1)],
        np.ravel(_rec_2020_inverse_transfer_function_vectorise(DATA1)))


# _rec_2020_inverse_transfer_function_analysis()

from colour.models.dataset.s_gamut import (
    _s_log_transfer_function,
    _s_log_inverse_transfer_function,
    _s_log2_transfer_function,
    _s_log2_inverse_transfer_function,
    _s_log3_transfer_function,
    _s_log3_inverse_transfer_function)
from colour.models.dataset.s_gamut import *


def _s_log_transfer_function_vectorise(value):
    value = np.asarray(value)

    return (0.432699 * np.log10(value + 0.037584) + 0.616596) + 0.03


def _s_log_transfer_function_analysis():
    message_box('_s_log_transfer_function')

    print(_s_log_transfer_function(RGB[0]))

    print(_s_log_transfer_function_vectorise(RGB[0]))

    print(_s_log_transfer_function_vectorise(RGB))

    print(_s_log_transfer_function_vectorise(RGB_t))

    print('\n')


# _s_log_transfer_function_analysis()


def _s_log_inverse_transfer_function_vectorise(value):
    value = np.asarray(value)

    return 10 ** (((value - 0.616596 - 0.03) / 0.432699)) - 0.037584


def _s_log_inverse_transfer_function_analysis():
    message_box('_s_log_inverse_transfer_function')

    print(_s_log_inverse_transfer_function(RGB[0]))

    print(_s_log_inverse_transfer_function_vectorise(RGB[0]))

    print(_s_log_inverse_transfer_function_vectorise(RGB))

    print(_s_log_inverse_transfer_function_vectorise(RGB_t))

    print('\n')


# _s_log_inverse_transfer_function_analysis()


def _s_log2_transfer_function_vectorise(value):
    value = np.asarray(value)

    return ((4 * (16 + 219 *
                  (0.616596 + 0.03 + 0.432699 *
                   (np.log10(0.037584 + value / 0.9))))) / 1023)


def _s_log2_transfer_function_analysis():
    message_box('_s_log2_transfer_function')

    print(_s_log2_transfer_function(RGB[0]))

    print(_s_log2_transfer_function_vectorise(RGB[0]))

    print(_s_log2_transfer_function_vectorise(RGB))

    print(_s_log2_transfer_function_vectorise(RGB_t))

    print('\n')


# _s_log2_transfer_function_analysis()


def _s_log2_inverse_transfer_function_vectorise(value):
    value = np.asarray(value)

    return ((10 ** (((((value * 1023 / 4 - 16) / 219) - 0.616596 - 0.03) /
                     0.432699)) - 0.037584) * 0.9)


def _s_log2_inverse_transfer_function_analysis():
    message_box('_s_log2_inverse_transfer_function')

    print(_s_log2_inverse_transfer_function(RGB[0]))

    print(_s_log2_inverse_transfer_function_vectorise(RGB[0]))

    print(_s_log2_inverse_transfer_function_vectorise(RGB))

    print(_s_log2_inverse_transfer_function_vectorise(RGB_t))

    print('\n')


# _s_log2_inverse_transfer_function_analysis()


def _s_log3_transfer_function_vectorise(value):
    value = np.asarray(value)

    return np.where(value >= 0.01125000,
                    (420 + np.log10((value + 0.01) /
                                    (0.18 + 0.01)) * 261.5) / 1023,
                    (value * (171.2102946929 - 95) / 0.01125000 + 95) / 1023)


def _s_log3_transfer_function_analysis():
    message_box('_s_log3_transfer_function')

    print(_s_log3_transfer_function(RGB[0]))

    print(_s_log3_transfer_function_vectorise(RGB[0]))

    print(_s_log3_transfer_function_vectorise(RGB))

    print(_s_log3_transfer_function_vectorise(RGB_t))

    print('\n')

    np.testing.assert_almost_equal(
        [_s_log3_transfer_function(x) for x in np.ravel(DATA1)],
        np.ravel(_s_log3_transfer_function_vectorise(DATA1)))


# _s_log3_transfer_function_analysis()


def _s_log3_inverse_transfer_function_vectorise(value):
    value = np.asarray(value)

    return np.where(value >= 171.2102946929 / 1023,
                    ((10 ** ((value * 1023 - 420) / 261.5)) *
                     (0.18 + 0.01) - 0.01),
                    (value * 1023 - 95) * 0.01125000 / (171.2102946929 - 95))


def _s_log3_inverse_transfer_function_analysis():
    message_box('_s_log3_inverse_transfer_function')

    print(_s_log3_inverse_transfer_function(RGB[0]))

    print(_s_log3_inverse_transfer_function_vectorise(RGB[0]))

    print(_s_log3_inverse_transfer_function_vectorise(RGB))

    print(_s_log3_inverse_transfer_function_vectorise(RGB_t))

    print('\n')

    np.testing.assert_almost_equal(
        [_s_log3_inverse_transfer_function(x) for x in np.ravel(DATA1)],
        np.ravel(_s_log3_inverse_transfer_function_vectorise(DATA1)))


# _s_log3_inverse_transfer_function_analysis()

from colour.models.dataset.srgb import (
    _srgb_transfer_function,
    _srgb_inverse_transfer_function)
from colour.models.dataset.srgb import *


def _srgb_transfer_function_vectorise(value):
    value = np.asarray(value)

    return np.where(value <= 0.0031308,
                    value * 12.92,
                    1.055 * (value ** (1 / 2.4)) - 0.055)


def _srgb_transfer_function_analysis():
    message_box('_srgb_transfer_function')

    print(_srgb_transfer_function_vectorise(RGB))

    print(_srgb_transfer_function_vectorise(RGB_t))

    print('\n')

    np.testing.assert_almost_equal(
        [_srgb_transfer_function(x) for x in np.ravel(DATA1)],
        np.ravel(_srgb_transfer_function_vectorise(DATA1)))


# _srgb_transfer_function_analysis()


def _srgb_inverse_transfer_function_vectorise(value):
    value = np.asarray(value)

    return np.where(value <= _srgb_transfer_function_vectorise(0.0031308),
                    value / 12.92,
                    ((value + 0.055) / 1.055) ** 2.4)


def _srgb_inverse_transfer_function_analysis():
    message_box('_srgb_inverse_transfer_function')

    print(_srgb_inverse_transfer_function_vectorise(RGB))

    print(_srgb_inverse_transfer_function_vectorise(RGB_t))

    print('\n')

    np.testing.assert_almost_equal(
        [_srgb_inverse_transfer_function(x) for x in np.ravel(DATA1)],
        np.ravel(_srgb_inverse_transfer_function_vectorise(DATA1)))


# _srgb_inverse_transfer_function_analysis()

# #############################################################################
# ### colour.XYZ_to_RGB
# #############################################################################
from colour.models.rgb import *

W_R = (0.34567, 0.35850)
W_T = (0.31271, 0.32902)
CAT = 'Bradford'
M = np.array([
    [3.24100326, -1.53739899, -0.49861587],
    [-0.96922426, 1.87592999, 0.04155422],
    [0.05563942, -0.2040112, 1.05714897]])


def XYZ_to_RGB_2d(XYZ):
    for i in range(len(XYZ)):
        XYZ_to_RGB(XYZ[i], W_R, W_T, M, CAT)


def XYZ_to_RGB_vectorise(XYZ,
                         illuminant_XYZ,
                         illuminant_RGB,
                         XYZ_to_RGB_matrix,
                         chromatic_adaptation_transform='CAT02',
                         transfer_function=None):
    M = chromatic_adaptation_matrix_VonKries_vectorise(
        xy_to_XYZ_vectorise(illuminant_XYZ),
        xy_to_XYZ_vectorise(illuminant_RGB),
        transform=chromatic_adaptation_transform)

    XYZ_a = np.einsum('...ij,...j->...i', M, XYZ)

    RGB = np.einsum('...ij,...j->...i', XYZ_to_RGB_matrix, XYZ_a)

    if transfer_function is not None:
        RGB = transfer_function(RGB)

    return RGB


def XYZ_to_RGB_analysis():
    message_box('XYZ_to_RGB')

    print('Reference:')
    XYZ = np.array([0.07049534, 0.1008, 0.09558313])

    print(XYZ_to_RGB(XYZ, W_R, W_T, M, CAT))

    print('\n')

    print('1d array input:')
    print(XYZ_to_RGB_vectorise(XYZ, W_R, W_T, M, CAT))

    print('\n')

    print('2d array input:')
    XYZ = np.tile(XYZ, (6, 1))
    print(XYZ_to_RGB_vectorise(XYZ, W_R, W_T, M, CAT))

    print('\n')

    print('3d array input:')
    XYZ = np.reshape(XYZ, (2, 3, 3))
    print(XYZ_to_RGB_vectorise(XYZ, W_R, W_T, M, CAT))

    print('\n')


# XYZ_to_RGB_analysis()


def XYZ_to_RGB_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            XYZ_to_RGB_2d,
            DATA_HD1)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            XYZ_to_RGB_vectorise,
            DATA_HD1, W_R, W_T, M, CAT)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('XYZ_to_RGB\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# XYZ_to_RGB_profile(3, 3)

# #############################################################################
# # ### colour.RGB_to_XYZ
# #############################################################################
W_R = (0.31271, 0.32902)
W_T = (0.34567, 0.35850)
CAT = 'Bradford'
M = np.array([
    [0.41238656, 0.35759149, 0.18045049],
    [0.21263682, 0.71518298, 0.0721802],
    [0.01933062, 0.11919716, 0.95037259]])


def RGB_to_XYZ_2d(RGB):
    for i in range(len(RGB)):
        RGB_to_XYZ(RGB[i], W_R, W_T, M, CAT)


def RGB_to_XYZ_vectorise(RGB,
                         illuminant_RGB,
                         illuminant_XYZ,
                         RGB_to_XYZ_matrix,
                         chromatic_adaptation_transform='CAT02',
                         inverse_transfer_function=None):
    if inverse_transfer_function is not None:
        RGB = inverse_transfer_function(RGB)

    XYZ = np.einsum('...ij,...j->...i', RGB_to_XYZ_matrix, RGB)

    M = chromatic_adaptation_matrix_VonKries_vectorise(
        xy_to_XYZ_vectorise(illuminant_RGB),
        xy_to_XYZ_vectorise(illuminant_XYZ),
        transform=chromatic_adaptation_transform)

    XYZ_a = np.einsum('...ij,...j->...i', M, XYZ)

    return XYZ_a


def RGB_to_XYZ_analysis():
    message_box('RGB_to_XYZ')

    print('Reference:')
    RGB = np.array([0.86969452, 1.00516431, 1.41715848])
    print(RGB_to_XYZ(RGB, W_R, W_T, M, CAT))

    print('\n')

    print('1d array input:')
    print(RGB_to_XYZ_vectorise(RGB, W_R, W_T, M, CAT))

    print('\n')

    print('2d array input:')
    RGB = np.tile(RGB, (6, 1))
    print(RGB_to_XYZ_vectorise(RGB, W_R, W_T, M, CAT))

    print('\n')

    print('3d array input:')
    RGB = np.reshape(RGB, (2, 3, 3))
    print(RGB_to_XYZ_vectorise(RGB, W_R, W_T, M, CAT))

    print('\n')


# RGB_to_XYZ_analysis()


def RGB_to_XYZ_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            RGB_to_XYZ_2d,
            DATA_HD1)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            RGB_to_XYZ_vectorise,
            DATA_HD1, W_R, W_T, M, CAT)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('RGB_to_XYZ\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# RGB_to_XYZ_profile(3, 3)

# #############################################################################
# ### colour.RGB_to_RGB
# #############################################################################
from colour import sRGB_COLOURSPACE, PROPHOTO_RGB_COLOURSPACE

C_I = sRGB_COLOURSPACE
C_O = PROPHOTO_RGB_COLOURSPACE
CAT = 'CAT02'


def RGB_to_RGB_2d(RGB):
    for i in range(len(RGB)):
        RGB_to_RGB(RGB[i], C_I, C_O, CAT)


def RGB_to_RGB_vectorise(RGB,
                         input_colourspace,
                         output_colourspace,
                         chromatic_adaptation_transform='CAT02'):
    cat = chromatic_adaptation_matrix_VonKries_vectorise(
        xy_to_XYZ_vectorise(input_colourspace.whitepoint),
        xy_to_XYZ_vectorise(output_colourspace.whitepoint),
        chromatic_adaptation_transform)

    M = np.einsum('...ij,...jk->...ik',
                  cat,
                  input_colourspace.RGB_to_XYZ_matrix)
    M = np.einsum('...ij,...jk->...ik',
                  output_colourspace.XYZ_to_RGB_matrix,
                  M)

    RGB = np.einsum('...ij,...j->...i', M, RGB)

    return RGB


def RGB_to_RGB_analysis():
    message_box('RGB_to_RGB')

    print('Reference:')
    RGB = np.array([0.86969452, 1.00516431, 1.41715848])
    print(RGB_to_RGB(RGB, C_I, C_O, CAT))

    print('\n')

    print('1d array input:')
    print(RGB_to_RGB_vectorise(RGB, C_I, C_O, CAT))

    print('\n')

    print('2d array input:')
    RGB = np.tile(RGB, (6, 1))
    print(RGB_to_RGB_vectorise(RGB, C_I, C_O, CAT))
    print('\n')

    print('3d array input:')
    RGB = np.reshape(RGB, (2, 3, 3))
    print(RGB_to_RGB_vectorise(RGB, C_I, C_O, CAT))

    print('\n')


# RGB_to_RGB_analysis()


def RGB_to_RGB_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            RGB_to_RGB_2d,
            DATA_HD1)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            RGB_to_RGB_vectorise,
            DATA_HD1, C_I, C_O, CAT)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('RGB_to_RGB\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# RGB_to_RGB_profile()

# #############################################################################
# #############################################################################
# ### colour.notation.munsell
# #############################################################################
# #############################################################################

# #############################################################################
# ### colour.munsell_value_Priest1920
# #############################################################################
from colour.notation.munsell import *

Y = np.linspace(0, 100, 1000000)


def munsell_value_Priest1920_2d(Y):
    for i in range(len(Y)):
        munsell_value_Priest1920(Y[i])


def munsell_value_Priest1920_vectorise(Y):
    Y = np.asarray(Y)

    V = 10 * np.sqrt(Y / 100)

    return V


def munsell_value_Priest1920_analysis():
    message_box('munsell_value_Priest1920')

    print('Reference:')
    print(munsell_value_Priest1920(10.08))

    print('\n')

    print('Numeric input:')
    print(munsell_value_Priest1920_vectorise(10.08))

    print('\n')

    print('0d array input:')
    print(munsell_value_Priest1920_vectorise(np.array(10.08)))

    print('\n')

    print('1d array input:')
    Y = [10.08] * 6
    print(munsell_value_Priest1920_vectorise(Y))

    print('\n')

    print('2d array input:')
    Y = np.reshape(Y, (2, 3))
    print(munsell_value_Priest1920_vectorise(Y))

    print('\n')

    print('3d array input:')
    Y = np.reshape(Y, (2, 3, 1))
    print(munsell_value_Priest1920_vectorise(Y))

    print('\n')


# munsell_value_Priest1920_analysis()


def munsell_value_Priest1920_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            munsell_value_Priest1920_2d,
            Y)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            munsell_value_Priest1920_vectorise,
            Y)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('munsell_value_Priest1920\t{0}\t{1}\t{2}'.format(
        len(Y), a, b))


# munsell_value_Priest1920_profile()

# #############################################################################
# ### colour.munsell_value_Munsell1933
# #############################################################################


def munsell_value_Munsell1933_2d(Y):
    for i in range(len(Y)):
        munsell_value_Munsell1933(Y[i])


def munsell_value_Munsell1933_vectorise(Y):
    Y = np.asarray(Y)

    V = np.sqrt(1.4742 * Y - 0.004743 * (Y * Y))

    return V


def munsell_value_Munsell1933_analysis():
    message_box('munsell_value_Munsell1933')

    print('Reference:')
    print(munsell_value_Munsell1933(10.08))

    print('\n')

    print('Numeric input:')
    print(munsell_value_Munsell1933_vectorise(10.08))

    print('\n')

    print('0d array input:')
    print(munsell_value_Munsell1933_vectorise(np.array(10.08)))

    print('\n')

    print('1d array input:')
    Y = [10.08] * 6
    print(munsell_value_Munsell1933_vectorise(Y))

    print('\n')

    print('2d array input:')
    Y = np.reshape(Y, (2, 3))
    print(munsell_value_Munsell1933_vectorise(Y))

    print('\n')

    print('3d array input:')
    Y = np.reshape(Y, (2, 3, 1))
    print(munsell_value_Munsell1933_vectorise(Y))

    print('\n')


# munsell_value_Munsell1933_analysis()


def munsell_value_Munsell1933_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            munsell_value_Munsell1933_2d,
            Y)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            munsell_value_Munsell1933_vectorise,
            Y)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('munsell_value_Munsell1933\t{0}\t{1}\t{2}'.format(
        len(Y), a, b))


# munsell_value_Munsell1933_profile()

# #############################################################################
# ### colour.munsell_value_Moon1943
# #############################################################################


def munsell_value_Moon1943_2d(Y):
    for i in range(len(Y)):
        munsell_value_Moon1943(Y[i])


def munsell_value_Moon1943_vectorise(Y):
    Y = np.asarray(Y)

    V = 1.4 * Y ** 0.426

    return V


def munsell_value_Moon1943_analysis():
    message_box('munsell_value_Moon1943')

    print('Reference:')
    print(munsell_value_Moon1943(10.08))

    print('\n')

    print('Numeric input:')
    print(munsell_value_Moon1943_vectorise(10.08))

    print('\n')

    print('0d array input:')
    print(munsell_value_Moon1943_vectorise(np.array(10.08)))

    print('\n')

    print('1d array input:')
    Y = [10.08] * 6
    print(munsell_value_Moon1943_vectorise(Y))

    print('\n')

    print('2d array input:')
    Y = np.reshape(Y, (2, 3))
    print(munsell_value_Moon1943_vectorise(Y))

    print('\n')

    print('3d array input:')
    Y = np.reshape(Y, (2, 3, 1))
    print(munsell_value_Moon1943_vectorise(Y))

    print('\n')


# munsell_value_Moon1943_analysis()


def munsell_value_Moon1943_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            munsell_value_Moon1943_2d,
            Y)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            munsell_value_Moon1943_vectorise,
            Y)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('munsell_value_Moon1943\t{0}\t{1}\t{2}'.format(
        len(Y), a, b))


# munsell_value_Moon1943_profile()

# #############################################################################
# ### colour.munsell_value_Saunderson1944
# #############################################################################


def munsell_value_Saunderson1944_2d(Y):
    for i in range(len(Y)):
        munsell_value_Saunderson1944(Y[i])


def munsell_value_Saunderson1944_vectorise(Y):
    Y = np.asarray(Y)

    V = 2.357 * (Y ** 0.343) - 1.52

    return V


def munsell_value_Saunderson1944_analysis():
    message_box('munsell_value_Saunderson1944')

    print('Reference:')
    print(munsell_value_Saunderson1944(10.08))

    print('\n')

    print('Numeric input:')
    print(munsell_value_Saunderson1944_vectorise(10.08))

    print('\n')

    print('0d array input:')
    print(munsell_value_Saunderson1944_vectorise(np.array(10.08)))

    print('\n')

    print('1d array input:')
    Y = [10.08] * 6
    print(munsell_value_Saunderson1944_vectorise(Y))

    print('\n')

    print('2d array input:')
    Y = np.reshape(Y, (2, 3))
    print(munsell_value_Saunderson1944_vectorise(Y))

    print('\n')

    print('3d array input:')
    Y = np.reshape(Y, (2, 3, 1))
    print(munsell_value_Saunderson1944_vectorise(Y))

    print('\n')


# munsell_value_Saunderson1944_analysis()


def munsell_value_Saunderson1944_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            munsell_value_Saunderson1944_2d,
            Y)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            munsell_value_Saunderson1944_vectorise,
            Y)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('munsell_value_Saunderson1944\t{0}\t{1}\t{2}'.format(
        len(Y), a, b))


# munsell_value_Saunderson1944_profile()

# #############################################################################
# ### colour.munsell_value_Ladd1955
# #############################################################################


def munsell_value_Ladd1955_2d(Y):
    for i in range(len(Y)):
        munsell_value_Ladd1955(Y[i])


def munsell_value_Ladd1955_vectorise(Y):
    Y = np.asarray(Y)

    V = 2.468 * (Y ** (1 / 3)) - 1.636

    return V


def munsell_value_Ladd1955_analysis():
    message_box('munsell_value_Ladd1955')

    print('Reference:')
    print(munsell_value_Ladd1955(10.08))

    print('\n')

    print('Numeric input:')
    print(munsell_value_Ladd1955_vectorise(10.08))

    print('\n')

    print('0d array input:')
    print(munsell_value_Ladd1955_vectorise(np.array(10.08)))

    print('\n')

    print('1d array input:')
    Y = [10.08] * 6
    print(munsell_value_Ladd1955_vectorise(Y))

    print('\n')

    print('2d array input:')
    Y = np.reshape(Y, (2, 3))
    print(munsell_value_Ladd1955_vectorise(Y))

    print('\n')

    print('3d array input:')
    Y = np.reshape(Y, (2, 3, 1))
    print(munsell_value_Ladd1955_vectorise(Y))

    print('\n')


# munsell_value_Ladd1955_analysis()


def munsell_value_Ladd1955_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            munsell_value_Ladd1955_2d,
            Y)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            munsell_value_Ladd1955_vectorise,
            Y)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('munsell_value_Ladd1955\t{0}\t{1}\t{2}'.format(
        len(Y), a, b))


# munsell_value_Ladd1955_profile()

# #############################################################################
# ### colour.munsell_value_McCamy1987
# #############################################################################


def munsell_value_McCamy1987_2d(Y):
    V = []
    for i in range(len(Y)):
        V.append(munsell_value_McCamy1987(Y[i]))
    return V


@ignore_numpy_errors
def munsell_value_McCamy1987_vectorise(Y):
    Y = np.asarray(Y)

    V = np.where(Y <= 0.9,
                 0.87445 * (Y ** 0.9967),
                 (2.49268 * (Y ** (1 / 3)) - 1.5614 -
                  (0.985 / (((0.1073 * Y - 3.084) ** 2) + 7.54)) +
                  (0.0133 / (Y ** 2.3)) +
                  0.0084 * np.sin(4.1 * (Y ** (1 / 3)) + 1) +
                  (0.0221 / Y) * np.sin(0.39 * (Y - 2)) -
                  (0.0037 / (0.44 * Y)) * np.sin(1.28 * (Y - 0.53))))

    return V


def munsell_value_McCamy1987_analysis():
    message_box('munsell_value_McCamy1987')

    print('Reference:')
    print(munsell_value_McCamy1987(10.08))

    print('\n')

    print('Numeric input:')
    print(munsell_value_McCamy1987_vectorise(10.08))

    print('\n')

    print('0d array input:')
    print(munsell_value_McCamy1987_vectorise(np.array(10.08)))

    print('\n')

    print('1d array input:')
    Y = [10.08] * 6
    print(munsell_value_McCamy1987_vectorise(Y))

    print('\n')

    print('2d array input:')
    Y = np.reshape(Y, (2, 3))
    print(munsell_value_McCamy1987_vectorise(Y))

    print('\n')

    print('3d array input:')
    Y = np.reshape(Y, (2, 3, 1))
    print(munsell_value_McCamy1987_vectorise(Y))

    print('\n')

    Y = np.linspace(0, 100, 10000)
    np.testing.assert_almost_equal(
        munsell_value_McCamy1987_2d(Y),
        munsell_value_McCamy1987_vectorise(Y))


# munsell_value_McCamy1987_analysis()

def munsell_value_McCamy1987_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            munsell_value_McCamy1987_2d,
            Y)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            munsell_value_McCamy1987_vectorise,
            Y)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('munsell_value_McCamy1987\t{0}\t{1}\t{2}'.format(
        len(Y), a, b))


# munsell_value_McCamy1987_profile()

# #############################################################################
# ### colour.munsell_value_ASTMD153508
# #############################################################################


def munsell_value_ASTMD153508_2d(Y):
    for i in range(len(Y)):
        munsell_value_ASTMD153508(Y[i])


from colour.algebra import *

_MUNSELL_VALUE_ASTM_D1535_08_INTERPOLATOR_CACHE = None


def _munsell_value_ASTMD153508_interpolator():
    global _MUNSELL_VALUE_ASTM_D1535_08_INTERPOLATOR_CACHE
    munsell_values = np.arange(0, 10, 0.001)
    if _MUNSELL_VALUE_ASTM_D1535_08_INTERPOLATOR_CACHE is None:
        _MUNSELL_VALUE_ASTM_D1535_08_INTERPOLATOR_CACHE = Extrapolator1d(
            LinearInterpolator1d(
                luminance_ASTMD153508_vectorise(munsell_values),
                munsell_values))

    return _MUNSELL_VALUE_ASTM_D1535_08_INTERPOLATOR_CACHE


def munsell_value_ASTMD153508_vectorise(Y):
    Y = np.asarray(Y)

    V = _munsell_value_ASTMD153508_interpolator()(Y)

    return V


def munsell_value_ASTMD153508_analysis():
    message_box('munsell_value_ASTMD153508')

    print('Reference:')
    print(munsell_value_ASTMD153508(10.08))

    print('\n')

    print('Numeric input:')
    print(munsell_value_ASTMD153508_vectorise(10.08))

    print('\n')

    print('0d array input:')
    print(munsell_value_ASTMD153508_vectorise(np.array(10.08)))

    print('\n')

    print('1d array input:')
    Y = [10.08] * 6
    print(munsell_value_ASTMD153508_vectorise(Y))

    print('\n')

    print('2d array input:')
    Y = np.reshape(Y, (2, 3))
    print(munsell_value_ASTMD153508_vectorise(Y))

    print('\n')

    print('3d array input:')
    Y = np.reshape(Y, (2, 3, 1))
    print(munsell_value_ASTMD153508_vectorise(Y))

    print('\n')


# munsell_value_ASTMD153508_analysis()


def munsell_value_ASTMD153508_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            munsell_value_ASTMD153508_2d,
            Y)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            munsell_value_ASTMD153508_vectorise,
            Y)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('munsell_value_ASTMD153508\t{0}\t{1}\t{2}'.format(
        len(Y), a, b))


# munsell_value_ASTMD153508_profile()

# #############################################################################
# #############################################################################
# ### colour.notation.triplet
# #############################################################################
# #############################################################################

# #############################################################################
# ### colour.notation.triplet.RGB_to_HEX
# #############################################################################
from colour.notation.triplet import *


def RGB_to_HEX_2d(RGB):
    for i in range(len(RGB)):
        RGB_to_HEX(RGB[i])


def RGB_to_HEX_vectorise(RGB):
    to_HEX = np.vectorize('{0:02x}'.format)

    HEX = to_HEX((RGB * 255).astype(np.uint8)).astype(object)
    HEX = np.asarray('#') + HEX[..., 0] + HEX[..., 1] + HEX[..., 2]

    return HEX


def RGB_to_HEX_analysis():
    message_box('RGB_to_HEX')

    print('Reference:')
    RGB = np.array([0.66666667, 0.86666667, 1])
    print(RGB_to_HEX(RGB))

    print('\n')

    print('1d array input:')
    print(RGB_to_HEX_vectorise(RGB))

    print('\n')

    print('2d array input:')
    RGB = np.tile(RGB, (6, 1))
    print(RGB_to_HEX_vectorise(RGB))

    print('\n')

    print('3d array input:')
    RGB = np.reshape(RGB, (2, 3, 3))
    print(RGB_to_HEX_vectorise(RGB))

    print('\n')


# RGB_to_HEX_analysis()


def RGB_to_HEX_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            RGB_to_HEX_2d,
            DATA_HD1)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            RGB_to_HEX_vectorise,
            DATA_HD1)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('RGB_to_HEX\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1), a, b))


# RGB_to_HEX_profile()

# #############################################################################
# ### colour.notation.triplet.HEX_to_RGB
# #############################################################################
from colour.notation.triplet import *


def HEX_to_RGB_2d(HEX):
    for i in range(len(HEX)):
        HEX_to_RGB(HEX[i])


def HEX_to_RGB_vectorise(HEX):
    HEX = np.core.defchararray.lstrip(HEX, '#')

    def to_RGB(x):
        length = len(x)
        return [int(x[i:i + length // 3], 16)
                for i in range(0, length, length // 3)]

    toRGB = np.vectorize(to_RGB, otypes=[np.ndarray])

    RGB = np.asarray(toRGB(HEX).tolist()) / 255

    return RGB


def HEX_to_RGB_analysis():
    message_box('HEX_to_RGB')

    print('Reference:')
    HEX = '#aaddff'
    print(HEX_to_RGB(HEX))

    print('\n')

    print('Numeric input:')
    print(HEX_to_RGB_vectorise(HEX))

    print('\n')

    print('1d array input:')
    HEX = np.tile(HEX, (6,))
    print(HEX_to_RGB_vectorise(HEX))

    print('\n')

    print('2d array input:')
    HEX = np.reshape(HEX, (2, 3))
    print(HEX_to_RGB_vectorise(HEX))

    # HEX1 = ['#aaddff'] * (1920 * 1080)

    print('\n')


# HEX_to_RGB_analysis()


def HEX_to_RGB_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    H = ['#aaddff'] * (1920 * 1080)
    times = timeit.Timer(
        functools.partial(
            HEX_to_RGB_2d,
            H)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            HEX_to_RGB_vectorise,
            H)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('HEX_to_RGB\t{0}\t{1}\t{2}'.format(
        len(H), a, b))


# HEX_to_RGB_profile()

# #############################################################################
# #############################################################################
# ### colour.phenomenons.rayleigh
# #############################################################################
# #############################################################################

# #############################################################################
# ### colour.phenomenons.rayleigh.air_refraction_index_Penndorf1957
# #############################################################################
from colour.phenomenons.rayleigh import *


def air_refraction_index_Penndorf1957_2d(wl):
    for i in range(len(wl)):
        air_refraction_index_Penndorf1957(wl[i])


def air_refraction_index_Penndorf1957_vectorise(wavelength, *args):
    wl = np.asarray(wavelength)

    n = 6432.8 + 2949810 / (146 - wl ** (-2)) + 25540 / (41 - wl ** (-2))
    n = n / 1.0e8 + 1

    return n


def air_refraction_index_Penndorf1957_analysis():
    message_box('air_refraction_index_Penndorf1957')

    print('Reference:')
    print(air_refraction_index_Penndorf1957(0.555))

    print('\n')

    print('Numeric input:')
    print(air_refraction_index_Penndorf1957_vectorise(0.555))

    print('\n')

    print('0d array input:')
    print(air_refraction_index_Penndorf1957_vectorise(np.array(0.555)))

    print('\n')

    print('1d array input:')
    wl = [0.555] * 6
    print(air_refraction_index_Penndorf1957_vectorise(wl))

    print('\n')

    print('2d array input:')
    wl = np.reshape(wl, (2, 3))
    print(air_refraction_index_Penndorf1957_vectorise(wl))

    print('\n')

    print('3d array input:')
    wl = np.reshape(wl, (2, 3, 1))
    print(air_refraction_index_Penndorf1957_vectorise(wl))

    print('\n')


# air_refraction_index_Penndorf1957_analysis()


def air_refraction_index_Penndorf1957_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            air_refraction_index_Penndorf1957_2d,
            DATA_HD1[..., 0])).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            air_refraction_index_Penndorf1957_vectorise,
            DATA_HD1[..., 0])).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('air_refraction_index_Penndorf1957\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1[..., 0]), a, b))


# air_refraction_index_Penndorf1957_profile()

# #############################################################################
# ### colour.phenomenons.rayleigh.air_refraction_index_Edlen1966
# #############################################################################


def air_refraction_index_Edlen1966_2d(wl):
    for i in range(len(wl)):
        air_refraction_index_Edlen1966(wl[i])


def air_refraction_index_Edlen1966_vectorise(wavelength, *args):
    wl = np.asarray(wavelength)

    n = 8342.13 + 2406030 / (130 - wl ** (-2)) + 15997 / (38.9 - wl ** (-2))
    n = n / 1.0e8 + 1

    return n


def air_refraction_index_Edlen1966_analysis():
    message_box('air_refraction_index_Edlen1966')

    print('Reference:')
    print(air_refraction_index_Edlen1966(0.555))

    print('\n')

    print('Numeric input:')
    print(air_refraction_index_Edlen1966_vectorise(0.555))

    print('\n')

    print('0d array input:')
    print(air_refraction_index_Edlen1966_vectorise(np.array(0.555)))

    print('\n')

    print('1d array input:')
    wl = [0.555] * 6
    print(air_refraction_index_Edlen1966_vectorise(wl))

    print('\n')

    print('2d array input:')
    wl = np.reshape(wl, (2, 3))
    print(air_refraction_index_Edlen1966_vectorise(wl))

    print('\n')

    print('3d array input:')
    wl = np.reshape(wl, (2, 3, 1))
    print(air_refraction_index_Edlen1966_vectorise(wl))

    print('\n')


# air_refraction_index_Edlen1966_analysis()


def air_refraction_index_Edlen1966_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            air_refraction_index_Edlen1966_2d,
            DATA_HD1[..., 0])).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            air_refraction_index_Edlen1966_vectorise,
            DATA_HD1[..., 0])).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('air_refraction_index_Edlen1966\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1[..., 0]), a, b))


# air_refraction_index_Edlen1966_profile()

# #############################################################################
# ### colour.phenomenons.rayleigh.air_refraction_index_Peck1972
# #############################################################################


def air_refraction_index_Peck1972_2d(wl):
    for i in range(len(wl)):
        air_refraction_index_Peck1972(wl[i])


def air_refraction_index_Peck1972_vectorise(wavelength, *args):
    wl = np.asarray(wavelength)

    n = (8060.51 + 2480990 / (132.274 - wl ** (-2)) + 17455.7 /
         (39.32957 - wl ** (-2)))
    n = n / 1.0e8 + 1

    return n


def air_refraction_index_Peck1972_analysis():
    message_box('air_refraction_index_Peck1972')

    print('Reference:')
    print(air_refraction_index_Peck1972(0.555))

    print('\n')

    print('Numeric input:')
    print(air_refraction_index_Peck1972_vectorise(0.555))

    print('\n')

    print('0d array input:')
    print(air_refraction_index_Peck1972_vectorise(np.array(0.555)))

    print('\n')

    print('1d array input:')
    wl = [0.555] * 6
    print(air_refraction_index_Peck1972_vectorise(wl))

    print('\n')

    print('2d array input:')
    wl = np.reshape(wl, (2, 3))
    print(air_refraction_index_Peck1972_vectorise(wl))

    print('\n')

    print('3d array input:')
    wl = np.reshape(wl, (2, 3, 1))
    print(air_refraction_index_Peck1972_vectorise(wl))

    print('\n')


# air_refraction_index_Peck1972_analysis()


def air_refraction_index_Peck1972_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            air_refraction_index_Peck1972_2d,
            DATA_HD1[..., 0])).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            air_refraction_index_Peck1972_vectorise,
            DATA_HD1[..., 0])).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('air_refraction_index_Peck1972\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1[..., 0]), a, b))


# air_refraction_index_Peck1972_profile()

# #############################################################################
# ### colour.phenomenons.rayleigh.air_refraction_index_Bodhaine1999
# #############################################################################


def air_refraction_index_Bodhaine1999_2d(wl):
    for i in range(len(wl)):
        air_refraction_index_Bodhaine1999(wl[i])


def air_refraction_index_Bodhaine1999_vectorise(
        wavelength,
        CO2_concentration=STANDARD_CO2_CONCENTRATION):
    wl = np.asarray(wavelength)
    CO2_c = np.asarray(CO2_concentration)

    n = ((1 + 0.54 * ((CO2_c * 1e-6) - 300e-6)) *
         (air_refraction_index_Peck1972(wl) - 1) + 1)

    return n


def air_refraction_index_Bodhaine1999_analysis():
    message_box('air_refraction_index_Bodhaine1999')

    print('Reference:')
    print(air_refraction_index_Bodhaine1999(0.555))

    print('\n')

    print('Numeric input:')
    print(air_refraction_index_Bodhaine1999_vectorise(0.555))

    print('\n')

    print('0d array input:')
    print(air_refraction_index_Bodhaine1999_vectorise(np.array(0.555)))

    print('\n')

    print('1d array input:')
    wl = [0.555] * 6
    print(air_refraction_index_Bodhaine1999_vectorise(wl))

    print('\n')

    print('2d array input:')
    wl = np.reshape(wl, (2, 3))
    print(air_refraction_index_Bodhaine1999_vectorise(wl))

    print('\n')

    print('3d array input:')
    wl = np.reshape(wl, (2, 3, 1))
    print(air_refraction_index_Bodhaine1999_vectorise(wl))

    print('\n')


# air_refraction_index_Bodhaine1999_analysis()


def air_refraction_index_Bodhaine1999_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            air_refraction_index_Bodhaine1999_2d,
            DATA_HD1[..., 0])).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            air_refraction_index_Bodhaine1999_vectorise,
            DATA_HD1[..., 0])).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('air_refraction_index_Bodhaine1999\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1[..., 0]), a, b))


# air_refraction_index_Bodhaine1999_profile()

# #############################################################################
# ### colour.phenomenons.rayleigh.N2_depolarisation
# #############################################################################


def N2_depolarisation_2d(wl):
    for i in range(len(wl)):
        N2_depolarisation(wl[i])


def N2_depolarisation_vectorise(wavelength):
    wl = np.asarray(wavelength)

    N2 = 1.034 + 3.17 * 1.0e-4 * (1 / wl ** 2)

    return N2


def N2_depolarisation_analysis():
    message_box('N2_depolarisation')

    print('Reference:')
    print(N2_depolarisation(0.555))

    print('\n')

    print('Numeric input:')
    print(N2_depolarisation_vectorise(0.555))

    print('\n')

    print('0d array input:')
    print(N2_depolarisation_vectorise(np.array(0.555)))

    print('\n')

    print('1d array input:')
    wl = [0.555] * 6
    print(N2_depolarisation_vectorise(wl))

    print('\n')

    print('2d array input:')
    wl = np.reshape(wl, (2, 3))
    print(N2_depolarisation_vectorise(wl))

    print('\n')

    print('3d array input:')
    wl = np.reshape(wl, (2, 3, 1))
    print(N2_depolarisation_vectorise(wl))

    print('\n')


# N2_depolarisation_analysis()


def N2_depolarisation_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            N2_depolarisation_2d,
            DATA_HD1[..., 0])).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            N2_depolarisation_vectorise,
            DATA_HD1[..., 0])).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('N2_depolarisation\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1[..., 0]), a, b))


# N2_depolarisation_profile()

# #############################################################################
# ### colour.phenomenons.rayleigh.O2_depolarisation
# #############################################################################


def O2_depolarisation_2d(wl):
    for i in range(len(wl)):
        O2_depolarisation(wl[i])


def O2_depolarisation_vectorise(wavelength):
    wl = np.asarray(wavelength)

    O2 = (1.096 + 1.385 * 1.0e-3 * (1 / wl ** 2) + 1.448 * 1.0e-4 *
          (1 / wl ** 4))

    return O2


def O2_depolarisation_analysis():
    message_box('O2_depolarisation')

    print('Reference:')
    print(O2_depolarisation(0.555))

    print('\n')

    print('Numeric input:')
    print(O2_depolarisation_vectorise(0.555))

    print('\n')

    print('0d array input:')
    print(O2_depolarisation_vectorise(np.array(0.555)))

    print('\n')

    print('1d array input:')
    wl = [0.555] * 6
    print(O2_depolarisation_vectorise(wl))

    print('\n')

    print('2d array input:')
    wl = np.reshape(wl, (2, 3))
    print(O2_depolarisation_vectorise(wl))

    print('\n')

    print('3d array input:')
    wl = np.reshape(wl, (2, 3, 1))
    print(O2_depolarisation_vectorise(wl))

    print('\n')


# O2_depolarisation_analysis()


def O2_depolarisation_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            O2_depolarisation_2d,
            DATA_HD1[..., 0])).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            O2_depolarisation_vectorise,
            DATA_HD1[..., 0])).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('O2_depolarisation\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1[..., 0]), a, b))


# O2_depolarisation_profile()

# #############################################################################
# ### colour.phenomenons.rayleigh.F_air_Penndorf1957
# #############################################################################


def F_air_Penndorf1957_vectorise(wavelength, *args):
    wl = np.asarray(wavelength)

    return np.resize(np.array([1.0608]), wl.shape)


def F_air_Penndorf1957_analysis():
    message_box('F_air_Penndorf1957')

    print('Reference:')
    print(F_air_Penndorf1957(0.555))

    print('\n')

    print('Numeric input:')
    print(F_air_Penndorf1957_vectorise(0.555))

    print('\n')

    print('0d array input:')
    print(F_air_Penndorf1957_vectorise(np.array(0.555)))

    print('\n')

    print('1d array input:')
    wl = [0.555] * 6
    print(F_air_Penndorf1957_vectorise(wl))

    print('\n')

    print('2d array input:')
    wl = np.reshape(wl, (2, 3))
    print(F_air_Penndorf1957_vectorise(wl))

    print('\n')

    print('3d array input:')
    wl = np.reshape(wl, (2, 3, 1))
    print(F_air_Penndorf1957_vectorise(wl))


# F_air_Penndorf1957_analysis()

# #############################################################################
# ### colour.phenomenons.rayleigh.F_air_Young1981
# #############################################################################


def F_air_Young1981_vectorise(wavelength, *args):
    wl = np.asarray(wavelength)

    return np.resize(np.array([1.0480]), wl.shape)


def F_air_Young1981_analysis():
    message_box('F_air_Young1981')

    print('Reference:')
    print(F_air_Young1981(0.555))

    print('\n')

    print('Numeric input:')
    print(F_air_Young1981_vectorise(0.555))

    print('\n')

    print('0d array input:')
    print(F_air_Young1981_vectorise(np.array(0.555)))

    print('\n')

    print('1d array input:')
    wl = [0.555] * 6
    print(F_air_Young1981_vectorise(wl))

    print('\n')

    print('2d array input:')
    wl = np.reshape(wl, (2, 3))
    print(F_air_Young1981_vectorise(wl))

    print('\n')

    print('3d array input:')
    wl = np.reshape(wl, (2, 3, 1))
    print(F_air_Young1981_vectorise(wl))


# F_air_Young1981_analysis()

# #############################################################################
# ### colour.phenomenons.rayleigh.F_air_Bates1984
# #############################################################################


def F_air_Bates1984_2d(wl):
    for i in range(len(wl)):
        F_air_Bates1984(wl[i])


def F_air_Bates1984_vectorise(wavelength, *args):
    O2 = O2_depolarisation_vectorise(wavelength)
    N2 = N2_depolarisation_vectorise(wavelength)
    Ar = 1.00
    CO2 = 1.15

    F_air = ((78.084 * N2 + 20.946 * O2 + CO2 + Ar) /
             (78.084 + 20.946 + Ar + CO2))

    return F_air


def F_air_Bates1984_analysis():
    message_box('F_air_Bates1984')

    print('Reference:')
    print(F_air_Bates1984(0.555))

    print('\n')

    print('Numeric input:')
    print(F_air_Bates1984_vectorise(0.555))

    print('\n')

    print('0d array input:')
    print(F_air_Bates1984_vectorise(np.array(0.555)))

    print('\n')

    print('1d array input:')
    wl = [0.555] * 6
    print(F_air_Bates1984_vectorise(wl))

    print('\n')

    print('2d array input:')
    wl = np.reshape(wl, (2, 3))
    print(F_air_Bates1984_vectorise(wl))

    print('\n')

    print('3d array input:')
    wl = np.reshape(wl, (2, 3, 1))
    print(F_air_Bates1984_vectorise(wl))

    print('\n')


# F_air_Bates1984_analysis()


def F_air_Bates1984_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            F_air_Bates1984_2d,
            DATA_HD1[..., 0])).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            F_air_Bates1984_vectorise,
            DATA_HD1[..., 0])).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('F_air_Bates1984\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1[..., 0]), a, b))


# F_air_Bates1984_profile()

# #############################################################################
# ### colour.phenomenons.rayleigh.F_air_Bodhaine1999
# #############################################################################


def F_air_Bodhaine1999_2d(wl):
    for i in range(len(wl)):
        F_air_Bodhaine1999(wl[i])


def F_air_Bodhaine1999_vectorise(wavelength,
                                 CO2_concentration=STANDARD_CO2_CONCENTRATION):
    O2 = O2_depolarisation_vectorise(wavelength)
    N2 = N2_depolarisation_vectorise(wavelength)
    CO2_c = np.asarray(CO2_concentration)

    F_air = ((78.084 * N2 + 20.946 * O2 + 0.934 * 1 + CO2_c * 1.15) /
             (78.084 + 20.946 + 0.934 + CO2_c))

    return F_air


def F_air_Bodhaine1999_analysis():
    message_box('F_air_Bodhaine1999')

    print('Reference:')
    print(F_air_Bodhaine1999(0.555))

    print('\n')

    print('Numeric input:')
    print(F_air_Bodhaine1999_vectorise(0.555))

    print('\n')

    print('0d array input:')
    print(F_air_Bodhaine1999_vectorise(np.array(0.555)))

    print('\n')

    print('1d array input:')
    wl = [0.555] * 6
    print(F_air_Bodhaine1999_vectorise(wl))

    print('\n')

    print('2d array input:')
    wl = np.reshape(wl, (2, 3))
    print(F_air_Bodhaine1999_vectorise(wl))

    print('\n')

    print('3d array input:')
    wl = np.reshape(wl, (2, 3, 1))
    print(F_air_Bodhaine1999_vectorise(wl))

    print('\n')


# F_air_Bodhaine1999_analysis()


def F_air_Bodhaine1999_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            F_air_Bodhaine1999_2d,
            DATA_HD1[..., 0])).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            F_air_Bodhaine1999_vectorise,
            DATA_HD1[..., 0])).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('F_air_Bodhaine1999\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1[..., 0]), a, b))


# F_air_Bodhaine1999_profile()

# #############################################################################
# ### colour.phenomenons.rayleigh.molecular_density
# #############################################################################
from colour.constants import AVOGADRO_CONSTANT


def molecular_density_2d(temperature):
    for i in range(len(temperature)):
        molecular_density(temperature[i])


def molecular_density_vectorise(temperature=STANDARD_AIR_TEMPERATURE,
                                avogadro_constant=AVOGADRO_CONSTANT):
    # Review doctests to use coherent temperature values.
    T = np.asarray(temperature)

    N_s = (avogadro_constant / 22.4141) * (273.15 / T) * (1 / 1000)

    return N_s


def molecular_density_analysis():
    message_box('molecular_density')

    print('Reference:')
    print(molecular_density(15))

    print('\n')

    print('Numeric input:')
    print(molecular_density_vectorise(15))

    print('\n')

    print('0d array input:')
    print(molecular_density_vectorise(np.array(15)))

    print('\n')

    print('1d array input:')
    t = [15] * 6
    print(molecular_density_vectorise(t))

    print('\n')

    print('2d array input:')
    t = np.reshape(t, (2, 3))
    print(molecular_density_vectorise(t))

    print('\n')

    print('3d array input:')
    t = np.reshape(t, (2, 3, 1))
    print(molecular_density_vectorise(t))

    print('\n')


# molecular_density_analysis()


def molecular_density_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            molecular_density_2d,
            DATA_HD1[..., 0])).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            molecular_density_vectorise,
            DATA_HD1[..., 0])).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('molecular_density\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1[..., 0]), a, b))


# molecular_density_profile()

# #############################################################################
# ### colour.phenomenons.rayleigh.mean_molecular_weights
# #############################################################################


def mean_molecular_weights_2d(C):
    for i in range(len(C)):
        mean_molecular_weights(C[i])


def mean_molecular_weights_vectorise(
        CO2_concentration=STANDARD_CO2_CONCENTRATION):
    CO2_c = np.asarray(CO2_concentration) * 1.0e-6

    m_a = 15.0556 * CO2_c + 28.9595

    return m_a


def mean_molecular_weights_analysis():
    message_box('mean_molecular_weights')

    print('Reference:')
    print(mean_molecular_weights(300))

    print('\n')

    print('Numeric input:')
    print(mean_molecular_weights_vectorise(300))

    print('\n')

    print('0d array input:')
    print(mean_molecular_weights_vectorise(np.array(300)))

    print('\n')

    print('1d array input:')
    c = [300] * 6
    print(mean_molecular_weights_vectorise(c))

    print('\n')

    print('2d array input:')
    c = np.reshape(c, (2, 3))
    print(mean_molecular_weights_vectorise(c))

    print('\n')

    print('3d array input:')
    c = np.reshape(c, (2, 3, 1))
    print(mean_molecular_weights_vectorise(c))

    print('\n')


# mean_molecular_weights_analysis()


def mean_molecular_weights_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            mean_molecular_weights_2d,
            DATA_HD1[..., 0])).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            mean_molecular_weights_vectorise,
            DATA_HD1[..., 0])).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('mean_molecular_weights\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1[..., 0]), a, b))


# mean_molecular_weights_profile()

# #############################################################################
# ### colour.phenomenons.rayleigh.gravity_List1968
# #############################################################################


def gravity_List1968_2d(C):
    for i in range(len(C)):
        gravity_List1968(C[i])


def gravity_List1968_vectorise(latitude=DEFAULT_LATITUDE,
                               altitude=DEFAULT_ALTITUDE):
    latitude = np.asarray(latitude)
    altitude = np.asarray(altitude)

    cos2phi = np.cos(2 * np.radians(latitude))

    # Sea level acceleration of gravity.
    g0 = 980.6160 * (1 - 0.0026373 * cos2phi + 0.0000059 * cos2phi ** 2)

    g = (g0 - (3.085462e-4 + 2.27e-7 * cos2phi) * altitude +
         (7.254e-11 + 1.0e-13 * cos2phi) * altitude ** 2 -
         (1.517e-17 + 6e-20 * cos2phi) * altitude ** 3)

    return g


def gravity_List1968_analysis():
    message_box('gravity_List1968')

    print('Reference:')
    print(gravity_List1968(0, 0))

    print('\n')

    print('Numeric input:')
    print(gravity_List1968_vectorise(0, 0))

    print('\n')

    print('1d array input:')
    print(gravity_List1968_vectorise([0] * 6, [0]))

    print('\n')

    print('2d array input:')
    l = np.reshape([0] * 6, (2, 3))
    print(gravity_List1968_vectorise(l, [0]))

    print('\n')

    print('2d array input:')
    l = np.reshape([0] * 6, (2, 3, 1))
    print(gravity_List1968_vectorise(l, [0]))

    print('\n')


# gravity_List1968_analysis()


def gravity_List1968_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            gravity_List1968_2d,
            DATA_HD1[..., 0])).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            gravity_List1968_vectorise,
            DATA_HD1[..., 0])).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('gravity_List1968\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1[..., 0]), a, b))


# gravity_List1968_profile()

# #############################################################################
# ### colour.phenomenons.rayleigh.scattering_cross_section
# #############################################################################


def scattering_cross_section_2d(wl):
    for i in range(len(wl)):
        scattering_cross_section(wl[i])


def scattering_cross_section_vectorise(wavelength,
                                       CO2_concentration=STANDARD_CO2_CONCENTRATION,
                                       temperature=STANDARD_AIR_TEMPERATURE,
                                       avogadro_constant=AVOGADRO_CONSTANT,
                                       n_s=air_refraction_index_Bodhaine1999,
                                       F_air=F_air_Bodhaine1999):
    wl = np.asarray(wavelength)
    CO2_c = np.asarray(CO2_concentration)
    temperature = np.asarray(temperature)

    wl_micrometers = wl * 10e3

    n_s = n_s(wl_micrometers)
    N_s = molecular_density(temperature, avogadro_constant)
    F_air = F_air(wl_micrometers, CO2_c)

    sigma = (24 * np.pi ** 3 * (n_s ** 2 - 1) ** 2 /
             (wl ** 4 * N_s ** 2 * (n_s ** 2 + 2) ** 2))
    sigma *= F_air

    return sigma


def scattering_cross_section_analysis():
    message_box('scattering_cross_section')

    print('Reference:')
    print(scattering_cross_section(555 * 10e-8))

    print('\n')

    print('Numeric input:')
    print(scattering_cross_section_vectorise(555 * 10e-8))

    print('\n')

    print('0d array input:')
    print(scattering_cross_section_vectorise(np.array(555 * 10e-8)))

    print('\n')

    print('1d array input:')
    wl = [555 * 10e-8] * 6
    print(scattering_cross_section_vectorise(wl))

    print('\n')

    print('2d array input:')
    wl = np.reshape(wl, (2, 3))
    print(scattering_cross_section_vectorise(wl))

    print('\n')

    print('3d array input:')
    wl = np.reshape(wl, (2, 3, 1))
    print(scattering_cross_section_vectorise(wl))

    print('\n')


# scattering_cross_section_analysis()


def scattering_cross_section_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            scattering_cross_section_2d,
            DATA_HD1[..., 0])).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            scattering_cross_section_vectorise,
            DATA_HD1[..., 0])).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('scattering_cross_section\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1[..., 0]), a, b))


# scattering_cross_section_profile()

# #############################################################################
# ### colour.phenomenons.rayleigh.rayleigh_optical_depth
# #############################################################################


def rayleigh_optical_depth_2d(wl):
    for i in range(len(wl)):
        rayleigh_optical_depth(wl[i])


def rayleigh_optical_depth_vectorise(wavelength,
                                     CO2_concentration=STANDARD_CO2_CONCENTRATION,
                                     temperature=STANDARD_AIR_TEMPERATURE,
                                     pressure=AVERAGE_PRESSURE_MEAN_SEA_LEVEL,
                                     latitude=DEFAULT_LATITUDE,
                                     altitude=DEFAULT_ALTITUDE,
                                     avogadro_constant=AVOGADRO_CONSTANT,
                                     n_s=air_refraction_index_Bodhaine1999,
                                     F_air=F_air_Bodhaine1999):
    wavelength = np.asarray(wavelength)
    CO2_c = np.asarray(CO2_concentration)
    latitude = np.asarray(latitude)
    altitude = np.asarray(altitude)
    # Conversion from pascal to dyne/cm2.
    P = np.asarray(pressure * 10)

    sigma = scattering_cross_section(wavelength,
                                     CO2_c,
                                     temperature,
                                     avogadro_constant,
                                     n_s,
                                     F_air)

    m_a = mean_molecular_weights(CO2_c)
    g = gravity_List1968(latitude, altitude)

    T_R = sigma * (P * avogadro_constant) / (m_a * g)

    return T_R


def rayleigh_optical_depth_analysis():
    message_box('rayleigh_optical_depth')

    print('Reference:')
    print(rayleigh_optical_depth(555 * 10e-8))

    print('\n')

    print('Numeric input:')
    print(rayleigh_optical_depth_vectorise(555 * 10e-8))

    print('\n')

    print('0d array input:')
    print(rayleigh_optical_depth_vectorise(np.array(555 * 10e-8)))

    print('\n')

    print('1d array input:')
    wl = [555 * 10e-8] * 6
    print(rayleigh_optical_depth_vectorise(wl))

    print('\n')

    print('2d array input:')
    wl = np.reshape(wl, (2, 3))
    print(rayleigh_optical_depth_vectorise(wl))

    print('\n')

    print('3d array input:')
    wl = np.reshape(wl, (2, 3, 1))
    print(rayleigh_optical_depth_vectorise(wl))

    print('\n')


# rayleigh_optical_depth_analysis()


def rayleigh_optical_depth_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            rayleigh_optical_depth_2d,
            DATA_HD1[..., 0])).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            rayleigh_optical_depth_vectorise,
            DATA_HD1[..., 0])).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('rayleigh_optical_depth\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1[..., 0]), a, b))


# rayleigh_optical_depth_profile()

# #############################################################################
# ### colour.phenomenons.rayleigh.rayleigh_scattering_spd
# #############################################################################


def rayleigh_scattering_spd_vectorise(shape=DEFAULT_SPECTRAL_SHAPE,
                                      CO2_concentration=STANDARD_CO2_CONCENTRATION,
                                      temperature=STANDARD_AIR_TEMPERATURE,
                                      pressure=AVERAGE_PRESSURE_MEAN_SEA_LEVEL,
                                      latitude=DEFAULT_LATITUDE,
                                      altitude=DEFAULT_ALTITUDE,
                                      avogadro_constant=AVOGADRO_CONSTANT,
                                      n_s=air_refraction_index_Bodhaine1999,
                                      F_air=F_air_Bodhaine1999):
    wavelengths = shape.range()
    return SpectralPowerDistribution(
        name=('Rayleigh Scattering - {0} ppm, {1} K, {2} Pa, {3} Degrees, '
              '{4} m').format(CO2_concentration,
                              temperature,
                              pressure,
                              latitude,
                              altitude),
        data=dict(zip(wavelengths,
                      rayleigh_optical_depth_vectorise(wavelengths * 10e-8,
                                                       CO2_concentration,
                                                       temperature,
                                                       pressure,
                                                       latitude,
                                                       altitude,
                                                       avogadro_constant,
                                                       n_s,
                                                       F_air))))


def rayleigh_scattering_spd_analysis():
    message_box('rayleigh_scattering_spd')

    print(rayleigh_scattering_spd_vectorise().values)

    print('\n')


# rayleigh_scattering_spd_analysis()

# #############################################################################
# #############################################################################
# ### colour.quality.cqs
# #############################################################################
# #############################################################################

# #############################################################################
# ### colour.quality.cqs.gamut_area
# #############################################################################
from colour.quality.cqs import *


def gamut_area_vectorise(Lab):
    Lab = np.asarray(Lab)
    Lab_s = np.roll(np.copy(Lab), -3)

    L, a, b = tsplit(Lab)
    L_s, a_s, b_s = tsplit(Lab_s)

    A = np.linalg.norm(Lab[..., 1:3], axis=-1)
    B = np.linalg.norm(Lab_s[..., 1:3], axis=-1)
    C = np.linalg.norm(np.dstack((a_s - a, b_s - b)), axis=-1)
    t = (A + B + C) / 2
    S = np.sqrt(t * (t - A) * (t - B) * (t - C))

    return np.sum(S)


def gamut_area_vectorise_analysis():
    message_box('gamut_area_vectorise')

    Lab = [np.array([39.94996006, 34.59018231, -19.86046321]),
           np.array([38.88395498, 21.44348519, -34.87805301]),
           np.array([36.60576301, 7.06742454, -43.21461177]),
           np.array([46.60142558, -15.90481586, -34.64616865]),
           np.array([56.50196523, -29.5465555, -20.50177194]),
           np.array([55.73912101, -43.39520959, -5.08956953]),
           np.array([56.2077687, -53.68997662, 20.2113441]),
           np.array([66.16683122, -38.64600327, 42.77396631]),
           np.array([76.7295211, -23.9214821, 61.04740432]),
           np.array([82.85370708, -3.98679065, 75.43320144]),
           np.array([69.26458861, 13.11066359, 68.83858372]),
           np.array([69.63154351, 28.24532497, 59.45609803]),
           np.array([61.26281449, 40.87950839, 44.97606172]),
           np.array([41.62567821, 57.34129516, 27.4671817]),
           np.array([40.52565174, 48.87449192, 3.4512168])]

    print(gamut_area(Lab))

    print(gamut_area_vectorise(Lab))

    print('\n')


# gamut_area_vectorise_analysis()

# #############################################################################
# #############################################################################
# ### colour.temperature.cct
# #############################################################################
# #############################################################################

# #############################################################################
# ### colour.xy_to_CCT_McCamy1992
# #############################################################################
from colour.temperature.cct import *


def xy_to_CCT_McCamy1992_2d(xy):
    for i in range(len(xy)):
        xy_to_CCT_McCamy1992(xy[i])


def xy_to_CCT_McCamy1992_vectorise(xy):
    x, y = tsplit(xy)

    n = (x - 0.3320) / (y - 0.1858)
    CCT = -449 * n ** 3 + 3525 * n ** 2 - 6823.3 * n + 5520.33

    return CCT


def xy_to_CCT_McCamy1992_analysis():
    message_box('xy_to_CCT_McCamy1992')

    print('Reference:')
    xy = np.array([0.31271, 0.32902])
    print(xy_to_CCT_McCamy1992(xy))

    print('\n')

    print('1d array input:')
    print(xy_to_CCT_McCamy1992_vectorise(xy))

    print('\n')

    print('2d array input:')
    xy = np.tile(xy, (6, 1))
    print(xy_to_CCT_McCamy1992_vectorise(xy))

    print('\n')

    print('3d array input:')
    xy = np.reshape(xy, (2, 3, 2))
    print(xy_to_CCT_McCamy1992_vectorise(xy))

    print('\n')


# xy_to_CCT_McCamy1992_analysis()


def xy_to_CCT_McCamy1992_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            xy_to_CCT_McCamy1992_2d,
            DATA_HD1[..., 0:2])).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            xy_to_CCT_McCamy1992_vectorise,
            DATA_HD1[..., 0:2])).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('xy_to_CCT_McCamy1992\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1[..., 0:2]), a, b))


# xy_to_CCT_McCamy1992_profile()

# #############################################################################
# ### colour.xy_to_CCT_Hernandez1999
# #############################################################################


def xy_to_CCT_Hernandez1999_2d(xy):
    CCT = []
    for i in range(len(xy)):
        CCT.append(xy_to_CCT_Hernandez1999(xy[i]))
    return CCT


def xy_to_CCT_Hernandez1999_vectorise(xy):
    x, y = tsplit(xy)

    n = (x - 0.3366) / (y - 0.1735)
    CCT = (-949.86315 +
           6253.80338 * np.exp(-n / 0.92159) +
           28.70599 * np.exp(-n / 0.20039) +
           0.00004 * np.exp(-n / 0.07125))

    n = np.where(CCT > 50000,
                 (x - 0.3356) / (y - 0.1691),
                 n)

    CCT = np.where(CCT > 50000,
                   36284.48953 + 0.00228 * np.exp(-n / 0.07861) +
                   5.4535e-36 * np.exp(-n / 0.01543),
                   CCT)

    return CCT


def xy_to_CCT_Hernandez1999_analysis():
    message_box('xy_to_CCT_Hernandez1999')

    print('Reference:')
    xy = np.array([0.31271, 0.32902])
    print(xy_to_CCT_Hernandez1999(xy))

    print('\n')

    print('1d array input:')
    print(xy_to_CCT_Hernandez1999_vectorise(xy))

    print('\n')

    print('2d array input:')
    xy = np.tile(xy, (6, 1))
    print(xy_to_CCT_Hernandez1999_vectorise(xy))

    print('\n')

    print('3d array input:')
    xy = np.reshape(xy, (2, 3, 2))
    print(xy_to_CCT_Hernandez1999_vectorise(xy))

    print('\n')

    np.testing.assert_almost_equal(
        np.ravel(xy_to_CCT_Hernandez1999_2d(DATA1[:, 0:2])),
        np.ravel(xy_to_CCT_Hernandez1999_vectorise(DATA1[:, 0:2])))


# xy_to_CCT_Hernandez1999_analysis()


def xy_to_CCT_Hernandez1999_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            xy_to_CCT_Hernandez1999_2d,
            DATA_HD1[..., 0:2])).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            xy_to_CCT_Hernandez1999_vectorise,
            DATA_HD1[..., 0:2])).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('xy_to_CCT_Hernandez1999\t{0}\t{1}\t{2}'.format(
        len(DATA_HD1[..., 0:2]), a, b))


# xy_to_CCT_Hernandez1999_profile()

# #############################################################################
# ### colour.CCT_to_xy_Kang2002
# #############################################################################
CCT = np.linspace(4000, 20000, 1000000)


def CCT_to_xy_Kang2002_2d(CCT):
    xy = []
    for i in range(len(CCT)):
        xy.append(CCT_to_xy_Kang2002(CCT[i]))
    return xy


def CCT_to_xy_Kang2002_vectorise(CCT):
    CCT = np.asarray(CCT)

    if np.any(CCT[np.asarray(np.logical_or(CCT < 1667, CCT > 25000))]):
        warning(('Correlated colour temperature must be in domain '
                 '[1667, 25000], unpredictable results may occur!'))

    x = np.where(CCT <= 4000,
                 -0.2661239 * 10 ** 9 / CCT ** 3 -
                 0.2343589 * 10 ** 6 / CCT ** 2 +
                 0.8776956 * 10 ** 3 / CCT +
                 0.179910,
                 -3.0258469 * 10 ** 9 / CCT ** 3 +
                 2.1070379 * 10 ** 6 / CCT ** 2 +
                 0.2226347 * 10 ** 3 / CCT +
                 0.24039)

    y = np.select([CCT <= 2222,
                   np.logical_and(CCT > 2222, CCT <= 4000),
                   CCT > 4000],
                  [-1.1063814 * x ** 3 -
                   1.34811020 * x ** 2 +
                   2.18555832 * x -
                   0.20219683,
                   -0.9549476 * x ** 3 -
                   1.37418593 * x ** 2 +
                   2.09137015 * x -
                   0.16748867,
                   3.0817580 * x ** 3 -
                   5.8733867 * x ** 2 +
                   3.75112997 * x -
                   0.37001483])

    xy = tstack((x, y))

    return xy


def CCT_to_xy_Kang2002_analysis():
    message_box('CCT_to_xy_Kang2002')

    print('Reference:')
    print(CCT_to_xy_Kang2002(6504.38938305))

    print('\n')

    print('Numeric input:')
    print(CCT_to_xy_Kang2002_vectorise(6504.38938305))

    print('\n')

    print('0d array input:')
    print(CCT_to_xy_Kang2002_vectorise(np.array(6504.38938305)))

    print('\n')

    print('1d array input:')
    CCT = [6504.38938305] * 6
    print(CCT_to_xy_Kang2002_vectorise(CCT))

    print('\n')

    print('2d array input:')
    CCT = np.reshape(CCT, (2, 3))
    print(CCT_to_xy_Kang2002_vectorise(CCT))

    print('\n')

    print('3d array input:')
    CCT = np.reshape(CCT, (2, 3, 1))
    print(CCT_to_xy_Kang2002_vectorise(CCT))

    print('\n')

    CCT = np.linspace(1667, 25000, 10000)
    np.testing.assert_almost_equal(
        np.ravel(CCT_to_xy_Kang2002_2d(CCT)),
        np.ravel(CCT_to_xy_Kang2002_vectorise(CCT)))


# CCT_to_xy_Kang2002_analysis()


def CCT_to_xy_Kang2002_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            CCT_to_xy_Kang2002_2d,
            CCT)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            CCT_to_xy_Kang2002_vectorise,
            CCT)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('CCT_to_xy_Kang2002\t{0}\t{1}\t{2}'.format(
        len(CCT), a, b))


# CCT_to_xy_Kang2002_profile()

# #############################################################################
# ### colour.CCT_to_xy_CIE_D
# #############################################################################


def CCT_to_xy_CIE_D_2d(CCT):
    xy = []
    for i in range(len(CCT)):
        xy.append(CCT_to_xy_CIE_D(CCT[i]))
    return xy


def CCT_to_xy_CIE_D_vectorise(CCT):
    CCT = np.asarray(CCT)

    if np.any(CCT[np.asarray(np.logical_or(CCT < 4000, CCT > 25000))]):
        warning(('Correlated colour temperature must be in domain '
                 '[4000, 25000], unpredictable results may occur!'))

    x = np.where(CCT <= 7000,
                 -4.607 * 10 ** 9 / CCT ** 3 +
                 2.9678 * 10 ** 6 / CCT ** 2 +
                 0.09911 * 10 ** 3 / CCT +
                 0.244063,
                 -2.0064 * 10 ** 9 / CCT ** 3 +
                 1.9018 * 10 ** 6 / CCT ** 2 +
                 0.24748 * 10 ** 3 / CCT +
                 0.23704)

    y = -3 * x ** 2 + 2.87 * x - 0.275

    xy = tstack((x, y))

    return xy


def CCT_to_xy_CIE_D_analysis():
    message_box('CCT_to_xy_CIE_D')

    print('Reference:')
    print(CCT_to_xy_CIE_D(6504.38938305))

    print('\n')

    print('Numeric input:')
    print(CCT_to_xy_CIE_D_vectorise(6504.38938305))

    print('\n')

    print('0d array input:')
    print(CCT_to_xy_CIE_D_vectorise(np.array(6504.38938305)))

    print('\n')

    print('1d array input:')
    CCT = [6504.38938305] * 6
    print(CCT_to_xy_CIE_D_vectorise(CCT))

    print('\n')

    print('2d array input:')
    CCT = np.reshape(CCT, (2, 3))
    print(CCT_to_xy_CIE_D_vectorise(CCT))

    print('\n')

    print('3d array input:')
    CCT = np.reshape(CCT, (2, 3, 1))
    print(CCT_to_xy_CIE_D_vectorise(CCT))

    print('\n')

    CCT = np.linspace(4000, 25000, 10000)
    np.testing.assert_almost_equal(
        np.ravel(CCT_to_xy_CIE_D_2d(CCT)),
        np.ravel(CCT_to_xy_CIE_D_vectorise(CCT)))


# CCT_to_xy_CIE_D_analysis()


def CCT_to_xy_CIE_D_profile(
        repeat_a=3, number_a=5, repeat_b=3, number_b=10):
    times = timeit.Timer(
        functools.partial(
            CCT_to_xy_CIE_D_2d,
            CCT)).repeat(repeat_a, number_a)

    a = min(times) / number_a

    times = timeit.Timer(
        functools.partial(
            CCT_to_xy_CIE_D_vectorise,
            CCT)).repeat(repeat_b, number_b)

    b = min(times) / number_b

    print('CCT_to_xy_CIE_D\t{0}\t{1}\t{2}'.format(
        len(CCT), a, b))


# CCT_to_xy_CIE_D_profile()
