# -*- coding: utf-8 -*-
"""
Fresnel Complex Equations
=========================

Implements support for *Fresnel Complex Equations*:

-   :func:``

See Also
--------
`Fresnel Complex Equations Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/phenomena/fresnel.ipynb>`_

References
----------
-   :cite:`` :
"""

from __future__ import division, unicode_literals

import numpy as np
from scipy import arcsin

from colour.constants import DEFAULT_COMPLEX_DTYPE
from colour.utilities import (tsplit, tstack, as_float_array, as_complex_array,
                              dot_vector)


def snell_law(n_1, n_2, theta_i):
    n_1 = as_complex_array(n_1).real
    n_2 = as_complex_array(n_2).real
    theta_i = np.radians(theta_i)

    return np.degrees(arcsin(n_1 * np.sin(theta_i) / n_2))


# https://en.wikipedia.org/wiki/Fresnel_equations


def _polarised_light_magnitude_elements(n_1, n_2, theta_i, theta_t):
    n_1 = as_complex_array(n_1)
    n_2 = as_complex_array(n_2)

    cos_theta_i = np.cos(np.radians(theta_i))
    cos_theta_t = np.cos(np.radians(theta_t))

    n_1_cos_theta_i = n_1 * cos_theta_i
    n_1_cos_theta_t = n_1 * cos_theta_t
    n_2_cos_theta_i = n_2 * cos_theta_i
    n_2_cos_theta_t = n_2 * cos_theta_t

    return n_1_cos_theta_i, n_1_cos_theta_t, n_2_cos_theta_i, n_2_cos_theta_t


def polarised_light_reflection_magnitude(n_1, n_2, theta_i, theta_t):
    n_1_cos_theta_i, n_1_cos_theta_t, n_2_cos_theta_i, n_2_cos_theta_t = (
        _polarised_light_magnitude_elements(n_1, n_2, theta_i, theta_t))

    r_s = ((n_1_cos_theta_i - n_2_cos_theta_t) /
           (n_1_cos_theta_i + n_2_cos_theta_t))

    r_p = ((n_2_cos_theta_i - n_1_cos_theta_t) /
           (n_2_cos_theta_i + n_1_cos_theta_t))

    return tstack([r_s, r_p], dtype=DEFAULT_COMPLEX_DTYPE)


def polarised_light_reflection_coefficient(n_1, n_2, theta_i, theta_t):
    R = np.abs(
        polarised_light_reflection_magnitude(n_1, n_2, theta_i, theta_t)) ** 2

    return as_complex_array(R)


def polarised_light_transmission_magnitude(n_1, n_2, theta_i, theta_t):
    n_1_cos_theta_i, n_1_cos_theta_t, n_2_cos_theta_i, n_2_cos_theta_t = (
        _polarised_light_magnitude_elements(n_1, n_2, theta_i, theta_t))

    _2_n_1_cos_theta_i = 2 * n_1_cos_theta_i

    t_s = (_2_n_1_cos_theta_i / (n_1_cos_theta_i + n_2_cos_theta_t))

    t_p = (_2_n_1_cos_theta_i / (n_2_cos_theta_i + n_1_cos_theta_t))

    return tstack([t_s, t_p], dtype=DEFAULT_COMPLEX_DTYPE)


def polarised_light_transmission_coefficient(n_1, n_2, theta_i, theta_t):
    n_1 = as_complex_array(n_1)
    n_2 = as_complex_array(n_2)

    n_1_cos_theta_i, n_1_cos_theta_t, n_2_cos_theta_i, n_2_cos_theta_t = (
        _polarised_light_magnitude_elements(n_1, n_2, theta_i, theta_t))

    T = (n_2_cos_theta_t / n_1_cos_theta_i)[..., np.newaxis] * np.abs(
        polarised_light_transmission_magnitude(n_1, n_2, theta_i,
                                               theta_t)) ** 2
    return as_complex_array(T)


# Matrix of refraction
def interface_matrix(n_1, n_2, theta_1, theta_2):
    r_jk = polarised_light_reflection_magnitude(n_1, n_2, theta_1, theta_2)
    r_jk_s, r_jk_p = tsplit(r_jk)
    t_jk = polarised_light_transmission_magnitude(n_1, n_2, theta_1, theta_2)
    t_jk_s, t_jk_p = tsplit(t_jk)

    ones = np.ones(r_jk_s.shape)

    I_jk = dot_vector([
        [ones, r_jk_s],
        [r_jk_s, ones],
    ], ones / t_jk_s)

    return I_jk


# Phase matrix
def layer_matrix(n_1, n_2, theta_i, wl, d_j):
    n_1 = as_complex_array(n_1)
    n_2 = as_complex_array(n_2)
    theta_i = np.radians(theta_i)
    wl = as_float_array(wl)
    d_j = as_float_array(d_j)

    q_j = np.sqrt(n_2 ** 2 - n_1 ** 2 * np.sin(theta_i))
    xi_j = 2 * np.pi / wl * q_j
    xi_j_d_j = xi_j * d_j

    L_jk = [
        [np.exp(-xi_j_d_j), 0],
        [0, np.exp(xi_j_d_j)],
    ]

    return L_jk


n_1 = [1.5, 1.5, 1.5, 1.5]
n_2 = np.sqrt(n_1)
d = 700 / n_2 / 4
theta_i = 45

theta_t = snell_law(n_1, n_2, theta_i)
print(theta_t)
print(polarised_light_reflection_magnitude(n_1, n_2, theta_i, theta_t))
print(polarised_light_transmission_magnitude(n_1, n_2, theta_i, theta_t))
print(polarised_light_reflection_coefficient(n_1, n_2, theta_i, theta_t))
print(polarised_light_transmission_coefficient(n_1, n_2, theta_i, theta_t))
# print(
#     polarised_light_reflection_magnitude([1 + 0.5j, 1 + 0.5j],
#                                            [2 + 0.5j, 2 + 0.5j], 45))
# print(polarised_light_transmission_magnitude(1 + 0.5j, 2 + 0.5j, 45))
# print(interface_matrix(1 + 0.5j, 2 + 0.5j, 45))
# print(layer_matrix(1 + 0.5j, 2 + 0.5j, 45, 555, ))

# print(interface_matrix(n_1, n_2, 0, 0))
# print(interface_matrix(n_2, 1, 0))
# print(layer_matrix(n_1, n_2, 0, 400, d))
# print(layer_matrix(n_2, 1, 0, 400, d))
