# -*- coding: utf-8 -*-
"""
Optimization Utilities
======================

Defines various optimization utilities and objects:

-   :func:`colour.algebra.spow`: Safe (symmetrical) power.
-   :func:`colour.algebra.smoothstep_function`: *Smoothstep* sigmoid-like
    function.

References
----------
-   :cite:`` :
"""

import numpy as np

from colour.utilities import as_float_array, vector_dot, runtime_warning

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['minimize_NewtonRaphson']


def minimize_NewtonRaphson(x,
                           x_0,
                           x_d,
                           gradient_callable,
                           print_callable=lambda x: x,
                           tolerance=1e-10,
                           max_iterations=int(1e3)):
    x = as_float_array(x)
    shape = x.shape
    x_0 = as_float_array(x_0)
    x_d = as_float_array(x_d)

    x = np.reshape(x, [-1, shape[-1]])
    x_0 = np.resize(x_0, x.shape)
    x_d = np.resize(x_d, x.shape)

    i = 0
    while i < max_iterations:
        print_callable('\nIteration {0} of {1}:'.format(i + 1, max_iterations))

        x_1 = np.copy(x_0)
        x_2 = np.copy(x_0)
        x_3 = np.copy(x_0)

        x_1[..., 0] = x_0[..., 0] + x_d[..., 0]
        x_2[..., 1] = x_0[..., 1] + x_d[..., 1]
        x_3[..., 2] = x_0[..., 2] + x_d[..., 2]

        y_0 = gradient_callable(x_0)
        y_1 = gradient_callable(x_1)
        y_2 = gradient_callable(x_2)
        y_3 = gradient_callable(x_3)

        M_j = as_float_array([[
            (y_1[..., j] - y_0[..., j]) / x_d[..., j],
            (y_2[..., j] - y_0[..., j]) / x_d[..., j],
            (y_3[..., j] - y_0[..., j]) / x_d[..., j],
        ] for j in range(shape[-1])])

        M_j = np.rollaxis(M_j, -1)

        print_callable('\nJacobian matrix:\n{0}'.format(M_j))

        try:
            x_j = vector_dot(np.linalg.inv(M_j), x - y_0) + x_0
        except np.linalg.LinAlgError as error:
            runtime_warning(
                'Linear Algebra-related condition occured at iteration {0}: '
                '"{1}", breaking loop!'.format(i + 1, error))
            break

        error = np.abs(x_j - x_0)

        print_callable('\nError:\n{0}'.format(error))

        if np.all(error <= tolerance):
            break
        else:
            x_0 = x_j

        i += 1

    return np.reshape(x_0, shape)
