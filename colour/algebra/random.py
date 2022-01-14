# -*- coding: utf-8 -*-
"""
Random Numbers Utilities
========================

Defines the random number generator objects:

-   :func:`colour.algebra.random_triplet_generator`
References
----------
-   :cite:`Laurent2012a` : Laurent. (2012). Reproducibility of python
    pseudo-random numbers across systems and versions? Retrieved January 20,
    2015, from
    http://stackoverflow.com/questions/8786084/\
reproducibility-of-python-pseudo-random-numbers-across-systems-and-versions
"""

from __future__ import annotations

import numpy as np

from colour.hints import ArrayLike, Integer, NDArray
from colour.utilities import as_float_array, tstack

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'RANDOM_STATE',
    'random_triplet_generator',
]

RANDOM_STATE = np.random.RandomState()


def random_triplet_generator(
        size: Integer,
        limits: ArrayLike = np.array([[0, 1], [0, 1], [0, 1]]),
        random_state: np.random.RandomState = RANDOM_STATE) -> NDArray:
    """
    Returns a generator yielding random triplets.

    Parameters
    ----------
    size
        Generator size.
    limits
        Random values limits on each triplet axis.
    random_state
         Mersenne Twister pseudo-random number generator.

    Returns
    -------
    :class:`numpy.ndarray`
        Random triplet generator.

    Notes
    -----
    -   The test is assuming that :func:`np.random.RandomState` definition
        will return the same sequence no matter which *OS* or *Python* version
        is used. There is however no formal promise about the *prng* sequence
        reproducibility of either *Python* or *Numpy* implementations, see
        :cite:`Laurent2012a`.

    Examples
    --------
    >>> from pprint import pprint
    >>> prng = np.random.RandomState(4)
    >>> random_triplet_generator(10, random_state=prng)
    ... # doctest: +ELLIPSIS
    array([[ 0.9670298...,  0.7793829...,  0.4361466...],
           [ 0.5472322...,  0.1976850...,  0.9489773...],
           [ 0.9726843...,  0.8629932...,  0.7863059...],
           [ 0.7148159...,  0.9834006...,  0.8662893...],
           [ 0.6977288...,  0.1638422...,  0.1731654...],
           [ 0.2160895...,  0.5973339...,  0.0749485...],
           [ 0.9762744...,  0.0089861...,  0.6007427...],
           [ 0.0062302...,  0.3865712...,  0.1679721...],
           [ 0.2529823...,  0.0441600...,  0.7333801...],
           [ 0.4347915...,  0.9566529...,  0.4084438...]])
    """

    limit_x, limit_y, limit_z = as_float_array(limits)

    return tstack([
        random_state.uniform(limit_x[0], limit_x[1], size=size),
        random_state.uniform(limit_y[0], limit_y[1], size=size),
        random_state.uniform(limit_z[0], limit_z[1], size=size),
    ])
