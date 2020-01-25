# -*- coding: utf-8 -*-
"""
Random Numbers Utilities
========================

Defines random numbers generator objects:

-   :func:`colour.algebra.random_triplet_generator`
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.constants import DEFAULT_INT_DTYPE
from colour.utilities import runtime_warning, tstack

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['RANDOM_STATE', 'random_triplet_generator']

RANDOM_STATE = np.random.RandomState()


def random_triplet_generator(size,
                             limits=np.array([[0, 1], [0, 1], [0, 1]]),
                             random_state=RANDOM_STATE):
    """
    Returns a generator yielding random triplets.

    Parameters
    ----------
    size : int
        Generator size.
    limits : array_like, (3, 2)
        Random values limits on each triplet axis.
    random_state : RandomState
         Mersenne Twister pseudo-random number generator.

    Returns
    -------
    generator
        Random triplets generator.

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

    integer_size = DEFAULT_INT_DTYPE(size)
    if integer_size != size:
        runtime_warning(
            '"size" has been cast to integer: {0}'.format(integer_size))

    return tstack([
        random_state.uniform(*limits[0], size=integer_size),
        random_state.uniform(*limits[1], size=integer_size),
        random_state.uniform(*limits[2], size=integer_size),
    ])
