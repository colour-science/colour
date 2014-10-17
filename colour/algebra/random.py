#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Random Numbers Utilities
========================

Defines random numbers generator objects:

-   :func:`random_triplet_generator`
"""

from __future__ import division, unicode_literals

import numpy as np

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['RANDOM_STATE',
           'random_triplet_generator']

RANDOM_STATE = np.random.RandomState()


def random_triplet_generator(size,
                             limits=np.array([[0, 1], [0, 1], [0, 1]]),
                             random_state=RANDOM_STATE):
    """
    Returns a generator yielding random triplets.

    Parameters
    ----------
    size : integer
        Generator size.
    limits : array_like, (3, 2)
        Random values limits on each triplet axis.
    random_state : RandomState
         Mersenne Twister pseudo-random number generator.

    Returns
    -------
    generator
        Random triplets generator.

    Examples
    --------
    >>> from pprint import pprint
    >>> prng = np.random.RandomState(4)
    >>> pprint(list(random_triplet_generator(10, random_state=prng)))  # noqa  # doctest: +ELLIPSIS
    [array([ 0.9670298...,  0.5472322...,  0.9726843...]),
     array([ 0.7148159...,  0.6977288...,  0.2160895...]),
     array([ 0.9762744...,  0.0062302...,  0.2529823...]),
     array([ 0.4347915...,  0.7793829...,  0.1976850...]),
     array([ 0.8629932...,  0.9834006...,  0.1638422...]),
     array([ 0.5973339...,  0.0089861...,  0.3865712...]),
     array([ 0.0441600...,  0.9566529...,  0.4361466...]),
     array([ 0.9489773...,  0.7863059...,  0.8662893...]),
     array([ 0.1731654...,  0.0749485...,  0.6007427...]),
     array([ 0.1679721...,  0.7333801...,  0.4084438...])]
    """

    x_limits, y_limits, z_limits = limits

    for _ in range(size):
        yield np.array((random_state.uniform(*x_limits),
                        random_state.uniform(*y_limits),
                        random_state.uniform(*z_limits)))
