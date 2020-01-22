# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.algebra.random` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.algebra import random_triplet_generator

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['RANDOM_TRIPLETS', 'TestRandomTripletGenerator']

RANDOM_TRIPLETS = np.array([
    [0.96702984, 0.77938292, 0.43614665],
    [0.54723225, 0.19768507, 0.94897731],
    [0.97268436, 0.86299324, 0.78630599],
    [0.71481599, 0.98340068, 0.86628930],
    [0.69772882, 0.16384224, 0.17316542],
    [0.21608950, 0.59733394, 0.07494859],
    [0.97627445, 0.00898610, 0.60074272],
    [0.00623026, 0.38657128, 0.16797218],
    [0.25298236, 0.04416006, 0.73338017],
    [0.43479153, 0.95665297, 0.40844386],
])


class TestRandomTripletGenerator(unittest.TestCase):
    """
    Defines :func:`colour.algebra.random.random_triplet_generator` definition
    unit tests methods.
    """

    def test_random_triplet_generator(self):
        """
        Tests :func:`colour.algebra.random.random_triplet_generator`
        definition.

        Notes
        -----
        -   The test is assuming that :func:`np.random.RandomState` definition
            will return the same sequence no matter which *OS* or *Python*
            version is used. There is however no formal promise about the
            *prng* sequence reproducibility of either *Python* or *Numpy*
            implementations, see :cite:`Laurent2012a`.
        """

        prng = np.random.RandomState(4)
        np.testing.assert_almost_equal(
            RANDOM_TRIPLETS,
            random_triplet_generator(10, random_state=prng),
            decimal=7)

        # TODO: Use "assertWarns" when dropping Python 2.7.
        random_triplet_generator(5.5, random_state=prng)


if __name__ == '__main__':
    unittest.main()
