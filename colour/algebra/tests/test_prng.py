"""
Define the unit tests for the :mod:`colour.algebra.prng` module.

References
----------
-   :cite:`Laurent2012a` : Laurent. (2012). Reproducibility of python
    pseudo-random numbers across systems and versions? Retrieved January 20,
    2015, from
    http://stackoverflow.com/questions/8786084/\
reproducibility-of-python-pseudo-random-numbers-across-systems-and-versions
"""

from __future__ import annotations

import unittest

import numpy as np

from colour.algebra import random_triplet_generator
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.hints import NDArrayFloat

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "RANDOM_TRIPLETS",
    "TestRandomTripletGenerator",
]

RANDOM_TRIPLETS: NDArrayFloat = np.array(
    [
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
    ]
)


class TestRandomTripletGenerator(unittest.TestCase):
    """
    Define :func:`colour.algebra.prng.random_triplet_generator` definition
    unit tests methods.
    """

    def test_random_triplet_generator(self):
        """
        Test :func:`colour.algebra.prng.random_triplet_generator` definition.

        Notes
        -----
        -   The test is assuming that :func:`np.random.RandomState` definition
            will return the same sequence no matter which *OS* or *Python*
            version is used. There is however no formal promise about the
            *prng* sequence reproducibility of either *Python* or *Numpy*
            implementations, see :cite:`Laurent2012a`.
        """

        prng = np.random.RandomState(4)
        np.testing.assert_allclose(
            RANDOM_TRIPLETS,
            random_triplet_generator(10, random_state=prng),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


if __name__ == "__main__":
    unittest.main()
