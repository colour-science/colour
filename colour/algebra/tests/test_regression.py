# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.algebra.regression` module."""

import unittest

import numpy as np

from colour.algebra import least_square_mapping_MoorePenrose
from colour.constants import TOLERANCE_ABSOLUTE_TESTS

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestLeastSquareMappingMoorePenrose",
]


class TestLeastSquareMappingMoorePenrose(unittest.TestCase):
    """
    Define :func:`colour.algebra.regression.\
least_square_mapping_MoorePenrose` definition unit tests methods.
    """

    def test_least_square_mapping_MoorePenrose(self):
        """
        Test :func:`colour.algebra.regression.\
least_square_mapping_MoorePenrose` definition.
        """

        prng = np.random.RandomState(2)
        y = prng.random_sample((24, 3))
        x = y + (prng.random_sample((24, 3)) - 0.5) * 0.5

        np.testing.assert_allclose(
            least_square_mapping_MoorePenrose(y, x),
            np.array(
                [
                    [1.05263767, 0.13780789, -0.22763399],
                    [0.07395843, 1.02939945, -0.10601150],
                    [0.05725508, -0.20526336, 1.10151945],
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        y = prng.random_sample((4, 3, 2))
        x = y + (prng.random_sample((4, 3, 2)) - 0.5) * 0.5
        np.testing.assert_allclose(
            least_square_mapping_MoorePenrose(y, x),
            np.array(
                [
                    [
                        [
                            [1.05968114, -0.0896093, -0.02923021],
                            [3.77254737, 0.06682885, -2.78161763],
                        ],
                        [
                            [-0.77388532, 1.78761209, -0.44050114],
                            [-4.1282882, 0.55185528, 5.049136],
                        ],
                        [
                            [0.36246422, -0.56421525, 1.4208154],
                            [2.07589501, 0.40261387, -1.47059455],
                        ],
                    ],
                    [
                        [
                            [0.237067, 0.4794514, 0.04004058],
                            [0.67778963, 0.15901967, 0.23854131],
                        ],
                        [
                            [-0.4225357, 0.99316309, -0.14598921],
                            [-3.46789045, 1.09102153, 3.31051434],
                        ],
                        [
                            [-0.91661817, 1.49060435, -0.45074387],
                            [-4.18896905, 0.25487186, 4.75951391],
                        ],
                    ],
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


if __name__ == "__main__":
    unittest.main()
