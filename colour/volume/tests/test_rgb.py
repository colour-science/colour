"""
Define the unit tests for the :mod:`colour.volume.rgb` module.

Notes
-----
The MonteCarlo sampling based unit tests are assuming that
:func:`np.random.RandomState` definition will return the same sequence no
matter which *OS* or *Python* version is used. There is however no formal
promise about the *prng* sequence reproducibility of either *Python* or *Numpy*
implementations:

References
----------
-   :cite:`Laurent2012a` : Laurent. (2012). Reproducibility of python
    pseudo-random numbers across systems and versions? Retrieved January 20,
    2015, from
    http://stackoverflow.com/questions/8786084/\
reproducibility-of-python-pseudo-random-numbers-across-systems-and-versions
"""


import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models import (
    RGB_COLOURSPACE_BT709,
)
from colour.volume import (
    RGB_colourspace_pointer_gamut_coverage_MonteCarlo,
    RGB_colourspace_visible_spectrum_coverage_MonteCarlo,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestRGB_colourspacePointerGamutCoverageMonteCarlo",
    "TestRGB_colourspaceVisibleSpectrumCoverageMonteCarlo",
]


class TestRGB_colourspacePointerGamutCoverageMonteCarlo:
    """
    Define :func:`colour.volume.rgb.\
RGB_colourspace_pointer_gamut_coverage_MonteCarlo` definition unit tests
    methods.

    References
    ----------
    :cite:`Laurent2012a`
    """

    def test_RGB_colourspace_pointer_gamut_coverage_MonteCarlo(self):
        """
        Test :func:`colour.volume.rgb.\
RGB_colourspace_pointer_gamut_coverage_MonteCarlo` definition.
        """

        np.testing.assert_allclose(
            RGB_colourspace_pointer_gamut_coverage_MonteCarlo(
                RGB_COLOURSPACE_BT709,
                10e3,
                random_state=np.random.RandomState(2),
            ),
            81.044349070100140,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestRGB_colourspaceVisibleSpectrumCoverageMonteCarlo:
    """
    Define :func:`colour.volume.rgb.\
RGB_colourspace_visible_spectrum_coverage_MonteCarlo` definition unit tests
    methods.

    References
    ----------
    :cite:`Laurent2012a`
    """

    def test_RGB_colourspace_visible_spectrum_coverage_MonteCarlo(self):
        """
        Test :func:`colour.volume.rgb.\
RGB_colourspace_visible_spectrum_coverage_MonteCarlo` definition.
        """

        np.testing.assert_allclose(
            RGB_colourspace_visible_spectrum_coverage_MonteCarlo(
                RGB_COLOURSPACE_BT709,
                10e3,
                random_state=np.random.RandomState(2),
            ),
            46.931407942238266,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
