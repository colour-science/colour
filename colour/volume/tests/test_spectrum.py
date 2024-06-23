"""Define the unit tests for the :mod:`colour.volume.spectrum` module."""


import numpy as np

from colour.volume import (
    is_within_visible_spectrum,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestIsWithinVisibleSpectrum",
]


class TestIsWithinVisibleSpectrum:
    """
    Define :func:`colour.volume.spectrum.is_within_visible_spectrum`
    definition unit tests methods.
    """

    def test_is_within_visible_spectrum(self):
        """
        Test :func:`colour.volume.spectrum.is_within_visible_spectrum`
        definition.
        """

        assert is_within_visible_spectrum(np.array([0.3205, 0.4131, 0.5100]))

        assert not is_within_visible_spectrum(np.array([-0.0005, 0.0031, 0.0010]))

        assert is_within_visible_spectrum(np.array([0.4325, 0.3788, 0.1034]))

        assert not is_within_visible_spectrum(np.array([0.0025, 0.0088, 0.0340]))
