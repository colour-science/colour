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

from __future__ import annotations

import typing
from contextlib import nullcontext

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models import (
    RGB_COLOURSPACE_ACES2065_1,
    RGB_COLOURSPACE_BT709,
    RGB_COLOURSPACE_BT2020,
)
from colour.utilities import (
    xp_as_array,
    xp_as_float_array,
    xp_assert_close,
)
from colour.volume import (
    RGB_colourspace_limits,
    RGB_colourspace_pointer_gamut_coverage_MonteCarlo,
    RGB_colourspace_visible_spectrum_coverage_MonteCarlo,
    RGB_colourspace_volume_coverage_MonteCarlo,
    RGB_colourspace_volume_MonteCarlo,
    is_within_pointer_gamut,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestRGB_colourspaceLimits",
    "TestRGB_colourspaceVolumeMonteCarlo",
    "TestRGB_colourspace_volume_coverage_MonteCarlo",
    "TestRGB_colourspacePointerGamutCoverageMonteCarlo",
    "TestRGB_colourspaceVisibleSpectrumCoverageMonteCarlo",
]


class TestRGB_colourspaceLimits:
    """
    Define :func:`colour.volume.rgb.RGB_colourspace_limits` definition unit
    tests methods.
    """

    def test_RGB_colourspace_limits(self) -> None:
        """Test :func:`colour.volume.rgb.RGB_colourspace_limits` definition."""

        xp_assert_close(
            RGB_colourspace_limits(RGB_COLOURSPACE_BT709),
            [
                [0.00000000, 100.00000000],
                [-86.18159689, 98.23744381],
                [-107.85546554, 94.48384002],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            RGB_colourspace_limits(RGB_COLOURSPACE_BT2020),
            [
                [0.00000000, 100.00000000],
                [-172.32005590, 130.52657313],
                [-120.27412558, 136.88564561],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            RGB_colourspace_limits(RGB_COLOURSPACE_ACES2065_1),
            [
                [-65.15706201, 102.72462756],
                [-380.86283223, 281.23227495],
                [-284.75355519, 177.11142683],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestRGB_colourspaceVolumeMonteCarlo:
    """
    Define :func:`colour.volume.rgb.RGB_colourspace_volume_MonteCarlo`
    definition unit tests methods.

    References
    ----------
    :cite:`Laurent2012a`
    """

    def test_RGB_colourspace_volume_MonteCarlo(self) -> None:
        """
        Test :func:`colour.volume.rgb.RGB_colourspace_volume_MonteCarlo`
        definition.
        """

        xp_assert_close(
            RGB_colourspace_volume_MonteCarlo(
                RGB_COLOURSPACE_BT709,
                int(10e3),
                random_state=np.random.RandomState(2),
            )
            * 1e-6,
            821700.0 * 1e-6,
            atol=max(TOLERANCE_ABSOLUTE_TESTS, 1),
        )


class TestRGB_colourspace_volume_coverage_MonteCarlo:
    """
    Define :func:`colour.volume.rgb.\
RGB_colourspace_volume_coverage_MonteCarlo` definition unit tests methods.

    References
    ----------
    :cite:`Laurent2012a`
    """

    def test_RGB_colourspace_volume_coverage_MonteCarlo(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.volume.rgb.\
RGB_colourspace_volume_coverage_MonteCarlo` definition.
        """

        xp_assert_close(
            RGB_colourspace_volume_coverage_MonteCarlo(
                RGB_COLOURSPACE_BT709,
                is_within_pointer_gamut,
                int(10e3),
                random_state=np.random.RandomState(2),
            ),
            81.044349070100140,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        points = xp_as_float_array(
            [[0.2, 0.2, 0.2], [0.4, 0.4, 0.4], [2.0, 2.0, 2.0]], xp=xp
        )
        for samples in (points, xp.stack([points, points])):

            def random_generator(
                _count: int, samples: typing.Any = samples, **_kwargs: typing.Any
            ) -> typing.Any:
                """Return deterministic native samples, including trailing batches."""

                return samples

            def coverage_sampler(
                XYZ: typing.Any, samples: typing.Any = samples
            ) -> typing.Any:
                """Keep every sample on the native device."""

                assert XYZ is samples
                if hasattr(samples, "device"):
                    assert XYZ.device == samples.device
                return xp_as_array([True] * XYZ.shape[0], xp=xp, like=XYZ)

            # Two of every three XYZ points lie inside BT.709. Native samples
            # must also work when the array creation default is the CPU.
            with xp.device("cpu") if xp.__name__ == "torch" else nullcontext():
                result = RGB_colourspace_volume_coverage_MonteCarlo(
                    RGB_COLOURSPACE_BT709,
                    coverage_sampler,
                    samples.shape[0],
                    random_generator=random_generator,
                )
            assert isinstance(result, float)
            xp_assert_close(result, 100 * 2 / 3, atol=TOLERANCE_ABSOLUTE_TESTS)


class TestRGB_colourspacePointerGamutCoverageMonteCarlo:
    """
    Define :func:`colour.volume.rgb.\
RGB_colourspace_pointer_gamut_coverage_MonteCarlo` definition unit tests
    methods.

    References
    ----------
    :cite:`Laurent2012a`
    """

    def test_RGB_colourspace_pointer_gamut_coverage_MonteCarlo(
        self,
        xp: ModuleType,  # noqa: ARG002
    ) -> None:
        """
        Test :func:`colour.volume.rgb.\
RGB_colourspace_pointer_gamut_coverage_MonteCarlo` definition.
        """

        xp_assert_close(
            RGB_colourspace_pointer_gamut_coverage_MonteCarlo(
                RGB_COLOURSPACE_BT709,
                int(10e3),
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

    def test_RGB_colourspace_visible_spectrum_coverage_MonteCarlo(
        self,
        xp: ModuleType,  # noqa: ARG002
    ) -> None:
        """
        Test :func:`colour.volume.rgb.\
RGB_colourspace_visible_spectrum_coverage_MonteCarlo` definition.
        """

        xp_assert_close(
            RGB_colourspace_visible_spectrum_coverage_MonteCarlo(
                RGB_COLOURSPACE_BT709,
                int(10e3),
                random_state=np.random.RandomState(2),
            ),
            46.931407942238266,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
