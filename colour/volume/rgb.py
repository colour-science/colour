"""
RGB Colourspace Volume Computation
==================================

Define RGB colourspace volume computation functionality.

-   :func:`colour.RGB_colourspace_limits`
-   :func:`colour.RGB_colourspace_volume_MonteCarlo`
-   :func:`colour.RGB_colourspace_volume_coverage_MonteCarlo`
-   :func:`colour.RGB_colourspace_pointer_gamut_coverage_MonteCarlo`
-   :func:`colour.RGB_colourspace_visible_spectrum_coverage_MonteCarlo`
"""

from __future__ import annotations

import itertools
import typing

import numpy as np

from colour.algebra import random_triplet_generator
from colour.colorimetry import CCS_ILLUMINANTS
from colour.constants import DTYPE_INT_DEFAULT

if typing.TYPE_CHECKING:
    from colour.hints import (
        ArrayLike,
        Callable,
        LiteralChromaticAdaptationTransform,
        NDArrayFloat,
    )

from colour.models import (
    Lab_to_XYZ,
    RGB_Colourspace,
    RGB_to_XYZ,
    XYZ_to_Lab,
    XYZ_to_RGB,
)
from colour.utilities import (
    array_namespace,
    as_float_array,
)
from colour.volume import is_within_pointer_gamut, is_within_visible_spectrum

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "sample_RGB_colourspace_volume_MonteCarlo",
    "RGB_colourspace_limits",
    "RGB_colourspace_volume_MonteCarlo",
    "RGB_colourspace_volume_coverage_MonteCarlo",
    "RGB_colourspace_pointer_gamut_coverage_MonteCarlo",
    "RGB_colourspace_visible_spectrum_coverage_MonteCarlo",
]


def sample_RGB_colourspace_volume_MonteCarlo(
    colourspace: RGB_Colourspace,
    samples: int = 1000000,
    limits: ArrayLike = ([0, 100], [-150, 150], [-150, 150]),
    illuminant_Lab: ArrayLike = CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"][
        "D65"
    ],
    chromatic_adaptation_transform: (
        LiteralChromaticAdaptationTransform | str | None
    ) = "CAT02",
    random_generator: Callable = random_triplet_generator,
    random_state: np.random.RandomState | None = None,
) -> int:
    """
    Randomly sample the *CIE L\\*a\\*b\\** colourspace volume and return the
    ratio of samples within the specified *RGB* colourspace volume.

    Parameters
    ----------
    colourspace
        *RGB* colourspace to compute the volume of.
    samples
        Sample count.
    limits
        *CIE L\\*a\\*b\\** colourspace volume.
    illuminant_Lab
        *CIE L\\*a\\*b\\** colourspace *illuminant* chromaticity coordinates.
    chromatic_adaptation_transform
        *Chromatic adaptation* transform.
    random_generator
        Random triplet generator providing the random samples within the
        *CIE L\\*a\\*b\\** colourspace volume.
    random_state
        Mersenne Twister pseudo-random number generator to use in the random
        number generator.

    Returns
    -------
    :class:`int`
        Within *RGB* colourspace volume sample count.

    Notes
    -----
    -   The doctest is assuming that :func:`np.random.RandomState`
        definition will return the same sequence no matter which *OS* or
        *Python* version is used. There is however no formal promise about
        the *prng* sequence reproducibility of either *Python* or *Numpy*
        implementations: Laurent. (2012). Reproducibility of python
        pseudo-random numbers across systems and versions? Retrieved January
        20, 2015, from http://stackoverflow.com/questions/8786084/\
reproducibility-of-python-pseudo-random-numbers-across-systems-and-versions

    Examples
    --------
    >>> from colour.models import RGB_COLOURSPACE_sRGB as sRGB
    >>> prng = np.random.RandomState(2)
    >>> sample_RGB_colourspace_volume_MonteCarlo(sRGB, 10e3, random_state=prng)
    ... # doctest: +ELLIPSIS
    9...
    """

    random_state = random_state if random_state is not None else np.random.RandomState()

    Lab = random_generator(DTYPE_INT_DEFAULT(samples), limits, random_state)
    RGB = XYZ_to_RGB(
        Lab_to_XYZ(Lab, illuminant_Lab),
        colourspace,
        illuminant_Lab,
        chromatic_adaptation_transform,
    )

    xp = array_namespace(RGB)

    RGB_w = RGB[xp.logical_and(xp.min(RGB, axis=-1) >= 0, xp.max(RGB, axis=-1) <= 1)]
    return len(RGB_w)


def RGB_colourspace_limits(colourspace: RGB_Colourspace) -> NDArrayFloat:
    """
    Compute the specified *RGB* colourspace volume limits in
    *CIE L\\*a\\*b\\** colourspace.

    Parameters
    ----------
    colourspace
        *RGB* colourspace to compute the volume of.

    Returns
    -------
    :class:`numpy.ndarray`
        *RGB* colourspace volume limits.

    Notes
    -----
    The limits are computed for the specified *RGB* colourspace illuminant.
    This is important to account for if the intent is to compare various
    *RGB* colourspaces together. In this instance, they must be
    chromatically adapted to the same illuminant beforehand. See
    :meth:`colour.RGB_Colourspace.chromatically_adapt` method for more
    information.

    Examples
    --------
    >>> from colour.models import RGB_COLOURSPACE_sRGB as sRGB
    >>> RGB_colourspace_limits(sRGB)  # doctest: +ELLIPSIS
    array([[   0.       ...,  100.       ...],
           [ -86.182855 ...,   98.2563272...],
           [-107.8503557...,   94.4894974...]])
    """

    corners = as_float_array(list(itertools.product([0, 1], repeat=3)))
    Lab = XYZ_to_Lab(
        RGB_to_XYZ(corners, colourspace),
        colourspace.whitepoint,
    )

    xp = array_namespace(Lab)

    return xp.stack([xp.min(Lab, axis=0), xp.max(Lab, axis=0)], axis=-1)


def RGB_colourspace_volume_MonteCarlo(
    colourspace: RGB_Colourspace,
    samples: int = 1000000,
    limits: ArrayLike = ([0, 100], [-150, 150], [-150, 150]),
    illuminant_Lab: ArrayLike = CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"][
        "D65"
    ],
    chromatic_adaptation_transform: (
        LiteralChromaticAdaptationTransform | str | None
    ) = "CAT02",
    random_generator: Callable = random_triplet_generator,
    random_state: np.random.RandomState | None = None,
) -> float:
    """
    Compute the specified *RGB* colourspace volume using the *Monte Carlo*
    method with multiprocessing.

    Parameters
    ----------
    colourspace
        *RGB* colourspace to compute the volume of.
    samples
        Sample count.
    limits
        *CIE L\\*a\\*b\\** colourspace volume boundaries.
    illuminant_Lab
        *CIE L\\*a\\*b\\** colourspace *illuminant* chromaticity
        coordinates.
    chromatic_adaptation_transform
        *Chromatic adaptation* method.
    random_generator
        Random triplet generator providing the random samples within the
        *CIE L\\*a\\*b\\** colourspace volume.
    random_state
        Mersenne Twister pseudo-random number generator to use in the
        random number generator.

    Returns
    -------
    :class:`float`
        *RGB* colourspace volume.

    Notes
    -----
    -   The doctest is assuming that :func:`np.random.RandomState`
        definition will return the same sequence no matter which *OS* or
        *Python* version is used. There is however no formal promise about
        the *prng* sequence reproducibility of either *Python* or *Numpy*
        implementations: Laurent. (2012). Reproducibility of python
        pseudo-random numbers across systems and versions? Retrieved January
        20, 2015, from http://stackoverflow.com/questions/8786084/\
reproducibility-of-python-pseudo-random-numbers-across-systems-and-versions

    Examples
    --------
    >>> from colour.models import RGB_COLOURSPACE_sRGB as sRGB
    >>> prng = np.random.RandomState(2)
    >>> RGB_colourspace_volume_MonteCarlo(sRGB, 10e3, random_state=prng)
    ... # doctest: +SKIP
    ...
    8...
    """

    samples = int(DTYPE_INT_DEFAULT(samples))
    within = sample_RGB_colourspace_volume_MonteCarlo(
        colourspace,
        samples,
        limits,
        illuminant_Lab,
        chromatic_adaptation_transform,
        random_generator,
        random_state,
    )

    limits = as_float_array(limits)

    xp = array_namespace(limits)

    Lab_volume = xp.prod(xp.sum(xp.abs(limits), axis=-1))

    return float(Lab_volume * within / samples)


def RGB_colourspace_volume_coverage_MonteCarlo(
    colourspace: RGB_Colourspace,
    coverage_sampler: Callable,
    samples: int = 1000000,
    random_generator: Callable = random_triplet_generator,
    random_state: np.random.RandomState | None = None,
) -> float:
    """
    Compute the specified *RGB* colourspace percentage coverage of an
    arbitrary volume.

    Parameters
    ----------
    colourspace
        *RGB* colourspace to compute the volume coverage percentage.
    coverage_sampler
        Python object responsible for checking the volume coverage.
    samples
        Sample count.
    random_generator
        Random triplet generator providing the random samples.
    random_state
        Mersenne Twister pseudo-random number generator to use in the
        random number generator.

    Returns
    -------
    :class:`float`
        Percentage coverage of volume.

    Examples
    --------
    >>> from colour.models import RGB_COLOURSPACE_sRGB as sRGB
    >>> prng = np.random.RandomState(2)
    >>> RGB_colourspace_volume_coverage_MonteCarlo(
    ...     sRGB, is_within_pointer_gamut, 10e3, random_state=prng
    ... )
    ... # doctest: +ELLIPSIS
    81...
    """

    random_state = random_state if random_state is not None else np.random.RandomState()

    XYZ = random_generator(DTYPE_INT_DEFAULT(samples), random_state=random_state)
    XYZ_vs = XYZ[coverage_sampler(XYZ)]

    RGB = XYZ_to_RGB(XYZ_vs, colourspace)

    xp = array_namespace(RGB)

    RGB_c = RGB[xp.logical_and(xp.min(RGB, axis=-1) >= 0, xp.max(RGB, axis=-1) <= 1)]

    return 100 * RGB_c.size / XYZ_vs.size


def RGB_colourspace_pointer_gamut_coverage_MonteCarlo(
    colourspace: RGB_Colourspace,
    samples: int = 1000000,
    random_generator: Callable = random_triplet_generator,
    random_state: np.random.RandomState | None = None,
) -> float:
    """
    Compute the specified *RGB* colourspace percentage coverage of
    *Pointer's Gamut* volume using the *Monte Carlo* method.

    Parameters
    ----------
    colourspace
        *RGB* colourspace to compute the *Pointer's Gamut* coverage
        percentage.
    samples
        Sample count.
    random_generator
        Random triplet generator providing the random samples.
    random_state
        Mersenne Twister pseudo-random number generator to use in the
        random number generator.

    Returns
    -------
    :class:`float`
        Percentage coverage of *Pointer's Gamut* volume.

    Examples
    --------
    >>> from colour.models import RGB_COLOURSPACE_sRGB as sRGB
    >>> prng = np.random.RandomState(2)
    >>> RGB_colourspace_pointer_gamut_coverage_MonteCarlo(
    ...     sRGB, 10e3, random_state=prng
    ... )  # doctest: +ELLIPSIS
    81...
    """

    return RGB_colourspace_volume_coverage_MonteCarlo(
        colourspace,
        is_within_pointer_gamut,
        samples,
        random_generator,
        random_state,
    )


def RGB_colourspace_visible_spectrum_coverage_MonteCarlo(
    colourspace: RGB_Colourspace,
    samples: int = 1000000,
    random_generator: Callable = random_triplet_generator,
    random_state: np.random.RandomState | None = None,
) -> float:
    """
    Compute the specified *RGB* colourspace percentage coverage of the visible
    spectrum volume using the *Monte Carlo* method.

    Parameters
    ----------
    colourspace
        *RGB* colourspace to compute the visible spectrum coverage percentage.
    samples
        Sample count.
    random_generator
        Random triplet generator providing the random samples.
    random_state
        Mersenne Twister pseudo-random number generator to use in the random
        number generator.

    Returns
    -------
    :class:`float`
        Percentage coverage of visible spectrum volume.

    Examples
    --------
    >>> from colour.models import RGB_COLOURSPACE_sRGB as sRGB
    >>> prng = np.random.RandomState(2)
    >>> RGB_colourspace_visible_spectrum_coverage_MonteCarlo(
    ...     sRGB, 10e3, random_state=prng
    ... )  # doctest: +ELLIPSIS
    46...
    """

    return RGB_colourspace_volume_coverage_MonteCarlo(
        colourspace,
        is_within_visible_spectrum,
        samples,
        random_generator,
        random_state,
    )
