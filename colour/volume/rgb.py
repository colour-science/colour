"""
RGB Colourspace Volume Computation
==================================

Defines various RGB colourspace volume computation objects:

-   :func:`colour.RGB_colourspace_limits`
-   :func:`colour.RGB_colourspace_volume_MonteCarlo`
-   :func:`colour.RGB_colourspace_volume_coverage_MonteCarlo`
-   :func:`colour.RGB_colourspace_pointer_gamut_coverage_MonteCarlo`
-   :func:`colour.RGB_colourspace_visible_spectrum_coverage_MonteCarlo`
"""

from __future__ import annotations

import itertools
import numpy as np

from colour.algebra import random_triplet_generator
from colour.colorimetry import CCS_ILLUMINANTS
from colour.constants import DEFAULT_INT_DTYPE
from colour.hints import (
    ArrayLike,
    Callable,
    Floating,
    Integer,
    Literal,
    NDArray,
    Optional,
    Tuple,
    Union,
)
from colour.models import (
    Lab_to_XYZ,
    RGB_Colourspace,
    RGB_to_XYZ,
    XYZ_to_Lab,
    XYZ_to_RGB,
)
from colour.volume import is_within_pointer_gamut, is_within_visible_spectrum
from colour.utilities import as_float_array, multiprocessing_pool

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
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


def _wrapper_RGB_colourspace_volume_MonteCarlo(arguments: Tuple) -> Integer:
    """
    Call the :func:`colour.volume.rgb.sample_RGB_colourspace_volume_MonteCarlo`
    definition with multiple arguments.

    Parameters
    ----------
    arguments
        Arguments.

    Returns
    -------
    :class:`numpy.integer`
        Inside *RGB* colourspace volume samples count.
    """

    return sample_RGB_colourspace_volume_MonteCarlo(*arguments)


def sample_RGB_colourspace_volume_MonteCarlo(
    colourspace: RGB_Colourspace,
    samples: Integer = 1000000,
    limits: ArrayLike = np.array([[0, 100], [-150, 150], [-150, 150]]),
    illuminant_Lab: ArrayLike = CCS_ILLUMINANTS[
        "CIE 1931 2 Degree Standard Observer"
    ]["D65"],
    chromatic_adaptation_transform: Optional[
        Union[
            Literal[
                "Bianco 2010",
                "Bianco PC 2010",
                "Bradford",
                "CAT02 Brill 2008",
                "CAT02",
                "CAT16",
                "CMCCAT2000",
                "CMCCAT97",
                "Fairchild",
                "Sharp",
                "Von Kries",
                "XYZ Scaling",
            ],
            str,
        ]
    ] = "CAT02",
    random_generator: Callable = random_triplet_generator,
    random_state: Optional[np.random.RandomState] = None,
) -> Integer:
    """
    Randomly sample the *CIE L\\*a\\*b\\** colourspace volume and returns the
    ratio of samples within the given *RGB* colourspace volume.

    Parameters
    ----------
    colourspace
        *RGB* colourspace to compute the volume of.
    samples
        Samples count.
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
    :class:`numpy.integer`
        Within *RGB* colourspace volume samples count.

    Notes
    -----
    -   The doctest is assuming that :func:`np.random.RandomState` definition
        will return the same sequence no matter which *OS* or *Python*
        version is used. There is however no formal promise about the *prng*
        sequence reproducibility of either *Python* or *Numpy*
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

    random_state = (
        random_state if random_state is not None else np.random.RandomState()
    )

    Lab = random_generator(DEFAULT_INT_DTYPE(samples), limits, random_state)
    RGB = XYZ_to_RGB(
        Lab_to_XYZ(Lab, illuminant_Lab),
        illuminant_Lab,
        colourspace.whitepoint,
        colourspace.matrix_XYZ_to_RGB,
        chromatic_adaptation_transform=chromatic_adaptation_transform,
    )
    RGB_w = RGB[
        np.logical_and(np.min(RGB, axis=-1) >= 0, np.max(RGB, axis=-1) <= 1)
    ]
    return len(RGB_w)


def RGB_colourspace_limits(colourspace: RGB_Colourspace) -> NDArray:
    """
    Compute given *RGB* colourspace volume limits in *CIE L\\*a\\*b\\**
    colourspace.

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
    The limits are computed for the given *RGB* colourspace illuminant. This is
    important to account for, if the intent is to compare various *RGB*
    colourspaces together. In this instance, they must be chromatically adapted
    to the same illuminant before-hand.
    See :meth:`colour.RGB_Colourspace.chromatically_adapt` method for more
    information.

    Examples
    --------
    >>> from colour.models import RGB_COLOURSPACE_sRGB as sRGB
    >>> RGB_colourspace_limits(sRGB)  # doctest: +ELLIPSIS
    array([[   0.       ...,  100.       ...],
           [ -86.182855 ...,   98.2563272...],
           [-107.8503557...,   94.4894974...]])
    """

    Lab_c = []
    for combination in list(itertools.product([0, 1], repeat=3)):
        Lab_c.append(
            XYZ_to_Lab(
                RGB_to_XYZ(
                    combination,
                    colourspace.whitepoint,
                    colourspace.whitepoint,
                    colourspace.matrix_RGB_to_XYZ,
                ),
                colourspace.whitepoint,
            )
        )
    Lab = np.array(Lab_c)

    limits = []
    for i in np.arange(3):
        limits.append((np.min(Lab[..., i]), np.max(Lab[..., i])))

    return np.array(limits)


def RGB_colourspace_volume_MonteCarlo(
    colourspace: RGB_Colourspace,
    samples: Integer = 1000000,
    limits: ArrayLike = np.array([[0, 100], [-150, 150], [-150, 150]]),
    illuminant_Lab: ArrayLike = CCS_ILLUMINANTS[
        "CIE 1931 2 Degree Standard Observer"
    ]["D65"],
    chromatic_adaptation_transform: Optional[
        Union[
            Literal[
                "Bianco 2010",
                "Bianco PC 2010",
                "Bradford",
                "CAT02 Brill 2008",
                "CAT02",
                "CAT16",
                "CMCCAT2000",
                "CMCCAT97",
                "Fairchild",
                "Sharp",
                "Von Kries",
                "XYZ Scaling",
            ],
            str,
        ]
    ] = "CAT02",
    random_generator: Callable = random_triplet_generator,
    random_state: Optional[np.random.RandomState] = None,
) -> Floating:
    """
    Perform given *RGB* colourspace volume computation using *Monte Carlo*
    method and multiprocessing.

    Parameters
    ----------
    colourspace\
        *RGB* colourspace to compute the volume of.
    samples\
        Samples count.
    limits\
        *CIE L\\*a\\*b\\** colourspace volume.
    illuminant_Lab\
        *CIE L\\*a\\*b\\** colourspace *illuminant* chromaticity coordinates.
    chromatic_adaptation_transform\
        *Chromatic adaptation* method.
    random_generator\
        Random triplet generator providing the random samples within the
        *CIE L\\*a\\*b\\** colourspace volume.
    random_state\
        Mersenne Twister pseudo-random number generator to use in the random
        number generator.

    Returns
    -------
    :class:`numpy.floating`
        *RGB* colourspace volume.

    Notes
    -----
    -   The doctest is assuming that :func:`np.random.RandomState` definition
        will return the same sequence no matter which *OS* or *Python*
        version is used. There is however no formal promise about the *prng*
        sequence reproducibility of either *Python* or *Numpy*
        implementations: Laurent. (2012). Reproducibility of python
        pseudo-random numbers across systems and versions? Retrieved January
        20, 2015, from http://stackoverflow.com/questions/8786084/\
reproducibility-of-python-pseudo-random-numbers-across-systems-and-versions

    Examples
    --------
    >>> from colour.models import RGB_COLOURSPACE_sRGB as sRGB
    >>> from colour.utilities import disable_multiprocessing
    >>> prng = np.random.RandomState(2)
    >>> with disable_multiprocessing():
    ...     RGB_colourspace_volume_MonteCarlo(sRGB, 10e3, random_state=prng)
    ... # doctest: +SKIP
    ...
    8...
    """

    import multiprocessing

    processes = multiprocessing.cpu_count()
    process_samples = DEFAULT_INT_DTYPE(np.round(samples / processes))

    arguments = (
        colourspace,
        process_samples,
        limits,
        illuminant_Lab,
        chromatic_adaptation_transform,
        random_generator,
        random_state,
    )

    with multiprocessing_pool() as pool:
        results = pool.map(
            _wrapper_RGB_colourspace_volume_MonteCarlo,
            [arguments for _ in range(processes)],
        )

    Lab_volume = np.product(
        [np.sum(np.abs(x)) for x in as_float_array(limits)]
    )

    return Lab_volume * np.sum(results) / (process_samples * processes)


def RGB_colourspace_volume_coverage_MonteCarlo(
    colourspace: RGB_Colourspace,
    coverage_sampler: Callable,
    samples: Integer = 1000000,
    random_generator: Callable = random_triplet_generator,
    random_state: Optional[np.random.RandomState] = None,
) -> Floating:
    """
    Return given *RGB* colourspace percentage coverage of an arbitrary volume.

    Parameters
    ----------
    colourspace
        *RGB* colourspace to compute the volume coverage percentage.
    coverage_sampler
        Python object responsible for checking the volume coverage.
    samples
        Samples count.
    random_generator
        Random triplet generator providing the random samples.
    random_state
        Mersenne Twister pseudo-random number generator to use in the random
        number generator.

    Returns
    -------
    :class:`numpy.floating`
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

    random_state = (
        random_state if random_state is not None else np.random.RandomState()
    )

    XYZ = random_generator(
        DEFAULT_INT_DTYPE(samples), random_state=random_state
    )
    XYZ_vs = XYZ[coverage_sampler(XYZ)]

    RGB = XYZ_to_RGB(
        XYZ_vs,
        colourspace.whitepoint,
        colourspace.whitepoint,
        colourspace.matrix_XYZ_to_RGB,
    )

    RGB_c = RGB[
        np.logical_and(np.min(RGB, axis=-1) >= 0, np.max(RGB, axis=-1) <= 1)
    ]

    return 100 * RGB_c.size / XYZ_vs.size


def RGB_colourspace_pointer_gamut_coverage_MonteCarlo(
    colourspace: RGB_Colourspace,
    samples: Integer = 1000000,
    random_generator: Callable = random_triplet_generator,
    random_state: Optional[np.random.RandomState] = None,
) -> Floating:
    """
    Return given *RGB* colourspace percentage coverage of Pointer's Gamut
    volume using *Monte Carlo* method.

    Parameters
    ----------
    colourspace
        *RGB* colourspace to compute the *Pointer's Gamut* coverage percentage.
    samples
        Samples count.
    random_generator
        Random triplet generator providing the random samples.
    random_state
        Mersenne Twister pseudo-random number generator to use in the random
        number generator.

    Returns
    -------
    :class:`numpy.floating`
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
    samples: Integer = 1000000,
    random_generator: Callable = random_triplet_generator,
    random_state: Optional[np.random.RandomState] = None,
) -> Floating:
    """
    Return given *RGB* colourspace percentage coverage of visible spectrum
    volume using *Monte Carlo* method.

    Parameters
    ----------
    colourspace
        *RGB* colourspace to compute the visible spectrum coverage percentage.
    samples
        Samples count.
    random_generator
        Random triplet generator providing the random samples.
    random_state
        Mersenne Twister pseudo-random number generator to use in the random
        number generator.

    Returns
    -------
    :class:`numpy.floating`
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
