# -*- coding: utf-8 -*-
"""
RGB Colourspace Volume Computation
==================================

Defines various RGB colourspace volume computation objects:

-   :func:`colour.RGB_colourspace_limits`
-   :func:`colour.RGB_colourspace_volume_MonteCarlo`
-   :func:`colour.RGB_colourspace_volume_coverage_MonteCarlo`
-   :func:`colour.RGB_colourspace_pointer_gamut_coverage_MonteCarlo`
-   :func:`colour.RGB_colourspace_visible_spectrum_coverage_MonteCarlo`

See Also
--------
`RGB Colourspace Volume Computation Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/volume/rgb.ipynb>`_
"""

from __future__ import division, unicode_literals

import itertools
import multiprocessing
import numpy as np

from colour.algebra import random_triplet_generator
from colour.colorimetry import ILLUMINANTS
from colour.constants import DEFAULT_INT_DTYPE
from colour.models import (Lab_to_XYZ, RGB_to_XYZ, XYZ_to_Lab, XYZ_to_RGB)
from colour.volume import is_within_pointer_gamut, is_within_visible_spectrum
from colour.utilities import as_float_array

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'sample_RGB_colourspace_volume_MonteCarlo', 'RGB_colourspace_limits',
    'RGB_colourspace_volume_MonteCarlo',
    'RGB_colourspace_volume_coverage_MonteCarlo',
    'RGB_colourspace_pointer_gamut_coverage_MonteCarlo',
    'RGB_colourspace_visible_spectrum_coverage_MonteCarlo'
]


def _wrapper_RGB_colourspace_volume_MonteCarlo(args):
    """
    Convenient wrapper to be able to call
    :func:`colour.volume.rgb.sample_RGB_colourspace_volume_MonteCarlo`:
    definition with multiple arguments.

    Parameters
    ----------
    args : array_like, optional
        Arguments.

    Returns
    -------
    integer
        Inside *RGB* colourspace volume samples count.
    """

    return sample_RGB_colourspace_volume_MonteCarlo(*args)


def sample_RGB_colourspace_volume_MonteCarlo(
        colourspace,
        samples=10e6,
        limits=np.array([[0, 100], [-150, 150], [-150, 150]]),
        illuminant_Lab=ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][
            'D65'],
        chromatic_adaptation_method='CAT02',
        random_generator=random_triplet_generator,
        random_state=None):
    """
    Randomly samples the *Lab* colourspace volume and returns the ratio of
    samples within the given *RGB* colourspace volume.

    Parameters
    ----------
    colourspace : RGB_Colourspace
        *RGB* colourspace to compute the volume of.
    samples : numeric, optional
        Samples count.
    limits : array_like, optional
        *Lab* colourspace volume.
    illuminant_Lab : array_like, optional
        *Lab* colourspace *illuminant* chromaticity coordinates.
    chromatic_adaptation_method : unicode, optional
        **{'CAT02', 'XYZ Scaling', 'Von Kries', 'Bradford', 'Sharp',
        'Fairchild', 'CMCCAT97', 'CMCCAT2000', 'CAT02_BRILL_CAT', 'Bianco',
        'Bianco PC'}**,
        *Chromatic adaptation* method.
    random_generator : generator, optional
        Random triplet generator providing the random samples within the *Lab*
        colourspace volume.
    random_state : RandomState, optional
        Mersenne Twister pseudo-random number generator to use in the random
        number generator.

    Returns
    -------
    integer
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
    >>> from colour.models import sRGB_COLOURSPACE as sRGB
    >>> prng = np.random.RandomState(2)
    >>> sample_RGB_colourspace_volume_MonteCarlo(sRGB, 10e3, random_state=prng)
    ... # doctest: +ELLIPSIS
    9...
    """

    random_state = (random_state
                    if random_state is not None else np.random.RandomState())

    Lab = as_float_array(list(random_generator(samples, limits, random_state)))
    RGB = XYZ_to_RGB(
        Lab_to_XYZ(Lab, illuminant_Lab),
        illuminant_Lab,
        colourspace.whitepoint,
        colourspace.XYZ_to_RGB_matrix,
        chromatic_adaptation_transform=chromatic_adaptation_method)
    RGB_w = RGB[np.logical_and(
        np.min(RGB, axis=-1) >= 0,
        np.max(RGB, axis=-1) <= 1)]
    return len(RGB_w)


def RGB_colourspace_limits(
        colourspace,
        illuminant=ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']):
    """
    Computes given *RGB* colourspace volume limits in *Lab* colourspace.

    Parameters
    ----------
    colourspace : RGB_Colourspace
        *RGB* colourspace to compute the volume of.
    illuminant : array_like, optional
        *Lab* colourspace *illuminant* chromaticity coordinates.

    Returns
    -------
    ndarray
        *RGB* colourspace volume limits.

    Examples
    --------
    >>> from colour import sRGB_COLOURSPACE as sRGB
    >>> RGB_colourspace_limits(sRGB)  # doctest: +ELLIPSIS
    array([[   0.       ...,  100.       ...],
           [ -86.182855 ...,   98.2563272...],
           [-107.8503557...,   94.4894974...]])
    """

    Lab = []
    for combination in list(itertools.product([0, 1], repeat=3)):
        Lab.append(
            XYZ_to_Lab(
                RGB_to_XYZ(combination, colourspace.whitepoint, illuminant,
                           colourspace.RGB_to_XYZ_matrix)))
    Lab = np.array(Lab)

    limits = []
    for i in np.arange(3):
        limits.append((np.min(Lab[..., i]), np.max(Lab[..., i])))

    return np.array(limits)


def RGB_colourspace_volume_MonteCarlo(
        colourspace,
        samples=10e6,
        limits=np.array([[0, 100], [-150, 150], [-150, 150]], dtype=np.float),
        illuminant_Lab=ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][
            'D65'],
        chromatic_adaptation_method='CAT02',
        random_generator=random_triplet_generator,
        random_state=None,
        processes=None):
    """
    Performs given *RGB* colourspace volume computation using *Monte Carlo*
    method and multiprocessing.

    Parameters
    ----------
    colourspace : RGB_Colourspace
        *RGB* colourspace to compute the volume of.
    samples : numeric, optional
        Samples count.
    limits : array_like, optional
        *Lab* colourspace volume.
    illuminant_Lab : array_like, optional
        *Lab* colourspace *illuminant* chromaticity coordinates.
    chromatic_adaptation_method : unicode, optional
        **{'CAT02', 'XYZ Scaling', 'Von Kries', 'Bradford', 'Sharp',
        'Fairchild', 'CMCCAT97', 'CMCCAT2000', 'CAT02_BRILL_CAT', 'Bianco',
        'Bianco PC'}**,
        *Chromatic adaptation* method.
    random_generator : generator, optional
        Random triplet generator providing the random samples within the *Lab*
        colourspace volume.
    random_state : RandomState, optional
        Mersenne Twister pseudo-random number generator to use in the random
        number generator.
    processes : integer, optional
        Processes count, default to :func:`multiprocessing.cpu_count`
        definition.

    Returns
    -------
    float
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
    >>> from colour import sRGB_COLOURSPACE as sRGB
    >>> prng = np.random.RandomState(2)
    >>> processes = 1
    >>> RGB_colourspace_volume_MonteCarlo(sRGB, 10e3, random_state=prng,
    ...                                   processes=processes)
    ... # doctest: +ELLIPSIS
    816...
    """

    cpu_count = processes if processes else multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cpu_count)

    process_samples = DEFAULT_INT_DTYPE(np.round(samples / cpu_count))

    arguments = (colourspace, process_samples, limits, illuminant_Lab,
                 chromatic_adaptation_method, random_generator, random_state)

    results = pool.map(_wrapper_RGB_colourspace_volume_MonteCarlo,
                       [arguments for _ in range(cpu_count)])

    Lab_volume = np.product([np.sum(np.abs(x)) for x in limits])

    return Lab_volume * np.sum(results) / (process_samples * cpu_count)


def RGB_colourspace_volume_coverage_MonteCarlo(
        colourspace,
        coverage_sampler,
        samples=10e6,
        random_generator=random_triplet_generator,
        random_state=None):
    """
    Returns given *RGB* colourspace percentage coverage of an arbitrary volume.

    Parameters
    ----------
    colourspace : RGB_Colourspace
        *RGB* colourspace to compute the volume coverage percentage.
    coverage_sampler : object
        Python object responsible for checking the volume coverage.
    samples : numeric, optional
        Samples count.
    random_generator : generator, optional
        Random triplet generator providing the random samples.
    random_state : RandomState, optional
        Mersenne Twister pseudo-random number generator to use in the random
        number generator.

    Returns
    -------
    float
        Percentage coverage of volume.

    Examples
    --------
    >>> from colour import sRGB_COLOURSPACE as sRGB
    >>> prng = np.random.RandomState(2)
    >>> RGB_colourspace_volume_coverage_MonteCarlo(
    ...     sRGB, is_within_pointer_gamut, 10e3, random_state=prng)
    ... # doctest: +ELLIPSIS
    81...
    """

    random_state = (random_state
                    if random_state is not None else np.random.RandomState())

    # TODO: Investigate for generator yielding directly a ndarray.
    XYZ = as_float_array(
        list(random_generator(samples, random_state=random_state)))
    XYZ_vs = XYZ[coverage_sampler(XYZ)]

    RGB = XYZ_to_RGB(XYZ_vs, colourspace.whitepoint, colourspace.whitepoint,
                     colourspace.XYZ_to_RGB_matrix)

    RGB_c = RGB[np.logical_and(
        np.min(RGB, axis=-1) >= 0,
        np.max(RGB, axis=-1) <= 1)]

    return 100 * RGB_c.size / XYZ_vs.size


def RGB_colourspace_pointer_gamut_coverage_MonteCarlo(
        colourspace,
        samples=10e6,
        random_generator=random_triplet_generator,
        random_state=None):
    """
    Returns given *RGB* colourspace percentage coverage of Pointer's Gamut
    volume using *Monte Carlo* method.

    Parameters
    ----------
    colourspace : RGB_Colourspace
        *RGB* colourspace to compute the *Pointer's Gamut* coverage percentage.
    samples : numeric, optional
        Samples count.
    random_generator : generator, optional
        Random triplet generator providing the random samples.
    random_state : RandomState, optional
        Mersenne Twister pseudo-random number generator to use in the random
        number generator.

    Returns
    -------
    float
        Percentage coverage of *Pointer's Gamut* volume.

    Examples
    --------
    >>> from colour import sRGB_COLOURSPACE as sRGB
    >>> prng = np.random.RandomState(2)
    >>> RGB_colourspace_pointer_gamut_coverage_MonteCarlo(
    ...     sRGB, 10e3, random_state=prng)  # doctest: +ELLIPSIS
    81...
    """

    return RGB_colourspace_volume_coverage_MonteCarlo(
        colourspace, is_within_pointer_gamut, samples, random_generator,
        random_state)


def RGB_colourspace_visible_spectrum_coverage_MonteCarlo(
        colourspace,
        samples=10e6,
        random_generator=random_triplet_generator,
        random_state=None):
    """
    Returns given *RGB* colourspace percentage coverage of visible spectrum
    volume using *Monte Carlo* method.

    Parameters
    ----------
    colourspace : RGB_Colourspace
        *RGB* colourspace to compute the visible spectrum coverage percentage.
    samples : numeric, optional
        Samples count.
    random_generator : generator, optional
        Random triplet generator providing the random samples.
    random_state : RandomState, optional
        Mersenne Twister pseudo-random number generator to use in the random
        number generator.

    Returns
    -------
    float
        Percentage coverage of visible spectrum volume.

    Examples
    --------
    >>> from colour import sRGB_COLOURSPACE as sRGB
    >>> prng = np.random.RandomState(2)
    >>> RGB_colourspace_visible_spectrum_coverage_MonteCarlo(
    ...     sRGB, 10e3, random_state=prng)  # doctest: +ELLIPSIS
    47...
    """

    return RGB_colourspace_volume_coverage_MonteCarlo(
        colourspace, is_within_visible_spectrum, samples, random_generator,
        random_state)
