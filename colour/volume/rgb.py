#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RGB Colourspace Volume Computation
==================================

Defines various RGB colourspace volume computation objects:

-   :func:`RGB_colourspace_volume_MonteCarlo`
-   :func:`RGB_colourspace_limits`

See Also
--------
`RGB Colourspace Volume Computation IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/master/notebooks/volume/rgb.ipynb>`_  # noqa
"""

from __future__ import division, unicode_literals

import itertools
import multiprocessing
import numpy as np

from colour.algebra import random_triplet_generator
from colour.colorimetry import ILLUMINANTS
from colour.models import Lab_to_XYZ, RGB_to_XYZ, XYZ_to_Lab, XYZ_to_RGB

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['sample_RGB_colourspace_volume_MonteCarlo',
           'RGB_colourspace_limits',
           'RGB_colourspace_volume_MonteCarlo']


def _wrapper_RGB_colourspace_volume_MonteCarlo(args):
    """
    Convenient wrapper to be able to call
    :func:`sample_RGB_colourspace_volume_MonteCarlo`: definition with multiple
    arguments.

    Parameters
    ----------
    \*args : \*
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
        illuminant_Lab=ILLUMINANTS.get(
            'CIE 1931 2 Degree Standard Observer').get('D50'),
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
        {'CAT02', 'XYZ Scaling', 'Von Kries', 'Bradford', 'Sharp', 'Fairchild,
        'CMCCAT97', 'CMCCAT2000', 'Bianco', 'Bianco PC'},
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
    The doctest is assuming that :func:`np.random.RandomState` definition will
    return the same sequence no matter which *OS* or *Python* version is used.
    There is however no formal promise about the *prng* sequence
    reproducibility of either *Python or *Numpy* implementations: Laurent.
    (2012). Reproducibility of python pseudo-random numbers across systems and
    versions? Retrieved January 20, 2015, from
    http://stackoverflow.com/questions/8786084/reproducibility-of-python-pseudo-random-numbers-across-systems-and-versions  # noqa

    Examples
    --------
    >>> from colour import sRGB_COLOURSPACE as sRGB
    >>> prng = np.random.RandomState(2)
    >>> sample_RGB_colourspace_volume_MonteCarlo(sRGB, 10e3, random_state=prng)  # noqa  # doctest: +ELLIPSIS
    9...
    """

    random_state = (random_state
                    if random_state is not None else
                    np.random.RandomState())

    within = 0
    for Lab in random_generator(samples, limits, random_state):
        RGB = XYZ_to_RGB(Lab_to_XYZ(Lab, illuminant_Lab),
                         illuminant_Lab,
                         colourspace.whitepoint,
                         colourspace.XYZ_to_RGB_matrix,
                         chromatic_adaptation_transform=(
                             chromatic_adaptation_method))

        if np.min(RGB) >= 0 and np.max(RGB) <= 1:
            within += 1

    return within


def RGB_colourspace_limits(colourspace,
                           illuminant=ILLUMINANTS.get(
                               'CIE 1931 2 Degree Standard Observer').get(
                               'D50')):
    """
    Computes given *RGB* colourspace volume limits in *Lab* colourspace.

    Parameters
    ----------
    colourspace : RGB_Colourspace
        *RGB* colourspace to compute the volume of.
    illuminant_Lab : array_like, optional
        *Lab* colourspace *illuminant* chromaticity coordinates.

    Returns
    -------
    ndarray
        *RGB* colourspace volume limits.

    Examples
    --------
    >>> from colour import sRGB_COLOURSPACE as sRGB
    >>> RGB_colourspace_limits(sRGB)  # noqa  # doctest: +ELLIPSIS
    array([[   0...        ,  100...        ],
           [ -79.2263741...,   94.6657491...],
           [-114.7846271...,   96.7135199...]])
    """

    Lab = []
    for combination in list(itertools.product([0, 1], repeat=3)):
        Lab.append(XYZ_to_Lab(RGB_to_XYZ(combination,
                                         colourspace.whitepoint,
                                         illuminant,
                                         colourspace.RGB_to_XYZ_matrix)))
    Lab = np.array(Lab)

    limits = []
    for i in np.arange(3):
        limits.append((np.min(Lab[..., i]), np.max(Lab[..., i])))

    return np.array(limits)


def RGB_colourspace_volume_MonteCarlo(
        colourspace,
        samples=10e6,
        limits=np.array([[0, 100], [-150, 150], [-150, 150]]),
        illuminant_Lab=ILLUMINANTS.get(
            'CIE 1931 2 Degree Standard Observer').get('D50'),
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
        {'CAT02', 'XYZ Scaling', 'Von Kries', 'Bradford', 'Sharp', 'Fairchild,
        'CMCCAT97', 'CMCCAT2000', 'Bianco', 'Bianco PC'},
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
    The doctest is assuming that :func:`np.random.RandomState` definition will
    return the same sequence no matter which *OS* or *Python* version is used.
    There is however no formal promise about the *prng* sequence
    reproducibility of either *Python or *Numpy* implementations: Laurent.
    (2012). Reproducibility of python pseudo-random numbers across systems and
    versions? Retrieved January 20, 2015, from
    http://stackoverflow.com/questions/8786084/reproducibility-of-python-pseudo-random-numbers-across-systems-and-versions  # noqa


    Examples
    --------
    >>> from colour import sRGB_COLOURSPACE as sRGB
    >>> prng = np.random.RandomState(2)
    >>> processes = 1
    >>> RGB_colourspace_volume_MonteCarlo(sRGB, 10e3, random_state=prng, processes=processes)  # noqa  # doctest: +ELLIPSIS
    859...
    """

    cpu_count = processes if processes else multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cpu_count)

    process_samples = int(np.round(samples / cpu_count))

    arguments = [colourspace,
                 process_samples,
                 limits,
                 illuminant_Lab,
                 chromatic_adaptation_method,
                 random_generator,
                 random_state]

    results = pool.map(_wrapper_RGB_colourspace_volume_MonteCarlo,
                       [arguments for _ in range(cpu_count)])

    Lab_volume = np.product([np.sum(np.abs(x)) for x in limits])

    return Lab_volume * np.sum(results) / (process_samples * cpu_count)
