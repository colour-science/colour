#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RGB Colourspace Volume Computation
==================================

Defines various RGB colourspace volume computation objects:

-   :func:`RGB_colourspace_volume_MonteCarlo`

See Also
--------
`RGB Colourspace Volume Computation IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/master/notebooks/volume/rgb.ipynb>`_  # noqa
"""

from __future__ import division, unicode_literals

import multiprocessing
import numpy as np

from colour.colorimetry import ILLUMINANTS
from colour.models import Lab_to_XYZ, XYZ_to_RGB

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['sample_RGB_colourspace_volume_MonteCarlo',
           'RGB_colourspace_volume_MonteCarlo']


def _wrapper_RGB_colourspace_volume_MonteCarlo(args):
    """
    Convenient wrapper to be able to call
    :def:`sample_RGB_colourspace_volume_MonteCarlo`: definition with multiple
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
        L_limits=(0, 100),
        a_limits=(-128, 128),
        b_limits=(-128, 128),
        illuminant_Lab=ILLUMINANTS.get(
            'CIE 1931 2 Degree Standard Observer').get('D50'),
        chromatic_adaptation_method='CAT02'):
    """
    Randomly samples the *Lab* colourspace volume and returns the ratio of
    samples within the given *RGB* colourspace volume.

    Parameters
    ----------
    colourspace : RGB_Colourspace
        *RGB* colourspace to compute the volume of.
    samples : numeric, optional
        Samples count.
    L_limits : array_like, optional
        *Lab* colourspace volume :math:`L` limits.
    a_limits : array_like, optional
        *Lab* colourspace volume :math:`a` limits.
    b_limits : array_like, optional
        *Lab* colourspace volume :math:`b` limits.
    illuminant_Lab : array_like, optional
        *Lab* colourspace *illuminant* chromaticity coordinates.
    chromatic_adaptation_method : unicode, optional
        {'CAT02', 'XYZ Scaling', 'Von Kries', 'Bradford', 'Sharp', 'Fairchild,
        'CMCCAT97', 'CMCCAT2000', 'Bianco', 'Bianco PC'},
        *Chromatic adaptation* method.

    Returns
    -------
    integer
        Within *RGB* colourspace volume samples count.

    Examples
    --------
    >>> from colour import sRGB_COLOURSPACE as sRGB
    >>> sample_RGB_colourspace_volume_MonteCarlo(sRGB, 10e3)   # noqa  # doctest: +ELLIPSIS
    1...
    """

    # Ensuring unique random numbers across processes.
    np.random.seed(None)

    within = 0
    for _ in np.arange(samples):
        Lab = np.array((np.random.uniform(*L_limits),
                        np.random.uniform(*a_limits),
                        np.random.uniform(*b_limits)))

        RGB = XYZ_to_RGB(Lab_to_XYZ(Lab, illuminant_Lab),
                         illuminant_Lab,
                         colourspace.whitepoint,
                         colourspace.XYZ_to_RGB_matrix,
                         chromatic_adaptation_method=(
                             chromatic_adaptation_method))

        if np.min(RGB) >= 0 and np.max(RGB) <= 1:
            within += 1

    return within


def RGB_colourspace_volume_MonteCarlo(
        colourspace,
        samples=10e6,
        L_limits=(0, 100),
        a_limits=(-128, 128),
        b_limits=(-128, 128),
        illuminant_Lab=ILLUMINANTS.get(
            'CIE 1931 2 Degree Standard Observer').get('D50'),
        chromatic_adaptation_method='CAT02'):
    """
    Performs given *RGB* colourspace volume computation using *Monte Carlo*
    method and multiprocessing.

    Parameters
    ----------
    colourspace : RGB_Colourspace
        *RGB* colourspace to compute the volume of.
    samples : numeric, optional
        Samples count.
    L_limits : array_like, optional
        *Lab* colourspace volume :math:`L` limits.
    a_limits : array_like, optional
        *Lab* colourspace volume :math:`a` limits.
    b_limits : array_like, optional
        *Lab* colourspace volume :math:`b` limits.
    illuminant_Lab : array_like, optional
        *Lab* colourspace *illuminant* chromaticity coordinates.
    chromatic_adaptation_method : unicode, optional
        {'CAT02', 'XYZ Scaling', 'Von Kries', 'Bradford', 'Sharp', 'Fairchild,
        'CMCCAT97', 'CMCCAT2000', 'Bianco', 'Bianco PC'},
        *Chromatic adaptation* method.

    Returns
    -------
    float
        *RGB* colourspace volume.

    Examples
    --------
    >>> from colour import sRGB_COLOURSPACE as sRGB
    >>> RGB_colourspace_volume_MonteCarlo(sRGB, 10e3)   # doctest: +ELLIPSIS
    8...
    """

    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cpu_count)

    arguments = [colourspace,
                 samples / cpu_count,
                 L_limits,
                 a_limits,
                 b_limits,
                 illuminant_Lab,
                 chromatic_adaptation_method]

    results = pool.map(_wrapper_RGB_colourspace_volume_MonteCarlo,
                       [arguments for _ in range(cpu_count)])

    Lab_limits = (L_limits, a_limits, b_limits)
    Lab_volume = np.product([np.sum(np.abs(x)) for x in Lab_limits])

    return Lab_volume * np.sum(results) / samples