# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**macadam_limits.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package colour *MacAdam* limits objects.

**Others:**

"""

from __future__ import unicode_literals

import numpy as np

from colour.models import xyY_to_XYZ
from colour.optimal import ILLUMINANTS_OPTIMAL_COLOUR_STIMULI
from colour.utilities import is_scipy_installed

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["is_within_macadam_limits"]

_XYZ_OPTIMAL_COLOUR_STIMULI_CACHE = {}
_XYZ_OPTIMAL_COLOUR_STIMULI_TRIANGULATIONS_CACHE = {}


def _get_XYZ_optimal_colour_stimuli(illuminant):
    """
    Returns given illuminant optimal colour stimuli in *CIE XYZ* colourspace
    and caches it if not existing.

    :param illuminant: Illuminant.
    :type illuminant: unicode
    :return: Illuminant optimal colour stimuli.
    :rtype: tuple
    """

    optimal_colour_stimuli = ILLUMINANTS_OPTIMAL_COLOUR_STIMULI.get(illuminant)

    if optimal_colour_stimuli is None:
        raise KeyError(
            "'{0}' not found in factory optimal colour stimuli: '{1}'.".format(
                illuminant, sorted(ILLUMINANTS_OPTIMAL_COLOUR_STIMULI.keys())))

    cached_ocs = _XYZ_OPTIMAL_COLOUR_STIMULI_CACHE.get(illuminant)
    if cached_ocs is None:
        _XYZ_OPTIMAL_COLOUR_STIMULI_CACHE[illuminant] = cached_ocs = (
            np.array([np.ravel(xyY_to_XYZ(x) / 100.)
                      for x in optimal_colour_stimuli]))
    return cached_ocs


def is_within_macadam_limits(xyY, illuminant):
    """
    Returns if given *CIE xyY* colourspace matrix is within *MacAdam* limits of
    given illuminant.

    :param xyY: *CIE xyY* colourspace matrix.
    :type xyY: array_like (3, 1)
    :param illuminant: Illuminant.
    :type illuminant: unicode
    :return: Is within *MacAdam* limits.
    :rtype: bool

    :note: Input *CIE xyY* colourspace matrix is in domain [0, 1].
    """

    if is_scipy_installed(raise_exception=True):
        from scipy.spatial import Delaunay

        optimal_colour_stimuli = _get_XYZ_optimal_colour_stimuli(illuminant)
        triangulation = _XYZ_OPTIMAL_COLOUR_STIMULI_TRIANGULATIONS_CACHE.get(
            illuminant)
        if triangulation is None:
            _XYZ_OPTIMAL_COLOUR_STIMULI_TRIANGULATIONS_CACHE[illuminant] = \
                triangulation = Delaunay(optimal_colour_stimuli)

        simplex = triangulation.find_simplex(np.ravel(xyY_to_XYZ(xyY)))
        return True if simplex != -1 else False