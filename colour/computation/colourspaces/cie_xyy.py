# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**cie_xyy.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package colour *CIE xyY* colourspace objects.

**Others:**

"""

from __future__ import unicode_literals

import numpy

import colour.dataset.illuminants.chromaticity_coordinates
import colour.dataset.illuminants.optimal_colour_stimuli
import colour.utilities.common
import colour.utilities.exceptions
from colour.cache.runtime import RuntimeCache

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["XYZ_to_xyY",
           "xyY_to_XYZ",
           "xy_to_XYZ",
           "XYZ_to_xy",
           "is_within_macadam_limits"]


def __get_XYZ_optimal_colour_stimuli(illuminant):
    """
    Returns given illuminant optimal colour stimuli in *CIE XYZ* colourspace and caches it if not existing.

    :param illuminant: Illuminant.
    :type illuminant: unicode
    :return: Illuminant optimal colour stimuli.
    :rtype: tuple
    """

    optimal_colour_stimuli = \
        colour.dataset.illuminants.optimal_colour_stimuli.ILLUMINANTS_OPTIMAL_COLOUR_STIMULI.get(illuminant)

    if optimal_colour_stimuli is None:
        raise colour.utilities.exceptions.ProgrammingError(
            "'{0}' not found in factory optimal colour stimuli: '{1}'.".format(illuminant,
                                                                               sorted(
                                                                                   colour.dataset.illuminants.optimal_colour_stimuli.ILLUMINANTS_OPTIMAL_COLOUR_STIMULI.keys())))

    cached_optimal_colour_stimuli = RuntimeCache.XYZ_optimal_colour_stimuli.get(illuminant)
    if cached_optimal_colour_stimuli is None:
        RuntimeCache.XYZ_optimal_colour_stimuli[illuminant] = cached_optimal_colour_stimuli = \
            numpy.array(map(lambda x: numpy.ravel(xyY_to_XYZ(x) / 100.), optimal_colour_stimuli))
    return cached_optimal_colour_stimuli


def XYZ_to_xyY(XYZ,
               illuminant=colour.dataset.illuminants.chromaticity_coordinates.ILLUMINANTS.get(
                   "CIE 1931 2 Degree Standard Observer").get("D50")):
    """
    Converts from *CIE XYZ* colourspace to *CIE xyY* colourspace and reference *illuminant*.

    References:

    -  http://www.brucelindbloom.com/Eqn_XYZ_to_xyY.html (Last accessed 24 February 2014)

    Usage::

        >>> XYZ_to_xyY(numpy.array([0.1180583421, 0.1034, 0.0515089229]))
        array([[ 0.4325]
               [ 0.3788]
               [ 0.1034]])

    :param XYZ: *CIE XYZ* colourspace matrix.
    :type XYZ: array_like (3, 1)
    :param illuminant: Reference *illuminant* chromaticity coordinates.
    :type illuminant: array_like
    :return: *CIE xyY* colourspace matrix.
    :rtype: ndarray (3, 1)

    :note: *CIE XYZ* is in domain [0, 1].
    :note: *CIE xyY* is in domain [0, 1].
    """

    X, Y, Z = numpy.ravel(XYZ)

    if X == 0 and Y == 0 and Z == 0:
        return numpy.array([illuminant[0], illuminant[1], Y]).reshape((3, 1))
    else:
        return numpy.array([X / (X + Y + Z), Y / (X + Y + Z), Y]).reshape((3, 1))

def xyY_to_XYZ(xyY):
    """
    Converts from *CIE xyY* colourspace to *CIE XYZ* colourspace.

    References:

    -  http://www.brucelindbloom.com/Eqn_xyY_to_XYZ.html (Last accessed 24 February 2014)

    Usage::

        >>> xyY_to_XYZ(numpy.array([0.4325, 0.3788, 0.1034]))
        array([[ 0.11805834]
               [ 0.1034    ]
               [ 0.05150892]])

    :param xyY: *CIE xyY* colourspace matrix.
    :type xyY: array_like (3, 1)
    :return: *CIE XYZ* colourspace matrix.
    :rtype: ndarray (3, 1)

    :note: *CIE xyY* is in domain [0, 1].
    :note: *CIE XYZ* is in domain [0, 1].
    """

    x, y, Y = numpy.ravel(xyY)

    if y == 0:
        return numpy.array([0., 0., 0.]).reshape((3, 1))
    else:
        return numpy.array([x * Y / y, Y, (1. - x - y) * Y / y]).reshape((3, 1))

def xy_to_XYZ(xy):
    """
    Returns the *CIE XYZ* colourspace matrix from given *xy* chromaticity coordinates.

    Usage::

        >>> xy_to_XYZ((0.25, 0.25))
        array([[ 1.],
               [ 1.],
               [ 2.]])

    :param xy: *xy* chromaticity coordinate.
    :type xy: array_like
    :return: *CIE XYZ* colourspace matrix.
    :rtype: ndarray (3, 1)

    :note: *xy* is in domain [0, 1].
    :note: *CIE XYZ* is in domain [0, 1].
    """

    return xyY_to_XYZ(numpy.array([xy[0], xy[1], 1.]).reshape((3, 1)))


def XYZ_to_xy(XYZ,
              illuminant=colour.dataset.illuminants.chromaticity_coordinates.ILLUMINANTS.get(
                  "CIE 1931 2 Degree Standard Observer").get("D50")):
    """
    Returns the *xy* chromaticity coordinates from given *CIE XYZ* colourspace matrix.

    Usage::

        >>> XYZ_to_xy(numpy.array([0.97137399, 1., 1.04462134]))
        (0.32207410281368043, 0.33156550013623531)
        >>> XYZ_to_xy((0.97137399, 1., 1.04462134))
        (0.32207410281368043, 0.33156550013623531)

    :param XYZ: *CIE XYZ* colourspace matrix.
    :type XYZ: array_like (3, 1)
    :param illuminant: Reference *illuminant* chromaticity coordinates.
    :type illuminant: array_like
    :return: *xy* chromaticity coordinates.
    :rtype: tuple

    :note: *CIE XYZ* is in domain [0, 1].
    :note: *xy* is in domain [0, 1].
    """

    xyY = numpy.ravel(XYZ_to_xyY(XYZ, illuminant))
    return xyY[0], xyY[1]


def is_within_macadam_limits(xyY, illuminant):
    """
    Returns if given *CIE xyY* colourspace matrix is within *MacAdam* limits of given illuminant.

    :param xyY: *CIE xyY* colourspace matrix.
    :type xyY: array_like (3, 1)
    :param illuminant: Illuminant.
    :type illuminant: unicode
    :return: Is within *MacAdam* limits.
    :rtype: bool

    :note: *CIE xyY* is in domain [0, 1].
    """

    if colour.utilities.common.is_scipy_installed(raise_exception=True):
        from scipy.spatial import Delaunay

        optimal_colour_stimuli = __get_XYZ_optimal_colour_stimuli(illuminant)
        triangulation = RuntimeCache.XYZ_optimal_colour_stimuli_triangulations.get(illuminant)
        if triangulation is None:
            RuntimeCache.XYZ_optimal_colour_stimuli_triangulations[illuminant] = triangulation = \
                Delaunay(optimal_colour_stimuli)

        return True if triangulation.find_simplex(numpy.ravel(xyY_to_XYZ(xyY))) != -1 else False