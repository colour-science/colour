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

import colour.computation.chromatic_adaptation
import colour.dataset.illuminants.chromaticity_coordinates
import colour.dataset.illuminants.optimal_colour_stimuli
import colour.utilities.common
import colour.utilities.verbose

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
           "XYZ_to_RGB",
           "RGB_to_XYZ",
           "xyY_to_RGB",
           "RGB_to_xyY"]

LOGGER = colour.utilities.verbose.install_logger()


def XYZ_to_xyY(XYZ,
               illuminant=colour.dataset.illuminants.chromaticity_coordinates.ILLUMINANTS.get(
                   "CIE 1931 2 Degree Standard Observer").get("D50")):
    """
    Converts from *CIE XYZ* colourspace to *CIE xyY* colourspace and reference *illuminant*.

    References:

    -  http://www.brucelindbloom.com/Eqn_XYZ_to_xyY.html

    Usage::

        >>> XYZ_to_xyY(numpy.matrix([11.80583421, 10.34, 5.15089229]).reshape((3, 1)))
        matrix([[  0.4325],
                [  0.3788],
                [ 10.34  ]])

    :param XYZ: *CIE XYZ* matrix.
    :type XYZ: matrix (3x1)
    :param illuminant: Reference *illuminant* chromaticity coordinates.
    :type illuminant: tuple
    :return: *CIE xyY* matrix.
    :rtype: matrix (3x1)
    """

    X, Y, Z = numpy.ravel(XYZ)

    if X == 0 and Y == 0 and Z == 0:
        return numpy.matrix([illuminant[0], illuminant[1], Y]).reshape((3, 1))
    else:
        return numpy.matrix([X / (X + Y + Z), Y / (X + Y + Z), Y]).reshape((3, 1))


def xyY_to_XYZ(xyY):
    """
    Converts from *CIE xyY* colourspace to *CIE XYZ* colourspace.

    References:

    -  http://www.brucelindbloom.com/Eqn_xyY_to_XYZ.html

    Usage::

        >>> xyY_to_XYZ(numpy.matrix([0.4325, 0.3788, 10.34]).reshape((3, 1)))
        matrix([[ 11.80583421],
                [ 10.34      ],
                [  5.15089229]])

    :param xyY: *CIE xyY* matrix.
    :type xyY: matrix (3x1)
    :return: *CIE XYZ* matrix.
    :rtype: matrix (3x1)
    """

    x, y, Y = numpy.ravel(xyY)

    if y == 0:
        return numpy.matrix([0., 0., 0.]).reshape((3, 1))
    else:
        return numpy.matrix([x * Y / y, Y, (1. - x - y) * Y / y]).reshape((3, 1))


def xy_to_XYZ(xy):
    """
    Returns the *CIE XYZ* matrix from given *xy* chromaticity coordinates.

    Usage::

        >>> xy_to_XYZ((0.25, 0.25))
        matrix([[ 1.],
                [ 1.],
                [ 2.]])

    :param xy: *xy* chromaticity coordinate.
    :type xy: tuple
    :return: *CIE XYZ* matrix.
    :rtype: matrix (3x1)
    """

    return xyY_to_XYZ(numpy.matrix([xy[0], xy[1], 1.]).reshape((3, 1)))


def XYZ_to_xy(XYZ,
              illuminant=colour.dataset.illuminants.chromaticity_coordinates.ILLUMINANTS.get(
                  "CIE 1931 2 Degree Standard Observer").get("D50")):
    """
    Returns the *xy* chromaticity coordinates from given *CIE XYZ* matrix.

    Usage::

        >>> XYZ_to_xy(numpy.matrix([0.97137399, 1., 1.04462134]).reshape((3, 1)))
        (0.32207410281368043, 0.33156550013623531)
        >>> XYZ_to_xy((0.97137399, 1., 1.04462134))
        (0.32207410281368043, 0.33156550013623531)

    :param XYZ: *CIE XYZ* matrix.
    :type XYZ: matrix (3x1)
    :param illuminant: Reference *illuminant* chromaticity coordinates.
    :type illuminant: tuple
    :return: *xy* chromaticity coordinates.
    :rtype: tuple
    """

    xyY = numpy.ravel(XYZ_to_xyY(XYZ, illuminant))
    return xyY[0], xyY[1]


def XYZ_to_RGB(XYZ,
               illuminant_XYZ,
               illuminant_RGB,
               chromatic_adaptation_method,
               from_XYZ,
               transfer_function=None):
    """
    Converts from *CIE XYZ* colourspace to *RGB* colourspace using given *CIE XYZ* matrix, *illuminants*,
    *chromatic adaptation* method, *normalised primary matrix* and *transfer function*.

    Usage::

        >>> XYZ = numpy.matrix([11.51847498, 10.08, 5.08937252]).reshape((3, 1))
        >>> illuminant_XYZ =  (0.34567, 0.35850)
        >>> illuminant_RGB =  (0.31271, 0.32902)
        >>> chromatic_adaptation_method =  "Bradford"
        >>> from_XYZ =  numpy.matrix([3.24100326, -1.53739899, -0.49861587, -0.96922426,  1.87592999,  0.04155422, 0.05563942, -0.2040112 ,  1.05714897]).reshape((3, 3))
        >>> XYZ_to_RGB(XYZ, illuminant_XYZ, illuminant_RGB, chromatic_adaptation_method, from_XYZ)
        matrix([[ 17.303501],
                [ 8.8211033],
                [ 5.5672498]])

    :param XYZ: *CIE XYZ* colourspace matrix.
    :type XYZ: matrix (3x1)
    :param illuminant_XYZ: *CIE XYZ* colourspace *illuminant* chromaticity coordinates.
    :type illuminant_XYZ: tuple
    :param illuminant_RGB: *RGB* colourspace *illuminant* chromaticity coordinates.
    :type illuminant_RGB: tuple
    :param chromatic_adaptation_method: *Chromatic adaptation* method.
    :type chromatic_adaptation_method: unicode
    :param from_XYZ: *Normalised primary matrix*.
    :type from_XYZ: matrix (3x3)
    :param transfer_function: *Transfer function*.
    :type transfer_function: object
    :return: *RGB* colourspace matrix.
    :rtype: matrix (3x1)
    """

    cat = colour.computation.chromatic_adaptation.get_chromatic_adaptation_matrix(
        xy_to_XYZ(illuminant_XYZ),
        xy_to_XYZ(illuminant_RGB),
        method=chromatic_adaptation_method)

    adaptedXYZ = cat * XYZ

    RGB = from_XYZ * adaptedXYZ

    if transfer_function is not None:
        RGB = numpy.matrix(map(lambda x: transfer_function(x), numpy.ravel(RGB))).reshape((3, 1))

    LOGGER.debug("> 'Chromatic adaptation' matrix:\n{0}".format(repr(cat)))
    LOGGER.debug("> Adapted 'CIE XYZ' matrix:\n{0}".format(repr(adaptedXYZ)))
    LOGGER.debug("> 'RGB' matrix:\n{0}".format(repr(RGB)))

    return RGB


def RGB_to_XYZ(RGB,
               illuminant_RGB,
               illuminant_XYZ,
               chromatic_adaptation_method,
               to_XYZ,
               inverse_transfer_function=None):
    """
    Converts from *RGB* colourspace to *CIE XYZ* colourspace using given *RGB* matrix, *illuminants*,
    *chromatic adaptation* method, *normalised primary matrix* and *transfer function*.

    Usage::

        >>> RGB = numpy.matrix([17.303501, 8.211033, 5.672498]).reshape((3, 1))
        >>> illuminant_RGB = (0.31271, 0.32902)
        >>> illuminant_XYZ = (0.34567, 0.35850)
        >>> chromatic_adaptation_method =  "Bradford"
        >>> to_XYZ = numpy.matrix([0.41238656, 0.35759149, 0.18045049, 0.21263682, 0.71518298, 0.0721802, 0.01933062, 0.11919716, 0.95037259]).reshape((3, 3)))
        >>> RGB_to_XYZ(RGB, illuminant_RGB, illuminant_XYZ, chromatic_adaptation_method, to_XYZ)
        matrix([[ 11.51847498],
                [ 10.0799999 ],
                [  5.08937278]])

    :param RGB: *RGB* colourspace matrix.
    :type RGB: matrix (3x1)
    :param illuminant_RGB: *RGB* colourspace *illuminant* chromaticity coordinates.
    :type illuminant_RGB: tuple
    :param illuminant_XYZ: *CIE XYZ* colourspace *illuminant* chromaticity coordinates.
    :type illuminant_XYZ: tuple
    :param chromatic_adaptation_method: *Chromatic adaptation* method.
    :type chromatic_adaptation_method: unicode
    :param to_XYZ: *Normalised primary matrix*.
    :type to_XYZ: matrix (3x3)
    :param inverse_transfer_function: *Inverse transfer function*.
    :type inverse_transfer_function: object
    :return: *CIE XYZ* colourspace matrix.
    :rtype: matrix (3x1)
    """

    if inverse_transfer_function is not None:
        RGB = numpy.matrix(map(lambda x: inverse_transfer_function(x), numpy.ravel(RGB))).reshape((3, 1))

    XYZ = to_XYZ * RGB

    cat = colour.computation.chromatic_adaptation.get_chromatic_adaptation_matrix(
        xy_to_XYZ(illuminant_RGB),
        xy_to_XYZ(illuminant_XYZ),
        method=chromatic_adaptation_method)

    adaptedXYZ = cat * XYZ

    LOGGER.debug("> 'CIE XYZ' matrix:\n{0}".format(repr(XYZ)))
    LOGGER.debug("> 'Chromatic adaptation' matrix:\n{0}".format(repr(cat)))
    LOGGER.debug("> Adapted 'CIE XYZ' matrix:\n{0}".format(repr(adaptedXYZ)))

    return adaptedXYZ


def xyY_to_RGB(xyY,
               illuminant_xyY,
               illuminant_RGB,
               chromatic_adaptation_method,
               from_XYZ,
               transfer_function=None):
    """
    Converts from *CIE xyY* colourspace to *RGB* colourspace using given *CIE xyY* matrix, *illuminants*,
    *chromatic adaptation* method, *normalised primary matrix* and *transfer function*.

    Usage::

        >>> xyY = numpy.matrix([0.4316, 0.3777, 10.08]).reshape((3, 1))
        >>> illuminant_xyY = (0.34567, 0.35850)
        >>> illuminant_RGB = (0.31271, 0.32902)
        >>> chromatic_adaptation_method =  "Bradford"
        >>> from_XYZ = numpy.matrix([ 3.24100326, -1.53739899, -0.49861587, -0.96922426,  1.87592999,  0.04155422, 0.05563942, -0.2040112 ,  1.05714897]).reshape((3, 3)))
        >>> xyY_to_RGB(xyY, illuminant_xyY, illuminant_RGB, chromatic_adaptation_method, from_XYZ)
        matrix([[ 17.30350095],
                [  8.21103314],
                [  5.67249761]])

    :param xyY: *CIE xyY* matrix.
    :type xyY: matrix (3x1)
    :param illuminant_xyY: *CIE xyY* colourspace *illuminant* chromaticity coordinates.
    :type illuminant_xyY: tuple
    :param illuminant_RGB: *RGB* colourspace *illuminant* chromaticity coordinates.
    :type illuminant_RGB: tuple
    :param chromatic_adaptation_method: *Chromatic adaptation* method.
    :type chromatic_adaptation_method: unicode
    :param from_XYZ: *Normalised primary matrix*.
    :type from_XYZ: matrix (3x3)
    :param transfer_function: *Transfer function*.
    :type transfer_function: object
    :return: *RGB* colourspace matrix.
    :rtype: matrix (3x1)
    """

    return XYZ_to_RGB(xyY_to_XYZ(xyY),
                      illuminant_xyY,
                      illuminant_RGB,
                      chromatic_adaptation_method,
                      from_XYZ,
                      transfer_function)


def RGB_to_xyY(RGB,
               illuminant_RGB,
               illuminant_xyY,
               chromatic_adaptation_method,
               to_XYZ,
               inverse_transfer_function=None):
    """
    Converts from *RGB* colourspace to *CIE xyY* colourspace using given *RGB* matrix, *illuminants*,
    *chromatic adaptation* method, *normalised primary matrix* and *transfer function*.

    Usage::

        >>> RGB = numpy.matrix([17.303501, 8.211033, 5.672498]).reshape((3, 1))
        >>> illuminant_RGB = (0.31271, 0.32902)
        >>> illuminant_xyY = (0.34567, 0.35850)
        >>> chromatic_adaptation_method = "Bradford"
        >>> to_XYZ = numpy.matrix([0.41238656, 0.35759149, 0.18045049, 0.21263682, 0.71518298, 0.0721802, 0.01933062, 0.11919716, 0.95037259]).reshape((3, 3)))
        >>> RGB_to_xyY(RGB, illuminant_RGB, illuminant_xyY, chromatic_adaptation_method, to_XYZ)
        matrix([[  0.4316    ],
                [  0.37769999],
                [ 10.0799999 ]])

    :param RGB: *RGB* colourspace matrix.
    :type RGB: matrix (3x1)
    :param illuminant_RGB: *RGB* colourspace *illuminant* chromaticity coordinates.
    :type illuminant_RGB: tuple
    :param illuminant_xyY: *CIE xyY* colourspace *illuminant* chromaticity coordinates.
    :type illuminant_xyY: tuple
    :param chromatic_adaptation_method: *Chromatic adaptation* method.
    :type chromatic_adaptation_method: unicode
    :param to_XYZ: *Normalised primary* matrix.
    :type to_XYZ: matrix (3x3)
    :param inverse_transfer_function: *Inverse transfer* function.
    :type inverse_transfer_function: object
    :return: *CIE xyY* matrix.
    :rtype: matrix (3x1)
    """

    return XYZ_to_xyY(RGB_to_XYZ(RGB,
                                 illuminant_RGB,
                                 illuminant_xyY,
                                 chromatic_adaptation_method,
                                 to_XYZ,
                                 inverse_transfer_function))


def is_within_macadam_limits(xyY, illuminant):
    if not colour.utilities.common.is_scipy_installed():
        # TODO: Error.
        raise