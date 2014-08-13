#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RGB Colourspace Transformations
===============================

Defines the *RGB* colourspace transformations:

-   :func:`XYZ_to_RGB`
-   :func:`RGB_to_XYZ`
-   :func:`RGB_to_RGB`
"""

from __future__ import unicode_literals

import numpy as np

from colour.models import xy_to_XYZ
from colour.adaptation import get_chromatic_adaptation_matrix

__author__ = "Colour Developers"
__copyright__ = "Copyright (C) 2013 - 2014 - Colour Developers"
__license__ = "New BSD License - http://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-science@googlegroups.com"
__status__ = "Production"

__all__ = ["XYZ_to_RGB",
           "RGB_to_XYZ",
           "RGB_to_RGB"]


def XYZ_to_RGB(XYZ,
               illuminant_XYZ,
               illuminant_RGB,
               to_RGB,
               chromatic_adaptation_method="CAT02",
               transfer_function=None):
    """
    Converts from *CIE XYZ* colourspace to *RGB* colourspace using given
    *CIE XYZ* colourspace matrix, *illuminants*, *chromatic adaptation* method,
    *normalised primary matrix* and *transfer function*.

    Parameters
    ----------
    XYZ : array_like, (3, 1)
        *CIE XYZ* colourspace matrix.
    illuminant_XYZ : array_like
        *CIE XYZ* colourspace *illuminant* *xy* chromaticity coordinates.
    illuminant_RGB : array_like
        *RGB* colourspace *illuminant* *xy* chromaticity coordinates.
    to_RGB : array_like, (3, 3)
        *Normalised primary matrix*.
    chromatic_adaptation_method : unicode, optional
        ("XYZ Scaling", "Bradford", "Von Kries", "Fairchild", "CAT02")
        *Chromatic adaptation* method.
    transfer_function : object, optional
        *Transfer function*.

    Returns
    -------
    ndarray, (3, 1)
        *RGB* colourspace matrix.

    Notes
    -----
    -   Input *CIE XYZ* colourspace matrix is in domain [0, 1].
    -   Input *illuminant_XYZ* *xy* chromaticity coordinates are in domain
        [0, 1].
    -   Input *illuminant_RGB* *xy* chromaticity coordinates are in domain
        [0, 1].
    -   Output *RGB* colourspace matrix is in domain [0, 1].

    Examples
    --------
    >>> XYZ = np.array([0.1151847498, 0.1008, 0.0508937252])
    >>> illuminant_XYZ =  (0.34567, 0.35850)
    >>> illuminant_RGB =  (0.31271, 0.32902)
    >>> chromatic_adaptation_method =  "Bradford"
    >>> to_RGB =  np.array([3.24100326, -1.53739899, -0.49861587, -0.96922426,  1.87592999,  0.04155422, 0.05563942, -0.2040112 ,  1.05714897]).reshape((3, 3))
    >>> colour.XYZ_to_RGB(XYZ, illuminant_XYZ, illuminant_RGB, to_RGB, chromatic_adaptation_method)
    array([[ 0.17303501],
           [ 0.08211033],
           [ 0.05672498]])
    """

    cat = get_chromatic_adaptation_matrix(xy_to_XYZ(illuminant_XYZ),
                                          xy_to_XYZ(illuminant_RGB),
                                          method=chromatic_adaptation_method)

    adapted_XYZ = np.dot(cat, XYZ)

    RGB = np.dot(to_RGB, adapted_XYZ)

    if transfer_function is not None:
        RGB = np.array([transfer_function(x) for x in np.ravel(RGB)])

    return RGB.reshape((3, 1))


def RGB_to_XYZ(RGB,
               illuminant_RGB,
               illuminant_XYZ,
               to_XYZ,
               chromatic_adaptation_method="CAT02",
               inverse_transfer_function=None):
    """
    Converts from *RGB* colourspace to *CIE XYZ* colourspace using given
    *RGB* colourspace matrix, *illuminants*, *chromatic adaptation* method,
    *normalised primary matrix* and *transfer function*.

    Parameters
    ----------
    RGB : array_like, (3, 1)
        *RGB* colourspace matrix.
    illuminant_RGB : array_like
        *RGB* colourspace *illuminant* chromaticity coordinates.
    illuminant_XYZ : array_like
        *CIE XYZ* colourspace *illuminant* chromaticity coordinates.
    to_XYZ : array_like, (3, 3)
        *Normalised primary matrix*.
    chromatic_adaptation_method : unicode, optional
        ("XYZ Scaling", "Bradford", "Von Kries", "Fairchild", "CAT02")
        *Chromatic adaptation* method.
    inverse_transfer_function : object, optional
        *Inverse transfer function*.

    Returns
    -------
    ndarray, (3, 1)
        *CIE XYZ* colourspace matrix.

    Notes
    -----
    -   Input *RGB* colourspace matrix is in domain [0, 1].
    -   Input *illuminant_RGB* *xy* chromaticity coordinates are in domain
        [0, 1].
    -   Input *illuminant_XYZ* *xy* chromaticity coordinates are in domain
        [0, 1].
    -   Output *CIE XYZ* colourspace matrix is in domain [0, 1].

    Examples
    --------
    >>> RGB = np.array([0.17303501, 0.08211033, 0.05672498])
    >>> illuminant_RGB = (0.31271, 0.32902)
    >>> illuminant_XYZ = (0.34567, 0.35850)
    >>> chromatic_adaptation_method =  "Bradford"
    >>> to_XYZ = np.array([0.41238656, 0.35759149, 0.18045049, 0.21263682, 0.71518298, 0.0721802, 0.01933062, 0.11919716, 0.95037259]).reshape((3, 3))
    >>> colour.RGB_to_XYZ(RGB, illuminant_RGB, illuminant_XYZ, to_XYZ, chromatic_adaptation_method)
    array([[ 0.11518475],
           [ 0.1008    ],
           [ 0.05089373]])
    """

    if inverse_transfer_function is not None:
        RGB = np.array([inverse_transfer_function(x)
                        for x in np.ravel(RGB)]).reshape((3, 1))

    XYZ = np.dot(to_XYZ, RGB)

    cat = get_chromatic_adaptation_matrix(
        xy_to_XYZ(illuminant_RGB),
        xy_to_XYZ(illuminant_XYZ),
        method=chromatic_adaptation_method)

    adapted_XYZ = np.dot(cat, XYZ)

    return adapted_XYZ


def RGB_to_RGB(RGB,
               input_colourspace,
               output_colourspace,
               chromatic_adaptation_method="CAT02"):
    """
    Converts from given input *RGB* colourspace to output *RGB* colourspace
    using given *chromatic adaptation* method.

    Parameters
    ----------
    RGB : array_like, (3, 1)
        *RGB* colourspace matrix.
    input_colourspace : RGB_Colourspace
        *RGB* input colourspace.
    output_colourspace : RGB_Colourspace
        *RGB* output colourspace.
    chromatic_adaptation_method : unicode, optional
        ("XYZ Scaling", "Bradford", "Von Kries", "Fairchild", "CAT02")
        *Chromatic adaptation* method.

    ndarray, (3, 1)
        *RGB* colourspace matrix.

    Notes
    -----
    -   *RGB* colourspace matrices are in domain [0, 1].

    Examples
    --------
    >>> RGB = np.array([0.35521588, 0.41, 0.24177934])
    >>> colour.RGB_to_RGB(RGB, colour.sRGB_COLOURSPACE, colour.PROPHOTO_RGB_COLOURSPACE)
    array([[ 0.35735427],
           [ 0.39987346],
           [ 0.26348887]])
    """

    cat = get_chromatic_adaptation_matrix(
        xy_to_XYZ(input_colourspace.whitepoint),
        xy_to_XYZ(output_colourspace.whitepoint),
        chromatic_adaptation_method)

    trs_matrix = np.dot(output_colourspace.to_RGB,
                        np.dot(cat, input_colourspace.to_XYZ))

    return np.dot(trs_matrix, RGB).reshape((3, 1))