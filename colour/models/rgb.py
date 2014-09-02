#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RGB Colourspace Transformations
===============================

Defines the *RGB* colourspace transformations:

-   :func:`XYZ_to_RGB`
-   :func:`RGB_to_XYZ`
-   :func:`RGB_to_RGB`

See Also
--------
`RGB Colourspaces IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/master/notebooks/models/rgb.ipynb>`_  # noqa
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.models import xy_to_XYZ
from colour.adaptation import chromatic_adaptation_matrix

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['XYZ_to_RGB',
           'RGB_to_XYZ',
           'RGB_to_RGB']


def XYZ_to_RGB(XYZ,
               illuminant_XYZ,
               illuminant_RGB,
               to_RGB,
               chromatic_adaptation_method='CAT02',
               transfer_function=None):
    """
    Converts from *CIE XYZ* colourspace to *RGB* colourspace using given
    *CIE XYZ* colourspace matrix, *illuminants*, *chromatic adaptation* method,
    *normalised primary matrix* and *transfer function*.

    Parameters
    ----------
    XYZ : array_like, (3,)
        *CIE XYZ* colourspace matrix.
    illuminant_XYZ : array_like
        *CIE XYZ* colourspace *illuminant* *xy* chromaticity coordinates.
    illuminant_RGB : array_like
        *RGB* colourspace *illuminant* *xy* chromaticity coordinates.
    to_RGB : array_like, (3, 3)
        *Normalised primary matrix*.
    chromatic_adaptation_method : unicode, optional
        ('XYZ Scaling', 'Bradford', 'Von Kries', 'Fairchild', 'CAT02')
        *Chromatic adaptation* method.
    transfer_function : object, optional
        *Transfer function*.

    Returns
    -------
    ndarray, (3,)
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
    >>> illuminant_XYZ = (0.34567, 0.35850)
    >>> illuminant_RGB = (0.31271, 0.32902)
    >>> chromatic_adaptation_method =  'Bradford'
    >>> to_RGB = np.array([
    ...     [3.24100326, -1.53739899, -0.49861587],
    ...     [-0.96922426, 1.87592999, 0.04155422],
    ...     [0.05563942, -0.2040112, 1.05714897]])
    >>> XYZ_to_RGB(
    ...     XYZ,
    ...     illuminant_XYZ,
    ...     illuminant_RGB,
    ...     to_RGB,
    ...     chromatic_adaptation_method)  # doctest: +ELLIPSIS
    array([ 0.1730350...,  0.0821103...,  0.0567249...])
    """

    np.array([
        [3.24100326, -1.53739899, -0.49861587],
        [-0.96922426, 1.87592999, 0.04155422],
        [0.05563942, -0.2040112, 1.05714897]])
    cat = chromatic_adaptation_matrix(xy_to_XYZ(illuminant_XYZ),
                                      xy_to_XYZ(illuminant_RGB),
                                      method=chromatic_adaptation_method)

    adapted_XYZ = np.dot(cat, XYZ)

    RGB = np.dot(to_RGB.reshape((3, 3)), adapted_XYZ.reshape((3, 1)))

    if transfer_function is not None:
        RGB = np.array([transfer_function(x) for x in np.ravel(RGB)])

    return np.ravel(RGB)


def RGB_to_XYZ(RGB,
               illuminant_RGB,
               illuminant_XYZ,
               to_XYZ,
               chromatic_adaptation_method='CAT02',
               inverse_transfer_function=None):
    """
    Converts from *RGB* colourspace to *CIE XYZ* colourspace using given
    *RGB* colourspace matrix, *illuminants*, *chromatic adaptation* method,
    *normalised primary matrix* and *transfer function*.

    Parameters
    ----------
    RGB : array_like, (3,)
        *RGB* colourspace matrix.
    illuminant_RGB : array_like
        *RGB* colourspace *illuminant* chromaticity coordinates.
    illuminant_XYZ : array_like
        *CIE XYZ* colourspace *illuminant* chromaticity coordinates.
    to_XYZ : array_like, (3, 3)
        *Normalised primary matrix*.
    chromatic_adaptation_method : unicode, optional
        ('XYZ Scaling', 'Bradford', 'Von Kries', 'Fairchild', 'CAT02')
        *Chromatic adaptation* method.
    inverse_transfer_function : object, optional
        *Inverse transfer function*.

    Returns
    -------
    ndarray, (3,)
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
    >>> chromatic_adaptation_method = 'Bradford'
    >>> to_XYZ = np.array([
    ...     [0.41238656, 0.35759149, 0.18045049],
    ...     [0.21263682, 0.71518298, 0.0721802],
    ...     [0.01933062, 0.11919716, 0.95037259]])
    >>> RGB_to_XYZ(
    ...     RGB,
    ...     illuminant_RGB,
    ...     illuminant_XYZ,
    ...     to_XYZ,
    ...     chromatic_adaptation_method)  # doctest: +ELLIPSIS
    array([ 0.1151847...,  0.1008    ,  0.0508937...])
    """

    if inverse_transfer_function is not None:
        RGB = np.array([inverse_transfer_function(x)
                        for x in np.ravel(RGB)])

    XYZ = np.dot(to_XYZ.reshape((3, 3)), RGB.reshape((3, 1)))

    cat = chromatic_adaptation_matrix(
        xy_to_XYZ(illuminant_RGB),
        xy_to_XYZ(illuminant_XYZ),
        method=chromatic_adaptation_method)

    adapted_XYZ = np.dot(cat, XYZ.reshape((3, 1)))

    return np.ravel(adapted_XYZ)


def RGB_to_RGB(RGB,
               input_colourspace,
               output_colourspace,
               chromatic_adaptation_method='CAT02'):
    """
    Converts from given input *RGB* colourspace to output *RGB* colourspace
    using given *chromatic adaptation* method.

    Parameters
    ----------
    RGB : array_like, (3,)
        *RGB* colourspace matrix.
    input_colourspace : RGB_Colourspace
        *RGB* input colourspace.
    output_colourspace : RGB_Colourspace
        *RGB* output colourspace.
    chromatic_adaptation_method : unicode, optional
        ('XYZ Scaling', 'Bradford', 'Von Kries', 'Fairchild', 'CAT02')
        *Chromatic adaptation* method.

    ndarray, (3,)
        *RGB* colourspace matrix.

    Notes
    -----
    -   *RGB* colourspace matrices are in domain [0, 1].

    Examples
    --------
    >>> from colour import sRGB_COLOURSPACE, PROPHOTO_RGB_COLOURSPACE
    >>> RGB = np.array([0.35521588, 0.41, 0.24177934])
    >>> RGB_to_RGB(
    ...     RGB,
    ...     sRGB_COLOURSPACE,
    ...     PROPHOTO_RGB_COLOURSPACE)  # doctest: +ELLIPSIS
    array([ 0.3579334...,  0.4007138...,  0.2615704...])
    """

    cat = chromatic_adaptation_matrix(
        xy_to_XYZ(input_colourspace.whitepoint),
        xy_to_XYZ(output_colourspace.whitepoint),
        chromatic_adaptation_method)

    trs_matrix = np.dot(output_colourspace.to_RGB,
                        np.dot(cat, input_colourspace.to_XYZ))

    return np.dot(trs_matrix, RGB)
