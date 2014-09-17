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
               XYZ_to_RGB_matrix,
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
    XYZ_to_RGB_matrix : array_like, (3, 3)
        *Normalised primary matrix*.
    chromatic_adaptation_method : unicode, optional
        {'CAT02', 'XYZ Scaling', 'Von Kries', 'Bradford', 'Sharp', 'Fairchild,
        'CMCCAT97', 'CMCCAT2000', 'Bianco', 'Bianco PC'},
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
    >>> XYZ = np.array([0.07049534, 0.1008, 0.09558313])
    >>> illuminant_XYZ = (0.34567, 0.35850)
    >>> illuminant_RGB = (0.31271, 0.32902)
    >>> chromatic_adaptation_method = 'Bradford'
    >>> XYZ_to_RGB_matrix = np.array([
    ...     [3.24100326, -1.53739899, -0.49861587],
    ...     [-0.96922426, 1.87592999, 0.04155422],
    ...     [0.05563942, -0.2040112, 1.05714897]])
    >>> XYZ_to_RGB(
    ...     XYZ,
    ...     illuminant_XYZ,
    ...     illuminant_RGB,
    ...     XYZ_to_RGB_matrix,
    ...     chromatic_adaptation_method)  # doctest: +ELLIPSIS
    array([ 0.0110360...,  0.1273446...,  0.1163103...])
    """

    np.array([
        [3.24100326, -1.53739899, -0.49861587],
        [-0.96922426, 1.87592999, 0.04155422],
        [0.05563942, -0.2040112, 1.05714897]])
    cat = chromatic_adaptation_matrix(xy_to_XYZ(illuminant_XYZ),
                                      xy_to_XYZ(illuminant_RGB),
                                      method=chromatic_adaptation_method)

    adapted_XYZ = np.dot(cat, XYZ)

    RGB = np.dot(XYZ_to_RGB_matrix.reshape((3, 3)),
                 adapted_XYZ.reshape((3, 1)))

    if transfer_function is not None:
        RGB = np.array([transfer_function(x) for x in np.ravel(RGB)])

    return np.ravel(RGB)


def RGB_to_XYZ(RGB,
               illuminant_RGB,
               illuminant_XYZ,
               RGB_to_XYZ_matrix,
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
    RGB_to_XYZ_matrix : array_like, (3, 3)
        *Normalised primary matrix*.
    chromatic_adaptation_method : unicode, optional
        {'CAT02', 'XYZ Scaling', 'Von Kries', 'Bradford', 'Sharp', 'Fairchild,
        'CMCCAT97', 'CMCCAT2000', 'Bianco', 'Bianco PC'},
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
    >>> RGB = np.array([0.01103604, 0.12734466, 0.11631037])
    >>> illuminant_RGB = (0.31271, 0.32902)
    >>> illuminant_XYZ = (0.34567, 0.35850)
    >>> chromatic_adaptation_method = 'Bradford'
    >>> RGB_to_XYZ_matrix = np.array([
    ...     [0.41238656, 0.35759149, 0.18045049],
    ...     [0.21263682, 0.71518298, 0.0721802],
    ...     [0.01933062, 0.11919716, 0.95037259]])
    >>> RGB_to_XYZ(
    ...     RGB,
    ...     illuminant_RGB,
    ...     illuminant_XYZ,
    ...     RGB_to_XYZ_matrix,
    ...     chromatic_adaptation_method)  # doctest: +ELLIPSIS
    array([ 0.0704953...,  0.1008    ,  0.0955831...])
    """

    if inverse_transfer_function is not None:
        RGB = np.array([inverse_transfer_function(x)
                        for x in np.ravel(RGB)])

    XYZ = np.dot(RGB_to_XYZ_matrix.reshape((3, 3)), RGB.reshape((3, 1)))

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
        {'CAT02', 'XYZ Scaling', 'Von Kries', 'Bradford', 'Sharp', 'Fairchild,
        'CMCCAT97', 'CMCCAT2000', 'Bianco', 'Bianco PC'},
        *Chromatic adaptation* method.

    ndarray, (3,)
        *RGB* colourspace matrix.

    Notes
    -----
    -   *RGB* colourspace matrices are in domain [0, 1].

    Examples
    --------
    >>> from colour import sRGB_COLOURSPACE, PROPHOTO_RGB_COLOURSPACE
    >>> RGB = np.array([0.01103604, 0.12734466, 0.11631037])
    >>> RGB_to_RGB(
    ...     RGB,
    ...     sRGB_COLOURSPACE,
    ...     PROPHOTO_RGB_COLOURSPACE)  # doctest: +ELLIPSIS
    array([ 0.0643338...,  0.1157362...,  0.1157614...])
    """

    cat = chromatic_adaptation_matrix(
        xy_to_XYZ(input_colourspace.whitepoint),
        xy_to_XYZ(output_colourspace.whitepoint),
        chromatic_adaptation_method)

    trs_matrix = np.dot(output_colourspace.XYZ_to_RGB_matrix,
                        np.dot(cat, input_colourspace.RGB_to_XYZ_matrix))

    return np.dot(trs_matrix, RGB)
