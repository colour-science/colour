#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RGB Colourspace & Transformations
=================================

Defines the :class:`RGB_Colourspace` class for the *RGB* colourspaces dataset
from :mod:`colour.models.dataset.aces_rgb`, etc... and the following *RGB*
colourspace transformations:

-   :func:`XYZ_to_RGB`
-   :func:`RGB_to_XYZ`
-   :func:`RGB_to_RGB`

See Also
--------
`RGB Colourspaces IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/\
master/notebooks/models/rgb.ipynb>`_
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.models import xy_to_XYZ, xy_to_xyY, xyY_to_XYZ
from colour.adaptation import chromatic_adaptation_matrix_VonKries
from colour.utilities import dot_matrix, dot_vector

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['RGB_Colourspace',
           'XYZ_to_RGB',
           'RGB_to_XYZ',
           'RGB_to_RGB']


class RGB_Colourspace(object):
    """
    Implements support for the *RGB* colourspaces dataset from
    :mod:`colour.models.dataset.aces_rgb`, etc....

    Parameters
    ----------
    name : unicode
        *RGB* colourspace name.
    primaries : array_like
        *RGB* colourspace primaries.
    whitepoint : array_like
        *RGB* colourspace whitepoint.
    illuminant : unicode, optional
        *RGB* colourspace whitepoint name as illuminant.
    RGB_to_XYZ_matrix : array_like, optional
        Transformation matrix from colourspace to *CIE XYZ* tristimulus values.
    XYZ_to_RGB_matrix : array_like, optional
        Transformation matrix from *CIE XYZ* tristimulus values to colourspace.
    OECF : object, optional
        Opto-electronic conversion function (OECF) that maps estimated
        tristimulus values in a scene to :math:`R'G'B'` video component signal
        value.
    EOCF : object, optional
        Electro-optical conversion function (EOCF) that maps an :math:`R'G'B'`
        video component signal to a tristimulus value at the display.
    """

    def __init__(self,
                 name,
                 primaries,
                 whitepoint,
                 illuminant=None,
                 RGB_to_XYZ_matrix=None,
                 XYZ_to_RGB_matrix=None,
                 OECF=None,
                 EOCF=None):
        self.__name = None
        self.name = name
        self.__primaries = None
        self.primaries = primaries
        self.__whitepoint = None
        self.whitepoint = whitepoint
        self.__illuminant = None
        self.illuminant = illuminant
        self.__RGB_to_XYZ_matrix = None
        self.RGB_to_XYZ_matrix = RGB_to_XYZ_matrix
        self.__XYZ_to_RGB_matrix = None
        self.XYZ_to_RGB_matrix = XYZ_to_RGB_matrix
        self.__OECF = None
        self.OECF = OECF
        self.__EOCF = None
        self.EOCF = EOCF

    @property
    def name(self):
        """
        Property for **self.__name** private attribute.

        Returns
        -------
        unicode
            self.__name.
        """

        return self.__name

    @name.setter
    def name(self, value):
        """
        Setter for **self.__name** private attribute.

        Parameters
        ----------
        value : unicode
            Attribute value.
        """

        if value is not None:
            assert isinstance(value, basestring), (  # noqa
                ('"{0}" attribute: "{1}" is not a '
                 '"basestring" instance!').format('name', value))
        self.__name = value

    @property
    def primaries(self):
        """
        Property for **self.__primaries** private attribute.

        Returns
        -------
        array_like, (3, 2)
            self.__primaries.
        """

        return self.__primaries

    @primaries.setter
    def primaries(self, value):
        """
        Setter for **self.__primaries** private attribute.

        Parameters
        ----------
        value : array_like, (3, 2)
            Attribute value.
        """

        if value is not None:
            value = np.asarray(value)
        self.__primaries = value

    @property
    def whitepoint(self):
        """
        Property for **self.__whitepoint** private attribute.

        Returns
        -------
        array_like
            self.__whitepoint.
        """

        return self.__whitepoint

    @whitepoint.setter
    def whitepoint(self, value):
        """
        Setter for **self.__whitepoint** private attribute.

        Parameters
        ----------
        value : array_like
            Attribute value.
        """

        if value is not None:
            assert isinstance(value, (tuple, list, np.ndarray, np.matrix)), (
                ('"{0}" attribute: "{1}" is not a "tuple", "list", "ndarray" '
                 'or "matrix" instance!').format('whitepoint', value))
        self.__whitepoint = value

    @property
    def illuminant(self):
        """
        Property for **self.__illuminant** private attribute.

        Returns
        -------
        unicode
            self.__illuminant.
        """

        return self.__illuminant

    @illuminant.setter
    def illuminant(self, value):
        """
        Setter for **self.__illuminant** private attribute.

        Parameters
        ----------
        value : unicode
            Attribute value.
        """

        if value is not None:
            assert isinstance(value, basestring), (  # noqa
                ('"{0}" attribute: "{1}" is not a '
                 '"basestring" instance!').format('illuminant', value))
        self.__illuminant = value

    @property
    def RGB_to_XYZ_matrix(self):
        """
        Property for **self.__to_XYZ** private attribute.

        Returns
        -------
        array_like, (3, 3)
            self.__to_XYZ.
        """

        return self.__RGB_to_XYZ_matrix

    @RGB_to_XYZ_matrix.setter
    def RGB_to_XYZ_matrix(self, value):
        """
        Setter for **self.__to_XYZ** private attribute.

        Parameters
        ----------
        value : array_like
            Attribute value.
        """

        if value is not None:
            value = np.asarray(value)
        self.__RGB_to_XYZ_matrix = value

    @property
    def XYZ_to_RGB_matrix(self):
        """
        Property for **self.__to_RGB** private attribute.

        Returns
        -------
        array_like, (3, 3)
            self.__to_RGB.
        """

        return self.__XYZ_to_RGB_matrix

    @XYZ_to_RGB_matrix.setter
    def XYZ_to_RGB_matrix(self, value):
        """
        Setter for **self.__to_RGB** private attribute.

        Parameters
        ----------
        value : array_like
            Attribute value.
        """

        if value is not None:
            value = np.asarray(value)
        self.__XYZ_to_RGB_matrix = value

    @property
    def OECF(self):
        """
        Property for **self.__OECF** private attribute.

        Returns
        -------
        object
            self.__OECF.
        """

        return self.__OECF

    @OECF.setter
    def OECF(self, value):
        """
        Setter for **self.__OECF** private attribute.

        Parameters
        ----------
        value : object
            Attribute value.
        """

        if value is not None:
            assert hasattr(value, '__call__'), (
                '"{0}" attribute: "{1}" is not callable!'.format(
                    'OECF', value))
        self.__OECF = value

    @property
    def EOCF(self):
        """
        Property for **self.__EOCF** private attribute.

        Returns
        -------
        object
            self.__EOCF.
        """

        return self.__EOCF

    @EOCF.setter
    def EOCF(self, value):
        """
        Setter for **self.__EOCF** private attribute.

        Parameters
        ----------
        value : object
            Attribute value.
        """

        if value is not None:
            assert hasattr(value, '__call__'), (
                '"{0}" attribute: "{1}" is not callable!'.format(
                    'EOCF', value))
        self.__EOCF = value


def XYZ_to_RGB(XYZ,
               illuminant_XYZ,
               illuminant_RGB,
               XYZ_to_RGB_matrix,
               chromatic_adaptation_transform='CAT02',
               OECF=None):
    """
    Converts from *CIE XYZ* tristimulus values to given *RGB* colourspace.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values.
    illuminant_XYZ : array_like
        *CIE XYZ* tristimulus values *illuminant* *xy* chromaticity coordinates
        or *CIE xyY* colourspace array.
    illuminant_RGB : array_like
        *RGB* colourspace *illuminant* *xy* chromaticity coordinates or
        *CIE xyY* colourspace array.
    XYZ_to_RGB_matrix : array_like
        *Normalised primary matrix*.
    chromatic_adaptation_transform : unicode, optional
        **{'CAT02', 'XYZ Scaling', 'Von Kries', 'Bradford', 'Sharp',
        'Fairchild, 'CMCCAT97', 'CMCCAT2000', 'CAT02_BRILL_CAT', 'Bianco',
        'Bianco PC'}**,
        *Chromatic adaptation* transform.
    OECF : object, optional
        *Opto-electronic conversion function*.

    Returns
    -------
    ndarray
        *RGB* colourspace array.

    Notes
    -----
    -   Input *CIE XYZ* tristimulus values are in domain [0, 1].
    -   Input *illuminant_XYZ* *xy* chromaticity coordinates or *CIE xyY*
        colourspace array are in domain [0, :math:`\infty`].
    -   Input *illuminant_RGB* *xy* chromaticity coordinates or *CIE xyY*
        colourspace array are in domain [0, :math:`\infty`].
    -   Output *RGB* colourspace array is in domain [0, 1].

    Examples
    --------
    >>> XYZ = np.array([0.07049534, 0.10080000, 0.09558313])
    >>> illuminant_XYZ = np.array([0.34567, 0.35850])
    >>> illuminant_RGB = np.array([0.31271, 0.32902])
    >>> chromatic_adaptation_transform = 'Bradford'
    >>> XYZ_to_RGB_matrix = np.array([
    ...     [3.24100326, -1.53739899, -0.49861587],
    ...     [-0.96922426, 1.87592999, 0.04155422],
    ...     [0.05563942, -0.20401120, 1.05714897]])
    >>> XYZ_to_RGB(
    ...     XYZ,
    ...     illuminant_XYZ,
    ...     illuminant_RGB,
    ...     XYZ_to_RGB_matrix,
    ...     chromatic_adaptation_transform)  # doctest: +ELLIPSIS
    array([ 0.0110360...,  0.1273446...,  0.1163103...])
    """

    M = chromatic_adaptation_matrix_VonKries(
        xyY_to_XYZ(xy_to_xyY(illuminant_XYZ)),
        xyY_to_XYZ(xy_to_xyY(illuminant_RGB)),
        transform=chromatic_adaptation_transform)

    XYZ_a = dot_vector(M, XYZ)

    RGB = dot_vector(XYZ_to_RGB_matrix, XYZ_a)

    if OECF is not None:
        RGB = OECF(RGB)

    return RGB


def RGB_to_XYZ(RGB,
               illuminant_RGB,
               illuminant_XYZ,
               RGB_to_XYZ_matrix,
               chromatic_adaptation_transform='CAT02',
               EOCF=None):
    """
    Converts from given *RGB* colourspace to *CIE XYZ* tristimulus values.

    Parameters
    ----------
    RGB : array_like
        *RGB* colourspace array.
    illuminant_RGB : array_like
        *RGB* colourspace *illuminant* chromaticity coordinates or *CIE xyY*
        colourspace array.
    illuminant_XYZ : array_like
        *CIE XYZ* tristimulus values *illuminant* chromaticity coordinates or
        *CIE xyY* colourspace array.
    RGB_to_XYZ_matrix : array_like
        *Normalised primary matrix*.
    chromatic_adaptation_transform : unicode, optional
        **{'CAT02', 'XYZ Scaling', 'Von Kries', 'Bradford', 'Sharp',
        'Fairchild, 'CMCCAT97', 'CMCCAT2000', 'CAT02_BRILL_CAT', 'Bianco',
        'Bianco PC'}**,
        *Chromatic adaptation* transform.
    EOCF : object, optional
        *Electro-optical conversion function*.

    Returns
    -------
    ndarray
        *CIE XYZ* tristimulus values.

    Notes
    -----
    -   Input *RGB* colourspace array is in domain [0, 1].
    -   Input *illuminant_RGB* *xy* chromaticity coordinates or *CIE xyY*
        colourspace array are in domain [0, :math:`\infty`].
    -   Input *illuminant_XYZ* *xy* chromaticity coordinates or *CIE xyY*
        colourspace array are in domain [0, :math:`\infty`].
    -   Output *CIE XYZ* tristimulus values are in domain [0, 1].

    Examples
    --------
    >>> RGB = np.array([0.01103604, 0.12734466, 0.11631037])
    >>> illuminant_RGB = np.array([0.31271, 0.32902])
    >>> illuminant_XYZ = np.array([0.34567, 0.35850])
    >>> chromatic_adaptation_transform = 'Bradford'
    >>> RGB_to_XYZ_matrix = np.array([
    ...     [0.41238656, 0.35759149, 0.18045049],
    ...     [0.21263682, 0.71518298, 0.07218020],
    ...     [0.01933062, 0.11919716, 0.95037259]])
    >>> RGB_to_XYZ(
    ...     RGB,
    ...     illuminant_RGB,
    ...     illuminant_XYZ,
    ...     RGB_to_XYZ_matrix,
    ...     chromatic_adaptation_transform)  # doctest: +ELLIPSIS
    array([ 0.0704953...,  0.1008    ,  0.0955831...])
    """

    if EOCF is not None:
        RGB = EOCF(RGB)

    M = chromatic_adaptation_matrix_VonKries(
        xyY_to_XYZ(xy_to_xyY(illuminant_RGB)),
        xyY_to_XYZ(xy_to_xyY(illuminant_XYZ)),
        transform=chromatic_adaptation_transform)

    XYZ = dot_vector(RGB_to_XYZ_matrix, RGB)

    XYZ_a = dot_vector(M, XYZ)

    return XYZ_a


def RGB_to_RGB(RGB,
               input_colourspace,
               output_colourspace,
               chromatic_adaptation_transform='CAT02'):
    """
    Converts from given input *RGB* colourspace to output *RGB* colourspace
    using given *chromatic adaptation* method.

    Parameters
    ----------
    RGB : array_like
        *RGB* colourspace array.
    input_colourspace : RGB_Colourspace
        *RGB* input colourspace.
    output_colourspace : RGB_Colourspace
        *RGB* output colourspace.
    chromatic_adaptation_transform : unicode, optional
        **{'CAT02', 'XYZ Scaling', 'Von Kries', 'Bradford', 'Sharp',
        'Fairchild, 'CMCCAT97', 'CMCCAT2000', 'CAT02_BRILL_CAT', 'Bianco',
        'Bianco PC'}**,
        *Chromatic adaptation* transform.

    Returns
    -------
    ndarray
        *RGB* colourspace array.

    Notes
    -----
    -   *RGB* colourspace arrays are in domain [0, 1].

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

    cat = chromatic_adaptation_matrix_VonKries(
        xy_to_XYZ(input_colourspace.whitepoint),
        xy_to_XYZ(output_colourspace.whitepoint),
        chromatic_adaptation_transform)

    M = dot_matrix(cat, input_colourspace.RGB_to_XYZ_matrix)
    M = dot_matrix(output_colourspace.XYZ_to_RGB_matrix, M)

    RGB = dot_vector(M, RGB)

    return RGB
