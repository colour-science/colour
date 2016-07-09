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
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/blob/\
master/notebooks/models/rgb.ipynb>`_
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.models import xy_to_XYZ, xy_to_xyY, xyY_to_XYZ
from colour.adaptation import chromatic_adaptation_matrix_VonKries
from colour.utilities import dot_matrix, dot_vector

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2016 - Colour Developers'
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
    encoding_cctf : object, optional
        Encoding colour component transfer function (Encoding CCTF) /
        opto-electronic transfer function (OETF / OECF) that maps estimated
        tristimulus values in a scene to :math:`R'G'B'` video component signal
        value.
    decoding_cctf : object, optional
        Decoding colour component transfer function (Decoding CCTF) /
        electro-optical transfer function (EOTF / EOCF) that maps an
        :math:`R'G'B'` video component signal value to tristimulus values at
        the display.

    Attributes
    ----------
    name
    primaries
    whitepoint
    illuminant
    RGB_to_XYZ_matrix
    XYZ_to_RGB_matrix
    encoding_cctf
    decoding_cctf
    """

    def __init__(self,
                 name,
                 primaries,
                 whitepoint,
                 illuminant=None,
                 RGB_to_XYZ_matrix=None,
                 XYZ_to_RGB_matrix=None,
                 encoding_cctf=None,
                 decoding_cctf=None):
        self._name = None
        self.name = name
        self._primaries = None
        self.primaries = primaries
        self._whitepoint = None
        self.whitepoint = whitepoint
        self._illuminant = None
        self.illuminant = illuminant
        self._RGB_to_XYZ_matrix = None
        self.RGB_to_XYZ_matrix = RGB_to_XYZ_matrix
        self._XYZ_to_RGB_matrix = None
        self.XYZ_to_RGB_matrix = XYZ_to_RGB_matrix
        self._encoding_cctf = None
        self.encoding_cctf = encoding_cctf
        self._decoding_cctf = None
        self.decoding_cctf = decoding_cctf

    @property
    def name(self):
        """
        Property for **self._name** private attribute.

        Returns
        -------
        unicode
            self._name.
        """

        return self._name

    @name.setter
    def name(self, value):
        """
        Setter for **self._name** private attribute.

        Parameters
        ----------
        value : unicode
            Attribute value.
        """

        if value is not None:
            assert isinstance(value, basestring), (  # noqa
                ('"{0}" attribute: "{1}" is not a '
                 '"basestring" instance!').format('name', value))
        self._name = value

    @property
    def primaries(self):
        """
        Property for **self._primaries** private attribute.

        Returns
        -------
        array_like, (3, 2)
            self._primaries.
        """

        return self._primaries

    @primaries.setter
    def primaries(self, value):
        """
        Setter for **self._primaries** private attribute.

        Parameters
        ----------
        value : array_like, (3, 2)
            Attribute value.
        """

        if value is not None:
            value = np.asarray(value)
        self._primaries = value

    @property
    def whitepoint(self):
        """
        Property for **self._whitepoint** private attribute.

        Returns
        -------
        array_like
            self._whitepoint.
        """

        return self._whitepoint

    @whitepoint.setter
    def whitepoint(self, value):
        """
        Setter for **self._whitepoint** private attribute.

        Parameters
        ----------
        value : array_like
            Attribute value.
        """

        if value is not None:
            assert isinstance(value, (tuple, list, np.ndarray, np.matrix)), (
                ('"{0}" attribute: "{1}" is not a "tuple", "list", "ndarray" '
                 'or "matrix" instance!').format('whitepoint', value))
        self._whitepoint = value

    @property
    def illuminant(self):
        """
        Property for **self._illuminant** private attribute.

        Returns
        -------
        unicode
            self._illuminant.
        """

        return self._illuminant

    @illuminant.setter
    def illuminant(self, value):
        """
        Setter for **self._illuminant** private attribute.

        Parameters
        ----------
        value : unicode
            Attribute value.
        """

        if value is not None:
            assert isinstance(value, basestring), (  # noqa
                ('"{0}" attribute: "{1}" is not a '
                 '"basestring" instance!').format('illuminant', value))
        self._illuminant = value

    @property
    def RGB_to_XYZ_matrix(self):
        """
        Property for **self._to_XYZ** private attribute.

        Returns
        -------
        array_like, (3, 3)
            self._to_XYZ.
        """

        return self._RGB_to_XYZ_matrix

    @RGB_to_XYZ_matrix.setter
    def RGB_to_XYZ_matrix(self, value):
        """
        Setter for **self._to_XYZ** private attribute.

        Parameters
        ----------
        value : array_like
            Attribute value.
        """

        if value is not None:
            value = np.asarray(value)
        self._RGB_to_XYZ_matrix = value

    @property
    def XYZ_to_RGB_matrix(self):
        """
        Property for **self._to_RGB** private attribute.

        Returns
        -------
        array_like, (3, 3)
            self._to_RGB.
        """

        return self._XYZ_to_RGB_matrix

    @XYZ_to_RGB_matrix.setter
    def XYZ_to_RGB_matrix(self, value):
        """
        Setter for **self._to_RGB** private attribute.

        Parameters
        ----------
        value : array_like
            Attribute value.
        """

        if value is not None:
            value = np.asarray(value)
        self._XYZ_to_RGB_matrix = value

    @property
    def encoding_cctf(self):
        """
        Property for **self._encoding_cctf** private attribute.

        Returns
        -------
        object
            self._encoding_cctf.
        """

        return self._encoding_cctf

    @encoding_cctf.setter
    def encoding_cctf(self, value):
        """
        Setter for **self._encoding_cctf** private attribute.

        Parameters
        ----------
        value : object
            Attribute value.
        """

        if value is not None:
            assert hasattr(value, '__call__'), (
                '"{0}" attribute: "{1}" is not callable!'.format(
                    'encoding_cctf', value))
        self._encoding_cctf = value

    @property
    def decoding_cctf(self):
        """
        Property for **self._decoding_cctf** private attribute.

        Returns
        -------
        object
            self._decoding_cctf.
        """

        return self._decoding_cctf

    @decoding_cctf.setter
    def decoding_cctf(self, value):
        """
        Setter for **self._decoding_cctf** private attribute.

        Parameters
        ----------
        value : object
            Attribute value.
        """

        if value is not None:
            assert hasattr(value, '__call__'), (
                '"{0}" attribute: "{1}" is not callable!'.format(
                    'decoding_cctf', value))
        self._decoding_cctf = value


def XYZ_to_RGB(XYZ,
               illuminant_XYZ,
               illuminant_RGB,
               XYZ_to_RGB_matrix,
               chromatic_adaptation_transform='CAT02',
               encoding_cctf=None):
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
        'Fairchild', 'CMCCAT97', 'CMCCAT2000', 'CAT02_BRILL_CAT', 'Bianco',
        'Bianco PC'}**,
        *Chromatic adaptation* transform.
    encoding_cctf : object, optional
        Encoding colour component transfer function (Encoding CCTF) or
        opto-electronic transfer function (OETF / OECF).

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
    -   Output *RGB* colourspace array is in range [0, 1].

    Examples
    --------
    >>> XYZ = np.array([0.07049534, 0.10080000, 0.09558313])
    >>> illuminant_XYZ = np.array([0.34570, 0.35850])
    >>> illuminant_RGB = np.array([0.31270, 0.32900])
    >>> chromatic_adaptation_transform = 'Bradford'
    >>> XYZ_to_RGB_matrix = np.array([
    ...     [3.24062548, -1.53720797, -0.49862860],
    ...     [-0.96893071, 1.87575606, 0.04151752],
    ...     [0.05571012, -0.20402105, 1.05699594]])
    >>> XYZ_to_RGB(
    ...     XYZ,
    ...     illuminant_XYZ,
    ...     illuminant_RGB,
    ...     XYZ_to_RGB_matrix,
    ...     chromatic_adaptation_transform)  # doctest: +ELLIPSIS
    array([ 0.0110015...,  0.1273504...,  0.1163271...])
    """

    M = chromatic_adaptation_matrix_VonKries(
        xyY_to_XYZ(xy_to_xyY(illuminant_XYZ)),
        xyY_to_XYZ(xy_to_xyY(illuminant_RGB)),
        transform=chromatic_adaptation_transform)

    XYZ_a = dot_vector(M, XYZ)

    RGB = dot_vector(XYZ_to_RGB_matrix, XYZ_a)

    if encoding_cctf is not None:
        RGB = encoding_cctf(RGB)

    return RGB


def RGB_to_XYZ(RGB,
               illuminant_RGB,
               illuminant_XYZ,
               RGB_to_XYZ_matrix,
               chromatic_adaptation_transform='CAT02',
               decoding_cctf=None):
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
        'Fairchild', 'CMCCAT97', 'CMCCAT2000', 'CAT02_BRILL_CAT', 'Bianco',
        'Bianco PC'}**,
        *Chromatic adaptation* transform.
    decoding_cctf : object, optional
        Decoding colour component transfer function (Decoding CCTF) or
        electro-optical transfer function (EOTF / EOCF).

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
    -   Output *CIE XYZ* tristimulus values are in range [0, 1].

    Examples
    --------
    >>> RGB = np.array([0.01100154,  0.12735048,  0.11632713])
    >>> illuminant_RGB = np.array([0.31270, 0.32900])
    >>> illuminant_XYZ = np.array([0.34570, 0.35850])
    >>> chromatic_adaptation_transform = 'Bradford'
    >>> RGB_to_XYZ_matrix = np.array([
    ...     [0.41240000, 0.35760000, 0.18050000],
    ...     [0.21260000, 0.71520000, 0.07220000],
    ...     [0.01930000, 0.11920000, 0.95050000]])
    >>> RGB_to_XYZ(
    ...     RGB,
    ...     illuminant_RGB,
    ...     illuminant_XYZ,
    ...     RGB_to_XYZ_matrix,
    ...     chromatic_adaptation_transform)  # doctest: +ELLIPSIS
    array([ 0.0704953...,  0.1008    ,  0.0955831...])
    """

    if decoding_cctf is not None:
        RGB = decoding_cctf(RGB)

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
        'Fairchild', 'CMCCAT97', 'CMCCAT2000', 'CAT02_BRILL_CAT', 'Bianco',
        'Bianco PC'}**,
        *Chromatic adaptation* transform.

    Returns
    -------
    ndarray
        *RGB* colourspace array.

    Notes
    -----
    -   Input / output *RGB* colourspace arrays are in domain / range [0, 1].
    -   Input / output *RGB* colourspace arrays are assumed to be representing
        linear light values.

    Examples
    --------
    >>> from colour import sRGB_COLOURSPACE, PROPHOTO_RGB_COLOURSPACE
    >>> RGB = np.array([0.01103742, 0.12734226, 0.11632971])
    >>> RGB_to_RGB(
    ...     RGB,
    ...     sRGB_COLOURSPACE,
    ...     PROPHOTO_RGB_COLOURSPACE)  # doctest: +ELLIPSIS
    array([ 0.0643538...,  0.1157289...,  0.1158038...])
    """

    cat = chromatic_adaptation_matrix_VonKries(
        xy_to_XYZ(input_colourspace.whitepoint),
        xy_to_XYZ(output_colourspace.whitepoint),
        chromatic_adaptation_transform)

    M = dot_matrix(cat, input_colourspace.RGB_to_XYZ_matrix)
    M = dot_matrix(output_colourspace.XYZ_to_RGB_matrix, M)

    RGB = dot_vector(M, RGB)

    return RGB
