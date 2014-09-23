#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RGB Colourspace
===============

Defines the :class:`RGB_Colourspace` class for the *RGB* colourspaces dataset
from :mod:`colour.models.dataset.aces_rgb`, etc...

See Also
--------
`RGB Colourspaces IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/master/notebooks/models/rgb.ipynb>`_  # noqa
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.algebra import to_ndarray

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['RGB_Colourspace']


class RGB_Colourspace(object):
    """
    Implements support for the *RGB* colourspaces dataset from
    :mod:`colour.models.dataset.aces_rgb`, etc....

    Parameters
    ----------
    name : str or unicode
        *RGB* Colourspace name.
    primaries : array_like
        *RGB* Colourspace primaries.
    whitepoint : array_like
        *RGB* Colourspace whitepoint.
    RGB_to_XYZ_matrix : array_like
        Transformation matrix from colourspace to *CIE XYZ* colourspace.
    XYZ_to_RGB_matrix : array_like
        Transformation matrix from *CIE XYZ* colourspace to colourspace.
    transfer_function : object
        *RGB* Colourspace opto-electronic transfer function from linear to
        colourspace.
    inverse_transfer_function : object
        *RGB* Colourspace inverse opto-electronic transfer function from
        colourspace to linear.
    """

    def __init__(self,
                 name,
                 primaries,
                 whitepoint,
                 RGB_to_XYZ_matrix=None,
                 XYZ_to_RGB_matrix=None,
                 transfer_function=None,
                 inverse_transfer_function=None):
        self.__name = None
        self.name = name
        self.__primaries = None
        self.primaries = primaries
        self.__whitepoint = None
        self.whitepoint = whitepoint
        self.__RGB_to_XYZ_matrix = None
        self.RGB_to_XYZ_matrix = RGB_to_XYZ_matrix
        self.__XYZ_to_RGB_matrix = None
        self.XYZ_to_RGB_matrix = XYZ_to_RGB_matrix
        self.__transfer_function = None
        self.transfer_function = transfer_function
        self.__inverse_transfer_function = None
        self.inverse_transfer_function = inverse_transfer_function

    @property
    def name(self):
        """
        Property for **self.__name** private attribute.

        Returns
        -------
        str or unicode
            self.__name.
        """

        return self.__name

    @name.setter
    def name(self, value):
        """
        Setter for **self.__name** private attribute.

        Parameters
        ----------
        value : str or unicode
            Attribute value.
        """

        if value is not None:
            assert type(value) in (str, unicode), (
                ('"{0}" attribute: "{1}" type is not '
                 '"str" or "unicode"!').format('name', value))
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
            value = to_ndarray(value)
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
            assert type(value) in (tuple, list, np.ndarray, np.matrix), (
                ('"{0}" attribute: "{1}" type is not "tuple", "list", '
                 '"ndarray" or "matrix"!').format('whitepoint', value))
        self.__whitepoint = value

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
            value = to_ndarray(value)
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
            value = to_ndarray(value)
        self.__XYZ_to_RGB_matrix = value

    @property
    def transfer_function(self):
        """
        Property for **self.__transfer_function** private attribute.

        Returns
        -------
        object
            self.__transfer_function.
        """

        return self.__transfer_function

    @transfer_function.setter
    def transfer_function(self, value):
        """
        Setter for **self.__transfer_function** private attribute.

        Parameters
        ----------
        value : object
            Attribute value.
        """

        if value is not None:
            assert hasattr(value, '__call__'), (
                '"{0}" attribute: "{1}" is not callable!'.format(
                    'transfer_function', value))
        self.__transfer_function = value

    @property
    def inverse_transfer_function(self):
        """
        Property for **self.__inverse_transfer_function** private attribute.

        Returns
        -------
        object
            self.__inverse_transfer_function.
        """

        return self.__inverse_transfer_function

    @inverse_transfer_function.setter
    def inverse_transfer_function(self, value):
        """
        Setter for **self.__inverse_transfer_function** private attribute.

        Parameters
        ----------
        value : object
            Attribute value.
        """

        if value is not None:
            assert hasattr(value, '__call__'), (
                '"{0}" attribute: "{1}" is not callable!'.format(
                    'inverse_transfer_function', value))
        self.__inverse_transfer_function = value
