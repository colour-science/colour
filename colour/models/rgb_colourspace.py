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
    to_XYZ : array_like
        Transformation matrix from colourspace to *CIE XYZ* colourspace.
    to_RGB : array_like
        Transformation matrix from *CIE XYZ* colourspace to colourspace.
    transfer_function : object
        *RGB* Colourspace transfer function from linear to colourspace.
    inverse_transfer_function : object
        *RGB* Colourspace inverse transfer function from colourspace to linear.
    """

    def __init__(self,
                 name,
                 primaries,
                 whitepoint,
                 to_XYZ=None,
                 to_RGB=None,
                 transfer_function=None,
                 inverse_transfer_function=None):
        self.__name = None
        self.name = name
        self.__primaries = None
        self.primaries = primaries
        self.__whitepoint = None
        self.whitepoint = whitepoint
        self.__to_XYZ = None
        self.to_XYZ = to_XYZ
        self.__to_RGB = None
        self.to_RGB = to_RGB
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
    def to_XYZ(self):
        """
        Property for **self.__to_XYZ** private attribute.

        Returns
        -------
        array_like, (3, 3)
            self.__to_XYZ.
        """

        return self.__to_XYZ

    @to_XYZ.setter
    def to_XYZ(self, value):
        """
        Setter for **self.__to_XYZ** private attribute.

        Parameters
        ----------
        value : array_like
            Attribute value.
        """

        if value is not None:
            value = to_ndarray(value)
        self.__to_XYZ = value

    @property
    def to_RGB(self):
        """
        Property for **self.__to_RGB** private attribute.

        Returns
        -------
        array_like, (3, 3)
            self.__to_RGB.
        """

        return self.__to_RGB

    @to_RGB.setter
    def to_RGB(self, value):
        """
        Setter for **self.__to_RGB** private attribute.

        Parameters
        ----------
        value : array_like
            Attribute value.
        """

        if value is not None:
            value = to_ndarray(value)
        self.__to_RGB = value

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
