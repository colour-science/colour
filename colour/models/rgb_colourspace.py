# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**rgb_colourspace.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package *RGB* colourspaces base object.

**Others:**

"""

from __future__ import unicode_literals

import numpy as np

from colour.algebra import to_ndarray

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["RGB_Colourspace"]


class RGB_Colourspace(object):
    """
    Defines a *RGB* colourspace object.
    """

    def __init__(self,
                 name,
                 primaries,
                 whitepoint,
                 to_XYZ=None,
                 from_XYZ=None,
                 transfer_function=None,
                 inverse_transfer_function=None):
        """
        Initialises the class.

        :param name: *RGB* Colourspace name.
        :type name: str or unicode
        :param primaries: *RGB* Colourspace primaries.
        :type primaries: array_like
        :param whitepoint: *RGB* Colourspace whitepoint.
        :type whitepoint: array_like
        :param to_XYZ: Transformation matrix from colourspace to *CIE XYZ* \
        colourspace.
        :type to_XYZ: array_like
        :param from_XYZ: Transformation matrix from *CIE XYZ* colourspace to \
        colourspace.
        :type from_XYZ: array_like
        :param transfer_function: *RGB* Colourspace transfer function from \
        linear to colourspace.
        :type transfer_function: object
        :param inverse_transfer_function: *RGB* Colourspace inverse transfer \
        function from colourspace to linear.
        :type inverse_transfer_function: object
        """

        # --- Setting class attributes. ---
        self.__name = None
        self.name = name
        self.__primaries = None
        self.primaries = primaries
        self.__whitepoint = None
        self.whitepoint = whitepoint
        self.__to_XYZ = None
        self.to_XYZ = to_XYZ
        self.__from_XYZ = None
        self.from_XYZ = from_XYZ
        self.__transfer_function = None
        self.transfer_function = transfer_function
        self.__inverse_transfer_function = None
        self.inverse_transfer_function = inverse_transfer_function

    @property
    def name(self):
        """
        Property for **self.__name** attribute.

        :return: self.__name.
        :rtype: str or unicode
        """

        return self.__name

    @name.setter
    def name(self, value):
        """
        Setter for **self.__name** attribute.

        :param value: Attribute value.
        :type value: str or unicode
        """

        if value is not None:
            assert type(value) in (str, unicode), \
                "'{0}' attribute: '{1}' type is not in 'str' or 'unicode'!".format(
                    "name", value)
        self.__name = value

    @property
    def primaries(self):
        """
        Property for **self.__primaries** attribute.

        :return: self.__primaries.
        :rtype: array_like
        """

        return self.__primaries

    @primaries.setter
    def primaries(self, value):
        """
        Setter for **self.__primaries** attribute.

        :param value: Attribute value.
        :type value: array_like
        """

        if value is not None:
            value = to_ndarray(value)
        self.__primaries = value

    @property
    def whitepoint(self):
        """
        Property for **self.__whitepoint** attribute.

        :return: self.__whitepoint.
        :rtype: array_like
        """

        return self.__whitepoint

    @whitepoint.setter
    def whitepoint(self, value):
        """
        Setter for **self.__whitepoint** attribute.

        :param value: Attribute value.
        :type value: array_like
        """

        if value is not None:
            assert type(value) in (tuple, list, np.ndarray, np.matrix), \
                "'{0}' attribute: '{1}' type is not 'tuple', 'list', 'ndarray' or 'matrix'!".format(
                    "whitepoint", value)
        self.__whitepoint = value

    @property
    def to_XYZ(self):
        """
        Property for **self.__to_XYZ** attribute.

        :return: self.__to_XYZ.
        :rtype: array_like
        """

        return self.__to_XYZ

    @to_XYZ.setter
    def to_XYZ(self, value):
        """
        Setter for **self.__to_XYZ** attribute.

        :param value: Attribute value.
        :type value: array_like
        """

        if value is not None:
            value = to_ndarray(value)
        self.__to_XYZ = value

    @property
    def from_XYZ(self):
        """
        Property for **self.__from_XYZ** attribute.

        :return: self.__from_XYZ.
        :rtype: array_like
        """

        return self.__from_XYZ

    @from_XYZ.setter
    def from_XYZ(self, value):
        """
        Setter for **self.__from_XYZ** attribute.

        :param value: Attribute value.
        :type value: array_like
        """

        if value is not None:
            value = to_ndarray(value)
        self.__from_XYZ = value

    @property
    def transfer_function(self):
        """
        Property for **self.__transfer_function** attribute.

        :return: self.__transfer_function.
        :rtype: object
        """

        return self.__transfer_function

    @transfer_function.setter
    def transfer_function(self, value):
        """
        Setter for **self.__transfer_function** attribute.

        :param value: Attribute value.
        :type value: object
        """

        if value is not None:
            assert hasattr(value, "__call__"), \
                "'{0}' attribute: '{1}' is not callable!".format(
                    "transfer_function", value)
        self.__transfer_function = value

    @property
    def inverse_transfer_function(self):
        """
        Property for **self.__inverse_transfer_function** attribute.

        :return: self.__inverse_transfer_function.
        :rtype: object
        """

        return self.__inverse_transfer_function

    @inverse_transfer_function.setter
    def inverse_transfer_function(self, value):
        """
        Setter for **self.__inverse_transfer_function** attribute.

        :param value: Attribute value.
        :type value: object
        """

        if value is not None:
            assert hasattr(value, "__call__"), \
                "'{0}' attribute: '{1}' is not callable!".format(
                    "inverse_transfer_function", value)
        self.__inverse_transfer_function = value