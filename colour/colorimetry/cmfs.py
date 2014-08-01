# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**cmfs.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package *colour matching functions* objects.

**Others:**

"""

from __future__ import unicode_literals

from colour.colorimetry import TriSpectralPowerDistribution

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["LMS_ConeFundamentals",
           "RGB_ColourMatchingFunctions",
           "XYZ_ColourMatchingFunctions"]


class LMS_ConeFundamentals(TriSpectralPowerDistribution):
    """
    Defines a *LMS* cone fundamentals implementation object.
    """

    def __init__(self, name, data):
        """
        Initialises the class.

        :param name: Standard observer colour matching functions name.
        :type name: unicode
        :param data: Standard observer colour matching functions.
        :type data: dict
        """

        TriSpectralPowerDistribution.__init__(self,
                                              name,
                                              data,
                                              mapping={"x": "l_bar",
                                                       "y": "m_bar",
                                                       "z": "s_bar"},
                                              labels={"x": "l\u0304",
                                                      "y": "m\u0304",
                                                      "z": "s\u0304"})

    @property
    def l_bar(self):
        """
        Property for **self.__r_bar** attribute.

        :return: self.__r_bar.
        :rtype: unicode
        """

        return self.x

    @l_bar.setter
    def l_bar(self, value):
        """
        Setter for **self.__r_bar** attribute.

        :param value: Attribute value.
        :type value: unicode
        """

        raise AttributeError("{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "l_bar"))

    @property
    def m_bar(self):
        """
        Property for **self.__g_bar** attribute.

        :return: self.__g_bar.
        :rtype: unicode
        """

        return self.y

    @m_bar.setter
    def m_bar(self, value):
        """
        Setter for **self.__g_bar** attribute.

        :param value: Attribute value.
        :type value: unicode
        """

        raise AttributeError("{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "m_bar"))

    @property
    def s_bar(self):
        """
        Property for **self.__b_bar** attribute.

        :return: self.__b_bar.
        :rtype: unicode
        """

        return self.z

    @s_bar.setter
    def s_bar(self, value):
        """
        Setter for **self.__b_bar** attribute.

        :param value: Attribute value.
        :type value: unicode
        """

        raise AttributeError("{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "s_bar"))


class RGB_ColourMatchingFunctions(TriSpectralPowerDistribution):
    """
    Defines a *CIE RGB* standard observer colour matching functions object implementation.
    """

    def __init__(self, name, data):
        """
        Initialises the class.

        :param name: Standard observer colour matching functions name.
        :type name: unicode
        :param data: Standard observer colour matching functions.
        :type data: dict
        """

        TriSpectralPowerDistribution.__init__(self,
                                              name,
                                              data,
                                              mapping={"x": "r_bar",
                                                       "y": "g_bar",
                                                       "z": "b_bar"},
                                              labels={"x": "r\u0304",
                                                      "y": "g\u0304",
                                                      "z": "b\u0304"})

    @property
    def r_bar(self):
        """
        Property for **self.__r_bar** attribute.

        :return: self.__r_bar.
        :rtype: unicode
        """

        return self.x

    @r_bar.setter
    def r_bar(self, value):
        """
        Setter for **self.__r_bar** attribute.

        :param value: Attribute value.
        :type value: unicode
        """

        raise AttributeError("{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "r_bar"))

    @property
    def g_bar(self):
        """
        Property for **self.__g_bar** attribute.

        :return: self.__g_bar.
        :rtype: unicode
        """

        return self.y

    @g_bar.setter
    def g_bar(self, value):
        """
        Setter for **self.__g_bar** attribute.

        :param value: Attribute value.
        :type value: unicode
        """

        raise AttributeError("{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "g_bar"))

    @property
    def b_bar(self):
        """
        Property for **self.__b_bar** attribute.

        :return: self.__b_bar.
        :rtype: unicode
        """

        return self.z

    @b_bar.setter
    def b_bar(self, value):
        """
        Setter for **self.__b_bar** attribute.

        :param value: Attribute value.
        :type value: unicode
        """

        raise AttributeError("{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "b_bar"))


class XYZ_ColourMatchingFunctions(TriSpectralPowerDistribution):
    """
    Defines an *CIE XYZ* standard observer colour matching functions object implementation.
    """

    def __init__(self, name, data):
        """
        Initialises the class.

        :param name: Standard observer colour matching functions name.
        :type name: unicode
        :param data: Standard observer colour matching functions.
        :type data: dict
        """

        TriSpectralPowerDistribution.__init__(self,
                                              name,
                                              data,
                                              mapping={"x": "x_bar",
                                                       "y": "y_bar",
                                                       "z": "z_bar"},
                                              labels={"x": "x\u0304",
                                                      "y": "y\u0304",
                                                      "z": "z\u0304"})

    @property
    def x_bar(self):
        """
        Property for **self.__x_bar** attribute.

        :return: self.__x_bar.
        :rtype: unicode
        """

        return self.x

    @x_bar.setter
    def x_bar(self, value):
        """
        Setter for **self.__x_bar** attribute.

        :param value: Attribute value.
        :type value: unicode
        """

        raise AttributeError("{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "x_bar"))

    @property
    def y_bar(self):
        """
        Property for **self.__y_bar** attribute.

        :return: self.__y_bar.
        :rtype: unicode
        """

        return self.y

    @y_bar.setter
    def y_bar(self, value):
        """
        Setter for **self.__y_bar** attribute.

        :param value: Attribute value.
        :type value: unicode
        """

        raise AttributeError("{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "y_bar"))

    @property
    def z_bar(self):
        """
        Property for **self.__z_bar** attribute.

        :return: self.__z_bar.
        :rtype: unicode
        """

        return self.z

    @z_bar.setter
    def z_bar(self, value):
        """
        Setter for **self.__z_bar** attribute.

        :param value: Attribute value.
        :type value: unicode
        """

        raise AttributeError("{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "z_bar"))