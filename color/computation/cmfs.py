# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**cmfs.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Color** package *color matching functions* manipulation objects.

**Others:**

"""

from __future__ import unicode_literals

import numpy

import color.utilities.exceptions
import color.utilities.verbose
from color.computation.spectrum import SpectralPowerDistributionTriad

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["LMS_ConeFundamentals",
           "RGB_ColorMatchingFunctions",
           "XYZ_ColorMatchingFunctions",
           "RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs",
           "RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs"]

LOGGER = color.utilities.verbose.install_logger()


class LMS_ConeFundamentals(SpectralPowerDistributionTriad):
    """
    Defines a *LMS* cone fundamentals implementation object.
    """

    def __init__(self, name, triad):
        """
        Initializes the class.

        :param name: Standard observer color matching functions name.
        :type name: unicode
        :param triad: Standard observer color matching functions.
        :type triad: dict
        """

        SpectralPowerDistributionTriad.__init__(self,
                                                name,
                                                triad,
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

        raise color.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "l_bar"))

    @l_bar.deleter
    def l_bar(self):
        """
        Deleter for **self.__r_bar** attribute.
        """

        raise color.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is not deletable!".format(self.__class__.__name__, "l_bar"))

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

        raise color.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "m_bar"))

    @m_bar.deleter
    def m_bar(self):
        """
        Deleter for **self.__g_bar** attribute.
        """

        raise color.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is not deletable!".format(self.__class__.__name__, "m_bar"))

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

        raise color.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "s_bar"))

    @s_bar.deleter
    def s_bar(self):
        """
        Deleter for **self.__b_bar** attribute.
        """

        raise color.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is not deletable!".format(self.__class__.__name__, "s_bar"))


class RGB_ColorMatchingFunctions(SpectralPowerDistributionTriad):
    """
    Defines a *CIE RGB* standard observer color matching functions object implementation.
    """

    def __init__(self, name, triad):
        """
        Initializes the class.

        :param name: Standard observer color matching functions name.
        :type name: unicode
        :param triad: Standard observer color matching functions.
        :type triad: dict
        """

        SpectralPowerDistributionTriad.__init__(self,
                                                name,
                                                triad,
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

        raise color.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "r_bar"))

    @r_bar.deleter
    def r_bar(self):
        """
        Deleter for **self.__r_bar** attribute.
        """

        raise color.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is not deletable!".format(self.__class__.__name__, "r_bar"))

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

        raise color.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "g_bar"))

    @g_bar.deleter
    def g_bar(self):
        """
        Deleter for **self.__g_bar** attribute.
        """

        raise color.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is not deletable!".format(self.__class__.__name__, "g_bar"))

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

        raise color.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "b_bar"))

    @b_bar.deleter
    def b_bar(self):
        """
        Deleter for **self.__b_bar** attribute.
        """

        raise color.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is not deletable!".format(self.__class__.__name__, "b_bar"))


class XYZ_ColorMatchingFunctions(SpectralPowerDistributionTriad):
    """
    Defines an *CIE XYZ* standard observer color matching functions object implementation.
    """

    def __init__(self, name, triad):
        """
        Initializes the class.

        :param name: Standard observer color matching functions name.
        :type name: unicode
        :param triad: Standard observer color matching functions.
        :type triad: dict
        """

        SpectralPowerDistributionTriad.__init__(self,
                                                name,
                                                triad,
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

        raise color.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "x_bar"))

    @x_bar.deleter
    def x_bar(self):
        """
        Deleter for **self.__x_bar** attribute.
        """

        raise color.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is not deletable!".format(self.__class__.__name__, "x_bar"))

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

        raise color.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "y_bar"))

    @y_bar.deleter
    def y_bar(self):
        """
        Deleter for **self.__y_bar** attribute.
        """

        raise color.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is not deletable!".format(self.__class__.__name__, "y_bar"))

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

        raise color.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "z_bar"))

    @z_bar.deleter
    def z_bar(self):
        """
        Deleter for **self.__z_bar** attribute.
        """


def RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(wavelength):
    """
    Converts given *Wright & Guild 1931 2 Degree RGB CMFs* color matching functions into the
    *CIE 1931 2 Degree Standard Observer* color matching functions.

    Reference: Wyszecki & Stiles, Color Science - Concepts and Methods Data and Formulae - Second Edition, Pages 138, 139.

    Usage::

        >>> RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(700)
        [[ 0.01135774]
         [ 0.004102  ]
         [ 0.        ]]

    :param wavelength: Wavelength in nm.
    :type wavelength: float
    :return: *CIE 1931 2 Degree Standard Observer* spectral tristimulus values.
    :rtype: matrix (3x1)
    :note: Data for the *CIE 1931 2 Degree Standard Observer* already exists, this definition is intended for educational purpose.
    """

    cmfs = color.data.cmfs.RGB_CMFS.get("Wright & Guild 1931 2 Degree RGB CMFs")
    r_bar, y_bar, z_bar = cmfs.r_bar.get(wavelength), cmfs.g_bar.get(wavelength), cmfs.b_bar.get(wavelength)
    if None in (r_bar, y_bar, z_bar):
        raise color.utilities.exceptions.ProgrammingError(
            "'{0} nm' wavelength not available in '{1}' color matching functions with '{2}' shape!".format(wavelength,
                                                                                                           cmfs.name,
                                                                                                           cmfs.shape))

    r = r_bar / (r_bar + y_bar + z_bar)
    g = y_bar / (r_bar + y_bar + z_bar)
    b = z_bar / (r_bar + y_bar + z_bar)

    x = (0.49000 * r + 0.31000 * g + 0.20000 * b) / (0.66697 * r + 1.13240 * g + 1.20063 * b)
    y = (0.17697 * r + 0.81240 * g + 0.01063 * b) / (0.66697 * r + 1.13240 * g + 1.20063 * b)
    z = (0.00000 * r + 0.01000 * g + 0.99000 * b) / (0.66697 * r + 1.13240 * g + 1.20063 * b)

    V = color.data.lefs.PHOTOPIC_LEFS.get("CIE 1924 Photopic Standard Observer").clone()
    V.align(*cmfs.shape)
    L = V.get(wavelength)

    x_bar = x / y * L
    y_bar = L
    z_bar = z / y * L

    return numpy.array([x_bar, y_bar, z_bar])


def RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(wavelength):
    """
    Converts given *Stiles & Burch 1959 10 Degree RGB CMFs* color matching
    functions into the *CIE 1964 10 Degree Standard Observer* color matching functions.

    Reference: Wyszecki & Stiles, Color Science - Concepts and Methods Data and Formulae - Second Edition, Page 141.

    Usage::

        >>> RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(700)
        [[ 0.01135774]
         [ 0.004102  ]
         [ 0.        ]]

    :param wavelength: Wavelength in nm.
    :type wavelength: float
    :return: *CIE 1964 10 Degree Standard Observer* spectral tristimulus values.
    :rtype: matrix (3x1)
    :note: Data for the *CIE 1964 10 Degree Standard Observer* already exists, this definition is intended for educational purpose.
    """

    cmfs = color.data.cmfs.RGB_CMFS.get("Stiles & Burch 1959 10 Degree RGB CMFs")
    r_bar, y_bar, z_bar = cmfs.r_bar.get(wavelength), cmfs.g_bar.get(wavelength), cmfs.b_bar.get(wavelength)
    if None in (r_bar, y_bar, z_bar):
        raise color.utilities.exceptions.ProgrammingError(
            "'{0} nm' wavelength not available in '{1}' color matching functions with '{2}' shape!".format(wavelength,
                                                                                                           cmfs.name,
                                                                                                           cmfs.shape))

    x_bar = 0.341080 * r_bar + 0.189145 * y_bar + 0.387529 * z_bar
    y_bar = 0.139058 * r_bar + 0.837460 * y_bar + 0.073316 * z_bar
    z_bar = 0.000000 * r_bar + 0.039553 * y_bar + 2.026200 * z_bar

    return numpy.array([x_bar, y_bar, z_bar])