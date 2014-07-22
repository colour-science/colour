# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**cmfs.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package *colour matching functions* manipulation objects.

**Others:**

"""

from __future__ import unicode_literals

import numpy

import colour.utilities.exceptions
import colour.utilities.verbose
from colour.computation.spectrum import TriSpectralPowerDistribution

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["LMS_ConeFundamentals",
           "RGB_ColourMatchingFunctions",
           "XYZ_ColourMatchingFunctions",
           "RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs",
           "RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs",
           "RGB_10_degree_cmfs_to_LMS_10_degree_cmfs",
           "LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs"]

LOGGER = colour.utilities.verbose.install_logger()


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

        raise colour.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "l_bar"))

    @l_bar.deleter
    def l_bar(self):
        """
        Deleter for **self.__r_bar** attribute.
        """

        raise colour.utilities.exceptions.ProgrammingError(
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

        raise colour.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "m_bar"))

    @m_bar.deleter
    def m_bar(self):
        """
        Deleter for **self.__g_bar** attribute.
        """

        raise colour.utilities.exceptions.ProgrammingError(
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

        raise colour.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "s_bar"))

    @s_bar.deleter
    def s_bar(self):
        """
        Deleter for **self.__b_bar** attribute.
        """

        raise colour.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is not deletable!".format(self.__class__.__name__, "s_bar"))


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

        raise colour.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "r_bar"))

    @r_bar.deleter
    def r_bar(self):
        """
        Deleter for **self.__r_bar** attribute.
        """

        raise colour.utilities.exceptions.ProgrammingError(
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

        raise colour.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "g_bar"))

    @g_bar.deleter
    def g_bar(self):
        """
        Deleter for **self.__g_bar** attribute.
        """

        raise colour.utilities.exceptions.ProgrammingError(
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

        raise colour.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "b_bar"))

    @b_bar.deleter
    def b_bar(self):
        """
        Deleter for **self.__b_bar** attribute.
        """

        raise colour.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is not deletable!".format(self.__class__.__name__, "b_bar"))


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

        raise colour.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "x_bar"))

    @x_bar.deleter
    def x_bar(self):
        """
        Deleter for **self.__x_bar** attribute.
        """

        raise colour.utilities.exceptions.ProgrammingError(
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

        raise colour.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "y_bar"))

    @y_bar.deleter
    def y_bar(self):
        """
        Deleter for **self.__y_bar** attribute.
        """

        raise colour.utilities.exceptions.ProgrammingError(
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

        raise colour.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "z_bar"))

    @z_bar.deleter
    def z_bar(self):
        """
        Deleter for **self.__z_bar** attribute.
        """


def RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(wavelength):
    """
    Converts *Wright & Guild 1931 2 Degree RGB CMFs* colour matching functions into the
    *CIE 1931 2 Degree Standard Observer* colour matching functions.

    References:

    -  **Wyszecki & Stiles**, *Color Science - Concepts and Methods Data and Formulae - Second Edition*, Pages 138, 139.

    Usage::

        >>> RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(700)
        [ 0.01135774  0.004102    0.        ]

    :param wavelength: Wavelength in nm.
    :type wavelength: float
    :return: *CIE 1931 2 Degree Standard Observer* spectral tristimulus values.
    :rtype: ndarray (3x1)
    :note: Data for the *CIE 1931 2 Degree Standard Observer* already exists, this definition is intended for educational purpose.
    """

    # Accessing directly from *colour* namespace to avoid circular imports issues.
    cmfs = colour.RGB_CMFS.get("Wright & Guild 1931 2 Degree RGB CMFs")
    r_bar, g_bar, b_bar = cmfs.r_bar.get(wavelength), cmfs.g_bar.get(wavelength), cmfs.b_bar.get(wavelength)
    if None in (r_bar, g_bar, b_bar):
        raise colour.utilities.exceptions.ProgrammingError(
            "'{0} nm' wavelength not available in '{1}' colour matching functions with '{2}' shape!".format(wavelength,
                                                                                                           cmfs.name,
                                                                                                           cmfs.shape))

    r = r_bar / (r_bar + g_bar + b_bar)
    g = g_bar / (r_bar + g_bar + b_bar)
    b = b_bar / (r_bar + g_bar + b_bar)

    x = (0.49000 * r + 0.31000 * g + 0.20000 * b) / (0.66697 * r + 1.13240 * g + 1.20063 * b)
    y = (0.17697 * r + 0.81240 * g + 0.01063 * b) / (0.66697 * r + 1.13240 * g + 1.20063 * b)
    z = (0.00000 * r + 0.01000 * g + 0.99000 * b) / (0.66697 * r + 1.13240 * g + 1.20063 * b)

    V = colour.PHOTOPIC_LEFS.get("CIE 1924 Photopic Standard Observer").clone()
    V.align(*cmfs.shape)
    L = V.get(wavelength)

    x_bar = x / y * L
    y_bar = L
    z_bar = z / y * L

    return numpy.array([x_bar, y_bar, z_bar])


def RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(wavelength):
    """
    Converts *Stiles & Burch 1959 10 Degree RGB CMFs* colour matching
    functions into the *CIE 1964 10 Degree Standard Observer* colour matching functions.

    References:

    -  **Wyszecki & Stiles**, *Color Science - Concepts and Methods Data and Formulae - Second Edition*, Page 141.

    Usage::

        >>> RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(700)
        [  9.64321500e-03   3.75263179e-03  -4.10788300e-06]

    :param wavelength: Wavelength in nm.
    :type wavelength: float
    :return: *CIE 1964 10 Degree Standard Observer* spectral tristimulus values.
    :rtype: ndarray (3x1)
    :note: Data for the *CIE 1964 10 Degree Standard Observer* already exists, this definition is intended for educational purpose.
    """

    # Accessing directly from *colour* namespace to avoid circular imports issues.
    cmfs = colour.RGB_CMFS.get("Stiles & Burch 1959 10 Degree RGB CMFs")
    r_bar, g_bar, b_bar = cmfs.r_bar.get(wavelength), cmfs.g_bar.get(wavelength), cmfs.b_bar.get(wavelength)
    if None in (r_bar, g_bar, b_bar):
        raise colour.utilities.exceptions.ProgrammingError(
            "'{0} nm' wavelength not available in '{1}' colour matching functions with '{2}' shape!".format(wavelength,
                                                                                                           cmfs.name,
                                                                                                           cmfs.shape))

    x_bar = 0.341080 * r_bar + 0.189145 * g_bar + 0.387529 * b_bar
    y_bar = 0.139058 * r_bar + 0.837460 * g_bar + 0.073316 * b_bar
    z_bar = 0.000000 * r_bar + 0.039553 * g_bar + 2.026200 * b_bar

    return numpy.array([x_bar, y_bar, z_bar])


def RGB_10_degree_cmfs_to_LMS_10_degree_cmfs(wavelength):
    """
    Converts *Stiles & Burch 1959 10 Degree RGB CMFs* colour matching
    functions into the *Stockman & Sharpe 10 Degree Cone Fundamentals* spectral sensitivity functions.

    References:

    -  `CIE 170-1:2006 Fundamental Chromaticity Diagram with Physiological Axes - Part 1 <http://div1.cie.co.at/?i_ca_id=551&pubid=48>`_

    Usage::

        >>> RGB_10_degree_cmfs_to_LMS_10_degree_cmfs(700)
        [ 0.00528607  0.00032528  0.        ]

    :param wavelength: Wavelength in nm.
    :type wavelength: float
    :return: *Stockman & Sharpe 10 Degree Cone Fundamentals* spectral tristimulus values.
    :rtype: ndarray (3x1)
    :note: Data for the *Stockman & Sharpe 10 Degree Cone Fundamentals* already exists, this definition is intended for educational purpose.
    """

    # Accessing directly from *colour* namespace to avoid circular imports issues.
    cmfs = colour.RGB_CMFS.get("Stiles & Burch 1959 10 Degree RGB CMFs")
    r_bar, g_bar, z_bar = cmfs.r_bar.get(wavelength), cmfs.g_bar.get(wavelength), cmfs.b_bar.get(wavelength)
    if None in (r_bar, g_bar, z_bar):
        raise colour.utilities.exceptions.ProgrammingError(
            "'{0} nm' wavelength not available in '{1}' colour matching functions with '{2}' shape!".format(wavelength,
                                                                                                           cmfs.name,
                                                                                                           cmfs.shape))

    l_bar = 0.192325269 * r_bar + 0.749548882 * g_bar + 0.0675726702 * z_bar
    g_bar = 0.0192290085 * r_bar + 0.940908496 * g_bar + 0.113830196 * z_bar
    z_bar = 0.0105107859 * g_bar + 0.991427669 * z_bar if wavelength <= 505 else 0.

    return numpy.array([l_bar, g_bar, z_bar])


def LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs(wavelength):
    """
    Converts *Stockman & Sharpe 2 Degree Cone Fundamentals* colour matching
    functions into the *CIE 2012 2 Degree Standard Observer* colour matching functions.

    References:

    -  http://www.cvrl.org/database/text/cienewxyz/cie2012xyz2.htm

    Usage::

        >>> LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs(700)
        [ 0.01096778  0.00419594  0.        ]

    :param wavelength: Wavelength in nm.
    :type wavelength: float
    :return: *CIE 2012 2 Degree Standard Observer* spectral tristimulus values.
    :rtype: ndarray (3x1)
    :note: Data for the *CIE 2012 2 Degree Standard Observer* already exists, this definition is intended for educational purpose.
    """

    # Accessing directly from *colour* namespace to avoid circular imports issues.
    cmfs = colour.LMS_CMFS.get("Stockman & Sharpe 2 Degree Cone Fundamentals")
    l_bar, m_bar, s_bar = cmfs.l_bar.get(wavelength), cmfs.m_bar.get(wavelength), cmfs.s_bar.get(wavelength)
    if None in (l_bar, m_bar, s_bar):
        raise colour.utilities.exceptions.ProgrammingError(
            "'{0} nm' wavelength not available in '{1}' colour matching functions with '{2}' shape!".format(wavelength,
                                                                                                           cmfs.name,
                                                                                                           cmfs.shape))

    x_bar = 1.94735469 * l_bar - 1.41445123 * m_bar + 0.36476327 * s_bar
    y_bar = 0.68990272 * l_bar + 0.34832189 * m_bar
    z_bar = 1.93485343 * s_bar

    return numpy.array([x_bar, y_bar, z_bar])


def LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs(wavelength):
    """
    Converts *Stockman & Sharpe 10 Degree Cone Fundamentals* colour matching
    functions into the *CIE 2012 10 Degree Standard Observer* colour matching functions.

    References:

    -  http://www.cvrl.org/database/text/cienewxyz/cie2012xyz10.htm

    Usage::

        >>> LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs(700)
        [ 0.00981623  0.00377614  0.        ]

    :param wavelength: Wavelength in nm.
    :type wavelength: float
    :return: *CIE 2012 10 Degree Standard Observer* spectral tristimulus values.
    :rtype: ndarray (3x1)
    :note: Data for the *CIE 2012 10 Degree Standard Observer* already exists, this definition is intended for educational purpose.
    """

    # Accessing directly from *colour* namespace to avoid circular imports issues.
    cmfs = colour.LMS_CMFS.get("Stockman & Sharpe 10 Degree Cone Fundamentals")
    l_bar, m_bar, s_bar = cmfs.l_bar.get(wavelength), cmfs.m_bar.get(wavelength), cmfs.s_bar.get(wavelength)
    if None in (l_bar, m_bar, s_bar):
        raise colour.utilities.exceptions.ProgrammingError(
            "'{0} nm' wavelength not available in '{1}' colour matching functions with '{2}' shape!".format(wavelength,
                                                                                                           cmfs.name,
                                                                                                           cmfs.shape))

    x_bar = 1.93986443 * l_bar - 1.34664359 * m_bar + 0.43044935 * s_bar
    y_bar = 0.69283932 * l_bar + 0.34967567 * m_bar
    z_bar = 2.14687945 * s_bar

    return numpy.array([x_bar, y_bar, z_bar])