# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**exceptions.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package exceptions and others exception handling related objects.

**Others:**

"""

from __future__ import unicode_literals

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["AbstractError",
           "ProgrammingError",
           "AbstractAlgebraError",
           "LinearRegressionError"]


class AbstractError(Exception):
    """
    Defines the abstract base class for all **Foundations** package exceptions.

    References:

    -  https://github.com/KelSolaar/Foundations/blob/develop/foundations/exceptions.py

    """

    def __init__(self, value):
        """
        Initializes the class.

        :param value: Error value or message.
        :type value: unicode
        """

        # --- Setting class attributes. ---
        self.__value = value

    @property
    def value(self):
        """
        Property for **self.__value** attribute.

        :return: self.__value.
        :rtype: object
        """

        return self.__value

    @value.setter
    def value(self, value):
        """
        Setter for **self.__value** attribute.

        :param value: Attribute value.
        :type value: object
        """

        self.__value = value

    @value.deleter
    def value(self):
        """
        Deleter for **self.__value** attribute.
        """

        raise Exception("{0} | '{1}' attribute is not deletable!".format(self.__class__.__name__, "value"))

    def __str__(self):
        """
        Returns the exception representation.

        :return: Exception representation.
        :rtype: unicode
        """

        return str(self.__value)


class AbstractUserError(AbstractError):
    """
    Defines the abstract base class for user related exception.
    """

    pass


class ProgrammingError(AbstractUserError):
    """
    Defines programming exception.
    """

    pass


class AbstractAlgebraError(AbstractError):
    """
    Defines the abstract base class for algebra exception.
    """

    pass


class LinearRegressionError(AbstractAlgebraError):
    """
    Defines linear regression exception.
    """

    pass