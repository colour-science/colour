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
           "AbstractUserError",
           "ProgrammingError",
           "AbstractAlgebraError",
           "DimensionsError",
           "DomainError",
           "AbstractInterpolationError",
           "InterpolationError",
           "InterpolationRangeError",
           "AbstractExtrapolationError",
           "ExtrapolationError",
           "LinearRegressionError",
           "AbstractColourMatchingFunctionsError",
           "ColourMatchingFunctionsError",
           "AbstractLuminousEfficiencyFunctionError",
           "LuminousEfficiencyFunctionError",
           "AbstractCorrelatedColourTemperatureError",
           "CorrelatedColourTemperatureError",
           "AbstractMunsellColourError",
           "MunsellColourError"]


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


class AbstractApiError(AbstractError):
    """
    Defines the abstract base class for api exception.
    """

    pass


class UnavailableApiFeatureError(AbstractError):
    """
    Defines unavailable api feature exception.
    """

    pass


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


class DimensionsError(AbstractAlgebraError):
    """
    Defines dimensions related exception.
    """

    pass


class DomainError(AbstractAlgebraError):
    """
    Defines domain related exception.
    """

    pass


class AbstractInterpolationError(AbstractAlgebraError):
    """
    Defines the abstract base class for interpolation exception.
    """

    pass


class InterpolationError(AbstractInterpolationError):
    """
    Defines interpolation exception.
    """

    pass


class InterpolationRangeError(AbstractInterpolationError):
    """
    Defines interpolation range exception.
    """

    pass


class AbstractExtrapolationError(AbstractAlgebraError):
    """
    Defines the abstract base class for extrapolation exception.
    """

    pass


class ExtrapolationError(AbstractExtrapolationError):
    """
    Defines interpolation extrapolation.
    """

    pass


class LinearRegressionError(AbstractAlgebraError):
    """
    Defines linear regression exception.
    """

    pass


class AbstractColourMatchingFunctionsError(AbstractError):
    """
    Defines the abstract base class for colour matching functions exception.
    """

    pass


class ColourMatchingFunctionsError(AbstractColourMatchingFunctionsError):
    """
    Defines colour matching functions exception.
    """

    pass


class AbstractLuminousEfficiencyFunctionError(AbstractError):
    """
    Defines the abstract base class for luminous efficiency function exception.
    """

    pass


class LuminousEfficiencyFunctionError(AbstractLuminousEfficiencyFunctionError):
    """
    Defines luminous efficiency function exception.
    """

    pass


class AbstractCorrelatedColourTemperatureError(AbstractError):
    """
    Defines the abstract base class for correlated colour temperature exception.
    """

    pass


class CorrelatedColourTemperatureError(AbstractCorrelatedColourTemperatureError):
    """
    Defines correlated colour temperature exception.
    """

    pass


class AbstractMunsellColourError(AbstractError):
    """
    Defines the abstract base class for munsell colour exception.
    """

    pass


class MunsellColourError(AbstractMunsellColourError):
    """
    Defines munsell colour exception.
    """

    pass