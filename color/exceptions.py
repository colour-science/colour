#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**exceptions.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Color** package exceptions and others exception handling related objects.

**Others:**

"""

from __future__ import unicode_literals

import color.verbose
import foundations.exceptions

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["LOGGER",
           "AbstractError",
           "ProgrammingError",
           "AbstractAlgebraError",
           "LinearRegressionError"]

LOGGER = color.verbose.install_logger()

AbstractError = foundations.exceptions.AbstractError
ProgrammingError = foundations.exceptions.ProgrammingError

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