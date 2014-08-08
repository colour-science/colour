# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**common.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package common utilities objects that don't fall in any specific category.

**Others:**

"""

from __future__ import unicode_literals

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["is_scipy_installed",
           "is_string"]


def is_scipy_installed(raise_exception=False):
    """
    Returns if *scipy* is installed and available.

    :param raise_exception: Raise exception if *scipy* is unavailable.
    :type raise_exception: bool
    :return: Is *scipy* installed.
    :rtype: bool
    """

    try:
        # Importing *scipy* Api features used in *Colour*.
        import scipy.interpolate
        import scipy.ndimage
        import scipy.spatial

        return True
    except ImportError as error:
        if raise_exception:
            raise colour.utilities.exceptions.UnavailableApiFeatureError(
                "'scipy' or specific 'scipy' Api features are not available: '{1}'.".format(
                    error))
        return False


def is_string(data):
    """
    Returns if given data is a *string_like* variable

    :param data: Data to test.
    :type data: object
    :return: Is *string_like* variable.
    :rtype: bool
    """

    return True if isinstance(data, basestring) else False