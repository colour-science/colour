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

import colour.utilities.exceptions

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["is_scipy_installed"]


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
                "{0} | 'scipy' or specific 'scipy' Api features are not available: '{1}'.".format(__name__, error))
        return False
