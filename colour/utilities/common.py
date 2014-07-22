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

import foundations.verbose

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["is_scipy_installed"]

LOGGER = foundations.verbose.install_logger()

def is_scipy_installed():
    """
    Returns if *scipy* is installed and available.

    :return: Is *scipy* installed.
    :rtype: bool
    """

    try:
        import scipy
        return True
    except ImportError:
        return False
