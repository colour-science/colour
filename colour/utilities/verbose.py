#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**verbose.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package verbose objects.

**Others:**

"""

from __future__ import unicode_literals

import warnings

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["warning"]


def warning(*args, **kwargs):
    """
    Issues a warning.


    :param \*args: Arguments.
    :type \*args: \*
    :param \*\*kwargs: Keywords arguments.
    :type \*\*kwargs: \*\*
    """

    warnings.warn(*args, **kwargs)