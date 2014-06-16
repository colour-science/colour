#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**decorators.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Color** package generic decorators objects.

**Others:**

"""

from __future__ import unicode_literals

import color.utilities.verbose
import foundations.decorators

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2008 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["memoize"]

LOGGER = color.utilities.verbose.install_logger()

memoize = foundations.decorators.memoize