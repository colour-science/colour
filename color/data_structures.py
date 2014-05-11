#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**data_structures.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Color** package data structures objects.

**Others:**

"""

from __future__ import unicode_literals

import color.verbose
import foundations.data_structures

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["LOGGER",
           "Structure"]

LOGGER = color.verbose.install_logger()

Structure = foundations.data_structures.Structure
