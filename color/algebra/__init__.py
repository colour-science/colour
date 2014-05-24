#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**__init__.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Color** package *algebra* initialisation.

**Others:**

"""

from __future__ import unicode_literals

import color.verbose

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["LOGGER"]

LOGGER = color.verbose.install_logger()

from color.algebra.matrix import *
from color.algebra.regression import *