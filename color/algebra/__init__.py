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

from .interpolation import *
from .matrix import *
from .regression import *
from . import interpolation
from . import matrix
from . import regression

__all__.extend(interpolation.__all__)
__all__.extend(matrix.__all__)
__all__.extend(regression.__all__)
__all__ = map(lambda x: x.encode("ascii"), __all__)