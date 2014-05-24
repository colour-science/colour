#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**__init__.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Color** package initialisation.

**Others:**

"""

from __future__ import unicode_literals

import foundations.globals.constants
from globals.constants import Constants

foundations.globals.constants.Constants.__dict__.update(Constants.__dict__)

from color.verbose import *

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["LOGGER"]

LOGGER = install_logger()

get_logging_console_handler()
set_verbosity_level(Constants.verbosity_level)

from blackbody import *
from chromatic_adaptation import *
from color_checkers import *
from colorspaces import *
from derivation import *
from difference import *
from illuminants import *
from lightness import *
from luminosity import *
from spectral import *
from temperature import *
from transformations import *
