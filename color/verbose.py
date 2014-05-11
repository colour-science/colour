#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**verbose.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Color** package verbose and logging objects.

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

__all__ = ["install_logger",
           "get_logging_console_handler",
           "set_verbosity_level"]

install_logger = foundations.verbose.install_logger
get_logging_console_handler = foundations.verbose.get_logging_console_handler
set_verbosity_level = foundations.verbose.set_verbosity_level
