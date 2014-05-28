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

from .verbose import *

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

from .algebra import *
from .chromatic_adaptation import *
from .color_checkers import *
from .colorspaces import *
from .derivation import *
from .difference import *
from .illuminants import *
from .lightness import *
from .spectral import *
from .temperature import *
from .transformations import *

from . import algebra
from . import chromatic_adaptation
from . import color_checkers
from . import colorspaces
from . import derivation
from . import difference
from . import illuminants
from . import lightness
from . import spectral
from . import temperature
from . import transformations

__all__.extend(algebra.__all__)
__all__.extend(chromatic_adaptation.__all__)
__all__.extend(color_checkers.__all__)
__all__.extend(colorspaces.__all__)
__all__.extend(derivation.__all__)
__all__.extend(difference.__all__)
__all__.extend(illuminants.__all__)
__all__.extend(lightness.__all__)
__all__.extend(spectral.__all__)
__all__.extend(temperature.__all__)
__all__.extend(transformations.__all__)

__all__ = map(lambda x: x.encode("ascii"), __all__)