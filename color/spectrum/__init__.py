#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**__init__.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Color** package *spectrum* related data initialisation.

**Others:**

"""

# from __future__ import unicode_literals

import color.utilities.verbose

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["LOGGER"]

LOGGER = color.utilities.verbose.install_logger()

from .blackbody import *
from .cmfs import *
from .color_checkers import *
from .correction import *
from .illuminants import *
from .lefs import *
from .spd import *
from .tcs import *
from .transformations import *

from . import blackbody
from . import cmfs
from . import color_checkers
from . import correction
from . import illuminants
from . import lefs
from . import spd
from . import tcs
from . import transformations

__all__.extend(blackbody.__all__)
__all__.extend(cmfs.__all__)
__all__.extend(color_checkers.__all__)
__all__.extend(correction.__all__)
__all__.extend(illuminants.__all__)
__all__.extend(lefs.__all__)
__all__.extend(spd.__all__)
__all__.extend(tcs.__all__)
__all__.extend(transformations.__all__)

__all__ = map(lambda x: x.encode("ascii"), __all__)

