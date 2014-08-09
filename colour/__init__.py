#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**__init__.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package initialisation.

**Others:**

"""

from __future__ import absolute_import

import sys

if sys.version_info[0] >= 3:
    # Python 3 compatibility hacks.
    import builtins
    import itertools
    import functools

    builtins.basestring = str
    builtins.unicode = str
    builtins.reduce = functools.reduce
    itertools.izip = zip

from .adaptation import *
from . import adaptation
from .algebra import *
from . import algebra
from .colorimetry import *
from . import colorimetry
from . import constants
from .difference import *
from . import difference
from .characterization import *
from . import characterization
from .models import *
from . import models
from .optimal import *
from . import optimal
from .notation import *
from . import notation
from .quality import *
from . import quality
from .temperature import *
from . import temperature

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = []
__all__ += adaptation.__all__
__all__ += algebra.__all__
__all__ += characterization.__all__
__all__ += colorimetry.__all__
__all__ += difference.__all__
__all__ += models.__all__
__all__ += optimal.__all__
__all__ += notation.__all__
__all__ += quality.__all__
__all__ += temperature.__all__

__application_name__ = "Colour"

__major_version__ = "0"
__minor_version__ = "2"
__change_version__ = "1"
__version__ = ".".join((__major_version__,
                        __minor_version__,
                        __change_version__))
