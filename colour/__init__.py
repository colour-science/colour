#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Colour
======

**Colour** is a **Python** colour science package implementing a comprehensive
number of colour theory transformations and algorithms.

Subpackages
-----------
-   adaptation: Chromatic adaptation transformations.
-   algebra: Algebra utilities.
-   appearance: Colour appearance models.
-   characterization: Colour fitting and camera characterization.
-   colorimetry: Core objects for colour computations.
-   constants: *CIE* and *CODATA* constants.
-   difference: Colour difference computations.
-   examples: Examples for the sub-packages.
-   implementation: Various implementations of the API.
-   models: Colour models.
-   notation: Colour notation systems.
-   optimal: Optimal colour stimuli computation.
-   plotting: Diagrams, plots, etc...
-   quality: Colour quality computation.
-   temperature: Colour temperature and correlated colour temperature
    computation.
-   utilities: Various utilities and data structures.
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

__author__ = "Colour Developers"
__copyright__ = "Copyright (C) 2013 - 2014 - Colour Developers"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Colour Developers"
__email__ = "colour-science@googlegroups.com"
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
