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
-   characterisation: Colour fitting and camera characterisation.
-   colorimetry: Core objects for colour computations.
-   constants: *CIE* and *CODATA* constants.
-   difference: Colour difference computations.
-   examples: Examples for the sub-packages.
-   io: Input / output objects for reading and writing data.
-   models: Colour models.
-   notation: Colour notation systems.
-   optimal: Optimal colour stimuli computation.
-   phenomenons: Computation of various optical phenomenons.
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
from .appearance import *
from . import appearance
from .colorimetry import *
from . import colorimetry
from . import constants
from .difference import *
from . import difference
from .characterisation import *
from . import characterisation
from .io import *
from . import io
from .models import *
from . import models
from .optimal import *
from . import optimal
from .phenomenons import *
from . import phenomenons
from .notation import *
from . import notation
from .quality import *
from . import quality
from .temperature import *
from . import temperature
from . import plotting
from . import utilities

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = []
__all__ += adaptation.__all__
__all__ += algebra.__all__
__all__ += appearance.__all__
__all__ += characterisation.__all__
__all__ += colorimetry.__all__
__all__ += difference.__all__
__all__ += io.__all__
__all__ += models.__all__
__all__ += optimal.__all__
__all__ += phenomenons.__all__
__all__ += notation.__all__
__all__ += quality.__all__
__all__ += temperature.__all__

__application_name__ = 'Colour'

__major_version__ = '0'
__minor_version__ = '2'
__change_version__ = '1'
__version__ = '.'.join((__major_version__,
                        __minor_version__,
                        __change_version__))
