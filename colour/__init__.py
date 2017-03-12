#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Colour
======

*Colour* is a *Python* colour science package implementing a comprehensive
number of colour theory transformations and algorithms.

Subpackages
-----------
-   adaptation: Chromatic adaptation models and transformations.
-   algebra: Algebra utilities.
-   appearance: Colour appearance models.
-   characterisation: Colour fitting and camera characterisation.
-   colorimetry: Core objects for colour computations.
-   constants: *CIE* and *CODATA* constants.
-   corresponding: Corresponding colour chromaticities computations.
-   difference: Colour difference computations.
-   examples: Examples for the sub-packages.
-   io: Input / output objects for reading and writing data.
-   models: Colour models.
-   notation: Colour notation systems.
-   phenomenons: Computation of various optical phenomenons.
-   plotting: Diagrams, figures, etc...
-   quality: Colour quality computation.
-   recovery: Reflectance recovery.
-   temperature: Colour temperature and correlated colour temperature
    computation.
-   utilities: Various utilities and data structures.
-   volume: Colourspace volumes computation and optimal colour stimuli.
"""

from __future__ import absolute_import

from .utilities import *  # noqa
from . import utilities  # noqa
from .adaptation import *  # noqa
from . import adaptation  # noqa
from .algebra import *  # noqa
from . import algebra  # noqa
from .colorimetry import *  # noqa
from . import colorimetry  # noqa
from .appearance import *  # noqa
from . import appearance  # noqa
from .constants import *  # noqa
from . import constants  # noqa
from .difference import *  # noqa
from . import difference  # noqa
from .characterisation import *  # noqa
from . import characterisation  # noqa
from .io import *  # noqa
from . import io  # noqa
from .models import *  # noqa
from . import models  # noqa
from .corresponding import *  # noqa
from . import corresponding  # noqa
from .phenomenons import *  # noqa
from . import phenomenons  # noqa
from .notation import *  # noqa
from . import notation  # noqa
from .quality import *  # noqa
from . import quality  # noqa
from .recovery import *  # noqa
from . import recovery  # noqa
from .temperature import *  # noqa
from . import temperature  # noqa
from .volume import *  # noqa
from . import volume  # noqa

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = []
__all__ += utilities.__all__
__all__ += adaptation.__all__
__all__ += algebra.__all__
__all__ += colorimetry.__all__
__all__ += appearance.__all__
__all__ += constants.__all__
__all__ += difference.__all__
__all__ += characterisation.__all__
__all__ += io.__all__
__all__ += models.__all__
__all__ += corresponding.__all__
__all__ += phenomenons.__all__
__all__ += notation.__all__
__all__ += quality.__all__
__all__ += recovery.__all__
__all__ += temperature.__all__
__all__ += volume.__all__

__application_name__ = 'Colour'

__major_version__ = '0'
__minor_version__ = '3'
__change_version__ = '9'
__version__ = '.'.join((__major_version__,
                        __minor_version__,
                        __change_version__))
