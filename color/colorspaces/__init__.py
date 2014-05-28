#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**__init__.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Color** package *colorspaces* initialisation.

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

__all__ = ["LOGGER",
           "COLORSPACES"]

LOGGER = color.verbose.install_logger()

from .colorspace import *
from .aces_rgb import *
from .adobe_rgb_1998 import *
from .adobe_wide_gamut_rgb import *
from .alexa_wide_gamut_rgb import *
from .apple_rgb import *
from .best_rgb import *
from .beta_rgb import *
from .cie_rgb import *
from .c_log import *
from .color_match_rgb import *
from .dci_p3 import *
from .don_rgb_4 import *
from .eci_rgb_v2 import *
from .ekta_space_ps5 import *
from .max_rgb import *
from .ntsc_rgb import *
from .pal_secam_rgb import *
from .pointer_gamut import *
from .prophoto_rgb import *
from .rec_709 import *
from .rec_2020 import *
from .russell_rgb import *
from .s_log import *
from .smptec_rgb import *
from .srgb import *
from .xtreme_rgb import *

from . import colorspace
from . import aces_rgb
from . import adobe_rgb_1998
from . import adobe_wide_gamut_rgb
from . import alexa_wide_gamut_rgb
from . import apple_rgb
from . import best_rgb
from . import beta_rgb
from . import cie_rgb
from . import c_log
from . import color_match_rgb
from . import dci_p3
from . import don_rgb_4
from . import eci_rgb_v2
from . import ekta_space_ps5
from . import max_rgb
from . import ntsc_rgb
from . import pal_secam_rgb
from . import pointer_gamut
from . import prophoto_rgb
from . import rec_709
from . import rec_2020
from . import russell_rgb
from . import s_log
from . import smptec_rgb
from . import srgb
from . import xtreme_rgb

__all__.extend(colorspace.__all__)
__all__.extend(aces_rgb.__all__)
__all__.extend(adobe_rgb_1998.__all__)
__all__.extend(adobe_wide_gamut_rgb.__all__)
__all__.extend(alexa_wide_gamut_rgb.__all__)
__all__.extend(apple_rgb.__all__)
__all__.extend(best_rgb.__all__)
__all__.extend(beta_rgb.__all__)
__all__.extend(cie_rgb.__all__)
__all__.extend(c_log.__all__)
__all__.extend(color_match_rgb.__all__)
__all__.extend(dci_p3.__all__)
__all__.extend(don_rgb_4.__all__)
__all__.extend(eci_rgb_v2.__all__)
__all__.extend(ekta_space_ps5.__all__)
__all__.extend(max_rgb.__all__)
__all__.extend(ntsc_rgb.__all__)
__all__.extend(pal_secam_rgb.__all__)
__all__.extend(pointer_gamut.__all__)
__all__.extend(prophoto_rgb.__all__)
__all__.extend(rec_709.__all__)
__all__.extend(rec_2020.__all__)
__all__.extend(russell_rgb.__all__)
__all__.extend(s_log.__all__)
__all__.extend(smptec_rgb.__all__)
__all__.extend(srgb.__all__)
__all__.extend(xtreme_rgb.__all__)
__all__ = map(lambda x: x.encode("ascii"), __all__)

COLORSPACES = {ACES_RGB_COLORSPACE.name: ACES_RGB_COLORSPACE,
               ACES_RGB_LOG_COLORSPACE.name: ACES_RGB_LOG_COLORSPACE,
               ACES_RGB_PROXY_10_COLORSPACE.name: ACES_RGB_PROXY_10_COLORSPACE,
               ACES_RGB_PROXY_12_COLORSPACE.name: ACES_RGB_PROXY_12_COLORSPACE,
               ADOBE_RGB_1998_COLORSPACE.name: ADOBE_RGB_1998_COLORSPACE,
               ADOBE_WIDE_GAMUT_RGB_COLORSPACE.name: ADOBE_WIDE_GAMUT_RGB_COLORSPACE,
               ALEXA_WIDE_GAMUT_RGB_COLORSPACE.name: ALEXA_WIDE_GAMUT_RGB_COLORSPACE,
               APPLE_RGB_COLORSPACE.name: APPLE_RGB_COLORSPACE,
               BEST_RGB_COLORSPACE.name: BEST_RGB_COLORSPACE,
               BETA_RGB_COLORSPACE.name: BETA_RGB_COLORSPACE,
               CIE_RGB_COLORSPACE.name: CIE_RGB_COLORSPACE,
               C_LOG_COLORSPACE.name: C_LOG_COLORSPACE,
               COLOR_MATCH_RGB_COLORSPACE.name: COLOR_MATCH_RGB_COLORSPACE,
               DCI_P3_COLORSPACE.name: DCI_P3_COLORSPACE,
               DON_RGB_4_COLORSPACE.name: DON_RGB_4_COLORSPACE,
               ECI_RGB_V2_COLORSPACE.name: ECI_RGB_V2_COLORSPACE,
               EKTA_SPACE_PS_5_COLORSPACE.name: EKTA_SPACE_PS_5_COLORSPACE,
               MAX_RGB_COLORSPACE.name: MAX_RGB_COLORSPACE,
               NTSC_RGB_COLORSPACE.name: NTSC_RGB_COLORSPACE,
               PAL_SECAM_RGB_COLORSPACE.name: PAL_SECAM_RGB_COLORSPACE,
               PROPHOTO_RGB_COLORSPACE.name: PROPHOTO_RGB_COLORSPACE,
               REC_709_COLORSPACE.name: REC_709_COLORSPACE,
               REC_2020_COLORSPACE.name: REC_2020_COLORSPACE,
               RUSSELL_RGB_COLORSPACE.name: RUSSELL_RGB_COLORSPACE,
               S_LOG_COLORSPACE.name: S_LOG_COLORSPACE,
               SMPTE_C_RGB_COLORSPACE.name: SMPTE_C_RGB_COLORSPACE,
               sRGB_COLORSPACE.name: sRGB_COLORSPACE,
               XTREME_RGB_COLORSPACE.name: XTREME_RGB_COLORSPACE}
