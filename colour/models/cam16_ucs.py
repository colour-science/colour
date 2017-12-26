#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CAM16-LCD, CAM16-SCD, and CAM16-UCS Colourspaces - Li et al. (2017)
===================================================================

Defines the *Li, Li, Wang, Zu, Luo, Cui, Melgosa, Brill and Pointer (2017)*
*CAM16-LCD*, *CAM16-SCD*, and *CAM16-UCS* colourspaces transformations:

-   :func:`JMh_CAM16_to_CAM16LCD`
-   :func:`CAM16LCD_to_JMh_CAM16`
-   :func:`JMh_CAM16_to_CAM16SCD`
-   :func:`CAM16SCD_to_JMh_CAM16`
-   :func:`JMh_CAM16_to_CAM16UCS`
-   :func:`CAM16UCS_to_JMh_CAM16`

See Also
--------
`CAM16-LCD, CAM16-SCD, and CAM16-UCS Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/cam16_ucs.ipynb>`_

References
----------
.. [1]  Li, C., Li, Z., Wang, Z., Xu, Y., Luo, M. R., Cui, G., … Pointer, M.
        (2017). Comprehensive color solutions: CAM16, CAT16, and CAM16-UCS.
        Color Research & Application, (January), n/a-n/a. doi:10.1002/col.22131
"""

from __future__ import division, unicode_literals

from functools import partial

from colour.models.cam02_ucs import (COEFFICIENTS_UCS_LUO2006,
                                     JMh_CIECAM02_to_UCS_Luo2006,
                                     UCS_Luo2006_to_JMh_CIECAM02)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'JMh_CAM16_to_UCS_Li2017', 'UCS_Li2017_to_JMh_CAM16',
    'JMh_CAM16_to_CAM16LCD', 'CAM16LCD_to_JMh_CAM16', 'JMh_CAM16_to_CAM16SCD',
    'CAM16SCD_to_JMh_CAM16', 'JMh_CAM16_to_CAM16UCS', 'CAM16UCS_to_JMh_CAM16'
]

JMh_CAM16_to_UCS_Li2017 = JMh_CIECAM02_to_UCS_Luo2006

UCS_Li2017_to_JMh_CAM16 = UCS_Luo2006_to_JMh_CIECAM02

JMh_CAM16_to_CAM16LCD = partial(
    JMh_CAM16_to_UCS_Li2017,
    coefficients=COEFFICIENTS_UCS_LUO2006['CAM02-LCD'])

CAM16LCD_to_JMh_CAM16 = partial(
    UCS_Li2017_to_JMh_CAM16,
    coefficients=COEFFICIENTS_UCS_LUO2006['CAM02-LCD'])

JMh_CAM16_to_CAM16SCD = partial(
    JMh_CAM16_to_UCS_Li2017,
    coefficients=COEFFICIENTS_UCS_LUO2006['CAM02-SCD'])

CAM16SCD_to_JMh_CAM16 = partial(
    UCS_Li2017_to_JMh_CAM16,
    coefficients=COEFFICIENTS_UCS_LUO2006['CAM02-SCD'])

JMh_CAM16_to_CAM16UCS = partial(
    JMh_CAM16_to_UCS_Li2017,
    coefficients=COEFFICIENTS_UCS_LUO2006['CAM02-UCS'])

CAM16UCS_to_JMh_CAM16 = partial(
    UCS_Li2017_to_JMh_CAM16,
    coefficients=COEFFICIENTS_UCS_LUO2006['CAM02-UCS'])
