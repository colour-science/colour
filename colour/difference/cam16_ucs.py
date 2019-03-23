# -*- coding: utf-8 -*-
"""
:math:`\\Delta E'` - Delta E Colour Difference - Li et al. (2017)
================================================================

Defines :math:`\\Delta E'` colour difference computation objects based on
*Li, Li, Wang, Zu, Luo, Cui, Melgosa, Brill and Pointer (2017)*
*CAM16-LCD*, *CAM16-SCD*, and *CAM16-UCS* colourspaces:

-   :func:`colour.difference.delta_E_CAM16LCD`
-   :func:`colour.difference.delta_E_CAM16SCD`
-   :func:`colour.difference.delta_E_CAM16UCS`

See Also
--------
`CAM16-LCD, CAM16-SCD, and CAM16-UCS Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/cam16_ucs.ipynb>`_

References
----------
-   :cite:`Li2017` : Li, C., Li, Z., Wang, Z., Xu, Y., Luo, M. R., Cui, G.,
    Pointer, M. (2017). Comprehensive color solutions: CAM16, CAT16, and
    CAM16-UCS. Color Research & Application, 42(6), 703-718.
    doi:10.1002/col.22131
"""

from __future__ import division, unicode_literals

from colour.difference.cam02_ucs import (delta_E_Luo2006, delta_E_CAM02LCD,
                                         delta_E_CAM02SCD, delta_E_CAM02UCS)
from colour.models.cam16_ucs import (
    _UCS_Luo2006_callable_to_UCS_Li2017_docstring)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'delta_E_Li2017', 'delta_E_CAM16LCD', 'delta_E_CAM16SCD',
    'delta_E_CAM16UCS'
]

delta_E_Li2017 = delta_E_Luo2006
delta_E_Li2017.__doc__ = (
    _UCS_Luo2006_callable_to_UCS_Li2017_docstring(delta_E_Luo2006))

delta_E_CAM16LCD = delta_E_CAM02LCD
delta_E_CAM16LCD.__doc__ = (
    _UCS_Luo2006_callable_to_UCS_Li2017_docstring(delta_E_CAM02LCD))

delta_E_CAM16SCD = delta_E_CAM02SCD
delta_E_CAM16SCD.__doc__ = (
    _UCS_Luo2006_callable_to_UCS_Li2017_docstring(delta_E_CAM02SCD))

delta_E_CAM16UCS = delta_E_CAM02UCS
delta_E_CAM16UCS.__doc__ = (
    _UCS_Luo2006_callable_to_UCS_Li2017_docstring(delta_E_CAM02UCS))
