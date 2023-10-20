"""
:math:`\\Delta E'` - Delta E Colour Difference - Li et al. (2017)
=================================================================

Defines the :math:`\\Delta E'` colour difference computation objects based on
*Li, Li, Wang, Zu, Luo, Cui, Melgosa, Brill and Pointer (2017)* *CAM16-LCD*,
*CAM16-SCD*, and *CAM16-UCS* colourspaces:

-   :func:`colour.difference.delta_E_CAM16LCD`
-   :func:`colour.difference.delta_E_CAM16SCD`
-   :func:`colour.difference.delta_E_CAM16UCS`

References
----------
-   :cite:`Li2017` : Li, C., Li, Z., Wang, Z., Xu, Y., Luo, M. R., Cui, G.,
    Melgosa, M., Brill, M. H., & Pointer, M. (2017). Comprehensive color
    solutions: CAM16, CAT16, and CAM16-UCS. Color Research & Application,
    42(6), 703-718. doi:10.1002/col.22131
"""

from colour.difference.cam02_ucs import (
    delta_E_CAM02LCD,
    delta_E_CAM02SCD,
    delta_E_CAM02UCS,
    delta_E_Luo2006,
)
from colour.models.cam16_ucs import (
    _UCS_Luo2006_callable_to_UCS_Li2017_docstring,
)
from colour.utilities import copy_definition

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "delta_E_Li2017",
    "delta_E_CAM16LCD",
    "delta_E_CAM16SCD",
    "delta_E_CAM16UCS",
]

delta_E_Li2017 = copy_definition(delta_E_Luo2006, "delta_E_Li2017")
delta_E_Li2017.__doc__ = _UCS_Luo2006_callable_to_UCS_Li2017_docstring(
    delta_E_Luo2006
)

delta_E_CAM16LCD = copy_definition(delta_E_CAM02LCD, "delta_E_CAM16LCD")
delta_E_CAM16LCD.__doc__ = _UCS_Luo2006_callable_to_UCS_Li2017_docstring(
    delta_E_CAM02LCD
)

delta_E_CAM16SCD = copy_definition(delta_E_CAM02SCD, "delta_E_CAM16SCD")
delta_E_CAM16SCD.__doc__ = _UCS_Luo2006_callable_to_UCS_Li2017_docstring(
    delta_E_CAM02SCD
)

delta_E_CAM16UCS = copy_definition(delta_E_CAM02UCS, "delta_E_CAM16UCS")
delta_E_CAM16UCS.__doc__ = _UCS_Luo2006_callable_to_UCS_Li2017_docstring(
    delta_E_CAM02UCS
)
