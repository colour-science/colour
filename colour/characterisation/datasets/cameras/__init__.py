# -*- coding: utf-8 -*-
"""
References
----------
-   :cite:`Darrodi2015a` : Darrodi, M. M., Finlayson, G., Goodman, T., &
    Mackiewicz, M. (2015). Reference data set for camera spectral sensitivity
    estimation. Journal of the Optical Society of America A, 32(3), 381.
    doi:10.1364/JOSAA.32.000381
"""

from __future__ import absolute_import

from .dslr import DSLR_CAMERA_RGB_SPECTRAL_SENSITIVITIES
from colour.utilities import CaseInsensitiveMapping

CAMERA_RGB_SPECTRAL_SENSITIVITIES = CaseInsensitiveMapping(
    DSLR_CAMERA_RGB_SPECTRAL_SENSITIVITIES)
CAMERA_RGB_SPECTRAL_SENSITIVITIES.__doc__ = """
Camera *RGB* spectral sensitivities.

References
----------
:cite:`Darrodi2015a`

CAMERA_RGB_SPECTRAL_SENSITIVITIES : CaseInsensitiveMapping
    **{Nikon 5100 (NPL), Sigma SDMerill (NPL)}**
"""

__all__ = ['CAMERA_RGB_SPECTRAL_SENSITIVITIES']
