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

from .dslr import MSDS_CAMERA_SENSITIVITIES_DSLR
from colour.utilities import CaseInsensitiveMapping

MSDS_CAMERA_SENSITIVITIES = CaseInsensitiveMapping(
    MSDS_CAMERA_SENSITIVITIES_DSLR)
MSDS_CAMERA_SENSITIVITIES.__doc__ = """
Multi-spectral distributions of camera sensitivities.

References
----------
:cite:`Darrodi2015a`

MSDS_CAMERA_SENSITIVITIES : CaseInsensitiveMapping
    **{Nikon 5100 (NPL), Sigma SDMerill (NPL)}**
"""

__all__ = ['MSDS_CAMERA_SENSITIVITIES']
