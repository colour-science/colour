# -*- coding: utf-8 -*-

from __future__ import absolute_import

from colour.utilities import CaseInsensitiveMapping, filter_kwargs

from .common import (
    ij_to_polar, polar_to_ij, Jab_to_spherical, spherical_to_Jab,
    close_gamut_boundary_descriptor, fill_gamut_boundary_descriptor,
    sample_volume_boundary_descriptor, tessellate_volume_boundary_descriptor)
from .morovic2000 import (area_boundary_descriptor_Morovic2000,
                          volume_boundary_descriptor_Morovic2000,
                          gamut_boundary_descriptor_Morovic2000)

__all__ = [
    'ij_to_polar', 'polar_to_ij', 'Jab_to_spherical', 'spherical_to_Jab',
    'close_gamut_boundary_descriptor', 'fill_gamut_boundary_descriptor',
    'sample_volume_boundary_descriptor',
    'tessellate_volume_boundary_descriptor'
]
__all__ += [
    'area_boundary_descriptor_Morovic2000',
    'volume_boundary_descriptor_Morovic2000',
    'gamut_boundary_descriptor_Morovic2000'
]

GAMUT_BOUNDARY_DESCRIPTOR_METHODS = CaseInsensitiveMapping({
    'Morovic 2000': gamut_boundary_descriptor_Morovic2000
})
GAMUT_BOUNDARY_DESCRIPTOR_METHODS.__doc__ = """
Supported *Gamut Boundary Descriptor (GDB)* computation methods.

References
----------
:cite:``

GAMUT_BOUNDARY_DESCRIPTOR_METHODS : CaseInsensitiveMapping
    **{'Morovic 2000'}**
"""


def gamut_boundary_descriptor(Jab_ij, method='Morovic 2000', **kwargs):
    function = GAMUT_BOUNDARY_DESCRIPTOR_METHODS[method]

    return function(Jab_ij, **filter_kwargs(function, **kwargs))


__all__ += ['GAMUT_BOUNDARY_DESCRIPTOR_METHODS', 'gamut_boundary_descriptor']
