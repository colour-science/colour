#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Entities Metadata
===================

Defines the objects implementing the entities metadata system support:

-   :attr:`ENTITIES`
"""

from __future__ import division, unicode_literals

from colour.metadata import EntityMetadata
from colour.utilities import CaseInsensitiveMapping

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2016 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['ENTITIES']

ENTITIES = CaseInsensitiveMapping(
    {'CIE Lab': EntityMetadata('CIE Lab', '$CIE L^*a^*b^*$'),
     'CIE LCHab': EntityMetadata('CIE LCHab', '$CIE LCH^*a^*b^*$'),
     'CIE XYZ': EntityMetadata('CIE XYZ', '$CIE XYZ$'),
     'Lightness Lstar': EntityMetadata('Lightness Lstar', '$Lightness (L^*)$'),
     'Lightness L': EntityMetadata('Lightness L', '$Lightness (L)$'),
     'Lightness W': EntityMetadata('Lightness W', '$Lightness (W)$'),
     'Luminance Y': EntityMetadata('Luminance Y', '$Luminance (Y)$'),
     'Luminance R_Y': EntityMetadata('Luminance R_Y', '$Luminance (R_Y)$'),
     'Munsell Value': EntityMetadata('Munsell Value', '$Munsell Value (V)$'),
     'Video Signal': EntityMetadata('Video Signal', "$Video Signal (V')$"),
     'Tristimulus Value': EntityMetadata('Tristimulus Value',
                                         '$Tristimulus Value (L)$')})
