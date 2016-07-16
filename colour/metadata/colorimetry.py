#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Colorimetry Metadata
====================

Defines the objects implementing the colorimetry metadata system support:

-   :class:``
-   :func:`set_metadata`
"""

from __future__ import division, unicode_literals

from colour.colorimetry import (
    lightness_CIE1976,
    lightness_Glasser1958,
    lightness_Wyszecki1963,
    luminance_ASTMD153508,
    luminance_CIE1976,
    luminance_Newhall1943)
from colour.metadata import (
    FunctionMetadata,
    ENTITIES,
    set_callable_metadata)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2016 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['LightnessFunctionMetadata',
           'LuminanceFunctionMetadata']


class LightnessFunctionMetadata(FunctionMetadata):
    """
    Defines the metadata class for *Lightness* :math:`L*` computations
    functions.
    """

    _FAMILY = 'Lightness Function'


class LuminanceFunctionMetadata(FunctionMetadata):
    """
    Defines the metadata class for *luminance* :math:`Y` computations
    functions.
    """

    _FAMILY = 'Luminance Function'


# Lightness
set_callable_metadata(
    lightness_CIE1976, LightnessFunctionMetadata,
    ENTITIES['Luminance Y'], ENTITIES['Lightness Lstar'],
    (0, 100), (0, 100), 'CIE 1976', 'CIE 1976')

set_callable_metadata(
    lightness_Glasser1958, LightnessFunctionMetadata,
    ENTITIES['Luminance Y'], ENTITIES['Lightness L'],
    (0, 100), (0, 100), 'Glasser 1958', 'Glasser (1958)')

set_callable_metadata(
    lightness_Wyszecki1963, LightnessFunctionMetadata,
    ENTITIES['Luminance Y'], ENTITIES['Lightness W'],
    (0, 100), (0, 100), 'Glasser 1958', 'Glasser (1958)')

# Luminance
set_callable_metadata(
    luminance_ASTMD153508, LuminanceFunctionMetadata,
    ENTITIES['Munsell V'], ENTITIES['Luminance Y'],
    (0, 10), (0, 100),
    'ASTM D1535-08', 'ASTM D1535-08e1')

set_callable_metadata(
    luminance_CIE1976, LuminanceFunctionMetadata,
    ENTITIES['Lightness Lstar'], ENTITIES['Luminance Y'],
    (0, 100), (0, 100),
    'CIE 1976', 'CIE 1976')

set_callable_metadata(
    luminance_Newhall1943, LuminanceFunctionMetadata,
    ENTITIES['Munsell V'], ENTITIES['Luminance R_Y'],
    (0, 10), (0, 100),
    'Newhall 1943', 'Newhall, Nickerson, and Judd (1943)')

if __name__ == '__main__':
    for v in lightness_CIE1976.__metadata__.instances.values():
        print()
        print(v)
