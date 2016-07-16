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

from colour.metadata import FunctionMetadata

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
