#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

from .cameras import CAMERAS_RGB_SPECTRAL_SENSITIVITIES
from .colour_checkers import (
    COLOURCHECKERS,
    COLOURCHECKER_INDEXES_TO_NAMES_MAPPING,
    COLOURCHECKERS_SPDS)
from .displays import DISPLAYS_RGB_PRIMARIES

__all__ = []
__all__ += ['CAMERAS_RGB_SPECTRAL_SENSITIVITIES']
__all__ += ['COLOURCHECKERS',
            'COLOURCHECKER_INDEXES_TO_NAMES_MAPPING',
            'COLOURCHECKERS_SPDS']
__all__ += ['DISPLAYS_RGB_PRIMARIES']
