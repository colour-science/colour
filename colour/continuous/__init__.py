# -*- coding: utf-8 -*-

from __future__ import absolute_import

import sys

from colour.utilities.deprecation import ModuleAPI, build_API_changes
from colour.utilities.documentation import is_documentation_building

from .abstract import AbstractContinuousFunction
from .signal import Signal
from .multi_signals import MultiSignals

__all__ = []
__all__ += ['AbstractContinuousFunction']
__all__ += ['Signal']
__all__ += ['MultiSignals']


# ----------------------------------------------------------------------------#
# ---                API Changes and Deprecation Management                ---#
# ----------------------------------------------------------------------------#
class continuous(ModuleAPI):
    def __getattr__(self, attribute):
        return super(continuous, self).__getattr__(attribute)


# v0.3.14
API_CHANGES = {
    'ObjectRenamed': [[
        'colour.continuous.MultiSignal',
        'colour.continuous.MultiSignals',
    ], ]
}
"""
Defines *colour.continuous* sub-package API changes.

API_CHANGES : dict
"""

if not is_documentation_building():
    sys.modules['colour.continuous'] = continuous(
        sys.modules['colour.continuous'], build_API_changes(API_CHANGES))

    del ModuleAPI, is_documentation_building, build_API_changes, sys
