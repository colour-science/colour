# -*- coding: utf-8 -*-

from __future__ import absolute_import

import sys

from colour.utilities.deprecation import ModuleAPI, Renamed
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
    'Renamed': [[
        'colour.continuous.MultiSignal',
        'colour.continuous.MultiSignals',
    ], ]
}
"""
Defines *colour.continuous* sub-package API changes.

API_CHANGES : dict
"""


def _setup_api_changes():
    """
    Setups *Colour* API changes.
    """

    global API_CHANGES

    for renamed in API_CHANGES['Renamed']:
        name, access = renamed
        API_CHANGES[name.split('.')[-1]] = Renamed(name, access)  # noqa
    API_CHANGES.pop('Renamed')


if not is_documentation_building():
    _setup_api_changes()

    del ModuleAPI
    del Renamed
    del is_documentation_building
    del _setup_api_changes

    sys.modules['colour.continuous'] = continuous(
        sys.modules['colour.continuous'], API_CHANGES)

    del sys
