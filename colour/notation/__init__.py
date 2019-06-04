# -*- coding: utf-8 -*-

from __future__ import absolute_import

import sys

from colour.utilities.deprecation import ModuleAPI, Renamed
from colour.utilities.documentation import is_documentation_building

from .datasets import *  # noqa
from . import datasets
from .munsell import MUNSELL_VALUE_METHODS
from .munsell import munsell_value
from .munsell import (munsell_value_Priest1920, munsell_value_Munsell1933,
                      munsell_value_Moon1943, munsell_value_Saunderson1944,
                      munsell_value_Ladd1955, munsell_value_McCamy1987,
                      munsell_value_ASTMD1535)
from .munsell import munsell_colour_to_xyY, xyY_to_munsell_colour
from .triplet import RGB_to_HEX, HEX_to_RGB

__all__ = []
__all__ += datasets.__all__
__all__ += ['munsell_value']
__all__ += ['MUNSELL_VALUE_METHODS']
__all__ += [
    'munsell_value_Priest1920', 'munsell_value_Munsell1933',
    'munsell_value_Moon1943', 'munsell_value_Saunderson1944',
    'munsell_value_Ladd1955', 'munsell_value_McCamy1987',
    'munsell_value_ASTMD1535'
]
__all__ += ['munsell_colour_to_xyY', 'xyY_to_munsell_colour']
__all__ += ['RGB_to_HEX', 'HEX_to_RGB']


# ----------------------------------------------------------------------------#
# ---                API Changes and Deprecation Management                ---#
# ----------------------------------------------------------------------------#
class notation(ModuleAPI):
    def __getattr__(self, attribute):
        return super(notation, self).__getattr__(attribute)


# v0.3.14
API_CHANGES = {
    'Renamed': [[
        'colour.notation.munsell_value_ASTMD153508',
        'colour.notation.munsell_value_ASTMD1535',
    ], ]
}
"""
Defines *colour.notation* sub-package API changes.

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

    sys.modules['colour.notation'] = notation(sys.modules['colour.notation'],
                                              API_CHANGES)

    del sys
