# -*- coding: utf-8 -*-
"""
Fundamental Physical Constants
==============================

Defines various constants from recommended values by the Committee on Data for
Science and Technology (CODATA).
"""

from __future__ import division, unicode_literals

from colour.utilities.documentation import (DocstringFloat,
                                            is_documentation_building)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'AVOGADRO_CONSTANT', 'BOLTZMANN_CONSTANT', 'LIGHT_SPEED', 'PLANCK_CONSTANT'
]

AVOGADRO_CONSTANT = 6.02214179e23
if is_documentation_building():  # pragma: no cover
    AVOGADRO_CONSTANT = DocstringFloat(AVOGADRO_CONSTANT)
    AVOGADRO_CONSTANT.__doc__ = """
Avogadro constant.

AVOGADRO_CONSTANT : numeric
"""

BOLTZMANN_CONSTANT = 1.38065e-23
if is_documentation_building():  # pragma: no cover
    BOLTZMANN_CONSTANT = DocstringFloat(BOLTZMANN_CONSTANT)
    BOLTZMANN_CONSTANT.__doc__ = """
Boltzmann constant.

BOLTZMANN_CONSTANT : numeric
"""

LIGHT_SPEED = 299792458
if is_documentation_building():  # pragma: no cover
    LIGHT_SPEED = DocstringFloat(LIGHT_SPEED)
    LIGHT_SPEED.__doc__ = """
Speed of light in vacuum.

LIGHT_SPEED : numeric
"""

PLANCK_CONSTANT = 6.62607e-34
if is_documentation_building():  # pragma: no cover
    PLANCK_CONSTANT = DocstringFloat(PLANCK_CONSTANT)
    PLANCK_CONSTANT.__doc__ = """
Planck constant.

PLANCK_CONSTANT : numeric
"""
