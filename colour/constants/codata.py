# -*- coding: utf-8 -*-
"""
Fundamental Physical Constants
==============================

Defines various constants from recommended values by the Committee on Data for
Science and Technology (CODATA).
"""

from colour.utilities.documentation import (
    DocstringFloat,
    is_documentation_building,
)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'CONSTANT_AVOGADRO',
    'CONSTANT_BOLTZMANN',
    'CONSTANT_LIGHT_SPEED',
    'CONSTANT_PLANCK',
]

CONSTANT_AVOGADRO: float = 6.02214179e23
if is_documentation_building():  # pragma: no cover
    CONSTANT_AVOGADRO = DocstringFloat(CONSTANT_AVOGADRO)
    CONSTANT_AVOGADRO.__doc__ = """
Avogadro constant.
"""

CONSTANT_BOLTZMANN: float = 1.38065e-23
if is_documentation_building():  # pragma: no cover
    CONSTANT_BOLTZMANN = DocstringFloat(CONSTANT_BOLTZMANN)
    CONSTANT_BOLTZMANN.__doc__ = """
Boltzmann constant.
"""

CONSTANT_LIGHT_SPEED: float = 299792458
if is_documentation_building():  # pragma: no cover
    CONSTANT_LIGHT_SPEED = DocstringFloat(CONSTANT_LIGHT_SPEED)
    CONSTANT_LIGHT_SPEED.__doc__ = """
Speed of light in vacuum.
"""

CONSTANT_PLANCK: float = 6.62607e-34
if is_documentation_building():  # pragma: no cover
    CONSTANT_PLANCK = DocstringFloat(CONSTANT_PLANCK)
    CONSTANT_PLANCK.__doc__ = """
Planck constant.
"""
