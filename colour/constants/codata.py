# -*- coding: utf-8 -*-
"""
Fundamental Physical Constants
==============================

Defines various constants from recommended values by the Committee on Data for
Science and Technology (CODATA).
"""

from __future__ import division, unicode_literals

from colour.utilities.documentation import DocstringFloat

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'AVOGADRO_CONSTANT', 'BOLTZMANN_CONSTANT', 'LIGHT_SPEED', 'PLANCK_CONSTANT'
]

AVOGADRO_CONSTANT = DocstringFloat(6.02214179e23)
AVOGADRO_CONSTANT.__doc__ = """
Avogadro constant.

AVOGADRO_CONSTANT : numeric
"""

BOLTZMANN_CONSTANT = DocstringFloat(1.38065e-23)
BOLTZMANN_CONSTANT.__doc__ = """
Boltzmann constant.

BOLTZMANN_CONSTANT : numeric
"""

LIGHT_SPEED = DocstringFloat(299792458)
LIGHT_SPEED.__doc__ = """
Speed of light in vacuum.

LIGHT_SPEED : numeric
"""

PLANCK_CONSTANT = DocstringFloat(6.62607e-34)
PLANCK_CONSTANT.__doc__ = """
Planck constant.

PLANCK_CONSTANT : numeric
"""
