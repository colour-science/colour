# -*- coding: utf-8 -*-
"""
CIE Constants
=============

Defines the *CIE* constants.

References
----------
-   :cite:`Wyszecki2000s` : Wyszecki, Günther, & Stiles, W. S. (2000).
    Standard Photometric Observers. In Color Science: Concepts and Methods,
    Quantitative Data and Formulae (pp. 256-259,395). Wiley.
    ISBN:978-0-471-39918-6
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

__all__ = ['K_M', 'KP_M']

K_M = 683
if is_documentation_building():  # pragma: no cover
    K_M = DocstringFloat(K_M)
    K_M.__doc__ = """
Rounded maximum photopic luminous efficiency :math:`K_m` value in
:math:`lm\\cdot W^{-1}`.

K_M : numeric

Notes
-----
-   To be adequate for all practical applications the :math:`K_m` value has
    been rounded from the original 683.002 value.

References
----------
:cite:`Wyszecki2000s`
"""

KP_M = 1700
if is_documentation_building():  # pragma: no cover
    KP_M = DocstringFloat(KP_M)
    KP_M.__doc__ = """
Rounded maximum scotopic luminous efficiency :math:`K^{\\prime}_m` value in
:math:`lm\\cdot W^{-1}`.

KP_M : numeric

Notes
-----
-   To be adequate for all practical applications the :math:`K^{\\prime}_m`
    value has been rounded from the original 1700.06 value.

References
----------
:cite:`Wyszecki2000s`
"""
