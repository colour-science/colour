# -*- coding: utf-8 -*-
"""
CIE Constants
=============

Defines the *CIE* constants.

References
----------
-   :cite:`Lindbloom2003d` : Lindbloom, B. (2003). A Continuity Study of the
    CIE L* Function. Retrieved February 24, 2014, from
    http://brucelindbloom.com/LContinuity.html
-   :cite:`Wyszecki2000s` : Wyszecki, G., & Stiles, W. S. (2000). Standard
    Photometric Observers. In Color Science: Concepts and Methods,
    Quantitative Data and Formulae (p. 256-259,395). Wiley. ISBN:978-0471399186
"""

from __future__ import division, unicode_literals

from colour.utilities.documentation import DocstringFloat

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['CIE_E', 'CIE_K', 'K_M', 'KP_M']

CIE_E = DocstringFloat(216 / 24389)
CIE_E.__doc__ = """
*CIE* :math:`\epsilon` constant.

CIE_E : numeric

Notes
-----
-   The original *CIE* value for :math:`\epsilon` is :math:`\epsilon=0.008856`,
    Lindbloom (2003) has shown that this value is causing a discontinuity
    at the junction point of the two functions grafted together to create the
    *Lightness* :math:`L^*` function.

    That discontinuity can be avoided by using the rational representation as
    follows: :math:`\epsilon=216\ /\ 24389`.

References
----------
-   :cite:`Lindbloom2003d`

"""

CIE_K = DocstringFloat(24389 / 27)
CIE_K.__doc__ = """
*CIE* :math:`\kappa` constant.

CIE_K : numeric

Notes
-----
-   The original *CIE* value for :math:`\kappa` is :math:`\kappa=903.3`,
    Lindbloom (2003) has shown that this value is causing a discontinuity
    at the junction point of the two functions grafted together to create the
    *Lightness* :math:`L^*` function.

    That discontinuity can be avoided by using the rational representation as
    follows: :math:`k=24389\ /\ 27`.

References
----------
-   :cite:`Lindbloom2003d`
"""

K_M = DocstringFloat(683)
K_M.__doc__ = """
Rounded maximum photopic luminous efficiency :math:`K_m` value in
:math:`lm\cdot W^{-1}`.

K_M : numeric

Notes
-----
-   To be adequate for all practical applications the :math:`K_m` value has
    been rounded from the original 683.002 value.

References
----------
-   :cite:`Wyszecki2000s`
"""

KP_M = DocstringFloat(1700)
KP_M.__doc__ = """
Rounded maximum scotopic luminous efficiency :math:`K^{\prime}_m` value in
:math:`lm\cdot W^{-1}`.

KP_M : numeric

Notes
-----
-   To be adequate for all practical applications the :math:`K^{\prime}_m`
    value has been rounded from the original 1700.06 value.

References
----------
-   :cite:`Wyszecki2000s`
"""
