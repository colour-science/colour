#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CIE Constants
=============

Defines *CIE* constants.
"""

from __future__ import division, unicode_literals

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['CIE_E',
           'CIE_K']

CIE_E = 216 / 24389
"""
*CIE* :math:`\epsilon` constant.

CIE_E : numeric

Notes
-----
-   The original *CIE* value for :math:`\epsilon` is :math:`\epsilon=0.008856`,
    **Bruce Lindbloom** has shown that this value is causing a discontinuity
    at the junction point of the two functions grafted together to create the
    *Lightness* :math:`L^*` function.

    That discontinuity can be avoided by using the rational representation as
    follows: :math:`\epsilon=216\ /\ 24389`.

References
----------
.. [1]  Lindbloom, B. (2003). A Continuity Study of the CIE L* Function.
        Retrieved February 24, 2014, from

"""

CIE_K = 24389 / 27
"""
*CIE* :math:`\kappa` constant.

CIE_K : numeric

Notes
-----
-   The original *CIE* value for :math:`\kappa` is :math:`\kappa=903.3`,
    **Bruce Lindbloom** has shown that this value is causing a discontinuity
    at the junction point of the two functions grafted together to create the
    *Lightness* :math:`L^*` function. [2]_

    That discontinuity can be avoided by using the rational representation as
    follows: :math:`k=24389\ /\ 27`.
"""
