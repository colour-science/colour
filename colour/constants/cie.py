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
    a the junction point of the two functions grafted together to create the
    *Lightness* :math:`L^*` function.

    That discontinuity can be avoided by using the rational representation as
    follows: :math:`\epsilon=216/24389`.

References
----------
.. [1]  http://brucelindbloom.com/index.html?LContinuity.html
        (Last accessed 24 February 2014)

"""

CIE_K = 24389 / 27
"""
*CIE* :math:`k` constant.

CIE_K : numeric

Notes
-----
-   The original *CIE* value for :math:`k` is :math:`k=903.3`,
    **Bruce Lindbloom** has shown that this value is causing a discontinuity
    a the junction point of the two functions grafted together to create the
    *Lightness* :math:`L^*` function.

    That discontinuity can be avoided by using the rational representation as
    follows: :math:`k=24389/27`.

References
----------
.. [2]  http://brucelindbloom.com/index.html?LContinuity.html
        (Last accessed 24 February 2014)

"""
