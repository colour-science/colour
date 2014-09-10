#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fairchild (1990) Chromatic Adaptation Model
===========================================

Defines *Fairchild (1990)* chromatic adaptation model objects:


See Also
--------
`Fairchild (1990) Chromatic Adaptation Model IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/master/notebooks/adaptation/fairchild1990.ipynb>`_  # noqa

References
----------
"""

from __future__ import division, unicode_literals

import math
import numpy as np

from colour.adaptation.cat import VON_KRIES_CAT

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['VON_KRIES_CAT_INVERSE_CAT',
           'CMCCAT2000_InductionFactors',
           'CMCCAT2000_VIEWING_CONDITIONS',
           'CMCCAT2000_forward',
           'CMCCAT2000_reverse']

VON_KRIES_CAT_INVERSE_CAT = np.linalg.inv(VON_KRIES_CAT)
"""
Inverse *VON_KRIES_CAT_INVERSE_CAT* chromatic adaptation transform.

VON_KRIES_CAT_INVERSE_CAT : array_like, (3, 3)
"""


def fairchild1990(XYZ_n,  # Stimulus
                  XYZ_1,  # Viewing Condition 1
                  XYZ_2,  # Viewing Condition 2
                  Y_n,  # Adapting Stimulus Luminance
                  discount_illuminant=False):  # Not used atm.
    XYZ_n, XYZ_1, XYZ_2 = np.ravel(XYZ_n), np.ravel(XYZ_1), np.ravel(XYZ_2)

    LMS_1 = np.dot(VON_KRIES_CAT, XYZ_1)
    LMS_2 = np.dot(VON_KRIES_CAT, XYZ_2)

    LMS_n = np.dot(VON_KRIES_CAT, XYZ_n)
    L_n, M_n, S_n = np.ravel(LMS_n)

    LMS_E = np.dot(VON_KRIES_CAT, np.array([1, 1, 1])) # E illuminant.
    L_E, M_E, S_E = np.ravel(LMS_E)

    v = 1 / 3
    Y3_n = Y_n ** v

    # >>> Equations 9.29 - 9.31
    f_E = lambda x, y: (3 * (x / y)) / (L_n / L_E + M_n / M_E + S_n / S_E)
    f_P = lambda x: (1 + Y3_n + x) / (1 + Y3_n + 1 / x)

    # I'm not sure I'm using the right input, Fairchild mentions $p$ and $a$
    # terms for the short and long wavelength sensitive cones but I can't see
    # them in his equations, however after taking a look at Hunt equations
    # 12.11 - 12.16, it seems to be the way to do it.
    # TODO: Handle *discount_illuminant*.
    p_L = f_P(f_E(L_n, L_E))
    p_M = f_P(f_E(M_n, M_E))
    p_S = f_P(f_E(S_n, S_E))

    a_L = p_L / L_n
    a_M = p_M / M_n
    a_S = p_S / S_n
    # Equations 9.29 - 9.31 >>>

    diagonal = lambda x, y, z: np.diagflat(np.array([x, y, z])).reshape((3, 3))

    A_1 = diagonal(a_L, a_M, a_S)
    LMSp_1 = np.dot(A_1, LMS_1)

    c = 0.219 - 0.0784 * math.log10(Y_n)
    C_1 = diagonal(c)

    LMS_a = np.dot(C_1, LMSp_1)

    # Fairchild says that the A and C matrices must be derived for the second
    # viewing condition and inverted. I have probably missed something but I
    # don't see where in A_1 and C_1 the first viewing condition is present.

    # Huuuu I'm lost :)