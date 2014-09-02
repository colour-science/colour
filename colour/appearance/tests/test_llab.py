# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.appearance.llab` module.
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.appearance import LLAB_InductionFactors, XYZ_to_LLAB
from colour.appearance.tests.common import ColourAppearanceModelTest

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestLLABColourAppearanceModel']


class TestLLABColourAppearanceModel(ColourAppearanceModelTest):
    """
    Defines :mod:`colour.appearance.llab` module unit tests methods for
    *LLAB(l:c)* colour appearance model.
    """

    FIXTURE_BASENAME = 'llab.csv'

    OUTPUT_ATTRIBUTES = {'L_L': 'J',
                         'Ch_L': 'C',
                         'h_L': 'h',
                         's_L': 's',
                         'C_L': 'M',
                         'A_L': 'a',
                         'B_L': 'b'}

    def output_specification_from_data(self, data):
        """
        Returns the *LLAB(l:c)* colour appearance model output specification
        from given data.

        Parameters
        ----------
        data : list
            Fixture data.

        Returns
        -------
        LLAB_Specification
            *LLAB(L:c)* colour appearance model specification.
        """

        XYZ = np.array([data['X'], data['Y'], data['Z']])
        XYZ_0 = np.array([data['X_0'], data['Y_0'], data['Z_0']])

        specification = XYZ_to_LLAB(XYZ,
                                    XYZ_0,
                                    data['Y_b'],
                                    data['L'],
                                    LLAB_InductionFactors(1,
                                                          data['F_S'],
                                                          data['F_L'],
                                                          data['F_C']))
        return specification
