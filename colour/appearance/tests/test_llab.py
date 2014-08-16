# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines units tests for :mod:`colour.appearance.llab` module.
"""

from __future__ import division, unicode_literals

import numpy as np
from colour.appearance.llab import XYZ_to_LLAB
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
    Defines :mod:`colour.appearance.llab` module units tests methods for
    *LLAB(l:c)* colour appearance model.
    """

    FIXTURE_BASENAME = 'llab.csv'

    OUTPUT_ATTRIBUTES = {'L_L': 'L_L',
                         'Ch_L': 'Ch_L',
                         's_L': 's_L',
                         'h_L': 'h_L',
                         'C_L': 'C_L',
                         'A_L': 'A_L',
                         'B_L': 'B_L'}

    def get_output_specification_from_data(self, data):
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
                                    data['F_S'],
                                    data['F_L'],
                                    data['F_C'],
                                    data['L'])
        return specification
