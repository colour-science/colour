# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.appearance.hunt` module.
"""

from __future__ import division, unicode_literals

import numpy as np
from colour.appearance.hunt import XYZ_to_Hunt
from colour.appearance.tests.common import ColourAppearanceModelTest

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestHuntColourAppearanceModel']


class TestHuntColourAppearanceModel(ColourAppearanceModelTest):
    """
    Defines :mod:`colour.appearance.hunt` module unit tests methods for
    *Hunt* colour appearance model.
    """

    FIXTURE_BASENAME = 'hunt.csv'

    OUTPUT_ATTRIBUTES = {'h_S': 'h_S',
                         's': 's',
                         'Q': 'Q',
                         'J': 'J',
                         'C_94': 'C_94',
                         'M94': 'M_94'}

    def output_specification_from_data(self, data):
        """
        Returns the *Hunt* colour appearance model output specification
        from given data.

        Parameters
        ----------
        data : list
            Fixture data.

        Returns
        -------
        Hunt_Specification
            *Hunt* colour appearance model specification.
        """

        XYZ = np.array([data['X'], data['Y'], data['Z']])
        XYZ_b = np.array([data['X_W'], 0.2 * data['Y_W'], data['Z_W']])
        XYZ_w = np.array([data['X_W'], data['Y_W'], data['Z_W']])

        specification = XYZ_to_Hunt(XYZ,
                                    XYZ_b,
                                    XYZ_w,
                                    L_A=data['L_A'],
                                    N_c=data['N_c'],
                                    N_b=data['N_b'],
                                    CCT_w=data['T'])

        return specification
