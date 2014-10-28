# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.appearance.hunt` module.
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.appearance import Hunt_InductionFactors, XYZ_to_Hunt
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
    Hunt colour appearance model.
    """

    FIXTURE_BASENAME = 'hunt.csv'

    OUTPUT_ATTRIBUTES = {'J': 'J',
                         'C_94': 'C',
                         'h_S': 'h',
                         's': 's',
                         'Q': 'Q',
                         'M94': 'M'}

    def output_specification_from_data(self, data):
        """
        Returns the Hunt colour appearance model output specification
        from given data.

        Parameters
        ----------
        data : list
            Fixture data.

        Returns
        -------
        Hunt_Specification
            Hunt colour appearance model specification.
        """

        XYZ = np.array([data['X'], data['Y'], data['Z']])
        XYZ_w = np.array([data['X_w'], data['Y_w'], data['Z_w']])
        XYZ_b = np.array([data['X_w'], 0.2 * data['Y_w'], data['Z_w']])

        specification = XYZ_to_Hunt(XYZ,
                                    XYZ_w,
                                    XYZ_b,
                                    data['L_A'],
                                    Hunt_InductionFactors(
                                        data['N_c'],
                                        data['N_b']),
                                    CCT_w=data['T'])

        return specification
