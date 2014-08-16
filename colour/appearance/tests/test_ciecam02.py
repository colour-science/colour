# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines units tests for :mod:`colour.appearance.ciecam02` module.
"""

import numpy as np
from colour.appearance.ciecam02 import (
    CIECAM02_InductionFactors,
    XYZ_to_CIECAM02)
from colour.appearance.tests.common import ColourAppearanceModelTest

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestCIECAM02ColourAppearanceModel']


class TestCIECAM02ColourAppearanceModel(ColourAppearanceModelTest):
    """
    Defines :mod:`colour.appearance.ciecam02` module units tests methods for
    *CIECAM02* colour appearance model.
    """

    FIXTURE_BASENAME = 'ciecam02.csv'

    OUTPUT_ATTRIBUTES = {'J': 'J',
                         'Q': 'Q',
                         'C': 'C',
                         'M': 'M',
                         'S': 's'}

    def get_output_specification_from_data(self, data):
        """
        Returns the *CIECAM02* colour appearance model output specification
        from given data.

        Parameters
        ----------
        data : list
            Fixture data.

        Returns
        -------
        CIECAM02_Specification
            *CIECAM02* colour appearance model specification.
        """

        XYZ = np.array([data['X'], data['Y'], data['Z']])
        XYZ_w = np.array([data['X_W'], data['Y_W'], data['Z_W']])

        specification = XYZ_to_CIECAM02(XYZ,
                                        XYZ_w,
                                        data['L_A'],
                                        data['Y_b'],
                                        CIECAM02_InductionFactors(data['F'],
                                                                  data['c'],
                                                                  data['N_c']))
        return specification