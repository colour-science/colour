# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.appearance.rlab` module.
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.appearance import XYZ_to_RLAB
from colour.appearance.tests.common import ColourAppearanceModelTest

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestRLABColourAppearanceModel']


class TestRLABColourAppearanceModel(ColourAppearanceModelTest):
    """
    Defines :mod:`colour.appearance.rlab` module unit tests methods for
    RLAB colour appearance model.
    """

    FIXTURE_BASENAME = 'rlab.csv'

    OUTPUT_ATTRIBUTES = {'LR': 'J',
                         'CR': 'C',
                         'hR': 'h',
                         'sR': 's',
                         'aR': 'a',
                         'bR': 'b'}

    def output_specification_from_data(self, data):
        """
        Returns the RLAB colour appearance model output specification
        from given data.

        Parameters
        ----------
        data : list
            Fixture data.

        Returns
        -------
        RLAB_Specification
            RLAB colour appearance model specification.
        """

        XYZ = np.array([data['X'], data['Y'], data['Z']])
        XYZ_n = np.array([data['X_n'], data['Y_n'], data['Z_n']])

        specification = XYZ_to_RLAB(XYZ,
                                    XYZ_n,
                                    data['Y_n2'],
                                    data['sigma'],
                                    data['D'])
        return specification
