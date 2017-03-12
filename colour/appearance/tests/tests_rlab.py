# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.appearance.rlab` module.
"""

from __future__ import division, unicode_literals

import numpy as np
from itertools import permutations

from colour.appearance import XYZ_to_RLAB
from colour.appearance.tests.common import ColourAppearanceModelTest
from colour.utilities import ignore_numpy_errors, tstack

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestRLABColourAppearanceModel']


class TestRLABColourAppearanceModel(ColourAppearanceModelTest):
    """
    Defines :mod:`colour.appearance.rlab` module unit tests methods for
    *RLAB* colour appearance model.
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
        Returns the *RLAB* colour appearance model output specification
        from given data.

        Parameters
        ----------
        data : list
            Fixture data.

        Returns
        -------
        RLAB_Specification
            *RLAB* colour appearance model specification.
        """

        XYZ = tstack((data['X'], data['Y'], data['Z']))
        XYZ_n = tstack((data['X_n'], data['Y_n'], data['Z_n']))

        specification = XYZ_to_RLAB(XYZ,
                                    XYZ_n,
                                    data['Y_n2'],
                                    data['sigma'],
                                    data['D'])

        return specification

    @ignore_numpy_errors
    def test_nan_XYZ_to_RLAB(self):
        """
        Tests :func:`colour.appearance.rlab.XYZ_to_RLAB` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ = np.array(case)
            XYZ_n = np.array(case)
            Y_n = case[0]
            sigma = case[0]
            D = case[0]
            XYZ_to_RLAB(XYZ, XYZ_n, Y_n, sigma, D)
