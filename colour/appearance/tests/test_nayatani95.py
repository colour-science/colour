# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.appearance.nayatani95` module.
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.appearance import XYZ_to_Nayatani95
from colour.appearance.tests.common import ColourAppearanceModelTest

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestNayatani95ColourAppearanceModel']


class TestNayatani95ColourAppearanceModel(ColourAppearanceModelTest):
    """
    Defines :mod:`colour.appearance.nayatani95` module unit tests methods for
    Nayatani (1995) colour appearance model.
    """

    FIXTURE_BASENAME = 'nayatani95.csv'

    OUTPUT_ATTRIBUTES = {'Lstar_P': 'Lstar_P',
                         'C': 'C',
                         'theta': 'h',
                         'S': 's',
                         'B_r': 'Q',
                         'M': 'M',
                         'Lstar_N': 'Lstar_N'}

    def output_specification_from_data(self, data):
        """
        Returns the Nayatani (1995) colour appearance model output
        specification from given data.

        Parameters
        ----------
        data : list
            Fixture data.

        Returns
        -------
        Nayatani95_Specification
            Nayatani (1995) colour appearance model specification.
        """

        XYZ = np.array([data['X'], data['Y'], data['Z']])
        XYZ_n = np.array([data['X_n'], data['Y_n'], data['Z_n']])

        specification = XYZ_to_Nayatani95(XYZ,
                                          XYZ_n,
                                          data['Y_o'],
                                          data['E_o'],
                                          data['E_or'])
        return specification
