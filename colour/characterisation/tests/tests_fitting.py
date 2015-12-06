#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.characterisation.fitting.fitting` package.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from itertools import permutations
from numpy.linalg import LinAlgError

from colour.characterisation.fitting import first_order_colour_fit
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['M1',
           'M2',
           'TestFirstOrderColourFit']

M1 = ((0.17224809530, 0.09170660377, 0.06416938454),
      (0.49189645050, 0.27802050110, 0.21923398970),
      (0.10999751090, 0.18658946450, 0.29938611390),
      (0.11666119840, 0.14327904580, 0.05713804066),
      (0.18988879020, 0.18227648740, 0.36056247350),
      (0.12501329180, 0.42223441600, 0.37027445440),
      (0.64785605670, 0.22396782040, 0.03365194052),
      (0.06761093438, 0.11076895890, 0.39779138570),
      (0.49101796750, 0.09448929131, 0.11623838540),
      (0.11622385680, 0.04425752908, 0.14469985660),
      (0.36867946390, 0.44545230270, 0.06028680503),
      (0.61632937190, 0.32323905830, 0.02437088825),
      (0.03016472235, 0.06153243408, 0.29014596340),
      (0.11103654650, 0.30553066730, 0.08149136603),
      (0.41162189840, 0.05816655606, 0.04845933989),
      (0.73339205980, 0.53075188400, 0.02475212328),
      (0.47347718480, 0.08834791929, 0.30310314890),
      (0.00000000000, 0.25187015530, 0.35062450170),
      (0.76809638740, 0.78486239910, 0.77808296680),
      (0.53822392230, 0.54307997230, 0.54710882900),
      (0.35458526020, 0.35318419340, 0.35524430870),
      (0.17976704240, 0.18000531200, 0.17991487680),
      (0.09351416677, 0.09510602802, 0.09675027430),
      (0.03405071422, 0.03295076638, 0.03702046722))

M2 = ((0.15579558910, 0.09715754539, 0.07514556497),
      (0.39113140110, 0.25943419340, 0.21266707780),
      (0.12824821470, 0.18463569880, 0.31508022550),
      (0.12028973550, 0.13455659150, 0.07408399880),
      (0.19368988280, 0.21158945560, 0.37955963610),
      (0.19957424007, 0.36085438730, 0.40678122640),
      (0.48896604780, 0.20691688360, 0.05816533044),
      (0.09775521606, 0.16710692640, 0.47147724030),
      (0.39358648660, 0.12233400340, 0.10526425390),
      (0.10780332240, 0.07258529216, 0.16151472930),
      (0.27502670880, 0.34705454110, 0.09728099406),
      (0.43980440500, 0.26880559330, 0.05430532619),
      (0.05887211859, 0.11126271640, 0.38552469020),
      (0.12705825270, 0.25787860160, 0.13566464190),
      (0.35612928870, 0.07933257520, 0.05118732154),
      (0.48131975530, 0.42082843180, 0.07120611519),
      (0.34665584560, 0.15170714260, 0.24969804290),
      (0.08261115849, 0.24588716030, 0.48707732560),
      (0.66054904460, 0.65941137080, 0.66376411910),
      (0.48051509260, 0.47870296240, 0.48230081800),
      (0.33045354490, 0.32904183860, 0.33228886130),
      (0.18001304570, 0.17978566880, 0.18004415930),
      (0.10283975300, 0.10424679520, 0.10384974630),
      (0.04742204025, 0.04772202671, 0.04914225638))


class TestFirstOrderColourFit(unittest.TestCase):
    """
    Defines :func:`colour.characterisation.fitting.first_order_colour_fit`
    definition unit tests methods.
    """

    def test_first_order_colour_fit(self):
        """
        Tests :func:`colour.characterisation.fitting.first_order_colour_fit`
        definition.
        """

        np.testing.assert_almost_equal(
            first_order_colour_fit(M1, M2),
            np.array([[0.69822661, 0.03071629, 0.16210422],
                      [0.06893499, 0.67579611, 0.16430385],
                      [-0.06314955, 0.0921247, 0.97134152]]),
            decimal=7)

    @ignore_numpy_errors
    def test_nan_first_order_colour_fit(self):
        """
        Tests :func:`colour.characterisation.fitting.first_order_colour_fit`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            try:
                first_order_colour_fit(np.vstack((M1, case)),
                                       np.vstack((M2, case)))
            except (ValueError, LinAlgError):
                import traceback
                from colour.utilities import warning

                warning(traceback.format_exc())


if __name__ == '__main__':
    unittest.main()
