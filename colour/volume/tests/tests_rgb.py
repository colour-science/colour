#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.volume.rgb` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

from colour.models import sRGB_COLOURSPACE
from colour.volume import RGB_colourspace_volume_MonteCarlo

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestRGBColourspaceVolumeMonteCarlo']


class TestRGBColourspaceVolumeMonteCarlo(unittest.TestCase):
    """
    Defines :func:`colour.volume.rgb.RGB_colourspace_volume_MonteCarlo`
    definition unit tests methods.
    """

    def test_RGB_colourspace_volume_MonteCarlo(self):
        """
        Tests :func:`colour.volume.rgb.RGB_colourspace_volume_MonteCarlo`
        definition.
        """

        self.assertEquals(
            RGB_colourspace_volume_MonteCarlo(
                sRGB_COLOURSPACE,
                10e3,
                random_state=np.random.RandomState(2)),
            828800.0)


if __name__ == '__main__':
    unittest.main()
