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

        # TODO: Investigate a proper way of doing this, fixed seed, better
        # reference colourspace volume, etc...

        sRGB_volume = 857028.68992  # 10e6

        attempts, threshold = 10, 10000
        while attempts:
            volume = []
            for _ in range(10):
                volume.append(RGB_colourspace_volume_MonteCarlo(
                    sRGB_COLOURSPACE, 10e3))
            volume = np.average(volume)

            if np.abs(sRGB_volume - volume) <= threshold:
                break

            attempts -= 1

        self.assertNotEqual(attempts, 0)


if __name__ == '__main__':
    unittest.main()
