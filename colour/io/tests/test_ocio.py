# -*- coding: utf-8 -*-
"""
Defines the unit tests for the :mod:`colour.io.ocio` module.
"""

import numpy as np
import os
import unittest

from colour.io import process_image_OpenColorIO
from colour.utilities import full

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['RESOURCES_DIRECTORY', 'TestProcessImageOpenColorIO']

RESOURCES_DIRECTORY = os.path.join(os.path.dirname(__file__), 'resources')


class TestProcessImageOpenColorIO(unittest.TestCase):
    """
    Defines :func:`colour.io.ocio.process_image_OpenColorIO` definition unit
    tests methods.
    """

    def test_process_image_OpenColorIO(self):
        """
        Tests :func:`colour.io.ocio.process_image_OpenColorIO` definition.
        """

        import PyOpenColorIO as ocio

        config = os.path.join(RESOURCES_DIRECTORY,
                              'config-aces-reference.ocio.yaml')

        a = full([4, 2, 3], 0.18)

        np.testing.assert_almost_equal(
            process_image_OpenColorIO(
                a, 'ACES - ACES2065-1', 'ACES - ACEScct', config=config),
            np.array([
                [[0.41358781, 0.41358781, 0.41358781],
                 [0.41358781, 0.41358781, 0.41358781]],
                [[0.41358781, 0.41358781, 0.41358781],
                 [0.41358781, 0.41358781, 0.41358781]],
                [[0.41358781, 0.41358781, 0.41358781],
                 [0.41358781, 0.41358781, 0.41358781]],
                [[0.41358781, 0.41358781, 0.41358781],
                 [0.41358781, 0.41358781, 0.41358781]],
            ]),
            decimal=7)

        np.testing.assert_almost_equal(
            process_image_OpenColorIO(
                a,
                'ACES - ACES2065-1',
                'Display - sRGB',
                'Output - SDR Video - ACES 1.0',
                ocio.TRANSFORM_DIR_FORWARD,
                config=config),
            np.array([
                [[0.35595229, 0.35595256, 0.35595250],
                 [0.35595229, 0.35595256, 0.35595250]],
                [[0.35595229, 0.35595256, 0.35595250],
                 [0.35595229, 0.35595256, 0.35595250]],
                [[0.35595229, 0.35595256, 0.35595250],
                 [0.35595229, 0.35595256, 0.35595250]],
                [[0.35595229, 0.35595256, 0.35595250],
                 [0.35595229, 0.35595256, 0.35595250]],
            ]),
            decimal=7)


if __name__ == '__main__':
    unittest.main()
