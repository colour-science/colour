#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.colorimetry.transformations` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

from colour.colorimetry import CMFS
from colour.colorimetry import RGB_10_degree_cmfs_to_LMS_10_degree_cmfs
from colour.colorimetry import RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs
from colour.colorimetry import RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs
from colour.colorimetry import LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs
from colour.colorimetry import LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestRGB_2_degree_cmfs_to_XYZ_2_degree_cmfs',
           'TestRGB_10_degree_cmfs_to_XYZ_10_degree_cmfs',
           'TestRGB_10_degree_cmfs_to_LMS_10_degree_cmfs',
           'TestLMS_2_degree_cmfs_to_XYZ_2_degree_cmfs',
           'TestLMS_10_degree_cmfs_to_XYZ_10_degree_cmfs']


class TestRGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(unittest.TestCase):
    """
    Defines
    :func:`colour.colorimetry.transformations.RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs`  # noqa
    definition unit tests methods.
    """

    def test_RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(self):
        """
        Tests
        :func:`colour.colorimetry.transformations.RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs`  # noqa
        definition.
        """

        cmfs = CMFS.get('CIE 1931 2 Degree Standard Observer')
        np.testing.assert_allclose(
            RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(435),
            cmfs.get(435),
            atol=0.0025)

        np.testing.assert_allclose(
            RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(545),
            cmfs.get(545),
            atol=0.0025)

        np.testing.assert_allclose(
            RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(700),
            cmfs.get(700),
            atol=0.0025)


class TestRGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(unittest.TestCase):
    """
    Defines
    :func:`colour.colorimetry.transformations.RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs`  # noqa
    definition unit tests methods.
    """

    def test_RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(self):
        """
        Tests
        :func:`colour.colorimetry.transformations.RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs`  # noqa
        definition.
        """

        cmfs = CMFS.get('CIE 1964 10 Degree Standard Observer')
        np.testing.assert_allclose(
            RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(435),
            cmfs.get(435),
            atol=0.025)

        np.testing.assert_allclose(
            RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(545),
            cmfs.get(545),
            atol=0.025)

        np.testing.assert_allclose(
            RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(700),
            cmfs.get(700),
            atol=0.025)


class TestRGB_10_degree_cmfs_to_LMS_10_degree_cmfs(unittest.TestCase):
    """
    Defines
    :func:`colour.colorimetry.transformations.RGB_10_degree_cmfs_to_LMS_10_degree_cmfs`  # noqa
    definition unit tests methods.
    """

    def test_RGB_10_degree_cmfs_to_LMS_10_degree_cmfs(self):
        """
        Tests
        :func:`colour.colorimetry.transformations.RGB_10_degree_cmfs_to_LMS_10_degree_cmfs`  # noqa
        definition.
        """

        cmfs = CMFS.get('Stockman & Sharpe 10 Degree Cone Fundamentals')
        np.testing.assert_allclose(
            RGB_10_degree_cmfs_to_LMS_10_degree_cmfs(435),
            cmfs.get(435),
            atol=0.0025)

        np.testing.assert_allclose(
            RGB_10_degree_cmfs_to_LMS_10_degree_cmfs(545),
            cmfs.get(545),
            atol=0.0025)

        np.testing.assert_allclose(
            RGB_10_degree_cmfs_to_LMS_10_degree_cmfs(700),
            cmfs.get(700),
            atol=0.0025)


class TestLMS_2_degree_cmfs_to_XYZ_2_degree_cmfs(unittest.TestCase):
    """
    Defines
    :func:`colour.colorimetry.transformations.LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs`  # noqa
    definition unit tests methods.
    """

    def test_LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs(self):
        """
        Tests
        :func:`colour.colorimetry.transformations.LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs`  # noqa
        definition.
        """

        cmfs = CMFS.get('CIE 2012 2 Degree Standard Observer')
        np.testing.assert_allclose(
            LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs(435),
            cmfs.get(435),
            atol=0.00015)

        np.testing.assert_allclose(
            LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs(545),
            cmfs.get(545),
            atol=0.00015)

        np.testing.assert_allclose(
            LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs(700),
            cmfs.get(700),
            atol=0.00015)


class TestLMS_10_degree_cmfs_to_XYZ_10_degree_cmfs(unittest.TestCase):
    """
    Defines
    :func:`colour.colorimetry.transformations.LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs`  # noqa
    definition unit tests methods.
    """

    def test_LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs(self):
        """
        Tests
        :func:`colour.colorimetry.transformations.LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs`  # noqa
        definition.
        """

        cmfs = CMFS.get('CIE 2012 10 Degree Standard Observer')
        np.testing.assert_allclose(
            LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs(435),
            cmfs.get(435),
            atol=0.00015)

        np.testing.assert_allclose(
            LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs(545),
            cmfs.get(545),
            atol=0.00015)

        np.testing.assert_allclose(
            LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs(700),
            cmfs.get(700),
            atol=0.00015)


if __name__ == '__main__':
    unittest.main()
