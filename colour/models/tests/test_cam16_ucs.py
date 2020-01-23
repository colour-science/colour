# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.models.cam16_ucs` module.
"""

from __future__ import division, unicode_literals

import unittest

from colour.models.tests.test_cam02_ucs import (
    TestJMh_CIECAM02_to_UCS_Luo2006, TestUCS_Luo2006_to_JMh_CIECAM02)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['TestJMh_CAM16_to_UCS_Li2017', 'TestUCS_Li2017_to_JMh_CAM16']


class TestJMh_CAM16_to_UCS_Li2017(TestJMh_CIECAM02_to_UCS_Luo2006):
    """
    Defines :func:`colour.models.cam16_ucs.JMh_CAM16_to_UCS_Li2017`
    definition unit tests methods.

    Notes
    -----
    -   :func:`colour.models.cam16_ucs.JMh_CAM16_to_UCS_Li2017` is a wrapper
        of :func:`colour.models.cam02_ucs.JMh_CIECAM02_to_UCS_Luo2006` and thus
        currently adopts the same unittests.
    """


class TestUCS_Li2017_to_JMh_CAM16(TestUCS_Luo2006_to_JMh_CIECAM02):
    """
    Defines :func:`colour.models.cam16_ucs.UCS_Li2017_to_JMh_CAM16`
    definition unit tests methods.

    Notes
    -----
    -   :func:`colour.models.cam16_ucs.UCS_Li2017_to_JMh_CAM16` is a wrapper
        of :func:`colour.models.cam02_ucs.UCS_Luo2006_to_JMh_CIECAM02` and thus
        currently adopts the same unittests.
    """


if __name__ == '__main__':
    unittest.main()
