# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.utilities.deprecation` module.
"""

from __future__ import division, unicode_literals

import unittest

from colour.utilities.deprecation import get_attribute

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestGetAttribute']


class TestGetAttribute(unittest.TestCase):
    """
    Defines :func:`colour.utilities.deprecation.get_attribute` definition unit
    tests methods.
    """

    def test_get_attribute(self):
        """
        Tests :func:`colour.utilities.deprecation.get_attribute` definition.
        """

        from colour import adaptation
        self.assertIs(get_attribute('colour.adaptation'), adaptation)

        from colour.models import oetf_sRGB
        self.assertIs(get_attribute('colour.models.oetf_sRGB'), oetf_sRGB)

        from colour.utilities.array import as_numeric
        self.assertIs(
            get_attribute('colour.utilities.array.as_numeric'), as_numeric)


if __name__ == '__main__':
    unittest.main()
