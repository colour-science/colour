# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.io.luts.common` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.constants import DEFAULT_INT_DTYPE
from colour.io.luts.common import parse_array

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestParseArray']


class TestParseArray(unittest.TestCase):
    """
    Defines :func:`colour.io.luts.common.parse_array` definition unit tests
    methods.
    """

    def test_parse_array(self):
        """
        Tests :func:`colour.io.luts.common.parse_array` definition.
        """

        np.testing.assert_equal(
            parse_array('-0.25 0.5 0.75'),
            np.array([-0.25, 0.5, 0.75]),
        )

        np.testing.assert_equal(
            parse_array(['-0.25', '0.5', '0.75']),
            np.array([-0.25, 0.5, 0.75]),
        )

        a = np.linspace(0, 1, 10)
        np.testing.assert_almost_equal(
            parse_array(
                str(a.tolist()).replace(
                    '[',
                    '',
                ).replace(
                    ']',
                    '',
                ).replace(
                    ' ',
                    '',
                ),
                separator=','),
            a,
            decimal=7)

        self.assertEqual(
            parse_array(['1', '2', '3'], dtype=DEFAULT_INT_DTYPE).dtype,
            DEFAULT_INT_DTYPE)


if __name__ == '__main__':
    unittest.main()
