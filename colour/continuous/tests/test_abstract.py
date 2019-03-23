# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.continuous.abstract` module.
"""

from __future__ import division, unicode_literals

import unittest

from colour.continuous import AbstractContinuousFunction

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestAbstractContinuousFunction']


class TestAbstractContinuousFunction(unittest.TestCase):
    """
    Defines :class:`colour.continuous.abstract.AbstractContinuousFunction`
    class unit tests methods.
    """

    def test_required_attributes(self):
        """
        Tests presence of required attributes.
        """

        required_attributes = ('name', 'domain', 'range', 'interpolator',
                               'interpolator_args', 'extrapolator',
                               'extrapolator_args', 'function')

        for attribute in required_attributes:
            self.assertIn(attribute, dir(AbstractContinuousFunction))

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ('__str__', '__repr__', '__hash__', '__getitem__',
                            '__setitem__', '__contains__', '__len__', '__eq__',
                            '__ne__', '__iadd__', '__add__', '__isub__',
                            '__sub__', '__imul__', '__mul__', '__idiv__',
                            '__div__', '__ipow__', '__pow__',
                            'arithmetical_operation', 'fill_nan',
                            'domain_distance', 'is_uniform', 'copy')

        for method in required_methods:
            self.assertIn(method, dir(AbstractContinuousFunction))


if __name__ == '__main__':
    unittest.main()
