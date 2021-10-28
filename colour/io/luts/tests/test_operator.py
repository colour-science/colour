# -*- coding: utf-8 -*-
"""
Defines the unit tests for the :mod:`colour.io.luts.operator` module.
"""

import unittest

from colour.io.luts import AbstractLUTSequenceOperator

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['TestAbstractLUTSequenceOperator']


class TestAbstractLUTSequenceOperator(unittest.TestCase):
    """
    Defines :class:`colour.io.luts.operator.AbstractLUTSequenceOperator` class
    unit tests methods.
    """

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ('apply', )

        for method in required_methods:
            self.assertIn(method, dir(AbstractLUTSequenceOperator))


if __name__ == '__main__':
    unittest.main()
