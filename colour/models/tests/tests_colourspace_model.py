#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.models.colourspace_model` module.
"""

from __future__ import division, unicode_literals

import unittest

from colour.models import ColourspaceModel

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestColourspaceModel']


class TestColourspaceModel(unittest.TestCase):
    """
    Defines :class:`colour.colour.models.ColourspaceModel` class units
    tests methods.
    """

    def test_required_attributes(self):
        """
        Tests presence of required attributes.
        """

        required_attributes = ('name',
                               'mapping',
                               'encoding_function',
                               'decoding_function',
                               'title',
                               'labels')

        for attribute in required_attributes:
            self.assertIn(attribute, dir(ColourspaceModel))


if __name__ == '__main__':
    unittest.main()
