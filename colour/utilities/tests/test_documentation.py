# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.utilities.documentation` module.
"""

from __future__ import division, unicode_literals

import os
import unittest

from colour.utilities.documentation import is_documentation_building

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['TestIsDocumentationBuilding']


class TestIsDocumentationBuilding(unittest.TestCase):
    """
    Defines :func:`colour.utilities.documentation.is_documentation_building`
    definition unit tests methods.
    """

    def test_is_documentation_building(self):
        """
        Tests :func:`colour.utilities.documentation.is_documentation_building`
        definition.
        """

        try:
            self.assertFalse(is_documentation_building())

            os.environ['READTHEDOCS'] = 'True'
            self.assertTrue(is_documentation_building())

            os.environ['READTHEDOCS'] = 'False'
            self.assertTrue(is_documentation_building())

            del os.environ['READTHEDOCS']
            self.assertFalse(is_documentation_building())

            os.environ['COLOUR_SCIENCE_DOCUMENTATION_BUILD'] = 'True'
            self.assertTrue(is_documentation_building())

            os.environ['COLOUR_SCIENCE_DOCUMENTATION_BUILD'] = 'False'
            self.assertTrue(is_documentation_building())

            del os.environ['COLOUR_SCIENCE_DOCUMENTATION_BUILD']
            self.assertFalse(is_documentation_building())

        finally:  # pragma: no cover
            if os.environ.get('READTHEDOCS'):
                del os.environ['READTHEDOCS']

            if os.environ.get('COLOUR_SCIENCE_DOCUMENTATION_BUILD'):
                del os.environ['COLOUR_SCIENCE_DOCUMENTATION_BUILD']


if __name__ == '__main__':
    unittest.main()
