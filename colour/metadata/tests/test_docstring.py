#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.metadata.docstring` module.
"""

from __future__ import division, unicode_literals

from ast import literal_eval
import unittest

from colour.metadata.docstring import DocstringFields

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestDocstringFields']


class TestDocstringFields(unittest.TestCase):
    """
    Defines :class:`colour.metadata.docstring.DocstringFields` class units
    tests methods.
    """

    def test_required_attributes(self):
        """
        Tests presence of required attributes.
        """

        required_attributes = ('parameters', 'returns', 'notes')

        for attribute in required_attributes:
            self.assertIn(attribute, dir(DocstringFields))

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ('__str__', )

        for method in required_methods:
            self.assertIn(method, dir(DocstringFields))

    def test_parser(self):
        """
        Tests the parser behaviour with various docstrings.
        """

        docstring = """
            Summary of docstring.

            Multi-line
            description of docstring.

            Parameters
            ----------
            argument_1 : object
                Description of `argument_1`.
            argument_2 : object
                Description of `argument_2`.

            Returns
            -------
            object
                Description of `object`.

            Notes
            -----
            -   Note a.
            -   Note b.

            Examples
            -------
            >>> print('This is an example')
            'This is an example'
            """
        fields = DocstringFields(docstring)
        self.assertListEqual(fields.parameters,
                             [(['argument_1', 'object'],
                               'Description of `argument_1`.'),
                              (['argument_2', 'object'],
                               'Description of `argument_2`.')])
        self.assertListEqual(fields.returns, [(['object'],
                                               'Description of `object`.')])
        self.assertListEqual(fields.notes, [('-   Note a.', ''),
                                            ('-   Note b.', '')])

        docstring = """
            Parameters
            ----------
            argument_1 : object
                Description of `argument_1`.
            argument_2 : object
                Description of `argument_2`.

            Returns
            -------
            object
                Description of `object`.
            """
        fields = DocstringFields(docstring)
        self.assertListEqual(fields.parameters,
                             [(['argument_1', 'object'],
                               'Description of `argument_1`.'),
                              (['argument_2', 'object'],
                               'Description of `argument_2`.')])
        self.assertListEqual(fields.returns, [(['object'],
                                               'Description of `object`.')])

        docstring = """
            Parameters
            ----------
            argument_1 : object
                Description of `argument_1`.
            argument_2 : object
                Description of `argument_2`.
            """
        fields = DocstringFields(docstring)
        self.assertListEqual(fields.parameters,
                             [(['argument_1', 'object'],
                               'Description of `argument_1`.'),
                              (['argument_2', 'object'],
                               'Description of `argument_2`.')])
        self.assertListEqual(fields.returns, [])

        docstring = """
            Summary of docstring.

            Returns
            -------
            object
                Description of `object`.
            """

        fields = DocstringFields(docstring)
        self.assertListEqual(fields.parameters, [])
        self.assertListEqual(fields.returns, [(['object'],
                                               'Description of `object`.')])

        docstring = """
            Summary of docstring.

            Returns
            -------
            object
            """

        fields = DocstringFields(docstring)
        self.assertListEqual(fields.parameters, [])
        self.assertListEqual(fields.returns, [(['object'], '')])

        docstring = """
            Parameters
            ----------

            argument_1 : object
                Description of `argument_1`.

            argument_2 : object
            """
        fields = DocstringFields(docstring)
        self.assertListEqual(fields.parameters,
                             [(['argument_1', 'object'],
                               'Description of `argument_1`.'),
                              (['argument_2', 'object'], '')])
        self.assertListEqual(fields.returns, [])

    def test__str__(self):
        """
        Tests :meth:`colour.metadata.docstring.DocstringFields.__str__` method.
        """

        docstring = """
            Summary of docstring.

            Multi-line
            description of docstring.

            Parameters
            ----------
            argument_1 : object
                Description of `argument_1`.
            argument_2 : object
                Description of `argument_2`.

            Returns
            -------
            object
                Description of `object`.

            Notes
            -----
            -   Note a.
            -   Note b.

            Examples
            -------
            >>> print('This is an example')
            'This is an example'
            """
        evaluated = literal_eval(str(DocstringFields(docstring)))
        self.assertListEqual(evaluated['parameters'],
                             [(['argument_1', 'object'],
                               'Description of `argument_1`.'),
                              (['argument_2', 'object'],
                               'Description of `argument_2`.')])
        self.assertListEqual(evaluated['returns'],
                             [(['object'], 'Description of `object`.')])
        self.assertListEqual(evaluated['notes'], [('-   Note a.', ''),
                                                  ('-   Note b.', '')])


if __name__ == '__main__':
    unittest.main()
