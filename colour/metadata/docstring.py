#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Docstring Parsing
=================

Defines the objects implementing the docstring parsing for the metadata
system:

-   :class:`DocstringFields`
"""

from __future__ import division, unicode_literals

import re
from pprint import pformat

from colour.utilities import Peekable, is_string

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['DocstringFields']


class DocstringFields(object):
    """
    Extracts *Parameters*, *Returns* and *Notes* sections fields from *Numpy*
    style docstrings.

    Parameters
    ----------
    docstring : unicode
        Docstring text to extract the Parameters*, *Returns* and *Notes*
        sections fields.

    Attributes
    ----------
    parameters
    returns
    notes

    References
    ----------
    .. [1]  Ruana, R. (n.d.). sphinxcontrib.napoleon. Retrieved August 14,
            2016, from https://bitbucket.org/birkenfeld/sphinx-contrib/src/\
21e3b2dc70a43d3f7e8942c4aaf078f66e1c575b/\
napoleon/sphinxcontrib/napoleon/docstring.py

    Examples
    --------
    >>> docstring = '''
    ... Summary of docstring.
    ...
    ... Description of docstring.
    ...
    ... Parameters
    ... ----------
    ... argument_1 : object
    ...     Description of `argument_1`.
    ... argument_2 : object
    ...     Description of `argument_2`.
    ...
    ... Returns
    ... -------
    ... object
    ...     Description of `object`.
    ...
    ... Notes
    ... -----
    ... -   Note a.
    ... -   Note b.
    ...
    ... Examples
    ... -------
    ... >>> print('This is an example')
    ... 'This is an example'
    ... '''
    >>> print(DocstringFields(docstring))  # doctest: +SKIP
    {u'notes': [([u'-   Note a.'], u''), ([u'-   Note b.'], u'')],
     u'parameters': [([u'argument_1', u'object'],
                      u'Description of `argument_1`.'),
                     ([u'argument_2', u'object'],
                      u'Description of `argument_2`.')],
     u'returns': [([u'object'], u'Description of `object`.')]}
    """

    _SECTION_UNDERLINE_PATTERN = re.compile(r'^([=\-`:\'"~^_*+#<>])\1{2,}$')

    def __init__(self, docstring):
        self._docstring = docstring
        self._iterator = Peekable(
            [line.rstrip() for line in docstring.splitlines()])

        self._sections = {
            'args': self._parse_parameters_section,
            'arguments': self._parse_parameters_section,
            'attributes': None,
            'example': None,
            'examples': None,
            'keyword args': None,
            'keyword arguments': None,
            'methods': None,
            'note': self._parse_notes_section,
            'notes': self._parse_notes_section,
            'other parameters': None,
            'parameters': self._parse_parameters_section,
            'return': self._parse_returns_section,
            'returns': self._parse_returns_section,
            'raises': None,
            'references': None,
            'see also': None,
            'todo': None,
            'warning': None,
            'warnings': None,
            'warns': None,
            'yield': None,
            'yields': None}

        self._parameters = []
        self._returns = []
        self._notes = []

        self._parse()

    @property
    def parameters(self):
        """
        Property for **self.parameters** attribute.

        Returns
        -------
        list
            Docstring parameters.

        Warning
        -------
        :attr:`DocstringMetadata.parameters` is read only.
        """

        return self._parameters

    @parameters.setter
    def parameters(self, value):
        """
        Setter for **self.parameters** attribute.

        Parameters
        ----------
        value : object
            Attribute value.
        """

        raise AttributeError(
            '"{0}" attribute is read only!'.format('parameters'))

    @property
    def returns(self):
        """
        Property for **self.returns** attribute.

        Returns
        -------
        list
            Docstring returns.

        Warning
        -------
        :attr:`DocstringMetadata.returns` is read only.
        """

        return self._returns

    @returns.setter
    def returns(self, value):
        """
        Setter for **self.returns** attribute.

        Parameters
        ----------
        value : object
            Attribute value.
        """

        raise AttributeError(
            '"{0}" attribute is read only!'.format('returns'))

    @property
    def notes(self):
        """
        Property for **self.notes** attribute.

        Returns
        -------
        list
            Docstring notes.

        Warning
        -------
        :attr:`DocstringMetadata.notes` is read only.
        """

        return self._notes

    @notes.setter
    def notes(self, value):
        """
        Setter for **self.notes** attribute.

        Parameters
        ----------
        value : object
            Attribute value.
        """

        raise AttributeError(
            '"{0}" attribute is read only!'.format('notes'))

    def __str__(self):
        """
        Returns a pretty formatted string representation of the
        `DocstringFields` class.

        Returns
        -------
        unicode
            Pretty formatted string representation.
        """

        return pformat({'parameters': self._parameters,
                        'returns': self._returns,
                        'notes': self._notes})

    def _indentation_size(self, line):
        """
        Computes the indentation size of given line, i.e. the number of spaces.

        Parameters
        ----------
        line : unicode
            Line to compute the indentation size.

        Returns
        -------
        int
            Indentation size of given line.
        """

        for i, character in enumerate(line):
            if not character.isspace():
                return i

        return 0

    def _is_section_break(self):
        """
        Returns whether the parser is about to consume a section break.

        Returns
        -------
        bool
            Is parser about to consume a section break.
        """

        _line_1, _line_2 = self._iterator.peek(2)

        return self._iterator.exhausted() or self._is_section_header()

    def _is_section_header(self):
        """
        Returns whether the parser is about to consume a section header.

        Returns
        -------
        bool
            Is parser about to consume a section header.
        """

        line_1, line_2 = self._iterator.peek(2)

        if is_string(line_1) and is_string(line_2):
            if (line_1.lstrip().lower() in self._sections and
                    re.match(self._SECTION_UNDERLINE_PATTERN,
                             line_2.lstrip())):
                return True
        else:
            return False

    def _consume_empty(self):
        """
        Commands the parser to consume empty lines.
        """

        line = self._iterator.peek()
        while not self._iterator.exhausted() and not line:
            self._iterator.next()
            line = self._iterator.peek()

    def _consume_to_next_section(self):
        """
        Commands the parser to consume lines until reaching next section.
        """

        self._consume_empty()
        lines = []
        while not self._is_section_break():
            self._iterator.next()
        return ', '.join(lines)

    def _consume_fields(self):
        """
        Commands the parser to consume fields of current section.
        """

        self._consume_empty()
        fields = []
        while not self._is_section_break():
            line = self._iterator.next()
            tokens = re.split('\s*(\w+)\s*:\s*(.*)', line)
            field_summary = [token.strip() for token in tokens if token]
            field_description = []

            indentation = self._indentation_size(line)
            line = self._iterator.peek()
            while (self._indentation_size(line) > indentation and
                    not self._is_section_break()):
                field_description.append(self._iterator.next().lstrip())
                if self._iterator.exhausted():
                    break

                line = self._iterator.peek()
                self._consume_empty()

            fields.append((field_summary, ' '.join(field_description)))
            self._consume_empty()

        return fields

    def _consume_section_header(self):
        """
        Commands the parser to consume current section header.
        """

        section = self._iterator.next().lstrip().lower()
        self._iterator.next()

        return section

    def _parse_parameters_section(self):
        """
        Parses the *Parameters* section.
        """

        self._parameters = self._consume_fields()

    def _parse_returns_section(self):
        """
        Parses the *Returns* section.
        """

        self._returns = self._consume_fields()

    def _parse_notes_section(self):
        """
        Parses the *Notes* section.
        """

        self._consume_empty()
        notes = []
        while not self._is_section_break():
            header = self._iterator.next()
            paragraph = []
            indentation = self._indentation_size(header)
            line = self._iterator.peek()
            while (self._indentation_size(line) > indentation and
                    not self._is_section_break()):
                paragraph.append(self._iterator.next().lstrip())
                if self._iterator.exhausted():
                    break

                line = self._iterator.peek()
                self._consume_empty()

            notes.append((header.lstrip(), ' '.join(paragraph)))
            self._consume_empty()

        self._notes = notes

    def _parse(self):
        """
        Parses the docstring.
        """

        while not self._iterator.exhausted():
            if self._is_section_header():
                section = self._consume_section_header()
                if self._sections[section] is not None:
                    self._sections[section]()
                else:
                    self._consume_to_next_section()
            else:
                self._iterator.next()
