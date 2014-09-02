#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
References Scraping Utility
===========================
"""

from __future__ import division, unicode_literals

import codecs
import fnmatch
import os
import re
from io import BytesIO
from tokenize import tokenize

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['REFERENCE_PATTERN',
           'PREFIX_REFERENCE_PATTERN',
           'references_from_token',
           'references_from_file',
           'references_from_directory']

REFERENCE_PATTERN = '^\s*\.\.\s+\[\d+\]\s+'
PREFIX_REFERENCE_PATTERN = '^\s*\.\.\s+'


def references_from_token(token_info):
    """
    Returns the references from given token.

    Parameters
    ----------
    token_info : TokenInfo
        Token information.

    Returns
    -------
    list
        References from given token.
    """

    references = []
    if (token_info.type == 3 and token_info.type):
        in_reference = False
        for line in token_info.line.split('\n'):
            if re.match(REFERENCE_PATTERN, line):
                in_reference = True

            if re.match('^\s*$|\s*\"\"\"', line):
                in_reference = False

            if in_reference:
                if not re.match(REFERENCE_PATTERN, line):
                    references[-1] = '{0} {1}'.format(references[-1],
                                                      line.strip())
                else:
                    references.append(
                        re.sub(PREFIX_REFERENCE_PATTERN, '', line.strip()))

    return references


def references_from_file(path):
    """
    Returns the references from given file.

    Parameters
    ----------
    path : unicode
        File path.

    Returns
    -------
    list
        References from given file.
    """

    path = BytesIO(codecs.open(path,
                               encoding='utf-8',
                               errors='ignore').read().encode('utf-8'))

    references = []
    for token in tokenize(path.readline):
        token_references = references_from_token(token)
        if token_references:
            references.extend(token_references)

    return references


def references_from_directory(directory):
    """
    Returns the references from given directory.

    Parameters
    ----------
    directory : unicode
        Directory path.

    Returns
    -------
    dict
        References from given directory.
    """

    references = {}
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, '*.py'):
            file = os.path.join(root, filename)
            file_references = references_from_file(file)
            if file_references:
                references[file] = file_references
    return references


if __name__ == '__main__':
    from pprint import pprint

    directory = os.path.join('..', 'colour')
    references = references_from_directory(directory)
    pprint(sorted(references.items()), width=2048)

    for file_references in references.values():
        for reference in file_references:
            print(reference)
