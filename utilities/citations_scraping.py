#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Citations Scraping Utility
==========================
"""

from __future__ import division, unicode_literals

import codecs
import fnmatch
import os
import re
from io import BytesIO
from tokenize import tokenize

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['CITATION_PATTERN',
           'PREFIX_CITATION_PATTERN',
           'API_TO_APA_SUBSTITUTIONS',
           'citations_from_token',
           'citations_from_module',
           'citations_from_directory']

CITATION_PATTERN = '^\s*\.\.\s+\[\d+\]\s+'
PREFIX_CITATION_PATTERN = '^\s*\.\.\s+'
API_TO_APA_SUBSTITUTIONS = (('\*+', ''),
                            # ('\[\d+\]\s+', ''),
                            ('# noqa', ''))


def citations_from_token(token_info):
    """
    Returns the citations from given token.

    Parameters
    ----------
    token_info : TokenInfo
        Token information.

    Returns
    -------
    list
        Citations from given token.
    """

    citations = []
    if token_info.type == 3 and token_info.type:
        in_citation = False
        for line in token_info.line.split('\n'):
            if re.match(CITATION_PATTERN, line):
                in_citation = True

            if re.match('^\s*$|\s*\"\"\"', line):
                in_citation = False

            if in_citation:
                if not re.match(CITATION_PATTERN, line):
                    citations[-1] = '{0} {1}'.format(
                        citations[-1], line.strip())
                else:
                    citations.append(
                        re.sub(PREFIX_CITATION_PATTERN, '', line.strip()))

    citations = [citation.replace('\ ', '') for citation in citations]

    return citations


def citations_from_module(module):
    """
    Returns the citations from given module.

    Parameters
    ----------
    module : unicode
        Module path.

    Returns
    -------
    list
        Citations from given module.
    """

    module = BytesIO(codecs.open(module,
                                 encoding='utf-8',
                                 errors='ignore').read().encode('utf-8'))

    citations = []
    for token in tokenize(module.readline):
        token_citations = citations_from_token(token)
        if token_citations:
            citations.extend(token_citations)

    return citations


def citations_from_directory(directory):
    """
    Returns the citations from given directory.

    Parameters
    ----------
    directory : unicode
        Directory path.

    Returns
    -------
    dict
        Citations from given directory.
    """

    citations = {}
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, '*.py'):
            module = os.path.join(root, filename)
            module_citations = citations_from_module(module)
            if module_citations:
                citations[module] = module_citations
    return citations


def API_to_APA(citation):
    """
    Performs patterns substitution on given citation.

    Parameters
    ----------
    citation : unicode
        Citation to perform patterns substitution on.

    Returns
    -------
    citation
    """

    for pattern, substitution in API_TO_APA_SUBSTITUTIONS:
        citation = re.sub(pattern, substitution, citation)
    return citation


if __name__ == '__main__':
    from pprint import pprint

    directory = os.path.join('..', 'colour')
    citations = citations_from_directory(directory)
    # pprint(sorted(citations.items()), width=2048)

    for module, module_citations in citations.items():
        print('*' * 79)
        print('{0}\n'.format(module))
        for citation in module_citations:
            print('\t{0}'.format(API_to_APA(citation)))
