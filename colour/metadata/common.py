#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Common Metadata
===============

Defines the objects implementing the base metadata system support:

-   :func:`filter_metadata_registry`
"""

from __future__ import division, unicode_literals

import ast
import itertools
import re
import sys
from collections import defaultdict, namedtuple

from colour.metadata.docstring import DocstringFields
from colour.utilities import is_string

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'METADATA_REGISTRY', 'ParameterMetadata', 'NotesMetadata',
    'parse_parameters_field_metadata', 'parse_returns_field_metadata',
    'parse_notes_field_metadata', 'set_metadata', 'install_metadata',
    'filter_metadata', 'filter_metadata_registry'
]

if sys.version_info[0] >= 3:
    _DOCSTRING_ATTRIBUTE = '__doc__'
else:
    _DOCSTRING_ATTRIBUTE = 'func_doc'

METADATA_REGISTRY = []
"""
Registry for objects with defined metadata.

METADATA_REGISTRY : list
"""


class ParameterMetadata(
        namedtuple('ParameterMetadata', ('type', 'symbol', 'extent'))):
    """
    Defines the metadata class for parameters.

    Parameters
    ----------
    type : unicode
        Type name.
    symbol : unicode
        Type symbol.
    extent : unicode
        Scale hint to convert in [0, 1] range.
    """

    def __new__(cls, type, symbol, extent=None):
        """
        Returns a new instance of the :class:`ParameterMetadata` class.
        """

        return super(ParameterMetadata, cls).__new__(cls, type, symbol, extent)


class NotesMetadata(
        namedtuple('NotesMetadata', ('classifier', 'method_name',
                                     'method_strict_name'))):
    """
    Defines the metadata class for classifiers and methods.

    Parameters
    ----------
    classifier : unicode
        Classifier name.
    method_name : unicode
        Method name.
    method_strict_name : unicode
        Method strict name.
    """

    def __new__(cls, classifier, method_name, method_strict_name=None):
        """
        Returns a new instance of the :class:`NotesMetadata` class.
        """

        return super(NotesMetadata, cls).__new__(cls, classifier, method_name,
                                                 method_strict_name)


def parse_parameters_field_metadata(field):
    """
    Parses given *Parameters* field metadata from callable docstring.

    Parameters
    ----------
    field : unicode
        *Parameter* field metadata.

    Returns
    -------
    ParameterMetadata
        Type metadata object.
    """

    _summary, description = field

    search = re.search('\s*metadata\s*:\s((?<!\\\\)\{.*?(?<!\\\\)\})',
                       description)
    if search is not None:
        return ParameterMetadata(**ast.literal_eval(search.group(1)))


def parse_returns_field_metadata(field):
    """
    Parses given *Returns* field metadata from callable docstring.

    Parameters
    ----------
    field : unicode
        *Returns* field metadata.

    Returns
    -------
    ParameterMetadata
        Type metadata object.
    """

    return parse_parameters_field_metadata(field)


def parse_notes_field_metadata(field):
    """
    Parses given *Notes* field metadata from callable docstring.

    Parameters
    ----------
    field : unicode
        *Notes* field metadata.

    Returns
    -------
    NotesMetadata
        Method metadata object.
    """

    header, paragraph = field
    search = re.search('\s*metadata\s*:\s(\{.*\})', header + paragraph)
    if search is not None:
        return NotesMetadata(**ast.literal_eval(search.group(1)))


def set_metadata(callable_):
    """
    Sets given callable with given metadata.

    Parameters
    ----------
    callable_ : callable, optional
        Callable to store within the metadata.

    Returns
    -------
    bool
        Definition success.

    Examples
    --------
    >>> from pprint import pprint
    >>> def fn_a(argument_1):
    ...     '''
    ...     Summary of docstring.
    ...
    ...     Description of docstring.
    ...
    ...     Parameters
    ...     ----------
    ...     argument_1 : object
    ...         metadata : {'type': 'type', 'symbol': 'symbol',
    ...             'extent': 'extent'}
    ...         Description of `argument_1`.
    ...
    ...     Returns
    ...     -------
    ...     object
    ...         metadata : {'type': 'type', 'symbol': 'symbol',
    ...             'extent': 'extent'}
    ...         Description of `object`.
    ...
    ...     Notes
    ...     -----
    ...     metadata : {'classifier': 'classifier', 'method_name':
    ...         'method_name', 'method_strict_name': 'method_strict_name'}
    ...     '''
    ...
    ...     return argument_1
    >>> set_metadata(fn_a)
    True
    >>> pprint(dict(fn_a.__metadata__))  # doctest: +SKIP
    {u'notes': [NotesMetadata(classifier='classifier', \
method_name='method_name', method_strict_name='method_strict_name')],
     u'parameters': [ParameterMetadata(type='type', symbol='symbol', \
 extent='extent')],
     u'returns': ParameterMetadata(type='type', symbol='symbol', \
 extent='extent')}
    """

    if getattr(callable_, _DOCSTRING_ATTRIBUTE) is None:
        return False

    fields = DocstringFields(getattr(callable_, _DOCSTRING_ATTRIBUTE))
    metadata = defaultdict(list)
    for parameter in fields.parameters:
        field_metadata = parse_parameters_field_metadata(parameter)
        if field_metadata is not None:
            metadata['parameters'].append(field_metadata)

    if fields.returns:
        field_metadata = parse_returns_field_metadata(fields.returns[0])
        if field_metadata is not None:
            metadata['returns'].append(field_metadata)

    for note in fields.notes:
        field_metadata = parse_notes_field_metadata(note)
        if field_metadata is not None:
            metadata['notes'].append(field_metadata)

    if metadata:
        global METADATA_REGISTRY

        METADATA_REGISTRY.append(callable_)

        callable_.__metadata__ = metadata

    return True


def install_metadata():
    """
    Installs the metadata in objects exposed in *Colour* base namespace.

    Returns
    -------
    bool
        Definition success.
    """

    import colour

    for object_ in itertools.chain(colour.__dict__.values(),
                                   colour.adaptation.__dict__.values(),
                                   colour.algebra.__dict__.values(),
                                   colour.appearance.__dict__.values(),
                                   colour.biochemistry.__dict__.values(),
                                   colour.characterisation.__dict__.values(),
                                   colour.colorimetry.__dict__.values(),
                                   colour.constants.__dict__.values(),
                                   colour.continuous.__dict__.values(),
                                   colour.corresponding.__dict__.values(),
                                   colour.difference.__dict__.values(),
                                   colour.io.__dict__.values(),
                                   colour.models.__dict__.values(),
                                   # colour.metadata.__dict__.values(),
                                   colour.notation.__dict__.values(),
                                   colour.phenomena.__dict__.values(),
                                   # colour.plotting.__dict__.values(),
                                   colour.quality.__dict__.values(),
                                   colour.recovery.__dict__.values(),
                                   colour.temperature.__dict__.values(),
                                   colour.utilities.__dict__.values(),
                                   colour.volume.__dict__.values()):
        if hasattr(object_, _DOCSTRING_ATTRIBUTE):
            set_metadata(object_)

    return True


def filter_metadata(metadata, pattern, attributes, flags=re.IGNORECASE):
    """
    Filters given metadata attributes with given regular expression
    pattern.

    Parameters
    ----------
    metadata : dict
        Callable metadata.
    pattern : unicode
        Regular expression filter pattern.
    attributes : array_like
        **{'type', 'symbol', 'extent', 'classifier', 'method_name',
        'method_strict_name'}**,
        Metadata attributes to filter.
    flags : int
        Regular expression flags.

    Returns
    -------
    bool
        Whether if the metadata attributes are matching the filter pattern.
    """

    for attribute in attributes:
        if re.search(pattern, getattr(metadata, attribute), flags):
            return True

    return False


def filter_metadata_registry(pattern,
                             categories,
                             attributes,
                             any_parameter=False,
                             flags=re.IGNORECASE):
    """
    Filters the metadata registry :attr:`METADATA_REGISTRY` attribute and
    returns matching objects.

    Parameters
    ----------
    pattern : unicode
        Regular expression filter pattern.
    categories : array_like
        **{'parameters', 'returns', 'notes'}**,
        Metadata categories to filter.
    attributes : array_like
        **{'type', 'symbol', 'extent', 'classifier', 'method_name',
        'method_strict_name'}**,
        Metadata attributes to filter.
    any_parameter : bool, optional
        Whether to filter only the first parameter of the callables or any
        parameter.
    flags : int, optional
        Regular expression flags.

    Returns
    -------
    list
        Filtered objects with defined metadata.

    Examples
    --------
    >>> filter_metadata_registry(  # doctest: +SKIP
    ...     'Luminance', categories='parameters', attributes='type')
    [<function lightness_Glasser1958 at 0x...>,
     <function lightness_Wyszecki1963 at 0x...>,
     <function lightness_CIE1976 at 0x...>]
    """

    categories = (categories, ) if is_string(categories) else categories
    attributes = (attributes, ) if is_string(attributes) else attributes
    attributes_p = set(attributes).intersection(ParameterMetadata._fields)
    attributes_n = set(attributes).intersection(NotesMetadata._fields)

    filtered = []

    for callable_ in METADATA_REGISTRY:
        callable_metadata = callable_.__metadata__
        if 'parameters' in categories and callable_metadata.get('parameters'):
            for metadata in callable_metadata['parameters']:
                if filter_metadata(metadata, pattern, attributes_p, flags):
                    filtered.append(callable_)

                if not any_parameter:
                    break

        for category, attributes in zip(('returns', 'notes'), (attributes_p,
                                                               attributes_n)):
            if category in categories and callable_metadata.get(category):
                for metadata in callable_metadata[category]:
                    if filter_metadata(metadata, pattern, attributes, flags):
                        filtered.append(callable_)

    return filtered
