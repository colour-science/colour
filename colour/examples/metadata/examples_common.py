#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Showcases metadata related examples.
"""

from pprint import pprint

import colour
from colour.utilities.verbose import message_box

message_box('Metadata Examples')

message_box('Filtering definitions with a first *Luminance* argument.')
pprint(
    colour.filter_metadata_registry(
        'Luminance', categories='parameters', attributes='type'))

print('\n')

message_box('Filtering definitions with any *Luminance* argument.')
pprint(
    colour.filter_metadata_registry(
        'Luminance',
        categories='parameters',
        attributes='type',
        any_parameter=True))

print('\n')

message_box('Filtering definitions returning a *Luminance* value.')
pprint(
    colour.filter_metadata_registry(
        'Luminance', categories='returns', attributes='type'))

print('\n')

message_box('Filtering definitions using *CIE 1976* method.')
pprint(
    colour.filter_metadata_registry(
        'CIE 1976', categories='notes', attributes='method_name'))
