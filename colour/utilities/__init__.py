#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

from .common import (
    batch,
    is_scipy_installed,
    is_iterable,
    is_string,
    is_numeric,
    is_integer)
from .array import as_array, closest, normalise, steps, is_uniform
from .data_structures import Lookup, Structure, CaseInsensitiveMapping
from .verbose import message_box, warning

__all__ = ['batch',
           'is_scipy_installed',
           'is_iterable',
           'is_string',
           'is_numeric',
           'is_integer']
__all__ += ['as_array', 'closest', 'normalise', 'steps', 'is_uniform']
__all__ += ['Lookup', 'Structure', 'CaseInsensitiveMapping']
__all__ += ['message_box', 'warning']
