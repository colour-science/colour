#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Input / Output Common Utilities
===============================

Defines input / output common utilities objects that don"t fall in any specific
category.
"""

from __future__ import division, unicode_literals

from pprint import pformat

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['format_spectral_data']


def format_spectral_data(data):
    """
    Pretty formats given spectral data.

    Parameters
    ----------
    data : dict
        Spectral data to pretty format.

    Returns
    -------
    unicode
        Spectral data pretty representation.
    """

    return pformat(data)
