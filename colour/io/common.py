# -*- coding: utf-8 -*-
"""
Input / Output Common Utilities
===============================

Defines input / output common utilities objects that don't fall in any specific
category.
"""

from __future__ import division, unicode_literals

from pprint import pformat

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
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

    Examples
    --------
    >>> import os
    >>> from colour import read_spectral_data_from_csv_file
    >>> csv_file = os.path.join(os.path.dirname(__file__), 'tests',
    ...                         'resources', 'colorchecker_n_ohta.csv')
    >>> sds_data = {'1': read_spectral_data_from_csv_file(csv_file)['1']}
    >>> print(format_spectral_data(sds_data['1']))  # doctest: +ELLIPSIS
    {380.0: 0.0...,
     385.0: 0.0...,
     390.0: 0.0...,
     395.0: 0.0...,
     400.0: 0.0...,
     405.0: 0.0...,
     410.0: 0.0...,
     415.0: 0.0...,
     420.0: 0.0...,
     425.0: 0.0...,
     430.0: 0.0...,
     435.0: 0.0...,
     440.0: 0.0...,
     445.0: 0.0...,
     450.0: 0.0...,
     455.0: 0.0...,
     460.0: 0.0...,
     465.0: 0.0...,
     470.0: 0.0...,
     475.0: 0.0...,
     480.0: 0.0...,
     485.0: 0.0...,
     490.0: 0.0...,
     495.0: 0.0...,
     500.0: 0.0...,
     505.0: 0.0...,
     510.0: 0.0...,
     515.0: 0.0...,
     520.0: 0.0...,
     525.0: 0.0...,
     530.0: 0.0...,
     535.0: 0.0...,
     540.0: 0.0...,
     545.0: 0.0...,
     550.0: 0.0...,
     555.0: 0.0...,
     560.0: 0.0...,
     565.0: 0.0...,
     570.0: 0.1...,
     575.0: 0.1...,
     580.0: 0.1...,
     585.0: 0.1...,
     590.0: 0.1...,
     595.0: 0.1...,
     600.0: 0.1...,
     605.0: 0.1...,
     610.0: 0.1...,
     615.0: 0.1...,
     620.0: 0.1...,
     625.0: 0.1...,
     630.0: 0.1...,
     635.0: 0.1...,
     640.0: 0.1...,
     645.0: 0.1...,
     650.0: 0.1...,
     655.0: 0.1...,
     660.0: 0.2...,
     665.0: 0.2...,
     670.0: 0.2...,
     675.0: 0.2...,
     680.0: 0.2...,
     685.0: 0.2...,
     690.0: 0.2...,
     695.0: 0.2...,
     700.0: 0.2...,
     705.0: 0.2...,
     710.0: 0.3...,
     715.0: 0.3...,
     720.0: 0.3...,
     725.0: 0.3...,
     730.0: 0.3...,
     735.0: 0.3...,
     740.0: 0.4...,
     745.0: 0.4...,
     750.0: 0.4...,
     755.0: 0.4...,
     760.0: 0.4...,
     765.0: 0.4...,
     770.0: 0.4...,
     775.0: 0.4...,
     780.0: 0.4...}
    """

    return pformat(data)
