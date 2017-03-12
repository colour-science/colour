#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
X-Rite Data Input
=================

Defines input object for *X-Rite* spectral data files:

-   :func:`read_spds_from_xrite_file`
"""

from __future__ import division, unicode_literals

import codecs
import numpy as np
import re
from collections import OrderedDict

from colour.colorimetry import SpectralPowerDistribution

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['XRITE_FILE_ENCODING',
           'read_spds_from_xrite_file']

XRITE_FILE_ENCODING = 'utf-8'


def read_spds_from_xrite_file(path):
    """
    Reads the spectral data from given *X-Rite* file and returns it as an
    *OrderedDict* of
    :class:`colour.colorimetry.spectrum.SpectralPowerDistribution` classes.

    Parameters
    ----------
    path : unicode
        Absolute *X-Rite* file path.

    Returns
    -------
    OrderedDict
        :class:`colour.colorimetry.spectrum.SpectralPowerDistribution`
        classes of given *X-Rite* file.

    Notes
    -----
    -   This parser is minimalistic and absolutely not bullet proof.

    Examples
    --------
    >>> import os
    >>> from pprint import pprint
    >>> xrite_file = os.path.join(
    ...     os.path.dirname(__file__),
    ...     'tests',
    ...     'resources',
    ...     'xrite_digital_colour_checker.txt')
    >>> spds_data = read_spds_from_xrite_file(xrite_file)
    >>> pprint(list(spds_data.keys()))  # doctest: +SKIP
    ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']
    """

    with codecs.open(path, encoding=XRITE_FILE_ENCODING) as xrite_file:
        lines = xrite_file.read().strip().split('\n')

        xrite_spds = OrderedDict()
        is_spectral_data_format, is_spectral_data = False, False
        for line in lines:
            line = line.strip()

            if line == 'END_DATA_FORMAT':
                is_spectral_data_format = False

            if line == 'END_DATA':
                is_spectral_data = False

            if is_spectral_data_format:
                wavelengths = [np.float_(x)
                               for x in re.findall('nm(\d+)', line)]
                index = len(wavelengths)

            if is_spectral_data:
                tokens = line.split()
                values = [np.float_(x) for x in tokens[-index:]]
                xrite_spds[tokens[1]] = (
                    SpectralPowerDistribution(tokens[1],
                                              dict(zip(wavelengths, values))))

            if line == 'BEGIN_DATA_FORMAT':
                is_spectral_data_format = True

            if line == 'BEGIN_DATA':
                is_spectral_data = True

        return xrite_spds
