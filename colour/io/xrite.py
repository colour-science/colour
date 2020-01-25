# -*- coding: utf-8 -*-
"""
X-Rite Data Input
=================

Defines input object for *X-Rite* spectral data files:

-   :func:`colour.read_sds_from_xrite_file`
"""

from __future__ import division, unicode_literals

import codecs
import re
from collections import OrderedDict

from colour.colorimetry import SpectralDistribution
from colour.constants import DEFAULT_FLOAT_DTYPE

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['XRITE_FILE_ENCODING', 'read_sds_from_xrite_file']

XRITE_FILE_ENCODING = 'utf-8'


def read_sds_from_xrite_file(path):
    """
    Reads the spectral data from given *X-Rite* file and returns it as an
    *OrderedDict* of :class:`colour.SpectralDistribution` classes.

    Parameters
    ----------
    path : unicode
        Absolute *X-Rite* file path.

    Returns
    -------
    OrderedDict
        :class:`colour.SpectralDistribution` classes of given *X-Rite*
        file.

    Notes
    -----
    -   This parser is minimalistic and absolutely not bullet proof.

    Examples
    --------
    >>> import os
    >>> from pprint import pprint
    >>> xrite_file = os.path.join(os.path.dirname(__file__), 'tests',
    ...                           'resources',
    ...                           'X-Rite_Digital_Colour_Checker.txt')
    >>> sds_data = read_sds_from_xrite_file(xrite_file)
    >>> pprint(list(sds_data.keys()))  # doctest: +SKIP
    ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']
    """

    with codecs.open(path, encoding=XRITE_FILE_ENCODING) as xrite_file:
        lines = xrite_file.read().strip().split('\n')

        xrite_sds = OrderedDict()
        is_spectral_data_format, is_spectral_data = False, False
        for line in lines:
            line = line.strip()

            if line == 'END_DATA_FORMAT':
                is_spectral_data_format = False

            if line == 'END_DATA':
                is_spectral_data = False

            if is_spectral_data_format:
                wavelengths = [
                    DEFAULT_FLOAT_DTYPE(x)
                    for x in re.findall('nm(\\d+)', line)
                ]
                index = len(wavelengths)

            if is_spectral_data:
                tokens = line.split()
                values = [DEFAULT_FLOAT_DTYPE(x) for x in tokens[-index:]]
                xrite_sds[tokens[1]] = (SpectralDistribution(
                    dict(zip(wavelengths, values)), name=tokens[1]))

            if line == 'BEGIN_DATA_FORMAT':
                is_spectral_data_format = True

            if line == 'BEGIN_DATA':
                is_spectral_data = True

        return xrite_sds
